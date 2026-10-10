"""
run_submission.py
What every asynchronous submit script (IBM, IQM EQE1, IQM Resonance, IQM VLQ) does once it is
connected to its QPU: print the run config, work out the chunks still to submit, build() the
circuits (from the cache), pick the run's physical qubits, health check before, submit() the chunks,
health check after, and write the submit.log entry. The scripts differ only in how they connect, in
the job packing and in a few provider lines, so those parts stay in each script. (AQT PIAST-Q is
synchronous: its loop script submits one chunk per pass and keeps its own flow.)

Light at import (no qiskit, no myQLM): everything heavy is imported inside submit_async_run().
"""

import contextlib
import sys

from source.core.common import print_done_block
from source.core.runner_support.benchmark_args import DT


def submit_async_run(*, run_folder, backend_name, service_name, real_backend, collect_service, bench, layout, lnn,
                     chunks_to_submit=None, do_spam=True, dt=DT, submit_kwargs=None, config_extra=(), log_extra_lines=(),
                     health_check_session=None, after_layout=None, collect_script=None):
    """Submit run_folder's chunks without a submission record to real_backend; exits with
    EXIT_HC_NOT_SUBMITTED when the health check after the submission could not be submitted.

    bench / layout: BenchmarkArgs / LayoutArgs (source/core/runner_support/benchmark_args.py); lnn: the QPU's
    topology (backends.json); chunks_to_submit: only the first N chunks (None: the whole run); collect_service: what the health-check collection retrieves jobs with
    (the service, adapter or backend of the provider); submit_kwargs: the provider's extra arguments of
    qbcs.submit() (chunks_per_qpu_job_batch, max_circuits_per_job); config_extra: extra (label, value)
    pairs of the printed run config; log_extra_lines: provider lines of the submit.log entry, called after
    the submission (a list or a function returning one); health_check_session(): a context manager yielding
    the provider's extra keyword arguments for one health check (IBM: a fresh Batch and its sampler);
    after_layout(run_qubits): a hook run once the qubits are chosen; collect_script: the script named in the
    final hint. Returns the run folder path."""
    from source.core.qbcs import bacq_ps_qbcs
    from source.core.common import rel_path
    from source.core.health_check import require_functional_health_check, submit_post_health_check
    from source.core.qpu.calibration_input import resolve_run_layout, layout_summary
    from source.core.qpu.providers import ACCOMPANYING_HEALTH_CHECK_NSHOTS
    from source.core.records.health_check_records import (install_submit_crash_log, format_chunk_range, EXIT_HC_NOT_SUBMITTED,
                                                          append_submit_log, print_run_config)
    from source.core.records.job_records import chunks_without_submission

    nshots, sample_size, list_times = bench.nshots, bench.sample_size, bench.list_times
    chunk_size = int(len(list_times) * 6)           # 6 circuits per depth
    n_wanted   = min(chunks_to_submit or sample_size, sample_size)
    print_run_config(run_folder, nshots=nshots, sample_size=sample_size, lnn=lnn, do_spam=do_spam,
                     list_times=list_times, chunk_size=chunk_size,
                     extra=[("Chunks to submit", f"{n_wanted} / {sample_size}"), *config_extra])

    # ── The chunks this run submits: those without a 'submitted' / 'complete' record (new, or of failed ──
    # ── jobs; may have gaps). Both health checks and submit() get exactly this list. ──
    todo = chunks_without_submission(f"datafiles/{run_folder}", n_wanted)
    install_submit_crash_log(f"datafiles/{run_folder}", todo)   # an uncaught error still leaves an entry in submit.log
    if not todo:
        print(f"\n✅ All {n_wanted} chunks already submitted — nothing to do.")
        sys.exit(0)
    print(f"  This submission:  chunks {format_chunk_range(todo)} ({len(todo)})")

    # ── Build qbcs instance and the circuits (from the cache) ─────────────────────
    qbcs = bacq_ps_qbcs(
        infidelity_total            = 0.0,
        ratio_ct                    = 0.0,
        noise_type                  = "pauli",
        sample_size                 = sample_size,
        chunk_size                  = chunk_size,
        list_circuit_times          = list_times,
        error_mitigation_strategies = ["SPAM", "NEC", "RC+NEC", "cRC+NEC"] if do_spam else ["NEC", "RC+NEC", "cRC+NEC"],
        do_SPAM                     = do_spam,
        nshots                      = nshots,
        dt                          = dt,
    )
    qbcs.build(circuit_transpilation_LNN=lnn)

    # ── Physical qubits for this run — before the first health check, so it runs on them too ──
    # (recorded in <run>/transpiled_circuits/physical_qubits.json; submit() reuses the record)
    print(f"  [Layout] requested: {layout_summary(layout, layout.qubits, layout.calibration_file)}")
    run_qubits, cal_source = resolve_run_layout(qbcs, real_backend, run_folder, layout, label=backend_name)
    if after_layout is not None:
        after_layout(run_qubits)

    session = health_check_session or (lambda: contextlib.nullcontext({}))

    # ── Health check before the submission (saved in <run>/health_checks/, auto-numbered) ──
    # Blocking, with bounded retry on transient network errors; each attempt is its own numbered
    # record. Never returns unless the QPU is confirmed functional — see source/core/health_check.py.
    with session() as hc_kwargs:
        require_functional_health_check(
            run_folder      = run_folder,
            backend         = real_backend,
            service_name    = service_name,
            collect_service = collect_service,
            nshots          = ACCOMPANYING_HEALTH_CHECK_NSHOTS,
            reps            = 1,
            chunks          = todo,
            backend_name    = backend_name,
            **hc_kwargs,
        )

    # ── Submit (async) ────────────────────────────────────────────────────────────
    run_folder_path = qbcs.submit(
        run_folder        = run_folder,
        lang              = "qiskit",
        service           = service_name,
        backend           = real_backend,
        layout_connection = "linear_nearest_neighbor" if lnn else "all_to_all",
        chunks            = todo,
        **(submit_kwargs or {}),
    )
    print_done_block(f"{len(todo)} chunk(s) submitted to {backend_name}: {format_chunk_range(todo)}",
                     [("Run folder",  rel_path(run_folder_path)),
                      ("Job records", f"{rel_path(run_folder_path)}/job_records/")])

    # ── Health check after the submission — always, for the same chunks ───────────
    # Collected later by the collect script. Retries the submission itself on a transient network error;
    # never raises/exits here — the chunks are already on the QPU and submit.log must still be written
    # either way (source/core/health_check.py: submit_post_health_check).
    with session() as hc_kwargs:
        post_hc_ok, post_hc_numbers = submit_post_health_check(
            run_folder   = run_folder,
            backend      = real_backend,
            service_name = service_name,
            backend_name = backend_name,
            nshots       = ACCOMPANYING_HEALTH_CHECK_NSHOTS,
            reps         = 1,
            chunks       = todo,
            **hc_kwargs,
        )

    extra_lines = log_extra_lines() if callable(log_extra_lines) else list(log_extra_lines)
    log_path = append_submit_log(run_folder_path, backend=backend_name, chunks=todo, n_chunks=sample_size, nshots=nshots,
                                 layout=layout_summary(layout, run_qubits, cal_source), run_qubits=run_qubits,
                                 extra_lines=extra_lines, post_hc_ok=post_hc_ok, post_hc_numbers=post_hc_numbers)
    print(f"   Log saved:   {rel_path(log_path)}")
    if collect_script:       # last, after the health check: the chunks are on the QPU whatever its outcome
        print(f"\n▶ Run {collect_script} to retrieve results.")
    if not post_hc_ok:
        sys.exit(EXIT_HC_NOT_SUBMITTED)
    return run_folder_path
