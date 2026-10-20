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
import time
from pathlib import Path

from source.core.common import exit_with_error, print_done_block
from source.core.runner_support.benchmark_args import DT


def health_check_retry_folder(undated, root="datafiles"):
    """The run folder of `--retry-health-check`: the latest one of these settings, finished or not (never a new
    one, which dated_run_folder() would pick for a finished run). Exits with code 2 when there is none."""
    from source.core.records.run_names import latest_run_folder
    folder = latest_run_folder(undated, root)
    if folder is None:
        exit_with_error(f"--retry-health-check: no run folder of these settings exists in {root}/.")
    return folder


def _retry_post_health_check(*, run_folder, backend_name, service_name, real_backend, n_chunks, nshots, session):
    """--retry-health-check: the chunks of the run are all submitted, the health check after them is not (its
    submission failed: exit code 11 or 12 of the first launch). Submits that health check again, for the chunks of the
    run's newest health-check record, and logs it. Exits 0 when the newest record is a submitted check after the
    submission already, 2 when chunks are still unsubmitted or the run has no health check, else like submit_async_run."""
    from source.core.common import rel_path
    from source.core.health_check import submit_post_health_check
    from source.core.qpu.providers import ACCOMPANYING_HEALTH_CHECK_NSHOTS
    from source.core.records.health_check_records import (EXIT_HC_NOT_SUBMITTED, append_submit_log, format_chunk_range,
                                                          load_health_checks)
    from source.core.records.job_records import chunks_without_submission
    run_path = f"datafiles/{run_folder}"
    left = chunks_without_submission(run_path, n_chunks)
    if left:
        exit_with_error(f"--retry-health-check: chunks {format_chunk_range(left)} of {run_folder} are not submitted yet; "
                        f"start the script without it.")
    checks = [c for c in load_health_checks(run_path) if c.get("chunks")]
    if not checks:
        exit_with_error(f"--retry-health-check: {run_folder} has no health check to repeat.")
    last = checks[-1]
    if last.get("after_submission") and last.get("status") != "failed":
        print(f"\n✅ Health check #{last['number']} after the submission is already submitted — nothing to do.")
        sys.exit(0)
    chunks = list(last["chunks"])
    print(f"  Health check after the submission of chunks {format_chunk_range(chunks)} ({len(chunks)}): submitting it again")
    with session() as hc_kwargs:
        ok, numbers = submit_post_health_check(
            run_folder=run_folder, backend=real_backend, service_name=service_name, backend_name=backend_name,
            nshots=ACCOMPANYING_HEALTH_CHECK_NSHOTS, reps=1, chunks=chunks, **hc_kwargs)
    log_path = append_submit_log(run_path, backend=backend_name, chunks=[], n_chunks=n_chunks, nshots=nshots,
                                 extra_lines=[f"Retry of the POST health check for chunks {format_chunk_range(chunks)}"],
                                 post_hc_ok=ok, post_hc_numbers=numbers)
    print(f"   Log saved:   {rel_path(log_path)}")
    if not ok:
        sys.exit(EXIT_HC_NOT_SUBMITTED)


def _seconds_since_last_submission(run_path):
    """(seconds since the run's latest submitted job (chunk or health check), None for a run without one; the number of
    a health check that rules out skipping the check before the submission, else None). That check is the latest job
    and is degraded, or is a check before the chunks (not after) whose verdict is not functional (never collected)."""
    from source.core.runner_support.submit_with_retries import submitted_jobs
    jobs = [j for j in submitted_jobs(run_path) if j[0] > 0]
    if not jobs:
        return None, None
    t, label, rec = jobs[-1]
    unsure = label.startswith("health check") and (rec.get("verdict") == "degraded"
                                                  or (not rec.get("after_submission") and rec.get("verdict") != "functional"))
    return max(0.0, time.time() - t), (rec.get("number") if unsure else None)


def submit_async_run(*, run_folder, backend_name, service_name, real_backend, collect_service, bench, layout, lnn,
                     chunks_to_submit=None, do_spam=True, dt=DT, submit_kwargs=None, config_extra=(), log_extra_lines=(),
                     health_check_session=None, after_layout=None, collect_script=None, retry_health_check=False,
                     skip_pre_check_within_s=None, instance=None):
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
    final hint; retry_health_check: only submit the health check after the already submitted chunks again
    (_retry_post_health_check(); run_folder comes from health_check_retry_folder());
    skip_pre_check_within_s: no health check before the submission when the run's latest job (chunk or health check)
    was submitted less than this many seconds ago, unless that job is a degraded health check or one before the chunks that was
    never collected (None: always check); instance: IBM's (instance name, plan), recorded in the
    run's run_info.json (an existing record is kept). Returns the run folder path."""
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

    if retry_health_check:
        _retry_post_health_check(run_folder=run_folder, backend_name=backend_name, service_name=service_name,
                                 real_backend=real_backend, n_chunks=n_wanted, nshots=nshots,
                                 session=health_check_session or (lambda: contextlib.nullcontext({})))
        return Path("datafiles") / run_folder

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
    if instance is not None:
        qbcs.run_instance = tuple(instance)

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
    age_s, unsure_number = _seconds_since_last_submission(f"datafiles/{run_folder}") if skip_pre_check_within_s else (None, None)
    if unsure_number is not None and age_s < skip_pre_check_within_s:
        print(f"  Health check #{unsure_number}, the run's last job, was degraded or never collected: checking again before the submission.")
    elif age_s is not None and age_s < skip_pre_check_within_s:
        print(f"  Health check before the submission skipped: the run's last job was submitted {age_s / 60:.1f} min ago "
              f"(< {skip_pre_check_within_s / 60:g} min)")
    else:
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
