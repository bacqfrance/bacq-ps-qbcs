"""
run_local_simulation.py
What every source/benchmark_scripts/local_simulation_scripts/benchmark_*.py script does once it has
parsed its arguments: construct the pipeline, build() the circuits (timed), print the run
configuration, health check before and after, simulate every chunk still without results (service
"local"), write the submit.log entry, analyze and print the timing and the closing summary. The
scripts differ only in the run name, the pipeline's noise arguments, the topology and the sampler
(noiseless Aer, or Aer with the noise of a calibration snapshot), so only those stay in each script.

Light at import (no qiskit, no myQLM): everything heavy is imported inside run_local_simulation().
"""

import time

from source.core.runner_support.benchmark_args import DT


def run_local_simulation(*, run_folder, bench, lnn, backend_label, do_spam=True, sampler_factory=None,
                         extra_qbcs_kwargs=None, extra_config=None, noisy_setup=None):
    """Build, simulate and analyze run_folder as the local benchmark scripts do.

    bench: BenchmarkArgs (source/core/runner_support/benchmark_args.py); lnn: the circuit topology;
    backend_label: printed and logged as the backend; do_spam: with the SPAM circuits and mitigation or without; extra_qbcs_kwargs: further bacq_ps_qbcs arguments
    (the coherent-noise ones); extra_config: further (label, value) lines for print_run_config();
    sampler_factory(): a new sampler per use (the noiseless runs: Aer's SamplerV2).
    noisy_setup(): for a run on a calibration snapshot, called once the clock has started, returns
    (snapshot_path, snapshot_backend, sampler); the qubits are then chosen on the snapshot's backend, the
    health check only warns when degraded, and submit.log records the qubits and the snapshot."""
    from source.core.qpu.providers import ACCOMPANYING_HEALTH_CHECK_NSHOTS
    from source.core.qbcs import bacq_ps_qbcs
    from source.core.common import print_done_block
    from source.core.qpu.calibration_input import layout_summary, resolve_layout_or_exit
    from source.core.runner_support.benchmark_args import LayoutArgs
    from source.core.health_check import submit_health_check
    from source.core.records.health_check_records import (load_health_checks, format_chunk_range, install_submit_crash_log,
                                                   append_submit_log, health_check_log_line, print_run_config)

    t0 = time.perf_counter()
    noisy = noisy_setup is not None
    if noisy:
        snapshot_path, snapshot_backend, sampler = noisy_setup()
        sampler_factory = lambda: sampler   # one sampler for every use

    qbcs = bacq_ps_qbcs(
        **(extra_qbcs_kwargs or {}),
        sample_size                 = bench.sample_size,
        list_circuit_times          = bench.list_times,
        chunk_size                  = int(len(bench.list_times) * 6),   # 6 circuits per depth
        error_mitigation_strategies = ["SPAM", "NEC", "RC+NEC", "cRC+NEC"] if do_spam else ["NEC", "RC+NEC", "cRC+NEC"],
        do_SPAM                     = do_spam,
        nshots                      = bench.nshots,
        dt                          = DT,
        rng_seed                    = 0,         # change this to get a different set of twirled circuits
    )

    qbcs.build(circuit_transpilation_LNN=lnn)   # compile + interleave chunks (see USER_MANUAL §5.4); the cached circuits when noisy
    run_qubits = resolve_layout_or_exit(qbcs, snapshot_backend, run_folder=run_folder) if noisy else None
    t1 = time.perf_counter()
    print(f"build time:   {t1 - t0:.1f} s")

    extra = [("Backend", backend_label)] + list(extra_config or [])
    if noisy:
        extra.append(("Run qubits", run_qubits))
    print_run_config(run_folder, nshots=bench.nshots, sample_size=bench.sample_size, lnn=lnn, do_spam=do_spam,
                     list_times=bench.list_times, chunk_size=qbcs.CONFIG["chunk_size"], extra=extra)

    if not noisy:
        qbcs.record_run_info(run_folder)   # recorded before the health check, so an aborted run is still analyzable

    # The chunks this run simulates (those without complete results); the health checks before / after
    # the simulation (datafiles/<run>/health_checks/) record this list.
    todo_chunks = qbcs.chunks_to_submit(run_folder)
    install_submit_crash_log(f"datafiles/{run_folder}", todo_chunks)   # an uncaught error (e.g. the assert below) still leaves an entry in submit.log
    print(f"  Chunks to simulate: {format_chunk_range(todo_chunks) or 'none'} ({len(todo_chunks)})")

    hc_layout = {"layout_backend": snapshot_backend} if noisy else {}
    if todo_chunks:
        hc_pre = load_health_checks(submit_health_check(run_folder=run_folder, backend=sampler_factory(), nshots=ACCOMPANYING_HEALTH_CHECK_NSHOTS,
                                                        service_name="local", backend_name=backend_label, **hc_layout,
                                                        accompanying=True, chunks=todo_chunks))[-1]
        print(f"  Health check: P(011)={hc_pre['P011']:.3f} → {hc_pre['verdict']}")
        if not noisy:
            assert hc_pre["verdict"] == "functional", "health check failed before the submission"
        elif hc_pre["verdict"] != "functional":
            print("  ⚠️  Simulated QPU scores as degraded with this calibration — continuing anyway.")

    submit_layout = {"layout_backend": snapshot_backend,   # required for the noisy path: transpile onto the snapshot's qubits
                     "layout_connection": "linear_nearest_neighbor" if lnn else "all_to_all"} if noisy else {}
    run_folder_path = qbcs.submit(   # also run with nothing left: it merges the results for analyze_results()
        run_folder = run_folder,
        lang       = "qiskit",
        service    = "local",
        backend    = sampler_factory(),
        **submit_layout,
        chunks     = todo_chunks,
    )

    if todo_chunks:
        hc_post = load_health_checks(submit_health_check(run_folder=run_folder, backend=sampler_factory(), nshots=ACCOMPANYING_HEALTH_CHECK_NSHOTS,
                                                         service_name="local", backend_name=backend_label, **hc_layout,
                                                         accompanying=True, after_submission=True,
                                                         chunks=todo_chunks))[-1]
        print(f"  Health check after the simulation: P(011)={hc_post['P011']:.3f} → {hc_post['verdict']}")
        log_layout = ({"layout": layout_summary(LayoutArgs("optimized"), run_qubits), "run_qubits": run_qubits,
                       "snapshot_path": snapshot_path} if noisy else {})   # resolve_layout_or_exit above chose the default "optimized"
        log_path = append_submit_log(run_folder_path, backend=backend_label, chunks=todo_chunks, n_chunks=qbcs.n_chunks,
                                     nshots=bench.nshots, **log_layout,
                                     extra_lines=[health_check_log_line("PRE ", hc_pre),
                                                  health_check_log_line("POST", hc_post)])
    t2 = time.perf_counter()
    print(f"Local execution time:  {t2 - t1:.1f} s")

    qbcs.analyze_results()   # writes processed_results/run_data.json (+ _spam.json)
    t3 = time.perf_counter()
    print(f"analysis time: {t3 - t2:.1f} s")
    print(f"total wall time:   {t3 - t0:.1f} s")

    print_done_block(f"{len(todo_chunks)} chunk(s) simulated: {format_chunk_range(todo_chunks) or 'none'}",
                     [("Run folder", f"datafiles/{run_folder}"),
                      ("Job records", f"datafiles/{run_folder}/job_records/"),
                      ("Log saved", f"datafiles/{run_folder}/{log_path.name}" if todo_chunks else None)])
