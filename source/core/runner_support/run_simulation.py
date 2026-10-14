"""
run_simulation.py
What every qbcs_real_simulate_*.py script does once it has the simulated QPU's calibration
snapshot: build an Aer noise model from it, build() the circuits (from the cache), pick the run's
physical qubits as a real submission would, health check before and after, simulate every chunk
still without results (service "local"), write the submit.log entry and analyze. The scripts
differ only in how they get the snapshot (download, provider file, VLQ's calibration API), so
that part stays in each script.

Light at import (no qiskit, no myQLM): everything heavy is imported inside simulate_run().
"""

import time

from source.core.common import print_done_block
from source.core.runner_support.benchmark_args import DT


def simulate_run(*, run_folder, bench, layout, lnn, snapshot_path, cal_source, do_spam=True, dt=DT,
                 seed=None, customize_backend=None):
    """Simulate run_folder's chunks with the noise of the calibration snapshot at snapshot_path.

    bench / layout: BenchmarkArgs / LayoutArgs (source/core/runner_support/benchmark_args.py); lnn: the QPU's topology
    (backends.json); cal_source: the calibration_file= the noise (and, for an optimized layout, the layout scoring) comes from (None: the snapshot);
    seed: Aer sampling seed (None: fresh randomness per chunk); customize_backend(snapshot_backend):
    an optional hook run before the qubits are chosen (VLQ attaches its raw calibration there)."""
    from source.core.qpu.calibration_input import layout_summary, resolve_layout_or_exit
    from source.core.records.health_check_records import (load_health_checks, format_chunk_range, install_submit_crash_log,
                                                   append_submit_log, health_check_log_line, print_run_config)
    from source.core.qpu.providers import ACCOMPANYING_HEALTH_CHECK_NSHOTS
    from source.core.qbcs import bacq_ps_qbcs
    from source.core.health_check import submit_health_check
    from source.core.qpu.qpu_snapshot import noisy_sampler_from_snapshot

    t0 = time.perf_counter()
    snapshot_backend, noise_model, sampler = noisy_sampler_from_snapshot(snapshot_path, seed=seed)
    if customize_backend is not None:
        customize_backend(snapshot_backend)
    print(f"  [Noise model] {sorted(noise_model.noise_instructions)}")

    # ── Build (compile circuits — reuses the shared circuit cache) ────────────────
    nshots     = bench.nshots
    list_times = bench.list_times
    chunk_size = int(len(list_times) * 6)           # 6 circuits per depth
    qbcs = bacq_ps_qbcs(
        infidelity_total            = 0.0,
        ratio_ct                    = 0.0,
        noise_type                  = "pauli",
        sample_size                 = bench.sample_size,
        chunk_size                  = chunk_size,
        list_circuit_times          = list_times,
        error_mitigation_strategies = ["SPAM", "NEC", "RC+NEC", "cRC+NEC"] if do_spam else ["NEC", "RC+NEC", "cRC+NEC"],
        do_SPAM                     = do_spam,
        nshots                      = nshots,
        dt                          = dt,
    )
    qbcs.build(circuit_transpilation_LNN=lnn)

    # ── Physical qubits — before the first health check, so it runs on them too ───
    run_qubits = resolve_layout_or_exit(qbcs, snapshot_backend, run_folder=run_folder, layout=layout,
                                        calibration_source=cal_source)
    t1 = time.perf_counter()
    backend = f"local Aer simulator (noise: {snapshot_backend.name})"
    print_run_config(run_folder, nshots=nshots, sample_size=bench.sample_size, lnn=lnn, do_spam=do_spam,
                     list_times=list_times, chunk_size=chunk_size, extra=[("Backend", backend), ("Run qubits", run_qubits)])

    # ── The chunks to simulate (those without complete results) — the health checks record this list ──
    todo = qbcs.chunks_to_submit(run_folder)
    install_submit_crash_log(f"datafiles/{run_folder}", todo)   # an uncaught error still leaves an entry in submit.log
    print(f"  Chunks to simulate: {format_chunk_range(todo) or 'none'} ({len(todo)})")

    def health_check(**kw):
        return load_health_checks(submit_health_check(run_folder=run_folder, backend=sampler,
                                                      nshots=ACCOMPANYING_HEALTH_CHECK_NSHOTS, service_name="local",
                                                      layout_backend=snapshot_backend, accompanying=True, backend_name=backend,
                                                      chunks=todo, **kw))[-1]

    # ── Health check before the simulation (<run>/health_checks/, auto-numbered) ───
    if todo:
        hc_pre = health_check()
        print(f"  Health check: P(011)={hc_pre['P011']:.3f} → {hc_pre['verdict']}")
        if hc_pre["verdict"] != "functional":
            print("  ⚠️  Simulated QPU scores as degraded with this calibration — continuing anyway.")

    # ── Simulate (synchronous, local; also with nothing left: it merges the results for the analysis) ──
    run_folder_path = qbcs.submit(
        run_folder        = run_folder,
        lang              = "qiskit",
        service           = "local",
        backend           = sampler,
        layout_backend    = snapshot_backend,
        layout_connection = "linear_nearest_neighbor" if lnn else "all_to_all",
        chunks            = todo,
    )
    if todo:
        hc_post = health_check(after_submission=True)
        print(f"  Health check after the simulation: P(011)={hc_post['P011']:.3f} → {hc_post['verdict']}")
        log_path = append_submit_log(run_folder_path, backend=backend, chunks=todo, n_chunks=qbcs.n_chunks, nshots=nshots,
                                     layout=layout_summary(layout, run_qubits, cal_source), run_qubits=run_qubits,
                                     snapshot_path=snapshot_path,
                                     extra_lines=[health_check_log_line("PRE ", hc_pre),
                                                  health_check_log_line("POST", hc_post)])
    t2 = time.perf_counter()

    qbcs.analyze_results()   # processed_results/run_data.json (+ _spam.json)
    t3 = time.perf_counter()
    print(f"\nBuild {t1 - t0:.1f} s | simulation {t2 - t1:.1f} s | analysis {t3 - t2:.1f} s | total wall time {t3 - t0:.1f} s")
    print_done_block(f"{len(todo)} chunk(s) simulated: {format_chunk_range(todo) or 'none'}",
                     [("Run folder",  f"datafiles/{run_folder}"),
                      ("Job records", f"datafiles/{run_folder}/job_records/"),
                      ("Log saved",   f"datafiles/{run_folder}/{log_path.name}" if todo else None)])
    return run_folder_path
