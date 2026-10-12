"""
run_collection.py
The steps every qbcs_real_collect_*.py script shares, after it has connected to its provider:
collect the run's health checks (warn only), collect the finished chunks (non-blocking), then
merge and analyze. A collect script only parses its arguments, connects, and calls these
(collect_runs() for every run folder it found).
"""

import sys
from pathlib import Path

from source.core.qbcs import bacq_ps_qbcs
from source.core.health_check import collect_health_check
from source.core.records.health_check_records import load_health_checks
from source.core.common import exit_with_error, print_done_block
from source.core.runner_support.benchmark_args import DT


def collect_health_checks(path, service, service_name, backend_name=None):
    """Collect every uncollected health check of the run at `path` and print their verdicts.
    Never raises: a health check that cannot be collected is only warned about."""
    try:
        collect_health_check(exp_results_folder_path=path, service=service, service_name=service_name,
                             accompanying=True, backend_name=backend_name)
        hcs = load_health_checks(path)
        if not hcs:
            print(f"  ⚠️  No health checks found in {path}/health_checks — nothing to collect.")
        for hc in hcs:
            if hc.get("P011") is not None:
                print(f"  Health check #{hc['number']}: {'✅' if hc['verdict'] == 'functional' else '⚠️'} "
                      f"P011={hc['P011']:.3f} — {hc['verdict']}")
        if any(hc.get("verdict") == "degraded" for hc in hcs):
            print("  ⚠️  QPU degraded during the run — results may be less reliable.")
    except Exception as e:
        print(f"  ⚠️  Health checks not available: {e}")


def collect_and_analyze(path, service, service_name, bench, lnn, do_spam=True, backend_name=None, **collect_kw):
    """Collect the run's finished chunks (any order, no waiting for the rest), merge them and analyze.
    `bench`: the run's BenchmarkArgs; the chunk layout (6 circuits per depth) is the one submitted.
    collect_kw: passed on to collect_results() (e.g. poll_interval)."""
    qbcs = bacq_ps_qbcs(
        infidelity_total            = 0.0,
        ratio_ct                    = 0.0,
        noise_type                  = "pauli",
        sample_size                 = bench.sample_size,
        chunk_size                  = int(len(bench.list_times) * 6),   # 6 circuits per depth, as submitted
        list_circuit_times          = bench.list_times,
        error_mitigation_strategies = ["SPAM", "NEC", "RC+NEC", "cRC+NEC"] if do_spam else ["NEC", "RC+NEC", "cRC+NEC"],
        do_SPAM                     = do_spam,
        nshots                      = bench.nshots,
        dt                          = DT,
    )
    qbcs.collect_results(exp_results_folder_path=path, service=service, lang="qiskit",
                         service_name=service_name, backend_name=backend_name, **collect_kw)
    if not qbcs.CONFIG.get("folder_path"):      # set once a chunk has been collected
        print("\n  Nothing to analyze yet — run this script again when the jobs have finished.")
        return
    qbcs._use_circuit_LNN = lnn
    qbcs._status          = "submitted"
    qbcs.analyze_results()

    print_done_block("Collection and analysis complete.",
                     [("Results",                  f"{path}/processed_results/run_data.json"),
                      ("Results (SPAM corrected)", f"{path}/processed_results/run_data_spam.json" if do_spam else None)])


def collect_run(path, service, service_name, backend_name, bench, lnn, do_spam=True, **collect_kw):
    """What every qbcs_real_collect_*.py script does once connected: print the run, collect its health
    checks (warn only), then its finished chunks — non-blocking, run again later for the rest — and
    merge and analyze them (collect_health_checks(), collect_and_analyze())."""
    print(f"\n  Collecting BACQ results from {backend_name}...")
    print(f"  Run folder: {path}")
    print("  Collecting the finished jobs (no waiting) — run again later for the rest.\n")
    collect_health_checks(path, service, service_name, backend_name)
    if not (Path(path) / "job_records").exists():   # the submit script stopped before any chunk (e.g. a degraded pre-submission check)
        exit_with_error(f"No chunk was submitted for {path} (a degraded pre-submission health check, or the submit script "
                        f"stopped before it got that far) — nothing to collect. Rerun the submit script.")
    collect_and_analyze(path, service, service_name, bench, lnn, do_spam, backend_name, **collect_kw)


def collect_runs(paths, service, service_name, backend_name, bench, lnn, do_spam=True, **collect_kw):
    """collect_run() for each run folder of `paths` in turn (a header line when there are several). A failure
    in one (an exception or an exit) prints its ❌ line and the next one still runs; exit code 2 at the end
    when any failed."""
    failed = []
    for i, path in enumerate(paths, 1):
        if len(paths) > 1:
            print(f"\n[run {i}/{len(paths)}] {path}")
        try:
            collect_run(path, service, service_name, backend_name, bench, lnn, do_spam, **collect_kw)
        except (Exception, SystemExit) as e:
            if not isinstance(e, SystemExit):
                print(f"❌ {path}: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(path)
    if failed:
        exit_with_error(f"Collection failed for {len(failed)} of {len(paths)} run folder(s): {', '.join(failed)}")
