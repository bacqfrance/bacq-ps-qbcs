"""
run_collection.py
The steps every qbcs_real_collect_*.py script shares, after it has connected to its provider:
collect the run's health checks (warn only), collect the finished chunks (non-blocking), then
merge and analyze. A collect script only parses its arguments, connects, and calls these
(collect_runs() for every run folder it found).
"""

import json
import sys
from pathlib import Path

from source.core.qbcs import bacq_ps_qbcs
from source.core.health_check import collect_health_check
from source.core.records.health_check_records import format_chunk_range, load_health_checks
from source.core.common import exit_with_error, print_done_block, print_run_status_banner
from source.core.records.job_records import chunk_record_files
from source.core.records.chunk_scope import collected_chunks
from source.core.records.run_names import subset_suffix
from source.core.runner_support.benchmark_args import DT


def collect_health_checks(path, service, service_name, backend_name=None):
    """Collect every uncollected health check of the run at `path` and print their verdicts.
    A health check that cannot be collected (any Exception) is only warned about; an MQSS_LRZ record
    without backend_name still ends the script through exit_with_error (old records to upgrade)."""
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


def run_chunk_status(path, n_chunks):
    """Where the chunks [0, n_chunks) of the run at `path` stand, from the files on disk: {"collected": [...],
    "failed": [...] (the current record is 'failed': resubmit), "unsubmitted": [...] (no record), "pending": [...]
    (submitted, not collected yet: run the collection again)}."""
    done    = set(collected_chunks(path))
    records = chunk_record_files(Path(path) / "job_records")
    out     = {"collected": [], "failed": [], "unsubmitted": [], "pending": []}
    for nc in range(n_chunks):
        if nc in done:
            out["collected"].append(nc)
            continue
        try:
            status = json.loads(records[nc].read_text(encoding="utf-8")).get("status") if nc in records else None
        except (OSError, ValueError):
            status = None
        out["unsubmitted" if status is None else "failed" if status == "failed" else "pending"].append(nc)
    return out


def print_run_status(path, n_chunks):
    """The closing banner of a collection (print_run_status_banner()): a complete run, or an incomplete one with
    its pending, failed and unsubmitted chunks and what to do about each. Returns the run_chunk_status() dict."""
    st = run_chunk_status(path, n_chunks)
    n_done = len(st["collected"])
    if n_done == n_chunks:
        print_run_status_banner(f"RUN COMPLETE — all {n_chunks} chunks collected", [], True)
        return st
    lines = []
    if st["pending"]:
        lines.append(f"Pending  (chunks {format_chunk_range(st['pending'])}): jobs not finished yet — run this collection script again later.")
    if st["failed"]:
        lines.append(f"Failed   (chunks {format_chunk_range(st['failed'])}): jobs ended in error or were cancelled — rerun the submit script (it resubmits only these).")
    if st["unsubmitted"]:
        lines.append(f"Not submitted (chunks {format_chunk_range(st['unsubmitted'])}) — rerun the submit script.")
    print_run_status_banner(f"RUN INCOMPLETE — {n_done} of {n_chunks} chunks collected", lines, False)
    return st


def collect_and_analyze(path, service, service_name, bench, lnn, do_spam=True, backend_name=None, dt=DT, **collect_kw):
    """Collect the run's finished chunks (any order, no waiting for the rest), merge them and analyze, then print
    the run's complete / incomplete banner. Returns run_chunk_status().
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
        dt                          = dt,
    )
    qbcs.collect_results(exp_results_folder_path=path, service=service, lang="qiskit",
                         service_name=service_name, backend_name=backend_name, **collect_kw)
    if not qbcs.CONFIG.get("folder_path"):      # set once a chunk has been collected
        print("\n  Nothing to analyze yet — run this script again when the jobs have finished.")
        return print_run_status(path, bench.sample_size)
    qbcs._use_circuit_LNN = lnn
    qbcs._status          = "submitted"
    qbcs.analyze_results()

    st = run_chunk_status(path, bench.sample_size)
    complete = len(st["collected"]) == bench.sample_size
    print_done_block("Collection and analysis complete." if complete else "The collected chunks are analyzed (partial run).",
                     [("Results",                  f"{path}/processed_results/run_data.json" if complete else
                                                   f"{path}/processed_results/ (files with {subset_suffix(len(st['collected']))})"),
                      ("Results (SPAM corrected)", f"{path}/processed_results/run_data_spam.json" if do_spam and complete else None)])
    return print_run_status(path, bench.sample_size)


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
    return collect_and_analyze(path, service, service_name, bench, lnn, do_spam, backend_name, **collect_kw)


def collect_runs(paths, service, service_name, backend_name, bench, lnn, do_spam=True, **collect_kw):
    """collect_run() for each run folder of `paths` in turn (a header line when there are several). A failure
    in one (an exception or an exit) prints its ❌ line and the next one still runs; exit code 2 at the end
    when any failed. With several folders, a summary of every run (complete / incomplete / failed) closes the output."""
    failed, summary = [], []
    for i, path in enumerate(paths, 1):
        if len(paths) > 1:
            print(f"\n[run {i}/{len(paths)}] {path}")
        try:
            st = collect_run(path, service, service_name, backend_name, bench, lnn, do_spam, **collect_kw)
            n_done = len(st["collected"])
            what = f"{n_done} of {bench.sample_size} chunks collected, " + (
                f"{len(st['failed'])} failed (chunks {format_chunk_range(st['failed'])})" if st["failed"] else "0 failed")
            what += "".join(f", {len(st[k])} {k}" for k in ("pending", "unsubmitted") if st[k])
            summary.append((path, what, n_done == bench.sample_size))
        except (Exception, SystemExit) as e:
            if not isinstance(e, SystemExit):
                print(f"❌ {path}: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(path)
            summary.append((path, "collection failed", False))
    if len(paths) > 1:
        n_ok = sum(ok for _, _, ok in summary)
        print_run_status_banner(f"{n_ok} of {len(paths)} runs complete",
                                [f"{'✅' if ok else '⚠️ '} {p}: {what}" for p, what, ok in summary], n_ok == len(paths))
    if failed:
        exit_with_error(f"Collection failed for {len(failed)} of {len(paths)} run folder(s): {', '.join(failed)}")
