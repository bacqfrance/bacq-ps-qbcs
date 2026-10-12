"""
job_status_check.py
The shared body of the check_real_qpu_job_status_<qpu>.py scripts (IBM, IQM EQE1, IQM Resonance,
IQM VLQ): read a run's job records from disk, ask the provider for each job's live status, print
the jobs that are not done, a summary and a provider overview. Does not wait and changes nothing.
Each script supplies only what is provider-specific (the connection and three small callables),
so all of them print the same report.

No qiskit, no myQLM and no provider SDK import here: every venv imports it.
"""

import json
import sys

from source.core.common import PROJECT_ROOT, exit_with_error, section
from source.core.qpu.providers import job_state_of_status

HEALTH_CHECK_WORD = "health_check"   # bare command-line word: check the standalone health check instead of the benchmark run


def pop_health_check_flag(argv):
    """(whether the `health_check` word is in argv, argv without it) — to be called before the
    benchmark/backend argument parsers, which do not know the word."""
    return HEALTH_CHECK_WORD in argv, [a for a in argv if a != HEALTH_CHECK_WORD]


def classify(status):
    """DONE / ERROR / CANCELLED / PENDING from a provider's status text (an enum's repr or value)."""
    state, s = job_state_of_status(status), str(status).upper()
    if state == "failed":                       # cancelled counts as failed; the report only shows it apart
        return "ERROR" if "ERROR" in s or "FAIL" in s else "CANCELLED"
    return state.upper()                        # DONE / PENDING


def _load_records(folder, health_check):
    """[(label, job_id, status on disk)] of a run folder, in chunk order — or of its one health-check record."""
    if health_check:
        paths = PROJECT_ROOT.glob(f"datafiles/{folder}/job_records/health_check.json")
    else:
        paths = PROJECT_ROOT.glob(f"datafiles/{folder}/job_records/job_record_chunk_*.json")
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        label = "Health check" if health_check else f"Chunk {rec['chunk_id']:3d}"
        key   = -1 if health_check else rec["chunk_id"]
        records.append((key, label, str(rec["job_id"]), rec["status"]))
    return [r[1:] for r in sorted(records)]


def run_status_check(provider, run_folder, health_check_folder, health_check, remote_status,
                     overview=None, recent_jobs=None, n_recent=10):
    """Print the status report of a run (or, with health_check, of the standalone health check).

    provider            label printed in front of the live status (the script's BACKEND_NAME)
    run_folder          the benchmark run folder under datafiles/
    health_check_folder the standalone health-check folder under datafiles/
    remote_status       job_id -> the provider's status text; raises when the job cannot be queried
    overview            optional () -> list of lines: the state of the QPU/provider (queue length, …)
    recent_jobs         optional n -> list of lines: the account's latest jobs; None when the
                        provider's API has no job listing
    """
    folder  = health_check_folder if health_check else run_folder
    records = _load_records(folder, health_check)
    if not records:
        exit_with_error(f"No {'health-check' if health_check else 'chunk'} job record found in datafiles/{folder}/job_records/.")

    what = "health-check job record" if health_check else "chunk job record"
    print(f"\nFound {len(records)} {what}(s) in {folder}:\n")

    live   = {}                                   # job_id -> status text, queried once per job
    counts = {}
    for label, job_id, status_disk in records:
        if job_id not in live:
            try:
                live[job_id] = str(remote_status(job_id))
            except Exception as e:
                live[job_id] = f"UNAVAILABLE — {type(e).__name__}: {e}"
        status_live = live[job_id]
        kind = "UNAVAILABLE" if status_live.startswith("UNAVAILABLE") else classify(status_live)
        counts[kind] = counts.get(kind, 0) + 1
        if kind != "DONE":                        # only print the jobs that are not done
            print(f"  {label} | job_id={job_id} | disk={status_disk} | {provider}={status_live}")

    section("Summary")
    print(f"  Total records:  {len(records)}")
    print(f"  Distinct jobs:  {len(live)}")
    for kind in ("DONE", "PENDING", "ERROR", "CANCELLED", "UNAVAILABLE"):
        if counts.get(kind):
            print(f"  {kind:15s}: {counts[kind]}")
    n_disk_complete = sum(1 for _, _, s in records if s == "complete")
    print(f"  {'complete_disk':15s}: {n_disk_complete}")

    if overview is not None:
        print(f"\n{provider} overview:")
        try:
            for line in overview():
                print(f"  {line}")
        except Exception as e:
            print(f"  unavailable ({type(e).__name__}: {e})")

    print(f"\nRecent {provider} jobs (last {n_recent}):")
    if recent_jobs is None:
        print(f"  not available — the {provider} API has no job listing")
    else:
        try:
            for line in recent_jobs(n_recent):
                print(f"  {line}")
        except Exception as e:
            print(f"  unavailable ({type(e).__name__}: {e})")


def run_status_checks(provider, run_folders, health_check_folder, health_check, remote_status,
                      overview=None, recent_jobs=None, n_recent=10):
    """run_status_check() for each run folder of `run_folders` in turn (a header line when there are several;
    one report for the standalone health check). A failure in one (an exception or an exit) prints its ❌
    line and the next one still runs; exit code 2 at the end when any failed. Same arguments otherwise."""
    folders = [None] if health_check or not run_folders else list(run_folders)
    failed = []
    label = lambda f: f"datafiles/{f}" if f is not None else (
        f"datafiles/{health_check_folder}" if health_check_folder else "the health check")
    for i, folder in enumerate(folders, 1):
        if len(folders) > 1:
            print(f"\n[run {i}/{len(folders)}] {label(folder)}")
        try:
            run_status_check(provider, folder, health_check_folder, health_check, remote_status,
                             overview=overview, recent_jobs=recent_jobs, n_recent=n_recent)
        except (Exception, SystemExit) as e:
            if not isinstance(e, SystemExit):
                print(f"❌ {label(folder)}: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(folder)
    if failed:
        exit_with_error(f"Status check failed for {len(failed)} of {len(folders)} run folder(s): "
                        f"{', '.join(label(f) for f in failed)}")
