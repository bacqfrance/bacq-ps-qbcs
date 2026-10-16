"""
job_cancel.py
The shared body of the cancel_real_qpu_jobs_<qpu>.py scripts (IBM, IQM EQE1, IQM Resonance, IQM VLQ):
read a run's job records from disk, ask the provider for each job's live status and cancel the
jobs that are still queued or running. Jobs that already ended (done, failed, cancelled), records
without a job (never submitted, "N/A") and repeated job ids are left alone. The records on disk are
not changed: the collect script (or the health check's own rerun) updates them afterwards.
Each script supplies only what is provider-specific (the connection and two small callables).

No qiskit, no myQLM and no provider SDK import here: every venv imports it.
"""

import sys

from source.core.common import exit_with_error, section
from source.core.qpu.job_status_check import _load_records, classify

NO_JOB_IDS = ("None", "N/A", "")   # what a record without a submitted job holds as job_id (str() of it)


def run_cancel(provider, run_folder, health_check_folder, health_check, remote_status, cancel_job):
    """Cancel the jobs of a run (or, with health_check, of the standalone health check) that have not ended.

    provider            label printed in front of the live status (the script's BACKEND_NAME)
    run_folder          the benchmark run folder under datafiles/
    health_check_folder the standalone health-check folder under datafiles/
    remote_status       job_id -> the provider's status text; raises when the job cannot be queried
    cancel_job          job_id -> None; asks the provider to cancel the job, raises when it cannot

    Returns the number of jobs for which the cancellation failed or whose status could not be read.
    """
    folder  = health_check_folder if health_check else run_folder
    records = _load_records(folder, health_check)
    if not records:
        exit_with_error(f"No {'health-check' if health_check else 'chunk'} job record found in datafiles/{folder}/job_records/.")

    what = "health-check job record" if health_check else "chunk job record"
    print(f"\nFound {len(records)} {what}(s) in {folder}:\n")

    labels = {}                                   # job_id -> the labels of the records that name it
    for label, job_id, _ in records:
        labels.setdefault(job_id, []).append(label.strip())

    counts = {"requested": 0, "ended": 0, "not submitted": 0, "failed": 0}
    for job_id, job_labels in labels.items():
        who = f"{', '.join(job_labels)} | job_id={job_id}"
        if job_id in NO_JOB_IDS:
            counts["not submitted"] += len(job_labels)
            continue
        try:
            status = str(remote_status(job_id))
        except Exception as e:
            print(f"❌ {who}: status unavailable — {type(e).__name__}: {e}", file=sys.stderr)
            counts["failed"] += 1
            continue
        if classify(status) != "PENDING":
            print(f"  {who} | {provider}={status} — already ended, left alone")
            counts["ended"] += 1
            continue
        try:
            cancel_job(job_id)
        except Exception as e:
            print(f"❌ {who} | {provider}={status}: cancel failed — {type(e).__name__}: {e}", file=sys.stderr)
            counts["failed"] += 1
            continue
        try:
            after = str(remote_status(job_id))
        except Exception as e:
            after = f"UNAVAILABLE — {type(e).__name__}: {e}"
        print(f"  {who} | {provider}={status} — cancellation requested (status now {after})")
        counts["requested"] += 1

    section("Summary")
    print(f"  Distinct jobs:          {len(labels)}")
    print(f"  Cancellation requested: {counts['requested']}")
    print(f"  Already ended:          {counts['ended']}")
    if counts["not submitted"]:
        print(f"  No job (not submitted): {counts['not submitted']}")
    if counts["failed"]:
        print(f"  Failed:                 {counts['failed']}")
    if counts["requested"]:
        print("\nThe records on disk are unchanged and a provider may take a moment to settle a cancelled job: "
              "run the collect script afterwards (same arguments) to update their status.")
    return counts["failed"]


def run_cancels(provider, run_folders, health_check_folder, health_check, remote_status, cancel_job):
    """run_cancel() for each run folder of `run_folders` in turn (a header line when there are several;
    one pass for the standalone health check). A failure in one (an exception or an exit) prints its ❌
    line and the next one still runs; exit code 2 at the end when any failed. Same arguments otherwise."""
    folders = [None] if health_check or not run_folders else list(run_folders)
    failed = []
    label = lambda f: f"datafiles/{f}" if f is not None else (
        f"datafiles/{health_check_folder}" if health_check_folder else "the health check")
    for i, folder in enumerate(folders, 1):
        if len(folders) > 1:
            print(f"\n[run {i}/{len(folders)}] {label(folder)}")
        try:
            if run_cancel(provider, folder, health_check_folder, health_check, remote_status, cancel_job):
                failed.append(folder)
        except (Exception, SystemExit) as e:
            if not isinstance(e, SystemExit):
                print(f"❌ {label(folder)}: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(folder)
    if failed:
        exit_with_error(f"Cancelling failed for {len(failed)} of {len(folders)} run folder(s): "
                        f"{', '.join(label(f) for f in failed)}")
