"""
submit_with_retries.py
The loop of the qbcs_real_submit_<qpu>_with_retries.py scripts (IQM EQE1, IQM VLQ), the Python counterpart of
run_aqt_loop_with_retries.sbatch for QPUs without Slurm. It runs the QPU's submit script, which stops on the first
provider error, again and again until every chunk is submitted. By default (VLQ) it waits before each launch until the latest job
already submitted to the run has left the provider's queue; with wait_for_jobs=False (EQE1) it relaunches without
waiting for any job. After a failed launch it waits a QPU-specific time (VLQ: only when the launch submitted no chunk).

Light at import (no qiskit, no myQLM, no provider SDK): the provider enters as a status callable.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from source.core.common import PROJECT_ROOT, read_json_or_none, rel_path
from source.core.qpu.providers import job_state_of_status
from source.core.records.health_check_records import load_health_checks
from source.core.records.health_check_records import EXIT_HC_DEGRADED, EXIT_HC_NOT_COLLECTED, EXIT_HC_NOT_SUBMITTED
from source.core.records.job_records import chunk_record_files, chunks_without_submission
from source.core.records.job_timing import parse_utc
from source.core.records.run_names import dated_run_folder, dated_run_folders
from source.core.runner_support.benchmark_args import RETRY_HEALTH_CHECK_WORD

PAUSE_S           = 600     # between two status checks of the latest job
RETRY_LIMIT_HOURS = 48      # give up after this long without progress
EXIT_GAVE_UP      = 14      # the same code run_aqt_loop_with_retries.sbatch ends with after repeated failures
EXIT_ARGUMENTS    = 2       # the submit script's code for unusable arguments: a rerun cannot help
EXIT_DEGRADED     = EXIT_HC_DEGRADED   # a health check found the QPU degraded: the run is aborted, not retried
HC_RETRIES        = 3       # launches of the health check after the last chunks, when its submission failed
RETRY_HC_ARGUMENT = RETRY_HEALTH_CHECK_WORD

EXIT_MEANING = {
    0:                       "all chunks submitted",
    EXIT_ARGUMENTS:          "unusable arguments",
    EXIT_DEGRADED:           "the health check found the QPU degraded",
    EXIT_HC_NOT_COLLECTED:   "a health check was submitted but not collected",
    EXIT_HC_NOT_SUBMITTED:   "a health check could not be submitted (QPU health unknown)",
}


def describe_exit(code):
    """What an exit code of the submit script means."""
    return EXIT_MEANING.get(code, "error (see the output above)")


def _utc_seconds(timestamp):
    """Epoch seconds of an ISO timestamp as the records store them ('…Z', with or without fractions); 0 if unreadable."""
    dt = parse_utc(timestamp)
    return dt.timestamp() if dt else 0.0


def submitted_jobs(run_path):
    """[(submitted_at in epoch s, label, record)] of every record of a run folder that has a job_id: chunk records
    (job_records/job_record_chunk_<n>.json) and accompanying health checks (health_checks/health_check_<n>.json),
    sorted by submission time. A record of a submission that never reached the provider has no job_id and is left out."""
    jobs = []
    for nc, path in chunk_record_files(Path(run_path) / "job_records").items():
        rec = read_json_or_none(path)
        if isinstance(rec, dict) and rec.get("job_id"):
            jobs.append((_utc_seconds(rec.get("submitted_at")), f"chunk {nc}", rec))
    for rec in load_health_checks(run_path):
        if rec.get("job_id"):
            jobs.append((_utc_seconds(rec.get("submitted_at")), f"health check {rec['number']}", rec))
    return sorted(jobs, key=lambda j: j[0])


def latest_submitted_job(run_path):
    """(label, record) of the job submitted last (a chunk or a health check), or None when the run has no submission."""
    jobs = submitted_jobs(run_path)
    return (jobs[-1][1], jobs[-1][2]) if jobs else None


def may_proceed(status):
    """True when the provider's status text of a job is final: completed/done, or failed/cancelled/errored (the submit
    script resubmits failed chunks itself). Anything else (queued, waiting, running, an unknown text, an unavailable
    query) is not final and the loop waits."""
    # a failed query is reported as "UNAVAILABLE — <error>", which job_state_of_status() would read as failed (ERROR/FAIL)
    return job_state_of_status(status) in ("done", "failed") and not str(status).upper().startswith("UNAVAILABLE")


def launch_submit_script(venv_name, script, args, needed_for):
    """Run the QPU's submit script in its venv, its output going straight to ours; its exit code."""
    from source.core.runner_support.venvs import venv_python
    cmd = [venv_python(venv_name, needed_for), "-u", str(script), *args]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode


def run_with_retries(undated, status_of, launch, args, *, wait_after_failed_s, pause_s=PAUSE_S,
                     limit_hours=RETRY_LIMIT_HOURS, sleep=time.sleep, clock=time.time, new_run=False, root="datafiles",
                     wait_for_jobs=True, first_retry_at_once=False, retry_at_once_after_progress=False):
    """Run the loop for the run of the settings `undated` (qpu_run_name() without its date); returns the exit code
    of the wrapper script.

    status_of(job_id, record)  the provider's status text of a job (raises when it cannot be queried);
                               may be None when wait_for_jobs is False
    launch(args)               runs the submit script with these arguments and returns its exit code
    args                       the submit script's arguments, without --force-new-run
    new_run                    --force-new-run was given. Its run is the folder the submit script creates: the first
                               launch gets --force-new-run, and so does every later launch as long as no folder of these
                               settings has appeared since this loop started. Once one has, its run is that folder (the
                               submit script picks it up as the latest unfinished one, whatever its date). Without
                               new_run the run is dated_run_folder(): the latest unfinished folder, else today's new one.
    wait_after_failed_s        pause before the next pass after a launch that failed (0: none)
    wait_for_jobs              False: no status query and no wait for the run's latest job to leave the queue
    first_retry_at_once        True: the first retry after a failure follows at once, the later ones after the pause
    retry_at_once_after_progress  True: a failed launch that submitted a chunk is followed at once, one that submitted
                               none after the pause
    The budget (limit_hours) starts at the first wait or failed launch and restarts after a launch that submitted a chunk."""
    before     = set(dated_run_folders(undated, root))   # folders of these settings that exist before the first launch
    series     = None                       # when the current series of waits / failures began
    attempt    = 0
    failures   = 0                          # failed launches in a row without progress
    last_label = last_status = None

    def current():
        """The run folder name now, or None while a new run has not been created yet."""
        if not new_run:
            return dated_run_folder(undated, root)
        created = [f for f in dated_run_folders(undated, root) if f not in before]
        return created[-1] if created else None

    def job_gate(folder, run_path):
        """None: the run's latest job is final (or there is none), launch. False: it is not, and this waited
        pause_s. An exit code: the budget ran out."""
        nonlocal series, last_label, last_status
        if not wait_for_jobs:
            return None
        job = latest_submitted_job(run_path) if run_path and run_path.is_dir() else None
        if job is None:
            print("  No job submitted to this run yet.", flush=True)
            return None
        label, rec = job
        try:
            status = str(status_of(rec["job_id"], rec))
        except Exception as e:
            status = f"UNAVAILABLE — {type(e).__name__}: {e}"
        last_label, last_status = f"{label} (job {rec['job_id']})", status
        if may_proceed(status):
            print(f"  Latest job: {last_label} — {status}.", flush=True)
            return None
        series = clock() if series is None else series
        if clock() - series >= limit_hours * 3600:
            return _give_up(folder, last_label, last_status, limit_hours)
        print(f"  Latest job: {last_label} — {status}. Next check in {pause_s // 60} min "
              f"({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(clock() + pause_s))}); "
              f"giving up {limit_hours} h after {time.strftime('%Y-%m-%d %H:%M', time.localtime(series))}.", flush=True)
        sleep(pause_s)
        return False

    def retry_final_health_check(folder):
        """All chunks are submitted and only the health check after them failed (exit code 11 or 12): launch the
        submit script with --retry-health-check, at most HC_RETRIES times, each after the latest job is final.
        Never a plain launch: that would start a new run."""
        nonlocal series
        run_path = Path(root) / folder
        rc = 0
        for n in range(1, HC_RETRIES + 1):
            series = None
            while (gate := job_gate(folder, run_path)) is False:
                pass
            if gate is not None:
                return gate
            print(f"\n[health check retry {n}/{HC_RETRIES}] launching the submit script …", flush=True)
            rc = launch([*args, RETRY_HC_ARGUMENT])
            print(f"  Submit script ended with exit code {rc} — {describe_exit(rc)}.", flush=True)
            if rc in (0, EXIT_ARGUMENTS, EXIT_DEGRADED):
                return rc
            if n < HC_RETRIES and wait_after_failed_s and not (first_retry_at_once and n == 1):
                print(f"  Retrying in {_minutes(wait_after_failed_s)}.", flush=True)
                sleep(wait_after_failed_s)
        print(f"❌ All chunks of {rel_path(Path(root) / folder)} are submitted, but the health check after them "
              f"failed {HC_RETRIES} times (exit code {rc}). Resubmit it by starting "
              f"qbcs_real_submit_*.py {RETRY_HC_ARGUMENT} later, or collect the run as it is.", file=sys.stderr)
        return rc

    while True:
        folder   = current()
        run_path = Path(root) / folder if folder else None
        print(f"  Run folder: {folder or 'not created yet (a new run starts with the next launch)'}", flush=True)
        gate = job_gate(folder, run_path)
        if gate is False:                   # not final: waited, check again
            continue
        if gate is not None:                # the budget ran out
            return gate

        todo_before = _chunks_left(run_path) if run_path else 0
        sent_before = _n_submitted(run_path)
        attempt += 1
        force = new_run and folder is None
        print(f"\n[submit pass {attempt}] launching the submit script …", flush=True)
        rc = launch([*args, *(["--force-new-run"] if force else [])])
        print(f"  Submit script ended with exit code {rc} — {describe_exit(rc)}.", flush=True)
        if rc == 0:
            return 0
        if rc == EXIT_ARGUMENTS:
            return EXIT_ARGUMENTS
        if rc == EXIT_DEGRADED:             # the health check says the QPU is degraded: stop, do not retry
            print(f"❌ The health check found the QPU degraded (exit code {EXIT_DEGRADED}): the run is aborted. "
                  f"Check the QPU, then start this script again with the same arguments.", file=sys.stderr)
            return EXIT_DEGRADED
        used = folder if folder and run_path.is_dir() else current()   # the folder this launch worked in
        if rc in (EXIT_HC_NOT_COLLECTED, EXIT_HC_NOT_SUBMITTED) and used and _all_submitted(Path(root) / used):
            return retry_final_health_check(used)      # not the pre-check: that one left chunks, and is retried below
        folder_after = used
        path_after   = Path(root) / folder_after if folder_after else None
        progressed = _chunks_left(path_after) < todo_before or _n_submitted(path_after) > sent_before
        if progressed:
            series = None                   # progress (a chunk submitted): the budget starts again
            failures = 0
        failures += 1
        series = clock() if series is None else series
        if clock() - series >= limit_hours * 3600:
            return _give_up(folder_after, last_label, last_status, limit_hours)
        at_once = (first_retry_at_once and failures == 1) or (retry_at_once_after_progress and progressed)
        if wait_after_failed_s and not at_once:
            print(f"  Retrying in {_minutes(wait_after_failed_s)}.", flush=True)
            sleep(wait_after_failed_s)


def _n_chunks(run_path):
    """The run's chunk count from its run_info.json (0 before it exists, so nothing counts as left)."""
    info = read_json_or_none(Path(run_path) / "run_info.json")
    return int(info["sample_size"]) if isinstance(info, dict) and "sample_size" in info else 0


def _n_submitted(run_path):
    """How many chunks of the run have a job record with a job_id (0 for no run folder yet). Progress of a
    new run, whose run_info.json (and so _chunks_left()) only appears with the first launch."""
    if run_path is None or not Path(run_path).is_dir():
        return 0
    return sum(1 for _, label, _ in submitted_jobs(run_path) if label.startswith("chunk"))


def _all_submitted(run_path):
    """True when the run has a run_info.json and every chunk has a submission record."""
    return _n_chunks(run_path) > 0 and not chunks_without_submission(run_path, _n_chunks(run_path))


def _chunks_left(run_path):
    """How many chunks of the run still have no submission (0 while its run_info.json does not exist, so a folder
    that a failed launch created, or none at all, is no progress). A folder that appears is not progress by itself."""
    if run_path is None:
        return 0
    return len(chunks_without_submission(run_path, _n_chunks(run_path)))


def _minutes(seconds):
    """'1 min' / '10 min' for a wait in whole minutes, else '45 s'."""
    return f"{seconds // 60} min" if seconds % 60 == 0 else f"{seconds:g} s"


def _duration(hours):
    """A budget in hours as '48 h' or, below an hour, '20 min'."""
    return f"{hours:g} h" if hours >= 1 else f"{round(hours * 60)} min"


def _give_up(run_folder, last_label, last_status, limit_hours):
    where = rel_path(Path("datafiles") / run_folder) if run_folder else "no run folder (the first launch never created one)"
    print(f"❌ GIVING UP: {_duration(limit_hours)} without progress on {where}.\n"
          f"   Last job: {last_label or 'none'}, status {last_status or 'unknown'}. "
          f"Check the provider, then run this script again.", file=sys.stderr)
    return EXIT_GAVE_UP
