"""
test_27_submit_with_retries.py
The loop of the qbcs_real_submit_<qpu>_with_retries.py scripts (source/core/runner_support/submit_with_retries.py):
which job is the latest, when a launch is made, how the waits and the 48 h budget behave (and the EQE1 setting: no waits for jobs, the first retry at once, a 20 min budget), the exit codes, and which
run folder a --force-new-run launch works in (also across midnight). No provider, no sleeping: the status and the
launch are fakes, the clock is a counter that sleep() advances. Works in a temporary folder, not in datafiles/.
"""

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

from source.core.records.run_names import today_utc
from source.core.runner_support import submit_with_retries as SWR

UNDATED = "full_real-data_iqm-eqe1_lnn-True_service-mqss-lrz_spam-True_layout-optimized"   # the settings of a run, without its date
TODAY   = today_utc()   # the date the loop gives a new run's folder (a fixed date made two tests fail once UTC passed it)
FOLDER  = f"{UNDATED}_{TODAY}_#1"


class Clock:
    """A fake clock: time moves only through sleep()."""
    def __init__(self):
        self.t = 1_000_000.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


def _folder(root, name=FOLDER):
    return Path(root) / name


def _write_chunk(root, nc, job_id, submitted_at, status="submitted", name=FOLDER):
    d = _folder(root, name) / "job_records"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"job_record_chunk_{nc}.json").write_text(json.dumps(
        {"chunk_id": nc, "status": status, "job_id": job_id, "submitted_at": submitted_at}), encoding="utf-8")


def _write_health_check(root, number, job_id, submitted_at, name=FOLDER):
    d = _folder(root, name) / "health_checks"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"health_check_{number}.json").write_text(json.dumps(
        {"schema": 2, "number": number, "job_id": job_id, "submitted_at": submitted_at, "status": "submitted"}), encoding="utf-8")


def _write_run_info(root, sample_size, name=FOLDER):
    _folder(root, name).mkdir(parents=True, exist_ok=True)
    (_folder(root, name) / "run_info.json").write_text(json.dumps({"sample_size": sample_size}), encoding="utf-8")


def _status_sequence(*texts):
    """A status_of that returns the texts in turn, then the last one forever; records the job ids it was asked for."""
    seen = []
    def status_of(job_id, record):
        seen.append(job_id)
        return texts[min(len(seen), len(texts)) - 1]
    status_of.seen = seen
    return status_of


def _launcher(*return_codes, on_call=None):
    """A launch that returns the codes in turn (the last one forever); records its argument lists."""
    calls = []
    def launch(args):
        calls.append(list(args))
        if on_call:
            on_call(len(calls))
        return return_codes[min(len(calls), len(return_codes)) - 1]
    launch.calls = calls
    return launch


def _run(launch, status_of, root, clock, **kw):
    kw.setdefault("wait_after_failed_s", 600)
    return SWR.run_with_retries(UNDATED, status_of, launch, kw.pop("args", ["mini"]),
                                sleep=clock.sleep, clock=clock.now, root=str(root), **kw)


def test_may_proceed_only_on_final_statuses():
    for final in ("COMPLETED", "DONE", "JobStatus.DONE", "FAILED", "CANCELLED", "ERROR"):
        assert SWR.may_proceed(final), final
    for waiting in ("PENDING", "WAITING", "QUEUED", "RUNNING", "VALIDATING", "INITIALIZING", "SOMETHING_NEW"):
        assert not SWR.may_proceed(waiting), waiting
    # a status query that failed mentions "FAILED" in its text: it must still wait
    assert not SWR.may_proceed("UNAVAILABLE — RuntimeError: connection failed")


def test_latest_job_is_the_newer_of_chunk_and_health_check(tmp_path):
    _write_chunk(tmp_path, 0, "100", "2026-10-15T07:00:00Z")
    _write_chunk(tmp_path, 1, "101", "2026-10-15T07:10:00Z")
    _write_health_check(tmp_path, 1, "200", "2026-10-15T07:05:00Z")
    label, rec = SWR.latest_submitted_job(_folder(tmp_path))
    assert (label, rec["job_id"]) == ("chunk 1", "101")
    _write_health_check(tmp_path, 2, "300", "2026-10-15T07:20:00Z")
    label, rec = SWR.latest_submitted_job(_folder(tmp_path))
    assert (label, rec["job_id"]) == ("health check 2", "300")


def test_records_without_a_job_id_are_ignored(tmp_path):
    _write_chunk(tmp_path, 0, "100", "2026-10-15T07:00:00Z")
    d = _folder(tmp_path) / "job_records"
    (d / "job_record_chunk_1.json").write_text(json.dumps({"chunk_id": 1, "status": "submitted", "job_id": None,
                                                           "submitted_at": "2026-10-15T08:00:00Z"}), encoding="utf-8")
    assert SWR.latest_submitted_job(_folder(tmp_path))[1]["job_id"] == "100"


def test_no_submission_launches_at_once(tmp_path):
    clock, launch = Clock(), _launcher(0)
    status_of = _status_sequence("PENDING")
    assert _run(launch, status_of, tmp_path, clock) == 0
    assert status_of.seen == []                       # nothing to query
    assert launch.calls == [["mini"]]                 # no --force-new-run without new_run
    assert clock.sleeps == []


def test_completed_job_launches_without_waiting(tmp_path):
    _write_chunk(tmp_path, 0, "100", "2026-10-15T07:00:00Z")
    clock, launch = Clock(), _launcher(0)
    assert _run(launch, _status_sequence("COMPLETED"), tmp_path, clock) == 0
    assert len(launch.calls) == 1 and clock.sleeps == []


def test_pending_job_waits_in_10_minute_steps_then_launches(tmp_path):
    _write_chunk(tmp_path, 0, "100", "2026-10-15T07:00:00Z")
    clock, launch = Clock(), _launcher(0)
    assert _run(launch, _status_sequence("PENDING", "RUNNING", "COMPLETED"), tmp_path, clock) == 0
    assert clock.sleeps == [SWR.PAUSE_S, SWR.PAUSE_S]
    assert len(launch.calls) == 1


def test_failed_launch_waits_the_given_time_before_the_next_pass(tmp_path):
    _write_chunk(tmp_path, 0, "100", "2026-10-15T07:00:00Z", status="failed")
    clock = Clock()
    launch = _launcher(1, 0)
    # the chunk record stays "failed": the job status is final, so the next pass launches again
    assert _run(launch, _status_sequence("FAILED"), tmp_path, clock, wait_after_failed_s=600) == 0
    assert len(launch.calls) == 2
    assert clock.sleeps == [600]


def test_failed_launch_without_wait_retries_at_once(tmp_path):
    _write_chunk(tmp_path, 0, "100", "2026-10-15T07:00:00Z", status="failed")
    clock, launch = Clock(), _launcher(1, 0)
    assert _run(launch, _status_sequence("FAILED"), tmp_path, clock, wait_after_failed_s=0) == 0
    assert len(launch.calls) == 2 and clock.sleeps == []


def test_arguments_exit_code_is_returned_without_retry(tmp_path):
    clock, launch = Clock(), _launcher(2)
    assert _run(launch, _status_sequence("COMPLETED"), tmp_path, clock, args=[]) == 2
    assert len(launch.calls) == 1 and clock.sleeps == []


def test_degraded_qpu_aborts_without_retry(tmp_path):
    # exit 10: a health check found the QPU degraded. The run is aborted at once: one launch, no wait, no retry
    clock, launch = Clock(), _launcher(10)
    assert _run(launch, _status_sequence("COMPLETED"), tmp_path, clock) == SWR.EXIT_DEGRADED == 10
    assert len(launch.calls) == 1 and clock.sleeps == []


def test_gives_up_with_exit_14_after_the_budget(tmp_path):
    _write_chunk(tmp_path, 0, "100", "2026-10-15T07:00:00Z")
    clock, launch = Clock(), _launcher(0)
    assert _run(launch, _status_sequence("PENDING"), tmp_path, clock, limit_hours=1) == SWR.EXIT_GAVE_UP == 14
    assert launch.calls == []
    assert sum(clock.sleeps) >= 3600


def test_gives_up_after_repeated_failed_launches_without_progress(tmp_path):
    _write_chunk(tmp_path, 0, "100", "2026-10-15T07:00:00Z", status="failed")
    clock, launch = Clock(), _launcher(1)
    assert _run(launch, _status_sequence("FAILED"), tmp_path, clock, limit_hours=1) == 14
    assert len(launch.calls) >= 2


def test_a_folder_that_never_appears_gives_up(tmp_path):
    # the launch fails before creating anything (e.g. the provider is unreachable at connect): no progress, so the
    # 48 h budget must still end the loop with exit code 14 instead of retrying forever
    clock, launch = Clock(), _launcher(1)
    assert _run(launch, _status_sequence("COMPLETED"), tmp_path, clock, limit_hours=1) == SWR.EXIT_GAVE_UP
    assert len(launch.calls) >= 2


def test_progress_restarts_the_budget(tmp_path):
    # limit 1 h, wait 2000 s after each failed launch. Launch 1 fails (no progress) at t=0; launch 2 submits chunk 1
    # (progress) and fails at t=2000; launch 3 fails at t=4000, which is 1 h 7 min after launch 1 but only 33 min
    # after the progress: it must not give up. Launch 4 succeeds.
    _write_chunk(tmp_path, 0, "100", "2026-10-15T07:00:00Z", status="failed")
    _write_run_info(tmp_path, 2)
    clock = Clock()
    def progress(n):
        if n == 2:
            _write_chunk(tmp_path, 1, "101", "2026-10-15T07:30:00Z")
    launch = _launcher(1, 1, 1, 0, on_call=progress)
    assert _run(launch, _status_sequence("FAILED"), tmp_path, clock, wait_after_failed_s=2000, limit_hours=1) == 0
    assert len(launch.calls) == 4


def test_force_new_run_creates_a_new_folder_across_midnight(tmp_path):
    # Yesterday's run (2026-10-14) is unfinished, so a forced run is wanted. The first launch starts just after UTC
    # midnight: the submit script names its folder with the new date. The loop must not keep forcing new folders
    # (it used to look for a folder with the name it had computed at start, which never matched).
    yesterday = f"{UNDATED}_2026-10-14_#1"
    _write_run_info(tmp_path, 2, name=yesterday)
    _write_chunk(tmp_path, 0, "900", "2026-10-14T23:00:00Z", status="failed", name=yesterday)
    clock = Clock()
    def create_today(n):
        if n == 1:       # the child creates today's folder; its chunk 0 is submitted and then the child fails
            _write_run_info(tmp_path, 2)
            _write_chunk(tmp_path, 0, "100", "2026-10-15T00:00:05Z")
    launch = _launcher(1, 1, 0, on_call=create_today)
    # the status of the new folder's job is COMPLETED; the old folder's job (900) must never be asked about
    status_of = _status_sequence("COMPLETED")
    assert _run(launch, status_of, tmp_path, clock, new_run=True) == 0
    assert launch.calls[0] == ["mini", "--force-new-run"]
    assert launch.calls[1] == ["mini"] and launch.calls[2] == ["mini"]   # the new folder exists: no more forcing
    assert set(status_of.seen) == {"100"}                               # the new run's job, not the old run's (900)


def test_force_new_run_is_forced_until_its_folder_appears(tmp_path):
    def create_folder(n):
        if n == 2:
            _write_chunk(tmp_path, 0, "100", "2026-10-15T07:00:00Z")
    clock = Clock()
    launch = _launcher(1, 1, 0, on_call=create_folder)
    assert _run(launch, _status_sequence("COMPLETED"), tmp_path, clock, new_run=True, wait_after_failed_s=0) == 0
    assert launch.calls[0] == ["mini", "--force-new-run"]   # no folder yet
    assert launch.calls[1] == ["mini", "--force-new-run"]   # still none: forced again
    assert launch.calls[2] == ["mini"]                       # the folder is there now: resume it


def _all_chunks_submitted(tmp_path, n=2):
    _write_run_info(tmp_path, n)
    for nc in range(n):
        _write_chunk(tmp_path, nc, str(100 + nc), f"2026-10-15T07:0{nc}:00Z")


def test_failed_last_health_check_is_retried_three_times_without_a_new_run(tmp_path):
    # all chunks are submitted and only the health check after them could not be submitted (exit 12): the plain script
    # would start a new run, so the loop launches only --retry-health-check, 3 times, and then stops with that code
    clock, launch = Clock(), _launcher(12, on_call=lambda n: _all_chunks_submitted(tmp_path) if n == 1 else None)   # launch 1 submits the chunks
    assert _run(launch, _status_sequence("COMPLETED"), tmp_path, clock) == 12
    assert launch.calls == [["mini"]] + [["mini", "--retry-health-check"]] * 3
    assert clock.sleeps == [600, 600]            # between the retries, none after the last


def test_last_health_check_retry_that_succeeds_ends_the_loop(tmp_path):
    clock, launch = Clock(), _launcher(12, 12, 0, on_call=lambda n: _all_chunks_submitted(tmp_path) if n == 1 else None)
    assert _run(launch, _status_sequence("COMPLETED"), tmp_path, clock) == 0
    assert len(launch.calls) == 3 and launch.calls[-1] == ["mini", "--retry-health-check"]


def test_failed_pre_health_check_keeps_the_normal_wait_cycle(tmp_path):
    # exit 12 while chunks are still missing (the check BEFORE the submission): waits and relaunches the plain script,
    # more than 3 times, until the budget ends
    _write_run_info(tmp_path, 2)
    _write_chunk(tmp_path, 0, "100", "2026-10-15T07:00:00Z")
    clock, launch = Clock(), _launcher(12)
    assert _run(launch, _status_sequence("COMPLETED"), tmp_path, clock, limit_hours=2) == SWR.EXIT_GAVE_UP
    assert len(launch.calls) > SWR.HC_RETRIES
    assert all("--retry-health-check" not in c for c in launch.calls)


def test_health_check_retry_waits_for_the_latest_job(tmp_path):
    clock, launch = Clock(), _launcher(12, 0, on_call=lambda n: _all_chunks_submitted(tmp_path) if n == 1 else None)
    status_of = _status_sequence("PENDING", "COMPLETED")
    assert _run(launch, status_of, tmp_path, clock) == 0
    assert 600 in clock.sleeps and len(launch.calls) == 2


def test_chunks_submitted_by_a_launch_that_created_the_run_count_as_progress(tmp_path):
    # --force-new-run: nothing exists before the first launch, which submits chunk 0 and fails; the budget restarts
    def create(n):
        if n == 1:
            _write_run_info(tmp_path, 2)
            _write_chunk(tmp_path, 0, "100", "2026-10-15T07:00:00Z")
        if n == 2:
            _write_chunk(tmp_path, 1, "101", "2026-10-15T07:30:00Z")
    clock, launch = Clock(), _launcher(1, 1, 0, on_call=create)
    assert _run(launch, _status_sequence("COMPLETED"), tmp_path, clock, new_run=True, limit_hours=1,
                wait_after_failed_s=2000) == 0
    assert len(launch.calls) == 3


def test_eqe1_style_retries_do_not_wait_for_jobs_and_retry_first_at_once(tmp_path):
    # wait_for_jobs=False, first_retry_at_once=True, 1 min between retries, 20 min budget: no status query,
    # the first retry without a sleep, then 60 s sleeps until 20 min have passed, then exit code 14
    _write_chunk(tmp_path, 0, "100", "2026-10-15T07:00:00Z")
    clock, launch, status_of = Clock(), _launcher(1), _status_sequence("PENDING")
    rc = _run(launch, status_of, tmp_path, clock, wait_after_failed_s=60, limit_hours=20 / 60,
              wait_for_jobs=False, first_retry_at_once=True)
    assert rc == SWR.EXIT_GAVE_UP
    assert status_of.seen == []
    assert clock.sleeps == [60] * 20
    assert len(launch.calls) == 22


def test_first_retry_at_once_then_waits(tmp_path):
    clock, launch = Clock(), _launcher(1, 1, 0)
    assert _run(launch, _status_sequence("COMPLETED"), tmp_path, clock, wait_after_failed_s=60,
                first_retry_at_once=True) == 0
    assert len(launch.calls) == 3 and clock.sleeps == [60]


def test_pre_check_skip_age_of_the_latest_job(tmp_path):
    import time
    from source.core.runner_support.run_submission import _seconds_since_last_submission
    assert _seconds_since_last_submission(_folder(tmp_path)) is None          # no job yet
    now = time.time()
    stamp = lambda age: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - age))
    _write_chunk(tmp_path, 0, "100", stamp(3600))
    _write_health_check(tmp_path, 1, "200", stamp(300))
    age = _seconds_since_last_submission(_folder(tmp_path))
    assert 290 <= age <= 330                                                  # the health check is the latest


def test_describe_exit_codes():
    assert SWR.describe_exit(0) == "all chunks submitted"
    assert SWR.describe_exit(12).startswith("a health check could not be submitted")
    assert SWR.describe_exit(99) == "error (see the output above)"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
