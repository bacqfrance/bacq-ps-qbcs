"""
test_29_job_cancel.py
The shared body of the cancel_real_qpu_jobs_<qpu>.py scripts (source/core/qpu/job_cancel.py): only jobs that are still
queued or running are cancelled, once per job even when chunks share it; ended jobs, records without a job and failed
queries are left alone; the records on disk stay unchanged; the exit code is 2 when any cancellation fails.
No provider: the status and the cancellation are fakes. Works in a temporary folder, not in datafiles/.
"""

import json
import os
import sys

import pytest

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

from source.core.qpu import job_cancel, job_status_check

RUN = "run_a"
HC  = "hc_a"


@pytest.fixture
def datafiles(tmp_path, monkeypatch):
    monkeypatch.setattr(job_status_check, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _records(root, folder, jobs, health_check=False):
    """Write job records: jobs is the list of job ids, one chunk each (or the one health-check record)."""
    d = root / "datafiles" / folder / "job_records"
    d.mkdir(parents=True)
    if health_check:
        (d / "health_check.json").write_text(json.dumps({"job_id": jobs[0], "status": "submitted"}), encoding="utf-8")
        return
    for chunk, job_id in enumerate(jobs, 1):
        (d / f"job_record_chunk_{chunk}.json").write_text(
            json.dumps({"chunk_id": chunk, "job_id": job_id, "status": "submitted"}), encoding="utf-8")


class Fake:
    """status: job_id -> status text (or an exception); cancelling a queued job turns it into CANCELLED."""
    def __init__(self, status, fail_cancel=()):
        self.status, self.fail_cancel, self.cancelled = dict(status), set(fail_cancel), []

    def remote_status(self, job_id):
        s = self.status[job_id]
        if isinstance(s, Exception):
            raise s
        return s

    def cancel_job(self, job_id):
        if job_id in self.fail_cancel:
            raise RuntimeError("refused")
        self.cancelled.append(job_id)
        self.status[job_id] = "JobStatus.CANCELLED"


def _run(fake, folders=(RUN,), health_check=False):
    return job_cancel.run_cancels("FAKE", list(folders), HC, health_check, fake.remote_status, fake.cancel_job)


def test_only_unfinished_jobs_are_cancelled(datafiles, capsys):
    _records(datafiles, RUN, ["q", "r", "d", "e", "c"])
    fake = Fake({"q": "JobStatus.QUEUED", "r": "JobStatus.RUNNING", "d": "JobStatus.DONE",
                 "e": "JobStatus.ERROR", "c": "JobStatus.CANCELLED"})
    _run(fake)
    assert sorted(fake.cancelled) == ["q", "r"]
    assert "run the collect script" in capsys.readouterr().out


def test_a_shared_job_is_cancelled_once(datafiles):
    _records(datafiles, RUN, ["j", "j", "j"])
    fake = Fake({"j": "PENDING"})
    _run(fake)
    assert fake.cancelled == ["j"]


def test_records_without_a_job_are_skipped(datafiles):
    _records(datafiles, RUN, [None, "N/A", "j"])
    fake = Fake({"j": "RUNNING"})
    _run(fake)
    assert fake.cancelled == ["j"]


def test_health_check_mode_reads_only_the_health_check_record(datafiles):
    _records(datafiles, HC, ["h"], health_check=True)
    fake = Fake({"h": "QUEUED"})
    _run(fake, folders=(), health_check=True)
    assert fake.cancelled == ["h"]


def test_records_on_disk_are_unchanged(datafiles):
    _records(datafiles, RUN, ["q"])
    path = datafiles / "datafiles" / RUN / "job_records" / "job_record_chunk_1.json"
    before = path.read_text(encoding="utf-8")
    _run(Fake({"q": "QUEUED"}))
    assert path.read_text(encoding="utf-8") == before


def test_a_refused_cancellation_exits_with_2_and_the_others_still_run(datafiles, capsys):
    _records(datafiles, RUN, ["bad", "good"])
    fake = Fake({"bad": "QUEUED", "good": "QUEUED"}, fail_cancel={"bad"})
    with pytest.raises(SystemExit) as e:
        _run(fake)
    assert e.value.code == 2
    assert fake.cancelled == ["good"]
    assert "cancel failed" in capsys.readouterr().err


def test_an_unreadable_status_is_a_failure_and_cancels_nothing(datafiles):
    _records(datafiles, RUN, ["j"])
    fake = Fake({"j": ConnectionError("down")})
    with pytest.raises(SystemExit) as e:
        _run(fake)
    assert e.value.code == 2
    assert fake.cancelled == []


def test_one_failed_folder_does_not_stop_the_next(datafiles):
    _records(datafiles, "run_b", ["j"])                     # run_a has no records: exits with 2
    fake = Fake({"j": "QUEUED"})
    with pytest.raises(SystemExit) as e:
        _run(fake, folders=(RUN, "run_b"))
    assert e.value.code == 2
    assert fake.cancelled == ["j"]
