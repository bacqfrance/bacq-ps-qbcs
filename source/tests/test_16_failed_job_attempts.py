"""
test_16_failed_job_attempts.py
==============================
Unit tests (no simulation, no network, a temporary run folder) for keeping the records
of failed job attempts (source/core/records/job_records.py), so their QPU, queue and collection time
count in the analysis:

  - archive_failed_record() copies a 'failed' chunk record to
    job_records/failed_attempts/job_record_chunk_<nc>_failed_<k>.json before a
    resubmission overwrites it: k counts up, the same failure is not archived twice,
    records that are not 'failed' are left alone.
  - chunk_record_files() / _record_files() read only the current records — an archived
    job_record_chunk_3_failed_1.json is never taken for chunk 1 (or 3).
  - chunks_without_submission() still lists failed and missing chunks.
  - failed_record_files() returns the archived attempts plus a current record still
    'failed', once each.
  - failed_attempts_breakdown() sums the failed jobs' QPU / queue / collection time, a
    batched job shared by several chunks counted once.
  - Per-job services (IBM, IQM): a failed job that reached the provider counts in full in the workload
    (n_failed_jobs, n_failed_circuits, n_failed_shots, failed_chunks_per_job; a shared job once, string fields
    cast); a submission failure and a still 'submitted' job are only reported (n_may_exist_on_provider,
    n_uncollected).
  - collect_chunk_jobs() (the health check plot) lists the failed jobs, marked failed.
  - Chunk submission (IBM / IQM, against fake samplers): _submit_error_kind() tells
    "never reached the provider" (retried) from "answer lost" (the job may exist: not
    retried) and other errors; a retried submission records its submit_attempts; one that
    gives up writes a 'failed' record per chunk ("submission failed", failure_stage
    "submission", may_exist_on_provider, one submission_id per job, counted once), keeps
    chunks already on the QPU untouched, and re-raises; a rerun archives those records.
  - Written records show status near the top (ordered_record).
  - The run window (run_timing._job_intervals_for_window) includes failed attempts and health checks, and the
    active duration (_active_seconds) subtracts only the gaps between the jobs longer than the dead-time
    threshold (30 min; 2 h for AQT), a gap bridged by a failed attempt not at all.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import json

import pytest
from qiskit import QuantumCircuit
from qiskit.qpy import dump

import source.core.qbcs as Q
from source.core.qpu import remote_jobs as RJ
from source.core.records.job_records import (
    archive_failed_record, chunk_record_files, chunks_without_submission, failed_record_files,
    FAILED_DIRNAME, ordered_record,
)
from source.core.records.health_check_records import collect_chunk_jobs
from source.core.analysis import run_timing as RT
from source.core.analysis.run_timing import _record_files, failed_attempts_breakdown, _job_intervals_for_window, _active_seconds
from source.core.records.job_timing import DEAD_TIME_MIN_GAP_S, DEAD_TIME_MIN_GAP_S_SYNC


def _rec(nc, status, job_id, submitted_at, service="IBM_Cloud", **timing):
    return {"chunk_id": nc, "service_name": service, "job_id": job_id, "status": status,
            "submitted_at": submitted_at, "scheduled_at": timing.pop("start", "N/A"),
            "finished_at": timing.pop("end", "N/A"), "n_spam": 0, "n_circuits": 4, "nshots": 100,
            "timing": timing, **({"failure": "job ended ERROR"} if status == "failed" else {})}


def _write(jr, nc, rec):
    jr.mkdir(parents=True, exist_ok=True)
    (jr / f"job_record_chunk_{nc}.json").write_text(json.dumps(rec), encoding="utf-8")


def test_archive_counts_up_and_is_idempotent(tmp_path):
    jr = tmp_path / "job_records"
    _write(jr, 3, _rec(3, "failed", "A", "2026-09-01T10:00:00Z", qpu_s=5.0))
    p1 = archive_failed_record(jr, 3)
    assert p1.name == "job_record_chunk_3_failed_1.json" and p1.parent.name == FAILED_DIRNAME
    assert archive_failed_record(jr, 3) == p1                      # same failure: not archived again
    archived = json.loads(p1.read_text(encoding="utf-8"))
    assert archived["job_id"] == "A" and archived["superseded"] is True and "archived_at" in archived

    _write(jr, 3, _rec(3, "failed", "B", "2026-09-01T11:00:00Z", qpu_s=6.0))   # second failure
    assert archive_failed_record(jr, 3).name == "job_record_chunk_3_failed_2.json"

    _write(jr, 3, _rec(3, "submitted", "C", "2026-09-01T12:00:00Z"))          # not failed: untouched
    assert archive_failed_record(jr, 3) is None
    assert archive_failed_record(jr, 7) is None                              # no record at all
    assert len(list((jr / FAILED_DIRNAME).iterdir())) == 2


def test_readers_ignore_archived_files(tmp_path):
    jr = tmp_path / "job_records"
    _write(jr, 1, _rec(1, "complete", "X", "2026-09-01T09:00:00Z"))
    _write(jr, 3, _rec(3, "failed", "A", "2026-09-01T10:00:00Z"))
    archive_failed_record(jr, 3)
    # A stray archive-like file in job_records/ itself must not be read as a chunk either
    (jr / "job_record_chunk_3_failed_1.json").write_text(json.dumps(_rec(3, "failed", "Z", "x")), encoding="utf-8")
    assert list(chunk_record_files(jr)) == [1, 3]
    recs = _record_files(tmp_path)
    assert recs[1]["job_id"] == "X" and recs[3]["job_id"] == "A"
    assert chunks_without_submission(tmp_path, 5) == [0, 2, 3, 4]


def test_failed_records_and_breakdown(tmp_path):
    jr = tmp_path / "job_records"
    # chunks 0 and 1 shared a failed batched job "J1", then were resubmitted and completed
    for nc in (0, 1):
        _write(jr, nc, _rec(nc, "failed", "J1", "2026-09-01T10:00:00Z", qpu_s=10.0, queue_s=100.0,
                            collect_s=2.0, submit_s=1.0, transpile_s=3.0,
                            start="2026-09-01T10:01:40Z", end="2026-09-01T10:01:50Z"))
        archive_failed_record(jr, nc)
        _write(jr, nc, _rec(nc, "complete", "J2", "2026-09-01T11:00:00Z", qpu_s=10.0, queue_s=50.0,
                            start="2026-09-01T11:00:50Z", end="2026-09-01T11:01:00Z"))
    # chunk 2 failed and was not resubmitted yet
    _write(jr, 2, _rec(2, "failed", "J3", "2026-09-01T12:00:00Z", qpu_s=None, queue_s=30.0, collect_s=1.0))

    fr = failed_record_files(tmp_path)
    assert {nc: [r["job_id"] for r in v] for nc, v in fr.items()} == {0: ["J1"], 1: ["J1"], 2: ["J3"]}
    assert failed_record_files(tmp_path, [2]) == {2: fr[2]}                 # scoped

    fa = failed_attempts_breakdown(tmp_path)
    assert fa["n_attempts"] == 3 and fa["n_jobs"] == 2 and fa["chunks"] == [0, 1, 2]
    assert fa["qpu_s"] == 10.0                     # J1 once; J3 reports none
    assert fa["n_jobs_without_qpu_s"] == 1
    assert fa["queue_s"] == 130.0 and sorted(fa["queues"]) == [30.0, 100.0]
    assert fa["collect_s"] == 3.0 and fa["submit_s"] == 1.0
    assert fa["transpile_s"] == 6.0                # per chunk
    assert fa["sequential"] is False

    jobs = collect_chunk_jobs(tmp_path)
    failed = [j for j in jobs if j["failed"]]
    assert [j["chunks"] for j in failed] == [[0, 1], [2]]
    assert [j["chunks"] for j in jobs if not j["failed"]] == [[0, 1]]
    assert failed[0]["end"] is not None and failed[1]["end"] is None


def test_aqt_failed_attempts_are_separate_jobs(tmp_path):
    jr = tmp_path / "job_records"
    for i, sub in enumerate(("2026-09-01T10:00:00Z", "2026-09-01T11:00:00Z"), start=1):
        _write(jr, 5, _rec(5, "failed", None, sub, service="PCSS", qpu_s=4.0, queue_s=20.0))
        archive_failed_record(jr, 5)
    fa = failed_attempts_breakdown(tmp_path)
    assert fa["n_attempts"] == 2 and fa["n_jobs"] == 2 and fa["qpu_s"] == 8.0
    assert fa["sequential"] is True
    assert sum(j["failed"] for j in collect_chunk_jobs(tmp_path)) == 2


def test_aqt_failed_attempts_circuits_and_shots_enter_the_workload(tmp_path):
    """Circuits executed in a failed attempt are summed (n_executed_circuits / n_executed_shots, or an old
    record's batch_sizes x batch_shots) and reported by qpu_workload apart from the complete jobs."""
    jr = tmp_path / "job_records"
    new = _rec(5, "failed", None, "2026-09-01T10:00:00Z", service="PCSS", qpu_s=4.0, queue_s=20.0)
    new.update(n_executed_circuits=7, n_executed_shots=1400)
    _write(jr, 5, new)
    archive_failed_record(jr, 5)
    _write(jr, 5, _rec(5, "failed", None, "2026-09-01T11:00:00Z", service="PCSS", qpu_s=4.0, queue_s=20.0,
                       batch_sizes=[2, 3], batch_shots=[100, 100]))     # old record: from the timing block
    archive_failed_record(jr, 5)
    _write(jr, 0, _rec(0, "complete", None, "2026-09-01T12:00:00Z", service="PCSS", qpu_s=4.0, queue_s=1.0))
    fa = failed_attempts_breakdown(tmp_path)
    assert fa["n_circuits"] == 7 + 5 and fa["n_shots"] == 1400 + 500
    w = RT.qpu_workload(tmp_path)
    assert w["n_failed_circuits"] == 12 and w["n_failed_shots"] == 1900
    assert w["n_jobs"] == 1 and w["n_circuits"] == 4 and w["n_shots"] == 400    # the failed ones are not jobs


def test_per_job_services_count_failed_jobs_in_full(tmp_path):
    """IBM-style: 3 chunk records share the failed job J1 (string fields, no n_executed_circuits): one job, every
    chunk's n_spam + n_circuits circuits x nshots. A submission failure (no job_id, may exist) and a job still
    'submitted' are not counted but reported; a failed async health check with a job_id is one job, one circuit."""
    jr = tmp_path / "job_records"
    for nc in (0, 1, 2):
        r = _rec(nc, "failed", "J1", "2026-09-01T10:00:00Z", qpu_s=10.0, queue_s=5.0)
        r.update(n_spam="2", n_circuits="4", nshots="100", failure_stage="execution")
        _write(jr, nc, r)
        archive_failed_record(jr, nc)
    sub = _rec(3, "failed", None, "2026-09-01T11:00:00Z", qpu_s=0.0)
    sub.update(failure_stage="submission", may_exist_on_provider=True, submission_id="S1", failure="submission failed: x")
    _write(jr, 3, sub)
    _write(jr, 4, _rec(4, "submitted", "J9", "2026-09-01T12:00:00Z"))
    _write(jr, 0, _rec(0, "complete", "J2", "2026-09-01T13:00:00Z", qpu_s=7.0))
    hc = tmp_path / "health_checks"
    hc.mkdir()
    (hc / "health_check_1.json").write_text(json.dumps(
        {"number": 1, "status": "failed", "job_id": "H1", "nshots": 50, "failure": "job ended ERROR",
         "failure_stage": "execution", "service_name": "IBM_Cloud", "timing": {"qpu_s": 3.0}}), encoding="utf-8")
    (hc / "health_check_2.json").write_text(json.dumps(
        {"number": 2, "status": "submitted", "job_id": "H2", "nshots": 50, "service_name": "IBM_Cloud"}), encoding="utf-8")
    fa = failed_attempts_breakdown(tmp_path)
    assert fa["n_provider_jobs"] == 2 and fa["chunks_per_job"] == [3]       # J1 (3 chunks) + the health check
    assert fa["n_circuits"] == 3 * 6 + 1 and fa["n_shots"] == 3 * 6 * 100 + 50
    assert fa["n_may_exist_on_provider"] == 1 and fa["qpu_s"] == 13.0
    w = RT.qpu_workload(tmp_path)
    assert w["n_failed_jobs"] == 2 and w["n_failed_circuits"] == 19 and w["n_failed_shots"] == 1850
    assert w["failed_chunks_per_job"] == [3] and w["n_may_exist_on_provider"] == 1
    assert w["n_uncollected"] == 2                                          # chunk 4's job and health check 2
    assert w["n_jobs"] == 1 and w["n_circuits"] == 4                        # the successful run only
    assert RT.qpu_queue_breakdown(tmp_path)["qpu_total_s"] == 7.0


# ── Chunk submission: retry and "submission failed" records ──────────────────
N_CIRC = 4


def _circuits(n):
    out = []
    for _ in range(n):
        qc = QuantumCircuit(3, 3)
        qc.measure(range(3), range(3))
        out.append(qc)
    return out


def _qbcs(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "SUBMIT_RETRY_BACKOFF_S", (0, 0, 0))
    obj = Q.bacq_ps_qbcs(sample_size=4, chunk_size=6, list_circuit_times=[0.2], nshots=100,
                         do_SPAM=True, error_mitigation_strategies=["SPAM", "NEC", "RC+NEC", "cRC+NEC"])
    obj.CONFIG["folder_path"] = str(tmp_path)
    obj.transpile_chunk_for_backend_to_isa = lambda nc, backend: _circuits(N_CIRC)
    tdir = tmp_path / "transpiled_circuits"
    tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / "spam_transpiled.qpy", "wb") as f:
        dump(_circuits(2), f)
    return obj


class _Job:
    def __init__(self, job_id):
        self._id = job_id

    def job_id(self):
        return self._id


class _FakeSampler:
    """run() raises the given errors in turn, then returns a job."""

    def __init__(self, *errors):
        self.errors, self.calls = list(errors), 0

    def run(self, circuits, shots=None):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return _Job(f"job-{self.calls}")


# Errors shaped like what requests/urllib3 raise (local_sim.venv has no requests): classified by message
NOT_CONNECTED = ConnectionError(
    "HTTPSConnectionPool(host='x', port=443): Max retries exceeded (Caused by NewConnectionError("
    "'Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))")
ANSWER_LOST = ConnectionError("('Connection aborted.', ConnectionResetError(10054, 'reset'))")


def test_submit_error_kind():
    assert RJ._submit_error_kind(NOT_CONNECTED) == "not_connected"
    assert RJ._submit_error_kind(ConnectionRefusedError("refused")) == "not_connected"
    assert RJ._submit_error_kind(TimeoutError("Connection to x timed out. (connect timeout=10)")) == "not_connected"
    assert RJ._submit_error_kind(ANSWER_LOST) == "answer_lost"
    assert RJ._submit_error_kind(TimeoutError("Read timed out. (read timeout=60)")) == "answer_lost"
    assert RJ._submit_error_kind(Q.VLQ.VLQTimeout("no answer after 3600 s")) == "answer_lost"
    assert RJ._submit_error_kind(ValueError("bad payload")) is None
    try:                                              # a chain holding both: the job may exist
        try:
            raise NOT_CONNECTED
        except Exception:
            raise RuntimeError("wrapped") from TimeoutError("Read timed out. (read timeout=60)")
    except RuntimeError as e:
        assert RJ._submit_error_kind(e) == "answer_lost"


def test_unreachable_provider_is_retried(tmp_path, monkeypatch):
    obj = _qbcs(tmp_path, monkeypatch)
    sampler = _FakeSampler(NOT_CONNECTED, NOT_CONNECTED)
    obj.submit_chunk_ibm(0, sampler, "fake_backend")
    assert sampler.calls == 3
    rec = json.loads((tmp_path / "job_records" / "job_record_chunk_0.json").read_text(encoding="utf-8"))
    assert rec["status"] == "submitted" and rec["job_id"] == "job-3"
    assert [a["kind"] for a in rec["submit_attempts"]] == ["not_connected", "not_connected", None]
    assert list(rec)[:2] == ["chunk_id", "status"]               # status near the top


def test_retries_run_out(tmp_path, monkeypatch):
    obj = _qbcs(tmp_path, monkeypatch)
    sampler = _FakeSampler(*[NOT_CONNECTED] * 4)
    with pytest.raises(RJ.ChunkSubmitError):
        obj.submit_chunk_ibm(0, sampler, "fake_backend")
    assert sampler.calls == 4                                     # 1 + len(SUBMIT_RETRY_BACKOFF_S)
    rec = json.loads((tmp_path / "job_records" / "job_record_chunk_0.json").read_text(encoding="utf-8"))
    assert rec["status"] == "failed" and rec["failure"].startswith("submission failed")
    assert rec["failure_stage"] == "submission" and rec["may_exist_on_provider"] is False
    assert len(rec["submit_attempts"]) == 4 and rec["timing"]["qpu_s"] == 0.0


@pytest.mark.parametrize("error, may_exist", [(ANSWER_LOST, True), (ValueError("bad payload"), False)])
def test_not_retried_failure_writes_records(tmp_path, monkeypatch, error, may_exist):
    obj = _qbcs(tmp_path, monkeypatch)
    jr  = tmp_path / "job_records"
    # chunk 1 already submitted in an earlier job: it must keep its record
    _write(jr, 1, _rec(1, "submitted", "old-job", "2026-09-01T09:00:00Z"))
    sampler = _FakeSampler(error)
    with pytest.raises(RJ.ChunkSubmitError) as info:
        obj.submit_batch_ibm([0, 1, 2], sampler, "fake_backend", None)
    assert info.value.cause is error and sampler.calls == 1       # not retried
    recs = {nc: json.loads((jr / f"job_record_chunk_{nc}.json").read_text(encoding="utf-8")) for nc in (0, 1, 2)}
    assert recs[1]["status"] == "submitted" and recs[1]["job_id"] == "old-job"
    for nc in (0, 2):
        r = recs[nc]
        assert r["status"] == "failed" and r["failure_stage"] == "submission"
        assert r["may_exist_on_provider"] is may_exist and r["job_id"] is None
        assert list(r)[:5] == ["chunk_id", "status", "failure", "failure_stage", "may_exist_on_provider"]
    assert recs[0]["submission_id"] == recs[2]["submission_id"]

    fa = failed_attempts_breakdown(tmp_path)
    assert fa["n_jobs"] == 1 and fa["n_submission_failures"] == 1 and fa["chunks"] == [0, 2]
    assert fa["n_may_exist_on_provider"] == int(may_exist) and fa["qpu_s"] == 0.0
    jobs = [j for j in collect_chunk_jobs(tmp_path) if j["failed"]]
    assert len(jobs) == 1 and jobs[0]["chunks"] == [0, 2] and jobs[0]["submission_failed"]

    # A rerun resubmits chunks 0 and 2 and archives their failed records first
    obj.submit_chunk_ibm(0, _FakeSampler(), "fake_backend")
    assert json.loads((jr / "job_record_chunk_0.json").read_text(encoding="utf-8"))["status"] == "submitted"
    assert (jr / FAILED_DIRNAME / "job_record_chunk_0_failed_1.json").exists()


def test_iqm_submission_failure_records_every_chunk(tmp_path, monkeypatch):
    obj = _qbcs(tmp_path, monkeypatch)

    class _Backend:
        name = "fake_iqm"
        def run(self, circuits, shots=None):
            raise ValueError("payload rejected")

    with pytest.raises(RJ.ChunkSubmitError):
        obj.submit_batch_iqm([0, 1], _Backend(), "IQM_Resonance", max_circuits_per_job=1000)
    for nc in (0, 1):
        r = json.loads((tmp_path / "job_records" / f"job_record_chunk_{nc}.json").read_text(encoding="utf-8"))
        assert r["status"] == "failed" and r["service_name"] == "IQM_Resonance" and r["n_circuits"] == N_CIRC


def test_ordered_record_keeps_other_keys():
    rec = {"timing": {}, "backend_name": "b", "status": "complete", "chunk_id": 3, "zzz": 1}
    assert list(ordered_record(rec)) == ["chunk_id", "status", "backend_name", "timing", "zzz"]


def _at(t):
    return f"2026-09-01T{t}Z"


def _window_run(tmp_path, chunks, failed=(), health=()):
    """A run folder: chunks/failed as (start, end) times of 2026-09-01, health checks likewise."""
    jr = tmp_path / "job_records"
    for nc, (a, b) in enumerate(chunks):
        _write(jr, nc, _rec(nc, "complete", f"J{nc}", _at(a), end=_at(b)))
    for i, (a, b) in enumerate(failed):
        _write(jr, 90 + i, _rec(90 + i, "failed", f"F{i}", _at(a), end=_at(b)))
        archive_failed_record(jr, 90 + i)
        (jr / f"job_record_chunk_{90 + i}.json").unlink()
    (tmp_path / "health_checks").mkdir(exist_ok=True)
    for i, (a, b) in enumerate(health):
        rec = {"number": i, "submitted_at": _at(a), "finished_at": _at(b), "status": "failed", "chunks": [0]}
        (tmp_path / "health_checks" / f"health_check_{i}.json").write_text(json.dumps(rec), encoding="utf-8")


def _active(tmp_path, min_gap_s=DEAD_TIME_MIN_GAP_S):
    ivs, _ = _job_intervals_for_window(tmp_path, None)
    return (max(b for _, b in ivs) - min(a for a, _ in ivs)).total_seconds(), _active_seconds(ivs, min_gap_s)


def test_window_includes_failed_attempts_and_health_checks(tmp_path):
    _window_run(tmp_path, [("10:00:00", "10:10:00")], failed=[("09:00:00", "09:05:00")], health=[("10:20:00", "10:30:00")])
    ivs, n = _job_intervals_for_window(tmp_path, None)
    assert n == 1 and len(ivs) == 3
    window, _ = _active(tmp_path)
    assert window == 5400.0                          # 09:00 -> 10:30


def test_health_check_without_finish_contributes_its_start(tmp_path):
    _window_run(tmp_path, [("10:00:00", "10:10:00")])
    rec = {"number": 0, "submitted_at": _at("10:15:00"), "status": "submitted", "chunks": [0]}
    (tmp_path / "health_checks" / "health_check_0.json").write_text(json.dumps(rec), encoding="utf-8")
    assert _active(tmp_path)[0] == 900.0


def test_long_gap_is_dead_time_short_gap_is_kept(tmp_path):
    _window_run(tmp_path, [("10:00:00", "10:10:00"), ("13:10:00", "13:20:00")])      # 3 h gap
    window, active = _active(tmp_path)
    assert window == 12000.0 and active == 1200.0
    other = tmp_path / "b"
    _window_run(other, [("10:00:00", "10:10:00"), ("10:30:00", "10:40:00")])         # 20 min gap
    window, active = _active(other)
    assert window == active == 2400.0


def test_sync_threshold_is_two_hours(tmp_path):
    _window_run(tmp_path, [("10:00:00", "10:10:00"), ("11:10:00", "11:20:00")])      # 1 h gap: kept
    window, active = _active(tmp_path, DEAD_TIME_MIN_GAP_S_SYNC)
    assert window == active == 4800.0
    other = tmp_path / "b"
    _window_run(other, [("10:00:00", "10:10:00"), ("13:10:00", "13:20:00")])         # 3 h gap: subtracted
    assert _active(other, DEAD_TIME_MIN_GAP_S_SYNC)[1] == 1200.0


def test_gap_bridged_by_failed_attempt_is_kept(tmp_path):
    _window_run(tmp_path, [("10:00:00", "10:10:00"), ("13:10:00", "13:20:00")], failed=[("10:05:00", "13:00:00")])
    window, active = _active(tmp_path)
    assert active == window                          # the 10-minute gap 13:00 -> 13:10 is short; the 3 h one is bridged
