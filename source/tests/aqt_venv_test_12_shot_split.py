"""
aqt_venv_test_12_shot_split.py
==============================
Run by source/tests/test_12_aqt_shot_split.py in venvs/aqt.venv (not collected by pytest under that name).
Manually, with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 (qiskit-aqt-provider registers a pytest plugin that needs pytest-httpx):
  Linux, with aqt.venv active: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest --noconftest source/tests/aqt_venv_test_12_shot_split.py
  Windows Terminal, with aqt.venv active: $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"; py -m pytest --noconftest source/tests/aqt_venv_test_12_shot_split.py

Unit tests (no simulator, no QPU, no network) for AQT's 200-shots-per-circuit limit:

  - job_timing.aqt_shot_parts: the fewest, most even split of a shot count into parts
    of at most 200 shots;
  - qpu/aqt.py: _aqt_run: refuses more than 200 shots per circuit before calling the backend;
  - job_timing.aqt_timing: QPU time modelled per batch at that batch's shot count;
  - qbcs.submit_chunk_aqt against a stub backend: a 450-shot chunk runs as 3 x 150 shots,
    the parts' counts are summed per circuit, each part is checkpointed, and an
    interrupted chunk resumes at the first missing part;
  - each part run is bracketed by qpu_chunk{nc}_part{p}_start/_finished in the run's
    timestamp log (parts reused from disk log nothing), and those rows are not read as chunks;
  - qpu/aqt.py: _aqt_retry announces each retry of a dropped connection, confirms a reconnect, and says when it gives up;
  - qpu/aqt.py: the loop script's STOP file (taken once) and its AQT_PAUSE_AFTER_HOURS time limit;
  - qpu/aqt.py: a circuit whose task is not executed within AQT_TASK_TIMEOUT_S ends the job with
    AQTTaskTimeout (failure stage "execution") instead of waiting on;
  - health_check.submit_health_check: that timeout in an accompanying health check leaves a failed record
    (queue and transpile time, no QPU time) that the failed-attempts analysis counts;
  - run_timing.qpu_workload: each execution of a split chunk counts as circuits,
    the shots are the total.
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

from source.core.records import job_timing as JT
from source.core.analysis import run_timing as RT
from source.core.records import timestamps as TS
from source.core import qbcs as Q
from source.core.qpu import aqt as AQ


# ── Splitting the shots ───────────────────────────────────────────────────────
@pytest.mark.parametrize("nshots, parts", [
    (1, [1]), (10, [10]), (200, [200]), (201, [101, 100]), (401, [134, 134, 133]),
    (450, [150, 150, 150]), (600, [200, 200, 200]), (1000, [200] * 5),
])
def test_aqt_shot_parts(nshots, parts):
    assert JT.aqt_shot_parts(nshots) == parts
    assert sum(parts) == nshots and max(parts) <= JT.AQT_MAX_SHOTS


def test_aqt_shot_parts_rejects_non_positive():
    with pytest.raises(ValueError):
        JT.aqt_shot_parts(0)


def test_aqt_run_refuses_more_than_200_shots():
    class Backend:
        def run(self, circuits, shots):
            raise AssertionError("must not be called")
    with pytest.raises(ValueError, match="at most 200 shots"):
        AQ._aqt_run(Backend(), [QuantumCircuit(1)], 201)


def test_task_not_executed_in_time_ends_the_job(monkeypatch):
    import threading
    class NeverExecuted:
        name, version = "stub_aqt", 1
        def __init__(self):
            self.stop = threading.Event()
        def run(self, circuits, shots):
            options = _Obj(with_progress_bar=False, query_timeout_seconds=1, shots=shots, memory=False)
            payload = _Obj(payload=_Obj(circuits=list(range(len(circuits)))))
            return _Obj(options=options, circuits=circuits, api_submit_payload=payload, job_id=lambda: "stub_job")
        def submit(self, i):
            return i
        def result(self, task_id, timeout):
            self.stop.wait()                    # PCSS's result query polls without end
    monkeypatch.setattr(AQ, "AQT_TASK_TIMEOUT_S", 0.2)
    backend = NeverExecuted()
    try:
        with pytest.raises(AQ.AQTTaskTimeout, match=r"not executed within .* min of its submission.*PIAST-Q") as exc:
            AQ._aqt_run(backend, [QuantumCircuit(1)], 100).result()
    finally:
        backend.stop.set()
    assert exc.value.aqt_failure_stage == "execution"


def test_health_check_task_timeout_writes_a_failed_record(monkeypatch, tmp_path):
    """An accompanying AQT health check whose task times out is recorded as failed (stage "execution", the
    wait as queue time, the transpile time, no QPU time) and AQTTaskTimeout is re-raised; the analysis counts
    it under failed_attempts (queue and transpile) but not in the QPU time."""
    from source.core import health_check as HC
    from source.core.records.health_check_records import load_health_checks
    monkeypatch.setattr(HC, "PROJECT_ROOT", tmp_path)
    run = tmp_path / "datafiles" / "run"
    (run / "job_records").mkdir(parents=True)
    isa = HC.build_qpu_health_check_circuit()
    monkeypatch.setattr(HC, "_health_check_isa", lambda *a, **k: (isa, "health_check_transpiled.qpy", 0.25))
    monkeypatch.setattr(HC, "export_circuit_drawings", lambda *a, **k: None)
    def timed_out(backend, circuits, shots):
        raise AQ.AQTTaskTimeout("AQT task t (circuit 0) was not executed within 30 min of its submission")
    monkeypatch.setattr(HC, "_aqt_run", timed_out)

    class Backend:
        name = "stub_aqt"
    with pytest.raises(AQ.AQTTaskTimeout):
        HC.submit_health_check("run", Backend(), service_name=HC.PCSS, nshots=10, accompanying=True, chunks=[0])
    (rec,) = load_health_checks(run)
    assert rec["status"] == "failed" and rec["failure_stage"] == "execution" and rec["chunks"] == [0]
    assert rec["P011"] is None and rec["verdict"] is None and rec["counts"] is None
    t = rec["timing"]
    assert t["transpile_s"] == 0.25 and t["qpu_s"] is None and t["queue_s"] >= 0

    fa = RT.failed_attempts_breakdown(run)
    assert fa["n_attempts"] == 1 and fa["qpu_s"] == 0.0 and fa["n_jobs_without_qpu_s"] == 1
    assert fa["transpile_s"] == 0.25 and fa["queue_s"] == pytest.approx(t["queue_s"]) and fa["sequential"]
    assert fa["failures"][0]["failure_stage"] == "execution"
    assert RT.qpu_workload(run)["n_health_checks"] == 0


def test_connection_retries_are_announced_and_confirmed(monkeypatch, capsys):
    """A dropped connection prints one two-line notice per retry, then a reconnected line; with the
    retries used up it prints a giving-up line and raises."""
    import httpx
    monkeypatch.setattr(AQ.time, "sleep", lambda s: None)
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise httpx.ConnectError("All connection attempts failed")
        return "result"
    assert AQ._aqt_retry("result query", flaky) == "result"
    out = capsys.readouterr().out
    assert out.count("⚠️  [AQT] result query: connection error — retry") == 2
    assert "retry 1/7 in 5 s" in out and "retry 2/7 in 15 s" in out
    assert "ConnectError: All connection attempts failed" in out
    assert "✅ [AQT] result query: reconnected after 2 retries." in out

    def dead():
        raise httpx.ConnectError("down")
    with pytest.raises(httpx.ConnectError):
        AQ._aqt_retry("result query", dead)
    cap = capsys.readouterr()
    out, err = cap.out, cap.err
    assert out.count("connection error — retry") == 7 and "❌ [AQT] result query: still no connection after 7 retries" in err


def test_stop_file_is_taken_once(tmp_path):
    assert not AQ.take_stop_file(tmp_path)
    (tmp_path / "STOP").touch()
    assert AQ.take_stop_file(tmp_path)                   # honoured and deleted …
    assert not (tmp_path / "STOP").exists()
    assert not AQ.take_stop_file(tmp_path)               # … so the next job of the run is not stopped
    assert (AQ.STOP_EXIT_CODE, AQ.PAUSE_EXIT_CODE) == (4, 3)   # what the PCSS sbatch files act on


def test_pause_after_hours():
    assert AQ.pause_after_hours({}) == float("inf")
    assert AQ.pause_after_hours({"AQT_PAUSE_AFTER_HOURS": "0.25"}) == 0.25
    assert not AQ.pause_due(100.0, 0.25, now=100.0 + 900)    # exactly 15 min: not yet
    assert AQ.pause_due(100.0, 0.25, now=100.0 + 901)
    assert not AQ.pause_due(100.0, float("inf"), now=1e12)   # unset: no limit


def test_aqt_timing_per_batch_shots():
    f = JT.aqt_qpu_s_per_circuit
    t = JT.aqt_timing([100.0, 100.0], [14, 14], 400, batch_shots=[200, 200])
    assert t["qpu_s"] == pytest.approx(28 * f(200)) and t["shots"] == 400
    assert t["batch_shots"] == [200, 200] and t["qpu_s_per_circuit"] == f(200)
    t = JT.aqt_timing([100.0, 100.0], [14, 14], 334, batch_shots=[134, 200])   # parts of different sizes
    assert t["qpu_s"] == pytest.approx(14 * f(134) + 14 * f(200)) and t["qpu_s_per_circuit"] is None


# ── submit_chunk_aqt with a stub backend ──────────────────────────────────────
N_SPAM, N_MAIN = 8, 6


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _StubAQT:
    """Stands in for the PCSS direct-access backend, as far as qpu/aqt.py: _AQTResilientJob uses it
    (run() -> job with options / circuits / api_submit_payload; submit(); result(); name; version).
    Every circuit returns bitstring format(i) for i = its position in the batch run, with one
    shot moved to '111' so the merge has more than one key to add up. fail_on_call: raise
    on that call (1-based) to simulate an interruption. fail_after=(call, k): that call (1-based)
    finishes k circuits, then the result query of circuit k raises (a failure inside a batch)."""
    name, version = "stub_aqt", 1

    def __init__(self, fail_on_call=None, fail_after=None):
        self.calls, self.fail_on_call, self.fail_after = [], fail_on_call, fail_after
        self._samples = {}
    def run(self, circuits, shots):
        assert shots <= JT.AQT_MAX_SHOTS
        self.calls.append((len(circuits), shots))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("connection lost")
        options = _Obj(with_progress_bar=False, query_timeout_seconds=1, shots=shots, memory=False)
        payload = _Obj(payload=_Obj(circuits=list(range(len(circuits)))))
        return _Obj(options=options, circuits=circuits, api_submit_payload=payload, job_id=lambda: "stub_job")
    def submit(self, i):
        # the i-th circuit of the batch: shots-1 outcomes format(i % 7), one outcome 111 (qubit k = bit k)
        shots = self.calls[-1][1]
        bits  = lambda v: [(v >> k) & 1 for k in range(3)]
        self._samples[i] = [bits(i % 7)] * (shots - 1) + [bits(7)]
        return i
    def result(self, task_id, timeout):
        if self.fail_after == (len(self.calls), task_id):
            raise RuntimeError("connection lost inside the batch")
        return _Obj(payload=_Obj(result=self._samples[task_id]))
    def __str__(self):
        return "stub_aqt"


def _circuits(n):
    out = []
    for i in range(n):
        qc = QuantumCircuit(3)
        qc.rx(0.1 * (i + 1), 0)
        qc.measure_all()
        out.append(qc)
    return out


def _qbcs(tmp_path, nshots):
    obj = Q.bacq_ps_qbcs(sample_size=1, chunk_size=6, list_circuit_times=[0.2], nshots=nshots,
                         do_SPAM=True, error_mitigation_strategies=["SPAM", "NEC", "RC+NEC", "cRC+NEC"])
    obj.CONFIG["folder_path"] = str(tmp_path)
    obj.transpile_chunk_for_backend_to_isa = lambda nc, backend: _circuits(N_MAIN)
    tdir = tmp_path / "transpiled_circuits"
    tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / "spam_transpiled.qpy", "wb") as f:
        dump(_circuits(N_SPAM), f)
    return obj


def test_split_chunk_is_merged_and_recorded(tmp_path):
    obj, backend = _qbcs(tmp_path, 450), _StubAQT()
    obj.submit_chunk_aqt(0, backend)

    assert backend.calls == [(N_SPAM + N_MAIN, 150)] * 3
    for p in range(3):
        part = json.loads((tmp_path / "raw_results" / "shot_parts" / f"chunk_0_part_{p}.json").read_text(encoding="utf-8"))
        assert part["shots"] == 150 and len(part["counts"]) == N_SPAM + N_MAIN

    counts = json.loads((tmp_path / "raw_results" / "results_counts.json").read_text(encoding="utf-8"))["chunk_0"]
    for job in counts.values():
        (c,) = job.values()
        assert sum(c.values()) == 450 and c["111"] == 3               # 1 shot per part, summed
    spam = json.loads((tmp_path / "raw_results" / "spam_calibration_counts_per_chunk.json").read_text(encoding="utf-8"))["chunk_0"]
    assert len(spam) == N_SPAM and all(sum(c.values()) == 450 for c in spam.values())

    rec = json.loads((tmp_path / "job_records" / "job_record_chunk_0.json").read_text(encoding="utf-8"))
    assert rec["status"] == "complete" and rec["nshots"] == 450 and rec["shot_parts"] == [150, 150, 150]
    assert rec["n_executed_circuits"] == 3 * (N_SPAM + N_MAIN) and rec["max_shots_per_circuit"] == 200
    assert [p["resumed"] for p in rec["shot_part_times"]] == [False] * 3
    assert rec["timing"]["batch_shots"] == [150] * 3
    assert rec["timing"]["qpu_s"] == pytest.approx(3 * (N_SPAM + N_MAIN) * JT.aqt_qpu_s_per_circuit(150))

    w = RT.qpu_workload(tmp_path)
    assert w["n_circuits"] == 3 * (N_SPAM + N_MAIN) and w["n_shots"] == 450 * (N_SPAM + N_MAIN)
    assert w["n_split_chunks"] == 1


def test_interrupted_chunk_resumes_at_the_missing_part(tmp_path):
    obj = _qbcs(tmp_path, 600)
    with pytest.raises(RuntimeError, match="connection lost"):
        obj.submit_chunk_aqt(0, _StubAQT(fail_on_call=2))            # part 0 done, part 1 fails
    assert (tmp_path / "raw_results" / "shot_parts" / "chunk_0_part_0.json").exists()
    failed = json.loads((tmp_path / "job_records" / "job_record_chunk_0.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed" and failed["failed_part"] == 1   # the attempt's cost is kept
    assert "connection lost" in failed["failure"] and failed["timing"]["queue_s"] >= 0
    assert failed["failure_stage"] == "execution" and failed["may_exist_on_provider"] is False
    assert list(failed)[:2] == ["chunk_id", "status"]                  # status near the top

    backend = _StubAQT()
    obj.submit_chunk_aqt(0, backend)
    assert backend.calls == [(N_SPAM + N_MAIN, 200)] * 2             # only parts 1 and 2 re-run
    rec = json.loads((tmp_path / "job_records" / "job_record_chunk_0.json").read_text(encoding="utf-8"))
    assert rec["status"] == "complete"
    assert [p["resumed"] for p in rec["shot_part_times"]] == [True, False, False]
    archived = json.loads((tmp_path / "job_records" / "failed_attempts" / "job_record_chunk_0_failed_1.json").read_text(encoding="utf-8"))
    assert archived["status"] == "failed" and archived["submitted_at"] == failed["submitted_at"]
    counts = json.loads((tmp_path / "raw_results" / "results_counts.json").read_text(encoding="utf-8"))["chunk_0"]
    assert all(sum(c.values()) == 600 for job in counts.values() for c in job.values())


def test_circuits_finished_in_a_failed_batch_are_counted(tmp_path):
    """A failure after k circuits of a batch keeps those k circuits in the failed record (circuits, shots,
    QPU time), and the failing batch's wall time is booked once: as queue time beyond that QPU time."""
    k = 3
    obj = _qbcs(tmp_path, 600)
    with pytest.raises(RuntimeError, match="inside the batch"):
        obj.submit_chunk_aqt(0, _StubAQT(fail_after=(2, k)))         # part 0 done, part 1 fails after k circuits
    failed = json.loads((tmp_path / "job_records" / "job_record_chunk_0.json").read_text(encoding="utf-8"))
    t = failed["timing"]
    assert failed["failed_part"] == 1 and failed["n_executed_circuits"] == k and failed["n_executed_shots"] == k * 200
    assert t["batch_sizes"] == [k] and t["batch_shots"] == [200]
    assert t["qpu_s"] == pytest.approx(k * JT.aqt_qpu_s_per_circuit(200))
    assert t["queue_s"] == pytest.approx(max(t["failed_batch_s"] - t["qpu_s"], 0.0), abs=2e-3)
    assert t["queue_s"] <= t["failed_batch_s"]                       # not added a second time
    fa = RT.failed_attempts_breakdown(tmp_path)
    assert fa["n_circuits"] == k and fa["n_shots"] == k * 200
    w = RT.qpu_workload(tmp_path)
    assert w["n_failed_circuits"] == k and w["n_failed_shots"] == k * 200 and w["n_jobs"] == 0


def test_shot_parts_are_timestamped(tmp_path):
    obj = _qbcs(tmp_path, 600)
    obj.TIMESTAMP_FILE = tmp_path / TS.TIMESTAMP_FILENAME
    with pytest.raises(RuntimeError, match="connection lost"):
        obj.submit_chunk_aqt(0, _StubAQT(fail_on_call=2))            # part 0 done, part 1 fails
    obj.submit_chunk_aqt(0, _StubAQT())                               # part 0 reused, parts 1 and 2 run
    for path in (obj.TIMESTAMP_FILE, TS.utc_timestamp_path(obj.TIMESTAMP_FILE)):
        labels = [lab for lab, _ in TS._read_timestamp_rows(path) if "_part" in lab]
        assert labels == ["qpu_chunk0_part0_start", "qpu_chunk0_part0_finished",
                          "qpu_chunk0_part1_start",                                   # interrupted
                          "qpu_chunk0_part1_start", "qpu_chunk0_part1_finished",
                          "qpu_chunk0_part2_start", "qpu_chunk0_part2_finished"]
        assert not any(RT._QPU_CHUNK_LABEL_RE.match(lab) for lab in labels)


def test_unsplit_chunk_runs_once(tmp_path):
    obj, backend = _qbcs(tmp_path, 10), _StubAQT()
    obj.submit_chunk_aqt(0, backend)
    assert backend.calls == [(N_SPAM + N_MAIN, 10)]
    rec = json.loads((tmp_path / "job_records" / "job_record_chunk_0.json").read_text(encoding="utf-8"))
    assert rec["shot_parts"] == [10] and rec["n_executed_circuits"] == N_SPAM + N_MAIN
    assert RT.qpu_workload(tmp_path)["n_split_chunks"] == 0
