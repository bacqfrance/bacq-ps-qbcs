"""
test_17_local_job_records.py
============================
Unit tests (Aer only, no myQLM build, no network, a temporary simulation run folder) for the
records and logs a local simulation writes, aligned with those of a real-QPU run:

  - submit_chunk_local_qiskit() writes job_records/job_record_chunk_<nc>.json like a real
    QPU's: status 'complete', service_name 'local', submitted_at / finished_at, nshots,
    n_spam / n_circuits, a timing block (transpile / submit / collect), circuit_check
    'not_available' (the simulator returns no executed circuits), the run's CONFIG without
    folder_path — and no absolute path anywhere in it.
  - A sampler error writes a 'failed' record (failure, failure_stage 'execution') and
    re-raises; the rerun archives it to job_records/failed_attempts/ and the analysis counts
    it as a failed attempt (failed_attempts_breakdown, collect_chunk_jobs).
  - chunks_to_submit() of a simulation run decides by the results: a chunk whose record says
    'complete' but whose results are gone is simulated again.
  - collect_chunk_jobs() reads both the new records and an old run's
    chunk_times_UTC_time.json, also mixed in one run (a run resumed after the change).
  - append_submit_log() writes the shared submit.log entry (relative run folder, chunk
    range, shots, extra lines, health-check numbers, the POST-check line when it failed);
    health_check_log_line() and print_run_config() format their lines.
  - run_relative_path() turns any run path (Windows, Linux, relative) into datafiles/<run>/... — what
    records store, never an absolute path.
  - snapshot_log_lines() summarizes a calibration snapshot (median 2q error, the run
    qubits' pairs and T1/T2) and never raises on a missing file.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import json

import pytest
from source.tests.testing_functions import run_name_for_test
TEST_RUN = run_name_for_test("test17").split("/")[-1]   # the name alone: these runs live in tmp_path
from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

import source.core.qbcs as Q
from source.core.records.health_check_records import (append_submit_log, collect_chunk_jobs, create_health_check,
                                         health_check_log_line, print_run_config)
from source.core.analysis.run_timing import failed_attempts_breakdown
from source.core.qpu.qpu_snapshot import snapshot_log_lines


def _circuits(n):
    out = []
    for _ in range(n):
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.measure_all()                # creg 'meas', as submit_chunk_local_qiskit() reads it
        out.append(qc)
    return out


class _Sampler:
    """Aer's SamplerV2, whose first `fail` run() calls raise."""

    def __init__(self, fail=0):
        self.inner, self.fail = AerSamplerV2(), fail

    def run(self, circuits, shots=None):
        if self.fail:
            self.fail -= 1
            raise RuntimeError("simulated out-of-memory")
        return self.inner.run(circuits, shots=shots)


def _qbcs(run_path, monkeypatch):
    obj = Q.bacq_ps_qbcs(sample_size=8, chunk_size=6, list_circuit_times=[0.2], nshots=20,
                         do_SPAM=True, error_mitigation_strategies=["SPAM", "NEC", "RC+NEC", "cRC+NEC"])
    obj.CONFIG["folder_path"] = str(run_path)
    obj.TIMESTAMP_FILE        = run_path / "timestamps_local_time.csv"
    obj.n_chunks              = 2                                  # build() sets these two
    obj.chunk_size            = obj.n_depths * len(obj.roles)
    obj._qk_spam_circuits     = _circuits(8)
    obj._load_raw_chunk_qiskit = lambda nc: _circuits(obj.chunk_size)
    monkeypatch.setattr(Q, "export_circuit_drawings", lambda *a, **k: None)
    return obj


@pytest.fixture
def run(tmp_path, monkeypatch):
    run_path = tmp_path / TEST_RUN
    run_path.mkdir()
    return run_path, _qbcs(run_path, monkeypatch)


def _record(run_path, nc):
    return json.loads((run_path / "job_records" / f"job_record_chunk_{nc}.json").read_text(encoding="utf-8"))


def test_complete_record_like_a_real_qpu(run):
    run_path, obj = run
    assert obj.submit_chunk_local_qiskit(0, qiskit_sampler=_Sampler()) is True
    rec = _record(run_path, 0)
    assert (rec["status"], rec["service_name"], rec["chunk_id"], rec["job_id"]) == ("complete", "local", 0, "N/A")
    assert (rec["nshots"], rec["n_spam"], rec["n_circuits"]) == (20, 8, 6)
    assert rec["submitted_at"] <= rec["finished_at"]
    assert set(rec["timing"]) >= {"transpile_s", "transpile_reused", "submit_s", "collect_s"}
    assert "qpu_s" not in rec["timing"] and "queue_s" not in rec["timing"]
    assert rec["circuit_check"]["status"] == "not_available"
    assert rec["submitted_circuit_files"] == [] and rec["layout"] == "N/A"   # noiseless: nothing transpiled
    assert rec["result_path"] == f"datafiles/{TEST_RUN}/raw_results/results_counts.json"
    assert "folder_path" not in rec["CONFIG"]
    text = (run_path / "job_records" / "job_record_chunk_0.json").read_text(encoding="utf-8")
    assert str(run_path) not in text and str(run_path).replace("\\", "/") not in text   # no absolute path
    assert list(rec)[:2] == ["chunk_id", "status"]                                       # ordered_record()
    # A second call skips the chunk and leaves its record alone
    assert obj.submit_chunk_local_qiskit(0, qiskit_sampler=_Sampler()) is False
    assert _record(run_path, 0) == rec


def test_failed_simulation_is_recorded_then_archived(run):
    run_path, obj = run
    with pytest.raises(RuntimeError, match="out-of-memory"):
        obj.submit_chunk_local_qiskit(1, qiskit_sampler=_Sampler(fail=1))
    rec = _record(run_path, 1)
    assert (rec["status"], rec["failure_stage"], rec["may_exist_on_provider"]) == ("failed", "execution", False)
    assert "out-of-memory" in rec["failure"] and rec["finished_at"]
    assert obj.chunks_to_submit(TEST_RUN) == [0, 1]

    assert obj.submit_chunk_local_qiskit(1, qiskit_sampler=_Sampler()) is True
    assert _record(run_path, 1)["status"] == "complete"
    archived = list((run_path / "job_records" / "failed_attempts").glob("*.json"))
    assert [p.name for p in archived] == ["job_record_chunk_1_failed_1.json"]
    assert failed_attempts_breakdown(run_path)["n_attempts"] == 1
    jobs = collect_chunk_jobs(run_path)
    assert sorted((j["chunks"], j["failed"]) for j in jobs) == [([1], False), ([1], True)]


def test_sim_run_decides_by_results(run):
    run_path, obj = run
    obj.submit_chunk_local_qiskit(0, qiskit_sampler=_Sampler())
    assert obj.chunks_to_submit(TEST_RUN) == [1]
    (run_path / "raw_results" / "results_counts.json").unlink()        # record still says 'complete'
    assert _record(run_path, 0)["status"] == "complete"
    assert obj.chunks_to_submit(TEST_RUN) == [0, 1]


def test_chunk_jobs_from_records_and_old_chunk_times(run):
    run_path, obj = run
    (run_path / "chunk_times_UTC_time.json").write_text(json.dumps({     # an old run, resumed
        "chunk_0": {"started_at": "2026-09-01T10:00:00Z", "finished_at": "2026-09-01T10:00:01Z"},
        "chunk_1": {"started_at": "2026-09-01T10:00:01Z", "finished_at": "2026-09-01T10:00:02Z"}}), encoding="utf-8")
    obj.submit_chunk_local_qiskit(1, qiskit_sampler=_Sampler())
    jobs = {j["chunks"][0]: j for j in collect_chunk_jobs(run_path)}
    assert sorted(jobs) == [0, 1]
    assert jobs[0]["start"].isoformat().startswith("2026-09-01T10:00:00")   # from chunk_times
    assert jobs[1]["start"] > jobs[0]["end"]                               # from the new record (now), not chunk_times


def test_append_submit_log(tmp_path, capsys):
    run_path = tmp_path / "sim_log_test"
    create_health_check(run_path, {"status": "complete", "P011": 0.97, "verdict": "functional"})
    create_health_check(run_path, {"status": "complete", "P011": 0.42, "verdict": "degraded"})
    path = append_submit_log(run_path, backend="AerSimulator (noiseless)", chunks=[0, 1, 2, 5], n_chunks=8,
                             nshots=50, run_qubits=[1, 2, 3], extra_lines=["Extra:            x"])
    append_submit_log(run_path, backend="ibm_x", chunks=[6], n_chunks=8, nshots=50,
                      post_hc_ok=False, post_hc_numbers=[3, 4])
    entries = path.read_text(encoding="utf-8").split("\n\n")
    assert entries[-1] == ""                                           # every entry ends with a blank line
    first = entries[0].splitlines()
    assert first[0].startswith("Submitted at:     20")
    assert first[1:] == ["Backend:          AerSimulator (noiseless)",
                         "Run folder:       datafiles/sim_log_test",
                         "Chunks submitted: 0-2,5 (4 of 8)",
                         "Shots:            50",
                         "Run qubits:       [1, 2, 3]",                   # no calibration snapshot: no lines for it
                         "Extra:            x",
                         "Health checks:    [1, 2]"]
    assert entries[1].splitlines()[-1].startswith("POST health check NOT submitted (attempts #3, #4 failed)")
    assert str(tmp_path) not in path.read_text(encoding="utf-8")

    from source.core.records.health_check_records import load_health_checks
    assert health_check_log_line("PRE ", load_health_checks(run_path)[0]) == \
        "PRE  health check #1: P(011)=0.970 → functional"
    assert health_check_log_line("POST", None) == "POST health check: not collected"

    print_run_config("sim_x", nshots=50, sample_size=8, lnn=True, do_spam=True, list_times=[0.2, 0.6],
                     chunk_size=12, extra=[("Backend", "Aer")])
    out = capsys.readouterr().out.splitlines()
    assert out[1:] == ["[Run config]", "  Run folder:       sim_x", "  Shots:            50",
                       "  Samples:          8", "  LNN:              True", "  SPAM:             True",
                       "  Depths:           2 (0.2 → 0.6)", "  Chunk size:       12", "  Backend:          Aer"]


def test_run_relative_path():
    from source.core.records.job_records import run_relative_path
    for run in (r"C:\Users\x\proj\datafiles\bacq_real_aqt", "/homelocal/x/proj/datafiles/bacq_real_aqt",
                "datafiles/bacq_real_aqt"):
        assert run_relative_path(run, "raw_results", "results_counts.json") == \
            "datafiles/bacq_real_aqt/raw_results/results_counts.json"


def test_snapshot_log_lines(tmp_path):
    assert snapshot_log_lines(tmp_path / "missing.json", [0, 1]) == []
    snap = {"backend": "fake_x", "downloaded_at": "2026-09-01T00:00:00+00:00", "calibration_set_id": None,
            "operations": {"cz": [{"qargs": [0, 1], "error": 0.01}, {"qargs": [2, 1], "error": 0.03},
                                  {"qargs": [2, 3], "error": 0.02}],
                           "sx": [{"qargs": [0], "error": 1e-4}]},
            "qubit_properties": [{"t1": 1e-4, "t2": 5e-5}, {"t1": 3e-4, "t2": None}, {"t1": None, "t2": 1e-4}]}
    path = tmp_path / "calibration_snapshot.json"
    path.write_text(json.dumps(snap), encoding="utf-8")
    assert snapshot_log_lines(path, [0, 1, 2]) == [
        "Calibration:      calibration_snapshot.json of fake_x, from 2026-09-01T00:00:00+00:00",
        "CZ error (median): 2.000e-02",
        "CZ(0,1): 1.000e-02   CZ(1,2): 3.000e-02",                     # (1,2) found as (2,1)
        "T1_run_qubits: 200.0µs   T2_run_qubits: 75.0µs",
    ]

    # append_submit_log() writes them in a fixed order: layout, run qubits, the calibration, then extra_lines
    log = append_submit_log(tmp_path, backend="fake_x", chunks=[0], n_chunks=1, nshots=10, layout="optimized",
                            run_qubits=[0, 1, 2], extra_lines=["Pending jobs at submission: 0"]).read_text(encoding="utf-8")
    lines = log.splitlines()
    i = lines.index("Layout:           optimized")
    assert lines[i + 1] == "Run qubits:       [0, 1, 2]"
    assert lines[i + 2].startswith("Calibration:      calibration_snapshot.json of fake_x")   # the run's own snapshot by default
    assert lines[i + 6] == "Pending jobs at submission: 0"
