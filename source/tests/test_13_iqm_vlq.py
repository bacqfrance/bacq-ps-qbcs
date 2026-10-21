"""
test_13_iqm_vlq.py
==================
Unit tests (no simulator run, no QPU, no network) for IT4I's VLQ (IQM Star 24) support:

  - iqm_vlq.gate_roles: each virtual qubit's prx / MOVE-pair / CZ counts, read from a
    transpiled star circuit;
  - iqm_vlq.select_vlq_qubits on a saved calibration set (source/tests/iqm_vlq_check_data/iqm_vlq_calibration_fixture.json,
    trimmed from IT4I's public API) with the full benchmark's roles: every ordered choice of
    qubits with complete metrics is scored (a qubit without MOVE data is excluded) and the best
    one is chosen (T1/T2 idle decay breaks ties between equal fidelities);
  - iqm_vlq.snapshot_from_calibration: the effective all-to-all CZ (error 1 - F_cz·F_move of
    the better direction, duration 2·MOVE + CZ; prx error per native gate) rebuilds into a Target and an Aer noise model;
  - layout_selection.count_2q_ops ignores MOVEs;
  - iqm_vlq.vlq_collect_timing on a real job (source/tests/iqm_vlq_check_data/iqm_vlq_job_timing_fixture.json: the mini
    benchmark's IQM timeline and qaas runtimes, 2026-09-29) — reproduces what collection recorded:
    Resonance-equivalent billed time, execution time from remote_hw_runtime, queue split into
    IT4I's and IQM's parts;
  - iqm_vlq.call: every network call to IT4I / HEAppE is logged (OK / FAIL / TIMEOUT with its
    duration) to the terminal and the run's iqm_vlq_connection.log, and abandoned after its time
    limit with the place it was waiting in (a HEAppE request that is never answered);
  - iqm_vlq.use_run_calibration: a calibration_file= that is a run's calibration_snapshot.json is accepted only with
    that run's vlq_calibration.json beside it (the raw file is copied, the snapshot rebuilt); alone it exits with code 2.
The QPU execution time of every backend is tested in test_11_circuit_check_and_timing.py.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit

from source.core.qpu import iqm_vlq as V
from source.core.records import job_timing as JT
from source.core.qpu.layout_selection import count_2q_ops

DATA = Path(__file__).parent / "iqm_vlq_check_data"   # real VLQ data, see the docstring
CAL  = json.loads((DATA / "iqm_vlq_calibration_fixture.json").read_text(encoding="utf-8"))


# The full benchmark's deepest circuit on VLQ (transpiled, 2026-09-29): v0 is always moved,
# v2 always does the CZ with the resonator.
ROLES = {"n_prx": [96, 153, 210], "n_move": [76, 38, 0], "n_cz": [0, 38, 76]}


def test_gate_roles_from_a_transpiled_star_circuit():
    from qiskit.circuit import Gate
    qc = QuantumCircuit(25)                       # QB1..QB24 + the resonator (index 24)
    move = Gate("move", 2, [])
    qc.r(0.1, 0.2, 13)
    qc.append(move, [13, 24]); qc.cz(6, 24); qc.append(move, [13, 24])     # CX(v0, v1): v0 moved
    qc.append(move, [6, 24]);  qc.cz(0, 24); qc.append(move, [6, 24])      # CX(v1, v2): v1 moved
    assert V.gate_roles(qc, [13, 6, 0]) == {"n_prx": [1, 0, 0], "n_move": [1, 1, 0], "n_cz": [0, 1, 1]}


def test_selection_scores_every_ordered_choice_by_role():
    metrics = V.qubit_metrics(CAL)
    rec     = V.select_vlq_qubits(ROLES, CAL)
    assert rec["method"] == "star_roles" and len(rec["physical_qubits"]) == 3
    no_move = [q for q, v in metrics.items() if v["move"] is None]
    usable  = [q for q in metrics if q not in no_move]
    n = len(usable)                               # all ordered triples of qubits with complete metrics
    assert rec["n_candidates"] == n * (n - 1) * (n - 2)
    assert set(rec["excluded_qubits"]) == set(no_move) and not set(rec["qubit_names"]) & set(no_move)
    assert rec["physical_qubits"] == [int(q[2:]) - 1 for q in rec["qubit_names"]]
    assert rec["decoherence_included"] and rec["t_circuit_s"] > 0 and rec["idle_log"] <= 0   # the fixture has T1, T2 and durations
    assert rec["log_success"] == pytest.approx(sum(rec["chosen_grades"].values()) + rec["idle_log"])
    assert rec["log_success"] == max(s for _, s in rec["top_candidates"])

    # an independent recomputation of one grade: readout + prx + MOVE pairs + CZs
    q, v = rec["qubit_names"][2], 2
    m    = metrics[q]
    f_prx = 1 - (1 - m["f1q"]) / V.PRX_PER_CLIFFORD
    assert V.prx_fidelity(m["f1q"]) == pytest.approx(f_prx)
    assert rec["chosen_grades"][q] == pytest.approx(
        math.log(m["readout"]) + ROLES["n_prx"][v] * math.log(f_prx) + ROLES["n_cz"][v] * math.log(m["cz"]))
    assert rec["calibration_set_id"] == CAL["calibration_set_id"]


def test_idle_decay_decides_between_equal_fidelities():
    """With every fidelity equal, only T1/T2 and the gate durations separate the qubits: the long-lived ones win, and a
    calibration with no T2 gives up the term for every choice (the old, fidelity-only score)."""
    metrics = V.qubit_metrics(CAL)
    usable  = [q for q, m in metrics.items() if V._qubit_logs(m) is not None]
    assert len(usable) >= 4
    flat    = metrics[usable[0]]                                           # one qubit's fidelities and durations for all
    m_flat  = {q: dict(flat, t1=50e-6, t2=20e-6) for q in usable}
    m_flat[usable[3]].update(t1=500e-6, t2=400e-6)                         # one qubit that outlives the rest
    orig    = V.qubit_metrics
    try:
        V.qubit_metrics = lambda cal: {q: dict(m) for q, m in m_flat.items()}
        rec = V.select_vlq_qubits(ROLES, CAL)
        assert rec["decoherence_included"] and usable[3] in rec["qubit_names"]
        m_flat[usable[3]]["t2"] = None
        rec = V.select_vlq_qubits(ROLES, CAL)
        assert not rec["decoherence_included"] and rec["t_circuit_s"] is None and rec["idle_log"] is None
    finally:
        V.qubit_metrics = orig


def test_snapshot_folds_the_resonator_into_an_all_to_all_cz():
    from source.core.qpu.qpu_snapshot import noisy_sampler_from_snapshot
    snap    = V.snapshot_from_calibration(CAL)
    metrics = V.qubit_metrics(CAL)
    cz      = {tuple(e["qargs"]): e for e in snap["operations"]["cz"]}
    a, b    = "QB1", "QB2"
    best    = max(metrics[b]["cz"] * metrics[a]["move"], metrics[a]["cz"] * metrics[b]["move"])
    assert cz[(0, 1)]["error"] == pytest.approx(1 - best)
    r = {e["qargs"][0]: e["error"] for e in snap["operations"]["r"]}
    assert r[0] == pytest.approx((1 - metrics[a]["f1q"]) / V.PRX_PER_CLIFFORD)   # per native prx, not per Clifford
    assert snap["num_qubits"] == 24 and len(snap["operations"]["measure"]) == 24
    backend, noise_model, _ = noisy_sampler_from_snapshot(snap)
    assert {"cz", "measure", "r"} <= set(noise_model.noise_instructions)
    assert backend.target.num_qubits == 24


def test_save_vlq_calibration_writes_the_raw_calibration_and_its_snapshot(tmp_path):
    from source.core.qpu.qpu_snapshot import SNAPSHOT_FILENAME
    run_dir = tmp_path / "run"                     # created by the helper
    snapshot_path = V.save_vlq_calibration(run_dir, CAL)
    assert snapshot_path == run_dir / SNAPSHOT_FILENAME
    assert json.loads((run_dir / V.VLQ_CALIBRATION_FILENAME).read_text(encoding="utf-8")) == CAL
    assert json.loads(snapshot_path.read_text(encoding="utf-8")) == json.loads(json.dumps(V.snapshot_from_calibration(CAL)))


def test_move_is_not_a_two_qubit_gate():
    from qiskit.circuit import Gate
    qc = QuantumCircuit(2)
    qc.append(Gate("move", 2, []), [0, 1]); qc.cz(0, 1); qc.append(Gate("move", 2, []), [0, 1])
    assert count_2q_ops(qc) == 1


# A real VLQ job (the mini benchmark's, 2026-09-29): its IQM timeline, qaas runtimes and the
# values collection recorded from them.
JOB = json.loads((DATA / "iqm_vlq_job_timing_fixture.json").read_text(encoding="utf-8"))


def _qjob(saved):
    """A stand-in for a finished qaas QJob, rebuilt from the saved job (a QJob can't be pickled)."""
    timeline = [SimpleNamespace(source=t["source"], status=t["status"],
                                timestamp=datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00")))
                for t in saved["timeline"]]
    remote = SimpleNamespace(_iqm_job=SimpleNamespace(data=SimpleNamespace(timeline=timeline, compilation=None)),
                             job_id=lambda: saved["iqm_job_id"])
    return SimpleNamespace(remote_job=remote, remote_hw_runtime=saved["remote_hw_runtime"],
                           consumpted_resources=saved["consumpted_resources"],
                           allocation_amount=saved["allocation_amount"], events={})


def test_vlq_timing_of_a_real_job():
    t   = V.vlq_collect_timing(_qjob(JOB["job"]), JOB["upload_end_at"])
    exp = JOB["expected"]
    for k in ("qpu_s", "qpu_s_source", "qpu_exec_s", "qpu_exec_s_source", "created_at", "execution_start", "execution_end"):
        assert t[k] == exp[k], k
    for k in ("compilation_seconds", "execution_seconds", "queue_s", "it4i_wait_s", "iqm_queue_s"):
        assert t[k] == pytest.approx(exp[k]), k
    # billed like IQM Resonance: ceil(compilation + execution); the queue is IT4I's wait + IQM's
    assert t["qpu_s"] == math.ceil(t["compilation_seconds"] + t["execution_seconds"])
    assert t["queue_s"] == pytest.approx(t["it4i_wait_s"] + t["iqm_queue_s"])
    # qaas' remote_hw_runtime is IQM's execution_started -> execution_ended
    assert t["qpu_exec_s"] == pytest.approx(t["execution_seconds"])
    assert JT.qpu_execution_s(t, "QAAS_IT4I") == (t["qpu_exec_s"], "remote_hw_runtime")


def test_connection_calls_are_logged_and_time_limited(tmp_path):
    # Every call to IT4I / HEAppE is a logged stage: OK / FAIL / TIMEOUT with its duration, in the
    # terminal and the run's iqm_vlq_connection.log. Events before the log is set are written into it.
    import threading
    try:
        V.set_connection_log(None)
        V.conn_log("before the log was set")
        log = tmp_path / V.CONNECTION_LOG_NAME
        V.set_connection_log(log)
        assert V.call("public status", lambda: 42, timeout_s=5) == 42
        with pytest.raises(ValueError):
            V.call("job status", lambda: (_ for _ in ()).throw(ValueError("bad")), "HEAppE job 7", timeout_s=5)
        release = threading.Event()
        with pytest.raises(V.VLQTimeout, match="no answer"):
            V.call("get_backend", lambda: release.wait(30), "VLQ-EU", timeout_s=0.3)   # a request never answered
        release.set()
        text = log.read_text(encoding="utf-8")
        assert text.index("before the log was set") < text.index("START   public status")
        assert "OK      public status" in text and "FAIL    job status (HEAppE job 7)" in text and "ValueError: bad" in text
        assert "TIMEOUT get_backend (VLQ-EU)" in text and "Waiting in:" in text and "release.wait" in text
    finally:
        V.set_connection_log(None)
        V._pending_events.clear()


def test_calibration_file_snapshot_is_accepted_only_with_its_raw_file_beside_it(tmp_path):
    from source.core.qpu.qpu_snapshot import SNAPSHOT_FILENAME
    real = tmp_path / "real_run"                   # a real run's folder: raw file + the snapshot built from it
    snapshot_path = V.save_vlq_calibration(real, CAL)
    sim = tmp_path / "sim_run"
    cal, source = V.use_run_calibration(sim, calibration_file=snapshot_path)
    assert cal == CAL and (sim / V.VLQ_CALIBRATION_FILENAME).exists() and (sim / SNAPSHOT_FILENAME).exists()
    assert source.endswith(SNAPSHOT_FILENAME) is False and source.endswith(V.VLQ_CALIBRATION_FILENAME)


def test_calibration_file_snapshot_alone_exits_with_code_2(tmp_path):
    lone = tmp_path / "lone" / "calibration_snapshot.json"
    lone.parent.mkdir()
    lone.write_text(json.dumps(V.snapshot_from_calibration(CAL)), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        V.use_run_calibration(tmp_path / "sim_run", calibration_file=lone)
    assert e.value.code == 2
