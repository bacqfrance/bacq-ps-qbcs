"""
test_30_dummy_estimates.py
==========================
Unit tests (no simulation build, no network, temporary folders) for the dummy noisy emulators
(source/core/analysis/dummy_estimates.py and the driver's skip / resume rule):

  - gate counts of a circuit: rz free, other 1-qubit gates N1q, 2-qubit gates N2q, measure / reset Nm, a circuit
    without classical registers measured on every qubit, a gate on 3 qubits refused;
  - the time model (per_shot_overhead + N1q t_1q + N2q t_2q + t_meas per shot) and each pricing model
    (ibm_payg, braket, braket_scaled, hqc), against values computed by hand below;
  - a null or missing term of the definitions is an error, never 0; all eight shipped definitions are valid;
  - dummy_estimates.json: written once (a resumed run keeps its inputs), no absolute path;
  - run_jobs(): a run's chunk jobs (SPAM circuits first) and accompanying health checks read from a cached circuit
    folder and the job / health-check records, scoped to the first n_samples chunks;
  - the driver's rule dummy_run_folder(): a finished run is skipped (no #2), an unfinished one is resumed, none is new.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CCXGate

from source.core.analysis import dummy_estimates as DE
from source.core.records import run_names as RN

TIMING = {"t_1q": 1e-6, "t_2q": 1e-5, "t_meas": 1e-4, "per_shot_overhead": 1e-3}
A, B, HC = (3, 2, 3), (0, 0, 3), (3, 4, 3)   # (N1q, N2q, Nm) of the three circuits below
# one shot: A 1e-3 + 3e-6 + 2e-5 + 1e-4 = 1.123e-3 s; B 1e-3 + 1e-4 = 1.1e-3 s; HC 1e-3 + 3e-6 + 4e-5 + 1e-4 = 1.143e-3 s
JOBS = [{"kind": "chunk", "shots": 100, "circuits": [A, B]}, {"kind": "health_check", "shots": 50, "circuits": [HC]}]
CHUNKS_S = 100 * (1.123e-3 + 1.1e-3)    # 0.2223
HCS_S    = 50 * 1.143e-3                # 0.05715


def test_gate_counts():
    qc = QuantumCircuit(3)
    qc.h(0); qc.rz(0.1, 1); qc.sx(2); qc.x(0)       # rz free: 3 counted
    qc.cx(0, 1); qc.rzz(0.2, 1, 2); qc.cz(0, 2)     # 3 two-qubit gates
    qc.barrier()
    assert DE.circuit_op_counts(qc) == (3, 3, 3)    # no classical register: measured like measure_all (3 qubits)
    qc.measure_all()
    assert DE.circuit_op_counts(qc) == (3, 3, 3)    # the explicit measurements, the barrier ignored
    qc.reset(0)
    assert DE.circuit_op_counts(qc)[2] == 4
    big = QuantumCircuit(3)
    big.append(CCXGate(), [0, 1, 2])
    with pytest.raises(ValueError, match="3 qubits"):
        DE.circuit_op_counts(big)


def test_time_model():
    assert DE.circuit_time_s(A, TIMING) == pytest.approx(1.123e-3)
    total, chunks, hcs = DE.modelled_qpu_time(JOBS, TIMING)
    assert (chunks, hcs, total) == pytest.approx((CHUNKS_S, HCS_S, CHUNKS_S + HCS_S))
    assert total == pytest.approx(0.27945)


def test_ibm_payg_cost_is_not_rounded_to_whole_seconds():
    total = DE.modelled_qpu_time(JOBS, TIMING)[0]
    cost = DE.modelled_cost(JOBS, {"model": "ibm_payg", "usd_per_qpu_minute": 96.0}, total)
    assert cost["usd"] == pytest.approx(0.27945 / 60 * 96) and cost["usd"] == pytest.approx(0.44712)


@pytest.mark.parametrize("model", ["braket", "braket_scaled"])
def test_braket_cost(model):
    shared = {"model": model, "usd_per_task": 0.30, "usd_per_shot": 0.001}
    program_sets = DE.modelled_cost(JOBS, {**shared, "circuits_per_task": 100}, 0)   # the chunk job is 1 task, the health check 1
    assert (program_sets["n_tasks"], program_sets["n_shots"]) == (2, 250)             # 2 * 100 + 50 shots, per circuit
    assert program_sets["usd"] == pytest.approx(2 * 0.30 + 250 * 0.001) and program_sets["usd"] == pytest.approx(0.85)
    one_each = DE.modelled_cost(JOBS, {**shared, "circuits_per_task": 1}, 0)          # one circuit per task: 2 + 1 tasks
    assert one_each["n_tasks"] == 3 and one_each["usd"] == pytest.approx(3 * 0.30 + 0.25) and one_each["usd"] == pytest.approx(1.15)
    many = [{"kind": "chunk", "shots": 10, "circuits": [A] * 101}]
    assert DE.modelled_cost(many, {**shared, "circuits_per_task": 100}, 0)["n_tasks"] == 2   # 101 circuits: 2 program sets


def test_hqc_cost():
    pricing = {"model": "hqc", "hqc_fixed": 5, "weights": {"n1q": 1, "n2q": 10, "nm": 5}, "hqc_divisor": 5000, "usd_per_hqc": 12.5}
    # 5 + C/5000 (N1q + 10 N2q + 5 Nm): A at C = 100: 5 + 0.02 * 38 = 5.76; B: 5 + 0.02 * 15 = 5.3; HC at C = 50: 5 + 0.01 * 58 = 5.58
    cost = DE.modelled_cost(JOBS, pricing, 0)
    assert cost["hqc"] == pytest.approx(5.76 + 5.3 + 5.58) and cost["hqc"] == pytest.approx(16.64)
    assert cost["usd"] == pytest.approx(16.64 * 12.5) and cost["usd"] == pytest.approx(208.0)


def test_null_terms_are_errors():
    definitions = DE.load_definitions()
    assert len(definitions) == 8
    for name, entry in definitions.items():
        DE.validate_entry(name, entry)
        record = DE.estimates_record(name, entry)
        DE.validate_entry(name, record, saved=True)
        assert record["noise_type"] == "pauli" and isinstance(record["lnn"], bool)
    assert DE.estimates_record("quantinuum-h2", definitions["quantinuum-h2"])["infidelity_total"] == pytest.approx(1e-3)
    entry = json.loads(json.dumps(definitions["ionq-forte"]))
    entry["timing"]["t_2q"] = None
    with pytest.raises(ValueError, match="t_2q"):
        DE.validate_entry("ionq-forte", entry)
    entry = json.loads(json.dumps(definitions["quantinuum-h2"]))
    entry["pricing"]["usd_per_hqc"] = None
    with pytest.raises(ValueError, match="usd_per_hqc"):
        DE.validate_entry("quantinuum-h2", entry)
    entry["pricing"]["model"] = "free"
    with pytest.raises(ValueError, match="unknown pricing model"):
        DE.validate_entry("quantinuum-h2", entry)
    entry = json.loads(json.dumps(definitions["ionq-forte"]))
    del entry["architecture"]
    with pytest.raises(ValueError, match="architecture.*definition"):
        DE.validate_entry("ionq-forte", entry)
    record = DE.estimates_record("ionq-forte", definitions["ionq-forte"])
    assert record["architecture"] == "Forte"
    del record["architecture"]
    with pytest.raises(ValueError, match="architecture.*dummy_estimates.json"):
        DE.validate_entry("ionq-forte", record, saved=True)


def test_dummy_estimates_file_is_written_once(tmp_path):
    entry = DE.load_definitions()["ibm-marrakesh"]
    path = DE.write_dummy_estimates(tmp_path / "run", "ibm-marrakesh", entry)
    assert path.name == "dummy_estimates.json"
    first = path.read_text(encoding="utf-8")
    assert "Users" not in first and str(tmp_path) not in first
    changed = json.loads(json.dumps(entry))
    changed["timing"]["t_1q"] = 9.9
    DE.write_dummy_estimates(tmp_path / "run", "ibm-marrakesh", changed)      # a resumed run keeps what it started with
    assert path.read_text(encoding="utf-8") == first
    loaded = DE.load_dummy_estimates(tmp_path / "run")
    assert loaded["name"] == "ibm-marrakesh" and loaded["timing"]["t_1q"] == entry["timing"]["t_1q"]
    with pytest.raises(FileNotFoundError):
        DE.load_dummy_estimates(tmp_path / "other")


def _qpy(path, circuits):
    from qiskit.qpy import dump
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        dump(circuits, f)


def test_run_jobs_from_the_cache_and_records(tmp_path, monkeypatch):
    from source.core.common import CIRCUITS_ROOT_ENV, get_circuit_folder, get_chunk_folder
    monkeypatch.setenv(CIRCUITS_ROOT_ENV, str(tmp_path / "circuits"))
    estimates = DE.estimates_record("ionq-forte", DE.load_definitions()["ionq-forte"])
    info = SimpleNamespace(lnn=False, list_times=[0.2], dt=0.2, circuits_qiskit_v13_qpy=True)
    cfg = {"lnn_architecture": False, "T_list": [0.2], "dt": 0.2,
           "coherent_noise": {"noise_type": "pauli", "infidelity_total": estimates["infidelity_total"], "ratio_ct": estimates["ratio_ct"]}}
    folder = get_chunk_folder(get_circuit_folder(cfg, top_folder=True), "qiskit_v13")
    spam = QuantumCircuit(2); spam.x(0); spam.measure_all()          # (1, 0, 2)
    chunk = QuantumCircuit(2); chunk.h(0); chunk.rz(0.3, 0); chunk.cx(0, 1)   # (1, 1, 2): no register, measured like measure_all
    for nc in range(2):
        _qpy(Path(folder) / f"chunk_{nc}_qiskit_v13.qpy", [chunk])
    _qpy(Path(folder) / "spam_qiskit_v13.qpy", [spam, spam])
    run = tmp_path / "run"
    (run / "job_records").mkdir(parents=True)
    for nc in range(2):
        (run / "job_records" / f"job_record_chunk_{nc}.json").write_text(
            json.dumps({"chunk_id": nc, "status": "complete", "nshots": 10, "n_spam": 2, "n_circuits": 1}), encoding="utf-8")
    (run / "health_checks").mkdir()
    (run / "health_checks" / "health_check_1.json").write_text(json.dumps({"number": 1, "status": "complete", "nshots": 100}), encoding="utf-8")
    (run / "health_checks" / "health_check_2.json").write_text(json.dumps({"number": 2, "status": "submitted", "nshots": 100}), encoding="utf-8")

    jobs = DE.run_jobs(run, info, estimates)
    assert [(j["kind"], j["shots"], j["circuits"]) for j in jobs] == [
        ("chunk", 10, [(1, 0, 2), (1, 0, 2), (1, 1, 2)]), ("chunk", 10, [(1, 0, 2), (1, 0, 2), (1, 1, 2)]),
        ("health_check", 100, [(3, 4, 3)])]                          # the unfinished health check is not counted
    assert [j["kind"] for j in DE.run_jobs(run, info, estimates, n_samples=1)] == ["chunk", "health_check"]
    forte = DE.load_definitions()["ionq-forte"]["timing"]
    total = DE.modelled_qpu_time(jobs, forte)[0]
    one = lambda c, shots: shots * (forte["per_shot_overhead"] + c[0] * forte["t_1q"] + c[1] * forte["t_2q"] + forte["t_meas"])
    assert total == pytest.approx(2 * (2 * one((1, 0, 2), 10) + one((1, 1, 2), 10)) + one((3, 4, 3), 100))


def _run_folder(root, name, collected, sample_size=2):
    d = root / name
    (d / "raw_results").mkdir(parents=True)
    (d / "run_info.json").write_text(json.dumps({"sample_size": sample_size}), encoding="utf-8")
    (d / "raw_results" / "results_counts.json").write_text(json.dumps({f"chunk_{i}": {} for i in range(collected)}), encoding="utf-8")


def test_driver_skips_finished_runs_and_resumes_unfinished_ones(tmp_path):
    undated = RN.local_run_name("mini", "sim-dummy", True, device="ibm-marrakesh")
    root = str(tmp_path)
    folder, finished = DE.dummy_run_folder(undated, root)                      # nothing yet: a new run, #1 today
    assert not finished and folder == f"{undated}_{RN.today_utc()}_#1"
    first = f"{undated}_2026-10-01_#1"
    _run_folder(tmp_path, first, collected=1)
    assert DE.dummy_run_folder(undated, root) == (first, False)                # unfinished: resumed in the same folder
    (tmp_path / first / "raw_results" / "results_counts.json").write_text(json.dumps({"chunk_0": {}, "chunk_1": {}}), encoding="utf-8")
    assert DE.dummy_run_folder(undated, root) == (first, True)                 # finished: skipped, no #2
    other = RN.local_run_name("mini", "sim-dummy", False, device="ionq-forte")
    assert DE.dummy_run_folder(other, root)[1] is False                        # another dummy is independent


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
