"""
test_11_circuit_check_and_timing.py
===================================
Unit tests (no simulator, no QPU, no network) for

  - source/core/records/circuit_check.py: the executed-vs-submitted comparison, on
      * real data (source/tests/circuit_check_data_executed_vs_submitted/, see its manifest.json):
        what IBM (job.inputs["pubs"]) and IQM Resonance (GET jobs/{id}/payload) returned
        for real jobs of 2026-09-27, against the circuits we submitted — must all match,
        and a garnet submission compared with an emerald job must not;
      * synthetic data: circuits transpiled onto GenericBackendV2, with deliberately
        altered "executed" copies — every alteration must come back as "mismatch",
        malformed server data as "check_failed", and nothing may raise;
    plus the analysis summary (with its qubit-consistency fallback for old records) and
    the figure warning line;
  - source/core/records/job_timing.py: Europe/Berlin -> UTC conversion (incl. the repeated DST hour),
    AQT and legacy timing;
  - the timestamp logs (source/core/records/timestamps.py, source/core/analysis/run_timing.py): durations come from
    timestamps_utc.csv, fall back to timestamps_local_time.csv when the UTC log does not
    cover the calculation; the logs are append-only and an interrupted attempt is skipped
    when reading; the build time is the sum of its stages;
  - old runs: the analysis refuses AQT records without a timing block; old IBM, IQM Resonance and
    IQM EQE1 records are read as legacy runs, and so is a record whose QPU time comes from its old fields;
  - QPU execution time (QPU_execution_time_s): its source per backend (IQM VLQ
    remote_hw_runtime, IQM Resonance execution_seconds or the execution interval, IBM
    circuits_execution_seconds, none for AQT), summed over the run's jobs by
    qpu_queue_breakdown and None ("N/A") when any job lacks it; qpu_workload's chunks per job;
  - layout selection (source/core/qpu/layout_selection.py): the native single-qubit gates are counted on
    the circuit transpiled onto one SWAP-free probe placement (the same on every placement).
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import copy
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.qpy import load, dump

from source.core.records import circuit_check as CC
from source.core.records import job_timing as JT
from source.core.analysis import run_timing as RT
from source.core.records import timestamps as TS

DATA = Path(__file__).resolve().parent / "circuit_check_data_executed_vs_submitted"


def _qpy(path):
    with open(path, "rb") as f:
        return load(f)


# ── Real provider data ────────────────────────────────────────────────────────
MANIFEST = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", sorted(MANIFEST))
def test_real_executed_vs_submitted(case):
    m         = MANIFEST[case]
    submitted = _qpy(DATA / m["submitted"])
    if m["provider"] == "IBM":
        executed = _qpy(DATA / m["executed"])
        pubs     = [(c, None, m["shots"]) for c in executed]
        result   = CC.check_ibm(submitted, pubs, m["shots"])
    else:
        payload  = json.loads((DATA / m["executed"]).read_text(encoding="utf-8"))
        names    = json.loads((DATA / m["qubit_names"]).read_text(encoding="utf-8"))
        result   = CC.check_iqm(submitted, payload, lambda i: names[str(i)], m["shots"])
    assert result["status"] == m["expected"], result
    assert result["n_executed"] == len(submitted)


# ── Synthetic circuits ────────────────────────────────────────────────────────
def _circuit():
    qc = QuantumCircuit(3, 3)
    qc.x([0, 1, 2]); qc.barrier()
    qc.cx(0, 1); qc.rz(0.3, 1); qc.cx(1, 2); qc.barrier()
    qc.measure(range(3), range(3))
    return qc


@pytest.fixture(scope="module")
def backend():
    return GenericBackendV2(num_qubits=5, basis_gates=["rz", "sx", "x", "cz"], coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]], seed=1)


@pytest.fixture(scope="module")
def isa(backend):
    return [transpile(_circuit(), backend, optimization_level=1, initial_layout=[1, 2, 3], seed_transpiler=1)
            for _ in range(3)]


def _pubs(circuits, shots=100):
    return [(c, None, shots) for c in circuits]


def _drop_first_gate(c, name):
    out = c.copy_empty_like()
    dropped = False
    for ins in c.data:
        if not dropped and ins.operation.name == name:
            dropped = True
            continue
        out.append(ins)
    return out


def test_identical_circuits_match(isa):
    assert CC.check_ibm(isa, _pubs(copy.deepcopy(isa)), 100)["status"] == CC.MATCH


@pytest.mark.parametrize("alteration", ["missing_gate", "other_qubits", "reordered", "missing_circuit", "angle", "shots"])
def test_altered_executed_circuits_mismatch(isa, backend, alteration):
    executed, shots = [c.copy() for c in isa], 100
    if alteration == "missing_gate":
        executed[1] = _drop_first_gate(executed[1], "cz")
    elif alteration == "other_qubits":
        executed[2] = transpile(_circuit(), backend, optimization_level=1, initial_layout=[2, 3, 4], seed_transpiler=1)
    elif alteration == "reordered":
        other = transpile(_circuit().reverse_ops(), backend, optimization_level=1, initial_layout=[1, 2, 3])
        executed = [other] + executed[1:]
    elif alteration == "missing_circuit":
        executed = executed[:-1]
    elif alteration == "angle":
        c = executed[0].copy_empty_like()
        for ins in executed[0].data:
            op = ins.operation
            if op.name == "rz":
                op = op.to_mutable(); op.params = [op.params[0] + 1e-3]
            c.append(op, ins.qubits, ins.clbits)
        executed[0] = c
    elif alteration == "shots":
        shots = 99
    result = CC.check_ibm(isa, _pubs(executed, shots), 100)
    assert result["status"] == CC.MISMATCH, result
    assert result["differences"]


def _to_iqm_payload(circuits, shots, name_of):
    """A minimal IQM-style payload from ISA circuits (as iqm-client would serialize r/cz/measure)."""
    out = []
    for c in circuits:
        phys = CC.physical_index_map(c)
        ins_list = []
        for ins in c.data:
            op, locus = ins.operation, [name_of(phys[q]) for q in ins.qubits]
            if op.name == "measure":
                ins_list.append({"name": "measure", "locus": locus, "args": {"key": CC.clbit_key(c, ins.clbits[0])}})
            elif op.name == "r":
                ins_list.append({"name": "prx", "locus": locus, "args": {"angle": float(op.params[0]), "phase": float(op.params[1])}})
            else:
                ins_list.append({"name": op.name, "locus": locus, "args": {}})
        out.append({"name": c.name, "instructions": ins_list, "metadata": {}})
    return {"circuits": out, "shots": shots}


@pytest.fixture(scope="module")
def iqm_isa():
    be = GenericBackendV2(num_qubits=5, basis_gates=["r", "cz"], coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]], seed=1)
    return [transpile(_circuit(), be, optimization_level=1, initial_layout=[1, 2, 3], seed_transpiler=1) for _ in range(2)]


def test_iqm_payload_match_and_mismatch(iqm_isa):
    name_of = lambda i: f"QB{i + 1}"
    payload = _to_iqm_payload(iqm_isa, 50, name_of)
    assert CC.check_iqm(iqm_isa, payload, name_of, 50)["status"] == CC.MATCH
    bad = copy.deepcopy(payload)
    bad["circuits"][1]["instructions"][0]["locus"] = ["QB5"]
    assert CC.check_iqm(iqm_isa, bad, name_of, 50)["status"] == CC.MISMATCH
    bad = copy.deepcopy(payload)
    bad["circuits"][0]["instructions"][-1]["args"]["key"] = "c_3_0_0"     # read out into another bit
    assert CC.check_iqm(iqm_isa, bad, name_of, 50)["status"] == CC.MISMATCH


@pytest.mark.parametrize("payload", [{}, {"circuits": [{"no_instructions": []}]}, None, "garbage"])
def test_malformed_server_data_is_check_failed_not_an_exception(iqm_isa, payload):
    result = CC.check_iqm(iqm_isa, payload, lambda i: f"QB{i + 1}", 50)
    assert result["status"] == CC.CHECK_FAILED and result["error"]


def test_ibm_malformed_pubs_is_check_failed(isa):
    assert CC.check_ibm(isa, [object()], 100)["status"] == CC.CHECK_FAILED


# ── Figure warning line ───────────────────────────────────────────────────────
def _summary(chunks=(), hcs=(), failed_chunks=()):
    return {"chunks": {"mismatched_ids": list(chunks), "check_failed_ids": list(failed_chunks)},
            "health_checks": {"mismatched_ids": list(hcs), "check_failed_ids": []}}


def test_verification_warning_names_at_most_one_of_each():
    assert CC.verification_warning(_summary()) is None
    assert CC.verification_warning(_summary(chunks=[12])) == "⚠ Circuit mismatch: chunk 12"
    assert CC.verification_warning(_summary(chunks=[12, 40], hcs=["15"])) == \
        "⚠ Circuit mismatch: chunk 12 and others; health check 15"
    assert CC.verification_warning(_summary(hcs=["3", "4"])) == "⚠ Circuit mismatch: health check 3 and others"
    assert CC.verification_warning(_summary(failed_chunks=[7])) == "⚠ Circuit check failed: chunk 7"


# ── Analysis summary on a synthetic run folder ────────────────────────────────
def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_summarize_checked_and_not_available_records(tmp_path, backend):
    run = tmp_path / "run"
    tdir = run / "transpiled_circuits"
    tdir.mkdir(parents=True)
    on = lambda layout: [transpile(_circuit(), backend, optimization_level=1, initial_layout=layout)]
    with open(tdir / "chunk_2_transpiled.qpy", "wb") as f: dump(on([1, 2, 3]), f)
    with open(tdir / "chunk_3_transpiled.qpy", "wb") as f: dump(on([2, 3, 4]), f)   # legacy chunk on other qubits
    base = {"status": "complete", "n_spam": 0, "n_circuits": 1}
    _write(run / "job_records" / "job_record_chunk_0.json", {**base, "circuit_check": {"status": "match"}})
    _write(run / "job_records" / "job_record_chunk_1.json", {**base, "circuit_check": {"status": "mismatch", "differences": [{"what": "x"}]}})
    na = {"circuit_check": {"status": "not_available"}}                              # EQE1/AQT, or a converted old record
    _write(run / "job_records" / "job_record_chunk_2.json", {**base, **na})          # on the run's qubits
    _write(run / "job_records" / "job_record_chunk_3.json", {**base, **na})          # on other qubits
    _write(tdir / "physical_qubits.json", {"physical_qubits": [1, 2, 3]})
    _write(run / "health_checks" / "health_check_1.json",
           {"number": 1, "schema": 2, "status": "complete", "jobs": [[0, 1]], "circuit_check": {"status": "match"}})
    _write(run / "health_checks" / "health_check_2.json",
           {"number": 2, "schema": 2, "status": "complete", "jobs": [[2, 3]], "layout": {"0": 0, "1": 1, "2": 2}, **na})

    s = CC.summarize_circuit_checks(run)
    assert s["chunks"]["status"] == CC.MISMATCH
    assert s["chunks"]["mismatched_ids"] == [1, 3]
    assert "qubit_consistency" in s["chunks"]["methods"] and "executed_vs_submitted" in s["chunks"]["methods"]
    assert s["health_checks"]["mismatched_ids"] == ["2"]
    assert CC.verification_warning(s) == "⚠ Circuit mismatch: chunk 1 and others; health check 2"
    # subset: chunks [0, 1) and the health checks that accompany them
    sub = CC.summarize_circuit_checks(run, n_samples=1)
    assert sub["chunks"]["status"] == CC.MATCH and sub["health_checks"]["status"] == CC.MATCH
    assert (run / "processed_results" / CC.LEGACY_CACHE_FILENAME).exists()


# ── Time zones and per-job timing ─────────────────────────────────────────────
def test_berlin_to_utc_summer_winter_and_repeated_hour():
    tz = JT.MQSS_TIMEZONE
    assert JT.local_naive_to_utc("2026-08-27 15:17:12.5", tz) == datetime(2026, 8, 27, 13, 17, 12, 500000, tzinfo=timezone.utc)
    assert JT.local_naive_to_utc("2026-12-01 10:00:00", tz) == datetime(2026, 12, 1, 9, 0, tzinfo=timezone.utc)
    # 2026-10-25 02:30 happens twice in Berlin (00:30 and 01:30 UTC)
    first = datetime(2026, 10, 25, 0, 30, tzinfo=timezone.utc)
    assert JT.is_ambiguous_local("2026-10-25 02:30:00", tz)
    assert JT.local_naive_to_utc("2026-10-25 02:30:00", tz) == first
    assert JT.local_naive_to_utc("2026-10-25 02:30:00", tz, not_before=datetime(2026, 10, 25, 1, 0, tzinfo=timezone.utc)) \
        == datetime(2026, 10, 25, 1, 30, tzinfo=timezone.utc)


class _MQSSResult:
    timestamp_submitted = "2026-10-25 02:50:00.000000"   # before the clock goes back (00:50 UTC)
    timestamp_scheduled = "2026-10-25 02:10:00.000000"   # after it (01:10 UTC) — naively 40 min *earlier*
    timestamp_completed = "2026-10-25 02:10:05.000000"


def test_eqe_timing_across_the_dst_change():
    t = JT.eqe_collect_timing(_MQSSResult(), "2026-10-25T00:50:00Z", n_circuits=68)
    assert t["queue_s"] == pytest.approx(20 * 60)
    assert t["qpu_s"] == pytest.approx(68 * JT.EQE_QPU_S_PER_CIRCUIT) and t["qpu_s_source"] == "model_eqe_per_circuit"
    assert t["qpu_s_observed_upper_bound"] == pytest.approx(5.0)
    assert t["ambiguous_dst"] and t["raw_server_timestamps"]["timestamp_scheduled"] == _MQSSResult.timestamp_scheduled


def test_aqt_and_legacy_timing():
    # QPU time per circuit: 3 s up to 10 shots, 15 s at 200 (the maximum), linear in between
    f = JT.aqt_qpu_s_per_circuit
    assert (f(1), f(10), f(200), f(500)) == (3.0, 3.0, 15.0, 15.0)
    assert f(105) == pytest.approx(9.0) and f(100) == pytest.approx(3 + 12 * 90 / 190)
    t = JT.aqt_timing([200.0, 10.0], [50, 18], 10, qpu_s_physics_model=1.5)
    assert t["qpu_s"] == pytest.approx(68 * 3.0)
    assert t["queue_s"] == pytest.approx(200 - 150) and t["queue_clamped"]   # 10 s < 18 × 3 s → clamped to 0
    assert t["submit_s"] == 0.0 and t["qpu_s_physics_model"] == 1.5
    t = JT.aqt_timing([100.0], [1], 100)                                      # a 100-shot health check
    assert t["qpu_s"] == pytest.approx(f(100)) and t["queue_s"] == pytest.approx(100 - f(100))
    old_ibm = {"created_at": "2026-09-21T22:18:26Z", "scheduled_at": "2026-09-21T22:18:30Z", "finished_at": "2026-09-21T22:18:40Z"}
    t = JT.legacy_record_timing(old_ibm, "IBM_Cloud", 68)
    assert (t["qpu_s"], t["qpu_s_source"], t["queue_s"]) == (10.0, "legacy_interval_ibm", 4.0)
    old_eqe = {"submitted_at": "2026-08-27T13:13:02Z", "scheduled_at": "2026-08-27 15:13:12.0", "finished_at": "x"}
    t = JT.legacy_record_timing(old_eqe, "MQSS_LRZ", 68)
    assert t["queue_s"] == pytest.approx(10.0) and t["qpu_s_source"] == "model_eqe_per_circuit"


def test_queue_total_is_the_longest_wait_except_on_aqt(tmp_path):
    # Three EQE1 jobs queued together: waits 100, 200, 300 s overlap, so the queue time is the
    # longest single wait (300 s), not their sum (600 s). AQT runs jobs one after another: the sum.
    base = {"status": "complete", "service_name": "MQSS_LRZ", "n_spam": 0, "n_circuits": 1}
    for nc, (sub, sched) in enumerate([("00:00", "01:40"), ("00:00", "03:20"), ("00:00", "05:00")]):
        _write(tmp_path / "job_records" / f"job_record_chunk_{nc}.json",
               {**base, "job_id": str(nc), "timing": {"queue_s": 100.0 * (nc + 1), "queue_s_source": "x", "qpu_s": 0.9,
                "queue_start_at": f"2026-09-01T10:{sub}Z", "queue_end_at": f"2026-09-01T10:{sched}Z"}})
    _write(tmp_path / "health_checks" / "health_check_1.json",
           {"number": 1, "schema": 2, "status": "complete", "service_name": "MQSS_LRZ", "jobs": [[0]],
            "timing": {"queue_s": 450.0, "queue_s_source": "x", "qpu_s": 0.9}})
    q = RT.qpu_queue_breakdown(tmp_path)
    assert q["queue_total_s"] == 450.0                    # the health check waited longest
    assert q["chunks"]["queue"]["total_s"] == 300.0 and q["chunks"]["queue"]["sum_of_jobs_s"] == 600.0
    for nc in range(3):                                   # AQT: sequential chunks — the sum
        p = tmp_path / "job_records" / f"job_record_chunk_{nc}.json"
        _write(p, {**json.loads(p.read_text(encoding="utf-8")), "service_name": "PCSS"})
    hc = tmp_path / "health_checks" / "health_check_1.json"
    _write(hc, {**json.loads(hc.read_text(encoding="utf-8")), "service_name": "PCSS"})
    assert RT.qpu_queue_breakdown(tmp_path)["queue_total_s"] == 1050.0


# ── Old records ───────────────────────────────────────────────────────────────
def test_old_aqt_records_are_refused_until_converted(tmp_path):
    run = tmp_path / "bacq_real_aqt_piast-q_run"
    old_chunk = {"backend_name": "aqt_pcss_piast-q", "chunk_id": 0, "job_id": "N/A", "n_spam": 8, "n_circuits": 60,
                 "nshots": 10, "wall_seconds": 500.0, "qpu_seconds_estimate": 1.5, "status": "complete"}
    _write(run / "job_records" / "job_record_chunk_0.json", old_chunk)
    with pytest.raises(ValueError, match="old format"):
        CC.summarize_circuit_checks(run)
    with pytest.raises(ValueError, match="no timing block"):
        RT.qpu_queue_breakdown(run)


def test_aqt_transpile_rate_from_the_runs_that_measured_it(tmp_path):
    new = {"service_name": "PCSS", "status": "complete", "n_circuits": 60, "n_spam": 8,
           "timing": {"transpile_s": 6.0, "transpile_reused": False}}
    old = {"service_name": "PCSS", "status": "complete", "n_circuits": 60, "n_spam": 8,
           "timing": {"transpile_s": None, "converted_from_old_record": True}}
    assert RT.aqt_transpile_rate(datafiles_dir=tmp_path) == (0.0, [])                    # none: 0
    _write(tmp_path / "old_run" / "job_records" / "job_record_chunk_0.json", old)
    _write(tmp_path / "ibm_run" / "job_records" / "job_record_chunk_0.json", {**new, "service_name": "IBM_Cloud"})
    assert RT.aqt_transpile_rate(datafiles_dir=tmp_path) == (0.0, [])                    # converted / other services don't count
    _write(tmp_path / "new_run" / "job_records" / "job_record_chunk_0.json", new)
    _write(tmp_path / "new_run" / "job_records" / "job_record_chunk_1.json", {**new, "timing": {"transpile_s": 3.0}})
    assert RT.aqt_transpile_rate(datafiles_dir=tmp_path) == (pytest.approx(9.0 / 120), ["new_run"])
    assert RT.aqt_transpile_rate(exclude=tmp_path / "new_run", datafiles_dir=tmp_path) == (0.0, [])


def test_health_check_transpile_time_is_counted_once_per_run(tmp_path):
    """The CPU time's health check transpile time is the sum over the saved variants of
    health_check_transpiled.json, whole-run also for a subset; a legacy run without the file (or with an
    unreadable one) sums the health check records in scope."""
    from source.core.records.health_check_records import create_health_check
    base = {"status": "complete", "timing": {"transpile_s": 1.0, "transpile_reused": False, "submit_s": 0.5}}
    create_health_check(tmp_path, {**base, "chunks": [0]})
    create_health_check(tmp_path, {**base, "chunks": [3], "timing": {"transpile_s": None, "transpile_reused": True,
                                                                      "submit_s": 0.5}})
    hc = lambda nchunks: RT.cpu_time_breakdown(tmp_path, nchunks, False, [0.2], 0.2, total_chunks=4)
    assert hc(4)["health_check_transpile_s"] == 1.0 and hc(2)["health_check_transpile_s"] == 1.0   # records
    assert hc(4)["health_check_submit_s"] == 1.0 and hc(2)["health_check_submit_s"] == 0.5
    meta = tmp_path / "transpiled_circuits" / "health_check_transpiled.json"
    meta.parent.mkdir()
    meta.write_text("{not json", encoding="utf-8")
    assert hc(4)["health_check_transpile_s"] == 1.0                                              # unreadable: records
    _write(meta, {"variants": [{"transpile_s": 1.5}, {"transpile_s": 0.5}]})
    assert hc(4)["health_check_transpile_s"] == 2.0 and hc(2)["health_check_transpile_s"] == 2.0   # whole run, subset too


def test_old_eqe_run_is_read_as_a_legacy_run(tmp_path):
    run = tmp_path / "run"
    old_eqe = {"backend_name": "<mqss.qiskit_adapter.backend.MQSSQiskitBackend object>", "chunk_id": 0, "job_id": "1",
               "submitted_at": "2026-08-27T13:13:02Z", "scheduled_at": "2026-08-27 15:13:12.000000",
               "finished_at": "2026-08-27 15:13:20.000000", "n_spam": 8, "n_circuits": 60, "status": "complete"}
    _write(run / "job_records" / "job_record_chunk_0.json", old_eqe)
    q = RT.qpu_queue_breakdown(run)
    assert q["legacy_run"] and q["n_legacy_records"] == 1
    assert q["qpu_total_s"] == pytest.approx(68 * JT.EQE_QPU_S_PER_CIRCUIT) and q["queue_total_s"] == pytest.approx(10.0)
    s = CC.summarize_circuit_checks(run)
    assert s["legacy_run"] and s["chunks"]["status"] == CC.NOT_AVAILABLE              # no transpiled circuits here


def test_a_timing_block_from_the_old_fields_keeps_the_run_legacy(tmp_path):
    # a record whose QPU time could only be taken from its old execution window (source legacy_interval_*)
    rec = {"chunk_id": 0, "status": "complete", "service_name": "IQM_Resonance", "job_id": "j", "n_spam": 8, "n_circuits": 60,
           "timing": {"qpu_s": 0.8, "qpu_s_source": "legacy_interval_iqm", "queue_s": 5.0, "queue_s_source": "x"},
           "circuit_check": CC.not_available("the provider no longer holds the job")}
    _write(tmp_path / "job_records" / "job_record_chunk_0.json", rec)
    q = RT.qpu_queue_breakdown(tmp_path)
    assert q["legacy_run"] and q["qpu_total_s"] == pytest.approx(0.8)
    assert q["qpu_exec_total_s"] == pytest.approx(0.8)                                 # the interval is execution time


# ── Timestamp logs ────────────────────────────────────────────────────────────
def _log(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["label", "timestamp"])
        w.writerows(rows)


def _job_seconds(folder, nchunks):
    """(total seconds, chunks covered) of the single-chunk qpu_chunk intervals among chunks [0, nchunks), read the way the analysis reads them (UTC log, else the local one)."""
    (singles, _), _source = TS._from_timestamp_log(
        folder / "timestamps_local_time.csv",
        lambda rows: ((r := RT._job_intervals_from_rows(rows, nchunks)), set(r[0]) >= set(range(nchunks))))
    return sum(singles.values()), len(singles)


def test_durations_come_from_the_utc_log_across_a_dst_change(tmp_path):
    # Berlin, night of 2026-10-25: 02:55 CEST -> (clock back) -> 02:05 CET is 10 minutes, not -50.
    _log(tmp_path / "timestamps_local_time.csv", [
        ["qpu_chunk0_start", "2026-10-25 02:55:00.000000"], ["chunk0_submit_start", "2026-10-25 02:55:00.000000"],
        ["chunk0_submit_end", "2026-10-25 02:05:00.000000"], ["qpu_chunk0_finished", "2026-10-25 02:05:00.000000"]])
    _log(tmp_path / "timestamps_utc.csv", [
        ["qpu_chunk0_start", "2026-10-25T00:55:00.000000Z"], ["chunk0_submit_start", "2026-10-25T00:55:00.000000Z"],
        ["chunk0_submit_end", "2026-10-25T01:05:00.000000Z"], ["qpu_chunk0_finished", "2026-10-25T01:05:00.000000Z"]])
    assert _job_seconds(tmp_path, 1) == (600.0, 1)
    labels, source = TS._from_timestamp_log(tmp_path / "timestamps_local_time.csv",
                                            lambda rows: ((r := RT._cpu_label_times_from_rows(rows, 1)), bool(r["chunks_with_labels"])))
    assert source == "utc" and labels["submit_s"] == 600.0


def test_partial_utc_log_falls_back_to_the_local_one(tmp_path):
    # Two chunks; the UTC log only started with chunk 1 (logging added mid-run).
    _log(tmp_path / "timestamps_local_time.csv", [
        ["qpu_chunk0_start", "2026-09-01 10:00:00.000000"], ["qpu_chunk0_finished", "2026-09-01 10:00:07.000000"],
        ["qpu_chunk1_start", "2026-09-01 10:01:00.000000"], ["qpu_chunk1_finished", "2026-09-01 10:01:05.000000"]])
    _log(tmp_path / "timestamps_utc.csv", [
        ["qpu_chunk1_start", "2026-09-01T08:01:00.000000Z"], ["qpu_chunk1_finished", "2026-09-01T08:01:05.000000Z"]])
    assert _job_seconds(tmp_path, 2) == (12.0, 2)
    assert _job_seconds(tmp_path, 1) == (7.0, 1)


def test_timestamp_logs_are_append_only_and_an_interrupted_attempt_is_skipped(tmp_path):
    log = tmp_path / "timestamps_local_time.csv"
    for label in ["Starting chunk submission.", "qpu_chunk0_start", "chunk0_transpile_start", "chunk0_transpile_end",
                  "chunk0_submit_start", "chunk0_submit_end", "qpu_chunk0_finished",
                  "qpu_chunk1_start", "chunk1_transpile_start",                 # killed here
                  "Starting chunk submission.", "qpu_chunk1_start", "qpu_chunk1_finished"]:
        TS.timestamp(label, log)
    utc = tmp_path / "timestamps_utc.csv"
    assert [r[0] for r in TS._read_timestamp_rows(log)] == [r[0] for r in TS._read_timestamp_rows(utc)]
    assert all(r[1].endswith("Z") for r in TS._read_timestamp_rows(utc))
    # the killed attempt's start is not paired with the retry's finish
    rows = [(lab, datetime(2026, 9, 1, 10, i, tzinfo=timezone.utc)) for i, lab in enumerate(
        ["qpu_chunk1_start", "chunk1_transpile_start", "Starting chunk submission.", "qpu_chunk1_start", "qpu_chunk1_finished"])]
    singles, _ = RT._job_intervals_from_rows(rows, 2)
    assert singles == {1: 60.0}
    # a start directly followed by another chunk's start is dropped too
    rows = [(lab, datetime(2026, 9, 1, 10, i, tzinfo=timezone.utc)) for i, lab in enumerate(
        ["qpu_chunk0_start", "qpu_chunk1_start", "qpu_chunk1_finished", "qpu_chunk0_finished"])]
    assert RT._job_intervals_from_rows(rows, 2)[0] == {1: 60.0}


def test_build_log_sync_appends_and_skips_the_other_formats_conversion(tmp_path):
    _log(tmp_path / "timestamps_utc.csv", [
        ["Starting compilation for Nt=1.", "2026-09-01T08:00:00.000000Z"],
        ["Finished myQLM interleaving 8 chunks.", "2026-09-01T08:01:40.000000Z"],
        ["Starting Qiskit conversion of chunks.", "2026-09-01T08:01:40.000000Z"],
        ["Finished Qiskit conversion 8 chunks.", "2026-09-01T08:02:00.000000Z"]])
    v13 = tmp_path / "timestamps_qiskit_v13_qpy_utc.csv"
    _log(v13, [["something earlier", "2026-09-05T08:00:00.000000Z"]])
    before = v13.read_text(encoding="utf-8")
    TS.sync_circuit_timestamp_logs(str(tmp_path), qiskit_v13_qpy=True)
    after = v13.read_text(encoding="utf-8")
    assert after.startswith(before)                                     # appended, nothing rewritten
    labels = [r[0] for r in TS._read_timestamp_rows(v13)]
    assert "Starting compilation for Nt=1." in labels and not any("Qiskit conversion" in l for l in labels)
    # the QPY-v13 format is converted days later: only its own conversion is counted, not the gap
    with open(v13, "a", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerows([["Starting Qiskit conversion of chunks.", "2026-09-05T09:00:00.000000Z"],
                                                  ["Finished Qiskit conversion 8 chunks.", "2026-09-05T09:00:30.000000Z"]])
    b = TS.circuit_build_breakdown(lnn=False, qiskit_v13_qpy=True, circuit_folder=str(tmp_path))
    assert (b["myqlm_build_s"], b["qiskit_conversion_s"], b["build_total_s"]) == (100.0, 30.0, 130.0)


def test_build_log_split_into_myqlm_and_conversion(tmp_path):
    _log(tmp_path / "timestamps_utc.csv", [
        ["Starting compilation for Nt=1.", "2026-09-01T08:00:00.000000Z"],
        ["Finished myQLM interleaving 8 chunks.", "2026-09-01T08:01:40.000000Z"],
        ["Starting Qiskit conversion of chunks.", "2026-09-01T08:01:40.000000Z"],
        ["Finished Qiskit conversion 8 chunks.", "2026-09-01T08:02:00.000000Z"]])
    b = TS.circuit_build_breakdown(lnn=False, circuit_folder=str(tmp_path))
    assert (b["myqlm_build_s"], b["qiskit_conversion_s"], b["build_total_s"]) == (100.0, 20.0, 120.0)
    assert b["timestamp_source"] == "utc"


def test_layout_selection_counts_native_single_qubit_gates_on_the_transpiled_circuit():
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2
    from source.core.qpu.layout_selection import probe_placement, transpiled_1q_counts, select_physical_qubits
    backend = GenericBackendV2(num_qubits=5, basis_gates=["cz", "rz", "sx", "x"],
                               coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]], seed=1)
    qc = QuantumCircuit(3)
    qc.h(0); qc.rx(0.3, 1); qc.ry(0.2, 2); qc.rz(0.1, 0)
    qc.cx(0, 1); qc.cx(1, 2)
    qc.measure_all()
    probe = probe_placement(qc, backend.target)
    assert probe == [0, 1, 2]                                   # first SWAP-free path, unscored
    counts = transpiled_1q_counts(qc, backend, probe)
    assert counts == transpiled_1q_counts(qc, backend, [4, 3, 2])   # the same on every placement
    assert sum(counts.values()) > 0
    rec = select_physical_qubits(qc, backend.target, counts)
    assert rec["n_1q_source"] == "transpiled" and rec["n_1q"] == [counts.get(v, 0) for v in range(3)]


# ── QPU execution time (QPU_execution_time_s) of every backend ─────────────────
def _record(tmp_path, nc, job_id, timing, service):
    jr = tmp_path / "job_records"
    jr.mkdir(exist_ok=True)
    (jr / f"job_record_chunk_{nc}.json").write_text(json.dumps(
        {"chunk_id": nc, "job_id": job_id, "status": "complete", "service_name": service,
         "n_spam": 8, "n_circuits": 12, "nshots": 600, "timing": timing}), encoding="utf-8")


def test_qpu_execution_time_and_workload(tmp_path):
    for nc in range(3):
        _record(tmp_path, nc, "j1" if nc < 2 else "j2",
                {"qpu_s": 3.0, "qpu_s_source": "iqm_equivalent_billed", "qpu_exec_s": 2.0,
                 "qpu_exec_s_source": "remote_hw_runtime"}, "QAAS_IT4I")
    qq = RT.qpu_queue_breakdown(tmp_path)
    assert qq["qpu_exec_total_s"] == 4.0 and qq["qpu_total_s"] == 6.0     # two jobs
    w = RT.qpu_workload(tmp_path)
    assert sorted(w["chunks_per_job"]) == [1, 2] and w["circuits_per_chunk"] == 20

    _record(tmp_path, 3, "j3", {"qpu_s": 5.0, "qpu_s_source": "model_eqe_per_circuit"}, "MQSS_LRZ")
    assert RT.qpu_queue_breakdown(tmp_path)["qpu_exec_total_s"] is None   # one job without it → "N/A"


def test_resonance_and_ibm_execution_time():
    assert JT.qpu_execution_s({"iqm_metrics": {"execution_seconds": 25.2}}, "IQM_Resonance") == (25.2, "iqm_execution_seconds")
    assert JT.qpu_execution_s({"execution_start": "2026-09-28T10:00:00Z", "execution_end": "2026-09-28T10:00:04Z"},
                              "IQM_Resonance") == (4.0, "execution_interval")
    assert JT.qpu_execution_s({"circuits_execution_seconds": 7.5}, "IBM_Cloud") == (7.5, "ibm_circuits_execution")
    assert JT.qpu_execution_s({"qpu_s": 3.0}, "PCSS") == (None, "unavailable")


def test_utc_helpers_are_shared():
    from source.core.records import health_check_records as HCR
    aware = datetime(2026, 8, 27, 13, 17, 12, 123456, tzinfo=timezone.utc)
    assert JT.parse_utc("2026-08-27T13:17:12.123456789Z") == aware
    assert JT.parse_utc("2026-08-27T13:17:12.123456789z") == aware
    assert JT.parse_any_utc is JT.parse_utc and HCR.parse_utc is JT.parse_utc
    assert HCR.now_utc_iso is JT.now_utc_iso and HCR.to_utc_iso is JT.iso_utc
    assert JT.iso_utc("2026-08-27T13:17:12Z") == "2026-08-27T13:17:12.000000Z"
    assert JT.iso_utc(aware) == "2026-08-27T13:17:12.123456Z"
    assert JT.iso_utc("N/A") is None and JT.iso_utc(None) is None
    assert JT.parse_utc("2026-08-27 15:17:12", naive_tz=JT.MQSS_TIMEZONE) == datetime(2026, 8, 27, 13, 17, 12, tzinfo=timezone.utc)
    assert JT.parse_utc("2026-08-27 15:17:12") == datetime(2026, 8, 27, 15, 17, 12, tzinfo=timezone.utc)
