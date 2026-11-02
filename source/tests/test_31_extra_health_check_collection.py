"""
test_31_extra_health_check_collection.py
========================================
Unit tests (no simulator, no QPU, no network, no provider SDK) for the collection side of extra_health_checks (the
health-check circuits packed into the submitted jobs):

  - bacq_ps_qbcs._save_extra_health_checks: the file names (extra_chunk_N_*, extra_job_N_*), the counts each offset picks,
    the bit reversal, no write without hc_pre_idx, a degraded check warns on stdout and never exits;
  - the collectors get_results_by_chunk_ibm (SamplerV2 results with a `c` or a `meas` register), get_results_by_chunk_iqm
    (EQE1, result.counts), get_results_by_chunk_iqm_resonance (IQM Resonance and VLQ): the health-check counts are
    saved as extra health checks and are absent from the chunk's results, SPAM and fidelity files. The provider objects are
    small stand-ins shaped as the provider returned them: IBM pub results with `data.meas` / `data.c` and `get_counts()`;
    EQE1 `Result` with `counts` and the naive `timestamp_*` strings; IQM Resonance `get_counts()` list and GET
    jobs/{id}/payload with `circuits[].instructions[]` (name, locus, args.key); VLQ `get_transpiled_circuits()`;
  - pack_job_circuits -> the job record's offsets -> the collector, per-chunk and per-job with several chunks in one job:
    every circuit comes back from the position it was packed at (each position of the stand-in job has its own counts);
  - bacq_ps_qbcs._check_extra_hc: one comparison per slot (pre, post), so an altered post marks only the post record;
    match, mismatch, and check_failed (no raise) for a missing QPY file or a comparison that raises;
  - the circuit_check of the extra records: compared per slot on IQM Resonance and VLQ (a mismatch is printed at collection),
    not_available on IBM without a service (simulated), EQE1 and the local simulator (AQT: aqt_venv_test_12_shot_split.py);
  - circuit_check.summarize_circuit_checks: an extra record's mismatch shows in the "health_checks" summary and the banner
    text under "extra chunk N post", a not_available one is checked for its recorded qubits once, and an extra record
    without circuit_check (an older run) is left out;
  - a repeated collection of a per-job job leaves the records and results as they were;
  - plot_health_checks.py on a run with a degraded, a half-missing (pre without post) and a failed-job record: the json
    summary's incomplete_extra_health_checks names the missing post, and is empty for a complete run; a filtered analysis
    (QPU_HC_THRESHOLD) records filter_threshold in its summary, an unfiltered one does not;
  - the filter: chunks_passing_extra_health_checks (per-chunk and per-job pairs, a missing post, P011 equal to the threshold),
    HealthCheckScope / figure_filter_label (file names and figure labels), load_merged_spam_calibration_counts with a chunk set.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import json
import shutil
import subprocess
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit
from qiskit.qpy import dump

from source.core import qbcs as Q
from source.core.records import circuit_check as CC
from source.core.records import health_check_records as HR
from source.core.records.run_info import write_run_info
from source.core.records.run_names import QAAS_IT4I, plots_dir as run_plots_dir
from source.core.runner_support.venvs import venv_python
from source.tests.testing_functions import run_name_for_test

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HC = ("hc",)                     # the health-check circuit, as a label (pack_job_circuits does list work only)
N_SPAM, N_CIRC = 4, 2            # SPAM circuits and circuits of every chunk in the stand-in jobs
ROLES = ["raw", "raw_nec"]
SUBMITTED = "2026-10-01T10:00:00Z"
T_START, T_END = "2026-10-01T10:01:00.000000Z", "2026-10-01T10:02:00.000000Z"


# ── Stand-ins ────────────────────────────────────────────────────────────────

def _qbcs(mode):
    """A bacq_ps_qbcs with only what collection reads (no circuits are built)."""
    obj = Q.bacq_ps_qbcs.__new__(Q.bacq_ps_qbcs)
    obj.extra_health_checks = mode
    obj._extra_hc_file = "hc.qpy"
    return obj


def _counts_at(i, is_hc):
    """The counts of the job's circuit at position i: unique per position. A health check's P(011) is (90 - i) / 100."""
    return {"110": 90 - i, "000": 10 + i} if is_hc else {"100": 1000 + i}


def _job_counts(circuits):
    return [_counts_at(i, c == HC) for i, c in enumerate(circuits)]


def _hc_circuit(extra_x=False):
    qc = QuantumCircuit(3, 3)
    if extra_x:
        qc.x(0)
    qc.measure(range(3), range(3))
    return qc


def _iqm_circuit(bad=False):
    """The health-check circuit as GET jobs/{id}/payload lists it (circuits[].instructions[]: name, locus, args.key)."""
    return {"name": "circuit-0", "metadata": {},
            "instructions": [{"name": "measure", "locus": [f"QB{i + 1}"], "implementation": None,
                              "args": {"key": "c_3_0_9" if bad and i == 0 else f"c_3_0_{i}"}} for i in range(3)]}


def _packed(mode, chunk_nums):
    chunks = [(nc, [("circ", nc, k) for k in range(N_CIRC)]) for nc in chunk_nums]
    return Q.pack_job_circuits(chunks, [("spam", si) for si in range(N_SPAM)], HC, mode)


def _make_run(tmp_path, mode, chunk_nums, service_name="IQM_Resonance"):
    """A run folder holding the job records of one packed job (all chunk_nums in it); returns (qbcs, circuits, offsets)."""
    obj = _qbcs(mode)
    circuits, offsets = _packed(mode, chunk_nums)
    (tmp_path / "job_records").mkdir()
    tdir = tmp_path / "transpiled_circuits"
    tdir.mkdir()
    with open(tdir / "hc.qpy", "wb") as f:
        dump([_hc_circuit()], f)
    for nc in chunk_nums:
        rec = {"chunk_id": nc, "job_id": "job-A", "status": "submitted", "service_name": service_name, "backend_name": "backend-x",
               "nshots": 100, "submitted_at": SUBMITTED, "n_spam": N_SPAM, "n_circuits": N_CIRC, "nsites": 2,
               "trotter_steps_list": [1], "roles": ROLES, "layout": {"0": 5, "1": 6, "2": 7}}
        rec.update(obj._hc_record_fields(offsets[nc], len(circuits), chunk_nums))
        (tmp_path / "job_records" / f"job_record_chunk_{nc}.json").write_text(json.dumps(rec), encoding="utf-8")
    return obj, circuits, offsets


def _record(tmp_path, nc):
    return json.loads((tmp_path / "job_records" / f"job_record_chunk_{nc}.json").read_text(encoding="utf-8"))


def _read(tmp_path, *parts):
    return json.loads(tmp_path.joinpath(*parts).read_text(encoding="utf-8"))


class _Backend:
    """IQM Resonance / VLQ backend stand-in: retrieve_job() gives a finished job with a get_counts() list."""
    index_to_qubit_name = staticmethod(lambda i: f"QB{i + 1}")

    def __init__(self, counts, transpiled=None):
        self.counts, self.transpiled, self.retrieved = counts, transpiled, []

    def retrieve_job(self, job_id):
        self.retrieved.append(job_id)
        counts, transpiled = self.counts, self.transpiled
        return SimpleNamespace(wait_for_final_state=lambda: None, done=lambda: True,
                               result=lambda: SimpleNamespace(get_counts=lambda: counts, success=True),
                               get_transpiled_circuits=lambda: transpiled)


def _patch_resonance(monkeypatch, obj, payload=None):
    """The timing and circuit-check calls of the IQM Resonance collection (network in production) replaced; the payload, when
    given, is what the job-payload cache holds for the health-check comparison."""
    monkeypatch.setattr(Q, "iqm_resonance_collect_timing",
                        lambda backend, job_id: {"execution_start": T_START, "execution_end": T_END, "completed_at": T_END})

    def check(backend, record, run_path, nc):
        if payload is not None:
            obj.__dict__.setdefault("_iqm_payload_cache", {})[record["job_id"]] = payload
        return CC.not_available("stand-in")
    obj._check_executed_iqm_resonance = check


def _patch_vlq(monkeypatch, obj):
    monkeypatch.setattr(Q.VLQ, "retrieve_job", lambda backend, job_id: backend.retrieve_job(job_id))
    monkeypatch.setattr(Q.VLQ, "wait_for_result", lambda job, job_id="", timeout_s=None: job.result())
    monkeypatch.setattr(Q.VLQ, "vlq_collect_timing",
                        lambda job, upload_end_at: {"execution_start": T_START, "execution_end": T_END, "completed_at": T_END})
    monkeypatch.setattr(Q.VLQ, "transpiled_circuits", lambda job, job_id="": job.get_transpiled_circuits())
    obj._check_executed_iqm_vlq = lambda job, record, run_path, nc: CC.not_available("stand-in")


def _hc_positions(circuits, mode, nc):
    """(pre, post) positions of chunk nc's health checks, read off the packed list."""
    if mode == "per-job":
        return 0, len(circuits) - 1
    first = circuits.index(("circ", nc, 0))
    return first - N_SPAM - 1, circuits.index(("circ", nc, N_CIRC - 1)) + 1


def _assert_collected(run_path, circuits, mode, chunk_nums):
    """Every chunk's results come from the positions it was packed at; the health checks are saved apart, from their own."""
    counts, fid, spam = (_read(run_path, "raw_results", f) for f in
                         ("results_counts.json", "results_fidelity.json", "spam_calibration_counts_per_chunk.json"))
    for nc in chunk_nums:
        at = lambda k, nc=nc: {"001": 1000 + circuits.index(("circ", nc, k))}   # the packed "100", reversed by the collector
        assert counts[f"chunk_{nc}"] == {"job_Nt_01_raw_ems0": {"Nt_01_raw_circuit_0": at(0)}}
        assert fid[f"chunk_{nc}"] == {"job_Nt_01_raw_nec_ems1": {"Nt_01_raw_nec_circuit_0": at(1)}}
        base = circuits.index(("circ", nc, 0)) - N_SPAM
        assert spam[f"chunk_{nc}"] == {format(si, "02b"): {"001": 1000 + base + si} for si in range(N_SPAM)}
    for blob in (counts, fid, spam):
        assert '"011"' not in json.dumps(blob) and '"000"' not in json.dumps(blob)   # no health-check counts among the chunk results
    first = min(chunk_nums)
    for nc in chunk_nums:
        pre, post = _hc_positions(circuits, mode, nc)
        assert circuits[pre] == HC and circuits[post] == HC
        for position, idx in (("pre", pre), ("post", post)):
            name = f"extra_chunk_{nc}_{position}.json" if mode == "per-chunk" else f"extra_job_{first}_{position}.json"
            rec = _read(run_path, "health_checks", name)
            assert rec["circuit_index"] == idx and rec["position"] == position and rec["mode"] == mode
            assert rec["n_circuits_in_job"] == len(circuits)
            assert rec["chunks"] == ([nc] if mode == "per-chunk" else list(chunk_nums))
            assert rec["counts"] == {"011": 90 - idx, "000": 10 + idx}   # reversed from the job's "110"
            assert rec["P011"] == pytest.approx((90 - idx) / 100)
            assert rec["layout"] == {"0": 5, "1": 6, "2": 7} and rec["job_id"] == "job-A"
    n_files = len(list((run_path / "health_checks").glob("extra_*.json")))
    assert n_files == (2 * len(chunk_nums) if mode == "per-chunk" else 2)


# ── 1. _save_extra_health_checks ─────────────────────────────────────────────

def _save_record(mode, **over):
    rec = {"extra_health_checks": mode, "hc_pre_idx": 3, "hc_post_idx": 9, "n_circuits_in_job": 12, "nshots": 100,
           "backend_name": "backend-x", "service_name": "MQSS_LRZ", "job_id": "job-A", "submitted_at": SUBMITTED,
           "layout": {"0": 5, "1": 6, "2": 7}}
    rec.update(over)
    return rec


_TABLE = {3: {"110": 90, "001": 10}, 9: {"110": 70, "000": 30}}   # the raw counts of two positions of the job


def test_save_extra_health_checks_per_chunk_names_offsets_and_bit_reversal(tmp_path):
    _qbcs("per-chunk")._save_extra_health_checks(tmp_path, _save_record("per-chunk"), 7, lambda i: _TABLE[i], T_START, T_END)
    assert sorted(p.name for p in (tmp_path / "health_checks").iterdir()) == ["extra_chunk_7_post.json", "extra_chunk_7_pre.json"]
    pre, post = (_read(tmp_path, "health_checks", f"extra_chunk_7_{p}.json") for p in ("pre", "post"))
    assert (pre["circuit_index"], post["circuit_index"]) == (3, 9)
    assert pre["counts"] == {"011": 90, "100": 10}      # "110" and "001" reversed
    assert post["counts"] == {"011": 70, "000": 30}
    assert pre["P011"] == pytest.approx(0.9) and post["P011"] == pytest.approx(0.7)
    assert pre["chunks"] == [7] and pre["scheduled_at"] == T_START and pre["finished_at"] == T_END


def test_save_extra_health_checks_per_job_is_named_after_the_first_chunk_and_written_once(tmp_path, capsys):
    obj, rec = _qbcs("per-job"), _save_record("per-job", job_chunks=[4, 5])
    obj._save_extra_health_checks(tmp_path, rec, 5, lambda i: _TABLE[i], T_START, T_END)   # the job's second chunk comes first
    assert sorted(p.name for p in (tmp_path / "health_checks").iterdir()) == ["extra_job_4_post.json", "extra_job_4_pre.json"]
    assert _read(tmp_path, "health_checks", "extra_job_4_pre.json")["chunks"] == [4, 5]
    assert "Extra health checks (per-job)" in capsys.readouterr().out
    before = {p.name: p.read_bytes() for p in (tmp_path / "health_checks").iterdir()}
    obj._save_extra_health_checks(tmp_path, rec, 4, lambda i: _TABLE[i], T_START, T_END)   # the other chunk writes nothing
    assert {p.name: p.read_bytes() for p in (tmp_path / "health_checks").iterdir()} == before
    assert capsys.readouterr().out == ""


def test_save_extra_health_checks_without_hc_pre_idx_writes_nothing(tmp_path):
    rec = _save_record("per-chunk")
    del rec["hc_pre_idx"], rec["hc_post_idx"]
    _qbcs("per-chunk")._save_extra_health_checks(tmp_path, rec, 0, lambda i: pytest.fail("counts_at must not be called"), "N/A", "N/A")
    _qbcs(None)._save_extra_health_checks(tmp_path, {}, 0, lambda i: pytest.fail("counts_at must not be called"), "N/A", "N/A")
    assert not (tmp_path / "health_checks").exists()


def test_save_extra_health_checks_degraded_warns_on_stdout_and_does_not_exit(tmp_path, capsys):
    obj, table = _qbcs("per-chunk"), {3: {"110": 20, "000": 80}, 9: {"110": 95, "000": 5}}
    obj._save_extra_health_checks(tmp_path, _save_record("per-chunk"), 2, lambda i: table[i], "N/A", "N/A")   # raises no SystemExit
    out = capsys.readouterr()
    assert "[Chunk 2] Extra health check (pre, circuit 3 of the job) found the QPU degraded: P(011)=0.200" in out.out
    assert "degraded" not in out.err and out.out.count("found the QPU degraded") == 1   # the functional post does not warn
    assert _read(tmp_path, "health_checks", "extra_chunk_2_pre.json")["verdict"] == "degraded"
    assert _read(tmp_path, "health_checks", "extra_chunk_2_post.json")["verdict"] == "functional"
    assert _read(tmp_path, "health_checks", "extra_chunk_2_pre.json")["scheduled_at"] is None   # "N/A" window
    obj._save_extra_health_checks(tmp_path, _save_record("per-chunk"), 2, lambda i: table[i], "N/A", "N/A")
    assert "degraded" not in capsys.readouterr().out   # a re-collection writes nothing and warns about nothing


def test_save_extra_health_checks_layout_from_physical_qubits_file_or_na(tmp_path):
    obj, table = _qbcs("per-chunk"), _TABLE
    rec = _save_record("per-chunk")
    del rec["layout"]
    obj._save_extra_health_checks(tmp_path / "a", rec, 0, lambda i: table[i], "N/A", "N/A")
    assert _read(tmp_path / "a", "health_checks", "extra_chunk_0_pre.json")["layout"] == "N/A"
    (tmp_path / "b" / "transpiled_circuits").mkdir(parents=True)
    (tmp_path / "b" / "transpiled_circuits" / "physical_qubits.json").write_text(json.dumps({"physical_qubits": [11, 12, 13]}), encoding="utf-8")
    obj._save_extra_health_checks(tmp_path / "b", rec, 0, lambda i: table[i], "N/A", "N/A")
    assert _read(tmp_path / "b", "health_checks", "extra_chunk_0_pre.json")["layout"] == {"0": 11, "1": 12, "2": 13}


# ── 2. IBM collection (SamplerV2-shaped result) ──────────────────────────────

class _Reg:
    def __init__(self, counts):
        self._counts = counts

    def get_counts(self):
        return self._counts


@pytest.mark.parametrize("registers", ["hc_c_chunks_meas", "all_meas", "all_c"])
@pytest.mark.parametrize("mode", ["per-chunk", "per-job"])
def test_ibm_collection_saves_the_health_checks_apart_from_the_chunks(tmp_path, registers, mode):
    chunk_nums = [3, 4]
    obj, circuits, _ = _make_run(tmp_path, mode, chunk_nums, "IBM_Cloud")

    def pub(c, counts):   # a pub result: data.<register>.get_counts(); the health-check circuit's register is `c`, the chunks' `meas`
        name = "c" if registers == "all_c" or (registers == "hc_c_chunks_meas" and c == HC) else "meas"
        return SimpleNamespace(data=SimpleNamespace(**{name: _Reg(counts)}))
    result = [pub(c, n) for c, n in zip(circuits, _job_counts(circuits))]
    obj._fake_job_registry = {"job-A": SimpleNamespace(result=lambda: result)}
    for nc in chunk_nums:
        obj.get_results_by_chunk_ibm(tmp_path, nc, service=None, lang="qiskit")
    _assert_collected(tmp_path, circuits, mode, chunk_nums)
    assert _record(tmp_path, 3)["status"] == "complete" and _record(tmp_path, 3)["circuit_check"]["status"] == CC.NOT_AVAILABLE
    for p in (tmp_path / "health_checks").glob("extra_*.json"):   # a simulated backend has no server copy: both slots say so
        assert _read(tmp_path, "health_checks", p.name)["circuit_check"]["status"] == CC.NOT_AVAILABLE


# ── 3. IQM Resonance and VLQ collection ──────────────────────────────────────

@pytest.mark.parametrize("altered, expected", [(False, CC.MATCH), (True, CC.MISMATCH)])
def test_iqm_resonance_collection_saves_the_health_checks_with_their_circuit_check(tmp_path, monkeypatch, altered, expected):
    chunk_nums = [3, 4]
    obj, circuits, _ = _make_run(tmp_path, "per-job", chunk_nums)
    payload = {"job_id": "job-A", "payload": {"shots": 100, "circuits": [_iqm_circuit() for _ in circuits]}}
    if altered:
        payload["payload"]["circuits"][len(circuits) - 1] = _iqm_circuit(bad=True)   # the last circuit (hc_post) was altered by the provider
    _patch_resonance(monkeypatch, obj, payload)
    backend = _Backend(_job_counts(circuits))
    for nc in chunk_nums:
        obj.get_results_by_chunk_iqm_resonance(tmp_path, nc, backend, lang="qiskit")
    _assert_collected(tmp_path, circuits, "per-job", chunk_nums)
    # each slot is compared on its own: the altered last circuit (hc_post) marks only the post record
    assert _read(tmp_path, "health_checks", "extra_job_3_pre.json")["circuit_check"]["status"] == CC.MATCH
    assert _read(tmp_path, "health_checks", "extra_job_3_post.json")["circuit_check"]["status"] == expected
    assert backend.retrieved == ["job-A"]                       # the packed job is fetched once for its chunks
    rec = _record(tmp_path, 4)
    assert rec["status"] == "complete" and rec["circuit_check"]["status"] == CC.NOT_AVAILABLE and rec["hc_post_idx"] == len(circuits) - 1
    assert _read(tmp_path, "health_checks", "extra_job_3_pre.json")["scheduled_at"] == T_START


def test_vlq_collection_saves_the_health_checks_with_their_circuit_check(tmp_path, monkeypatch, capsys):
    chunk_nums = [3, 4]
    obj, circuits, _ = _make_run(tmp_path, "per-chunk", chunk_nums, QAAS_IT4I)
    executed = [_hc_circuit() if c == HC else _hc_circuit(extra_x=True) for c in circuits]
    _, post4 = _hc_positions(circuits, "per-chunk", 4)
    executed[post4] = _hc_circuit(extra_x=True)                   # chunk 4's closing health check ran another circuit
    _patch_vlq(monkeypatch, obj)
    backend = _Backend(_job_counts(circuits), transpiled=executed)
    for nc in chunk_nums:
        obj.get_results_by_chunk_iqm_resonance(tmp_path, nc, backend, lang="qiskit")
    out = capsys.readouterr().out
    _assert_collected(tmp_path, circuits, "per-chunk", chunk_nums)
    status = {p.name: _read(tmp_path, "health_checks", p.name)["circuit_check"]["status"]
              for p in (tmp_path / "health_checks").glob("extra_*.json")}
    assert status == {"extra_chunk_3_pre.json": CC.MATCH, "extra_chunk_3_post.json": CC.MATCH,
                      "extra_chunk_4_pre.json": CC.MATCH, "extra_chunk_4_post.json": CC.MISMATCH}   # each slot on its own
    assert "[Extra health check (chunk 4, post)] THE PROVIDER'S CIRCUITS DIFFER" in out   # the mismatch is printed at collection
    assert "[Extra health check (chunk 4, pre)] The provider's circuits match" in out
    assert backend.retrieved == ["job-A"]


# ── 4. IQM EQE1 collection ───────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["per-chunk", "per-job"])
def test_eqe1_collection_saves_the_health_checks_from_result_counts(tmp_path, monkeypatch, mode):
    chunk_nums = [3, 4]
    obj, circuits, _ = _make_run(tmp_path, mode, chunk_nums, "MQSS_LRZ")
    mqss = types.ModuleType("mqss_client")
    mqss_job = types.ModuleType("mqss_client.job")
    mqss_job.CircuitJobRequest = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "mqss_client", mqss)
    monkeypatch.setitem(sys.modules, "mqss_client.job", mqss_job)
    # an MQSS Result: counts and naive Europe/Berlin timestamp strings
    result = SimpleNamespace(counts=_job_counts(circuits), timestamp_submitted="2026-10-01T12:00:01.000000",
                             timestamp_scheduled="2026-10-01T12:00:02.000000", timestamp_completed="2026-10-01T12:00:07.000000")
    fetched = []
    adapter = SimpleNamespace(client=SimpleNamespace(wait_for_job_result=lambda job_id, req: fetched.append(job_id) or result))
    for nc in chunk_nums:
        obj.get_results_by_chunk_iqm(tmp_path, nc, adapter, lang="qiskit")
    _assert_collected(tmp_path, circuits, mode, chunk_nums)
    assert fetched == ["job-A"]
    for p in (tmp_path / "health_checks").glob("extra_*.json"):   # EQE1 returns no executed circuits: both slots say so
        assert _read(tmp_path, "health_checks", p.name)["circuit_check"]["status"] == CC.NOT_AVAILABLE
    rec = _record(tmp_path, 3)
    assert rec["circuit_check"]["status"] == CC.NOT_AVAILABLE and rec["status"] == "complete"
    pre = _read(tmp_path, "health_checks", f"extra_{'chunk' if mode == 'per-chunk' else 'job'}_3_pre.json")
    assert pre["scheduled_at"].startswith("2026-10-01T10:00:02") and pre["finished_at"].startswith("2026-10-01T10:00:07")   # UTC


# ── 5. Packing -> offsets -> collection round trip ───────────────────────────

@pytest.mark.parametrize("chunk_nums", [[0], [5, 6, 7]])
@pytest.mark.parametrize("mode", ["per-chunk", "per-job"])
def test_packed_job_round_trip_gives_every_circuit_back_from_its_own_position(tmp_path, monkeypatch, mode, chunk_nums):
    obj, circuits, offsets = _make_run(tmp_path, mode, chunk_nums)
    assert circuits.count(HC) == (2 * len(chunk_nums) if mode == "per-chunk" else 2)
    for nc in chunk_nums:   # the recorded offsets point at what was packed there
        r = _record(tmp_path, nc)
        assert circuits[r["circ_start"]:r["circ_end"]] == [("circ", nc, k) for k in range(N_CIRC)]
        assert circuits[r["spam_start"]:r["circ_start"]] == [("spam", si) for si in range(N_SPAM)]
        assert circuits[r["hc_pre_idx"]] == HC and circuits[r["hc_post_idx"]] == HC
        assert r["n_circuits_in_job"] == len(circuits)
    _patch_resonance(monkeypatch, obj)
    for nc in chunk_nums:
        obj.get_results_by_chunk_iqm_resonance(tmp_path, nc, _Backend(_job_counts(circuits)), lang="qiskit")
    _assert_collected(tmp_path, circuits, mode, chunk_nums)
    pre = [_read(tmp_path, "health_checks", p.name)["circuit_index"] for p in (tmp_path / "health_checks").glob("extra_*_pre.json")]
    assert len(set(pre)) == len(pre)   # per-chunk: each chunk's own pre; per-job: the one pre


# ── 6. _check_extra_hc ───────────────────────────────────────────────────────

def test_check_extra_hc_match_mismatch_and_a_missing_qpy_file(tmp_path):
    (tmp_path / "transpiled_circuits").mkdir()
    with open(tmp_path / "transpiled_circuits" / "hc.qpy", "wb") as f:
        dump([_hc_circuit()], f)
    obj = _qbcs("per-chunk")
    record = {"extra_hc_circuit_file": "hc.qpy", "hc_pre_idx": 1, "hc_post_idx": 4}
    asked = []
    executed = {1: _hc_circuit(), 4: _hc_circuit()}

    def compare(submitted, idx):
        asked.append((len(submitted), idx))
        return CC.check_vlq(submitted, [executed[i] for i in idx], "stand-in")
    checks = obj._check_extra_hc(tmp_path, record, compare, "stand-in")
    assert checks["pre"]["status"] == CC.MATCH and checks["post"]["status"] == CC.MATCH
    assert asked == [(1, [1]), (1, [4])]               # the health-check circuit once per slot
    executed[4] = _hc_circuit(extra_x=True)
    checks = obj._check_extra_hc(tmp_path, record, compare, "stand-in")
    assert checks["pre"]["status"] == CC.MATCH         # only the post slot differs
    assert checks["post"]["status"] == CC.MISMATCH and checks["post"]["differences"][0]["circuit"] == 0
    record["extra_hc_circuit_file"] = "missing.qpy"
    checks = obj._check_extra_hc(tmp_path, record, compare, "stand-in")   # does not raise
    for check in checks.values():
        assert check["status"] == CC.CHECK_FAILED and check["method"] == "stand-in"
        assert "FileNotFoundError" in check["error"] and str(tmp_path) not in check["error"]
    record["extra_hc_circuit_file"] = "hc.qpy"
    boom = obj._check_extra_hc(tmp_path, record, lambda s, i: 1 / 0 if i == [4] else compare(s, i), "stand-in")
    assert boom["pre"]["status"] == CC.MATCH           # a failing slot does not spoil the other
    assert boom["post"]["status"] == CC.CHECK_FAILED and "ZeroDivisionError" in boom["post"]["error"]


def test_save_extra_health_checks_stores_each_slots_own_circuit_check(tmp_path):
    cc = {"pre": CC.not_available("a"), "post": CC.check_failed("m", ValueError("b"))}
    _qbcs("per-chunk")._save_extra_health_checks(tmp_path, _save_record("per-chunk"), 0, lambda i: _TABLE[i], T_START, T_END, cc)
    assert _read(tmp_path, "health_checks", "extra_chunk_0_pre.json")["circuit_check"]["status"] == CC.NOT_AVAILABLE
    assert _read(tmp_path, "health_checks", "extra_chunk_0_post.json")["circuit_check"]["status"] == CC.CHECK_FAILED


# ── 6b. The circuit verification of a run counts the extra records ───────────

def _extra_record(chunk, position, check, layout=None):
    return HR.extra_health_check_record(
        mode="per-chunk", position=position, chunks=[chunk], circuit_index=0, n_circuits_in_job=4, counts={"011": 95, "000": 5},
        nshots=100, layout=layout or {"0": 5, "1": 6, "2": 7}, backend_name="b", service_name="MQSS_LRZ", job_id=f"job-{chunk}",
        submitted_at=SUBMITTED, scheduled_at=T_START, finished_at=T_END, circuit_check=check)


def _verified_run(tmp_path):
    """A run with one complete chunk and extra records: chunk 0 (pre match, post mismatch), chunk 1 (not_available pre on the
    run's qubits, post on other qubits), chunk 2 (an older record without circuit_check)."""
    (tmp_path / "job_records").mkdir()
    (tmp_path / "transpiled_circuits").mkdir()
    (tmp_path / "transpiled_circuits" / "physical_qubits.json").write_text(json.dumps({"physical_qubits": [5, 6, 7]}), encoding="utf-8")
    rec = {"chunk_id": 0, "status": "complete", "service_name": "MQSS_LRZ", "n_spam": 0, "circuit_check": CC.not_available("x")}
    (tmp_path / "job_records" / "job_record_chunk_0.json").write_text(json.dumps(rec), encoding="utf-8")
    for r in (_extra_record(0, "pre", {"status": CC.MATCH, "method": "m"}),
              _extra_record(0, "post", {"status": CC.MISMATCH, "method": "m", "differences": [{"circuit": 0}]}),
              _extra_record(1, "pre", CC.not_available("x")),
              _extra_record(1, "post", CC.not_available("x"), {"0": 5, "1": 6, "2": 9}),
              _extra_record(2, "pre", None)):   # the last one: written before circuit_check existed
        HR.save_extra_health_check(tmp_path, r)
    return tmp_path


def test_run_verification_counts_mismatched_and_not_available_extra_records(tmp_path):
    summary = CC.summarize_circuit_checks(_verified_run(tmp_path))
    hc = summary["health_checks"]
    assert hc["n_records"] == 4                                   # the record without circuit_check is left out, none counted twice
    assert hc["status"] == CC.MISMATCH
    assert hc["mismatched_ids"] == ["extra chunk 0 post", "extra chunk 1 post"]   # the second one: other qubits than the run's
    assert hc["counts"] == {CC.MATCH: 2, CC.MISMATCH: 2}
    assert summary["chunks"]["n_records"] == 1                    # the chunk itself is counted once, unaffected
    assert "health check extra chunk 0 post and others" in CC.verification_warning(summary)


def test_run_verification_leaves_out_an_extra_record_without_circuit_check(tmp_path):
    run = _verified_run(tmp_path)
    for p in (run / "health_checks").glob("extra_*.json"):
        p.unlink()
    HR.save_extra_health_check(run, _extra_record(2, "pre", None))
    summary = CC.summarize_circuit_checks(run)
    assert summary["health_checks"]["n_records"] == 0 and summary["health_checks"]["status"] == "no_records"
    assert CC.verification_warning(summary) is None


# ── 7. A repeated collection of a per-job job ────────────────────────────────

def test_repeated_per_job_collection_leaves_records_and_results_as_they_were(tmp_path, monkeypatch, capsys):
    chunk_nums = [3, 4]
    obj, circuits, _ = _make_run(tmp_path, "per-job", chunk_nums)
    _patch_resonance(monkeypatch, obj)
    for nc in chunk_nums:
        obj.get_results_by_chunk_iqm_resonance(tmp_path, nc, _Backend(_job_counts(circuits)), lang="qiskit")

    def snapshot():
        files = {p.name: p.read_bytes() for p in (tmp_path / "health_checks").iterdir()}
        raw = {p.name: p.read_bytes() for p in (tmp_path / "raw_results").iterdir()}
        return files, raw

    first = snapshot()
    first_records = {nc: _record(tmp_path, nc) for nc in chunk_nums}
    # a second session re-collecting both chunks of the job whose records were reset to "submitted"
    for nc in chunk_nums:
        rec = _record(tmp_path, nc)
        rec["status"] = "submitted"
        (tmp_path / "job_records" / f"job_record_chunk_{nc}.json").write_text(json.dumps(rec), encoding="utf-8")
    obj2, backend = _qbcs("per-job"), _Backend(_job_counts(circuits))
    _patch_resonance(monkeypatch, obj2)
    capsys.readouterr()
    for nc in chunk_nums:
        obj2.get_results_by_chunk_iqm_resonance(tmp_path, nc, backend, lang="qiskit")
    printed = capsys.readouterr().out
    assert "degraded" not in printed and "Extra health checks (" not in printed
    assert snapshot() == first                      # no duplicated, rewritten or extra file
    assert sorted(first[0]) == ["extra_job_3_post.json", "extra_job_3_pre.json"]
    assert backend.retrieved == ["job-A"]
    for nc in chunk_nums:
        again = _record(tmp_path, nc)
        assert again["status"] == "complete"
        for key in ("hc_pre_idx", "hc_post_idx", "n_circuits_in_job", "job_chunks", "extra_health_checks", "spam_start", "circ_start", "circ_end"):
            assert again[key] == first_records[nc][key]
    # a complete chunk is skipped without fetching the job again
    backend3 = _Backend(_job_counts(circuits))
    obj2.__dict__.pop("_iqm_job_cache")
    obj2.get_results_by_chunk_iqm_resonance(tmp_path, 3, backend3, lang="qiskit")
    assert backend3.retrieved == [] and "Already collected" in capsys.readouterr().out
    _assert_collected(tmp_path, circuits, "per-job", chunk_nums)


# ── 8. plot_health_checks.py with degraded, half-missing and failed-job records ──

def _plot_extras(name, specs, failed_chunk=True, hc_threshold=None):
    """Run plot_health_checks.py on a run whose extra records are specs = [(chunk, position, counts, index)]; returns (summary, stdout).
    With hc_threshold=T the analysis is filtered (QPU_HC_THRESHOLD) and the summary read is health_checks+hc-gt-T.json."""
    folder = run_name_for_test(name)
    run_path = PROJECT_ROOT / "datafiles" / folder
    plots = PROJECT_ROOT / run_plots_dir(folder)
    shutil.rmtree(run_path, ignore_errors=True)
    try:
        (run_path / "job_records").mkdir(parents=True)
        for nc, job_id, status in ((0, "job-0", "complete"), (1, "job-1", "complete"), (2, "job-2", "failed" if failed_chunk else "complete")):
            rec = {"chunk_id": nc, "job_id": job_id, "status": status, "service_name": "MQSS_LRZ", "backend_name": "b",
                   "submitted_at": f"2026-10-01T10:0{nc}:00Z", "scheduled_at": f"2026-10-01T10:0{nc}:10Z",
                   "finished_at": f"2026-10-01T10:0{nc}:50Z"}
            if status == "failed":
                rec.update(failure="stand-in", failure_stage="execution")
            (run_path / "job_records" / f"job_record_chunk_{nc}.json").write_text(json.dumps(rec), encoding="utf-8")

        def extra(chunk, position, p011_counts, index):
            return HR.extra_health_check_record(
                mode="per-chunk", position=position, chunks=[chunk], circuit_index=index, n_circuits_in_job=10, counts=p011_counts,
                nshots=100, layout={"0": 5, "1": 6, "2": 7}, backend_name="b", service_name="MQSS_LRZ", job_id=f"job-{chunk}",
                submitted_at=f"2026-10-01T10:0{chunk}:00Z", scheduled_at=f"2026-10-01T10:0{chunk}:10Z",
                finished_at=f"2026-10-01T10:0{chunk}:50Z")
        for chunk, position, counts, index in specs:
            HR.save_extra_health_check(run_path, extra(chunk, position, counts, index))
        write_run_info(config=dict(shots=100, samples=3, T_list=[0.2], dt=0.2, SPAM_correction=True, chunk_size=1),
                       run_folder_path=run_path, lnn=True)
        env = {**os.environ, "QPU_RUN_FOLDER": folder, "QPU_N_SAMPLES": ""}
        if hc_threshold is not None:
            env["QPU_HC_THRESHOLD"] = str(hc_threshold)
        r = subprocess.run([venv_python("local_sim.venv", "test_31 (plot_health_checks.py)"),
                            str(PROJECT_ROOT / "source" / "analysis_scripts" / "plot_health_checks.py")],
                           cwd=PROJECT_ROOT, env=env, capture_output=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0, r.stdout + r.stderr
        suffix = "" if hc_threshold is None else f"+hc-gt-{hc_threshold}"
        assert (PROJECT_ROOT / plots / f"health_checks{suffix}.png").exists()
        return _read(run_path, "processed_results", f"health_checks{suffix}.json"), r.stdout
    finally:
        shutil.rmtree(run_path, ignore_errors=True)
        shutil.rmtree(plots, ignore_errors=True)


def test_plot_health_checks_handles_degraded_half_missing_and_failed_extra_records():
    good, bad = {"011": 95, "000": 5}, {"011": 20, "000": 80}
    out, stdout = _plot_extras("test31-extra-plot", [(0, "pre", bad, 0), (0, "post", good, 9), (1, "pre", good, 0)])   # chunk 1: its post is missing
    rows = [(e["chunks"], e["position"], e["verdict"]) for e in out["extra_health_checks"]]
    assert rows == [([0], "pre", "degraded"), ([0], "post", "functional"), ([1], "pre", "functional")]
    assert all(e["x"] is not None and e["estimated_time"] is not None for e in out["extra_health_checks"])
    assert [j["chunks"] for j in out["failed_chunk_jobs"]] == [[2]]
    assert not any(e["chunks"] == [1] and e["position"] == "post" for e in out["extra_health_checks"])
    assert out["incomplete_extra_health_checks"] == [{"mode": "per-chunk", "chunks": [1], "missing": "post"}]
    assert "⚠️  1 extra health check(s) without their pre or post" in stdout and "no post" in stdout


def test_plot_health_checks_reports_no_incomplete_extra_records_for_a_complete_run():
    good = {"011": 95, "000": 5}
    out, stdout = _plot_extras("test31-extra-plot-ok", [(0, "pre", good, 0), (0, "post", good, 9), (1, "pre", good, 0), (1, "post", good, 9)],
                               failed_chunk=False)
    assert out["incomplete_extra_health_checks"] == []
    assert "without their pre or post" not in stdout
    assert "filter_threshold" not in out   # an unfiltered analysis records no threshold
    filtered, _ = _plot_extras("test31-extra-plot-ok", [(0, "pre", good, 0), (0, "post", good, 9), (1, "pre", good, 0), (1, "post", good, 9)],
                               failed_chunk=False, hc_threshold=0.5)
    assert filtered["filter_threshold"] == 0.5


def _extra(position, chunks, p011):
    return {"schema": HR.EXTRA_HEALTH_CHECK_SCHEMA, "position": position, "chunks": chunks, "P011": p011}


def test_chunks_passing_extra_health_checks_per_chunk_per_job_and_edge(tmp_path):
    d = tmp_path / "health_checks"
    d.mkdir()
    recs = [_extra("pre", [0], 0.9), _extra("post", [0], 0.8),    # passes
            _extra("pre", [1], 0.9), _extra("post", [1], 0.5),    # P011 == threshold: not above
            _extra("pre", [2], 0.9),                              # no post
            _extra("pre", [3, 4], 0.7), _extra("post", [3, 4], 0.6),   # a per-job pair: both chunks pass
            _extra("pre", [5, 6], 0.7), _extra("post", [5, 6], 0.2)]   # a per-job pair failing: both chunks out
    for i, r in enumerate(recs):
        (d / f"extra_{i}.json").write_text(json.dumps(r), encoding="utf-8")
    assert HR.chunks_passing_extra_health_checks(tmp_path, 0.5) == frozenset({0, 3, 4})
    assert HR.chunks_passing_extra_health_checks(tmp_path, 0.75) == frozenset({0})


def test_chunks_passing_extra_health_checks_is_none_without_extra_checks(tmp_path):
    assert HR.chunks_passing_extra_health_checks(tmp_path, 0.5) is None


def test_health_check_scope_names_files_and_figures():
    from source.core.records.chunk_scope import HealthCheckScope, figure_filter_label
    from source.core.records.run_names import subset_suffix
    scope = HealthCheckScope({0, 3}, 0.5, 48)
    assert subset_suffix(scope) == "+hc-gt-0.5"
    assert figure_filter_label(scope) == ", health checks P011 > 0.5 (2 of 48 chunks kept)"
    assert figure_filter_label(None) == "" and figure_filter_label(8) == ""


def test_spam_calibration_counts_of_a_chunk_set(tmp_path):
    from source.core.circuits.SPAM_functions import load_merged_spam_calibration_counts
    run = tmp_path / "run"
    (run / "raw_results").mkdir(parents=True)
    per_chunk = {f"chunk_{c}": {"0": {"0": 10 * (c + 1)}} for c in range(3)}
    (run / "raw_results" / "spam_calibration_counts_per_chunk.json").write_text(json.dumps(per_chunk), encoding="utf-8")
    config = {"folder_path": str(run)}
    assert load_merged_spam_calibration_counts(config, n_samples=frozenset({0, 2})) == {"0": {"0": 40}}
    assert load_merged_spam_calibration_counts(config, n_samples=2) == {"0": {"0": 30}}
    with pytest.raises(AssertionError):
        load_merged_spam_calibration_counts(config, n_samples=frozenset({7}))
