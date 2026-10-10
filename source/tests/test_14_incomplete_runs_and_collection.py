"""
test_14_incomplete_runs_and_collection.py
=========================================
Unit tests (no simulator, no QPU, no network) for

  - runs with uncollected chunks in the middle (source/core/records/chunk_scope.py): merging keeps each
    chunk's own number (the NEC analysis derives the sample's input state from it), an
    analysis of n chunks covers the first n collected ones — in timing, workload —, its SPAM
    correction uses every collected chunk's calibration, and the completeness block lists what is missing;
  - the collection time (Collect_time_s): chunk{nc}_collect_* labels of the run's log and the
    health checks' timing.collect_s, "N/A" for runs collected before it was timed, and its
    part of the total time (FT_collect);
  - non-blocking collection: finished chunks are collected in any order, unfinished ones left
    pending, failed ones marked failed — for chunk jobs and health checks alike;
  - the chunks a submission covers: the ones without a 'submitted' / 'complete' record (gaps
    included), required by every accompanying health check, which records them as `chunks`
    (older records: `jobs` / `prior_chunks`, still read);
  - the health check plot of an incomplete run: +nsamples-N file names, the uncollected health
    checks and chunk jobs listed.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import json
import shutil
import numpy as np
import subprocess
from source.core.runner_support.venvs import venv_python
from pathlib import Path

import pytest
from source.tests.testing_functions import run_name_for_test
from source.core.records.run_names import plots_dir as run_plots_dir

import source.core.qbcs as Q
from source.core.analysis import derived_quantities as DQ
from source.core import health_check as HC
from source.core.analysis import mitigation as MI
from source.core.analysis import run_timing as RT
from source.core.records import timestamps as TS
from source.core.records import chunk_scope as CS
from source.core.circuits import SPAM_functions as SF
from source.core.common import get_target_bitstring
from source.core.records.run_info import write_run_info

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLES_CHUNK  = {"job_Nt_01_rc_ems2": "Nt_01_rc_circuit_0"}


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _counts(chunks):
    return {f"chunk_{c}": {job: {circ: {"000": c + 1}} for job, circ in ROLES_CHUNK.items()} for c in chunks}


def _log(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("label\ttimestamp\n")
        for lab, t in rows:
            f.write(f"{lab}\t{t}\n")


# ── Gaps: merge, scope, completeness ─────────────────────────────────────────
def test_merge_keeps_chunk_numbers_so_nec_input_states_stay_right():
    merged = MI.merge_chunks(_counts([0, 1, 3, 9]))["job_Nt_01_rc_ems2"]
    assert list(merged) == [f"Nt_01_rc_circuit_{c}" for c in (0, 1, 3, 9)]
    # sample 9's input state is 9 % 8 = 1 -> '001', not that of its position (3 -> '011')
    nec_map = {format(i, "03b"): format(i, "03b") for i in range(8)}
    assert get_target_bitstring("Nt_01_rc_circuit_9", 16, 3, nec_map) == "001"
    # a chunk holding several circuits per job (old data) keeps the running index
    old = {"chunk_0": {"j": {"c_0": 1, "c_1": 2}}, "chunk_2": {"j": {"c_0": 3, "c_1": 4}}}
    assert list(MI.merge_chunks(old)["j"]) == ["c_0", "c_1", "c_2", "c_3"]


def test_scope_is_the_first_n_collected_chunks(tmp_path):
    _write(tmp_path / "raw_results" / "results_counts.json", _counts([0, 1, 3, 4]))
    assert CS.collected_chunks(tmp_path) == [0, 1, 3, 4]
    assert CS.chunk_scope(tmp_path, 3) == {0, 1, 3}
    assert CS.chunk_scope(tmp_path, None) is None
    assert CS.chunk_scope(tmp_path / "nothing", 2) == {0, 1}            # no counts: [0, n)
    assert CS.as_chunk_set(3) == {0, 1, 2} and CS.scope_last_chunk({0, 1, 3}) == 3


def test_completeness_block_and_warning(tmp_path):
    _write(tmp_path / "raw_results" / "results_counts.json", _counts([0, 1, 3]))
    _write(tmp_path / "health_checks" / "health_check_1.json", {"number": 1, "schema": 2, "status": "complete"})
    _write(tmp_path / "health_checks" / "health_check_2.json", {"number": 2, "schema": 2, "status": "submitted"})
    c = CS.completeness(tmp_path, 6)
    assert c["n_chunks_analyzed"] == 3 and c["chunks_analyzed"] == "0-1,3"
    assert c["uncollected_chunks"] == [2, 4, 5] and c["uncollected_health_checks"] == [2] and not c["complete"]
    w = CS.completeness_warning(c)
    assert "3 of 6 chunks" in w and "chunks 2, 4-5 are uncollected" in w and "health check 2 is uncollected" in w
    _write(tmp_path / "raw_results" / "results_counts.json", _counts(range(6)))
    _write(tmp_path / "health_checks" / "health_check_2.json", {"number": 2, "schema": 2, "status": "complete"})
    assert CS.completeness(tmp_path, 6)["complete"] and CS.completeness_warning(CS.completeness(tmp_path, 6)) is None


def test_spam_calibration_of_a_subset_uses_the_first_collected_chunks(tmp_path):
    per_input = lambda c: {format(i, "03b"): {format(i, "03b"): c} for i in range(8)}
    _write(tmp_path / "raw_results" / "spam_calibration_counts_per_chunk.json",
           {f"chunk_{c}": per_input(c + 1) for c in (0, 1, 3)})
    totals = SF.load_merged_spam_calibration_counts({"folder_path": str(tmp_path)}, n_samples=3)
    assert totals["000"]["000"] == 1 + 2 + 4
    with pytest.raises(AssertionError):
        SF.load_merged_spam_calibration_counts({"folder_path": str(tmp_path)}, n_samples=4)


def test_subset_analysis_uses_all_collected_spam_calibration(tmp_path):
    # 4 chunks of SPAM calibration; the analysis of the first 2 RC samples corrects with the matrix of all 4
    # (spam_confusion_matrix_inv.npy) and writes no +nsamples-2 matrix, unless spam_samples=2 asks for it
    per_input = lambda c: {format(i, "03b"): {format(i, "03b"): 9 + c, format((i + 1) % 8, "03b"): 1} for i in range(8)}
    _write(tmp_path / "raw_results" / "spam_calibration_counts_per_chunk.json", {f"chunk_{c}": per_input(c) for c in range(4)})
    _write(tmp_path / "raw_results" / "results_counts_merged.json",
           {"chunk_0": {"job_Nt_01_raw_ems0": {f"rc_circuit_{c}": {"000": 10 - c, "001": c} for c in range(4)}}})
    _write(tmp_path / "raw_results" / "results_fidelity_merged.json", {"chunk_0": {"job_Nt_01_raw_nec_ems1": [1.0] * 4}})
    config = {"folder_path": str(tmp_path), "samples": 4, "nsites": 3, "T_list": [1], "dt": 1, "SPAM_correction": True,
              "nec_measurement_counts": False, "nec_randomize_input_state": False}
    processed = tmp_path / "processed_results"
    kw = dict(rc_samples=2, roles=["raw", "raw_nec"])
    with pytest.warns(UserWarning):
        MI.compute_mitigated_expectation_values(config, **kw)
    assert (processed / "spam_confusion_matrix_inv.npy").exists()
    assert not list(processed.glob("*+nsamples-*"))
    full = np.load(processed / "spam_confusion_matrix_inv.npy")
    assert SF.count_spam_calibration_chunks(config) == 4
    with pytest.warns(UserWarning):
        MI.compute_mitigated_expectation_values(config, spam_samples=2, **kw)
    assert (processed / "spam_confusion_matrix_inv+nsamples-2.npy").exists()
    assert not np.allclose(full, np.load(processed / "spam_confusion_matrix_inv+nsamples-2.npy"))


def test_workload_and_qpu_time_of_a_run_with_a_gap(tmp_path):
    _write(tmp_path / "raw_results" / "results_counts.json", _counts([0, 1, 3]))
    for nc, status in [(0, "complete"), (1, "complete"), (2, "submitted"), (3, "complete")]:
        _write(tmp_path / "job_records" / f"job_record_chunk_{nc}.json",
               {"chunk_id": nc, "job_id": f"j{nc}", "status": status, "service_name": "IQM_Resonance",
                "n_spam": 8, "n_circuits": 12, "nshots": 10,
                "timing": {"qpu_s": 1.0, "qpu_s_source": "x", "queue_s": 5.0}})
    assert RT.qpu_workload(tmp_path, 3)["n_chunks"] == 3                 # chunks 0, 1, 3
    assert RT.qpu_workload(tmp_path, 2)["n_chunks"] == 2                 # chunks 0, 1
    assert RT.qpu_queue_breakdown(tmp_path, 3)["qpu_total_s"] == 3.0


# ── Collection time ───────────────────────────────────────────────────────────
def test_collect_time_from_the_log_labels_and_the_health_checks(tmp_path):
    _write(tmp_path / "raw_results" / "results_counts.json", _counts([0, 1, 3]))
    for nc in (0, 1, 3):
        _write(tmp_path / "job_records" / f"job_record_chunk_{nc}.json", {"chunk_id": nc, "status": "complete"})
    # chunk 3 finished first and was collected first; chunk 1 was collected before the collection was timed
    _log(tmp_path / "timestamps_utc.csv", [
        ("chunk3_collect_start", "2026-09-29T10:00:00.000000Z"), ("chunk3_collect_end", "2026-09-29T10:00:04.000000Z"),
        ("chunk0_collect_start", "2026-09-29T10:05:00.000000Z"), ("chunk0_collect_end", "2026-09-29T10:05:02.500000Z"),
        ("chunk0_transpile_start", "2026-09-29T09:00:00.000000Z"), ("chunk0_transpile_end", "2026-09-29T09:00:09.000000Z")])
    shutil.copy(tmp_path / "timestamps_utc.csv", tmp_path / "timestamps_local_time.csv")
    _write(tmp_path / "health_checks" / "health_check_1.json",
           {"number": 1, "schema": 2, "status": "complete", "jobs": [], "prior_chunks": [], "timing": {"collect_s": 1.5}})
    c = RT.collect_time_breakdown(tmp_path)
    assert c["chunks_s"] == 6.5 and c["health_checks_s"] == 1.5 and c["total_s"] == 8.0
    assert c["n_chunks_timed"] == 2 and c["n_chunks_without"] == 1
    assert RT.collect_time_breakdown(tmp_path, {0, 1})["chunks_s"] == 2.5     # scoped: chunk 3 left out
    # the collect labels are not CPU time
    rows = TS._read_timestamp_rows(tmp_path / "timestamps_utc.csv")
    rows = [(l, TS._parse_timestamp(t)) for l, t in rows]
    assert RT._cpu_label_times_from_rows(rows, None)["transpile_s"] == 9.0


def test_runs_collected_before_it_was_timed_are_na(tmp_path):
    _write(tmp_path / "job_records" / "job_record_chunk_0.json", {"chunk_id": 0, "status": "complete"})
    c = RT.collect_time_breakdown(tmp_path)
    assert c["total_s"] is None and c["n_chunks_without"] == 1


def test_collection_is_part_of_the_total_time():
    awae = {e: [0.1] for e in ["bare", "NEC", "RC", "NEC+RC", "cRC", "NEC+cRC"]}
    d = DQ.compute_derived_quantities(awae, {}, 10.0, 5.0, Nt_vals=[1], T_queue_total=3.0, T_collect_total=2.0)
    assert d["T_generation_submission_execution_s"] == 20.0 and d["Collect_time_s"] == 2.0
    assert d["FT_clas"] + d["FT_quant"] + d["FT_queue"] + d["FT_collect"] == pytest.approx(1.0)
    d = DQ.compute_derived_quantities(awae, {}, 10.0, 5.0, Nt_vals=[1], T_queue_total=3.0)
    assert d["Collect_time_s"] == "N/A" and d["FT_collect"] == "N/A" and d["T_generation_submission_execution_s"] == 18.0


def test_best_ems_skips_nan_and_is_none_when_nothing_beats_one():
    names = ["bare", "NEC", "RC", "NEC+RC", "cRC", "NEC+cRC"]
    awae = {e: [0.5] for e in names}
    awae["bare"] = [float("nan")]       # an earlier scheme without data must not shift the index
    awae["NEC+RC"] = [0.2]
    d = DQ.compute_derived_quantities(awae, {}, 10.0, 5.0, Nt_vals=[1])
    assert d["best_ems_Nmax"] == "NEC+RC" and d["E_min_Nmax"] == pytest.approx(0.2)
    awae = {e: [0.995] for e in names}  # below 1: the best scheme is named (no tolerance)
    assert DQ.compute_derived_quantities(awae, {}, 10.0, 5.0, Nt_vals=[1])["best_ems_Nmax"] == "bare"
    awae = {e: [1.2] for e in names}    # none below 1: no best scheme, no crash
    d = DQ.compute_derived_quantities(awae, {}, 10.0, 5.0, Nt_vals=[1])
    assert d["best_ems_Nmax"] is None and d["best_ems_Nacc"] is None


# ── Non-blocking collection ──────────────────────────────────────────────────
@pytest.fixture
def pending_run(tmp_path, monkeypatch):
    """A run of 4 chunks in 3 IBM jobs: A (chunks 0, 1) finished, B (chunk 2) still running,
    C (chunk 3) failed. The provider is faked: _remote_job_state answers from STATES, and
    _collect_chunk writes the chunk's counts like the real collection does."""
    q = Q.bacq_ps_qbcs(sample_size=4, list_circuit_times=[0.2], error_mitigation_strategies=["NEC"],
                       do_SPAM=False, nshots=10)
    cfg = {k: v for k, v in q.CONFIG.items() if isinstance(v, (str, int, float, bool, list, type(None)))}
    jobs = {0: "A", 1: "A", 2: "B", 3: "C"}
    for nc, job in jobs.items():
        _write(tmp_path / "job_records" / f"job_record_chunk_{nc}.json",
               {"chunk_id": nc, "job_id": job, "status": "submitted", "CONFIG": cfg, "nshots": 10})
    states = {"A": ("done", "DONE"), "B": ("pending", "RUNNING"), "C": ("failed", "ERROR")}
    queried = []

    def state(service_name, service, job_id, *a):
        queried.append(job_id)
        return states[job_id]

    def collect(self, service_name, path, nc, service, lang, poll, backend_name=None):
        counts = path / "raw_results" / "results_counts.json"
        data = json.loads(counts.read_text(encoding="utf-8")) if counts.exists() else {}
        data.update(_counts([nc]))
        _write(counts, data)
        _write(path / "raw_results" / "results_fidelity.json", data)
        rec_path = path / "job_records" / f"job_record_chunk_{nc}.json"
        _write(rec_path, {**json.loads(rec_path.read_text(encoding="utf-8")), "status": "complete"})

    monkeypatch.setattr(Q, "_remote_job_state", state)
    monkeypatch.setattr(Q.bacq_ps_qbcs, "_collect_chunk", collect)
    monkeypatch.setattr(Q.bacq_ps_qbcs, "merge_all_chunks", lambda self: None)
    return q, tmp_path, states, queried


def test_collection_takes_the_finished_jobs_and_leaves_the_rest(pending_run):
    q, run, states, queried = pending_run
    pending = q.collect_results(run, service=object(), service_name="IBM_Cloud")
    assert pending == [2] and sorted(queried) == ["A", "B", "C"]          # one status query per job
    rec = lambda nc: json.loads((run / "job_records" / f"job_record_chunk_{nc}.json").read_text(encoding="utf-8"))
    assert rec(0)["status"] == rec(1)["status"] == "complete" and rec(2)["status"] == "submitted"
    assert rec(3)["status"] == "failed" and "ERROR" in rec(3)["failure"]
    assert rec(0)["timing"]["collect_s"] >= 0                            # timed, and logged
    labels = [r[0] for r in TS._read_timestamp_rows(run / "timestamps_local_time.csv")]
    assert labels == ["chunk0_collect_start", "chunk0_collect_end", "chunk1_collect_start", "chunk1_collect_end"]
    # later: job B has finished — only chunk 2 is collected
    states["B"] = ("done", "DONE")
    queried.clear()
    assert q.collect_results(run, service=object(), service_name="IBM_Cloud") == []
    assert queried == ["B"] and rec(2)["status"] == "complete"


def test_collect_and_analyze_with_no_finished_job_stops_without_analyzing(pending_run, capsys):
    from source.core.runner_support.run_collection import collect_and_analyze
    from source.core.runner_support.benchmark_args import parse_benchmark_args
    q, run, states, queried = pending_run
    for job in states:
        states[job] = ("pending", "RUNNING")
    collect_and_analyze(run, object(), "IBM_Cloud", parse_benchmark_args(["mini"]), True)      # must not raise
    assert "Nothing to analyze yet" in capsys.readouterr().out


def test_collecting_an_aborted_run_ends_with_one_line_not_a_traceback(tmp_path, capsys):
    # the submit script stopped at a degraded pre-submission health check: health_checks/ but no job_records/
    from source.core.runner_support.run_collection import collect_run
    from source.core.runner_support.benchmark_args import parse_benchmark_args
    _write(tmp_path / "health_checks" / "health_check_1.json",
           {"number": 1, "schema": 2, "status": "complete", "service_name": "IBM_Cloud", "backend_name": "ibm_fez",
            "job_id": "h1", "nshots": 100, "P011": 0.29, "verdict": "degraded"})
    with pytest.raises(SystemExit) as e:
        collect_run(tmp_path, object(), "IBM_Cloud", "ibm_fez", parse_benchmark_args(["mini"]), True)
    assert e.value.code == 2
    captured = capsys.readouterr()
    assert "degraded" in captured.out and "No chunk was submitted" in captured.err and "Traceback" not in captured.err


def test_health_check_collection_skips_unfinished_and_marks_failed(tmp_path, monkeypatch):
    for n in (1, 2):
        _write(tmp_path / "health_checks" / f"health_check_{n}.json",
               {"number": n, "schema": 2, "status": "submitted", "job_id": f"h{n}", "service_name": "QAAS_IT4I",
                "backend_name": "iqm_vlq", "nshots": 100})
    monkeypatch.setattr(HC, "_remote_job_state",
                        lambda s, svc, job_id, *a: {"h1": ("pending", "RUNNING"), "h2": ("failed", "ERROR")}[job_id])
    called = []
    monkeypatch.setattr(HC, "_collect_health_check_one", lambda *a, **k: called.append(a))
    HC.collect_health_check(tmp_path, service=object(), service_name="QAAS_IT4I", accompanying=True)
    hc = lambda n: json.loads((tmp_path / "health_checks" / f"health_check_{n}.json").read_text(encoding="utf-8"))
    assert not called and hc(1)["status"] == "submitted" and hc(2)["status"] == "failed"
    assert CS.uncollected_health_checks(tmp_path) == [1, 2]
    # one requested check (a submission's pre-submission check) is waited for, not skipped
    HC.collect_health_check(tmp_path, service=object(), service_name="QAAS_IT4I", accompanying=True, number=1)
    assert len(called) == 1


def test_remote_job_state_unknown_service_and_missing_backend_name_raise():
    from source.core.qpu.remote_jobs import _remote_job_state
    with pytest.raises(ValueError, match="unknown service"):
        _remote_job_state("NEW_SERVICE", object(), "j1")
    assert _remote_job_state("PCSS", object(), "j1")[0] == "done"
    assert _remote_job_state("local", object(), "j1") == ("done", "local")
    with pytest.raises(SystemExit) as exc:   # a user-fixable problem: exit_with_error, code 2
        _remote_job_state("MQSS_LRZ", object(), "j1", backend_name=None)
    assert exc.value.code == 2


@pytest.mark.parametrize("status, state, report", [
    ("JobStatus.DONE", "done", "DONE"), ("COMPLETED", "done", "DONE"),
    ("JobStatus.ERROR", "failed", "ERROR"), ("FAILED", "failed", "ERROR"),
    ("JobStatus.CANCELLED", "failed", "CANCELLED"),          # a cancelled job counts as failed
    ("QUEUED", "pending", "PENDING"), ("RUNNING", "pending", "PENDING"),
])
def test_one_reading_of_job_status_texts(status, state, report):
    """Collection, the IBM health-check wait and the status report read status texts the same way."""
    from source.core.qpu.providers import job_state_of_status
    from source.core.qpu.job_status_check import classify
    assert job_state_of_status(status) == state
    assert classify(status) == report


# ── The chunks a submission and its health checks cover ───────────────────────
def test_chunks_to_submit_are_the_ones_without_a_submitted_record_gaps_included(tmp_path):
    from source.core.records.job_records import chunks_without_submission
    for nc, status in {0: "complete", 1: "failed", 2: "submitted", 3: "failed", 5: "complete"}.items():
        _write(tmp_path / "job_records" / f"job_record_chunk_{nc}.json", {"chunk_id": nc, "status": status})
    # failed jobs' chunks (1, 3) and chunks without a record (4, 6, 7) — with gaps
    assert chunks_without_submission(tmp_path, 8) == [1, 3, 4, 6, 7]
    assert chunks_without_submission(tmp_path, 4) == [1, 3]
    assert chunks_without_submission(tmp_path / "no_such_run", 3) == [0, 1, 2]


def test_accompanying_health_check_needs_its_chunks():
    with pytest.raises(AssertionError, match="chunks"):
        HC.submit_health_check(run_folder="tests/sim_test14_never_created", backend=None, service_name="local",
                              accompanying=True)
    assert not (PROJECT_ROOT / "datafiles" / "tests" / "sim_test14_never_created").exists()


def test_health_check_chunks_new_records_and_old_ones():
    from source.core.records.circuit_check import health_check_chunks, health_check_in_scope
    # new records: the chunks passed by the submit script, whatever else is on disk
    assert health_check_chunks({"chunks": [12, 13, 30], "after_submission": False}) == [12, 13, 30]
    assert health_check_chunks({"chunks": [12, 13, 30], "after_submission": True}) == [12, 13, 30]
    assert health_check_in_scope({"chunks": [12, 30]}, 13) and not health_check_in_scope({"chunks": [12, 30]}, 12)
    # records written before `chunks` existed: jobs, or the first chunk after prior_chunks
    assert health_check_chunks({"jobs": [[4, 5], [6]], "prior_chunks": []}) == [4, 5, 6]
    assert health_check_chunks({"jobs": [], "prior_chunks": [0, 1, 2]}) == [3]
    assert health_check_chunks({"jobs": [], "prior_chunks": []}) == [0]
    assert health_check_chunks({}) is None


# ── Health check plot of an incomplete run ────────────────────────────────────
def test_health_check_plot_of_an_incomplete_run():
    folder = run_name_for_test("test14")
    run, plots = PROJECT_ROOT / "datafiles" / folder, PROJECT_ROOT / run_plots_dir(folder)
    shutil.rmtree(run, ignore_errors=True)
    try:
        write_run_info({"shots": 10, "samples": 4, "T_list": [0.2], "dt": 0.2, "chunk_size": 6}, run, lnn=False)
        _write(run / "raw_results" / "results_counts.json", _counts([0, 1, 3]))
        t = lambda m: f"2026-09-29T10:{m:02d}:00Z"
        na = {"circuit_check": {"status": "not_available"}}
        for nc, (job, done) in enumerate([("A", True), ("A", True), ("B", False), ("C", True)]):
            _write(run / "job_records" / f"job_record_chunk_{nc}.json",
                   {"chunk_id": nc, "job_id": job, "status": "complete" if done else "submitted", **na,
                    "submitted_at": t(nc), **({"scheduled_at": t(10 + nc), "finished_at": t(20 + nc)} if done else {})})
        _write(run / "health_checks" / "health_check_1.json",
               {"number": 1, "schema": 2, "status": "complete", "P011": 0.8, "submitted_at": t(0),
                "scheduled_at": t(5), "finished_at": t(6), **na})
        _write(run / "health_checks" / "health_check_2.json",
               {"number": 2, "schema": 2, "status": "submitted", "submitted_at": t(30)})
        env = {**os.environ, "QPU_RUN_FOLDER": folder, "QPU_N_SAMPLES": "3"}
        r = subprocess.run([venv_python("local_sim.venv", "test_14 (plot_health_checks.py)"), str(PROJECT_ROOT / "source" / "analysis_scripts" / "plot_health_checks.py")], cwd=PROJECT_ROOT,
                           env=env, capture_output=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0, r.stdout + r.stderr
        assert (plots / "health_checks+nsamples-3.png").exists()
        out = json.loads((run / "processed_results" / "health_checks+nsamples-3.json").read_text(encoding="utf-8"))
        assert out["uncollected_health_checks"] == [2]
        assert out["completeness"]["uncollected_chunks"] == [2] and out["completeness"]["n_chunks_analyzed"] == 3
        assert [j["end"] is None for j in out["jobs"]] == [False, True, False]   # job B (chunk 2) not collected
    finally:
        shutil.rmtree(run, ignore_errors=True)
        shutil.rmtree(plots, ignore_errors=True)
