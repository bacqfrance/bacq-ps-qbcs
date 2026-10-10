"""
test_07_full_pipeline_local_sim.py
===================================
End-to-end integration test for the local-simulator pipeline: builds +
submits + analyzes (mirroring benchmark_noiseless_all_to_all.py), checks every output file
lands where §8 documents,
verifies source/analysis_scripts/run_post_process.py's local-sim post-processing chain
(merge_and_analyze + the four downstream scripts) runs cleanly, and checks
that AWAE (shot-noise error) is small and decays with more shots for all six
roles (bare, NEC, RC, NEC+RC, cRC, NEC+cRC) — run at two shot counts on the
SAME pre-built circuits (see qbcs.reset_for_resubmit()). A ratio far below the expected
sqrt(shots) scaling would indicate coherent residual noise in that role's
correction rather than ordinary shot noise.

All six roles are safe to assert here, unlike test_01: RC/cRC's flakiness
there came from Qiskit and myQLM each drawing their OWN independent random
twirl in separate pipeline calls. Here, qbcs.build() (with a fixed rng_seed)
runs once and both shot-count runs resubmit the SAME twirled circuits via
reset_for_resubmit() — no cross-run twirl-sampling variance to confound the
shot-noise-scaling check.

The analysis side runs through source/analysis_scripts/run_post_process.py's analyze_run()
exactly as for a real-QPU run: the run's own settings (shots, samples, depths,
topology) are read back from the run folder's run_info.json, so summary JSON
"shots_per_circuit" and plot filenames carry each run's real shot count. Also
covered: the merge+analyze branch (derived files deleted, then regenerated from
the raw counts must reproduce the same AWAE) and subset analysis (n_samples).
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import json
import pytest
import numpy as np
from pathlib import Path
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from source.core.qbcs import bacq_ps_qbcs
from source.core.health_check import submit_health_check
from source.core.records.health_check_records import load_health_checks
from source.analysis_scripts.run_post_process import merge_and_analyze, analyze_run
from source.core.records.run_info import load_run_info
from source.core.common import get_circuit_folder
from source.tests.testing_functions import rm_glob, run_name_for_test, TEST_CIRCUITS_ROOT
from source.core.records.run_names import plots_dir as run_plots_dir
from source.core.records.run_names import run_summary_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LIST_TIMES = [0.4, 0.6]   # same reduced depths as test_00/test_01
DT         = 0.2
SAMPLE_SIZE = 16          # must be a multiple of 2**nqubits=8 (source/core/common.py)
SHOTS_A, SHOTS_B = 250, 4000
ROLES = ["bare", "NEC", "RC", "NEC+RC", "cRC", "NEC+cRC"]   # 16x apart -> expected AWAE ratio ~ sqrt(16) = 4


def _run_one_shot_value(qbcs, nshots, run_folder):
    qbcs.CONFIG["shots"] = nshots
    # Health checks accompanying the run (same pattern as benchmark_noiseless_all_to_all.py): one before, one after
    # the submission, both given the chunks the submission runs
    todo = qbcs.chunks_to_submit(run_folder)
    assert todo == list(range(SAMPLE_SIZE)), todo   # a fresh run folder: every chunk
    submit_health_check(run_folder=run_folder, backend=AerSamplerV2(seed=0), nshots=100,
                        service_name="local", accompanying=True, chunks=todo)
    qbcs.submit(
        run_folder = run_folder,
        lang       = "qiskit",
        service    = "local",
        # Fixed seed: without it, AerSamplerV2's shot sampling is a fresh random
        # draw every run, so the AWAE-ratio check below is only *usually* right —
        # it failed once in 5 runs on an unseeded sampler (ratio 1.78 vs the
        # expected ~4, well within ordinary shot-noise spread but under the
        # tolerance floor). Seeding makes both shot-count runs deterministic.
        backend    = AerSamplerV2(seed=0),
        chunks     = todo,
    )
    assert qbcs.chunks_to_submit(run_folder) == [], "chunks left after the submission"
    submit_health_check(run_folder=run_folder, backend=AerSamplerV2(seed=0), nshots=100,
                        service_name="local", accompanying=True, after_submission=True, chunks=todo)
    qbcs.analyze_results()
    qbcs.reset_for_resubmit()


def _assert_local_run_outputs(run_path, do_spam=True):
    """§8's output tree, restricted to what a local run driven through bacq_ps_qbcs
    directly produces — verified against a real sim run's directory tree. It has job
    records like a real-QPU run, one 'complete' record per chunk (no chunk_times_UTC_time.json
    any more); submit.log is written by the scripts, not by submit(), so it is not here."""
    for rel in [
        "raw_results/results_counts.json",
        "raw_results/results_counts_merged.json",
        "raw_results/results_fidelity.json",
        "raw_results/results_fidelity_merged.json",
        "processed_results/run_data.json",
        "timestamps_local_time.csv",
        "timestamps_utc.csv",
        "transpiled_circuits/health_check_transpiled.qpy",
        "job_records/job_record_chunk_0.json",
        "health_checks/health_check_1.json",
        "health_checks/health_check_2.json",
    ] + ([
        "raw_results/spam_calibration_counts_per_chunk.json",
        "processed_results/run_data_spam.json",
        "processed_results/spam_confusion_matrix.npy",
        "processed_results/spam_confusion_matrix_inv.npy",
    ] if do_spam else []):
        path = run_path / rel
        assert path.exists(), f"missing expected output: {path}"
    assert (run_path / "processed_results" / "run_data_spam.json").exists() == do_spam
    assert not (run_path / "chunk_times_UTC_time.json").exists()
    for p in (run_path / "job_records").glob("job_record_chunk_*.json"):
        rec = json.loads(p.read_text(encoding="utf-8"))
        assert (rec["status"], rec["service_name"]) == ("complete", "local"), (p.name, rec["status"])


def _load_run_data(run_path, name="run_data.json"):
    with open(run_path / "processed_results" / name, encoding="utf-8") as f:
        return json.load(f)


def _assert_post_process_outputs(backend_name, run_path, nshots, do_spam=True):
    # submit() recorded what this run was
    info = load_run_info(backend_name)
    assert info.source == "manifest", info
    assert (info.nshots, info.sample_size, info.lnn, info.do_spam) == (nshots, SAMPLE_SIZE, False, do_spam), info
    assert np.allclose(info.list_times, LIST_TIMES) and info.dt == DT and info.chunk_size == len(LIST_TIMES) * 6

    # Already analyzed by the run itself -> recognized as done, nothing to redo
    ok, n_samples = merge_and_analyze(backend_name)
    assert ok, f"{backend_name}: merge_and_analyze() did not recognize the local run as already analyzed"
    assert n_samples is None, f"{backend_name}: unexpected n_samples override {n_samples!r}"

    # Drop everything derived, keep only the raw per-chunk counts: the merge + analyze branch must rebuild it
    data_names = ["run_data.json"] + (["run_data_spam.json"] if do_spam else [])
    before = {n: _load_run_data(run_path, n) for n in data_names}
    for rel in ["raw_results/results_counts_merged.json", "raw_results/results_fidelity_merged.json",
                *(f"processed_results/{n}" for n in data_names)]:
        (run_path / rel).unlink()
    assert analyze_run(backend_name)[0], f"analyze_run failed for {backend_name}"
    for name, old in before.items():
        new = _load_run_data(run_path, name)
        assert new["nchunks"] == SAMPLE_SIZE
        for nt, depth in old["depths"].items():
            for role in ROLES:
                assert new["depths"][nt][role]["AWAE"] == depth[role]["AWAE"], (name, nt, role)

    summary = json.loads((run_path / "processed_results" / run_summary_name()).read_text(encoding="utf-8"))
    assert summary["shots_per_circuit"] == nshots
    assert summary["n_samples_analyzed"] == SAMPLE_SIZE

    # Timing (source/core/records/job_timing.py): a local run has CPU time only — build, conversion, submission
    # (the simulation) from the UTC log — and no QPU or queue time; the fractions add up to 1.
    assert summary["timing_definitions_version"] == 3
    assert summary["QPU_time_s"] == 0 and summary["Queue_time_s"] == 0
    cpu = summary["timing_breakdown"]["cpu"]
    assert cpu["submit_s"] > 0 and cpu["run_timestamp_source"] == "utc", cpu
    assert summary["CPU_time_s"] == pytest.approx(cpu["total_s"], abs=1e-3)
    # collection: every chunk's results handling is logged (chunk{nc}_collect_*), plus the health checks
    col = summary["timing_breakdown"]["collect"]
    assert col["n_chunks_timed"] == SAMPLE_SIZE and col["n_chunks_without"] == 0, col
    assert summary["Collect_time_s"] == pytest.approx(col["total_s"], abs=1e-3) and col["total_s"] > 0
    assert summary["T_generation_submission_execution_s"] == pytest.approx(
        summary["CPU_time_s"] + summary["Collect_time_s"], abs=1e-3)
    assert summary["FT_clas"] + summary["FT_quant"] + summary["FT_queue"] + summary["FT_collect"] \
        == pytest.approx(1.0, abs=1e-3)
    assert summary["completeness"]["complete"] and summary["completeness"]["uncollected_chunks"] == []

    # Health checks: PRE and POST, both recording the chunks the submission ran (every chunk of the run)
    hcs = load_health_checks(run_path)
    assert [c["number"] for c in hcs] == [1, 2], hcs
    assert all(c["verdict"] == "functional" and c["P011"] > 0.9 for c in hcs)
    assert [c["after_submission"] for c in hcs] == [False, True]
    assert all(c["chunks"] == list(range(SAMPLE_SIZE)) for c in hcs), [c["chunks"] for c in hcs]
    assert all("jobs" not in c and "prior_chunks" not in c for c in hcs)
    # the health check circuit is transpiled once and the saved copy reused
    assert [c["isa_circuit_file"] for c in hcs] == ["health_check_transpiled.qpy"] * 2
    assert hcs[0]["timing"]["transpile_s"] is not None and hcs[1]["timing"]["transpile_reused"]
    assert all(c["circuit_check"]["status"] == "not_available" for c in hcs)
    assert (run_path / "processed_results" / "health_checks.json").exists()

    # Plot PNGs — filenames carry this run's real shot count
    plots_dir = PROJECT_ROOT / run_plots_dir(backend_name)   # plots are grouped per run folder
    for fname in [
        f"obs_lnnFalse_ns{nshots}_no_spam.png",
        f"max_error_lnnFalse_ns{nshots}.png",
        "awae_no_spam.png",
        "health_checks.png",
    ] + ([f"obs_lnnFalse_ns{nshots}_spam.png", "awae_spam.png", "awae_best_ems_spam_no_spam_comparison.png"]
         if do_spam else []):
        path = plots_dir / fname
        assert path.exists(), f"missing expected plot: {path}"
    if not do_spam:   # a run without SPAM has no SPAM figures
        for fname in [f"obs_lnnFalse_ns{nshots}_spam.png", "awae_spam.png", "awae_best_ems_spam_no_spam_comparison.png"]:
            assert not (plots_dir / fname).exists(), f"unexpected SPAM plot in a no-SPAM run: {fname}"


def _assert_subset_analysis(backend_name, run_path, nshots, n):
    """Analyzing only the first n chunks writes *+nsamples-{n} files next to — never over — the full run's."""
    full_before = _load_run_data(run_path)
    assert analyze_run(backend_name, n_samples_override=n)[0], f"subset analyze_run failed for {backend_name}"
    subset = _load_run_data(run_path, f"run_data+nsamples-{n}.json")
    assert subset["n_samples_analyzed"] == n
    assert _load_run_data(run_path) == full_before, "subset analysis modified the full run's run_data.json"
    summary = json.loads((run_path / "processed_results" / run_summary_name(n)).read_text(encoding="utf-8"))
    assert summary["n_samples_analyzed"] == n and summary["shots_per_circuit"] == nshots
    plots_dir = PROJECT_ROOT / run_plots_dir(backend_name)
    for fname in [f"obs_lnnFalse_ns{nshots}_no_spam+nsamples-{n}.png", f"max_error_lnnFalse_ns{nshots}+nsamples-{n}.png"]:
        assert (plots_dir / fname).exists(), f"missing expected plot: {plots_dir / fname}"

    # a subset larger than what was collected is refused, not silently clipped
    assert not analyze_run(backend_name, n_samples_override=SAMPLE_SIZE + 8)[0]   # (ok, run_folder, n_samples)


def _load_awae(run_path, role, nt):
    with open(run_path / "processed_results" / "run_data.json", encoding="utf-8") as f:
        run_data = json.load(f)
    return run_data["depths"][str(nt)][role]["AWAE"]


def test_07_full_pipeline_local_sim():
    rm_glob("datafiles/tests/test07+*")
    rm_glob(run_plots_dir("tests/test07+").as_posix() + "*")
    rm_glob(f"{TEST_CIRCUITS_ROOT}/all_to_all_Nt2-3")   # the no-SPAM test builds its own cache of the same name

    folder_a = run_name_for_test(f"test07+nshots-{SHOTS_A}")
    folder_b = run_name_for_test(f"test07+nshots-{SHOTS_B}")

    qbcs = bacq_ps_qbcs(
        sample_size                 = SAMPLE_SIZE,
        list_circuit_times          = LIST_TIMES,
        error_mitigation_strategies = ["SPAM", "NEC", "RC+NEC", "cRC+NEC"],
        do_SPAM                     = True,
        nshots                      = SHOTS_A,   # placeholder — overridden per run below
        dt                          = DT,
        rng_seed                    = 0,
    )
    qbcs.build(circuit_transpilation_LNN=False)
    circuit_folder = PROJECT_ROOT / get_circuit_folder(qbcs.CONFIG, top_folder=True)   # this config's own cache (datafiles/tests/circuits/all_to_all_Nt2-3)
    assert (circuit_folder / "chunks_qiskit_v13_qpy" / "spam_qiskit_v13.qpy").exists(), (   # all_to_all: QPY version 13
        "build() didn't cache the SPAM circuits (convert_spam_to_qiskit())")

    for nshots, folder in [(SHOTS_A, folder_a), (SHOTS_B, folder_b)]:
        _run_one_shot_value(qbcs, nshots, folder)
        run_path = PROJECT_ROOT / "datafiles" / folder
        _assert_local_run_outputs(run_path)
        _assert_post_process_outputs(folder, run_path, nshots)

    _assert_subset_analysis(folder_a, PROJECT_ROOT / "datafiles" / folder_a, SHOTS_A, n=8)

    nt = int(round(LIST_TIMES[0] / DT))
    run_path_a = PROJECT_ROOT / "datafiles" / folder_a
    run_path_b = PROJECT_ROOT / "datafiles" / folder_b

    for role in ROLES:
        awae_a = _load_awae(run_path_a, role, nt)
        awae_b = _load_awae(run_path_b, role, nt)

        assert awae_a < 0.5, f"{role} AWAE unexpectedly large at {SHOTS_A} shots: {awae_a}"
        assert awae_b < 0.5, f"{role} AWAE unexpectedly large at {SHOTS_B} shots: {awae_b}"
        assert awae_b < awae_a, (
            f"{role} AWAE did not decrease with more shots: "
            f"{awae_a} (n={SHOTS_A}) -> {awae_b} (n={SHOTS_B})"
        )

        # If RC/cRC left behind coherent residual noise (a systematic bias from
        # the twirl/correction itself, not shot noise), AWAE would plateau above
        # some floor instead of following sqrt(shots) — the ratio would fall well
        # short of ~4 regardless of how the shot count changes. Since both runs
        # here share the SAME built/twirled circuits (qbcs.build() called once,
        # only resubmitted via reset_for_resubmit() — see below), there's no
        # cross-run twirl-sampling variance to confound this the way there was
        # in test_01 (which compared two *independently* twirled pipelines).
        ratio    = awae_a / awae_b
        expected = np.sqrt(SHOTS_B / SHOTS_A)   # = 4
        assert 2 < ratio < 10, (
            f"{role} AWAE ratio {ratio:.2f} far from the expected shot-noise "
            f"scaling ~{expected:.1f} (sqrt({SHOTS_B}/{SHOTS_A})) — "
            f"AWAE(n={SHOTS_A})={awae_a:.3e}, AWAE(n={SHOTS_B})={awae_b:.3e} — "
            f"possible coherent residual noise in {role} rather than pure shot noise"
        )


def test_07_full_pipeline_local_sim_no_spam():
    """The same pipeline for a run with do_SPAM=False: no SPAM files or figures, everything else as above."""
    rm_glob("datafiles/tests/test07nospam*", run_plots_dir("tests/test07nospam").as_posix() + "*", f"{TEST_CIRCUITS_ROOT}/all_to_all_Nt2-3")
    folder = run_name_for_test("test07nospam")

    qbcs = bacq_ps_qbcs(
        sample_size                 = SAMPLE_SIZE,
        list_circuit_times          = LIST_TIMES,
        error_mitigation_strategies = ["NEC", "RC+NEC", "cRC+NEC"],
        do_SPAM                     = False,
        nshots                      = SHOTS_A,
        dt                          = DT,
        rng_seed                    = 0,
    )
    qbcs.build(circuit_transpilation_LNN=False)
    _run_one_shot_value(qbcs, SHOTS_A, folder)
    run_path = PROJECT_ROOT / "datafiles" / folder
    _assert_local_run_outputs(run_path, do_spam=False)
    _assert_post_process_outputs(folder, run_path, SHOTS_A, do_spam=False)
    _assert_subset_analysis(folder, run_path, SHOTS_A, n=8)


if __name__ == "__main__":
    test_07_full_pipeline_local_sim()
