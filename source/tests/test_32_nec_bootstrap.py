"""
test_32_nec_bootstrap.py
========================
Unit tests (synthetic data, no simulator, no QPU, no circuit cache) of the NEC attenuation and bootstrap in
source/core/analysis/mitigation.py and of the strict accuracy depth in derived_quantities.py:

  - sign_matrix() bit convention and nec_attenuation_samples() (zero noise gives F = f = 1);
  - bootstrap_weights(): reproducible with a seed, stratification keeps the input-state counts;
  - floor_attenuation() / attenuation_floor(): never NaN, 0 becomes +floor;
  - compute_mitigated_expectation_values() on a tiny synthetic run: a zero attenuation gives a finite large value,
    the "global" switch reproduces mean(obs) / mean(F), per-observable differs, fewer than 5 samples per input
    state means no bootstrap, zero noise gives zero-width intervals;
  - compute_derived_quantities(): N_acc_strict chosen by the upper bound, no bootstrap -> None fields, N_acc_strict = 0;
    N_acc_bare over bare and bare+SPAM only (never a mitigated scheme), the depth-1 values at 0, no bootstrap needed.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import json

import numpy as np
import pytest

import source.core.analysis.mitigation as MIT
from source.core.analysis.derived_quantities import compute_derived_quantities
from source.core.circuits.building import EXACT_OBS_LIST


# ── helpers ──────────────────────────────────────────────────────────────────────
def test_sign_matrix_bit_convention():
    S = MIT.sign_matrix(3)
    assert S.shape == (8, 7) and list(EXACT_OBS_LIST) == ["X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"]
    assert np.all(S[0] == 1)                                         # "000": every observable +1
    # "100": character 0 is qubit 0 (X): the observables with X flip
    assert list(S[int("100", 2)]) == [-1, 1, 1, -1, -1, 1, -1]
    assert list(S[int("001", 2)]) == [1, 1, -1, 1, -1, -1, -1]       # qubit 2 (Z)
    assert list(S[int("111", 2)]) == [-1, -1, -1, 1, 1, 1, -1]


def test_zero_noise_attenuation_is_one():
    S = MIT.sign_matrix(3)
    targets = np.array([0, 5, 7, 2, 3])
    C = np.zeros((5, 8))
    C[np.arange(5), targets] = 100
    F, f = MIT.nec_attenuation_samples(C, targets, S)
    assert np.allclose(F, 1) and f.shape == (5, 7) and np.allclose(f, 1)


def test_bootstrap_weights_seed_and_stratification():
    strata = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 3])
    W = MIT.bootstrap_weights(strata, n_resamples=50, seed=7)
    assert W.shape == (50, 10)
    assert np.array_equal(W, MIT.bootstrap_weights(strata, n_resamples=50, seed=7))
    assert not np.array_equal(W, MIT.bootstrap_weights(strata, n_resamples=50, seed=8))
    for s in np.unique(strata):
        assert np.all(W[:, strata == s].sum(axis=1) == np.sum(strata == s))   # the input-state counts are kept
    assert np.all(W.sum(axis=1) == 10) and np.all(W >= 0)


def test_floor_attenuation():
    floor = 0.1
    x, hit = MIT.floor_attenuation([0.0, -0.01, 0.01, 0.5, -0.5, 0.1], floor)
    assert list(x) == [0.1, -0.1, 0.1, 0.5, -0.5, 0.1]
    assert list(hit) == [True, True, True, False, False, False]
    assert not np.any(np.isnan(x))
    assert MIT.attenuation_floor(480, 10) == pytest.approx(2 / np.sqrt(4800))
    assert MIT.attenuation_floor(480, 0) == pytest.approx(2 / np.sqrt(480 * MIT.UNLIMITED_SHOTS))


# ── a tiny synthetic run ─────────────────────────────────────────────────────────
def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _run(tmp_path, monkeypatch, n=10, nec_counts=None, raw_counts=None, **config):
    """One depth (Nt = 1), roles raw / raw_nec, n samples, one NEC input state (target "000")."""
    raw_counts = raw_counts or (lambda c: {"000": 10 - c % 3, "100": c % 3})
    nec_counts = nec_counts or (lambda c: {"000": 90 + c, "001": 10 - c})
    _write(tmp_path / "raw_results" / "results_counts_merged.json",
           {"chunk_0": {"job_Nt_01_raw_ems0": {f"rc_circuit_{c}": raw_counts(c) for c in range(n)}}})
    _write(tmp_path / "raw_results" / "results_fidelity_merged.json",
           {"chunk_0": {"job_Nt_01_raw_nec_ems1": {f"nec_circuit_{c}": nec_counts(c) for c in range(n)}}})
    monkeypatch.setattr(MIT, "load_classical_nec_map", lambda cfg, label: {"000": "000"})
    cfg = {"folder_path": str(tmp_path), "samples": n, "nsites": 3, "T_list": [1], "dt": 1, "shots": 100,
           "nec_measurement_counts": True, "nec_randomize_input_state": False}
    cfg.update(config)
    return MIT.compute_mitigated_expectation_values(cfg, roles=["raw", "raw_nec"])


def _obs_and_F(n=10):
    S = MIT.sign_matrix(3)
    raw = np.zeros((n, 8))
    for c in range(n):
        raw[c, 0], raw[c, 4] = 10 - c % 3, c % 3
    obs = (raw @ S) / 10
    F = np.array([(90 + c) / 100 for c in range(n)])
    return obs, F


def test_global_switch_is_the_old_global_formula(tmp_path, monkeypatch):
    res, info = _run(tmp_path, monkeypatch, nec_correction="global")
    obs, F = _obs_and_F()
    assert np.allclose(res[1]["raw"]["mean"], obs.mean(axis=0))
    assert np.allclose(res[1]["raw_nec"]["mean"], obs.mean(axis=0) / F.mean())
    assert info["nec_correction"] == "global" and info["bootstrap_done"]


def test_per_observable_differs_by_weight(tmp_path, monkeypatch):
    res, info = _run(tmp_path, monkeypatch)   # per_observable is the default
    obs, F = _obs_and_F()
    nec = res[1]["raw_nec"]["mean"]
    assert info["nec_correction"] == "per_observable"
    # target "000": observables without qubit 2 (X, Y, XY) see no error (f = 1), those with Z see 2F - 1
    expected = obs.mean(axis=0) / np.array([1, 1, 2 * F.mean() - 1, 1, 2 * F.mean() - 1, 2 * F.mean() - 1, 2 * F.mean() - 1])
    assert np.allclose(nec, expected)
    assert not np.allclose(nec, obs.mean(axis=0) / F.mean())


def test_zero_attenuation_gives_a_finite_large_value(tmp_path, monkeypatch):
    res, info = _run(tmp_path, monkeypatch, nec_correction="global", nec_counts=lambda c: {"111": 100})
    out = res[1]["raw_nec"]
    floor = MIT.attenuation_floor(10, 100)
    assert np.all(np.isfinite(out["mean"])) and np.all(np.isfinite(out["boot"]))
    assert np.allclose(out["mean"], res[1]["raw"]["mean"] / floor)
    assert out["floor_hits"] == 7 and out["floor_hit_rate_bootstrap"] == 1.0 and np.allclose(out["attenuation"], 0)


def test_few_samples_per_input_state_means_no_bootstrap(tmp_path, monkeypatch):
    res, info = _run(tmp_path, monkeypatch, n=4)
    assert info["bootstrap_done"] is False and "fewer than 5" in info["bootstrap_skipped_because"]
    assert res[1]["raw"]["boot"] is None and res[1]["raw_nec"]["boot"] is None
    assert res[1]["raw_nec"]["floor_hit_rate_bootstrap"] is None


def test_five_samples_give_a_bootstrap(tmp_path, monkeypatch):
    res, info = _run(tmp_path, monkeypatch, n=5)
    assert info["bootstrap_done"] is True and "bootstrap_skipped_because" not in info
    assert res[1]["raw"]["boot"].shape == (1000, 7) and res[1]["raw_nec"]["boot"].shape == (1000, 7)


def test_zero_noise_gives_zero_width_intervals(tmp_path, monkeypatch):
    res, info = _run(tmp_path, monkeypatch, nec_counts=lambda c: {"000": 100}, raw_counts=lambda c: {"000": 10})
    out = res[1]["raw_nec"]
    assert np.allclose(out["attenuation"], 1) and np.allclose(out["mean"], 1) and out["floor_hits"] == 0
    assert np.allclose(out["boot"], out["mean"]) and np.allclose(MIT.ci95(out["boot"]), 1)
    assert np.allclose(res[1]["raw"]["boot"], 1)


# ── N_acc_strict ─────────────────────────────────────────────────────────────────
EMS = ["bare", "NEC", "RC", "NEC+RC", "cRC", "NEC+cRC"]


def _derived(cum, upper, bootstrap_done=True):
    ci = {e: [[0.0, u] for u in upper.get(e, [0.9] * 3)] for e in EMS}
    cumul = {e: cum.get(e, [0.5] * 3) for e in EMS}
    return compute_derived_quantities(cumul, cumul, 1.0, 1.0, Nt_vals=[1, 2, 3], lnn=False,
                                      awae_cumul_ci95=ci if bootstrap_done else None, bootstrap_done=bootstrap_done)


def test_n_acc_strict_is_chosen_by_the_upper_bound():
    r = _derived({"NEC": [0.01, 0.02, 0.03]}, {"NEC": [0.04, 0.04, 0.2], "RC": [0.03, 0.045, 0.045]})
    assert r["N_acc"] == 3 and r["best_ems_Nacc"] == "NEC" and r["bootstrap_done"] is True
    assert r["N_acc_strict"] == 3 and r["best_ems_Nacc_strict"] == "RC"       # its own best scheme, by upper bound
    assert r["E_min_Nacc_strict"] == 0.5 and r["E_min_Nacc_strict_ci95"] == [0.0, 0.045]
    assert r["E_min_Nacc_ci95"] == [0.0, 0.2]
    assert r["N_cnot_acc_strict"] == 6 * 3


def test_no_bootstrap_means_no_strict_or_interval_fields():
    r = _derived({"NEC": [0.01, 0.02, 0.03]}, {}, bootstrap_done=False)
    assert r["N_acc"] == 3 and r["bootstrap_done"] is False
    for k, v in r.items():
        if k.endswith("_strict") or "_strict_" in k or k.endswith("_ci95"):
            assert v is None, k
    assert r["E_bare_Nacc_strict"] is None and r["E_bare_Nacc_ci95"] is None


def test_n_acc_strict_zero_gives_depth_one_values():
    r = _derived({"NEC": [0.01, 0.02, 0.03]}, {"NEC": [0.3, 0.3, 0.3]})
    assert r["N_acc"] == 3 and r["N_acc_strict"] == 0 and r["N_cnot_acc_strict"] == 0
    assert r["best_ems_Nacc_strict"] == "NEC" and r["E_min_Nacc_strict"] == 0.01      # depth 1
    assert r["E_bare_Nacc_strict"] == 0.5   # bare cumulative AWAE at depth 1


def test_n_acc_bare_counts_only_bare_schemes_and_needs_no_bootstrap():
    # a mitigated scheme stays below EPSILON at every depth, bare only up to depth 2; no bootstrap data at all
    r = _derived({"NEC": [0.01, 0.01, 0.01], "bare": [0.02, 0.03, 0.9]}, {}, bootstrap_done=False)
    assert r["N_acc"] == 3 and r["N_acc_bare"] == 2 and r["bootstrap_done"] is False and r["N_acc_strict"] is None
    assert r["best_ems_Nacc_bare"] == "bare" and r["E_min_Nacc_bare"] == 0.03
    # N_acc_bare = 0: the *_Nacc_bare fields hold the depth-1 values (bare cumulative AWAE 0.6 there)
    r = _derived({"NEC": [0.01, 0.01, 0.01], "bare": [0.6, 0.01, 0.01]}, {}, bootstrap_done=False)
    assert r["N_acc"] == 3 and r["N_acc_bare"] == 0 and r["N_cnot_acc_bare"] == 0
    assert r["best_ems_Nacc_bare"] == "bare" and r["E_min_Nacc_bare"] == 0.6
