"""
test_26_awae_helpers.py
=======================
Unit tests of the shared AWAE helpers in source/core/analysis/derived_quantities.py: cumulative_awae() reproduces the
Z-weighted cumulative AWAE the summary has always used (bit for bit) and agrees with the other forms the plot scripts
once used, EPSILON is the one accuracy threshold, and best_schemes() lists the best mitigation schemes of a summary.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import inspect

import numpy as np
import pytest

from source.core.analysis.derived_quantities import EPSILON, best_schemes, compute_derived_quantities, cumulative_awae


def _formula_a(Z, a):
    return np.array([(1 / np.sum(Z[:k+1])) * np.sum(Z[:k+1] * a[:k+1]) for k in range(len(Z))])


def _formula_b(Z, a):
    return np.array([np.sum(Z[:k+1] * a[:k+1]) / np.sum(Z[:k+1]) for k in range(len(Z))])


def _formula_c(Z, a):
    out, zw, zsum = [], 0.0, 0.0
    for x, w in zip(a, Z):
        zw, zsum = zw + w * x, zsum + w
        out.append(zw / zsum)
    return np.array(out)


@pytest.mark.parametrize("seed, n", [(0, 10), (1, 11), (2, 12), (3, 10)])
def test_cumulative_awae_matches_the_forms(seed, n):
    rng = np.random.default_rng(seed)
    Z, a = rng.uniform(0.1, 1.0, n), rng.uniform(0.0, 2.0, n)
    got = cumulative_awae(Z, a)
    assert np.array_equal(got, _formula_a(Z, a))
    assert np.allclose(got, _formula_b(Z, a), rtol=1e-12)
    assert np.allclose(got, _formula_c(Z, a), rtol=1e-12)


def test_cumulative_awae_nan_and_single_depth():
    Z = np.array([0.5, 0.4, 0.3])
    got = cumulative_awae(Z, np.array([0.1, np.nan, 0.2]))
    assert got[0] == pytest.approx(0.1) and np.isnan(got[1]) and np.isnan(got[2])
    assert np.array_equal(cumulative_awae([0.7], [0.25]), np.array([0.25]))


def test_epsilon():
    assert EPSILON == 0.05
    assert inspect.signature(compute_derived_quantities).parameters["epsilon"].default == EPSILON


@pytest.mark.parametrize("summary, expected", [
    ({"best_ems_Nmax": "NEC", "best_ems_Nacc": "NEC"}, [("N_max", "NEC")]),
    ({"best_ems_Nmax": "NEC", "best_ems_Nacc": "RC+SPAM"}, [("N_max", "NEC"), ("N_acc", "RC+SPAM")]),
    ({"best_ems_Nmax": "NEC"}, [("N_max", "NEC")]),
    ({"best_ems_Nmax": None, "best_ems_Nacc": "RC"}, [("N_acc", "RC")]),
    ({"best_ems_Nmax": "NEC", "best_ems_Nacc": None}, [("N_max", "NEC")]),
    ({"best_ems_Nmax": None, "best_ems_Nacc": None}, []),
    ({}, []),
])
def test_best_schemes(summary, expected):
    assert best_schemes(summary) == expected
