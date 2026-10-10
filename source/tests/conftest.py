"""
conftest.py — the pytest session setup for source/tests/: project root on sys.path and as the
working directory, the non-interactive matplotlib backend, and the tests' own circuit caches.
"""
import os
import sys
# pytest may be started from any directory: put the project root on sys.path (for `source.`
# imports) and make it the working directory (datafiles/ paths are relative to it) before any test
# module is imported.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import matplotlib
# Force the non-interactive Agg backend before anything below (or any test
# module) can import pyplot and let matplotlib pick a default. The pipeline
# code under test only ever needs to save PNGs, never display a window, so
# an interactive backend (TkAgg by default on most installs) is both
# unnecessary here and a source of environment-dependent breakage — e.g. a
# broken/mismatched Tcl-Tk install raises _tkinter.TclError the first time a
# test triggers plt.subplots(), on an otherwise fully passing run.
matplotlib.use("Agg")

import json
import pytest
from source.tests.testing_functions import TEST_CIRCUITS_ROOT, rm_test_caches
from source.core.common import CIRCUITS_ROOT_ENV
from source.core.runner_support.venvs import venv_python

# The suite runs in local_sim.venv (test_12 runs in aqt.venv, through its launcher): stop early, with
# instructions, when that venv was never created.
try:
    venv_python("local_sim.venv", "the test suite")
except SystemExit:   # venv_python() ends through exit_with_error(), which already printed the ❌ line
    raise pytest.UsageError("venvs/local_sim.venv is missing: create it as in USER_MANUAL §3")

# ── The tests' own circuit caches ──────────────────────────────────────────────
# Every circuit cache lives under common.circuits_root(). Setting BACQ_CIRCUITS_ROOT here — at
# import, before any test module is imported, and inherited by every script a test launches as a
# subprocess — sends all the caches the tests build to datafiles/tests/circuits/. The caches real
# runs use (datafiles/circuits/<topology>_Nt<steps>/) are never read, moved or deleted, so pytest
# can run while a benchmark script is running. Only this file sets it for the suite (test_19 sets it with monkeypatch to test circuits_root() itself).
os.environ[CIRCUITS_ROOT_ENV] = TEST_CIRCUITS_ROOT

# What datafiles/circuits/ held when the session started: test_19 checks that the tests added
# nothing there but the tests/ subfolder (a cache path built without circuits_root() would).
# Kept in the environment because a test importing this file again would re-run it.
_REAL_CIRCUITS = os.path.join("datafiles", "circuits")
os.environ.setdefault("BACQ_TEST_CIRCUITS_AT_START", json.dumps(
    sorted(os.listdir(_REAL_CIRCUITS)) if os.path.isdir(_REAL_CIRCUITS) else []))


@pytest.fixture(scope="session", autouse=True)
def _fresh_test_circuit_caches():
    """Start every session from empty test caches. They are left in place afterwards, for debugging."""
    rm_test_caches()
    yield
