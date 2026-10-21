"""
test_12_aqt_shot_split.py
=========================
Launcher for the AQT shot-split tests, which need aqt.venv (Qiskit 1.4.6, the version AQT runs on),
not local_sim.venv: it runs source/tests/aqt_venv_test_12_shot_split.py with venvs/aqt.venv's Python and
fails with instructions when aqt.venv (or pytest inside it) is missing. See that file for what is tested.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import subprocess
from pathlib import Path

from source.core.runner_support.venvs import venv_python

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_12_aqt_shot_split_in_aqt_venv():
    python = venv_python("aqt.venv", "test_12 (AQT shot split)")
    # --noconftest: source/tests/conftest.py is for local_sim.venv (matplotlib, test cache root).
    # PYTEST_DISABLE_PLUGIN_AUTOLOAD: qiskit-aqt-provider registers a pytest plugin that needs pytest-httpx.
    proc = subprocess.run([python, "-m", "pytest", "--noconftest", "-p", "no:cacheprovider", "-q",
                           str(PROJECT_ROOT / "source" / "tests" / "aqt_venv_test_12_shot_split.py")],
                          cwd=PROJECT_ROOT, env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}, capture_output=True, encoding="utf-8", errors="replace")
    out = proc.stdout + proc.stderr
    if "No module named pytest" in out:
        raise RuntimeError(
            f"pytest is not installed in venvs/aqt.venv. Install it there:\n"
            f"    {python} -m pip install pytest")
    assert proc.returncode == 0, f"tests in aqt.venv failed (exit {proc.returncode}):\n{out[-6000:]}"
