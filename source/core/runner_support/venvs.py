"""
venvs.py
The per-backend virtual environments under <project root>/venvs/ (USER_MANUAL §3). A script that launches another one as a subprocess takes the child's
interpreter from here, never from sys.executable or PATH, which can silently pick
the wrong environment. Works on Windows (Scripts/python.exe) and Linux/macOS
(bin/python).
"""

import os

from source.core.common import PROJECT_ROOT, exit_with_error


def venv_python_path(venv_name, os_name=None):
    """Where venvs/<venv_name>'s interpreter lives on this OS (os_name: os.name by default)."""
    os_name = os_name or os.name
    rel = ("Scripts", "python.exe") if os_name == "nt" else ("bin", "python")
    return PROJECT_ROOT.joinpath("venvs", venv_name, *rel)


def venv_python(venv_name, needed_for, packages_hint=""):
    """Path (str) of venvs/<venv_name>'s Python interpreter. Ends the script
    (exit_with_error), explaining how to create the venv, when it does not exist."""
    exe = venv_python_path(venv_name)
    if not exe.exists():
        exit_with_error(
            f"{needed_for} must run in venvs/{venv_name}, but venvs/{venv_name} has no interpreter "
            f"({exe.relative_to(PROJECT_ROOT).as_posix()}).\n"
            f"Create it first: see USER_MANUAL.md §3 (local_sim.venv) or the readme of the QPU (§1; e.g. "
            f"source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/readme_aqt.md for aqt.venv), "
            f"then install the requirements file named there into it.{packages_hint}")
    return str(exe)
