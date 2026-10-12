"""
test_23_no_absolute_paths_in_saved_files.py
===========================================
Nothing a run or an analysis saves may contain an absolute path (it leaks the user name and breaks when the
folder moves; see the project instructions). This test runs the mini local simulation
(source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_all_to_all.py with `mini`, in a test
folder, like test_08), analyzes it with source/analysis_scripts/run_post_process.py, then reads every text file
(json, log, csv, txt, md, ...) of THAT run folder and of its output under analysis/tests/ (binary files such as
qpy, png, pdf, npz and the circuit cache are skipped) and asserts that none contains
  - the project root (backslash, forward-slash and JSON-escaped spellings) or the home folder,
  - a Windows drive path (drive letter + colon + slash) or a UNC path (two backslashes + server),
  - an absolute POSIX path under /home, /Users, /mnt, /tmp, /var, /root or /opt.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import re
import subprocess
from pathlib import Path

from source.analysis_scripts.run_post_process import analyze_run
from source.core.runner_support.venvs import venv_python
from source.core.records.run_names import plots_dir as run_plots_dir, summaries_dir
from source.tests.testing_functions import rm_glob, run_name_for_test, TEST_CIRCUITS_ROOT

BS = chr(92)   # a backslash, spelled out so no string escaping can change it
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT       = PROJECT_ROOT / "source" / "benchmark_scripts" / "local_simulation_scripts" / "benchmark_noiseless_all_to_all.py"
FOLDER_LINE  = 'RUN_FOLDER = dated_run_folder(local_run_name(BENCH.tag, "sim-noiseless", lnn=False))'
TEST_FOLDER  = run_name_for_test("test23")

TEXT_SUFFIXES = {".json", ".log", ".csv", ".txt", ".md", ".tsv", ".yaml", ".yml", ".ini", ".cfg", ".qasm", ".html", ".svg"}

ABSOLUTE_PATH_PATTERNS = {
    "Windows drive path":   re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[" + BS + BS + "/]"),
    "UNC path":             re.compile(r"(?<![" + r"\w" + BS + BS + "])(?:" + BS * 4 + "){1,2}" + r"\w"),
    "absolute POSIX path":  re.compile(r"(?<![\w.:/~-])/(?:home|Users|mnt|tmp|var|root|opt)/"),
}


def _forbidden_literals():
    """Literal strings that must not appear: the project root and the home folder, in every spelling."""
    out = set()
    for p in (PROJECT_ROOT, Path.home()):
        s = str(p)
        for spelling in (s, s.replace("\\", "/"), s.replace("\\", "\\\\"), s.replace("\\", "/").lower(), s.lower()):
            out.add(spelling)
    return {s for s in out if len(s) > 3}


def _text_files(folder):
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if d != "circuits"]    # the circuit cache holds only binary QPY and exact values
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in TEXT_SUFFIXES:
                yield path


def _leaks(path, literals):
    text = path.read_text(encoding="utf-8", errors="replace")
    found = []
    for lit in literals:
        i = text.lower().find(lit.lower())
        if i >= 0:
            found.append(f"{path.as_posix()}: contains {lit!r}: ...{text[max(0, i - 40):i + 80]!r}")
    for label, pattern in ABSOLUTE_PATH_PATTERNS.items():
        m = pattern.search(text)
        if m:
            found.append(f"{path.as_posix()}: {label}: ...{text[max(0, m.start() - 40):m.end() + 80]!r}")
    return found


def test_patterns_catch_absolute_paths_and_spare_urls():
    hit = lambda s: any(p.search(s) for p in ABSOLUTE_PATH_PATTERNS.values())
    for bad in (BS.join(["C:", "Users", "x"]), "C:/Users/x", BS * 2 + "server" + BS + "share", BS * 4 + "server" + BS * 2 + "share",
                "/home/x/proj", "see /Users/x/y", '"/mnt/c/proj"', "/tmp/x"):
        assert hit(bad), bad
    for good in ("https://example.org/a", "datafiles/run/raw_results/x.json", "http://host:8080/home/", "10:30/11",
                 "a/home/b", "P(011)=0.97 -> ok", "line1" + BS + "nline2", "ibm_fez:/x"):
        assert not hit(good), good


def test_saved_config_keeps_json_values_and_drops_paths_and_arrays():
    import numpy as np
    from source.core.records.job_records import saved_config
    noise = {"noise_type": "pauli", "infidelity_total": 0.01, "ratio_ct": 0.5}
    out = saved_config({"coherent_noise": noise, "other": None, "shots": 100, "name": "x", "L": [1, 2],
                        "epsilon": np.array([-1, 0, 1]), "folder_path": "C:/x"})
    assert out == {"coherent_noise": noise, "other": None, "shots": 100, "name": "x", "L": [1, 2]}


def test_23_saved_files_hold_no_absolute_path():
    src = SCRIPT.read_text(encoding="utf-8")
    src = f"__file__ = {str(SCRIPT)!r}\nimport sys; sys.argv = [__file__, 'mini']\n" + src
    assert src.count(FOLDER_LINE) == 1, "benchmark_noiseless_all_to_all.py's RUN_FOLDER line changed — update this test"
    src = src.replace(FOLDER_LINE, f'RUN_FOLDER = "{TEST_FOLDER}"')

    plots, summaries = run_plots_dir(TEST_FOLDER), summaries_dir(TEST_FOLDER)
    rm_glob(f"datafiles/{TEST_FOLDER}", plots.as_posix(), f"{TEST_CIRCUITS_ROOT}/all_to_all_Nt1-3")
    result = subprocess.run([venv_python("local_sim.venv", "test_23 (the mini benchmark child process)"), "-c", src],
                            encoding="utf-8", errors="replace", cwd=PROJECT_ROOT, capture_output=True)
    assert result.returncode == 0, f"benchmark_noiseless_all_to_all.py failed:\n{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    assert analyze_run(TEST_FOLDER)[0], "analyzing the mini run failed"

    run_path = PROJECT_ROOT / "datafiles" / TEST_FOLDER
    files = list(_text_files(run_path))
    names = {p.name for p in files}
    assert {"submit.log", "run_info.json", "run_data.json"} <= names, sorted(names)
    assert any(p.parent.name == "job_records" for p in files), "no job records were written"
    for folder in (PROJECT_ROOT / plots, PROJECT_ROOT / summaries):
        if folder.is_dir():
            files += list(_text_files(folder))

    literals = _forbidden_literals()
    found = [line for p in files for line in _leaks(p, literals)]
    assert not found, f"{len(found)} absolute path(s) in saved files:\n" + "\n".join(found[:20])
