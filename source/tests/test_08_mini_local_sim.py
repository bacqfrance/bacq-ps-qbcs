"""
test_08_mini_local_sim.py
=========================
source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_all_to_all.py itself, run the way a
user starts with it — with the `mini` argument — and unmodified except for its
output folder name (and, in the no-SPAM case, do_spam=True replaced by do_spam=False) (so it can't touch a real mini_sim-noiseless_… run). Its results are then
analyzed through source/analysis_scripts/run_post_process.py like any other run: summary JSON + plots must carry
the mini benchmark's own 200 shots / 8 samples / 2 depths, and the run must hold the drawings of
the first circuit of every error-mitigation role (source/core/circuits/circuit_drawings.py).
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import json
import subprocess
import pytest
from source.core.runner_support.venvs import venv_python
import numpy as np
from pathlib import Path

from source.analysis_scripts.run_post_process import analyze_run
from source.core.records.run_info import load_run_info
from source.core.circuits.circuit_drawings import ROLE_LABELS
from source.tests.testing_functions import rm_glob, run_name_for_test, TEST_CIRCUITS_ROOT
from source.core.records.run_names import plots_dir as run_plots_dir
from source.core.records.run_names import run_summary_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT       = PROJECT_ROOT / "source" / "benchmark_scripts" / "local_simulation_scripts" / "benchmark_noiseless_all_to_all.py"
FOLDER_LINE  = 'RUN_FOLDER = local_run_name(BENCH.tag, "sim-noiseless", lnn=False)'
TEST_FOLDER  = run_name_for_test("test08")


@pytest.mark.parametrize("do_spam", [True, False], ids=["spam", "no_spam"])
def test_08_mini_local_sim_results_are_analyzable(do_spam):
    test_folder = TEST_FOLDER if do_spam else run_name_for_test("test08nospam")
    src = SCRIPT.read_text(encoding="utf-8")
    # run via -c below, where __file__ is otherwise undefined and there is no benchmark argument (the script has no default)
    src = f"__file__ = {str(SCRIPT)!r}\nimport sys; sys.argv = [__file__, 'mini']\n" + src
    assert src.count(FOLDER_LINE) == 1, "benchmark_noiseless_all_to_all.py's RUN_FOLDER line changed — update this test"
    src = src.replace(FOLDER_LINE, f'RUN_FOLDER = "{test_folder}"')
    if not do_spam:   # the same script with SPAM switched off
        for old, new in [("do_spam=True,", "do_spam=False,")]:
            assert src.count(old) == 1, f"benchmark_noiseless_all_to_all.py changed ({old!r}) — update this test"
            src = src.replace(old, new)

    # Delete the test session's mini circuit cache too (datafiles/tests/circuits/, never a real one), so its build runs
    rm_glob(f"datafiles/{test_folder}", run_plots_dir(test_folder).as_posix(), f"{TEST_CIRCUITS_ROOT}/all_to_all_Nt1-3")
    # encoding="utf-8": the child reconfigures ITS stdout to UTF-8 and prints non-ASCII characters;
    # decode them as UTF-8 here too, instead of the Windows console codepage (cp1252).
    result = subprocess.run([venv_python("local_sim.venv", "test_08 (the mini benchmark child process)"), "-c", src],
                            encoding="utf-8", errors="replace", cwd=PROJECT_ROOT, capture_output=True)
    assert result.returncode == 0, f"benchmark_noiseless_all_to_all.py failed:\n{result.stdout[-2000:]}\n{result.stderr[-3000:]}"

    run_path = PROJECT_ROOT / "datafiles" / test_folder
    info = load_run_info(test_folder)
    assert info.source == "manifest"
    assert (info.nshots, info.sample_size, info.do_spam, info.lnn) == (200, 8, do_spam, False)
    assert np.allclose(info.list_times, [0.2, 0.6]) and info.chunk_size == 2 * 6

    # the first circuit of every role and of SPAM (run as they are: no "after" drawing), and the
    # health check (transpiled for the generic AerSimulator: both drawings)
    drawings = sorted(p.name for p in (run_path / "circuit_drawings").iterdir())
    expected = (["00_SPAM_before_transpilation.pdf"] if do_spam else []) + (
                [f"{i + 1:02d}_{label}_before_transpilation.pdf" for i, label in enumerate(ROLE_LABELS.values())]
                + ["health_check_after_transpilation.pdf", "health_check_before_transpilation.pdf"])
    assert drawings == sorted(expected), drawings

    assert analyze_run(test_folder)[0], "analyzing the mini run failed"
    summary = json.loads((run_path / "processed_results" / run_summary_name()).read_text(encoding="utf-8"))
    assert summary["shots_per_circuit"] == 200 and summary["n_samples_analyzed"] == 8
    run_data = json.loads((run_path / "processed_results" / "run_data.json").read_text(encoding="utf-8"))
    assert sorted(int(k) for k in run_data["depths"]) == [1, 3]

    plots_dir = PROJECT_ROOT / run_plots_dir(test_folder)
    for fname in ["obs_lnnFalse_ns200_no_spam.png", "max_error_lnnFalse_ns200.png", "awae_no_spam.png",
                  "health_checks.png"] + (["obs_lnnFalse_ns200_spam.png", "awae_spam.png"] if do_spam else []):
        assert (plots_dir / fname).exists(), f"missing plot {fname}"
    if not do_spam:
        assert not (plots_dir / "awae_spam.png").exists() and not (run_path / "processed_results" / "run_data_spam.json").exists()
