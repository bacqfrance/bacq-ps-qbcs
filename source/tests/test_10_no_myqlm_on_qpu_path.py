"""
test_10_no_myqlm_on_qpu_path.py
===============================
The real-QPU venvs have no myQLM (USER_MANUAL §3.1): build() runs in local_sim.venv
and caches every circuit as Qiskit QPY plus the exact reference values, and
everything after that (build() from the cache, submission, collection,
analysis) must run on those files alone.

This test builds a small config here (with myQLM), then runs a child process
in which any import of `qat` raises, the way it would in a venv without
myQLM. The child:
  - imports source.core.qbcs (no module may import myQLM at load time);
  - calls build() — which must take the cached-QPY path;
  - submits every chunk to the local Aer sampler and analyzes the run;
  - builds an uncached config, which must stop with the RuntimeError that
    points to local_sim.venv, not an ImportError from deep inside;
and finally reports whether `qat` ever made it into sys.modules.

A second child imports every source.core module except circuits/main.py, rc_functions.py and
circuit_functions.py (the only ones allowed to import myQLM at load time) with `qat`
blocked the same way.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import json
import subprocess
from pathlib import Path
from source.core.qbcs import bacq_ps_qbcs
from source.tests.testing_functions import rm_glob, run_name_for_test, TEST_CIRCUITS_ROOT
from source.core.records.run_names import plots_dir as run_plots_dir
from source.core.runner_support.venvs import venv_python

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_FOLDER   = run_name_for_test("test10")
LIST_TIMES   = [0.4, 0.6]
SAMPLE_SIZE  = 8          # multiple of 2**nqubits=8

CHILD = r'''
import sys, os, json, importlib.abc
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.getcwd())

class _NoMyQLM(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "qat" or name.startswith("qat.") or name.startswith("qlmaas"):
            raise ImportError(f"myQLM import blocked: {name}")
        return None
sys.meta_path.insert(0, _NoMyQLM())

from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from source.core.qbcs import bacq_ps_qbcs

def new(list_times):
    return bacq_ps_qbcs(sample_size=SAMPLE_SIZE, chunk_size=len(list_times) * 6, list_circuit_times=list_times,
                        error_mitigation_strategies=["SPAM", "NEC", "RC+NEC", "cRC+NEC"], do_SPAM=True,
                        nshots=200, dt=0.2, rng_seed=0)

q = new(LIST_TIMES)
q.build(circuit_transpilation_LNN=False)
q.submit(run_folder=RUN_FOLDER, lang="qiskit", service="local", backend=AerSamplerV2(seed=0))
q.analyze_results()

try:
    new([0.4, 0.8]).build(circuit_transpilation_LNN=False)   # Nt=4 was never built
    uncached = "no error"
except RuntimeError as e:
    uncached = str(e)

print("RESULT " + json.dumps({
    "qat_loaded": any(m == "qat" or m.startswith("qat.") for m in sys.modules),
    "uncached":   uncached,
}))
'''


def _local_sim_python():
    return venv_python("local_sim.venv", "test_10 (the child process that blocks myQLM)")


def test_10_no_myqlm_on_qpu_path():
    # Start from an empty cache for this config (datafiles/tests/circuits/all_to_all_Nt2-3;
    # conftest.py keeps any real one safe), so the build below really runs myQLM.
    rm_glob(f"datafiles/{RUN_FOLDER}", f"{TEST_CIRCUITS_ROOT}/all_to_all_Nt2-3")

    # Build the small config here, where myQLM is available (as in local_sim.venv).
    bacq_ps_qbcs(sample_size=SAMPLE_SIZE, chunk_size=len(LIST_TIMES) * 6, list_circuit_times=LIST_TIMES,
                 error_mitigation_strategies=["SPAM", "NEC", "RC+NEC", "cRC+NEC"], do_SPAM=True,
                 nshots=200, dt=0.2, rng_seed=0
                 ).build(circuit_transpilation_LNN=False)

    code = f"RUN_FOLDER = {RUN_FOLDER!r}\nLIST_TIMES = {LIST_TIMES!r}\nSAMPLE_SIZE = {SAMPLE_SIZE}\n" + CHILD
    proc = subprocess.run([_local_sim_python(), "-c", code], cwd=PROJECT_ROOT,
                          capture_output=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, f"child failed (exit {proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"

    result_lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert result_lines, f"child printed no result:\n{proc.stdout[-3000:]}"
    result = json.loads(result_lines[-1][len("RESULT "):])

    assert "myQLM not needed" in proc.stdout, "build() didn't take the cached-QPY path"
    assert not result["qat_loaded"], "myQLM was imported on the submit/analysis path"
    assert "local_sim.venv" in result["uncached"], (
        f"building an uncached config without myQLM should say to build in local_sim.venv, got: {result['uncached']!r}")

    run_data = PROJECT_ROOT / "datafiles" / RUN_FOLDER / "processed_results" / "run_data.json"
    assert run_data.exists(), "analysis produced no run_data.json"
    depths = json.loads(run_data.read_text(encoding="utf-8"))["depths"]
    assert sorted(int(nt) for nt in depths) == [2, 3]
    for nt, d in depths.items():
        assert len(d["perfect_circuit_obs_values"]) == 7

    rm_glob(f"datafiles/{RUN_FOLDER}")
    rm_glob(f"{run_plots_dir(RUN_FOLDER).as_posix()}*")


IMPORT_CHILD = r'''
import sys, os, json, importlib, importlib.abc
sys.path.insert(0, os.getcwd())

class _NoMyQLM(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "qat" or name.startswith("qat.") or name.startswith("qlmaas"):
            raise ImportError(f"myQLM import blocked: {name}")
        return None
sys.meta_path.insert(0, _NoMyQLM())

failed = {}
for m in MODULES:
    try:
        importlib.import_module(m)
    except Exception as e:
        failed[m] = f"{type(e).__name__}: {e}"
print("RESULT " + json.dumps({"failed": failed,
                              "qat_loaded": any(m == "qat" or m.startswith("qat.") for m in sys.modules)}))
'''


MYQLM_AT_IMPORT = {"source.core.circuits.main", "source.core.circuits.rc_functions",
                   "source.core.circuits.circuit_functions"}   # the only modules allowed to import myQLM at load time


def test_10_only_the_myqlm_circuit_modules_need_it_at_import():
    # Every other source.core module must import in a process where `qat` is blocked.
    core    = PROJECT_ROOT / "source" / "core"
    modules = sorted(m for m in (".".join(p.relative_to(PROJECT_ROOT).with_suffix("").parts)
                                 for p in core.rglob("*.py") if p.name != "__init__.py")
                     if m not in MYQLM_AT_IMPORT)
    assert {"source.core.qbcs", "source.core.qpu.iqm_vlq", "source.core.circuits.building"} <= set(modules)
    proc = subprocess.run([_local_sim_python(), "-c", f"MODULES = {modules!r}\n" + IMPORT_CHILD], cwd=PROJECT_ROOT,
                          capture_output=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, f"child failed (exit {proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    result = json.loads([l for l in proc.stdout.splitlines() if l.startswith("RESULT ")][-1][len("RESULT "):])
    assert not result["failed"], f"modules that do not import without myQLM: {result['failed']}"
    assert not result["qat_loaded"], "importing the modules that must load without myQLM loaded it"


if __name__ == "__main__":
    # Through pytest, so conftest.py protects the real circuit cache this test wipes.
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
