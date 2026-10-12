"""
health_check_real_iqm_resonance.py
Submit AND collect a QPU health check on a real IQM QPU on IQM Resonance (cloud),
in one script. submit_health_check() returns immediately (async); collect_health_check()
then blocks on job.wait_for_final_state() and writes the result.
"""

import os
import sys
import shutil
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # datafiles/ paths are relative to the project root
from source.core.records.run_names import health_check_run_name, service_name, dated_run_folder
from source.core.qpu.providers import IQM_RESONANCE_URL
from source.core.qpu.providers import STANDALONE_HEALTH_CHECK_REPS, STANDALONE_HEALTH_CHECK_NSHOTS
from source.core.runner_support.benchmark_args import make_usage, usage_if_requested   # light (numpy only): the usage is printed before anything else loads
from source.core.qpu.tokens import read_token
usage_if_requested(sys.argv[1:], make_usage(__file__, "Health check of an IQM Resonance QPU: submit and collect in one go. Standalone, unrelated to any benchmark run.", benchmark=False))   # no arguments or --help: print the usage and do nothing

from iqm.qiskit_iqm import IQMProvider
from source.core.runner_support.benchmark_args import parse_backend_config
from source.core.health_check import submit_health_check, collect_health_check, print_standalone_summary
from source.core.common import rel_path

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_NAME = parse_backend_config(sys.argv[1:], Path(__file__).parent, others_ok=False)[0]   # required backend=NAME, checked against backends.json
SERVICE_NAME = service_name("IQM_RESONANCE")   # the access platform, as records and qbcs call it (source/core/records/run_names.py)
NSHOTS       = STANDALONE_HEALTH_CHECK_NSHOTS
REPS         = STANDALONE_HEALTH_CHECK_REPS
QUBIT_SET    = None           # physical qubit indices for the 3-qubit health check, e.g. [0, 1, 2]; None = let the transpiler route
RUN_FOLDER   = dated_run_folder(health_check_run_name("IQM_RESONANCE", BACKEND_NAME, REPS, NSHOTS))

shutil.rmtree(f"datafiles/{RUN_FOLDER}", ignore_errors=True)   # fresh submission every run

# ── Connect to IQM Resonance ─────────────────────────────────────────────────
print(f"\n[{BACKEND_NAME} — connection]")
token        = read_token("iqm_resonance_token")
provider     = IQMProvider(IQM_RESONANCE_URL, quantum_computer=BACKEND_NAME, token=token)
real_backend = provider.get_backend()

print(f"  Server:  {IQM_RESONANCE_URL}")
print(f"  QC:      {BACKEND_NAME}")
print(f"  Backend: {real_backend.name}")
print(f"  Qubits:  {real_backend.num_qubits}")

# ── Submit health check (async) ─────────────────────────────────────────────
path = submit_health_check(
    run_folder    = RUN_FOLDER,
    backend       = real_backend,
    sampler       = None,
    nshots        = NSHOTS,
    service       = None,
    fake_registry = None,
    service_name  = SERVICE_NAME,
    backend_name  = BACKEND_NAME,
    reps          = REPS,
    qubit_set     = QUBIT_SET,
)
print(f"\n✅ Submitted to {BACKEND_NAME}")
print(f"   Job record: {rel_path(path)}/job_records/health_check.json")

# ── Collect result (blocks on job.wait_for_final_state) ─────────────────────
print(f"\n⏳ Waiting for the result ...")
result = collect_health_check(
    exp_results_folder_path = path,
    service                 = real_backend,   # IQMBackend — used for retrieve_job()
    fake_registry           = None,
    service_name            = SERVICE_NAME,
    backend_name            = BACKEND_NAME,
)

# ── Summary ────────────────────────────────────────────────────────────────
p_011  = result["P011"]
status = result["verdict"]

sys.exit(print_standalone_summary(result, p_011, status == "functional", backend_name=BACKEND_NAME))
