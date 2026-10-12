"""
health_check_real_iqm_eqe1.py
Submit AND collect a QPU health check on real IQM EQE1 (via MQSS), in one script.
Standalone — unrelated to any benchmark run (own RUN_FOLDER). RUN_FOLDER is wiped
before every submission, so each run is fresh (same pattern as health_check_real_ibm.py /
health_check_real_iqm_resonance.py). collect_health_check() blocks until the job
reaches a final state.
"""

import os
import shutil
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # datafiles/ paths are relative to the project root
from source.core.records.run_names import health_check_run_name, service_name, dated_run_folder
from source.core.qpu.providers import STANDALONE_HEALTH_CHECK_REPS, STANDALONE_HEALTH_CHECK_NSHOTS
from source.core.runner_support.benchmark_args import make_usage, usage_if_requested   # light (numpy only): the usage is printed before anything else loads
from source.core.qpu.tokens import read_token
usage_if_requested(sys.argv[1:], make_usage(__file__, "Health check of IQM EQE1 (MQSS): submit and collect in one go. Standalone, unrelated to any benchmark run.", benchmark=False))   # no arguments or --help: print the usage and do nothing

from mqss.qiskit_adapter import MQSSQiskitAdapter
from source.core.runner_support.benchmark_args import parse_backend_config
from source.core.health_check import submit_health_check, collect_health_check, print_standalone_summary
from source.core.common import rel_path

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_NAME = parse_backend_config(sys.argv[1:], Path(__file__).parent, others_ok=False)[0]   # required backend=NAME, checked against backends.json
SERVICE_NAME = service_name("IQM_EQE1")   # the access platform, as records and qbcs call it (source/core/records/run_names.py)
NSHOTS       = STANDALONE_HEALTH_CHECK_NSHOTS
REPS         = STANDALONE_HEALTH_CHECK_REPS               # 1 repetition = 4 CNOTs
QUBIT_SET    = None           # physical qubit indices for the 3-qubit health check, e.g. [2, 3, 4]; None = let the transpiler route
RUN_FOLDER   = dated_run_folder(health_check_run_name("IQM_EQE1", BACKEND_NAME, REPS, NSHOTS))

shutil.rmtree(f"datafiles/{RUN_FOLDER}", ignore_errors=True)   # fresh submission every run

# ── Connect to EQE1 via MQSS ──────────────────────────────────────────────────
print(f"\n[{BACKEND_NAME} — connection]")
token        = read_token("iqm_eqe_token")
adapter      = MQSSQiskitAdapter(token=token)
real_backend = adapter.get_backend(BACKEND_NAME)
print(f"  Backend:      {real_backend.name}")
print(f"  Pending jobs: {real_backend.num_pending_jobs}")

# ── Submit ────────────────────────────────────────────────────────────────────
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

# ── Collect result (blocks until the MQSS job is finished) ────────────────────
print(f"\n⏳ Waiting for the result ...")
result = collect_health_check(
    exp_results_folder_path = path,
    service                 = adapter,
    fake_registry           = None,
    service_name            = SERVICE_NAME,
    backend_name            = BACKEND_NAME,
)

# ── Summary ────────────────────────────────────────────────────────────────
p_011  = result["P011"]
status = result["verdict"]

sys.exit(print_standalone_summary(result, p_011, status == "functional", backend_name=BACKEND_NAME))
