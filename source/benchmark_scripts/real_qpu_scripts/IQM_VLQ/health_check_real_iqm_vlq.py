"""
health_check_real_iqm_vlq.py
Submit AND collect a QPU health check on IT4Innovations' VLQ (LEXIS/HEAppE, qaas), in one
script. submit_health_check() returns once the circuit is uploaded; collect_health_check()
then blocks until the job is finished and writes the result.
"""

import os
import sys
from pathlib import Path
import shutil
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # datafiles/ paths are relative to the project root
from source.core.runner_support.benchmark_args import make_usage, usage_if_requested   # light (numpy only): the usage is printed before anything else loads
from source.core.records.run_names import health_check_run_name, service_name, dated_run_folder
from source.core.qpu.providers import STANDALONE_HEALTH_CHECK_REPS, STANDALONE_HEALTH_CHECK_NSHOTS
usage_if_requested(sys.argv[1:], make_usage(__file__, "Health check of IQM VLQ (LEXIS/HEAppE): submit and collect in one go. Standalone, unrelated to any benchmark run.", benchmark=False))   # no arguments or --help: print the usage and do nothing

from source.core.runner_support.benchmark_args import parse_backend_config
from source.core.qpu import iqm_vlq as VLQ
from source.core.health_check import submit_health_check, collect_health_check, print_standalone_summary
from source.core.common import rel_path

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_NAME = parse_backend_config(sys.argv[1:], Path(__file__).parent, others_ok=False)[0]   # required backend=NAME, checked against backends.json
SERVICE_NAME = service_name("IQM_VLQ")   # the access platform, as records and qbcs call it (source/core/records/run_names.py)
NSHOTS      = STANDALONE_HEALTH_CHECK_NSHOTS
REPS         = STANDALONE_HEALTH_CHECK_REPS
QUBIT_SET    = None           # physical qubit indices (0 = QB1) for the 3-qubit health check, e.g. [0, 1, 2]; None = let the transpiler choose
RUN_FOLDER   = dated_run_folder(health_check_run_name("IQM_VLQ", BACKEND_NAME, REPS, NSHOTS))

shutil.rmtree(f"datafiles/{RUN_FOLDER}", ignore_errors=True)   # fresh submission every run

# ── Connect to VLQ ────────────────────────────────────────────────────────────
print(f"\n[{BACKEND_NAME} — connection]")
qpu_status   = VLQ.qpu_status()
real_backend = VLQ.lexis_backend(log_file=f"datafiles/{RUN_FOLDER}/{VLQ.LEXIS_LOG_NAME}")   # py4lexis' log, in the run folder
print(f"  Status:  QPU {qpu_status.get('QPU_status')}, total {qpu_status.get('total_status')} ({qpu_status.get('timestamp')})")
print(f"  Backend: {real_backend.backend_name}")
print(f"  Qubits:  {real_backend.num_qubits} + resonator {VLQ.RESONATOR}")

# ── Submit health check (async) ───────────────────────────────────────────────
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

# ── Collect result (blocks until the HEAppE job is finished) ──────────────────
print(f"\n⏳ Waiting for the result ...")
result = collect_health_check(
    exp_results_folder_path = path,
    service                 = real_backend,   # qaas backend — used for retrieve_job()
    fake_registry           = None,
    service_name            = SERVICE_NAME,
    backend_name            = BACKEND_NAME,
)

# ── Summary ───────────────────────────────────────────────────────────────────
p_011  = result["P011"]
status = result["verdict"]

sys.exit(print_standalone_summary(result, p_011, status == "functional", "(virtual → physical, 0 = QB1)", backend_name=BACKEND_NAME))
