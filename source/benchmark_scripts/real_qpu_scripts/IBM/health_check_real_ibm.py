"""
health_check_real_ibm.py
Submit AND collect a QPU health check on a real IBM QPU, in one script.
RUN_FOLDER is wiped before every submission, so each run is fresh (same
pattern as health_check_real_iqm_eqe1.py / health_check_real_iqm_resonance.py).
collect_health_check() blocks (polls every 30s internally) until the job
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
from source.core.runner_support.benchmark_args import make_usage, usage_if_requested   # light (numpy only): the usage is printed before anything else loads
from source.core.qpu.providers import STANDALONE_HEALTH_CHECK_REPS, STANDALONE_HEALTH_CHECK_NSHOTS, ibm_batch
from source.core.records.run_names import health_check_run_name, service_name, dated_run_folder
from source.core.qpu.tokens import read_token
usage_if_requested(sys.argv[1:], make_usage(__file__, "Health check of a real IBM QPU: submit and collect in one go. Standalone, unrelated to any benchmark run.", benchmark=False, instance=True))   # no arguments or --help: print the usage and do nothing

from source.core.runner_support.benchmark_args import parse_backend_config, parse_instance_arg
from source.core.qpu.providers import connect_ibm
from source.core.health_check import submit_health_check, collect_health_check, print_standalone_summary
from source.core.common import rel_path

# ── Config ────────────────────────────────────────────────────────────────────
INSTANCE_ARG = parse_instance_arg(sys.argv[1:])   # instance=NAME, default the Open Plan instance
BACKEND_NAME = parse_backend_config([a for a in sys.argv[1:] if not a.startswith("instance=")], Path(__file__).parent, others_ok=False)[0]   # required backend=NAME, checked against backends.json
SERVICE_NAME = service_name("IBM")   # the access platform, as records and qbcs call it (source/core/records/run_names.py)
NSHOTS       = STANDALONE_HEALTH_CHECK_NSHOTS
REPS         = STANDALONE_HEALTH_CHECK_REPS
RUN_FOLDER   = dated_run_folder(health_check_run_name("IBM", BACKEND_NAME, REPS, NSHOTS))

shutil.rmtree(f"datafiles/{RUN_FOLDER}", ignore_errors=True)   # fresh submission every run

# ── Connect to IBM ────────────────────────────────────────────────────────────
print(f"\n[{BACKEND_NAME} — connection]")
token        = read_token("ibm_token")
service, INSTANCE = connect_ibm(token, INSTANCE_ARG, BACKEND_NAME)
real_backend = service.backend(BACKEND_NAME)

print(f"  Channel:  ibm_quantum_platform")
print(f"  Backend:  {real_backend.name}")
print(f"  Qubits:   {real_backend.num_qubits}")

# ── Submit health check ───────────────────────────────────────────────────────
with ibm_batch(real_backend) as sampler:
    path = submit_health_check(
        run_folder   = RUN_FOLDER,
        backend      = real_backend,
        sampler      = sampler,
        nshots       = NSHOTS,
        service      = service,
        service_name = SERVICE_NAME,
        backend_name = BACKEND_NAME,
        reps         = REPS,
    )
print(f"\n✅ Submitted → {rel_path(path)}")
print(f"   Job record: {rel_path(path)}/job_records/health_check.json")

# ── Collect result — blocks, polling every 30s until the job finishes ────────
print(f"\n⏳ Waiting for the result ...")
result = collect_health_check(
    exp_results_folder_path = path,
    service                 = service,
    fake_registry           = None,
    service_name            = SERVICE_NAME,
    backend_name            = BACKEND_NAME,
)

# ── Summary ────────────────────────────────────────────────────────────────
p_011  = result["P011"]
status = result["verdict"]

sys.exit(print_standalone_summary(result, p_011, status == "functional", backend_name=BACKEND_NAME))
