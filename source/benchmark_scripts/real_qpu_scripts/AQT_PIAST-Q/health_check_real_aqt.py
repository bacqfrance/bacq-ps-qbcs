"""
health_check_real_aqt.py
Submit AND collect a QPU health check on the real AQT QPU (PIAST-Q), in one
script. AQT's submit_health_check() call is synchronous — it blocks until the
job completes and writes the full result straight to the job record, so
submitting and collecting happen in this one call.
"""

import os
import json
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
usage_if_requested(sys.argv[1:], make_usage(__file__, "Health check of AQT PIAST-Q: submit and collect in one go (the call is synchronous). Standalone, unrelated to any benchmark run.", benchmark=False))   # no arguments or --help: print the usage and do nothing
from source.core.runner_support.benchmark_args import parse_backend_config
BACKEND_NAME = parse_backend_config(sys.argv[1:], Path(__file__).parent, others_ok=False)[0]   # required backend=NAME, checked against backends.json
SERVICE_NAME = service_name("AQT_PIAST-Q")   # the access platform, as records and qbcs call it (source/core/records/run_names.py)

from pcss_qapi import AuthorizationService
from pcss_qapi.aqt.provider import PCSS_AQTProvider
from source.core.health_check import submit_health_check, print_standalone_summary
from source.core.records.health_check_records import EXIT_AQT_TASK_TIMEOUT, EXIT_HC_NOT_SUBMITTED
from source.core.qpu.aqt import AQTTaskTimeout

# ── Config ────────────────────────────────────────────────────────────────────
NSHOTS       = STANDALONE_HEALTH_CHECK_NSHOTS
REPS         = STANDALONE_HEALTH_CHECK_REPS
RUN_FOLDER   = dated_run_folder(health_check_run_name("AQT_PIAST-Q", BACKEND_NAME, REPS, NSHOTS))

shutil.rmtree(f"datafiles/{RUN_FOLDER}", ignore_errors=True)

# ── Connect to PIAST-Q ────────────────────────────────────────────────────────
print(f"\n[{BACKEND_NAME} — connection]")
ACCESS_TOKEN = read_token("aqt_piast-q_token")
AuthorizationService.login(ACCESS_TOKEN)
provider     = PCSS_AQTProvider()
real_backend = provider.get_direct_access_backend()

print(f"  Backend: {real_backend.name}")

# ── Submit health check (synchronous — completes inline) ────────────────────
try:
    path = submit_health_check(
        run_folder   = RUN_FOLDER,
        backend      = real_backend,
        nshots       = NSHOTS,
        service_name = SERVICE_NAME,
        backend_name = BACKEND_NAME,
        reps         = REPS,
    )
except Exception as e:
    print(f"❌ Health check not executed on {BACKEND_NAME}: {e}", file=sys.stderr)
    sys.exit(EXIT_AQT_TASK_TIMEOUT if isinstance(e, AQTTaskTimeout) else EXIT_HC_NOT_SUBMITTED)

record = json.loads((path / "job_records" / "health_check.json").read_text(encoding="utf-8"))

# ── Summary ────────────────────────────────────────────────────────────────
# Note: for AQT, submit_health_check() writes status="complete" — the job ran, it does not
# judge pass/fail; the verdict (the shared health_check_verdict(), as for every QPU and in the
# PRE/POST checks of qbcs_real_submit_aqt_loop.py) is in the record's "verdict".
sys.exit(print_standalone_summary(record, record["P011"], record["verdict"] == "functional", backend_name=BACKEND_NAME))
