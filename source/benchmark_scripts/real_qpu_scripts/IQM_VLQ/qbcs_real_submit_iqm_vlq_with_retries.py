"""
qbcs_real_submit_iqm_vlq_with_retries.py
Submit a BACQ benchmark run to IT4Innovations' VLQ (LEXIS/HEAppE) and keep retrying until every chunk is submitted:
the Python counterpart of PCSS_Quantum_UI/run_aqt_loop_with_retries.sbatch (VLQ has no Slurm). Before each launch of
qbcs_real_submit_iqm_vlq.py it waits until the latest job submitted to the run (chunk or health check) has finished
on HEAppE, checking every 10 min. After a failed launch it retries at once (the status gate still applies). It gives up
after 48 h without progress (exit code 14). Same arguments as qbcs_real_submit_iqm_vlq.py (backend=…, mini/full, layout=, …).

Logs in like the submit script: with the saved LEXIS token, else through the browser page, so start it from a session
where you can approve that login (see readme_iqm_vlq.md §2). If every chunk is submitted and only the health check after
them could not be submitted, it repeats just that health check (--retry-health-check), at most 3 times.
Stop it with Ctrl-C; start it again with the same arguments to resume (the run folder decides where).
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # datafiles/ paths are relative to the project root
from source.core.runner_support.benchmark_args import make_usage, usage_if_requested   # light (numpy only): the usage is printed before anything else loads
from source.core.records.run_names import latest_run_folder, qpu_run_name
usage_if_requested(sys.argv[1:], make_usage(__file__, "Submit a BACQ benchmark run to IQM VLQ (LEXIS/HEAppE), retrying until every chunk is submitted: waits for the latest job to finish before each launch of qbcs_real_submit_iqm_vlq.py, gives up after 48 h without progress.", layout="submit_live"))   # no arguments or --help: print the usage and do nothing

from pathlib import Path
from source.core.qpu import iqm_vlq as VLQ
from source.core.runner_support.benchmark_args import parse_benchmark_args, parse_backend_config, parse_layout_args, pop_force_new_run_flag
from source.core.runner_support.submit_with_retries import run_with_retries, launch_submit_script, PAUSE_S

SUBMIT_SCRIPT = Path(__file__).parent / "qbcs_real_submit_iqm_vlq.py"
WAIT_AFTER_FAILED_SUBMIT_S = 0   # VLQ: a failed launch is retried at once; the status gate before each launch still waits

# ── Config: the same arguments as qbcs_real_submit_iqm_vlq.py, which decide the run folder ──
FORCE_NEW_RUN, ARGV = pop_force_new_run_flag(sys.argv[1:])   # --force-new-run: only for the first launch (a new folder); later passes resume it
BACKEND_NAME, BACKEND_CFG = parse_backend_config(ARGV, Path(__file__).parent)   # required backend=NAME, checked against backends.json
BENCH            = parse_benchmark_args(ARGV, extra_keys=("backend",))   # full / mini / custom — see source/core/runner_support/benchmark_args.py
LAYOUT           = parse_layout_args(ARGV, live_calibration="IQM VLQ")   # layout=optimized | random | lowest | 15,14,21 — see source/core/runner_support/benchmark_args.py
LNN              = BACKEND_CFG["lnn"]   # the backend's topology, from backends.json
DO_SPAM          = True
UNDATED          = qpu_run_name("IQM_VLQ", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM, LAYOUT.tag)   # the run settings; the run folder is the dated one of these (dated_run_folder)

# ── Status queries: the qaas backend, logged in like the submit script ──
_backend = None

def status_of(job_id, record):
    """The HEAppE status text of a job (DONE, QUEUED, RUNNING, ERROR, CANCELLED, …); raises when it cannot be queried.
    The backend is logged in on first use and again after a failure, so a rotated token is picked up."""
    global _backend
    if _backend is None:
        folder = latest_run_folder(UNDATED)   # py4lexis' log goes in the run folder, like the submit script's
        _backend = VLQ.lexis_backend(log_file=None if folder is None else f"datafiles/{folder}/{VLQ.LEXIS_LOG_NAME}")
    try:
        return str(VLQ.job_status(VLQ.retrieve_job(_backend, job_id), job_id))
    except Exception:
        _backend = None
        raise

def launch(args):
    """Run the submit script with these arguments (in the VLQ venv); its exit code."""
    return launch_submit_script("iqm_vlq.venv", SUBMIT_SCRIPT, args, "qbcs_real_submit_iqm_vlq_with_retries.py")

print(f"\n[{BACKEND_NAME} — connection]")
print(f"  Checking the latest job every {PAUSE_S // 60} min; gives up after 48 h without progress.", flush=True)
try:
    rc = run_with_retries(UNDATED, status_of, launch, ARGV, wait_after_failed_s=WAIT_AFTER_FAILED_SUBMIT_S,
                          new_run=FORCE_NEW_RUN)
except KeyboardInterrupt:
    print("\n  Stopped (Ctrl-C). Run it again with the same arguments to resume.", file=sys.stderr)
    rc = 130
sys.exit(rc)
