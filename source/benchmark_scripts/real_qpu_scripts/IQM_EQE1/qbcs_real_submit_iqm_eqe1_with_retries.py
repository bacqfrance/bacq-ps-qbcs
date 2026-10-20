"""
qbcs_real_submit_iqm_eqe1_with_retries.py
Submit a BACQ benchmark run to IQM EQE1 (MQSS) and keep retrying until every chunk is submitted: the Python
counterpart of PCSS_Quantum_UI/run_aqt_loop_with_retries.sbatch (EQE1 has no Slurm). It does not wait for the
jobs already submitted: after a failed launch of qbcs_real_submit_iqm_eqe1.py it retries at once, then every 1 min,
and gives up after 20 min without a chunk submitted (exit code 14). A launch within 21 min of the run's last
job (a chunk, or a functional health check) skips the health check before the submission. If every chunk is submitted and only the health check after them could not be submitted, it repeats just
that health check (--retry-health-check), at most 3 times. Same arguments as qbcs_real_submit_iqm_eqe1.py (backend=EQE1, mini/full, layout=, …).

Stop it with Ctrl-C; start it again with the same arguments to resume (the run folder decides where).
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # datafiles/ paths are relative to the project root
from source.core.runner_support.benchmark_args import make_usage, usage_if_requested   # light (numpy only): the usage is printed before anything else loads
from source.core.records.run_names import qpu_run_name
usage_if_requested(sys.argv[1:], make_usage(__file__, "Submit a BACQ benchmark run to IQM EQE1 (MQSS), retrying a failed launch of qbcs_real_submit_iqm_eqe1.py at once, then every 1 min, until every chunk is submitted; gives up after 20 min without progress.", layout="submit_no_live"))   # no arguments or --help: print the usage and do nothing

from pathlib import Path
from source.core.runner_support.benchmark_args import parse_benchmark_args, parse_backend_config, parse_layout_args, pop_force_new_run_flag
from source.core.runner_support.submit_with_retries import run_with_retries, launch_submit_script

SUBMIT_SCRIPT = Path(__file__).parent / "qbcs_real_submit_iqm_eqe1.py"
WAIT_AFTER_FAILED_SUBMIT_S = 60    # EQE1 (MQSS): the first retry is at once, the later ones 1 min after the previous failure
RETRY_LIMIT_HOURS = 20 / 60        # give up after 20 min of failed launches without a chunk submitted

# ── Config: the same arguments as qbcs_real_submit_iqm_eqe1.py, which decide the run folder ──
FORCE_NEW_RUN, ARGV = pop_force_new_run_flag(sys.argv[1:])   # --force-new-run: only for the first launch (a new folder); later passes resume it
BACKEND_NAME, BACKEND_CFG = parse_backend_config(ARGV, Path(__file__).parent)   # required backend=NAME, checked against backends.json
BENCH            = parse_benchmark_args(ARGV, extra_keys=("backend",))   # full / mini / custom — see source/core/runner_support/benchmark_args.py
LAYOUT           = parse_layout_args(ARGV, calibration_required=True)   # layout=... [calibration_file=...] — see source/core/runner_support/benchmark_args.py
LNN              = BACKEND_CFG["lnn"]   # the backend's topology, from backends.json
DO_SPAM          = True
UNDATED          = qpu_run_name("IQM_EQE1", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM, LAYOUT.tag)   # the run settings; the run folder is the dated one of these (dated_run_folder)

def launch(args):
    """Run the submit script with these arguments (in the EQE1 venv); its exit code."""
    return launch_submit_script("iqm_eqe1.venv", SUBMIT_SCRIPT, args, "qbcs_real_submit_iqm_eqe1_with_retries.py")

print("  Retries a failed launch at once, then every 1 min; gives up after 20 min without progress.", flush=True)
try:
    rc = run_with_retries(UNDATED, None, launch, ARGV, wait_after_failed_s=WAIT_AFTER_FAILED_SUBMIT_S,
                          limit_hours=RETRY_LIMIT_HOURS, wait_for_jobs=False, first_retry_at_once=True,
                          new_run=FORCE_NEW_RUN)
except KeyboardInterrupt:
    print("\n  Stopped (Ctrl-C). Run it again with the same arguments to resume.", file=sys.stderr)
    rc = 130
sys.exit(rc)
