"""
cancel_real_qpu_jobs_iqm_vlq.py
Cancel the jobs of a run on IQM VLQ (HEAppE) that are still queued or running, according to the job records on disk.
Jobs that already ended are left alone and the records on disk are not changed: run the collect script
afterwards to update them. Without layout= and date=, every run of the benchmark with
queued or running jobs is handled in turn.

  backend=NAME            required (VLQ)
  health_check            cancel the standalone health check of that backend (its latest) instead of a benchmark run
  layout=... / date=...   optional: that one run; without them, every run of the benchmark with queued or running jobs (all layouts and dates)

Benchmark size (required — there is no default), from the command line (source/core/runner_support/benchmark_args.py; readme: "Benchmark size and run folders"):
  mini · full · mini|full nshots_per_circuit=N ·
  nshots_per_circuit=N n_samples=M n_depths=K (or nt_list=1,5,9)
"""

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # datafiles/ paths are relative to the project root
from source.core.runner_support.benchmark_args import make_usage, usage_if_requested   # light (numpy only): the usage is printed before anything else loads
from source.core.qpu.providers import STANDALONE_HEALTH_CHECK_REPS, STANDALONE_HEALTH_CHECK_NSHOTS
from source.core.records.run_names import qpu_run_prefix, health_check_run_name, latest_run_folder
usage_if_requested(sys.argv[1:], make_usage(__file__, "Cancel the jobs of a run on IQM VLQ (HEAppE) that are still queued or running (jobs that already ended are left alone; the records on disk are not changed). Waits for nothing.", health_check_flag=True, layout="collect"))   # no arguments or --help: print the usage and do nothing

from source.core.qpu import iqm_vlq as VLQ
from source.core.runner_support.benchmark_args import parse_benchmark_args, parse_backend_config, parse_layout_args, find_collect_run_folders, parse_date_arg
from source.core.qpu.job_status_check import pop_health_check_flag
from source.core.qpu.job_cancel import run_cancels

# ── Config — must match the submit and health-check scripts ───────────────────
HEALTH_CHECK, ARGV = pop_health_check_flag(sys.argv[1:])
BACKEND_NAME, BACKEND_CFG = parse_backend_config(ARGV, Path(__file__).parent)   # required backend=NAME, checked against backends.json
BENCH        = parse_benchmark_args(ARGV, extra_keys=("backend", "date"), required=not HEALTH_CHECK)   # full / mini / custom; none (None) for a health check
LAYOUT       = parse_layout_args(ARGV, collect=True)   # layout=... / date=... pick the run folder(s) (date= alone: each layout's run of that date); None: every run with something left to collect
LNN          = BACKEND_CFG["lnn"]   # the backend's topology, from backends.json
DO_SPAM      = True
RUN_FOLDERS  = [] if BENCH is None else find_collect_run_folders(qpu_run_prefix("IQM_VLQ", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM), LAYOUT, parse_date_arg(ARGV))   # one status report per folder
HC_NAME      = health_check_run_name("IQM_VLQ", BACKEND_NAME, STANDALONE_HEALTH_CHECK_REPS, STANDALONE_HEALTH_CHECK_NSHOTS)
HC_LATEST    = latest_run_folder(HC_NAME)   # None: no standalone health check on disk
HC_FOLDER    = HC_LATEST or HC_NAME   # the standalone health check's folder

# ── Connect ───────────────────────────────────────────────────────────────────
print(f"\n[{BACKEND_NAME} — connection]")
LOG_FOLDER   = HC_LATEST if HEALTH_CHECK else (RUN_FOLDERS[0] if RUN_FOLDERS else None)   # an existing folder, or None (the project root)
real_backend = VLQ.lexis_backend(log_file=None if LOG_FOLDER is None else f"datafiles/{LOG_FOLDER}/{VLQ.LEXIS_LOG_NAME}")   # py4lexis' log, in the run folder when it exists

def cancel_job(job_id):
    if not VLQ.cancel_job(VLQ.retrieve_job(real_backend, job_id), job_id):   # logged, time-limited; HEAppE only accepts the request
        raise RuntimeError("HEAppE did not accept the cancellation")

run_cancels(BACKEND_NAME, RUN_FOLDERS, HC_FOLDER, HEALTH_CHECK,
            remote_status=lambda job_id: VLQ.job_status(VLQ.retrieve_job(real_backend, job_id), job_id),   # logged, time-limited
            cancel_job=cancel_job)
