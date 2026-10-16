"""
cancel_real_qpu_jobs_ibm.py
Cancel the jobs of a run on IBM that are still queued or running, according to the job records on disk.
Jobs that already ended are left alone and the records on disk are not changed: run the collect script
afterwards to update them. Without layout= and date=, every run of the benchmark with
queued or running jobs is handled in turn.

  backend=NAME            required, one of the backends in backends.json
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
from source.core.qpu.tokens import read_token
usage_if_requested(sys.argv[1:], make_usage(__file__, "Cancel the jobs of a run on IBM that are still queued or running (jobs that already ended are left alone; the records on disk are not changed). Waits for nothing.", health_check_flag=True, layout="collect", instance=True))   # no arguments or --help: print the usage and do nothing

from source.core.runner_support.benchmark_args import parse_benchmark_args, parse_backend_config, parse_layout_args, find_collect_run_folders, parse_date_arg, parse_instance_arg
from source.core.qpu.providers import connect_ibm, ibm_instance_for_runs
from source.core.qpu.job_status_check import pop_health_check_flag
from source.core.qpu.job_cancel import run_cancels

# ── Config — must match the submit and health-check scripts ───────────────────
HEALTH_CHECK, ARGV = pop_health_check_flag(sys.argv[1:])
BACKEND_NAME, BACKEND_CFG = parse_backend_config(ARGV, Path(__file__).parent)   # required backend=NAME, checked against backends.json
BENCH        = parse_benchmark_args(ARGV, extra_keys=("backend", "date", "instance"), required=not HEALTH_CHECK)   # full / mini / custom; none (None) for a health check
LAYOUT       = parse_layout_args(ARGV, collect=True)   # layout=... / date=... pick the run folder(s) (date= alone: each layout's run of that date); None: every run with something left to collect
LNN          = BACKEND_CFG["lnn"]   # the backend's topology, from backends.json
DO_SPAM      = True
RUN_FOLDERS  = [] if BENCH is None else find_collect_run_folders(qpu_run_prefix("IBM", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM), LAYOUT, parse_date_arg(ARGV))
HC_FOLDER    = latest_run_folder(health_check_run_name("IBM", BACKEND_NAME, STANDALONE_HEALTH_CHECK_REPS, STANDALONE_HEALTH_CHECK_NSHOTS)) or health_check_run_name("IBM", BACKEND_NAME, STANDALONE_HEALTH_CHECK_REPS, STANDALONE_HEALTH_CHECK_NSHOTS)   # the standalone health check's folder

# ── Connect ───────────────────────────────────────────────────────────────────
print(f"\n[{BACKEND_NAME} — connection]")
token   = read_token("ibm_token")
service, INSTANCE = connect_ibm(token, ibm_instance_for_runs(parse_instance_arg(ARGV), RUN_FOLDERS), BACKEND_NAME)   # the runs' recorded instance (the standalone health check has none: instance= or the Open Plan one)

run_cancels(BACKEND_NAME, RUN_FOLDERS, HC_FOLDER, HEALTH_CHECK,
            remote_status=lambda job_id: service.job(job_id).status(),
            cancel_job=lambda job_id: service.job(job_id).cancel())
