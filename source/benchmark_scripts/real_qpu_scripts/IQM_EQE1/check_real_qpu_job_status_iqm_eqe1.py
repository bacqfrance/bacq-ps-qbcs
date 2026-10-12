"""
check_real_qpu_job_status_iqm_eqe1.py
Check the status of a run's jobs on IQM EQE1 (MQSS) against the job records on disk.
Does not wait and changes nothing. Without layout= and date=, every run of the benchmark with
something left to collect is reported in turn.

  backend=NAME            required (EQE1)
  health_check            check the standalone health check of that backend instead of a benchmark run

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
usage_if_requested(sys.argv[1:], make_usage(__file__, "Check the live status of a run's jobs on IQM EQE1 (MQSS) against the job records on disk. Waits for nothing and changes nothing.", health_check_flag=True, layout="collect"))   # no arguments or --help: print the usage and do nothing

from mqss.qiskit_adapter import MQSSQiskitAdapter
from mqss_client.job import CircuitJobRequest
from source.core.runner_support.benchmark_args import parse_benchmark_args, parse_backend_config, parse_layout_args, find_collect_run_folders, parse_date_arg
from source.core.qpu.job_status_check import pop_health_check_flag, run_status_checks

# ── Config — must match the submit and health-check scripts ───────────────────
HEALTH_CHECK, ARGV = pop_health_check_flag(sys.argv[1:])
BACKEND_NAME, BACKEND_CFG = parse_backend_config(ARGV, Path(__file__).parent)   # required backend=NAME, checked against backends.json
BENCH        = parse_benchmark_args(ARGV, extra_keys=("backend", "date"), required=not HEALTH_CHECK)   # full / mini / custom; none (None) for a health check
LAYOUT       = parse_layout_args(ARGV, collect=True)   # layout=... / date=... pick the run folder(s) (date= alone: each layout's run of that date); None: every run with something left to collect
LNN          = BACKEND_CFG["lnn"]   # the backend's topology, from backends.json
DO_SPAM      = True
RUN_FOLDERS  = [] if BENCH is None else find_collect_run_folders(qpu_run_prefix("IQM_EQE1", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM), LAYOUT, parse_date_arg(ARGV))
HC_FOLDER    = latest_run_folder(health_check_run_name("IQM_EQE1", BACKEND_NAME, STANDALONE_HEALTH_CHECK_REPS, STANDALONE_HEALTH_CHECK_NSHOTS)) or health_check_run_name("IQM_EQE1", BACKEND_NAME, STANDALONE_HEALTH_CHECK_REPS, STANDALONE_HEALTH_CHECK_NSHOTS)   # the standalone health check's folder

# ── Connect to EQE1 via MQSS ──────────────────────────────────────────────────
print(f"\n[{BACKEND_NAME} — connection]")
token   = read_token("iqm_eqe_token")
adapter = MQSSQiskitAdapter(token=token)

# MQSS has no retrieve_job(): a status query takes a job request naming the resource (as in source/core/qbcs.py)
JOB_TYPE = CircuitJobRequest(circuits="", circuit_format="qasm", resource_name=BACKEND_NAME,
                             shots=1, no_modify=True, queued=False)

def overview():
    return [f"{BACKEND_NAME}: pending jobs={adapter.get_backend(BACKEND_NAME).num_pending_jobs}"]

run_status_checks(BACKEND_NAME, RUN_FOLDERS, HC_FOLDER, HEALTH_CHECK,
                 remote_status=lambda job_id: adapter.client.job_status(job_id, JOB_TYPE).value,
                 overview=overview, recent_jobs=None)   # MQSS's API has no job listing
