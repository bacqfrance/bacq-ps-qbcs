"""
qbcs_real_collect_ibm.py
Collect BACQ benchmarking results from real IBM QPU.
Can be run in a new session — reconstructs everything from job records on disk.
Collects the jobs that have finished, in any order, without waiting for the rest — run it
again later for those. Without layout= and date=, every run of the benchmark with something left
to collect is collected in turn. An incomplete run is analyzed as +nsamples-<collected> files that list what is missing.

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
from source.core.records.run_names import qpu_run_prefix, service_name
from source.core.qpu.tokens import read_token
usage_if_requested(sys.argv[1:], make_usage(__file__, "Collect the results of a BACQ benchmark run from IBM: every job that has finished, in any order, then merge and analyze. Run it again later for the rest. Needs the same arguments as the submit script.", layout="collect"))   # no arguments or --help: print the usage and do nothing

from qiskit_ibm_runtime import QiskitRuntimeService

from source.core.runner_support.benchmark_args import parse_benchmark_args, parse_backend_config, parse_layout_args, find_collect_run_folders, parse_date_arg
from source.core.runner_support.run_collection import collect_runs

# ── Config — must match submit script ────────────────────────────────────────
BACKEND_NAME, BACKEND_CFG = parse_backend_config(sys.argv[1:], Path(__file__).parent)   # required backend=NAME, checked against backends.json
SERVICE_NAME = service_name("IBM")   # the access platform, as records and qbcs call it (source/core/records/run_names.py)
BENCH        = parse_benchmark_args(sys.argv[1:], extra_keys=("backend", "date"))   # full / mini / custom — see source/core/runner_support/benchmark_args.py
LAYOUT       = parse_layout_args(sys.argv[1:], collect=True)   # layout=... / date=... pick the run folder(s) (date= alone: each layout's run of that date); None: every run with something left to collect
LNN          = BACKEND_CFG["lnn"]   # the backend's topology, from backends.json
DO_SPAM      = True
RUN_FOLDERS  = find_collect_run_folders(qpu_run_prefix("IBM", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM), LAYOUT, parse_date_arg(sys.argv[1:]))
PATHS        = [f"datafiles/{folder}" for folder in RUN_FOLDERS]   # oldest first

# ── Connect to IBM ────────────────────────────────────────────────────────────
print(f"\n[{BACKEND_NAME} — connection]")
token   = read_token("ibm_token")
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)

# ── Health checks, then the finished chunks (non-blocking), merge, analyze (source/core/runner_support/run_collection.py) ──
collect_runs(PATHS, service, SERVICE_NAME, BACKEND_NAME, BENCH, LNN, DO_SPAM)
