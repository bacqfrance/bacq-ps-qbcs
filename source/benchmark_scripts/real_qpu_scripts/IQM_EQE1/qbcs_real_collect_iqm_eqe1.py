"""
qbcs_real_collect_iqm_eqe1.py
Collect BACQ benchmarking results from real IQM QPU (EQE1 via MQSS).
Can be run in a new session — reconstructs everything from job records on disk.
Collects the jobs that have finished, in any order, without waiting for the rest — run it
again later for those. An incomplete run is analyzed as +nsamples-<collected> files that list what is missing.

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
usage_if_requested(sys.argv[1:], make_usage(__file__, "Collect the results of a BACQ benchmark run from IQM EQE1 (MQSS): every job that has finished, in any order, then merge and analyze. Run it again later for the rest. Needs the same arguments as the submit script.", layout="collect"))   # no arguments or --help: print the usage and do nothing

from mqss.qiskit_adapter import MQSSQiskitAdapter
from source.core.runner_support.benchmark_args import parse_benchmark_args, parse_backend_config, parse_layout_args, find_collect_run_folder
from source.core.runner_support.run_collection import collect_run

# ── Config — must match submit script ────────────────────────────────────────
BACKEND_NAME, BACKEND_CFG = parse_backend_config(sys.argv[1:], Path(__file__).parent)   # required backend=NAME, checked against backends.json
SERVICE_NAME = service_name("IQM_EQE1")   # the access platform, as records and qbcs call it (source/core/records/run_names.py)
BENCH        = parse_benchmark_args(sys.argv[1:], extra_keys=("backend",))   # full / mini / custom — see source/core/runner_support/benchmark_args.py
LAYOUT       = parse_layout_args(sys.argv[1:], collect=True)   # layout=... names the run folder; None: the only one on disk
LNN          = BACKEND_CFG["lnn"]   # the backend's topology, from backends.json
DO_SPAM      = True
RUN_FOLDER   = find_collect_run_folder(qpu_run_prefix("IQM_EQE1", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM), LAYOUT)
PATH         = f"datafiles/{RUN_FOLDER}"

# ── Connect to EQE1 via MQSS ──────────────────────────────────────────────────
print(f"\n[{BACKEND_NAME} — connection]")
token   = read_token("iqm_eqe_token")
adapter = MQSSQiskitAdapter(token=token)

# ── Health checks, then the finished chunks (non-blocking), merge, analyze (source/core/runner_support/run_collection.py) ──
collect_run(PATH, adapter, SERVICE_NAME, BACKEND_NAME, BENCH, LNN, DO_SPAM)
