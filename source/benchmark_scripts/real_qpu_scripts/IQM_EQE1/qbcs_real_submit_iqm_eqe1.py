"""
qbcs_real_submit_iqm_eqe1.py
Submit BACQ benchmarking run to real IQM QPU (EQE1 via MQSS).
Async — returns immediately after all chunks submitted.
Run qbcs_real_collect_iqm_eqe1.py separately to retrieve results.

Benchmark size (required — there is no default), from the command line (source/core/runner_support/benchmark_args.py; readme: "Benchmark size and run folders"):
  mini · full · mini|full nshots_per_circuit=N ·
  nshots_per_circuit=N n_samples=M n_depths=K (or nt_list=1,5,9)
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # datafiles/ paths are relative to the project root
from source.core.runner_support.benchmark_args import make_usage, usage_if_requested   # light (numpy only): the usage is printed before anything else loads
from source.core.records.run_names import qpu_run_name, service_name
from source.core.qpu.tokens import read_token
usage_if_requested(sys.argv[1:], make_usage(__file__, "Submit a BACQ benchmark run to IQM EQE1 (MQSS). Async: returns once every chunk is submitted; collect the results with qbcs_real_collect_iqm_eqe1.py (same arguments).", layout="submit_no_live"))   # no arguments or --help: print the usage and do nothing

from pathlib import Path
from mqss.qiskit_adapter import MQSSQiskitAdapter
from source.core.runner_support.benchmark_args import parse_benchmark_args, parse_backend_config, parse_layout_args
from source.core.qbcs import IQM_MAX_CIRCUITS_PER_JOB
from source.core.runner_support.run_submission import submit_async_run

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_NAME, BACKEND_CFG = parse_backend_config(sys.argv[1:], Path(__file__).parent)   # required backend=NAME, checked against backends.json
SERVICE_NAME = service_name("IQM_EQE1")   # the access platform, as records and qbcs call it (source/core/records/run_names.py)
BENCH            = parse_benchmark_args(sys.argv[1:], extra_keys=("backend",))   # full / mini / custom — see source/core/runner_support/benchmark_args.py
LAYOUT           = parse_layout_args(sys.argv[1:], calibration_required=True)   # layout=... [calibration_file=...] — see source/core/runner_support/benchmark_args.py
SAMPLE_SIZE      = BENCH.sample_size
LNN              = BACKEND_CFG["lnn"]   # the backend's topology, from backends.json
DO_SPAM          = True
RUN_FOLDER       = qpu_run_name("IQM_EQE1", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM, LAYOUT.tag)
CHUNKS_TO_SUBMIT = SAMPLE_SIZE   # the whole run; lower it to submit only the first N chunks
MAX_CIRCUITS_PER_JOB = IQM_MAX_CIRCUITS_PER_JOB[SERVICE_NAME]   # 70; chunks are packed into jobs of at most this many circuits (each chunk with its SPAM circuits)

# ── Connect to EQE1 via MQSS ──────────────────────────────────────────────────
print(f"\n[{BACKEND_NAME} — connection]")
token   = read_token("iqm_eqe_token")
adapter      = MQSSQiskitAdapter(token=token)
real_backend = adapter.get_backend(BACKEND_NAME)

print(f"  Backend:      {real_backend.name}")
print(f"  Pending jobs: {real_backend.num_pending_jobs}")


# ── Config print, health checks, build, submit and log: shared by every async submit script ──
submit_async_run(
    run_folder = RUN_FOLDER, backend_name = BACKEND_NAME, service_name = SERVICE_NAME,
    real_backend = real_backend, collect_service = adapter,
    bench = BENCH, layout = LAYOUT, lnn = LNN, do_spam = DO_SPAM, chunks_to_submit = CHUNKS_TO_SUBMIT,
    submit_kwargs = dict(max_circuits_per_job = MAX_CIRCUITS_PER_JOB),
    config_extra = [("Max circuits/job", f"{MAX_CIRCUITS_PER_JOB}")],
    log_extra_lines = lambda: [f"Pending jobs at submission: {real_backend.num_pending_jobs}"],
    collect_script = "qbcs_real_collect_iqm_eqe1.py",
)
