"""
qbcs_real_submit_iqm_vlq.py
Submit a BACQ benchmarking run (the benchmark size is given on the command line) to
IT4Innovations' VLQ (IQM Star 24) through LEXIS/HEAppE (qaas). Async — returns once every
chunk is uploaded and submitted. Run qbcs_real_collect_iqm_vlq.py separately (with the same
arguments) to retrieve the results.

Chunks go out in jobs of up to 12 chunks (each chunk with its SPAM circuits; 816 circuits per
job in the full run, so its 480 chunks are 40 jobs; the mini run's 8 chunks are one job). Jobs
of 48 chunks (3,264 circuits) were uploaded but then failed when IT4I forwarded them to IQM.
qaas uploads every circuit as its own HTTP request (~0.3 s each): the full submission takes
about 2.7 h, plus transpilation. Login: the LEXIS refresh token saved in <project root>/tokens/iqm_vlq_token
is reused; a browser login is asked for only when there is none or it was rejected.

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
from source.core.records.run_names import qpu_run_name, service_name, dated_run_folder
usage_if_requested(sys.argv[1:], make_usage(__file__, "Submit a BACQ benchmark run to IQM VLQ (LEXIS/HEAppE). Async: returns once every chunk is uploaded; collect the results with qbcs_real_collect_iqm_vlq.py (same arguments).", layout="submit_live"))   # no arguments or --help: print the usage and do nothing

from pathlib import Path
from source.core.qpu import iqm_vlq as VLQ
from source.core.runner_support.benchmark_args import parse_benchmark_args, parse_backend_config, parse_layout_args, pop_force_new_run_flag, pop_retry_health_check_flag
from source.core.runner_support.run_submission import submit_async_run, health_check_retry_folder
from source.core.qbcs import VLQ_CHUNKS_PER_JOB

# ── Config ────────────────────────────────────────────────────────────────────
FORCE_NEW_RUN, ARGV = pop_force_new_run_flag(sys.argv[1:])   # --force-new-run: start a new run folder even if the latest run of these settings is unfinished
RETRY_HC, ARGV = pop_retry_health_check_flag(ARGV)   # --retry-health-check (from the *_with_retries.py script): only the health check after the last chunks, in the latest run
BACKEND_NAME, BACKEND_CFG = parse_backend_config(ARGV, Path(__file__).parent)   # required backend=NAME, checked against backends.json
SERVICE_NAME = service_name("IQM_VLQ")   # the access platform, as records and qbcs call it (source/core/records/run_names.py)
BENCH            = parse_benchmark_args(ARGV, extra_keys=("backend",))   # full / mini / custom — see source/core/runner_support/benchmark_args.py
LAYOUT           = parse_layout_args(ARGV, live_calibration="IQM VLQ")   # layout=optimized | random | lowest | 15,14,21 — see source/core/runner_support/benchmark_args.py
SAMPLE_SIZE      = BENCH.sample_size
LNN              = BACKEND_CFG["lnn"]   # the backend's topology, from backends.json
DO_SPAM          = True
UNDATED          = qpu_run_name("IQM_VLQ", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM, LAYOUT.tag)
RUN_FOLDER       = health_check_retry_folder(UNDATED) if RETRY_HC else dated_run_folder(UNDATED, new_run=FORCE_NEW_RUN)
CHUNKS_TO_SUBMIT = SAMPLE_SIZE   # the whole run; lower it to submit only the first N chunks
CHUNKS_PER_JOB   = min(VLQ_CHUNKS_PER_JOB, SAMPLE_SIZE)   # full: 816 circuits, ~4 min upload each (48-chunk jobs failed at IT4I → IQM; see readme §3)

# ── Connect to VLQ ────────────────────────────────────────────────────────────
print(f"\n[{BACKEND_NAME} — connection]")
status = VLQ.qpu_status()
print(f"  Status:          QPU: {status.get('QPU_status')}, control: {status.get('QCCW_status')}, "
      f"total: {status.get('total_status')} ({status.get('timestamp')})")
real_backend = VLQ.lexis_backend(log_file=f"datafiles/{RUN_FOLDER}/{VLQ.LEXIS_LOG_NAME}")   # py4lexis' log, in the run folder
cal          = VLQ.attach_calibration(real_backend)   # the calibration it runs with → the run's qubits
print(f"  Backend:         {real_backend.backend_name}")
print(f"  Qubits:          {real_backend.num_qubits} + resonator {VLQ.RESONATOR}")
print(f"  Calibration set: {cal['calibration_set_id']}")

# ── Config print, health checks, build, submit and log: shared by every async submit script ──
# (build() loads the cached all_to_all circuits, QPY version 13 — iqm_vlq.venv's Qiskit 2.1 reads it)
submit_async_run(
    run_folder = RUN_FOLDER, backend_name = BACKEND_NAME, service_name = SERVICE_NAME,
    real_backend = real_backend, collect_service = real_backend,   # qaas backend — used for retrieve_job()
    bench = BENCH, layout = LAYOUT, lnn = LNN, do_spam = DO_SPAM, chunks_to_submit = CHUNKS_TO_SUBMIT,
    submit_kwargs = dict(chunks_per_qpu_job_batch = CHUNKS_PER_JOB),
    config_extra = [("Chunks per job", f"{CHUNKS_PER_JOB}")],
    log_extra_lines = [f"Calibration set:  {cal['calibration_set_id']}"],
    retry_health_check = RETRY_HC,
    collect_script = "qbcs_real_collect_iqm_vlq.py",
)
