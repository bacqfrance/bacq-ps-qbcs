"""
qbcs_real_submit_ibm.py
Submit a BACQ benchmarking run (the benchmark size is given on the command line) to a real IBM QPU.
Async — returns immediately after all chunks are submitted.
Run qbcs_real_collect_ibm.py separately to retrieve the results.

Benchmark size (required — there is no default), from the command line (source/core/runner_support/benchmark_args.py; readme: "Benchmark size and run folders"):
  mini · full · mini|full nshots_per_circuit=N ·
  nshots_per_circuit=N n_samples=M n_depths=K (or nt_list=1,5,9)
"""

import os
import sys
import contextlib
import statistics
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # datafiles/ paths are relative to the project root
from source.core.runner_support.benchmark_args import make_usage, usage_if_requested   # light (numpy only): the usage is printed before anything else loads
from source.core.records.run_names import qpu_run_name, service_name, dated_run_folder
from source.core.qpu.tokens import read_token
usage_if_requested(sys.argv[1:], make_usage(__file__, "Submit a BACQ benchmark run to a real IBM QPU. Async: returns once every chunk is submitted; collect the results with qbcs_real_collect_ibm.py (same arguments).", layout="submit_live", instance=True))   # no arguments or --help: print the usage and do nothing
from pathlib import Path

from source.core.runner_support.benchmark_args import parse_benchmark_args, parse_backend_config, parse_layout_args, parse_instance_arg, pop_force_new_run_flag
from source.core.qpu.providers import ibm_batch, connect_ibm, ibm_instance_for_runs
from source.core.qpu.qpu_snapshot import snapshot_log_lines, SNAPSHOT_FILENAME
from source.core.runner_support.run_submission import submit_async_run
from source.core.qbcs import IBM_CHUNKS_PER_JOB

# ── Config ────────────────────────────────────────────────────────────────────
FORCE_NEW_RUN, ARGV = pop_force_new_run_flag(sys.argv[1:])   # --force-new-run: start a new run folder even if the latest run of these settings is unfinished
BACKEND_NAME, BACKEND_CFG = parse_backend_config(ARGV, Path(__file__).parent)   # required backend=NAME, checked against backends.json
SERVICE_NAME = service_name("IBM")   # the access platform, as records and qbcs call it (source/core/records/run_names.py)
BENCH            = parse_benchmark_args(ARGV, extra_keys=("backend", "instance"))   # full / mini / custom — see source/core/runner_support/benchmark_args.py
INSTANCE_ARG     = parse_instance_arg(ARGV)   # instance=NAME; default the Open Plan instance; a run's recorded instance wins (providers.py: ibm_instance_for_runs)
LAYOUT           = parse_layout_args(ARGV, live_calibration="IBM")   # layout=... (live calibration only) — see source/core/runner_support/benchmark_args.py
SAMPLE_SIZE      = BENCH.sample_size
LNN              = BACKEND_CFG["lnn"]   # the backend's topology, from backends.json
DO_SPAM          = True
RUN_FOLDER       = dated_run_folder(qpu_run_name("IBM", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM, LAYOUT.tag), new_run=FORCE_NEW_RUN)
CHUNKS_TO_SUBMIT = SAMPLE_SIZE   # the whole run; lower it to submit only the first N chunks
CHUNKS_PER_JOB   = min(IBM_CHUNKS_PER_JOB, SAMPLE_SIZE)   # full: 10 jobs of 48 chunks; mini: all 8 chunks in one job
# ── Connect to IBM ────────────────────────────────────────────────────────────
print(f"\n[{BACKEND_NAME} — connection]")
token   = read_token("ibm_token")
service, INSTANCE = connect_ibm(token, ibm_instance_for_runs(INSTANCE_ARG, [RUN_FOLDER]), BACKEND_NAME)
real_backend = service.backend(BACKEND_NAME)

# ── Print calibration info ────────────────────────────────────────────────────
props    = real_backend.properties()
cz_gates = [g for g in props.gates if g.gate == "cz"]
avg_cz   = statistics.median(g.parameters[0].value for g in cz_gates)
t1_vals, t2_vals = [], []
for q in range(real_backend.num_qubits):
    try: t1_vals.append(props.qubit_property(q, "T1")[0])
    except Exception: pass
    try: t2_vals.append(props.qubit_property(q, "T2")[0])
    except Exception: pass
t1_avg = sum(t1_vals) / len(t1_vals) if t1_vals else 0
t2_avg = sum(t2_vals) / len(t2_vals) if t2_vals else 0

print(f"\n[{BACKEND_NAME} — live calibration, whole device]")
print(f"  CZ error (median): {avg_cz:.3e}")
print(f"  T1: {t1_avg*1e6:.1f}µs   T2: {t2_avg*1e6:.1f}µs")

# ── Health checks: one IBM Batch per check, closed when it is done ────────────
# The batch stays open across the retries of one check (a retry just submits another job into it).
@contextlib.contextmanager
def health_check_batch():
    with ibm_batch(real_backend) as sampler:
        yield dict(sampler=sampler, service=service)


def print_run_qubit_calibration(run_qubits):
    """Calibration of the run's own qubits (virtual 0-1-2 → run_qubits, a chain), from the run's snapshot."""
    print(f"\n[{BACKEND_NAME} — calibration of the run qubits {run_qubits}]")
    for line in snapshot_log_lines(Path("datafiles") / RUN_FOLDER / SNAPSHOT_FILENAME, run_qubits):
        print(f"  {line}")


# ── Config print, health checks, build, submit and log: shared by every async submit script ──
submit_async_run(
    run_folder = RUN_FOLDER, backend_name = BACKEND_NAME, service_name = SERVICE_NAME,
    real_backend = real_backend, collect_service = service,
    bench = BENCH, layout = LAYOUT, lnn = LNN, do_spam = DO_SPAM, chunks_to_submit = CHUNKS_TO_SUBMIT,
    submit_kwargs = dict(chunks_per_qpu_job_batch = CHUNKS_PER_JOB),
    config_extra = [("Chunks per job", f"{CHUNKS_PER_JOB}"), ("IBM instance", f"{INSTANCE['name']} ({INSTANCE['plan']})")],
    log_extra_lines = [f"IBM instance: {INSTANCE['name']} ({INSTANCE['plan']})"],
    instance = (INSTANCE["name"], INSTANCE["plan"]),
    health_check_session = health_check_batch,
    after_layout = print_run_qubit_calibration,
    collect_script = "qbcs_real_collect_ibm.py",
)
