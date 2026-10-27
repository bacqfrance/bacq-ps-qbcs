"""
benchmark_fake_fez_lnn.py
The LNN benchmark on the local Aer simulator with the noise of Qiskit's built-in
FakeFez backend — a frozen snapshot of IBM Fez's calibration that ships with
qiskit-ibm-runtime. No network calls, no token. run_local_simulation() prints the time of each stage.

Same path as source/benchmark_scripts/real_qpu_scripts/IBM/qbcs_real_simulate_ibm.py: FakeFez's target
is saved to datafiles/<RUN_FOLDER>/calibration_snapshot.json, an Aer noise
model is built from it, physical qubits are picked from it, and every chunk is
transpiled onto them, scheduled if scheduling_simulation=true (idle qubits decay, as in the IBM simulation) or not if false (faster).

Run in venvs/ibm.venv (qiskit-ibm-runtime; no myQLM). The LNN circuits of the same
benchmark size must already be cached — run benchmark_noiseless_lnn.py with the same
arguments in venvs/local_sim.venv first.

Usage (see source/core/runner_support/benchmark_args.py; the benchmark is required — without arguments, or with --help, the usage is printed):
  python3 benchmark_fake_fez_lnn.py mini                               mini benchmark: 8 samples, Nt = 1, 3, 200 shots
  python3 benchmark_fake_fez_lnn.py full scheduling_simulation=false   full benchmark: 480 samples, Nt = 1…19, 10 shots
  python3 benchmark_fake_fez_lnn.py mini nshots_per_circuit=N
  python3 benchmark_fake_fez_lnn.py nshots_per_circuit=N n_samples=M n_depths=K   (or nt_list=1,5,9)
Any of them takes the optional scheduling_simulation=true | false (default true): true transpiles the circuits scheduled, so idle qubits relax through T1/T2 as on
the QPU (slower: about 4 times the time of false on the full benchmark); false leaves idle-qubit relaxation out. It is recorded in run_info.json,
and a run folder started with the other value is refused.
Any of them takes layout=optimized (default) | random | lowest | 15,14,21 (calibration_file= is refused: FakeFez's calibration is built in).
The run folder is <benchmark>_sim-fake-fez_local-aer_lnn-True_service-local_spam-True_layout-<mode>_<YYYY-MM-DD>_#<N> (source/core/records/run_names.py).
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))   # datafiles/ paths are relative to the project root
from source.core.records.run_names import local_run_name, dated_run_folder
from source.core.runner_support.benchmark_args import make_usage, usage_if_requested   # light (numpy only): the usage is printed before anything else loads
usage_if_requested(sys.argv[1:], make_usage(__file__, "Run the LNN BACQ benchmark on the local Aer simulator with the noise of Qiskit's built-in FakeFez backend (no network, no token). Run it in venvs/ibm.venv; the LNN circuits of the same benchmark size must already be cached (benchmark_noiseless_lnn.py).", backend=False, layout="simulate_builtin"))   # no arguments or --help: print the usage and do nothing
from pathlib import Path
from source.core.runner_support.benchmark_args import parse_benchmark_args, parse_layout_args, parse_scheduling_arg
from source.core.runner_support.run_local_simulation import run_local_simulation

BENCH      = parse_benchmark_args(sys.argv[1:], extra_keys=("scheduling_simulation",))
LAYOUT     = parse_layout_args(sys.argv[1:], no_calibration_file=True)
SCHEDULING = parse_scheduling_arg(sys.argv[1:])   # scheduling_simulation=true|false (optional, default true): idle qubits relax (true) or not (false)
RUN_FOLDER = dated_run_folder(local_run_name(BENCH.tag, "sim-fake-fez", lnn=True, layout_tag=LAYOUT.tag))
BACKEND    = "Aer with FakeFez noise (local)"

def noisy_setup():
    """Noise model from FakeFez's bundled calibration (offline)."""
    from qiskit_ibm_runtime.fake_provider import FakeFez
    from source.core.qpu.qpu_snapshot import SNAPSHOT_FILENAME, save_calibration_snapshot, noisy_sampler_from_snapshot
    snapshot_path = Path("datafiles") / RUN_FOLDER / SNAPSHOT_FILENAME
    if not snapshot_path.exists():
        save_calibration_snapshot(FakeFez(), snapshot_path)
    snapshot_backend, noise_model, sampler = noisy_sampler_from_snapshot(snapshot_path)
    return snapshot_path, snapshot_backend, sampler

run_local_simulation(run_folder=RUN_FOLDER, bench=BENCH, lnn=True, backend_label=BACKEND, do_spam=True, noisy_setup=noisy_setup, layout=LAYOUT,
                     scheduling_simulation=SCHEDULING)
