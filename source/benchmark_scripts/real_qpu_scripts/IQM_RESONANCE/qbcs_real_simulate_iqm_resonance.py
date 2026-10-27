"""
qbcs_real_simulate_iqm_resonance.py
Run the BACQ benchmark on the local Aer simulator with a real IQM Resonance QPU's
calibration — same circuits, settings and physical-qubit selection as
qbcs_real_submit_iqm_resonance.py, but no QPU time.

  1. Download BACKEND_NAME's calibration (get_backend(use_metrics=True)) once and save it to
     datafiles/<RUN_FOLDER>/calibration_snapshot.json (source/core/qpu/qpu_snapshot.py). For Sirius (a star machine with a
     computational resonator) the raw calibration set is downloaded instead, saved as star_calibration.json, and folded
     into an all-to-all snapshot (source/core/qpu/iqm_star.py); layout=optimized then scores the qubits on it as on VLQ.
     A rerun of an unfinished run reuses that file (with a warning) — same calibration,
     no connection needed (a finished run is followed by a new dated folder, which
     downloads fresh calibration); remove the run folder to start over. With calibration_file=PATH
     the snapshot is taken from that file instead, and nothing is downloaded.
  2. Build an offline backend + Aer noise model from the snapshot (depolarizing
     gate errors, T1/T2 relaxation over gate durations and, with scheduling_simulation=true, while a
     qubit waits for the others, readout errors).
  3. build(), then pick the run's physical qubits from the snapshot exactly as a
     real submission would choose them.
  4. Health check before and after the simulation, like the submit script: the
     same transpiled 3-qubit circuit, on the run's qubits, under the same noise
     model — so it shows what that circuit would score with this calibration.
  5. submit(service="local", layout_backend=<snapshot backend>) — every chunk,
     SPAM on the same qubits — then analyze_results(). Post-process with
     source/analysis_scripts/run_post_process.py --include-simulations-real: every layout of the
     QPU's sim-qpu-noise_<provider>-<backend> runs is already in SIMULATED_QPUS of RUN_THESE (USER_MANUAL §10.2).

Run from the project root in venvs/iqm_resonance.venv.

scheduling_simulation=true|false is optional (default true): true transpiles the circuits scheduled, so idle qubits relax through T1/T2 as on the QPU
(slower: about 4 times the time of false on the full benchmark); false leaves idle-qubit relaxation out. It is recorded in run_info.json, and a run
folder started with the other value is refused.

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
from source.core.qpu.providers import IQM_RESONANCE_URL
from source.core.records.run_names import qpu_run_name, dated_run_folder
from source.core.qpu.tokens import read_token
usage_if_requested(sys.argv[1:], make_usage(__file__, "Run the BACQ benchmark on the local Aer simulator with an IQM Resonance QPU's calibration: same circuits, settings and qubit selection as qbcs_real_submit_iqm_resonance.py, but no QPU time.", layout="simulate"))   # no arguments or --help: print the usage and do nothing
import json
from pathlib import Path

from iqm.qiskit_iqm import IQMProvider

from source.core.runner_support.benchmark_args import parse_benchmark_args, DT, parse_backend_config, parse_layout_args, parse_scheduling_arg
from source.core.qpu.calibration_input import use_calibration_file
from source.core.qpu.iqm_star import STAR_CALIBRATION_FILENAME, attach_star_calibration, save_star_calibration
from source.core.runner_support.run_simulation import simulate_run
from source.core.qpu.qpu_snapshot import SNAPSHOT_FILENAME, save_calibration_snapshot, reuse_calibration_snapshot

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_NAME, BACKEND_CFG = parse_backend_config(sys.argv[1:], Path(__file__).parent)   # required backend=NAME, checked against backends.json
BENCH            = parse_benchmark_args(sys.argv[1:], extra_keys=("backend", "scheduling_simulation"))   # full / mini / custom — see source/core/runner_support/benchmark_args.py
LAYOUT           = parse_layout_args(sys.argv[1:], calibration_any_layout=True)   # layout=... [calibration_file=...] — see source/core/runner_support/benchmark_args.py
SCHEDULING       = parse_scheduling_arg(sys.argv[1:])   # scheduling_simulation=true|false (optional, default true): idle qubits relax (true) or not (false)
LNN              = BACKEND_CFG["lnn"]   # the backend's topology, from backends.json
DO_SPAM          = True
RUN_FOLDER       = dated_run_folder(qpu_run_name("IQM_RESONANCE", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM, LAYOUT.tag, data_type="sim-qpu-noise"))
SEED             = None   # Aer sampling seed; None = fresh randomness per chunk

# ── Calibration snapshot (download once, then reuse) ──────────────────────────
snapshot_path = Path("datafiles") / RUN_FOLDER / SNAPSHOT_FILENAME
CAL_SOURCE = None
if LAYOUT.calibration_file:   # calibration_file=: the noise data of the simulation (and the layout scoring), instead of the download
    _, _, CAL_SOURCE = use_calibration_file(LAYOUT.calibration_file, Path("datafiles") / RUN_FOLDER, BACKEND_NAME)
elif not reuse_calibration_snapshot(snapshot_path):   # a rerun warns and reuses the saved calibration
    token        = read_token("iqm_resonance_token")
    provider     = IQMProvider(IQM_RESONANCE_URL, quantum_computer=BACKEND_NAME, token=token)
    real_backend = provider.get_backend(use_metrics=True)   # without use_metrics the Target has no errors
    if attach_star_calibration(real_backend) is not None:   # Sirius (a star machine): the raw calibration, folded into an all-to-all snapshot
        save_star_calibration(snapshot_path.parent, real_backend.vlq_calibration, real_backend.star_name)
    else:
        save_calibration_snapshot(real_backend, snapshot_path)
STAR_CAL_PATH = Path("datafiles") / RUN_FOLDER / STAR_CALIBRATION_FILENAME
STAR_CAL      = json.loads(STAR_CAL_PATH.read_text(encoding="utf-8")) if STAR_CAL_PATH.exists() else None   # the layout is scored on it, as on the real backend

# ── Simulate: build, qubits, health checks, every chunk left, submit.log, analysis (source/core/runner_support/run_simulation.py) ──
simulate_run(run_folder=RUN_FOLDER, bench=BENCH, layout=LAYOUT, lnn=LNN, snapshot_path=snapshot_path, cal_source=CAL_SOURCE,
             do_spam=DO_SPAM, dt=DT, seed=SEED, scheduling_simulation=SCHEDULING,
             customize_backend=None if STAR_CAL is None else lambda b: setattr(b, "vlq_calibration", STAR_CAL))   # Sirius: qubit selection as on the real backend
