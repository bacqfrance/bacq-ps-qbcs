"""
qbcs_real_simulate_iqm_eqe1.py
Run the BACQ benchmark on the local Aer simulator with IQM EQE1's calibration —
same circuits and settings as qbcs_real_submit_iqm_eqe1.py, but no QPU time.

  1. EQE1's SDK (MQSS) exposes no calibration, so the calibration is a file: calibration_file=PATH
     (the operators' calibration_data_Euro-Q-Exa_<date>_active_qubits.json, converted
     automatically, or an already converted snapshot), by default the newest file in
     calibration_input_files/Euro-Q-Exa/. It is saved as datafiles/<RUN_FOLDER>/calibration_snapshot.json,
     with a warning. A rerun reuses that copy (it must be the same calibration) and resumes
     an unfinished run (a finished one is followed by a new dated folder); remove the run
     folder to start over.
  2. Build an offline backend + Aer noise model from the snapshot: depolarizing gate
     errors and readout errors, and T1/T2 relaxation over gate durations and, with scheduling_simulation=true, while a
     qubit waits for the others. The file has no gate durations, so IQM Emerald's are assumed. The file's own topology is used — couplers die and revive over
     time, so it may differ from the live backend's — and qubits with incomplete
     calibration (NaN) are left out of it.
  3. build(), then pick the run's physical qubits as layout= says, on the snapshot's Target.
  4. Health check before and after the simulation, like the submit script, under
     the same noise model.
  5. submit(service="local", layout_backend=<snapshot backend>) — every chunk,
     SPAM on the same qubits — then analyze_results(). Post-process with
     source/analysis_scripts/run_post_process.py --include-simulations-real: every layout of the
     QPU's sim-qpu-noise_<provider>-<backend> runs is already in SIMULATED_QPUS of RUN_THESE (USER_MANUAL §10.2).

Qubit layout (required — see the usage): EQE1 / PIAST-Q report no calibration, so a real submission uses a RANDOM
layout unless calibration_file= is given. layout=random|lowest|optimized|15,14,21 says which one to mimic;
layout=optimized scores the placements with the calibration file. The run folder ends in _layout-<mode>.

Run from the project root in venvs/iqm_eqe1.venv.

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
from source.core.records.run_names import qpu_run_name, dated_run_folder
usage_if_requested(sys.argv[1:], make_usage(__file__, "Run the BACQ benchmark on the local Aer simulator with IQM EQE1's calibration: same circuits and settings as qbcs_real_submit_iqm_eqe1.py, but no QPU time.", layout="simulate_required"))   # no arguments or --help: print the usage and do nothing
from pathlib import Path


from source.core.runner_support.benchmark_args import parse_benchmark_args, DT, parse_backend_config, parse_layout_args, parse_scheduling_arg
from source.core.qpu.calibration_input import use_calibration_file, newest_provider_file
from source.core.runner_support.run_simulation import simulate_run
from source.core.qpu.qpu_snapshot import SNAPSHOT_FILENAME

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_NAME, BACKEND_CFG = parse_backend_config(sys.argv[1:], Path(__file__).parent)   # required backend=NAME, checked against backends.json
BENCH            = parse_benchmark_args(sys.argv[1:], extra_keys=("backend", "scheduling_simulation"))   # full / mini / custom — see source/core/runner_support/benchmark_args.py
LAYOUT           = parse_layout_args(sys.argv[1:], calibration_required=True, layout_required=True, calibration_any_layout=True)   # layout=... [calibration_file=...] — see source/core/runner_support/benchmark_args.py
SCHEDULING       = parse_scheduling_arg(sys.argv[1:])   # scheduling_simulation=true|false (optional, default true): idle qubits relax (true) or not (false)
LNN              = BACKEND_CFG["lnn"]   # the backend's topology, from backends.json
DO_SPAM          = True
RUN_FOLDER       = dated_run_folder(qpu_run_name("IQM_EQE1", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM, LAYOUT.tag, data_type="sim-qpu-noise"))
SEED             = None   # Aer sampling seed; None = fresh randomness per chunk

# ── Calibration snapshot (newest local file, then reuse) ──────────────────────
snapshot_path = Path("datafiles") / RUN_FOLDER / SNAPSHOT_FILENAME
_, _, CAL_SOURCE = use_calibration_file(LAYOUT.calibration_file or newest_provider_file("Euro-Q-Exa"),
                                        Path("datafiles") / RUN_FOLDER, BACKEND_NAME)

# ── Simulate: build, qubits, health checks, every chunk left, submit.log, analysis (source/core/runner_support/run_simulation.py) ──
simulate_run(run_folder=RUN_FOLDER, bench=BENCH, layout=LAYOUT, lnn=LNN, snapshot_path=snapshot_path, cal_source=CAL_SOURCE,
             do_spam=DO_SPAM, dt=DT, seed=SEED, scheduling_simulation=SCHEDULING)
