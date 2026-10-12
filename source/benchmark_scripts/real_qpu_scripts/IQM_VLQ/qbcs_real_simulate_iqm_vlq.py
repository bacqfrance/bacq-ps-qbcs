"""
qbcs_real_simulate_iqm_vlq.py
Run a BACQ benchmark on the local Aer simulator with VLQ's calibration — same
circuits, settings and physical-qubit selection as qbcs_real_submit_iqm_vlq.py, but no
QPU time and no LEXIS login.

  1. Download VLQ's current calibration set from IT4I's public API (no authentication)
     once and save it, with the effective snapshot built from it, to
     datafiles/<RUN_FOLDER>/ (vlq_calibration.json, calibration_snapshot.json).
     A rerun of an unfinished run reuses the files (with a warning; a finished run is
     followed by a new dated folder, which downloads fresh calibration); remove the
     run folder to start over. With calibration_file=PATH (a run's vlq_calibration.json) that
     calibration is used instead, and nothing is downloaded.
  2. Build an offline backend + Aer noise model from the snapshot. The resonator is
     folded into an all-to-all CZ (source/core/qpu/iqm_vlq.snapshot_from_calibration): error
     1 - F_cz · F_move, duration 2·MOVE + CZ; prx error from the Clifford RB fidelity per native gate
     (iqm_vlq.prx_fidelity); readout and T1/T2 as measured.
  3. build(), then pick the run's physical qubits exactly as a real submission would: the
     prx / MOVE / CZ roles come from transpiling onto an offline VLQ target built from the
     downloaded architecture (iqm-client; no login), graded from the calibration.
  4. Health check before and after the simulation, on the run's qubits.
  5. submit(service="local", layout_backend=<snapshot backend>), then analyze_results().
     Post-process with source/analysis_scripts/run_post_process.py --include-simulations-real (add the run name without
     its <benchmark>_, sim-qpu-noise_iqm-vlq_…, to RUN_THESE — USER_MANUAL §10.2).

Run from the project root in venvs/iqm_vlq.venv (iqm-client builds the offline target; no login needed).

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
usage_if_requested(sys.argv[1:], make_usage(__file__, "Run the BACQ benchmark on the local Aer simulator with VLQ's calibration: same circuits, settings and qubit selection as qbcs_real_submit_iqm_vlq.py, but no QPU time and no LEXIS login.", layout="simulate"))   # no arguments or --help: print the usage and do nothing
from pathlib import Path

from source.core.qpu import iqm_vlq as VLQ
from source.core.runner_support.benchmark_args import parse_benchmark_args, parse_backend_config, parse_layout_args
from source.core.runner_support.run_simulation import simulate_run
from source.core.qpu.qpu_snapshot import SNAPSHOT_FILENAME

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_NAME, BACKEND_CFG = parse_backend_config(sys.argv[1:], Path(__file__).parent)   # required backend=NAME, checked against backends.json
BENCH            = parse_benchmark_args(sys.argv[1:], extra_keys=("backend",))   # full / mini / custom — see source/core/runner_support/benchmark_args.py
LAYOUT           = parse_layout_args(sys.argv[1:], calibration_any_layout=True)   # layout=... [calibration_file=...] — see source/core/runner_support/benchmark_args.py
LNN              = BACKEND_CFG["lnn"]   # the backend's topology, from backends.json
DO_SPAM          = True
DT               = 0.2
RUN_FOLDER       = dated_run_folder(qpu_run_name("IQM_VLQ", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM, LAYOUT.tag, data_type="sim-qpu-noise"))
CALIBRATION_SET  = "default"   # or a calibration set UUID, e.g. the one a real run recorded
SEED             = None        # Aer sampling seed; None = fresh randomness per chunk

# ── Calibration (download once, then reuse; or calibration_file=, e.g. to re-simulate an old run) ──
run_dir       = Path("datafiles") / RUN_FOLDER
snapshot_path = run_dir / SNAPSHOT_FILENAME
cal, CAL_SOURCE = VLQ.use_run_calibration(run_dir, LAYOUT.calibration_file, CALIBRATION_SET)

# ── Simulate: build, qubits, health checks, every chunk left, submit.log, analysis (source/core/runner_support/run_simulation.py) ──
simulate_run(run_folder=RUN_FOLDER, bench=BENCH, layout=LAYOUT, lnn=LNN, snapshot_path=snapshot_path, cal_source=CAL_SOURCE,
             do_spam=DO_SPAM, dt=DT, seed=SEED,
             customize_backend=lambda b: setattr(b, "vlq_calibration", cal))   # qubit selection as on the real backend
