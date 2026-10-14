# IQM EQE1 (via MQSS) — real-QPU submission

EQE1 is IQM's machine in [Euro-Q-Exa (EQE)](https://www.eurohpc-ju.europa.eu/eurohpc-quantum-computers/our-quantum-computers_en#euro-q-exa), one of the EuroHPC quantum
computers. It is hosted by LRZ, the Leibniz Supercomputing Centre ([lrz.de](https://www.lrz.de/en/)),
and reached through LRZ's Munich Quantum Software Stack (MQSS).

## 1. Prerequisites

Separate virtual environment: **iqm_eqe1.venv** (created at `venvs/iqm_eqe1.venv` in the project root). On top of the base
`local_sim.venv` package family, it needs one independent top-level package:

| Package | Version (this project's `iqm_eqe1.venv`) | Provides |
|---|---|---|
| `mqss-qiskit` | 0.2.0 | `mqss.qiskit_adapter.MQSSQiskitAdapter` |

Installing it pulls `mqss-client==0.2.1` as a dependency. Unlike IQM VLQ and AQT, this
does **not** force a Qiskit downgrade — `iqm_eqe1.venv` runs `qiskit==2.5.2`,
the same as the base env. Every other package in
this venv (`qiskit-aer`, `numpy`, `scipy`, …) is the same family as the
base requirements but independently resolved, so exact patch versions drift —
treat `requirements_iqm_eqe1.txt` as the source of truth, not the base
`venvs/requirements_local_sim.txt`.

**No myQLM in this venv.** The circuits are built in `local_sim.venv` — run
`build()` there first, with the same settings (the matching
`source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_lnn.py` script, with the same benchmark arguments, does it, see the main USER_MANUAL §4.3). It
caches every circuit as Qiskit QPY, plus the exact reference values the
analysis needs, and the scripts here only read those files. A script that
finds the cache incomplete stops with a message saying so.

**Build it in one pass**: fresh venv, then a single pinned
`pip install -r requirements_iqm_eqe1.txt` covering the base packages and
`mqss-qiskit` together, rather than adding `mqss-qiskit` on top of an
already-installed `local_sim.venv`.

## 2. Connecting

```python
from source.core.qpu.tokens import read_token
from mqss.qiskit_adapter import MQSSQiskitAdapter

token   = read_token("iqm_eqe_token")   # tokens/iqm_eqe_token
adapter = MQSSQiskitAdapter(token=token)
real_backend = adapter.get_backend("EQE1")
```

## 3. Config knobs (top of each script)

| Knob | Meaning |
|---|---|
| `BACKEND_NAME` | `backend=EQE1` on the command line — **required** by every script here (benchmark scripts, health check): without it, or with any other name, the script stops before it connects (MQSS also serves other machines — AQT20, Q20, QExa20, …), so nothing can be sent to another machine by mistake. The allowed names and each one's topology (`"lnn"`: linear-nearest-neighbour vs. all-to-all circuits) are in [`backends.json`](backends.json), which also sets `LNN`. |
| run size (shots, RC samples, evolution times) | the preset or sizes given on the command line (below), not edited in the script |
| `LNN` | linear-nearest-neighbour transpilation — taken from the backend's `"lnn"` entry in `backends.json` (`true` for EQE1) |
| `DO_SPAM` | also run SPAM calibration |
| `CHUNKS_TO_SUBMIT` | submit only the first N chunks |
| `MAX_CIRCUITS_PER_JOB` | 70 — most circuits in one EQE1 job. Chunks are packed: each job holds as many whole chunks (each with its own SPAM circuits) as fit under this limit. Full run: one chunk is 60 circuits + 8 SPAM = 68, so each job holds 1 chunk. Mini run: one chunk is 12 circuits + 8 SPAM = 20, so the 8 chunks go out as 3 jobs (3 + 3 + 2 chunks, 60 + 60 + 40 circuits). |

### Benchmark size and run folders (command-line arguments)

Every submit, collect, job-status and simulate script here takes the benchmark size on the command line:
`mini`, `full`, a preset with other shots or a custom size — the same arguments as every script in
`source/benchmark_scripts/` (main USER_MANUAL §4.3, with the table of all forms). A collect or status script needs the
same arguments as the submission it belongs to, since they pick the run folder, whose name is described in
main USER_MANUAL §5.4. Called without arguments, a script prints its usage and does nothing. Example: `python source/benchmark_scripts/real_qpu_scripts/IQM_EQE1/qbcs_real_submit_iqm_eqe1.py backend=EQE1 full`.
The circuits must be cached for the same depth list first: run `source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_lnn.py` with the same arguments in `local_sim.venv` (main USER_MANUAL §4.3).

**Qubit layout.** IQM EQE1 (MQSS) reports **no calibration data**, so a live run cannot be optimized without a calibration
file. Therefore:

- **Without `layout=` and `calibration_file=`, the run uses a random layout** (a notice is printed): any valid
  3-qubit layout, drawn once with a fresh seed that is recorded in `physical_qubits.json`. The valid layouts follow the qubits and couplers of the live Target; the calibration file may list more.
- `layout=optimized calibration_file=PATH` scores the placements with a calibration file. `calibration_file=` is
  mandatory with `optimized` here (an error without it). The file may be the provider's own,
  the operators' `calibration_data_Euro-Q-Exa_<date>_active_qubits.json`, or an already converted snapshot; a provider file is converted automatically and saved in the run folder
  as `calibration_snapshot.json` (its `measured_at` is the file's own date). The latest provider file is in `calibration_input_files/Euro-Q-Exa/`. A relative PATH is relative to the project root.
- `layout=lowest` is the lexicographically lowest valid layout, and `layout=15,14,21` forces exactly those qubits.
  A layout the QPU cannot run ends the script with a one-line message (exit code 2) before anything is submitted;
  so does `layout=optimized` when the QPU has no placement of the circuits without SWAPs, like `random` and `lowest`.

The run folder ends in `_layout-optimized`, `_layout-random`, `_layout-lowest` or `_layout-15-14-21`; the collect and status scripts take `layout=` too (optional). After the layout comes the UTC date the run started and its number that day (`_2026-10-11_#1`). Rerunning the submit script continues the latest run of its settings while it is unfinished (some chunk not yet submitted), whatever its date, else starts a new folder dated today with the next run number; `--force-new-run` starts a new folder even while the latest is unfinished. Without `layout=` and `date=`, the collect and status scripts take every run of that benchmark with something left to collect in turn, all layouts and dates (each is reported, and a failure in one does not stop the others); `layout=` picks one run (its latest, or the one of `date=`), also a finished one; `date=YYYY-MM-DD` alone takes each layout's latest run of that day, and `date=YYYY-MM-DD_#N` each layout's run with that number (layouts without one are skipped).
The **simulate script requires `layout=`** (it stops with an explanation otherwise): the live backend provides no calibration,
a real submission without a file is random, so the simulation must be told which of these to mimic. It takes its noise
data from `calibration_file=` (default: the newest file in `calibration_input_files/Euro-Q-Exa/`), under any layout.

## 4. Workflow

```
qbcs_real_submit_iqm_eqe1.py                qbcs_real_collect_iqm_eqe1.py
  build circuits    ─┐                   (run separately, any time later,
  choose the layout  │  (async,          even in a fresh session)
  PRE health check   │   returns         collect the finished chunks
  submit chunks      │   immediately)    (no waiting), then merge
  POST health check ─┘                   → calls analyze_results() itself
```

- **`qbcs_real_submit_iqm_eqe1.py`** — build → physical qubits → PRE health check
  (aborts if degraded) → submit every chunk asynchronously → POST health check.
  An aborted run stays on record — `run_info.json` is written before the PRE check
  (`qbcs.resolve_physical_qubits(run_folder=…)`) — so `source/analysis_scripts/run_post_process.py` reports it as
  a degraded QPU with the benchmark not run, and plots its health checks only. Both health
  checks are written into the run's own `health_checks/health_check_<N>.json`
  (auto-numbered, single counter).
  **A network blip during the PRE/POST health check is retried** — see the main
  USER_MANUAL §7 for the retry behaviour and the three exit codes (`10` degraded, `11` submitted but not
  collected, `12` not even submitted) `require_functional_health_check`/`submit_post_health_check`
  (`source/core/health_check.py`) use.
- **`qbcs_real_collect_iqm_eqe1.py`** — run separately/later, reconstructs `CONFIG`
  from job records on disk, collects the jobs that have finished (in any order,
  without waiting for the others — run it again later for those; failed jobs are
  marked failed; rerunning the submit script resubmits them and first copies their records to
  `job_records/failed_attempts/`, counted under `failed_attempts` in the summary — main USER_MANUAL §7; EQE1
  reports no times for a failed job. A submission that could not reach MQSS is retried, 3 more
  tries 30 / 60 / 120 s apart; one that still fails, or fails any other way, leaves `failed` records
  marked `"submission failed"` and stops the script — with `may_exist_on_provider: true` check MQSS
  for the job before resubmitting), merges, and calls `analyze_results()` itself. It collects every
  finished, uncollected health check of the run (warn-only if none exist).
- **`check_real_qpu_job_status_iqm_eqe1.py`** — checks the live MQSS status of every job in a run folder
  against the job records on disk, without waiting or changing anything. Same arguments as the
  submit/collect scripts (`backend=EQE1`, mini/full, …); add the word `health_check` to check the
  standalone health-check job instead. Prints the jobs that are not done, a summary (done / pending /
  error / cancelled / unavailable) and EQE1's pending-job count. MQSS has no job listing, so "recent
  jobs" is not available. Same report as the IBM and other IQM scripts (`source/core/qpu/job_status_check.py`).
- **`health_check_real_iqm_eqe1.py`** — a standalone submit-and-collect health check in one script,
  unrelated to any benchmark run (its own run folder, wiped before every submission); it blocks until
  the job finishes, like the other QPUs' health-check scripts. `backend=NAME` is required. The script's
  `QUBIT_SET` (three physical qubits, default `None`: the transpiler chooses) is, when set, checked against the backend
  before anything is transpiled or submitted: a repeated qubit, a qubit that does not exist or is unusable, a wrong
  number of qubits or a missing coupler ends the script with one `❌` line and exit code 2. Exit codes: `0` functional, `10` degraded, `2` for a problem you can fix (no argument, a missing or unknown `backend=`, a missing token, or a bad `QUBIT_SET`), `1` for any other error (a provider or SDK error, a failed submission, collection or job); `11` and `12` come only from the accompanying checks of the submit scripts.

### 4.1 Simulating locally with this QPU's calibration

The SDK (MQSS) reports no calibration data, so `qbcs_real_simulate_iqm_eqe1.py`
(in `iqm_eqe1.venv`) takes the calibration from a file: `calibration_file=PATH` — the operators'
`calibration_data_Euro-Q-Exa_<date>_active_qubits.json`, converted automatically, or an already converted
snapshot — and by default the newest file in `calibration_input_files/Euro-Q-Exa/`. It prints a warning saying that the data is not from the
backend, gives its measurement date, and lists the qubits it leaves out. `layout=` is **required** (see **Qubit
layout** above): with `layout=optimized` the same file also scores the placements. Otherwise it
works like `qbcs_real_simulate_iqm_resonance.py`. The calibration is saved as
`datafiles/<benchmark>_sim-qpu-noise_iqm-eqe1_…_layout-<mode>_<YYYY-MM-DD>_#<N>/calibration_snapshot.json`. A rerun reuses that copy (it must be the same calibration), prints that it is reusing the run's saved calibration, and
resumes an unfinished run (a finished one is followed by a new dated folder); remove the run folder to start over. It then builds an
Aer noise model, picks the physical qubits as `layout=` says, runs health
checks and every chunk locally, and analyzes the results. To post-process it, run
`source/analysis_scripts/run_post_process.py` with `--include-simulations-real` (every layout of this QPU's `sim-qpu-noise_…` runs is
already in `SIMULATED_QPUS` of `RUN_THESE`; `--layouts=<mode>` narrows it), or with `nshots_per_circuit=<its shots> --include-simulations-real`
(main USER_MANUAL §7). Examples of the simulation:

```bash
python source/benchmark_scripts/real_qpu_scripts/IQM_EQE1/qbcs_real_simulate_iqm_eqe1.py backend=EQE1 mini layout=random
python source/benchmark_scripts/real_qpu_scripts/IQM_EQE1/qbcs_real_simulate_iqm_eqe1.py backend=EQE1 mini layout=optimized calibration_file=calibration_input_files/Euro-Q-Exa/calibration_data_Euro-Q-Exa_20260827_active_qubits.json
```

What the noise model contains, and what it doesn't:

- Gate errors on `r` and `cz`, and readout errors. **No T1/T2 relaxation:** the source
  file has no gate durations. The T2 stored is the Ramsey value.
- The **topology of the calibration file**, not the live backend's. Dead couplers and
  qubits change over time, so qubit numbering and couplings may differ from today's
  device.
- Qubits with NaN in T1, T2, `1q_fidelity` or `readout_fidelity` are left out
  entirely (`excluded_qubits` in the file), together with their couplings.

**Newer calibration data:** pass the new provider file as `calibration_file=` (or copy it into `calibration_input_files/Euro-Q-Exa/`,
where the newest file by name is the default).

## 5. Timing, billing and circuit check

Every chunk and health-check record gets a `timing` block and a `circuit_check`:

- **QPU time** (`qpu_s`): MQSS reports neither QPU nor billed time, so it is modelled as **0.9 s per circuit**. A failed job that reached MQSS (it has a `job_id`) counts in full in the summary's QPU time and cost (all its chunks' circuits × `nshots`, once if several chunks share it); a submission failure without a `job_id`, and a job never collected, are not counted and are named in the cost text (main USER_MANUAL §7).
- **Queue time** (`queue_s`): from submission to the server's `scheduled_at` (reported in Europe/Berlin time,
  converted to UTC).
- **Circuit check**: `not_available` — MQSS does not return the executed circuits. The analysis checks instead
  that every transpiled circuit of the run (chunks, SPAM batch, health checks) sits on the same physical qubits
  (`qubit_consistency`).

The full comparison of all QPUs is the timing table in main USER_MANUAL §7; the code is in `source/core/records/job_timing.py` (timing) and `source/core/records/circuit_check.py` (circuit check).

## 6. Outputs & post-processing

**Circuit drawings.** The submit, simulate and health-check scripts save the first circuit of every
error-mitigation role (bare, NEC, RC, RC+NEC, cRC, cRC+NEC), the first SPAM circuit and the health-check
circuit, before and after transpilation, as PDFs in `datafiles/<run_folder>/circuit_drawings/`
(`01_bare_before_transpilation.pdf`, `01_bare_after_transpilation.pdf`, … numbered per role; `00_SPAM_…` and
`health_check_…` for SPAM and the health check; main USER_MANUAL §4).

Standard run-folder layout (main USER_MANUAL §8): `datafiles/<run_folder>/`. Health
checks live *inside* the run folder, in `health_checks/`:

```
datafiles/<run_folder>/
├── health_checks/
│   ├── health_check_1.json      # PRE  — before the submission (single counter)
│   └── health_check_2.json      # POST — after the submission; lists the chunks of the submission
└── job_records/
    └── job_record_chunk_{nc}.json
```

`source/analysis_scripts/run_post_process.py` plots them (with the QPU jobs, labelled by chunk numbers) into
`analysis/plots/<run_folder>/health_checks.png`; see main USER_MANUAL §7 and §10. Health checks are collected
by the collect script; the submit script skips checks and submission if all chunks are already submitted.

`submit.log` (appended, never overwritten) holds one entry per submission, the same in every QPU folder
(`append_submit_log()`): the time, backend, run folder, chunks and shots, the layout, the run qubits with the
calibration summary of `calibration_snapshot.json`, and the run's health-check numbers, plus EQE1's pending-job count at submission.

**Post-processing.** `python source/analysis_scripts/run_post_process.py mini --include-real-data` (or `full`) merges, analyzes, summarizes and plots
this QPU's runs together with the other QPUs' — see main USER_MANUAL §10 for the script, its `RUN_THESE` / `REAL_QPUS` lists,
the analysis of a still-collecting run, and the outputs.

`run_post_process.py` runs the other scripts in `source/analysis_scripts/`, passing each run to them
through the `QPU_*` environment variables; each also works standalone (main USER_MANUAL §10).

## 7. Known issues

None currently open.

## Creating the environment

From the project root (with Python 3.12: `python3.12`, or `python3` if that is 3.12; `py -3.12` on Windows):
```bash
python3.12 -m venv ./venvs/iqm_eqe1.venv
source ./venvs/iqm_eqe1.venv/bin/activate      # Windows PowerShell: .\venvs\iqm_eqe1.venv\Scripts\Activate.ps1
pip install -r source/benchmark_scripts/real_qpu_scripts/IQM_EQE1/requirements_iqm_eqe1.txt
```
