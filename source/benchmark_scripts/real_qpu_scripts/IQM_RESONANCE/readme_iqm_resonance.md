# IQM Resonance (cloud) — real-QPU submission

IQM's own quantum computers are reached through IQM's cloud service, [IQM Resonance](https://resonance.iqm.tech/),
where the API token (`iqm_resonance_token`, §2) comes from.

## 1. Prerequisites

Separate virtual environment: **iqm_resonance.venv** (created at `venvs/iqm_resonance.venv` in the project root). On top of the base
`local_sim.venv` package family, it needs one independent top-level package:

| Package | Version (this project's `iqm_resonance.venv`) | Provides |
|---|---|---|
| `iqm-client` | 35.0.2 | `iqm.qiskit_iqm.IQMProvider` |

Installing it pulls `iqm-pulse`, `iqm-station-control-client`, and
`iqm-data-definitions` as dependencies (plus an OpenTelemetry stack used for
its own tracing). **This is a different SDK from IQM EQE1's `mqss-qiskit`**,
even though both target IQM hardware — don't assume the two IQM environments
are interchangeable. Like EQE1, this does not force a Qiskit downgrade:
`iqm_resonance.venv` runs `qiskit==2.5.2`. Every other shared package
(`qiskit-aer`, `numpy`, `scipy`, …) is independently resolved in this
venv — treat `requirements_iqm_resonance.txt` as the source of truth.

**No myQLM in this venv.** The circuits are built in `local_sim.venv` — run
`build()` there first, with the same settings (the matching
`source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_lnn.py` script for Emerald and Garnet, `benchmark_noiseless_all_to_all.py` for
Sirius, which is all-to-all — with the same benchmark arguments, see the main USER_MANUAL §4.3). It
caches every circuit as Qiskit QPY, plus the exact reference values the
analysis needs, and the scripts here only read those files. A script that
finds the cache incomplete stops with a message saying so.

**Build it in one pass**: fresh venv, then a single pinned
`pip install -r requirements_iqm_resonance.txt` covering the base packages
and `iqm-client` together, rather than adding `iqm-client` on top of an
already-installed `local_sim.venv`.

## 2. Connecting

```python
from source.core.qpu.tokens import read_token
from iqm.qiskit_iqm import IQMProvider

token    = read_token("iqm_resonance_token")   # tokens/iqm_resonance_token
provider = IQMProvider("https://resonance.iqm.tech/", quantum_computer="emerald", token=token)
real_backend = provider.get_backend(use_metrics=True)   # with the calibration errors, to pick the run's qubits
```

## 3. Config knobs (top of each script)

| Knob | Meaning |
|---|---|
| `IQM_RESONANCE_URL` | `"https://resonance.iqm.tech/"` (from `source/core/qpu/providers.py`) |
| `BACKEND_NAME` | quantum-computer id/alias, given as `backend=NAME` on the command line — **required** by `qbcs_real_submit_iqm_resonance.py`, `qbcs_real_collect_iqm_resonance.py`, `qbcs_real_simulate_iqm_resonance.py` and `check_real_qpu_job_status_iqm_resonance.py` (they stop with a usage line without it), e.g. `backend=emerald` (Crystal-54). The name must be listed in [`backends.json`](backends.json); add a line there to use another machine. Pass the same `backend=` to submit/collect/status for a given run — the run folder name is derived from it, which is how they find the run on disk. `health_check_real_iqm_resonance.py` requires it too — there is no default. Listed: `emerald`, `garnet` (grid, LNN) and `sirius` (all-to-all). |
| run size (shots, RC samples, evolution times) | the preset or sizes given on the command line (below), not edited in the script |
| `LNN` | linear-nearest-neighbour transpilation. Taken from the chosen machine's `"lnn"` entry in `backends.json`. **It depends on the device's topology** — IQM Resonance serves several machines and they don't all share one connectivity: a grid-connected device like `emerald` (Crystal-54) needs `"lnn": true` (matching its local coupling), but a star/resonator-coupled device (all-to-all through the resonator, like `sirius` or IQM VLQ) needs `"lnn": false`. When adding a machine to `backends.json`, check its real topology — the wrong setting doesn't necessarily fail outright, but it changes what the transpiler optimizes for and can produce a worse or misleading benchmark. |
| `DO_SPAM` | also run SPAM calibration |
| `CHUNKS_TO_SUBMIT` | submit only the first N chunks |
| `MAX_CIRCUITS_PER_JOB` | 100 — most circuits in one Resonance job (100 is accepted, 101 is not). Chunks are packed: each job holds as many whole chunks (each with its own SPAM circuits) as fit under this limit. Full run: one chunk is 60 circuits + 8 SPAM = 68, so each job holds 1 chunk. Mini run: one chunk is 12 circuits + 8 SPAM = 20, so the 8 chunks go out as 2 jobs (5 + 3 chunks, 100 + 60 circuits). |

### Benchmark size and run folders (command-line arguments)

Every submit, collect, job-status and simulate script here takes the benchmark size on the command line:
`mini`, `full`, a preset with other shots or a custom size — the same arguments as every script in
`source/benchmark_scripts/` (main USER_MANUAL §4.3, with the table of all forms). A collect or status script needs the
same arguments as the submission it belongs to, since they pick the run folder, whose name is described in
main USER_MANUAL §5.4. Called without arguments, a script prints its usage and does nothing. Example: `python source/benchmark_scripts/real_qpu_scripts/IQM_RESONANCE/qbcs_real_submit_iqm_resonance.py backend=emerald full`.
The circuits must be cached for the same depth list first: run `source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_lnn.py` (`benchmark_noiseless_all_to_all.py` for Sirius) with the same arguments in `local_sim.venv` (main USER_MANUAL §4.3).

**Qubit layout.** `layout=optimized` (the default), `random`, `lowest` or a list such as `layout=15,14,21`
chooses the physical qubits (main USER_MANUAL §5.4): the placement with the best estimated success probability
from the live calibration, a random valid one (the seed is recorded in `physical_qubits.json`), the
lexicographically lowest valid one, or exactly the given qubits — the script stops with a one-line message (exit code 2) if this QPU cannot run them.
The submit script always scores with the live calibration and refuses `calibration_file=` (exit code 2, with an
explanation). The run folder
ends in `_layout-optimized`, `_layout-random`, `_layout-lowest` or `_layout-15-14-21`, so runs with different
layouts never collide; the collect and status scripts take `layout=` too (optional when the benchmark has one run
folder). The simulate script takes the same arguments plus `calibration_file=PATH`: a snapshot (e.g. an old run's
`calibration_snapshot.json`) as the simulated QPU's noise data instead of the download, to re-simulate that calibration.
A submission also saves the QPU's current calibration in the run folder (`calibration_snapshot.json`), once per run — what the layout was chosen on. A layout the QPU cannot run ends the script with a one-line message (exit code 2) before anything is submitted; so does `layout=optimized` when the QPU has no placement of the circuits without SWAPs, like `random` and `lowest`.

## 4. Workflow

```
qbcs_real_submit_iqm_resonance.py      qbcs_real_collect_iqm_resonance.py
  build circuits    ─┐                   (run separately, any time later,
  choose the layout  │  (async,          even in a fresh session)
  PRE health check   │   returns         collect the finished chunks
  submit chunks      │   immediately)    (no waiting), then merge
  POST health check ─┘                   → calls analyze_results() itself
```

- **`qbcs_real_submit_iqm_resonance.py`** — build → physical qubits → PRE health
  check (aborts if degraded) → submit every chunk asynchronously → POST health check.
  An aborted run stays on record — `run_info.json` is written before the PRE check
  (`qbcs.resolve_physical_qubits(run_folder=…)`) — so `source/analysis_scripts/run_post_process.py` reports it as
  a degraded QPU with the benchmark not run, and plots its health checks only.
  Health checks are written into the run's own
  `health_checks/health_check_<N>.json` (auto-numbered).
  **A network blip during the PRE/POST health check is retried** — see the main
  USER_MANUAL §7 for the retry behaviour and the three exit codes (`10` degraded, `11` submitted but not
  collected, `12` not even submitted) `require_functional_health_check`/`submit_post_health_check`
  (`source/core/health_check.py`) use.
- **`qbcs_real_collect_iqm_resonance.py`** — run separately/later; collects the jobs that have
  finished, in any order, without waiting for the others (run it again later for those; failed
  jobs are marked failed, with the billed time and timeline IQM still reports; rerunning the submit
  script resubmits them and first copies their records to `job_records/failed_attempts/`, counted
  under `failed_attempts` in the summary — main USER_MANUAL §7. A submission that could not reach IQM is
  retried, 3 more tries 30 / 60 / 120 s apart; one that still fails, or fails any other way, leaves
  `failed` records marked `"submission failed"` and stops the script — with
  `may_exist_on_provider: true` check the IQM Resonance web interface for the job before
  resubmitting), merges, and calls `analyze_results()` itself — an incomplete run as
  `+nsamples-<collected>` files listing what is missing. It collects every finished, uncollected health
  check of the run (warn-only if none exist).
- **`check_real_qpu_job_status_iqm_resonance.py`** — checks the live Resonance status of every job in a
  run folder against the job records on disk, without waiting or changing anything. Same arguments as
  the submit/collect scripts (`backend=NAME`, mini/full, …); add the word `health_check` to check the
  backend's standalone health-check job instead. Prints the jobs that are not done, a summary (done /
  pending / error / cancelled / unavailable) and the QPU's health. Resonance has no job listing, so
  "recent jobs" is not available. Same report as the IBM and other IQM scripts
  (`source/core/qpu/job_status_check.py`).
- **`health_check_real_iqm_resonance.py`** — a standalone submit-and-collect health check in one script,
  unrelated to any benchmark run (its own run folder, wiped before every submission); it blocks until
  the job finishes, like the other QPUs' health-check scripts. `backend=NAME` is required. The script's
  `QUBIT_SET` (three physical qubits, default `None`: the transpiler chooses) is, when set, checked against the backend
  before anything is transpiled or submitted: a repeated qubit, a qubit that does not exist or is unusable, a wrong
  number of qubits or a missing coupler ends the script with one `❌` line and exit code 2. Exit codes: `0` functional, `10` degraded, `2` for a problem you can fix (no argument, a missing or unknown `backend=`, a missing token, or a bad `QUBIT_SET`), `1` for any other error (a provider or SDK error, a failed submission, collection or job); `11` and `12` come only from the accompanying checks of the submit scripts.

### 4.1 Simulating locally with this QPU's calibration

`qbcs_real_simulate_iqm_resonance.py` runs the same benchmark (same settings block as the submit
script) on the local Aer simulator, with a noise model built from this QPU's
calibration, and costs no QPU time:

1. downloads the calibration once into
   `datafiles/<benchmark>_sim-qpu-noise_<provider>-<backend>_…/calibration_snapshot.json`. A rerun reuses that
   file (same calibration, no connection needed) and prints a warning that it uses
   pre-saved calibration data; to download fresh calibration and run with it,
   remove the run folder `datafiles/<benchmark>_sim-qpu-noise_<provider>-<backend>_…/`;
2. builds an offline backend + Aer noise model from it (depolarizing gate
   errors, T1/T2 relaxation over gate durations, readout errors —
   `source/core/qpu/qpu_snapshot.py`);
3. picks the physical qubits from the snapshot exactly as a real submission
   would, runs every chunk (SPAM on the same qubits), and analyzes the results.

Run it from the project root in `iqm_resonance.venv`. It connects with `get_backend(use_metrics=True)` — without it the backend carries no error data. Post-process by adding the
`…_sim-qpu-noise_…` folder name to `RUN_THESE` in `source/analysis_scripts/run_post_process.py` and
running it with `--include-simulations-real`; its
summary JSON names the backend and snapshot date, so the result is never
mistaken for a hardware run. Like the submit script, it runs a health check
before and after the simulation — the same transpiled circuit, on the run's
qubits, under the same noise model, so it shows what the health circuit scores
with this calibration; a "degraded" score is reported but doesn't stop the
run. It keeps the submit script's records too: a job record per chunk in `job_records/` (`service_name: "local"`) and a `submit.log` entry with the snapshot's calibration summary, the run qubits and both health-check verdicts.
The full 480-chunk run simulates
in roughly an hour (≈ 8 s per chunk; measured with IBM Fez's calibration, Resonance's are similar), plus
the usual circuit build.

## 5. Timing, billing and circuit check

Every chunk and health-check record gets a `timing` block and a `circuit_check`:

- **QPU time** (`qpu_s`): the time IQM Resonance bills, the job's raw `metrics.billed_seconds`. IQM Resonance bills per job: a failed job that reached IQM (it has a `job_id`) counts in full in the summary's QPU time and cost (all its chunks' circuits × `nshots`, once if several chunks share it); a submission failure without a `job_id`, and a job never collected, are not counted and are named in the cost text (main USER_MANUAL §7).
- **Queue time** (`queue_s`): from the job's creation to `compilation_started` on IQM's server timeline.
- **Circuit check**: the job's payload (`GET jobs/{id}/payload`) compared gate by gate with the submitted ISA
  circuits in `transpiled_circuits/`. A mismatch is warned about and recorded, never fatal.
- A failed job is recorded with the billed time and timeline IQM still reports.

The full comparison of all QPUs is the timing table in main USER_MANUAL §7; the code is in `source/core/records/job_timing.py` (timing) and `source/core/records/circuit_check.py` (circuit check).

## 6. Outputs & post-processing

**Circuit drawings.** The submit, simulate and health-check scripts save the first circuit of every
error-mitigation role (bare, NEC, RC, RC+NEC, cRC, cRC+NEC), the first SPAM circuit and the health-check
circuit, before and after transpilation, as PDFs in `datafiles/<run_folder>/circuit_drawings/`
(`01_bare_before_transpilation.pdf`, `01_bare_after_transpilation.pdf`, … numbered per role; `00_SPAM_…` and
`health_check_…` for SPAM and the health check; main USER_MANUAL §4).

Standard run-folder layout (main USER_MANUAL §8): `datafiles/<run_folder>/`.

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
calibration summary of `calibration_snapshot.json`, and the run's health-check numbers.

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
python3.12 -m venv ./venvs/iqm_resonance.venv
source ./venvs/iqm_resonance.venv/bin/activate      # Windows PowerShell: .\venvs\iqm_resonance.venv\Scripts\Activate.ps1
pip install -r source/benchmark_scripts/real_qpu_scripts/IQM_RESONANCE/requirements_iqm_resonance.txt
```
