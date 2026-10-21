# IBM Quantum — real-QPU submission

The IBM QPUs are reached through IBM's quantum cloud, [IBM Quantum Platform](https://quantum.cloud.ibm.com/),
where the API token (`ibm_token`, §2) comes from.

## 1. Prerequisites

Separate virtual environment: **ibm.venv** (created at `venvs/ibm.venv` in the project root). On top of the base
`local_sim.venv` package family, it needs one independent top-level package:

| Package | Version (this project's `ibm.venv`) | Provides |
|---|---|---|
| `qiskit-ibm-runtime` | 0.48.0 | real-hardware auth/submission (`QiskitRuntimeService`, `SamplerV2`, `Batch`) |

`qiskit-aer==0.17.2` is also in this venv (same version as base) — used by
the main USER_MANUAL's §4.2 NoiseModel example, *not* by hardware submission
itself. Unlike IQM VLQ and AQT, `qiskit-ibm-runtime` does not force a Qiskit
downgrade — `ibm.venv` runs `qiskit==2.5.2`, same as base.

**No myQLM in this venv.** The circuits are built in `local_sim.venv` — run
`build()` there first, with the same settings (the matching
`source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_lnn.py` script, with the same benchmark arguments, does it, see the main USER_MANUAL §4.3). It
caches every circuit as Qiskit QPY, plus the exact reference values the
analysis needs, and the scripts here only read those files. A script that
finds the cache incomplete stops with a message saying so.
Keep myQLM out of this venv as a design rule.

**Build it in one pass**: fresh venv, then a single pinned
`pip install -r requirements_ibm.txt` covering the base packages and
`qiskit-ibm-runtime` together.

## 2. Connecting

```python
from source.core.qpu.tokens import read_token
from source.core.qpu.providers import connect_ibm

token   = read_token("ibm_token")   # tokens/ibm_token
service, instance = connect_ibm(token, None, "ibm_marrakesh")   # None: the Open Plan instance; or an instance name
real_backend = service.backend("ibm_marrakesh")   # or another name listed in `backends.json`
```

Every script in this folder that connects to IBM authenticates this way, reading `tokens/ibm_token`, and always
chooses the IBM **instance** explicitly (an instance is a plan-bound resource of your IBM Quantum Platform account,
listed with its name and plan on the platform): the account's free Open Plan instance by default, another one with
`instance=NAME` (its name as the platform shows it). Left to itself, the library may pick a paid
instance, so the scripts never do. The scripts print the instance they use and every instance of the account (the names `instance=NAME` accepts), and `qbcs_real_submit_ibm.py` records it
(name and plan) in the run's `run_info.json` and `submit.log`.

Instances differ in two ways that matter here:

- **The QPUs they list.** A QPU not listed on the chosen instance stops the script with a one-line message (exit
  code 2) saying which ones are. `backends.json` holds every QPU any instance of the account may list; the Open Plan
  instance lists `ibm_marrakesh`, `ibm_fez` and `ibm_kingston`, so the others, which are listed only on paid instances, need `instance=NAME` (a run on a paid instance is billed, §5). Calibration
  downloads (`qbcs_real_simulate_ibm.py`) work through any instance that lists the QPU.
- **The jobs they hold.** A job is found only through the instance that received it. A run therefore stays on the
  instance recorded in its `run_info.json`: collect, status and a continued submission use it, and an `instance=NAME`
  naming another instance for that run stops the script (exit code 2). Runs on different instances in one call (a
  collect without `layout=` and `date=`) stop it too: narrow them with `layout=` or `date=`. The standalone health
  check keeps no record, so its status script takes the same `instance=` as `health_check_real_ibm.py` (default: the Open Plan instance).

## 3. Config knobs (top of each script)

| Knob | Meaning |
|---|---|
| `BACKEND_NAME` | `backend=NAME` on the command line — **required** by `qbcs_real_submit_ibm.py`, `qbcs_real_collect_ibm.py`, `qbcs_real_simulate_ibm.py`, `check_real_qpu_job_status_ibm.py` and `cancel_real_qpu_jobs_ibm.py` (they stop with a usage line without it), e.g. `backend=ibm_marrakesh`. The name must be listed in [`backends.json`](backends.json) (and be visible under your account/instance); add a line there to use another device. Pass the same `backend=` to submit/collect/status for a given run — the run folder name is derived from it, which is how they find the run on disk. `health_check_real_ibm.py` requires it too — there is no default. |
| `INSTANCE` | `instance=NAME` on the command line, optional, accepted by every script here: the IBM instance to use (§2). Default: the account's Open Plan instance; for a run that already has one recorded in its `run_info.json`, that one. |
| run size (shots, RC samples, evolution times) | the preset or sizes given on the command line (below), not edited in the script |
| `LNN` | `True` = linear-nearest-neighbour transpilation, matching the device's local coupling. Taken from the chosen backend's `"lnn"` entry in `backends.json`. IBM's own backends are all heavy-hex (locally connected), so `true` is right for every one listed; a device with different connectivity needs its own `"lnn"` value there (see `readme_iqm_resonance.md` §3 for why it matters). |
| `DO_SPAM` | also run SPAM calibration |
| `CHUNKS_TO_SUBMIT` | submit only the first N chunks (useful for a test run before committing the whole run) |
| `CHUNKS_PER_JOB` | packs N chunks into one IBM `Batch` job — reduces per-job overhead. `min(IBM_CHUNKS_PER_JOB, SAMPLE_SIZE)` (`IBM_CHUNKS_PER_JOB` = 48 in `source/core/qbcs.py`): the full run goes out as 10 jobs of 48 chunks, the mini run as one job of 8 |

### Benchmark size and run folders (command-line arguments)

Every submit, collect, job-status and simulate script here takes the benchmark size on the command line:
`mini`, `full`, a preset with other shots or a custom size — the same arguments as every script in
`source/benchmark_scripts/` (main USER_MANUAL §4.3, with the table of all forms). A collect or status script needs the
same arguments as the submission it belongs to, since they pick the run folder, whose name is described in
main USER_MANUAL §5.4. Called without arguments, a script prints its usage and does nothing. Example: `python source/benchmark_scripts/real_qpu_scripts/IBM/qbcs_real_submit_ibm.py backend=ibm_marrakesh full`.
The circuits must be cached for the same depth list first: run `source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_lnn.py` with the same arguments in `local_sim.venv` (main USER_MANUAL §4.3).

**Qubit layout.** `layout=optimized` (the default), `random`, `lowest` or a list such as `layout=15,14,21`
chooses the physical qubits (main USER_MANUAL §5.4): the placement with the best estimated success probability
from the live calibration (gate and readout errors, plus the T1/T2 decay of qubits idling while others work), a random valid one (the seed is recorded in `physical_qubits.json`), the
lexicographically lowest valid one, or exactly the given qubits — the script stops with a one-line message (exit code 2) if this QPU cannot run them. The chosen qubits are recorded in `transpiled_circuits/physical_qubits.json` and, as `physical_qubits`, in `run_info.json`.
The submit script always scores with the live calibration and refuses `calibration_file=` (exit code 2, with an
explanation). The run folder
ends in `_layout-optimized`, `_layout-random`, `_layout-lowest` or `_layout-15-14-21`, so runs with different
layouts never collide; the collect and status scripts take `layout=` too (optional). After the layout comes the UTC date the run started and its number that day (`_2026-10-11_#1`). Rerunning the submit script continues the latest run of its settings while it is unfinished (some chunk not yet submitted), whatever its date, else starts a new folder dated today with the next run number; `--force-new-run` starts a new folder even while the latest is unfinished. Without `layout=` and `date=`, the collect and status scripts take every run of that benchmark with something left to collect in turn, all layouts and dates (each is reported, and a failure in one does not stop the others); `layout=` picks one run (its latest, or the one of `date=`), also a finished one; `date=YYYY-MM-DD` alone takes each layout's latest run of that day, and `date=YYYY-MM-DD_#N` each layout's run with that number (layouts without one are skipped). The simulate script takes the same arguments plus `calibration_file=PATH`: a snapshot (e.g. an old run's
`calibration_snapshot.json`) as the simulated QPU's noise data instead of the download, to re-simulate that calibration.
A submission also saves the QPU's current calibration in the run folder (`calibration_snapshot.json`), once per run — what the layout was chosen on (the layout is scored on that saved copy; its `measured_at` is when IBM made the calibration). A layout the QPU cannot run ends the script with a one-line message (exit code 2) before anything is submitted; so does `layout=optimized` when the QPU has no placement of the circuits without SWAPs, like `random` and `lowest`.

## 4. Workflow

```
qbcs_real_submit_ibm.py                qbcs_real_collect_ibm.py
  build circuits    ─┐                   (run separately, any time later,
  choose the layout  │  (async,          even in a fresh session)
  PRE health check   │   returns         collect the finished chunks
  submit chunks      │   immediately)    (no waiting), then merge
  POST health check ─┘                   → calls analyze_results() itself
```

- **`qbcs_real_submit_ibm.py`** — builds the circuits (from the cache), picks the
  run's physical qubits, submits a PRE health check and aborts if the QPU is
  degraded, submits every chunk asynchronously (one `SamplerV2` job per chunk,
  or `CHUNKS_PER_JOB` chunks per `Batch` job; a `Batch` is closed even if its submission fails), then submits a POST health check.
  An aborted run stays on record — `run_info.json` is written before the PRE check
  (`qbcs.resolve_physical_qubits(run_folder=…)`) — so `source/analysis_scripts/run_post_process.py` reports it as
  a degraded QPU with the benchmark not run, and plots its health checks only. Prints live calibration (CZ error,
  T1/T2) and appends everything to `submit.log`. Returns immediately — it does
  **not** wait for hardware results.
  **A network blip during the PRE/POST health check is retried** — see the main
  USER_MANUAL §7 for the retry behaviour and the three exit codes (`10` degraded, `11` submitted but not
  collected, `12` not even submitted) `require_functional_health_check`/`submit_post_health_check`
  (`source/core/health_check.py`) use.
- **`qbcs_real_collect_ibm.py`** — run separately, any time after submission,
  even in a brand-new process: it reconstructs the run's `CONFIG` entirely
  from the job records on disk, collects the jobs that have finished (in any
  order, without waiting for the others — run it again later for those; failed
  jobs are marked failed, with the QPU/queue time IBM still reports), merges them, and **calls `qbcs.analyze_results()` itself** — you don't run that
  step manually for a real-QPU run. Rerunning the submit script resubmits the failed chunks; their
  failed records are first copied to `job_records/failed_attempts/job_record_chunk_<nc>_failed_<k>.json`
  and counted under `failed_attempts` in the summary (main USER_MANUAL §7). A submission that could not
  reach IBM is retried (3 more tries, 30 / 60 / 120 s apart); one that still fails, or fails any
  other way, leaves `failed` records marked `"submission failed"` and stops the script. If such a
  record says `may_exist_on_provider: true` (the connection broke after the request may have
  arrived), check `check_real_qpu_job_status_ibm.py` / the IBM dashboard for the job before
  resubmitting (main USER_MANUAL §7).
- **`check_real_qpu_job_status_ibm.py`** — checks the live IBM status of every job in a run folder
  against the job records on disk, without waiting or changing anything. Same arguments as the
  submit/collect scripts (`backend=NAME`, mini/full, …); add the word `health_check` to check the
  backend's standalone health-check job instead. Prints the jobs that are not done, a summary
  (done / pending / error / cancelled / unavailable), the backend's queue and your 10 most recent
  IBM jobs. Useful for checking on a run without doing a full collect. (Same report as the IQM scripts;
  shared code in `source/core/qpu/job_status_check.py`.)
- **`cancel_real_qpu_jobs_ibm.py`** — cancels the jobs of a run folder that are still queued or running on IBM, after reading their live status. Same arguments as the status script (`backend=NAME`, mini/full, …, `layout=`/`date=`; without them every run with something left to collect is handled in turn); add the word `health_check` to cancel the backend's latest standalone health-check job instead. Jobs that already ended are left alone, a job shared by several chunks is cancelled once, and the records on disk are not changed: run the collect script afterwards (same arguments) to update their status. `instance=NAME` as in the other IBM scripts. Exit code `2` when a cancellation fails. (Shared code in `source/core/qpu/job_cancel.py`.)
- **`health_check_real_ibm.py`** — a standalone submit-and-collect health check in one script, unrelated to
  any benchmark run (its own run folder, wiped before every submission); it blocks until the job finishes,
  like the other QPUs' health-check scripts. `backend=NAME` is required. Exit codes: `0` functional, `10` degraded, `2` for a problem you can fix (no argument, a missing or unknown `backend=`, a missing token), `1` for any other error (a provider or SDK error, a failed submission, collection or job); `11` and `12` come only from the accompanying checks of the submit scripts.

All of them are safe to re-run — see the main USER_MANUAL §5.5 (idempotency) and §7.

### 4.1 Simulating locally with this QPU's calibration

`qbcs_real_simulate_ibm.py` runs the same benchmark (same settings block as the submit
script) on the local Aer simulator, with a noise model built from this QPU's
calibration, and costs no QPU time:

1. downloads the calibration once into
   `datafiles/<benchmark>_sim-qpu-noise_<provider>-<backend>_…/calibration_snapshot.json`. A rerun of an unfinished run
   reuses that file (same calibration, no connection needed) and prints a warning that it uses
   pre-saved calibration data (a finished run is followed by a new dated folder, which
   downloads fresh calibration); to start over, remove the run folder
   `datafiles/<benchmark>_sim-qpu-noise_<provider>-<backend>_…/`;
2. builds an offline backend + Aer noise model from it (depolarizing gate
   errors, T1/T2 relaxation over gate durations and, with `scheduling_simulation=true`, while a qubit waits for the others, readout errors —
   `source/core/qpu/qpu_snapshot.py`);
3. picks the physical qubits from the snapshot exactly as a real submission
   would, runs every chunk (SPAM on the same qubits), and analyzes the results.

Run it from the project root in `ibm.venv`, e.g. `python source/benchmark_scripts/real_qpu_scripts/IBM/qbcs_real_simulate_ibm.py backend=ibm_marrakesh mini`; `scheduling_simulation=true|false` is optional, default `true`: `true` transpiles the circuits scheduled, so idle qubits relax through T1/T2 as on the QPU (full benchmark about 4 h); `false` does not schedule them, so idle relaxation is **not** taken into account (only the relaxation during gates), like the noiseless and coherent-noise simulations (about 1 h). The choice is recorded in `run_info.json` and `submit.log`, the health checks follow it, and an unfinished run folder cannot be resumed with the other value (main USER_MANUAL §4.2). Post-process with
`run_post_process.py --include-simulations-real` (`source/analysis_scripts/`; every `sim-qpu-noise_…` run of this QPU, whatever its layout, is
already covered by `SIMULATED_QPUS` in `RUN_THESE`); its
summary JSON names the backend and snapshot date, so the result is never
mistaken for a hardware run. Like the submit script, it runs a health check
before and after the simulation — the same transpiled circuit, on the run's
qubits, under the same noise model, so it shows what the health circuit scores
with this calibration; a "degraded" score is reported but doesn't stop the
run. It keeps the submit script's records too: a job record per chunk in `job_records/` (`service_name: "local"`) and a `submit.log` entry with the snapshot's calibration summary, the run qubits and both health-check verdicts.
The full 480-chunk run simulates
in about 4 h with the default `scheduling_simulation=true`, roughly an hour with `false` (≈ 8 s per chunk measured with IBM Fez's calibration), plus
the usual circuit build.

## 5. Timing, billing and circuit check

Every chunk and health-check record gets a `timing` block and a `circuit_check`:

- **QPU time** (`qpu_s`): the time IBM bills, `usage.qpu_charge_time_seconds` of the job. IBM bills per job: a failed job that reached IBM (it has a `job_id`) counts in full in the summary's QPU time and cost (all its chunks' circuits × `nshots`, once if several chunks share it); a submission failure without a `job_id`, and a job never collected, are not counted and are named in the cost text (main USER_MANUAL §7).
- **Cost** (summary): the job carries no money amount (IBM reports the billed seconds and abstract resource units), so the effective cost is the QPU time at the Pay-As-You-Go list price. A run on the Open Plan instance was free (actual cost 0); a run on any other (pay-as-you-go) instance was billed that price, so the summary's actual cost equals its effective cost. Which one applies comes from the run's recorded instance plan.
- **Queue time** (`queue_s`): from the job's creation to the start of its first execution span.
- **Circuit check**: the circuits IBM ran (`job.inputs["pubs"]`) compared gate by gate with the submitted ISA
  circuits in `transpiled_circuits/`. A mismatch is warned about and recorded, never fatal.

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
`analysis/plots+summaries/<run_folder>/health_checks.png`; see main USER_MANUAL §7 and §10. Health checks are collected
by the collect script; the submit script skips checks and submission if all chunks are already submitted.

`submit.log` (appended, never overwritten) holds one entry per submission, the same in every QPU folder
(`append_submit_log()`): the time, backend, run folder, chunks and shots, the layout, the run qubits with the
calibration summary of `calibration_snapshot.json`, and the run's health-check numbers, plus an `IBM instance: <name> (<plan>)` line.

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
python3.12 -m venv ./venvs/ibm.venv
source ./venvs/ibm.venv/bin/activate      # Windows PowerShell: .\venvs\ibm.venv\Scripts\Activate.ps1
pip install -r source/benchmark_scripts/real_qpu_scripts/IBM/requirements_ibm.txt
```
