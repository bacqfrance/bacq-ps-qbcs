# IQM VLQ (IT4Innovations, via LEXIS/HEAppE) — real-QPU submission

[VLQ](https://www.eurohpc-ju.europa.eu/eurohpc-quantum-computers/our-quantum-computers_en#vlq), one of the EuroHPC quantum computers, is hosted by IT4Innovations
([it4i.cz](https://www.it4i.cz/en)), the Czech national supercomputing center.
VLQ is IQM's **Star 24** machine at IT4Innovations (EuroHPC): 24 qubits `QB1`…`QB24`, each
coupled only to one computational resonator `COMPR1`. Native operations: `prx` (Qiskit `r`),
`cz` (qubit–resonator), `move` (moves a qubit's state into the resonator and back) and
`measure`. A CX between two qubits becomes MOVE, CZ with the resonator, MOVE back, so every
pair of qubits is connected: the benchmark runs the **all_to_all** circuits (`LNN = False`).

## 1. Prerequisites

Separate virtual environment: **iqm_vlq.venv** (created at `venvs/iqm_vlq.venv` in the project root). On top of the base
`local_sim.venv` package family, it needs these independent top-level packages:

| Package | Version (this project's `iqm_vlq.venv`) | Provides |
|---|---|---|
| `qaas` | 0.5.3 | IT4I's QaaS client: LEXIS authentication and submission to VLQ (`source/core/qpu/iqm_vlq.py`) |
| `py4lexis` | 7.1.1 | LEXIS login (MyAccessID) used by `lexis_backend()`; its refresh token is saved to `tokens/iqm_vlq_token` |
| `qiskit-qasm3-import` | 0.6.0 | OpenQASM 3 import, not imported by this project's code; listed so a rebuild gets the same version |
| `dill` | 0.4.1 | object serialisation, not imported by this project's code; listed so a rebuild gets the same version |

`qaas` pins `qiskit<2.2` (the venv runs `qiskit==2.1.2`) and its own `iqm-client`, so it
cannot share a venv with IQM Resonance (`iqm-client` 35, Qiskit 2.5) or AQT (Qiskit 1.4).
**Do not `pip install qiskit-iqm`** — it conflicts with `qaas`.
`requirements_iqm_vlq.txt` is the source of truth; see "Creating the environment" below.

**No myQLM in this venv.** The circuits are built in `local_sim.venv` — run
`source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_all_to_all.py` there first (main
USER_MANUAL §4.3). An all_to_all build caches its circuits at QPY format version 13
(`datafiles/circuits/all_to_all_Nt<depths>/chunks_qiskit_v13_qpy/`), which Qiskit 2.1 reads.
A script that finds the cache incomplete stops with a message saying so.
Qiskit 2.1 would warn on every load that the file was written by a newer Qiskit (2.5.x).
It checks the writer's Qiskit version recorded in the header, not the QPY format version.
Since the files are meant to be read this way, `source/core/qbcs.py` silences that one
warning. A file that really can't be read still fails on load.

**Build it in one pass**: fresh venv, then a single pinned
`pip install -r requirements_iqm_vlq.txt` (with IT4I's package index) covering the base packages and
`qaas` together.

**LEXIS access.** Membership of the LEXIS project that holds the VLQ allocation. The project
and resource names are read from `tokens/iqm_vlq_project_info` (two lines, `PROJECT = …` and `RESOURCE = …`;
its format is in `tokens/README.md`); pass others to `source.core.qpu.iqm_vlq.lexis_backend(project, resource)`
to use them instead.

## 2. Connecting

```python
from source.core.qpu import iqm_vlq as VLQ
real_backend = VLQ.lexis_backend()        # qaas QBackendIQM
```

`lexis_backend()` logs in to LEXIS with the refresh token saved in `<project root>/tokens/iqm_vlq_token`
(the `tokens/` folder is git-ignored). Only when there is no saved token, or LEXIS rejects it, does it print
**"LEXIS login needed"** and open the MyAccessID login page in the browser — that is when your
approval is needed. The saved token then keeps working for some time (it is renewed on every use),
so later runs log in without the browser until LEXIS asks again.

py4lexis logs every login to `datafiles/<run folder>/iqm_vlq_authentication_lexis_logs.log`
(the project root when the script has no run folder, as the status script without one; git-ignored). It creates the file at each login but writes to it only for a browser login, so after
token logins it is empty. It never contains the tokens.

### Connection log

Every call to IT4I / HEAppE (public status, LEXIS login, `get_backend`, job submission, job status,
result and executed-circuit downloads, calibration downloads) is logged with its UTC time and duration —
`START`, then `OK`, `FAIL` (with the error) or `TIMEOUT` — in the terminal and in
`datafiles/<run>/iqm_vlq_connection.log` (the project root when the script has no run folder, as the status script without one). The qaas client sets no time limits, and a HEAppE request can go
unanswered (seen for `JobManagement/CurrentInfoForJob` while the public status page showed VLQ up),
so each call has a limit (`TIMEOUTS` in `source/core/qpu/iqm_vlq.py`, e.g. 300 s for
`get_backend`, which itself submits and polls a short HEAppE `backend_init` job); after it the script stops
with "IT4I / HEAppE not responding", and the log shows where the call was waiting. During a collection, a
job whose status query times out is left pending and collected next time. A job upload that times out may
still complete on IT4I's side: check the job status before submitting again.

## 3. Config knobs (top of each script)

| Knob | Meaning |
|---|---|
| `BACKEND_NAME` | `backend=VLQ` on the command line — **required** by every script here (benchmark scripts, health check): without it, or with any other name, the script stops before it connects, so nothing can be sent to another machine by mistake. The allowed names and each one's topology (`"lnn"`: linear-nearest-neighbour vs. all-to-all circuits) are in [`backends.json`](backends.json), which also sets `LNN`. |
| run size (shots, RC samples, evolution times) | the preset or sizes given on the command line (below), not edited in the script |
| `LNN` | taken from the backend's `"lnn"` entry in `backends.json`: `false` — the star is all-to-all through the resonator |
| `DO_SPAM` | also run SPAM calibration |
| `CHUNKS_TO_SUBMIT` | submit only the first N chunks |
| `CHUNKS_PER_JOB` | whole chunks (each with its 8 SPAM circuits) per VLQ job. Mini: all 8 chunks in one job (8 × 20 = 160 circuits). Full: 12 (12 × 68 = 816 circuits, ~4 min to upload). Larger jobs (e.g. 48 chunks, 3,264 circuits) fail when IT4I forwards them to IQM, and each would keep one upload open for ~16–18 min, where any network drop loses the whole job; with 12 chunks a failure costs one short job, and rerunning the submit script resubmits only the chunks without records |

### Benchmark size and run folders (command-line arguments)

Every submit, collect, job-status and simulate script here takes the benchmark size on the command line:
`mini`, `full`, a preset with other shots or a custom size — the same arguments as every script in
`source/benchmark_scripts/` (main USER_MANUAL §4.3, with the table of all forms). A collect or status script needs the
same arguments as the submission it belongs to, since they pick the run folder, whose name is described in
main USER_MANUAL §5.4. Called without arguments, a script prints its usage and does nothing. Example: `python source/benchmark_scripts/real_qpu_scripts/IQM_VLQ/qbcs_real_submit_iqm_vlq.py backend=VLQ full`.
The circuits must be cached for the same depth list first: run `source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_all_to_all.py` with the same arguments in `local_sim.venv` (main USER_MANUAL §4.3).

**Qubit layout.** `layout=optimized` (the default), `random`, `lowest` or a list such as `layout=15,14,21`
chooses the physical qubits (main USER_MANUAL §5.4). The resonator is always part of the layout, so `random` and
`lowest` choose among the three qubits (of those with complete calibration) that go with it, and a list names the
three qubits (the script stops with a one-line message if one is not usable). `optimized` grades the qubits per role from IT4I's public
calibration (§4.1); the submit script refuses `calibration_file=` (it always uses the live calibration). The simulate
script takes `calibration_file=PATH` pointing at a `vlq_calibration.json` a VLQ run saved, to re-simulate that
calibration. The run folder ends in `_layout-optimized`, `_layout-random`, `_layout-lowest` or `_layout-15-14-21`; the collect and status scripts take `layout=` too (optional), and so does the simulate script. After the layout comes the UTC date the run started and its number that day (`_2026-10-11_#1`). Rerunning the submit script continues the latest run of its settings while it is unfinished (some chunk not yet submitted), whatever its date, else starts a new folder dated today with the next run number; `--force-new-run` starts a new folder even while the latest is unfinished. Without `layout=` and `date=`, the collect and status scripts take every run of that benchmark with something left to collect in turn, all layouts and dates (each is reported, and a failure in one does not stop the others); `layout=` picks one run (its latest, or the one of `date=`), also a finished one; `date=YYYY-MM-DD` alone takes each layout's latest run of that day, and `date=YYYY-MM-DD_#N` each layout's run with that number (layouts without one are skipped).
A submission also saves the QPU's current calibration in the run folder (`calibration_snapshot.json`, `vlq_calibration.json`), once per run — what the layout was chosen on. A layout the QPU cannot run ends the script with a one-line message (exit code 2) before anything is submitted.

## 4. Workflow

```
qbcs_real_submit_iqm_vlq.py            qbcs_real_collect_iqm_vlq.py
  calibration → physical qubits          (run separately, any time later,
  PRE health check (waits for it)         even in a fresh session)
  upload + submit the job(s)             collects the finished jobs
                                         (no waiting), then merges
  POST health check                      → calls analyze_results() itself
```

- **`qbcs_real_submit_iqm_vlq.py`** — prints VLQ's public status, connects, downloads the
  calibration set the backend runs with, picks the physical qubits (§4.1), runs the PRE health
  check (aborts if degraded), uploads and submits the chunks, then submits the POST health check.
  An aborted run stays on record — `run_info.json` is written before the PRE check
  (`qbcs.resolve_physical_qubits(run_folder=…)`) — so `source/analysis_scripts/run_post_process.py` reports it as
  a degraded QPU with the benchmark not run, and plots its health checks only.
  Uploading is the slow part: qaas sends every circuit as its own HTTP request, ~0.3 s each
  (1,000 circuits ≈ 5 min). Parallel uploads do not help — IT4I's server rejects them
  ("Failed to upload quantum job backend!!!"). An upload that could not reach IT4I is retried, 3 more
  tries 30 / 60 / 120 s apart; one that still fails, or fails any other way, leaves `failed` records
  marked `"submission failed"` and stops the script. An upload that timed out (`VLQTimeout`) or lost its
  connection is not retried and is marked `may_exist_on_provider: true`: the job may still have been
  created — check `check_real_qpu_job_status_iqm_vlq.py` / LEXIS before resubmitting.
  **A network blip during the PRE/POST health check is retried** — see the main
  USER_MANUAL §7 for the retry behaviour and the three exit codes (`10` degraded, `11` submitted but not
  collected, `12` not even submitted) `require_functional_health_check`/`submit_post_health_check`
  (`source/core/health_check.py`) use.
- **`qbcs_real_collect_iqm_vlq.py`** — collects the jobs that have finished (HEAppE), in any
  order, without waiting for the others (run it again later for those; failed jobs are marked
  failed, with the timing IQM still reports; rerunning the submit script resubmits them and first
  copies their records to `job_records/failed_attempts/`, counted under `failed_attempts` in the
  summary — main USER_MANUAL §7): saves the counts, checks the executed circuits (§5), merges and calls
  `analyze_results()` — an incomplete run as `+nsamples-<collected>` files listing what is missing. It
  collects every finished, uncollected health check of the run (warn-only if none exist).
- **`check_real_qpu_job_status_iqm_vlq.py`** — checks each job's live HEAppE status against the job
  records on disk, without waiting or changing anything. Same arguments as the submit/collect scripts
  (`backend=VLQ`, mini/full, …); add the word `health_check` to check the standalone health-check job
  instead. Prints the jobs that are not done, a summary (done / pending / error / cancelled /
  unavailable) and IT4I's public QPU status. HEAppE has no job listing, so "recent jobs" is not available.
  Same report as the IBM and other IQM scripts (`source/core/qpu/job_status_check.py`).
- **`health_check_real_iqm_vlq.py`** — a standalone submit-and-collect health check in one script,
  unrelated to any benchmark run (its own run folder, wiped before every submission); it blocks until
  the job finishes, like the other QPUs' health-check scripts. `backend=NAME` is required. Its optional
  `QUBIT_SET` (physical qubit indices, 0 = QB1; default `None`: the transpiler chooses) is validated, when set, before
  anything is transpiled or submitted, as on the other IQM QPUs: a repeated or nonexistent qubit, or a wrong count, gives one ❌ line and
  exit code 2. VLQ's target is all-to-all over its 24 circuit qubits (the resonator is not part of it), so any three distinct qubits 0…23 pass. Exit codes: `0` functional, `10` degraded, `2` for a problem you can fix (no argument, a missing or unknown `backend=`, a missing token, a missing or incomplete `tokens/iqm_vlq_project_info`, or a bad `QUBIT_SET`), `1` for any other error (a provider or SDK error, a failed submission, collection or job); `11` and `12` come only from the accompanying checks of the submit scripts.

### 4.1 Physical qubits

The backend's Qiskit target carries no error data, so the qubits are chosen from IT4I's public
calibration API (`https://extranet.it4i.cz/api/v1/calibration-sets/star24/<id>`, with `/metrics`
and `/dynamic-quantum-architecture`; no authentication), for the calibration set the backend
reports (`select_vlq_qubits()` in `source/core/qpu/iqm_vlq.py`; `star_roles` is the method label written into `physical_qubits.json`):

1. **Roles.** Every two-qubit gate is MOVE, CZ with the resonator, MOVE back, and the transpiler
   decides which of the two qubits is moved (on VLQ the lower virtual index, so for the benchmark's
   three qubits: v0 only moves, v1 moves for v2 and does the CZ for v0, v2 only does CZs). The
   representative circuit (chunk 0's deepest) is transpiled once — onto the live backend in the
   submit script, onto an offline target built from the calibration's architecture in the
   simulation; nothing is submitted — and each virtual qubit's `prx` gates, MOVE pairs and CZs are
   counted. Full benchmark: `prx` 96 / 153 / 210, MOVE pairs 76 / 38 / 0, CZs 0 / 38 / 76.
2. **Grades.** Placing virtual qubit v on physical qubit q scores

       log P(v → q) = log F_readout + n_prx(v)·log F_prx + n_move(v)·log F_move + n_cz(v)·log F_cz

   with the SSRO readout fidelity, the Clifford RB fidelity converted to one native gate
   (1 − (1 − F_Clifford)/1.875), the MOVE-pair fidelity (interleaved RB with two MOVEs) and the
   interleaved-RB CZ fidelity. A qubit missing any of these metrics (e.g. QB12, for which the
   calibration had no MOVE data) is excluded: a calibration gap signals a problem with the qubit.
3. **Choice.** Every ordered choice of three distinct usable qubits is scored (23·22·21 = 10,626) and the best
   sum is recorded in `<run>/transpiled_circuits/physical_qubits.json`, with the roles, each chosen
   qubit's grade and metrics and the calibration set id.

Transpilation (`qiskit.transpile` with that `initial_layout`) inserts the MOVEs; the check that
transpilation kept every two-qubit gate counts CZs and ignores MOVEs.

### 4.2 Simulating locally with this QPU's calibration

`qbcs_real_simulate_iqm_vlq.py` runs the same benchmark on the local Aer simulator. It downloads
the current calibration set (public API, no login) once into
`datafiles/<benchmark>_sim-qpu-noise_iqm-vlq_…/vlq_calibration.json`, builds an effective all-to-all snapshot from it
(`calibration_snapshot.json`: the resonator folded into a CZ with error 1 − F_cz · F_move and duration
2 · MOVE + CZ, the prx error from the per-gate fidelity above, readout and T1/T2 as measured) and picks the qubits exactly as the submit script does (transpiling onto an offline target built
from the downloaded architecture, with iqm-client).
A rerun of an unfinished run reuses the files and prints a warning that it uses pre-saved calibration data (a finished run is followed by a new dated folder, which downloads fresh calibration); remove the run folder to start over. It runs in
`iqm_vlq.venv` (iqm-client builds the offline target); it needs no login. Post-process by adding the
`sim-qpu-noise_…` folder name (without the benchmark) to `RUN_THESE` in `source/analysis_scripts/run_post_process.py` and
running it with `--include-simulations-real`; its
summary JSON names the backend and snapshot date, so the result is never
mistaken for a hardware run. Like the submit script, it runs a health check
before and after the simulation — the same transpiled circuit, on the run's
qubits, under the same noise model, so it shows what the health circuit scores
with this calibration; a "degraded" score is reported but doesn't stop the
run. It keeps the submit script's records too: a job record per chunk in `job_records/` (`service_name: "local"`) and a `submit.log` entry with the snapshot's calibration summary, the run qubits and both health-check verdicts.

## 5. Timing, billing and circuit check

Every chunk and health-check record gets a `timing` block and a `circuit_check` (see `source/core/records/job_timing.py`, `TIMING_NOTES["QAAS_IT4I"]`):

- **QPU time** (`qpu_s`): IQM Resonance-equivalent billed time, `ceil(compilation + execution)` of the job, from IQM's server timeline — the benchmark's QPU time, comparable with the other QPUs. A failed job that reached IT4I (it has a `job_id`) counts in full in the summary's QPU time and cost (all its chunks' circuits × `nshots`, priced as tasks like the complete jobs); a submission failure without a `job_id`, and a job never collected, are not counted and are named in the cost text (main USER_MANUAL §7).
- **QPU execution** (`qpu_exec_s`): QPU execution only, qaas' `remote_hw_runtime` (IQM `execution_started` → `execution_ended`); summed into the summary's `QPU_execution_time_s`.
- **Queue time** (`queue_s`): end of our upload → IQM `compilation_started`: the wait in IT4I's HEAppE job system (`it4i_wait_s`, upload end → IQM `created`) plus IQM's queue (`iqm_queue_s`).
- **Consumed resources** (`consumpted_resources`, sic, qaas's field name): the LEXIS project's cumulative consumption counter after the job; in every job we checked it grew by exactly `remote_hw_runtime`.
- **Timelines** (`timeline`, `qaas_events`): IQM's server timeline and qaas' client/remote events, UTC.
- **Circuit check**: qaas returns the circuits the server ran (`job.get_transpiled_circuits()`);
  the collect script compares them gate by gate with the submitted ones (`circuit_check` in each record).

IT4I documents a charge of at least 0.9 s per job; the counter we observed charges the execution
time only (≈ 0.2 s per job + 0.3 ms per shot + 3 ms per circuit). The summary's cost is a
synthetic estimate combining the IQM Resonance and Amazon Braket billing schemes (VLQ is not sold
commercially): $0.30 per task and Braket's Emerald shot price scaled by IQM Resonance's credits per
second for Sirius (a Star 24 machine, 0.3) over Emerald (0.75), $0.00064 per shot; each job counts
as the program sets of up to 100 circuits that its whole chunks fill, each billed as one task.

The full comparison of all QPUs is the timing table in main USER_MANUAL §7; the code is in `source/core/records/job_timing.py` (timing) and `source/core/records/circuit_check.py` (circuit check).

## 6. Outputs & post-processing

**Circuit drawings.** The submit, simulate and health-check scripts save the first circuit of every
error-mitigation role (bare, NEC, RC, RC+NEC, cRC, cRC+NEC), the first SPAM circuit and the health-check
circuit, before and after transpilation, as PDFs in `datafiles/<run_folder>/circuit_drawings/`
(`01_bare_before_transpilation.pdf`, `01_bare_after_transpilation.pdf`, … numbered per role; `00_SPAM_…` and
`health_check_…` for SPAM and the health check; main USER_MANUAL §4).

Standard run-folder layout (main USER_MANUAL §8): `datafiles/<run_folder>/` with `health_checks/`, `job_records/`,
`transpiled_circuits/`, `raw_results/`, `processed_results/`, `submit.log`. Health
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
calibration summary of `calibration_snapshot.json`, and the run's health-check numbers, plus the calibration set id.

**Post-processing.** `python source/analysis_scripts/run_post_process.py mini --include-real-data` (or `full`) merges, analyzes, summarizes and plots
this QPU's runs together with the other QPUs' — see main USER_MANUAL §10 for the script, its `RUN_THESE` / `REAL_QPUS` lists,
the analysis of a still-collecting run, and the outputs.

`run_post_process.py` runs the other scripts in `source/analysis_scripts/`, passing each run to them
through the `QPU_*` environment variables; each also works standalone (main USER_MANUAL §10).

## 7. Known issues

- **Upload time.** A 12-chunk job takes ~4 min to upload, so the full run's 40 jobs take ~2.7 h
  (plus ~1.5 h to transpile the chunks the first time). Rerunning the submit script resubmits the chunks of
  an upload marked `"submission failed"`; an upload marked `may_exist_on_provider: true` (timed out or
  lost its connection) may have created a job, so check it first (section 4).
- **Walltime.** IT4I accepts at most 7,200 s of HEAppE walltime per job (`iqm_vlq.WALLTIME_LIMIT_S`).
- **Job ids** are HEAppE job ids (integers); IQM's own job id is in each record's `timing.iqm_job_id`.
- **Errors at exit.** The HEAppE client (`py4heappe`) raises harmless errors in its destructor when Python exits (`AttributeError` on the already torn-down logger and `pickle`, and on Windows `OSError: [WinError 6] The handle is invalid` from its worker pool). The work is done and saved by then, so the scripts hide exactly these errors (`iqm_vlq._silence_heappe_shutdown_errors()`); any other error still prints.

## Creating the environment

From the project root (with Python 3.12: `python3.12`, or `python3` if that is 3.12; `py -3.12` on Windows):
```bash
python3.12 -m venv ./venvs/iqm_vlq.venv
source ./venvs/iqm_vlq.venv/bin/activate      # Windows PowerShell: .\venvs\iqm_vlq.venv\Scripts\Activate.ps1
pip install -r source/benchmark_scripts/real_qpu_scripts/IQM_VLQ/requirements_iqm_vlq.txt --extra-index-url https://opencode.it4i.eu/api/v4/projects/107/packages/pypi/simple
```
