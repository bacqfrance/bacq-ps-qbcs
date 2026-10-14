# AQT PIAST-Q (via PCSS, EuroHPC) — real-QPU submission

[PIAST-Q](https://www.eurohpc-ju.europa.eu/eurohpc-quantum-computers/our-quantum-computers_en#piast-q) is AQT's trapped-ion quantum computer, one of the EuroHPC quantum
computers. It is hosted by PCSS, the Poznań Supercomputing and Networking Center
([psnc.pl](https://www.psnc.pl/)), and reached over PCSS's API (`pcss_qapi`, §2). PCSS also
gives access to a Slurm cluster from which the benchmark can be run (§4.3).

## 1. Prerequisites

Separate virtual environment: **aqt.venv** (created at `venvs/aqt.venv` in the project root). On top of the base
`local_sim.venv` package family, it needs two independent top-level packages:

| Package | Version (this project's `aqt.venv`) | Provides |
|---|---|---|
| `pcss_qapi` | 0.2.2 | `AuthorizationService`, `pcss_qapi.aqt.provider.PCSS_AQTProvider` (own deps: just `pyjwt`, `requests`) |
| `qiskit-aqt-provider` | 1.14.0 | the actual AQT Qiskit backend integration |

**`qiskit-aqt-provider`'s own package metadata pins `qiskit>=1,<2`** — a hard
upper bound below Qiskit 2.0. That's why `aqt.venv` runs **`qiskit==1.4.6`**,
a full major version behind every other environment in this project
(base `local_sim.venv`, `ibm.venv`, `iqm_eqe1.venv` and `iqm_resonance.venv` are on `qiskit 2.5.x`;
`iqm_vlq.venv` is on `qiskit 2.1.2`, pinned by IT4I's qaas client).
This is the one place in the whole project where code written against a
current Qiskit API could silently break — double check anything
qiskit-version-sensitive before relying on it here.

One concrete consequence: `local_sim.venv`'s Qiskit writes raw circuit-chunk
QPY files (`chunk_{nc}_qiskit.qpy`) at QPY format version 17 by default, but
`aqt.venv`'s Qiskit 1.4.6 can only read up to QPY format version 13. Loading
a `chunk_{nc}_qiskit.qpy` directly in `aqt.venv` raises `QiskitError: "The
QPY format version being read, 17, isn't supported by this Qiskit
version."`. This surfaces the first time a given chunk is transpiled for AQT
(a chunk whose `transpiled_circuits/chunk_{nc}_transpiled.qpy` is already
cached is unaffected), which makes it look like an intermittent crash in a
long submission loop even though it's fully deterministic. See the QPY-format paragraph of §4 for the
build step that avoids this.

Every other package in this venv (`qiskit-aer`, `numpy`, `scipy`, …) is the same family as the
base requirements but independently resolved, so exact patch versions drift —
treat `requirements_aqt.txt` as the source of truth, not the base
`venvs/requirements_local_sim.txt`.

**Tests**: `source/tests/test_12_aqt_shot_split.py` starts the real tests (`source/tests/aqt_venv_test_12_shot_split.py`) in this venv (the rest of the suite runs in `local_sim.venv`); they need `pytest`, which `requirements_aqt.txt` installs, and errors with instructions when this venv or `pytest` is missing.

**No myQLM in this venv.** The circuits are built in `local_sim.venv` — run
`build()` there first, with the same settings (`source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_all_to_all.py`
does it, see the main USER_MANUAL §4.3). It
caches every circuit as Qiskit QPY, plus the exact reference values the
analysis needs, and the scripts here only read those files. A script that
finds the cache incomplete stops with a message saying so.

**Build it in one pass**: fresh venv, then a single pinned
`pip install -r requirements_aqt.txt` covering the base packages,
`pcss_qapi`, and `qiskit-aqt-provider` together. Installing
`qiskit-aqt-provider` on top of an already-installed newer `qiskit` lets pip
silently downgrade it to satisfy the `<2` constraint — resolving everything
in one pass avoids relying on that implicit downgrade.

## 2. Connecting

```python
from source.core.qpu.tokens import read_token
from pcss_qapi import AuthorizationService
from pcss_qapi.aqt.provider import PCSS_AQTProvider

token = read_token("aqt_piast-q_token")   # tokens/aqt_piast-q_token
AuthorizationService.login(token)
real_backend = PCSS_AQTProvider().get_direct_access_backend()
```

## 3. Config knobs (top of each script)

| Knob | Meaning |
|---|---|
| `BACKEND_NAME` | `backend=PIAST-Q` on the command line — **required** by every script here (benchmark scripts, health check): without it, or with any other name, the script stops before it connects, so nothing can be sent to another machine by mistake. The allowed names and each one's topology (`"lnn"`: linear-nearest-neighbour vs. all-to-all circuits) are in [`backends.json`](backends.json), which also sets `LNN`. |
| `NSHOTS`, `SAMPLE_SIZE`, `LIST_TIMES` | shots per circuit (above 200, AQT's limit, each chunk is run as several parts — §4.2), RC samples and evolution times — from the command line (below), not edited in the script |
| `LNN` | linear-nearest-neighbour transpilation — taken from the backend's `"lnn"` entry in `backends.json` (`false` for PIAST-Q) |
| `DO_SPAM` | also run SPAM calibration |
| `CHUNK_START` / `CHUNK_STOP` | the 1-based range of chunks `qbcs_real_submit_aqt_loop.py` submits, one `submit(chunks=[nc])` call per chunk (the 0-based chunk index); `CHUNK_STOP` is clamped to the real number of chunks (`qbcs.n_chunks`) right after `build()`, so setting it to e.g. `SAMPLE_SIZE` is a safe overshoot |

### Benchmark size and run folders (command-line arguments)

The submit and simulate scripts here take the benchmark size on the command line: `mini`, `full`, a preset
with other shots or a custom size — the same arguments as every script in `source/benchmark_scripts/` (main USER_MANUAL
§4.3, with the table of all forms; the run-folder names are in main USER_MANUAL §5.4). AQT runs synchronously, so there is no collect
or job-status script: the submit script saves each chunk's results as it completes. Called without arguments, a script prints its usage and does nothing. Example: `python source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/qbcs_real_submit_aqt_loop.py backend=PIAST-Q full`.
The circuits must be cached for the same depth list first: run `source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_all_to_all.py` with the same arguments in `local_sim.venv` (main USER_MANUAL §4.3).

**Qubit layout.** PIAST-Q reports **no calibration data**, so a live run cannot be optimized without a calibration
file. Therefore:

- **Without `layout=` and `calibration_file=`, the run uses a random layout** (a notice is printed): any valid
  3-qubit layout, drawn once with a fresh seed that is recorded in `physical_qubits.json`. PIAST-Q is all-to-all and has 20 qubits (ions 0…19; the live backend reports the count), so every ordered triple of ions is a valid layout.
- `layout=optimized calibration_file=PATH` scores the placements with a calibration file. `calibration_file=` is
  mandatory with `optimized` here (an error without it). The file may be the provider's own,
  AQT/PCSS's `piastq_characterisation_<date>.json`, or an already converted snapshot; a provider file is converted automatically and saved in the run folder
  as `calibration_snapshot.json` (its `measured_at` is the file's own date). The latest provider file is in `calibration_input_files/PIAST-Q/`. A relative PATH is relative to the project root.
- `layout=lowest` is the lexicographically lowest valid layout, and `layout=15,14,12` forces exactly those qubits
  (a layout the QPU cannot run ends the script with a one-line message, exit code 2).

The run folder ends in `_layout-optimized`, `_layout-random`, `_layout-lowest` or `_layout-15-14-21`. After the layout comes the UTC date the run started and its number that day (`_2026-10-11_#1`). Rerunning the loop script continues the latest run of its settings while it is unfinished (some chunk not yet submitted), whatever its date, else starts a new folder dated today with the next run number; `--force-new-run` starts a new folder even while the latest is unfinished. The loop script also accepts `run_folder=` to pin the folder (the sbatch files use it).
The **simulate script requires `layout=`** (it stops with an explanation otherwise): the live backend provides no calibration,
a real submission without a file is random, so the simulation must be told which of these to mimic. It takes its noise
data from `calibration_file=` (default: the newest file in `calibration_input_files/PIAST-Q/`), under any layout.

## 4. Workflow

AQT is the one synchronous backend: unlike IBM/IQM, AQT submission **blocks until results are back and saves
inline** — there is no separate collect script.

```
qbcs_real_submit_aqt_loop.py           (synchronous — one submit() call per chunk)
  load the cached circuits (once)
  STOP file in the run folder → exit 4, before the PRE check
  choose the layout (once per run)     → recorded in transpiled_circuits/physical_qubits.json
  record the run (run_info.json)       → an aborted run stays analyzable (health checks only)
  PRE health check (once, before the loop; exit 10 degraded, 12 not submitted, 13 task timed out)
  for n in CHUNK_START..CHUNK_STOP:    (1-based; chunk index nc = n - 1)
      STOP file in the run folder → exit 4, before the chunk
      submit(chunks=[nc])   → submit + wait + save this chunk, inline
      POST health check (exit 10 degraded, 12 not submitted, 13 task timed out)
  merge + analyze_results()            → run_data*.json written immediately
```

- **`qbcs_real_submit_aqt_loop.py`** — the only submit script; also the
  production path for a full run. It always submits the benchmark; for a health check on its
  own, use `health_check_real_aqt.py` (below). Its own outer loop calls
  `qbcs.submit(..., chunks=[nc])` once per chunk of
  `CHUNK_START..CHUNK_STOP` (1-based; `nc` is the 0-based chunk index, the loop value minus 1) (`submit_chunk_aqt` is idempotent, so
  already-completed chunks are instant skips — safe to re-run after an
  interruption, and safe to point at a small range for a quick connectivity
  test). Runs a health check once before the loop and once after every
  chunk, and prints a live wallclock ETA and a QPU-time-budget projection to
  stdout (capture with `nohup`/a redirect); a summary is appended to
  `submit.log`. Each health check is one synchronous call (submit + run +
  collect together, `source/core/qpu/aqt.py: _aqt_retry` already retrying a dropped connection —
  next bullet); if it still can't be submitted, the script aborts with exit
  code **12** (health unknown — nothing is written to disk for that attempt,
  unlike the other backends, where a failed attempt stays recorded), or **13** if the cause is a circuit
  PIAST-Q did not execute in time (time limit below; that attempt is recorded as a `failed` health check); if it
  ran and found the QPU degraded, exit code **10**. The standalone `health_check_real_aqt.py` exits
  `0` (functional), `10` (degraded), `12` (the health check could not be submitted) or `13` (its circuit hits the time limit); an unusable argument or a missing token ends it with `2`, and any other error with `1`.
- PIAST-Q runs each
  circuit as its own task, and `source/core/qpu/aqt.py: _aqt_retry` retries a failed API call on
  its own: a failed result query polls the same task again, a failed submit
  sends that circuit again. The chunk then goes on from that circuit instead of
  starting over. After about 19 min without a connection
  (the delays in `source/core/qpu/aqt.py: _AQT_RETRY_DELAYS_S`, which count towards the 30 min below) the error goes up to the loop script, which
  ends with exit code 1.
- **Time limit per circuit.** PCSS reports no queue state and its result query polls without end, so
  a task PIAST-Q never executes (down, in maintenance, reserved) would block the job forever. Each
  circuit (health check or chunk) must be executed within 30 min of its submission
  (`source/core/qpu/aqt.py: AQT_TASK_TIMEOUT_S`; the connection retries above count towards it).
  Otherwise the run stops with a warning (the `AQTTaskTimeout` message: which task timed out, check
  PIAST-Q's availability with PCSS) and exit code **13**, not a traceback; a chunk also
  writes a `failed` record (`failure_stage` `"execution"`, with the circuits finished before the failure), and so does a PRE or POST health check
  (`health_checks/health_check_<N>.json`, `status` `"failed"`): its wait is recorded as queue time, together
  with its transpile time and no QPU time, so the summary counts it under `failed_attempts` and not in the
  QPU time or cost. The standalone `health_check_real_aqt.py` keeps its single record in `job_records/health_check.json`. A rerun resumes. `run_aqt_loop_with_retries.sbatch`
  (§4.3) reruns automatically.
- The loop script can also pause itself between chunks: when the environment
  variable `AQT_PAUSE_AFTER_HOURS` is set and that many hours have passed since
  it started, it exits with code 3 before the next chunk. The PCSS sbatch file
  uses this (§4.3). Without the variable it runs to the end.
- A `STOP` file in the run folder (`datafiles/<run folder>/STOP`, made with `touch`) ends the loop script with exit code **4** at
  the next chunk boundary: the running chunk and its POST health check finish first, and the script deletes the file when it
  honours it (also when it finds it before the PRE health check, or when the loop has already ended). Details and the sbatch
  behaviour: §4.3.
- **`PCSS_Quantum_UI/aqt_run_folder.py`** — computes the run folder name from the same
  arguments as the loop script (prints it when run as a script). Used by both PCSS loop
  sbatch files.
- **`health_check_real_aqt.py`** — a standalone submit-and-collect health check in one script,
  unrelated to any benchmark run (its own run folder, wiped before every submission); it blocks until the job finishes, like the other QPUs' health-check
  scripts. `backend=NAME` is required. Its own folder is
  `datafiles/health-check+reps-<N>+nshots-<shots>_real-data_aqt-<backend>_lnn-none_service-pcss_spam-none_layout-none_<YYYY-MM-DD>_#<N>/`. Use it to see
  whether PIAST-Q is worth a run before starting one. Not a benchmark run, so it
  writes no `run_info.json` and `source/analysis_scripts/run_post_process.py` doesn't analyze it.

**all_to_all circuits are saved in an older QPY format.** The circuits are built beforehand in **`local_sim.venv`** (not `aqt.venv`, which
has no myQLM) with `qbcs.build(circuit_transpilation_LNN=False)` — the
`benchmark_noiseless_all_to_all.py` local-simulation script does it. An
all_to_all build always saves each chunk as `chunk_{nc}_qiskit_v13.qpy` in
`datafiles/circuits/all_to_all_Nt<depths>/chunks_qiskit_v13_qpy/` (`Nt1-3` for mini, `Nt1-3-5-7-9-11-13-15-17-19` for full), at QPY format
version 13 — the newest version `aqt.venv`'s Qiskit 1.4.6 can read (see §1).
(LNN builds keep Qiskit's default QPY version in `chunks_qiskit/`; AQT runs
all_to_all, so it never needs those.) The same build saves the SPAM calibration
circuits as `spam_qiskit_v13.qpy` in that folder, so `aqt.venv` loads them rather
than building them in myQLM at submit time. The build is idempotent: existing
chunk files are reused.

Qiskit 1.4.6 would warn on every load that the file was written by a newer Qiskit (2.5.x).
It checks the writer's Qiskit version recorded in the header, not the QPY format version.
Since the files are meant to be read this way, `source/core/qbcs.py` silences that one
warning. A file that really can't be read still fails on load.

### 4.1 Simulating locally with this QPU's calibration

The SDK reports no calibration data, so `qbcs_real_simulate_aqt.py` (in `aqt.venv`) takes the calibration from a
file: `calibration_file=PATH` — AQT/PCSS's `piastq_characterisation_<date>.json`, converted automatically, or an
already converted snapshot — and by default the newest file in `calibration_input_files/PIAST-Q/`. It prints a warning saying that the data is not from the
backend and gives its measurement date. `layout=` is **required** (see **Qubit layout** above): with
`layout=optimized` the same file also scores the ions. Otherwise it works like
`qbcs_real_simulate_iqm_resonance.py`. The calibration is saved as
`datafiles/<benchmark>_sim-qpu-noise_aqt-piast-q_…_layout-<mode>_<YYYY-MM-DD>_#<N>/calibration_snapshot.json`. A rerun reuses that copy
(it must be the same calibration), prints that it is reusing the run's saved calibration, and resumes an unfinished run (a finished one is followed by a new dated folder); remove the run folder to start over. It then builds
an Aer noise model, runs health checks and every chunk locally, and analyzes the
results. To post-process it, run
`source/analysis_scripts/run_post_process.py` with `--include-simulations-real` (every layout of this QPU's `sim-qpu-noise_…` runs is
already in `SIMULATED_QPUS` of `RUN_THESE`; `--layouts=<mode>` narrows it), or with `nshots_per_circuit=<its shots> --include-simulations-real`
(main USER_MANUAL §7). Examples of the simulation:

```bash
python source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/qbcs_real_simulate_aqt.py backend=PIAST-Q mini layout=random
python source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/qbcs_real_simulate_aqt.py backend=PIAST-Q mini layout=optimized calibration_file=calibration_input_files/PIAST-Q/piastq_characterisation_2026-09-29.json
```

What the noise model contains:

- Per-ion `r` errors, and `rxx` errors on every pair. The file gives only the
  device-mean two-qubit fidelity, so that value is used for all pairs.
- Readout error from the SPAM fidelity lower bound, and device-wide T1/T2.
- Gate durations and readout time from the file, so T1/T2 relaxation is included.
- Crosstalk figures are kept in the file but not modelled.

A newer provider file is used simply by passing it as `calibration_file=` (or copying it into `calibration_input_files/PIAST-Q/`,
where the newest file by name is the default).

### 4.2 At most 200 shots per circuit — larger shot counts are split

AQT accepts at most **200 shots per circuit** (`job_timing.AQT_MAX_SHOTS`); every
AQT `backend.run` goes through `source/core/qpu/aqt.py: _aqt_run`, which raises `ValueError` above
that. When `NSHOTS` is larger, `submit_chunk_aqt` runs each chunk (SPAM +
main circuits) several times, split as evenly as possible into the fewest
parts (`job_timing.aqt_shot_parts`): 600 → `[200, 200, 200]`, 450 →
`[150, 150, 150]`, 401 → `[134, 134, 133]`. The circuit cache is not
duplicated — the same circuits are simply run again.

- Each finished part is checkpointed to
  `<run>/raw_results/shot_parts/chunk_{nc}_part_{p}.json` (its raw counts,
  shots, batch wall times and timestamps). An interrupted chunk resumes at the
  first missing part; parts on disk are reused, even from an earlier session.
- Each part run in the session is also bracketed in the run's
  `timestamps_utc.csv` / `timestamps_local_time.csv` by
  `qpu_chunk{nc}_part{p}_start` / `qpu_chunk{nc}_part{p}_finished`, inside the
  chunk's own `qpu_chunk{nc}_start` / `_finished` rows. Parts reused from disk
  log nothing. These part labels are informational: the analysis does not read them.
- When all parts are in, their counts are summed per circuit (and checked to
  total `NSHOTS`) and the usual `results_counts.json`, `results_fidelity.json`
  and `spam_calibration_counts_per_chunk.json` are written — the analysis sees
  `NSHOTS` shots per circuit, and the run folder name does not depend on the split.
- `job_record_chunk_{nc}.json` is written only once the chunk is complete. It
  keeps `nshots` = the total per circuit and adds `shot_parts`,
  `max_shots_per_circuit`, `n_executed_circuits` (circuits × parts) and
  `shot_part_times` (per part: shots, submitted/finished, wall time, whether it
  was reused from disk). The QPU time is modelled per batch (§5).

### 4.3 Running on the PCSS cluster (Slurm)

A full run takes weeks of wall time, mostly PCSS queue wait, so it is best left to
the PCSS cluster rather than a PC. The folder `PCSS_Quantum_UI/` holds the files for that.
They are for the PCSS cluster (login node `quantum-ui`, partition `all`, worker nodes
`quantum-cpu-[01-04]` and `quantum-gpu-01`). Jobs there are submitted with Slurm
([Slurm quick start](https://slurm.schedmd.com/quickstart.html)).

**Access and setup.** The cluster is reached through PCSS's OpenVPN and then SSH; PCSS
provides the detailed instructions. To set up the environment:

1. Connect to PCSS via OpenVPN.
2. Connect to the cluster via SSH, and clone this repository there.
3. Open an interactive session on a worker node: `srun --pty /bin/bash`.
4. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
5. Create each venv with Python 3.12, from the project root, e.g.
   `uv venv --python 3.12 venvs/aqt.venv` (and `venvs/local_sim.venv`).
6. Install its packages inside it, e.g.
   `uv pip install --python venvs/aqt.venv -r source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/requirements_aqt.txt`
   (and `venvs/requirements_local_sim.txt` into `local_sim.venv`).

Then leave the interactive session (`exit`) and submit the jobs from the login node:

| File | What it does |
|---|---|
| `run_aqt_loop.sbatch` | the benchmark: runs `qbcs_real_submit_aqt_loop.py` with the given benchmark arguments and requeues itself (below); Slurm limit 24 h per job |
| `run_aqt_loop_with_retries.sbatch` | the same as `run_aqt_loop.sbatch`, but reruns the loop script every hour after a failure, for up to 7 days (below) |
| `aqt_run_folder.py` | prints the run folder for given benchmark arguments; used by both loop sbatch files |
| `run_health_check_aqt.sbatch` | `health_check_real_aqt.py` — one health check on its own; 2 h limit |
| `run_simulate_aqt.sbatch` | `qbcs_real_simulate_aqt.py` with the given benchmark arguments — the benchmark on Aer with PIAST-Q's calibration; 24 h limit |
| `run_noiseless_all_to_all.sbatch` | builds the circuit cache of the benchmark given as argument (`benchmark_noiseless_all_to_all.py <arguments>`, e.g. `mini` or `full`, `local_sim.venv`); at least one argument is required |
| `run_pytest.sbatch` | the test suite (`source/tests/`, `local_sim.venv`; `test_12` starts itself in `aqt.venv`, so both venvs must exist) |

Submit them from the project root (each job changes into the clone itself, `$HOME/Quantum_Benchmarking_project`
as set in the `PROJECT` line of the sbatch files — edit it if your clone is elsewhere); each writes its Slurm
log `<job-name>_<jobid>.out` into the folder you submit from, so into the project root. For example:

```bash
sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_noiseless_all_to_all.sbatch full           # once, before the first full run (`mini` for the mini one)
sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_aqt_loop.sbatch backend=PIAST-Q mini       # the mini benchmark (`full`: the full one); backend= and the benchmark are required
# add layout=... and calibration_file=... (below): without them the run uses a RANDOM layout
# add --force-new-run to start a new run folder even if the latest run of these settings is unfinished
squeue -u $USER                                       # its state; scancel <jobid> stops it
```

**How `run_aqt_loop.sbatch` works.** It runs the loop script directly, with no retry
script: short connection drops are retried inside the chunk (§4), and any other error ends the
job with exit code 1. A circuit PIAST-Q does not execute within 30 min ends it with `13` (§4),
and a health check uses its own codes: `10` if it
ran and found the QPU degraded, `12` if it couldn't be submitted, `13` if its task timed out — in each case, to be looked
at before resubmitting. The sbatch file only special-cases exit code 3 (pause/requeue, below) and
passes every other code straight through.

**`run_aqt_loop_with_retries.sbatch`** is the same job (same arguments, same pause and requeue after
10 h, same logs), except that a loop script that ends in any other way than `0` (done), `3` (pause),
`2` (unusable arguments, which a rerun cannot fix) or `4` (STOP file, below) is run again, 1 h after the failure and then every hour; a rerun
resumes at the first chunk without a record. The job sleeps between the attempts (a worker node cannot
`sbatch`), and the waiting counts towards its 10 h, after which it requeues itself as usual. After 7 days of failures
without progress (a chunk completed, or an end with `0` or `3`, starts the week again; the start of the series is kept in
`retry_state` in the run folder) the job prints a `GIVING UP` block into its Slurm log and `output.log` and ends
with exit code `14`. The cluster cannot send e-mail, so that block and `sacct -j <jobid>` are the notification.
`PAUSE_AFTER_HOURS`, `RETRY_WAIT_S` and `RETRY_LIMIT_DAYS` are at the top of the file.

**One run folder per job (both loop sbatch files).** The sbatch file resolves the run folder once, at the start of the job (`aqt_run_folder.py`, which applies `--force-new-run`). Every retry then calls the loop script without `--force-new-run` and with `run_folder=<that folder>`, so the job keeps one folder through all retries; after a pause the requeued job drops `--force-new-run` and continues the folder it started.

**Pause and requeue (both loop sbatch files).** The time limit is `PAUSE_AFTER_HOURS` (10), set inside the sbatch file:

1. The sbatch file passes it to the loop script as `AQT_PAUSE_AFTER_HOURS`.
2. Once that time has passed, the loop script finishes the chunk it is on, with its POST
   health check, and then exits with code 3 instead of starting the next chunk. Nothing
   is cut off, however long the chunk takes; the 24 h Slurm limit leaves room for that.
3. The sbatch file then requeues its own job (`scontrol requeue`). The job restarts after
   a few minutes with the same job ID and arguments, and resumes at the first chunk
   without a record (§4). Its Slurm log is appended to.
4. When all chunks are done, the loop script exits with code 0 and the job ends for good.

The loop script's output, and a start and end line for every requeue, are appended to
`output.log` in the run folder (found with `aqt_run_folder.py`). To test the chain on the
mini benchmark, submit a copy of the sbatch file with a smaller `PAUSE_AFTER_HOURS`, e.g.
`0.25` with `backend=PIAST-Q mini nshots_per_circuit=100` pauses at the first chunk boundary after 15 minutes (a value shorter than one chunk's run time pauses after every chunk). A requeue reruns the
copy of the sbatch file Slurm stored at submission, so edits to the file don't reach a
running job.

**Stopping a job cleanly (STOP file).** `touch datafiles/<run folder>/STOP` (the run folder is in the first lines of the Slurm
log). The loop script finishes the chunk it is on, with its POST health check, then deletes the file and ends with exit code `4`
instead of starting the next chunk; Slurm shows the job as `FAILED 4:0`, and `run_aqt_loop_with_retries.sbatch` neither
retries nor requeues after it. If the file is already there when the job (re)starts, it ends before the PRE health check. To
cancel a stop that has not happened yet, remove the file before the chunk ends. Unlike `scancel`, no running chunk is lost, and
finished chunks are kept: submit the same command again to resume.

**Changing the code or the sbatch settings in the middle of a run.** The STOP file and a dependent job do it without losing
a chunk. Edit the code (or the settings at the top of the sbatch file), then, from the project root on the login node:

```bash
sbatch --dependency=afterany:<jobid> source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_aqt_loop_with_retries.sbatch backend=PIAST-Q full
squeue -u $USER                                       # the new job waits as PD (Dependency)
touch datafiles/<run folder>/STOP                     # the running job ends after its chunk, with exit code 4
```

`<jobid>` is the running job, and the command is the one it was submitted with, with the same arguments (`layout=`,
`calibration_file=`, `nshots_per_circuit=` and so on): other arguments make another run name, so another run folder. When the
running job ends, the waiting one starts and resumes the run folder at the first chunk without a record, with the code and the
sbatch file as they are then (a running job keeps the copy of the sbatch file it was submitted with). Notes:

- Use `afterany`: the STOP exit code 4 counts as a failure for Slurm, so `afterok` would never start the job.
- Submit the waiting job before the run ends. A job that finds its run finished starts a new run folder, so if the run
  finished first, `scancel` the new job.
- A pause and requeue (above) keeps the job ID, so the dependency waits through it.
- `scancel` of the running job instead of the STOP file starts the waiting job too, but loses the chunk in progress.
- The chunks before and after the stop come from different code versions, so keep the change compatible with the records
  already saved in the run folder.

**Found on PCSS:**

- The worker nodes reach the PCSS API (`https://api.quantum.psnc.pl`) directly, and PIAST-Q
  returns the same objects there as from a PC.
- A job cannot submit new jobs from a worker node, so it continues by requeueing itself
  (`scontrol` works there). SSH from a worker node to the login node is refused as well.
- The venvs are Linux venvs built on the cluster (*Access and setup* above), not copies
  from a PC. The sbatch files call `venvs/aqt.venv/bin/python` (or `local_sim.venv`)
  directly.
- `tokens/aqt_piast-q_token` goes on the cluster too. It is a bearer
  credential, so make it readable only by yourself (`chmod 600 tokens/aqt_piast-q_token`).
- The job is network-bound: it waits on PIAST-Q almost all the time and uses the CPU only
  to transpile. All sbatch files ask for 1 CPU: Slurm hands out a whole core (2 threads)
  anyway, and the mini simulation ran no faster on 2 CPUs than on 1 (2026-09). The local
  AQT simulation is effectively single-threaded: its circuits are too small for Aer to parallelize.

## 5. Timing, billing and circuit check

Every chunk and health-check record gets a `timing` block and a `circuit_check`:

- **QPU time** (`qpu_s`): PCSS reports none, so it is modelled per circuit from its shot count: 3 s up to
  10 shots, 15 s at 200 (the maximum), linear in between (`aqt_qpu_s_per_circuit()` in `source/core/records/job_timing.py`). A chunk whose shots are
  split into parts (§4.2) is modelled per batch at that batch's shot count (`batch_shots` in the timing block).
- **Queue time** (`queue_s`): AQT runs synchronously, so per batch it is the wall time of the blocking call
  minus the QPU time; `submit_s` is 0.
- **Circuit check**: `not_available` — PCSS does not return the executed circuits. The analysis checks instead
  that every transpiled circuit of the run sits on the same physical qubits (`qubit_consistency`).
- The cost summary counts each execution of a split chunk as circuits (`n_split_chunks` in `qpu_workload`), and adds the
  circuits and shots executed in failed attempts, whose results were lost (`n_failed_circuits`, `n_failed_shots`; not jobs).
  AQT bills per circuit (each circuit is its own PCSS task), so a failed attempt counts only the circuits and shots that
  finished before the failure, and their modelled QPU time is part of the summary's QPU time; a health check that timed out
  adds nothing (PIAST-Q was unreachable) (main USER_MANUAL §7).

The full comparison of all QPUs is the timing table in main USER_MANUAL §7; the code is in `source/core/records/job_timing.py` (timing) and `source/core/records/circuit_check.py` (circuit check).

## 6. Outputs & post-processing

**Circuit drawings.** The submit, simulate and health-check scripts save the first circuit of every
error-mitigation role (bare, NEC, RC, RC+NEC, cRC, cRC+NEC), the first SPAM circuit and the health-check
circuit, before and after transpilation, as PDFs in `datafiles/<run_folder>/circuit_drawings/`
(`01_bare_before_transpilation.pdf`, `01_bare_after_transpilation.pdf`, … numbered per role; `00_SPAM_…` and
`health_check_…` for SPAM and the health check; main USER_MANUAL §4).

Per-chunk results land on disk synchronously as each chunk finishes — no
cloud job to poll later. Health checks live in the run folder, in `health_checks/`,
with a single counter that keeps growing across invocations:

```
datafiles/<run_folder>/
├── health_checks/
│   ├── health_check_1.json     # PRE of the first invocation; `chunks` = every chunk this invocation submits
│   ├── health_check_2.json     # after chunk 0  (`chunks` = [0])
│   ├── health_check_3.json     # after chunk 1  (`chunks` = [1])
│   └── …
├── job_records/job_record_chunk_{nc}.json
├── output.log                  # the sbatch files' log (§4.3)
├── retry_state                 # run_aqt_loop_with_retries.sbatch only, while it is retrying (§4.3)
└── STOP                        # only between `touch` and the loop script honouring it (§4)
```

The script decides once, at the start, which chunks this invocation submits: those
without a `submitted` / `complete` record (a failed chunk is resubmitted). Each circuit's submit
and result query is retried on connection errors (`_aqt_retry`). A chunk that still raises writes a
`failed` record — `failure_stage` `"submission"` (`"submission failed …"`: a circuit could not be
submitted) or `"execution"`, written with `status` near the top — (`n_executed_circuits` and `n_executed_shots`; QPU time of its finished batches and of the circuits finished inside the failing batch, the rest of that batch's
wall time as queue time) before the error propagates; the retry copies it to
`job_records/failed_attempts/job_record_chunk_<nc>_failed_<k>.json`, and the summary counts it under
`failed_attempts` (main USER_MANUAL §7). The PRE
health check records that whole list as `chunks`, each POST check the one chunk it
follows. `P011` is the pass metric (`> 0.5`). `source/analysis_scripts/run_post_process.py`
plots them next to the per-chunk time bars into
`analysis/plots+summaries/<run_folder>/health_checks.png` (with hundreds of chunks set
`ZOOM_X` in `source/analysis_scripts/plot_health_checks.py`).

`submit.log` (appended, never overwritten) holds one entry per submission, the same in every QPU folder
(`append_submit_log()`): the time, backend, run folder, chunks and shots, the layout, the run qubits with the
calibration summary of `calibration_snapshot.json`, and the run's health-check numbers, plus the loop's chunk range, total wallclock, QPU-time estimate and mean time per chunk.

**Logs of unattended runs** (all appended, never overwritten):

| Log | Written by | Where |
|---|---|---|
| `output.log` | `run_aqt_loop.sbatch`, `run_aqt_loop_with_retries.sbatch` (loop output plus a start/end line per requeue or retry) | `datafiles/<run_folder>/` |
| `<job-name>_<jobid>.out` | Slurm, for every sbatch file (appended across requeues) | the project root (the submission folder) |

None of them is committed (`.gitignore`).

**Post-processing.** `python source/analysis_scripts/run_post_process.py mini --include-real-data` (or `full`) merges, analyzes, summarizes and plots
this QPU's runs together with the other QPUs' — see main USER_MANUAL §10 for the script, its `RUN_THESE` / `REAL_QPUS` lists,
the analysis of a still-collecting run, and the outputs.

`run_post_process.py` runs the other scripts in `source/analysis_scripts/`, passing each run to them
through the `QPU_*` environment variables; each also works standalone (main USER_MANUAL §10).

## 7. Known issues

- **The Qiskit 1.4 gap.** `aqt.venv` runs Qiskit 1.4.6 (§1) — it's a real constraint from `qiskit-aqt-provider`, not a bug, but
worth remembering when debugging anything that behaves differently here than
on the other four backends. The QPY-version incompatibility this causes for
raw circuit chunks is worked around by the build step in §4; if a chunk still
crashes with a `QPY format version ... isn't supported` error, check whether
it has a `chunk_{nc}_qiskit_v13.qpy` in `datafiles/circuits/all_to_all_Nt<depths>/chunks_qiskit_v13_qpy/` and
re-run `build(circuit_transpilation_LNN=False)` in `local_sim.venv`
if not.

## Creating the environment

From the project root (with Python 3.12: `python3.12`, or `python3` if that is 3.12; `py -3.12` on Windows):
```bash
python3.12 -m venv ./venvs/aqt.venv
source ./venvs/aqt.venv/bin/activate      # Windows PowerShell: .\venvs\aqt.venv\Scripts\Activate.ps1
pip install -r source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/requirements_aqt.txt
```
