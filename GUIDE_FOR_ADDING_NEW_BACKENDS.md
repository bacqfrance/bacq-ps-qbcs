# Guide for adding a new backend

This guide is for developers adding a new QPU provider (or a new service for an existing
provider) to the BACQ benchmark pipeline. Part 1 lists what a new backend must deliver and how
to check it, part 2 where in the code each piece goes. The existing backends are the templates — a new
backend may use any SDK or API; the table only shows which existing one to start from:

| Service name (`service_name`) | Provider SDK / access | Kind | Closest template for |
|---|---|---|---|
| `IBM_Cloud` | `qiskit-ibm-runtime` | asynchronous, batched jobs | cloud services with billed time and executed-circuit readback |
| `IQM_Resonance` | `iqm-client` (Resonance cloud) | asynchronous, packed jobs | IQM-style servers with a timeline and a raw job API |
| `MQSS_LRZ` | `mqss-qiskit` (LRZ) | asynchronous | services that report little (no QPU time, no executed circuits) |
| `QAAS_IT4I` | `qaas` via LEXIS/HEAppE (IT4I) | asynchronous, HPC job system | services behind an HPC scheduler; unreliable connections |
| `PCSS` | `qiskit-aqt-provider`, `pcss_qapi` (PCSS) | **synchronous** (blocks per batch) | services that return results directly; shot limits |
| `local` | Aer / fake backends | synchronous | — |

Throughout, "chunk" is one random-compilation sample: the benchmark circuits of every depth and
error-mitigation role (plus SPAM calibration circuits). A run is `sample_size` chunks (8 in the
mini benchmark, 480 in the full one); a provider job holds one or several chunks.

---

## Part 1 — Steps

### 1. Virtual environment

1. Create `venvs/<backend>.venv` with the provider's SDK and the Qiskit version it requires.
   **No myQLM** (`qat`) in it: circuits are built once in `local_sim.venv` and cached as QPY.
2. Write `source/benchmark_scripts/real_qpu_scripts/<BACKEND>/requirements_<backend>.txt`.
3. Check the SDK's Qiskit version reads the cached circuits: LNN runs use the default QPY version
   (`chunks_qiskit/`), all-to-all runs QPY v13 (`chunks_qiskit_v13_qpy/`, readable by Qiskit ≥ 1.x).
   If the SDK's Qiskit is too old for both, add a format to `bacq_ps_qbcs.build()`.
   An older Qiskit warns that each file was written by a newer Qiskit. That warning is
   expected, and `source/core/qbcs.py` silences it. Only an actual load error means the format
   is unreadable.
4. Add the venv to USER_MANUAL §3/§7.
5. Check that `source/core/qbcs.py` and `source/core/health_check.py` import in the new venv with `qat` blocked.

### 2. Study the backend: documentation and probe jobs

Before writing pipeline code, find out how the backend reports and bills everything the
benchmark records. The code of steps 4–8 is built on these findings.

1. Store the credentials in a `tokens/<backend>_token` file (the `tokens/` folder is git-ignored;
   list the new file in `tokens/README.md`, with what it holds and where to get it) and read it with `source/core/qpu/tokens.py: read_token()`,
   never in a script.
2. Read the provider's documentation (SDK, REST API, pricing and billing pages, user guides of
   the hosting centre) for: job states, result format, timestamps, how QPU time is measured and
   billed, limits, how the qubit layout and calibration data are exposed.
3. Probe the real backend with a script: first a health check, then bigger jobs (more circuits,
   more shots, a real benchmark chunk). Some behaviour shows only at scale — the queue/execution
   split of a many-circuit job, upload timeouts (VLQ), billing per job vs. per circuit. For each
   job, dump everything the job and result objects offer and answer:
   - **Time:** status values, timestamps and their time zone, compilation / execution / billed
     time — which of them the provider bills, and with what rounding or minimum (compare with the
     provider's dashboard or consumption counter).
   - **Results:** result format, bit order.
   - **Layout:** which physical qubits the job used, and where that is reported.
   - **Calibration:** where the calibration data come from (API endpoint, calibration-set ID),
     their format and metrics, whether a job records which calibration it ran with, and when the provider made
     the calibration (saved as the snapshot's `measured_at`; `null` when it says nothing).
   - **Executed circuits:** whether the executed (server-transpiled) circuits can be read back, and
     in what form (gate names, qubit indexing, added gates such as MOVEs). The circuit check
     (step 6) compares them with the submitted ones.
4. **Keep the probe script, all its raw output and a note of the findings.** The code development relies on them later:
   they decide what the timing block records (step 4), how the cost is modelled (step 8), and
   they become the test fixtures of step 9.
5. Find out the provider's limits: circuits per job, shots per circuit, walltime, upload size,
   rate limits. Note which of them made a job fail (VLQ: 48-chunk jobs timed out between IT4I and
   IQM; AQT: at most 200 shots per circuit, at most 50 circuits per job, a limit of the `qiskit-aqt-provider` client library, which rejects larger jobs before sending) and which were given
   by the provider's staff (EQE1: 70 circuits per job).
6. If the service is reached through a flaky layer (an HPC scheduler, a gateway), plan for calls
   that never return: the SDK may set no network timeouts at all (qaas does not).

### 3. Scripts

Create `source/benchmark_scripts/real_qpu_scripts/<BACKEND>/` with, following the closest
existing backend:

| Script | Does |
|---|---|
| `qbcs_real_submit_<backend>.py` | health check before (waited for; abort if degraded), qubit selection, submission, health check after (submitted only) |
| `qbcs_real_collect_<backend>.py` | collects the finished jobs (non-blocking), the finished health checks, merges, analyzes |
| `qbcs_real_simulate_<backend>.py` | the same run on Aer with the backend's calibration (optional, recommended) |
| `health_check_real_<backend>.py` | a standalone submit-and-collect health check |
| `check_real_qpu_job_status_<backend>.py` | read-only: each job's status next to the records on disk |
| `backends.json` | the machines the scripts may run on and each one's topology: `{"backends": {"<name>": {"lnn": true}}}` (`true` = linear-nearest-neighbour circuits, `false` = all-to-all) |
| `readme_<backend>.md` | setup, the scripts, limits, timing and cost definitions |

The service name is the access platform. It is defined once, with the run-name service field
(`ibm-cloud`, …), in `QPUS` in `source/core/records/run_names.py`; scripts get it with `service_name("<QPU folder>")`.

A synchronous service (AQT) has no collect script: its submit loop receives the results.

Every script must:
- start with `sys.stdout/stderr.reconfigure(encoding="utf-8")`, then the `sys.path.insert` + `os.chdir` to the project root (`"../../../.."` from these folders), as the sibling scripts do;
- take the benchmark size from the command line: `BENCH = parse_benchmark_args(sys.argv[1:])`
  (`source/core/runner_support/benchmark_args.py`; `mini`, `full` or custom — there is no default, a call without arguments prints
  the usage), and whichever of `NSHOTS`, `SAMPLE_SIZE`, `LIST_TIMES` the script needs from it;
- take the qubit layout from `LAYOUT = parse_layout_args(sys.argv[1:])` (`layout=optimized|random|lowest|15,14,21`,
  `calibration_file=`): submit scripts call `resolve_run_layout(qbcs, backend, RUN_FOLDER, LAYOUT, label=BACKEND_NAME)`
  (`source/core/qpu/calibration_input.py`) right after `build()`; a QPU whose SDK reports its calibration passes
  `live_calibration="<QPU>"` in its submit script (the live calibration is always used, so `calibration_file=` is
  refused; the layout is scored on the saved `calibration_snapshot.json`, whose `measured_at` the live-calibration helper
  `provider_calibration_time()` in `source/core/qpu/qpu_snapshot.py` fills from the SDK, a new QPU adding its case there);
  a QPU that reports none passes `calibration_required=True` (random layout without a file, `optimized` only
  with one), and its simulate script `layout_required=True` and `calibration_any_layout=True` (the file is the simulated
  noise data, allowed with any layout); collect and status scripts use `parse_layout_args(..., collect=True)`, `parse_date_arg()` and
  `find_collect_run_folder()` (one run) / `find_collect_run_folders()` (without `layout=` and `date=`: every run with something left to collect, in turn);
- take the machine from the required `backend=NAME` argument, checked against the folder's `backends.json`:
  `BACKEND_NAME, BACKEND_CFG = parse_backend_config(sys.argv[1:], Path(__file__).parent)` and
  `LNN = BACKEND_CFG["lnn"]` (`source/core/runner_support/benchmark_args.py`; there is no default, and an unknown name stops
  the script before it connects). Pass `extra_keys=("backend",)` to `parse_benchmark_args`. A health check
  or other script without benchmark arguments passes `others_ok=False`;
- take the run folder name from `source/core/records/run_names.py`: `RUN_FOLDER = dated_run_folder(qpu_run_name("<QPU folder>", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM, LAYOUT.tag))` in the submit and simulate scripts (`data_type="sim-qpu-noise"`; `dated_run_folder()` adds the date and run number (`_<YYYY-MM-DD>_#<N>`): the latest unfinished folder of those settings, else a new one for today; submit scripts pass `new_run=` from `pop_force_new_run_flag()`, the `--force-new-run` option), `find_collect_run_folder(qpu_run_prefix(...), LAYOUT, parse_date_arg(sys.argv[1:]))` (in `source/core/runner_support/benchmark_args.py`; pass `"date"` in `parse_benchmark_args(extra_keys=...)`) in collect / status scripts, `dated_run_folder(health_check_run_name(...))` in the health check. Add the provider, its run-name service field and its service name to `QPUS` in `run_names.py` (the scripts then use `SERVICE_NAME = service_name("<QPU folder>")`) and a constant for the service name next to `IBM_CLOUD` & co. there, which the library code (`qbcs.py`, `health_check.py`, `job_timing.py`) compares against, the backend's metadata to `bacq_ps_qbcs_summary_json.py`, and a case in `source/tests/test_20_run_names.py`;
- use the same config block (`backend=`, `DO_SPAM`, the benchmark arguments) in submit and collect.
- reuse the shared steps: a simulate script gets the calibration snapshot (the provider-specific part) and
  calls `simulate_run()` (`source/core/runner_support/run_simulation.py`); a collect script connects and calls `collect_runs()`
  (`source/core/runner_support/run_collection.py`); an asynchronous submit script connects and calls `submit_async_run()` (`source/core/runner_support/run_submission.py`: it prints the config, builds, picks the layout, runs both health checks, submits and writes `submit.log`), passing only what the provider does differently; the accompanying health checks use `ACCOMPANYING_HEALTH_CHECK_NSHOTS`
  (`source/core/qpu/providers.py`).

### 4. Job records and the timing block

Each chunk gets `datafiles/<run>/job_records/job_record_chunk_<nc>.json`, each health check
`datafiles/<run>/health_checks/health_check_<n>.json`. Required fields (see the existing
`submit_batch_*` / `submit_chunk_*` functions):

- `service_name`, `backend_name`, `chunk_id`, `job_id` (shared by the chunks of one job),
  `status` (`submitted` → `complete` or `failed`), `submitted_at` (UTC ISO), `nshots`,
  `n_spam`, `n_circuits`, the offsets `spam_start` / `circ_start` / `circ_end` of the chunk in its
  job, `n_circuits_in_job`, `layout`, `submitted_circuit_files`, `trotter_steps_list`, `roles`,
  `nsites`, `CONFIG`;
- after collection: `scheduled_at`, `finished_at`, `circuit_check`, and for a chunk record `result_path` (a health check keeps `counts`, `P011`, `verdict`, `threshold` and `collected_at` in its own record);
- a `timing` block (`source/core/records/job_timing.py` terms):

| Key | Meaning | Written by |
|---|---|---|
| `transpile_s`, `transpile_reused`, `submit_s` | CPU side of the submission | `submit_timing()` at submission |
| `qpu_s`, `qpu_s_source` | the QPU time the benchmark reports (billed, or a model — say which in `QPU_S_SOURCES`) | the collect-timing function |
| `qpu_exec_s`, `qpu_exec_s_source` | pure execution time if the provider reports it, else leave it out | the collect-timing function |
| `queue_s`, `queue_s_source` | end of submission → start of execution | the collect-timing function |
| `execution_start`, `execution_end` | the execution window (UTC) | the collect-timing function |
| `collect_s` | collecting this chunk's results (download, circuit check, saving) | `collect_results()` (automatic) |

All times are UTC. If the provider gives local times, convert them (EQE1's are naive
Europe/Berlin: `job_timing.local_naive_to_utc`, which handles the repeated DST hour).

The timestamp log `datafiles/<run>/timestamps_local_time.csv` (+ `timestamps_utc.csv`, used for
every duration) gets, per chunk: `chunk<nc>_transpile_start/_end`, `chunk<nc>_submit_start/_end`
(or `chunk<a>-<b>_submit_*` for a job of several chunks) and `chunk<nc>_collect_start/_end`. The
CPU time is summed from these labels, the collection time from the `collect` ones. Use
`self._log_timestamp()` / `self._timed_submit()`; `collect_results()` writes the collect labels. The analysis also reads the `qpu_chunk<nc>_start/_finished` pair (only for runs without transpile/submit
labels) and the `chunk<a> to chunk<b>_start/_end` pair of a batched job; `Starting chunk submission.` marks
the start of a submission session and `All chunks submitted.` its end. All other labels (AQT
`qpu_chunk<nc>_part<p>_*`, health-check and build labels) are informational and not read by the analysis.

### 5. Collection must be non-blocking

The collect script is run repeatedly until everything is in. One pass must:
1. query each job's status **once**, without waiting (`_remote_job_state()`);
2. collect the finished jobs, in whatever order they finished;
3. leave unfinished ones as `submitted`, mark failed ones `failed` (with the reason);
4. do the same for health checks — except a single check requested by number (the submit
   script's pre-submission check), which is waited for.

A run with gaps is then analyzed from the collected chunks (`+nsamples-<collected>` files, see USER_MANUAL
§10.2, "Incomplete runs"). Do not rely on chunks finishing in order.

### 6. Circuit check

After collecting, compare the circuits the provider executed with the ones submitted
(`source/core/records/circuit_check.py`): `check_ibm` (IBM `job.inputs["pubs"]`), `check_iqm` (IQM job
payload), `check_vlq` (qaas transpiled circuits). If the provider cannot return them, record
`CC.not_available("<why>")`; the analysis then checks that all circuits used the same physical
qubits instead. The check never raises: wrap it and return `CC.check_failed(method, error)`.

### 7. Health checks

`submit_health_check()` / `collect_health_check()` in `source/core/health_check.py` handle every service:
the 3-qubit check circuit on the run's qubits, 100 shots, P(011) > 0.5 = functional. Health
check records get the same timing block (with `collect_s`, labels `health_check_<n>_collect_*`). Pass the script's `BACKEND_NAME` as `backend_name=` (the name in the check's messages), and a service that opens a session or batch should wrap it in a context manager that always closes it, as `ibm_batch()` in `source/core/qpu/providers.py` does. An explicit `qubit_set` is validated against the backend's target (`layout_selection.check_forced_layout()`) before anything is transpiled, so a target that lists the real qubits and couplers needs nothing extra.

### 8. Summary: metadata, QPU time, cost

In `source/analysis_scripts/bacq_ps_qbcs_summary_json.py`:
- `SERVICE`: comes from the run name's service field (`service_name_of_run_service()`); nothing to add;
- the metadata block: `QPU_PROVIDER`, `ARCHITECTURE`, `DEVICE_LABEL`, `QPU_SERVICE`, `PRICING`,
  `FREE_BECAUSE`;
- a pricing branch if none of the existing models (IBM per minute, Braket per task + shot,
  Scaleway per circuit + shot, VLQ's synthetic one) fits. Price from `qpu_workload()` — jobs,
  circuits, shots of the analysed chunks and health checks — and state the model, the prices,
  their source URL and date in `cost_explanation`. Mark estimates as estimates. Price the failed jobs too, in the
  provider's billing unit (`n_failed_jobs`, `failed_chunks_per_job`, `n_failed_circuits`, `n_failed_shots`; how
  each QPU counts them: USER_MANUAL §7).

In `source/core/records/job_timing.py`: a `TIMING_NOTES[<service>]` line explaining the QPU time, and new
entries in `QPU_S_SOURCES` / `QPU_EXEC_S_SOURCES` for new sources. If jobs run one after another
(no overlapping queues), add the service to `SYNCHRONOUS_SERVICES` — the queue time is then the
sum of the waits, not the longest.

### 9. Tests — before trusting any number

Tests go in `source/tests/`, run without network or QPU, and use saved real data as fixtures:

1. **Timing on a real job.** Save one finished job's raw data (timeline, metrics, timestamps —
   step 2) as a JSON fixture in `source/tests/<backend>_check_data/` and test that the collect-timing
   function reproduces `qpu_s`, `qpu_exec_s`, `queue_s` and the execution window by hand-checked
   values (template: `test_vlq_timing_of_a_real_job` in `source/tests/test_13_iqm_vlq.py`, the
   Resonance/IBM cases in `source/tests/test_11_circuit_check_and_timing.py`).
2. **Executed-vs-submitted circuits.** Save what the provider returned for a real job next to
   the submitted QPY (`source/tests/circuit_check_data_executed_vs_submitted/`, its `manifest.json`):
   it must come back `match`, and altered copies `mismatch`.
3. **Records → summary.** Write synthetic records with timing blocks into a `tmp_path` run and
   check `qpu_queue_breakdown`, `qpu_workload`, `collect_time_breakdown` and the cost
   (templates in test_11 / test_12 / test_14).
4. **Non-blocking collection.** Fake the provider's status (monkeypatch `_remote_job_state`)
   and check that finished, pending and failed jobs are handled (template: test_14).
5. **Provider limits** (like AQT's shot split, test_12): the splitting, and that the SDK call
   refuses anything above the limit.
6. **Connection failures**, for a flaky service: logged and time-limited calls (test_13's
   connection-log test).

Then: add every new or changed test to the test table in USER_MANUAL §9, and run the whole suite
(`pytest source/tests/`, in the background; it builds its own caches in `datafiles/tests/circuits/`, so it is safe while a
benchmark is running).

Finally, check on the real service, in this order: standalone health check → mini benchmark
(submit, collect twice — once while jobs are running, once after) → its post-processing. Compare
the summary's QPU time and cost with what the provider's own dashboard says.

### 10. Post-processing and docs

- Add the run-name fragment `real-data_<provider>-<backend>` (e.g. `real-data_ibm-fez`) to `REAL_QPUS` in `source/analysis_scripts/run_post_process.py`, and `sim-qpu-noise_<provider>-<backend>`
  (cut off after the backend, so every layout is covered) to `SIMULATED_QPUS` if the backend has a `qbcs_real_simulate_*.py`.
- USER_MANUAL: the backend in §3 (setup), §5.4 (the `<provider>-<backend>` and `service` fields), §7 (timing table: what QPU and queue time mean for it,
  the token table), §10.2 (the cost paragraph), and the test table; both QUICK_START guides.
- `readme_<backend>.md` in the backend's folder (`source/benchmark_scripts/real_qpu_scripts/<BACKEND>/`), with the
  same top-level sections as the other QPU readmes: 1. Prerequisites, 2. Connecting, 3. Config knobs, 4. Workflow (with a numbered
  4.x Simulating locally with this QPU's calibration subsection), 5. Timing, billing and circuit check, 6. Outputs & post-processing, 7. Known issues, then Creating the environment
  (test_21 checks this). Provider-specific material goes into subsections.

---

## Part 2 — Where the code changes go

Search for an existing service name (e.g. `"IQM_Resonance"`) to find every branch; the list
below is what a new asynchronous service needs. Its job-status query goes in `_remote_job_state()`
(`source/core/qpu/remote_jobs.py`); without it, collection stops with an "unknown service" error.

### `source/core/qbcs.py`, `source/core/health_check.py`, `source/core/qpu/remote_jobs.py`

| Place | Change |
|---|---|
| `remote_jobs._remote_job_state()` | the non-blocking status query → `"done"` / `"pending"` / `"failed"` |
| `bacq_ps_qbcs.submit()` | a `service == "<NEW>"` branch: transpile, SPAM, how many chunks per job, calling the batch/chunk submit function |
| `submit_batch_*` / `submit_chunk_*` | build the job (SPAM + chunk circuits, offsets), submit it through `_timed_submit`, write the records |
| `collect_results()` / `_collect_chunk()` | dispatch to the new `get_results_by_chunk_*` |
| `get_results_by_chunk_*` | fetch the job's result once (cache it for the job's other chunks), split the counts per chunk (reverse Qiskit bit order), save counts / fidelity / SPAM counts, timing, circuit check, record `complete` |
| `_check_executed_*` | the executed-vs-submitted comparison (step 6) |
| `health_check.submit_health_check()` | submission of the check circuit and its record for the new service |
| `health_check._collect_health_check_one()` | its collection (result, timing, circuit check) |
| `health_check._check_executed_health_check()` | its circuit check |
| `resolve_physical_qubits()` | only if qubit selection differs from the generic one (`layout_selection.select_physical_qubits`, scored on the transpiled circuit) — e.g. VLQ's star topology |

Keep provider-specific code in its own module when it grows (as `source/core/qpu/iqm_vlq.py` does for
VLQ, and `source/core/qpu/aqt.py` for AQT) and import the SDK inside functions, so `qbcs.py` and
`health_check.py` load in every venv.

### `source/core/records/job_timing.py`

- `<service>_collect_timing(...)`: the finished job → the timing block of step 4. Never raises;
  missing values are `None` with source `"unavailable"`.
- `qpu_execution_s()`: how to read the execution-only time from the block.
- `service_of_record()`: recognize the backend name.
- `TIMING_NOTES`, `QPU_S_SOURCES`, `QPU_EXEC_S_SOURCES`, `SYNCHRONOUS_SERVICES` (step 8).

### `source/core/records/circuit_check.py`

- a `check_<service>()` if the provider returns the executed circuits in a new form.

### `source/core/analysis/run_timing.py`

Usually nothing: `aggregate_run_timing`, `qpu_queue_breakdown`, `qpu_workload` and
`collect_time_breakdown` read any record with a timing block. Change them only for a new rule
(e.g. a new way jobs share time).

### `source/analysis_scripts/bacq_ps_qbcs_summary_json.py`

- the metadata block, `PRICING` and its cost branch (step 8).

### `source/analysis_scripts/run_post_process.py`

- `REAL_QPUS` (`real-data_<provider>-<backend>` entries), `SIMULATED_QPUS` (`sim-qpu-noise_<provider>-<backend>`) and `RUN_THESE`.

### Unreliable connections

If calls to the service can hang, route every network call through a logged, time-limited
wrapper, as `source/core/qpu/iqm_vlq.py` does (`call()`, `TIMEOUTS`, `iqm_vlq_connection.log` in the run
folder): each call is logged `START` / `OK` / `FAIL` / `TIMEOUT` with its duration, and a
timeout records where the call was waiting. A timed-out *submission* may still arrive at the
provider — check the job list before submitting again.

---

## Checklist

- [ ] venv without myQLM; `qbcs.py` and `health_check.py` import in it with `qat` blocked
- [ ] token file, git-ignored
- [ ] documentation read; probe jobs (health check, bigger jobs) run; script, raw output and
      findings (time, billing, layout, calibration, executed circuits) kept
- [ ] scripts in the backend's folder: project root, UTF-8, same config in submit and collect
- [ ] records with all fields and the timing block; UTC everywhere
- [ ] timestamp labels: transpile, submit, collect
- [ ] non-blocking collection; failed jobs marked; pre-submission health check waited for
- [ ] circuit check, or `not_available` with the reason
- [ ] health checks submitted and collected
- [ ] summary: service name, metadata, QPU time source, cost model with sources
- [ ] tests on saved real data (timing, circuit check), records → summary, non-blocking collection
- [ ] USER_MANUAL §9 test table updated; full suite passes
- [ ] real-service check: health check → mini benchmark → post-processing, compared with the provider's dashboard
- [ ] `backends.json` with every machine's real topology; `backend=` required by every script
- [ ] `REAL_QPUS`, `SIMULATED_QPUS`; USER_MANUAL §3/§5.4/§7/§9, §10.2 cost paragraph; both QUICK_START guides; `tokens/README.md`; `readme_<backend>.md`
