"""
qbcs_real_submit_aqt_loop.py
Submit a BACQ benchmarking run to the real AQT QPU (PIAST-Q), one chunk per
iteration, with a live wallclock ETA and a QPU-time-budget projection.

  - The loop runs over the chunks from the CHUNK_START to the CHUNK_STOP knob; each pass submits exactly
    one new chunk (submit_chunk_aqt is idempotent, so already-done chunks are
    instant skips -> the script resumes cleanly after an interruption).
  - Health check runs once before the loop (aborting the run if the QPU is
    degraded), then once after every chunk. For a health check on its own,
    use health_check_real_aqt.py.
  - Per-chunk wall_seconds and the modelled QPU time (timing.qpu_s, the same
    AQT model the summary uses) are read back from job_record_chunk_{nc}.json
    (written by submit_chunk_aqt) and printed to stdout (capture it with
    nohup / a redirect); a summary is appended to submit.log in the run folder.
  - iter_seconds = whole-iteration wallclock (submit + health check).
  - ETA(last N) = mean iter_seconds of the last ETA_WINDOW chunks x remaining.
  - ETA(avg)    = mean iter_seconds over all chunks this run x remaining.
  - QPU budget  = mean modelled QPU time per chunk x total chunks.
  - Time limit: if the environment variable AQT_PAUSE_AFTER_HOURS is set and that
    many hours have passed since the script started, it exits with code 3 before
    starting the next chunk (a clean pause; rerunning resumes). The running chunk
    and its POST health check always finish. PCSS_Quantum_UI/run_aqt_loop.sbatch and run_aqt_loop_with_retries.sbatch
    set it and requeue their Slurm job on code 3. Unset: no limit.
  - STOP file: if datafiles/<run folder>/STOP exists (`touch` it), the script ends with exit code 4 before the next
    chunk (or before the PRE health check, if it exists at start); the running chunk and its POST health check finish
    first. The file is deleted when honoured. run_aqt_loop_with_retries.sbatch does not rerun after code 4.
  - Dropped connections to PIAST-Q are retried inside source/core/qpu/aqt.py: _aqt_retry (per circuit).
    A circuit not executed within 30 min of its submission (AQT_TASK_TIMEOUT_S; PIAST-Q probably down,
    in maintenance or reserved) is not an error: the script prints a warning and ends with exit code 13
    (a chunk's attempt is saved as failed and a rerun resumes it; the same for a health check's circuit).
    A health check that still can't be executed after those retries ends the script
    with exit code 12 (QPU health unknown; no record is written for it); a health
    check that ran and found the QPU degraded ends it with exit code 10. Unusable
    arguments or a missing token end it with code 2; any other error with code 1.

Benchmark size (required — there is no default), from the command line (source/core/runner_support/benchmark_args.py; readme: "Benchmark size and run folders"):
  mini · full · mini|full nshots_per_circuit=N ·
  nshots_per_circuit=N n_samples=M n_depths=K (or nt_list=1,5,9)
"""

import os
import sys
import json
import time
import datetime
T_SCRIPT_START = time.perf_counter()   # the time limit (AQT_PAUSE_AFTER_HOURS) counts from here
from collections import deque
from statistics import mean
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))   # datafiles/ paths are relative to the project root
from source.core.qpu.providers import ACCOMPANYING_HEALTH_CHECK_NSHOTS
from source.core.runner_support.benchmark_args import make_usage, usage_if_requested   # light (numpy only): the usage is printed before anything else loads
from source.core.records.run_names import qpu_run_name, service_name, dated_run_folder
from source.core.qpu.tokens import read_token
usage_if_requested(sys.argv[1:], make_usage(__file__, "Submit a BACQ benchmark run to AQT PIAST-Q, one chunk per iteration, with a live ETA and a QPU-time-budget projection. Synchronous: each chunk's result is saved as it completes, so there is no separate collect script.", layout="submit_no_live"))   # no arguments or --help: print the usage and do nothing

from pathlib import Path
from pcss_qapi import AuthorizationService
from pcss_qapi.aqt.provider import PCSS_AQTProvider
from source.core.runner_support.benchmark_args import parse_benchmark_args, DT, parse_backend_config, parse_layout_args, parse_run_folder_arg, pop_force_new_run_flag
from source.core.qpu.calibration_input import resolve_run_layout, layout_summary
from source.core.qbcs import bacq_ps_qbcs
from source.core.health_check import submit_health_check
from source.core.records.health_check_records import (load_health_checks, install_submit_crash_log, format_chunk_range,
                                         EXIT_HC_DEGRADED, EXIT_HC_NOT_SUBMITTED, EXIT_AQT_TASK_TIMEOUT,
                                         append_submit_log, print_run_config, health_check_log_line)
from source.core.qpu.aqt import AQTTaskTimeout, AQT_TASK_TIMEOUT_S, STOP_EXIT_CODE, PAUSE_EXIT_CODE, take_stop_file, pause_after_hours, pause_due
from source.core.records.job_records import chunks_without_submission
from source.core.circuits.SPAM_functions import compute_and_save_spam_confusion_matrix
from source.core.records.job_timing import aqt_shot_parts, AQT_MAX_SHOTS
from source.core.common import rel_path

# ── Config ────────────────────────────────────────────────────────────────────
FORCE_NEW_RUN, ARGV = pop_force_new_run_flag(sys.argv[1:])   # --force-new-run: start a new run folder even if the latest run of these settings is unfinished
BACKEND_NAME, BACKEND_CFG = parse_backend_config(ARGV, Path(__file__).parent)   # required backend=NAME, checked against backends.json
SERVICE_NAME = service_name("AQT_PIAST-Q")   # the access platform, as records and qbcs call it (source/core/records/run_names.py)
BENCH            = parse_benchmark_args(ARGV, extra_keys=("backend", "run_folder"))   # full / mini / custom — see source/core/runner_support/benchmark_args.py
LAYOUT           = parse_layout_args(ARGV, calibration_required=True)   # random unless calibration_file= is given — see source/core/runner_support/benchmark_args.py
NSHOTS           = BENCH.nshots
SAMPLE_SIZE      = BENCH.sample_size
LNN              = BACKEND_CFG["lnn"]   # the backend's topology, from backends.json
DO_SPAM          = True
LIST_TIMES       = BENCH.list_times
CHUNK_SIZE       = int(len(LIST_TIMES) * 6)           # 6 circuits per depth
EMS              = ["SPAM", "NEC", "RC+NEC", "cRC+NEC"] if DO_SPAM else ["NEC", "RC+NEC", "cRC+NEC"]
RUN_SETTINGS     = qpu_run_name("AQT_PIAST-Q", BACKEND_NAME, BENCH.tag, LNN, DO_SPAM, LAYOUT.tag)   # the run name without its date
RUN_FOLDER       = parse_run_folder_arg(ARGV, RUN_SETTINGS) or dated_run_folder(RUN_SETTINGS, new_run=FORCE_NEW_RUN)   # run_folder=<name> pins the folder (the sbatch files resolve it once per job)

CHUNK_START      = 1                # first chunk to submit (1-based)
CHUNK_STOP       = SAMPLE_SIZE      # last value, inclusive
ETA_WINDOW       = 1                # ETA uses the mean iteration time (submit + health check) of the last N real chunks

# ── Connect to PIAST-Q ────────────────────────────────────────────────────────
print(f"\n[{BACKEND_NAME} — connection]")
ACCESS_TOKEN = read_token("aqt_piast-q_token")
AuthorizationService.login(ACCESS_TOKEN)
provider     = PCSS_AQTProvider()
real_backend = provider.get_direct_access_backend()

print(f"  Backend: {real_backend.name}")

# ── Print run config ─────────────────────────────────────────────────────────
print_run_config(RUN_FOLDER, nshots=NSHOTS, sample_size=SAMPLE_SIZE, lnn=LNN, do_spam=DO_SPAM,
                 list_times=LIST_TIMES, chunk_size=CHUNK_SIZE,
                 extra=[("HC folder", f"{RUN_FOLDER}/health_checks"),
                        ("Shot parts", f"{aqt_shot_parts(NSHOTS)}  (AQT runs at most {AQT_MAX_SHOTS} shots per circuit)"),
                        ("Chunk range", f"{CHUNK_START} → {CHUNK_STOP}")])

# ── Paths / timing state ────────────────────────────────────────────────────
run_folder_path = Path("datafiles") / RUN_FOLDER
job_records_dir = run_folder_path / "job_records"
run_folder_path.mkdir(parents=True, exist_ok=True)

# STOP file: `touch datafiles/<run folder>/STOP` ends the script at the next chunk boundary (or at once, before the
# PRE health check, when it exists at start) with STOP_EXIT_CODE; the running chunk and its POST health check finish first.
# The file is deleted when honoured, so a later job of the run is not stopped by it.
def stop_if_requested(where):
    if take_stop_file(run_folder_path):
        print(f"\n  ⚠️  STOP file found (removed) — stopping before {where} (exit code {STOP_EXIT_CODE}). "
              f"Finished chunks are kept; rerun to resume.", flush=True)
        sys.exit(STOP_EXIT_CODE)

# Build once, before the health check, so the run can be recorded (run_info.json) even if
# that check aborts it — see qbcs.record_run_info(). build() converts all CHUNK_STOP chunks
# to QPY and sets chunk_paths / n_chunks / chunk_size on the object. submit() then flips
# _status "built"->"submitted" and asserts "built" on entry, so the loop resets
# qbcs._status = "built" before each call.
qbcs = bacq_ps_qbcs(
    infidelity_total            = 0.0,
    ratio_ct                    = 0.0,
    noise_type                  = "pauli",
    sample_size                 = SAMPLE_SIZE,
    chunk_size                  = CHUNK_SIZE,
    list_circuit_times          = LIST_TIMES,
    error_mitigation_strategies = EMS,
    do_SPAM                     = DO_SPAM,
    nshots                      = NSHOTS,
    dt                          = DT,
)
qbcs.build(circuit_transpilation_LNN=LNN)   # all_to_all: loads the QPY-v13 chunks aqt.venv can read

# CHUNK_STOP was set to SAMPLE_SIZE above, which equals the number of submission
# chunks (qbcs.n_chunks, set by build()). Clamp it to n_chunks anyway, as a guard
# in case they ever differ, since submit(chunks=[...]) rejects a chunk number above n_chunks.
CHUNK_STOP = min(CHUNK_STOP, qbcs.n_chunks)

# The chunks this session submits: those of CHUNK_START..CHUNK_STOP (1-based) without a 'submitted' /
# 'complete' record (may have gaps). The PRE health check records the whole list, each POST check its chunk.
todo_chunks        = [nc for nc in chunks_without_submission(run_folder_path, CHUNK_STOP) if nc >= CHUNK_START - 1]
install_submit_crash_log(run_folder_path, todo_chunks)   # an uncaught error still leaves an entry in submit.log

# ── Physical qubits for this run (ions 0…19 of PIAST-Q) — after the crash log, before the first health check, so it runs on them too ──
# (recorded in <run>/transpiled_circuits/physical_qubits.json; submit() reuses the record)
print(f"  [Layout] requested: {layout_summary(LAYOUT, LAYOUT.qubits, LAYOUT.calibration_file)}")
run_qubits, CAL_SOURCE = resolve_run_layout(qbcs, real_backend, RUN_FOLDER, LAYOUT, label=BACKEND_NAME)

n_total     = CHUNK_STOP - CHUNK_START + 1
done_before = n_total - len(todo_chunks)
all_done    = not todo_chunks
print(f"  Already complete: {done_before}/{n_total}")
if todo_chunks:
    print(f"  This session:     chunks {format_chunk_range(todo_chunks)} ({len(todo_chunks)})")

# ── Health check PRE (once, before the loop; skipped if nothing left to do) ──
hc_pre_record = None    # the PRE check / the latest POST check record, for submit.log
hc_post_record = None
if not all_done:
    stop_if_requested("the PRE health check")
    # Health checks accompanying the run live in <run>/health_checks/ (auto-numbered).
    qbcs.record_run_info(RUN_FOLDER)   # so an aborted run is still analyzable (health checks only)
    # AQT's health check is one synchronous call (submit + run + collect in one) — the built-in
    # ~19-minute per-call retry (source/core/qpu/aqt.py: _aqt_retry) already covers a dropped connection.
    # If it still fails, nothing is written (the owner accepts no record here, unlike the other
    # backends) and the run aborts with EXIT_HC_NOT_SUBMITTED, distinct from a degraded QPU. Only a
    # task timeout (AQTTaskTimeout) leaves a failed health check record; the exit code is then EXIT_AQT_TASK_TIMEOUT.
    try:
        path_hc = submit_health_check(
            run_folder    = RUN_FOLDER,
            backend       = real_backend,
            nshots        = ACCOMPANYING_HEALTH_CHECK_NSHOTS,
            service_name  = SERVICE_NAME,
            backend_name  = BACKEND_NAME,
            reps          = 1,
            accompanying  = True,
            chunks        = todo_chunks,
        )
    except Exception as e:
        print(f"  ❌ PRE health check could not be executed on {BACKEND_NAME} (after the built-in connection "
              f"retries and the per-circuit time limit): {e} — QPU health unknown, no chunks submitted. Rerun to resume.", file=sys.stderr)
        sys.exit(EXIT_AQT_TASK_TIMEOUT if isinstance(e, AQTTaskTimeout) else EXIT_HC_NOT_SUBMITTED)
    hc_pre_record = load_health_checks(path_hc)[-1]
    p011_pre     = hc_pre_record["P011"]
    if hc_pre_record["verdict"] != "functional":
        print(f"  ❌ QPU degraded — check #{hc_pre_record['number']}, P(011)={p011_pre:.3f} — aborting!", file=sys.stderr)
        sys.exit(EXIT_HC_DEGRADED)
    print(f"  ✅ QPU functional — check #{hc_pre_record['number']}, P(011)={p011_pre:.3f} — proceeding.")
else:
    print("  All chunks already complete — skipping health checks and submission.")

durations   = deque(maxlen=ETA_WINDOW)   # iter_seconds of the last ETA_WINDOW real chunks
wall_sum    = 0.0                         # sum of iter_seconds over all real chunks this run
qpu_s_total = 0.0                         # sum of the modelled QPU time (timing.qpu_s)
qpu_known   = 0                           # chunks with a numeric estimate
n_real      = 0                           # chunks actually submitted this run
submitted_chunks = []                     # their chunk numbers, in submission order
t_run_start = time.perf_counter()

# Time limit: PCSS_Quantum_UI/run_aqt_loop.sbatch and run_aqt_loop_with_retries.sbatch set AQT_PAUSE_AFTER_HOURS. Once it has passed,
# the loop exits with PAUSE_EXIT_CODE before starting the next chunk (the running chunk and its
# POST health check finish first), and the sbatch file requeues its job. Unset: no limit.
PAUSE_AFTER_HOURS = pause_after_hours()

# ── Submission loop — one new chunk per iteration ────────────────────────────
for CHUNKS_TO_SUBMIT in range(CHUNK_START, CHUNK_STOP + 1):
    t0            = time.perf_counter()   # whole-iteration clock: submit + health check
    nc            = CHUNKS_TO_SUBMIT - 1
    record_path   = job_records_dir / f"job_record_chunk_{nc}.json"
    if nc not in todo_chunks:
        print(f"  [Chunk {nc}] already complete — ETA unchanged ({CHUNKS_TO_SUBMIT}/{CHUNK_STOP})")
        continue
    stop_if_requested(f"chunk {nc}")
    if pause_due(T_SCRIPT_START, PAUSE_AFTER_HOURS):
        print(f"\n  ⏸  Time limit ({PAUSE_AFTER_HOURS:.2f} h) reached — pausing before chunk {nc} "
              f"(exit code {PAUSE_EXIT_CODE}).", flush=True)
        sys.exit(PAUSE_EXIT_CODE)
    mtime_before  = record_path.stat().st_mtime if record_path.exists() else None   # a failed chunk's record is rewritten

    qbcs._status = "built"   # submit() set it to "submitted" last pass; its entry assert needs "built"

    try:
        run_folder_path = qbcs.submit(
            run_folder        = RUN_FOLDER,
            lang              = "qiskit",
            service           = SERVICE_NAME,
            backend           = real_backend,
            layout_connection = "linear_nearest_neighbor" if LNN else "all_to_all",
            chunks            = [nc],
        )
    except AQTTaskTimeout as e:
        # submit_chunk_aqt has already saved the attempt as a failed record (job_records/failed_attempts/ once rerun)
        print(f"\n  ⚠️  [Chunk {nc}] {BACKEND_NAME} did not execute a circuit within {AQT_TASK_TIMEOUT_S // 60} min\n"
              f"      {e}\n"
              f"      Stopping with exit code {EXIT_AQT_TASK_TIMEOUT}; finished chunks are kept, a rerun resumes at chunk {nc}.",
              flush=True)
        sys.exit(EXIT_AQT_TASK_TIMEOUT)

    if not record_path.exists() or record_path.stat().st_mtime == mtime_before:
        reason = "no record written"
        print(f"  [Chunk {nc}] {reason} — ETA unchanged ({CHUNKS_TO_SUBMIT}/{CHUNK_STOP})")
        continue

    rec     = json.loads(record_path.read_text(encoding="utf-8"))
    wall_s  = float(rec.get("wall_seconds", 0.0))          # chunk backend.run+result time (info column)
    qpu_raw = (rec.get("timing") or {}).get("qpu_s")          # AQT QPU-time model, as in the summary
    qpu_s   = float(qpu_raw) if isinstance(qpu_raw, (int, float)) and qpu_raw >= 0 else None

    # ── Health check POST (after every chunk) ────────────────────────────────
    p011_post  = None
    status_post = "skipped"
    if not all_done:
        # Relates to the chunk just submitted. Same "no record on failure" acceptance and the same
        # distinct exit code as the PRE check above.
        try:
            submit_health_check(
                run_folder       = RUN_FOLDER,
                backend          = real_backend,
                nshots           = ACCOMPANYING_HEALTH_CHECK_NSHOTS,
                service_name     = SERVICE_NAME,
                backend_name     = BACKEND_NAME,
                reps             = 1,
                accompanying     = True,
                after_submission = True,
                chunks           = [nc],
            )
        except Exception as e:
            print(f"  ❌ POST health check after chunk {nc} could not be executed (after the built-in "
                  f"connection retries and the per-circuit time limit): {e} — chunk {nc} was submitted and saved; stopping before the "
                  f"next chunk since QPU health is now unknown.", file=sys.stderr)
            sys.exit(EXIT_AQT_TASK_TIMEOUT if isinstance(e, AQTTaskTimeout) else EXIT_HC_NOT_SUBMITTED)
        hc_post_record = load_health_checks(run_folder_path)[-1]
        p011_post     = hc_post_record["P011"]
        status_post    = hc_post_record["verdict"]
        if status_post == "functional":
            print(f"  ✅ QPU functional — check #{hc_post_record['number']} (after chunk {nc}), P(011)={p011_post:.3f}")
        else:
            print(f"  ❌ QPU degraded — check #{hc_post_record['number']} (after chunk {nc}), P(011)={p011_post:.3f} — stopping.", file=sys.stderr)
            sys.exit(EXIT_HC_DEGRADED)
    iter_seconds = time.perf_counter() - t0   # whole iteration: submit + health check

    durations.append(iter_seconds)
    wall_sum += iter_seconds
    n_real   += 1
    submitted_chunks.append(nc)
    if qpu_s is not None:
        qpu_s_total += qpu_s
        qpu_known   += 1

    remaining  = CHUNK_STOP - CHUNKS_TO_SUBMIT
    eta_recent = mean(durations) * remaining        # last ETA_WINDOW chunk(s)
    eta_avg    = (wall_sum / n_real) * remaining     # mean over all chunks this run
    qpu_proj   = (qpu_s_total / qpu_known) * CHUNK_STOP if qpu_known else None

    now         = datetime.datetime.now().astimezone()
    finish_last = now + datetime.timedelta(seconds=eta_recent)
    finish_avg  = now + datetime.timedelta(seconds=eta_avg)

    qpu_txt   = f"{qpu_s:.2f}s" if qpu_s is not None else "n/a"
    proj_txt  = str(datetime.timedelta(seconds=round(qpu_proj))) if qpu_proj is not None else "n/a"
    eta_r_txt = str(datetime.timedelta(seconds=round(eta_recent)))
    eta_a_txt = str(datetime.timedelta(seconds=round(eta_avg)))
    print(
        f"\n  [{now.isoformat(timespec='seconds')}]  Chunk {nc}  |  done {CHUNKS_TO_SUBMIT}/{CHUNK_STOP}\n"
        f"      wall {wall_s:.1f}s   iter {iter_seconds:.1f}s   QPU {qpu_txt}\n"
        f"      ETA(last {ETA_WINDOW}) {eta_r_txt:>14}  📅  finish ~ {finish_last.strftime('%Y-%m-%d %H:%M %Z')}\n"
        f"      ETA(avg)    {eta_a_txt:>14}  📅  finish ~ {finish_avg.strftime('%Y-%m-%d %H:%M %Z')}\n"
        f"      QPU-budget proj {proj_txt}",
        flush=True,
    )
# ── Analyze ─────────────────────────────────────────────────────────────────
qbcs._use_circuit_LNN = LNN
qbcs._status          = "submitted"
qbcs.merge_all_chunks()
if qbcs.CONFIG["SPAM_correction"]:
    compute_and_save_spam_confusion_matrix(config=qbcs.CONFIG)
qbcs.analyze_results()

print(f"\n✅ Analysis complete.")
print(f"   Results: {rel_path(run_folder_path)}/processed_results/run_data.json")
if DO_SPAM:
    print(f"   Results (SPAM): {rel_path(run_folder_path)}/processed_results/run_data_spam.json")

if take_stop_file(run_folder_path):   # touched during the last chunk: nothing is left to stop, and a stale file would stop the next job of this run
    print("  STOP file removed: the loop had already reached its end.")

# ── Save log ────────────────────────────────────────────────────────────────
total_wall = time.perf_counter() - t_run_start
log_path   = append_submit_log(
    run_folder_path, backend=BACKEND_NAME, chunks=submitted_chunks, n_chunks=qbcs.n_chunks, nshots=NSHOTS,
    layout=layout_summary(LAYOUT, run_qubits, CAL_SOURCE), run_qubits=run_qubits,
    extra_lines=[f"Chunk range:      {CHUNK_START}-{CHUNK_STOP}",
                 *([health_check_log_line("PRE ", hc_pre_record)] if hc_pre_record is not None else []),
                 *([health_check_log_line("POST", hc_post_record)] if hc_post_record is not None else []),
                 f"Total wallclock:  {datetime.timedelta(seconds=round(total_wall))}",
                 f"QPU-time estimate (this run): {datetime.timedelta(seconds=round(qpu_s_total))}"
                 f"  [{qpu_known}/{n_real} chunks with a known estimate]",
                 *([f"Mean iter/chunk:  {wall_sum / n_real:.1f}s  (submit + health check)"] if n_real else [])])
print(f"   Log saved → {rel_path(log_path)}")
