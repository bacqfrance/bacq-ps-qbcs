#!/bin/bash
# Run the AQT PIAST-Q benchmark on a worker node like run_aqt_loop.sbatch (same arguments, same requeue every
# PAUSE_AFTER_HOURS), but a loop script that ends in any other way than done (exit code 0), planned pause (3),
# unusable arguments (2) or a STOP file (4, below) is run again: 1 h after the failure, then every hour (a worker node cannot sbatch, so the
# job sleeps and reruns the loop script itself). Typical failures: 13 = PIAST-Q did not execute a circuit within
# 30 min (down, in maintenance or reserved), 12 / 10 = a health check could not be executed / found the QPU degraded,
# 1 = any other error. A rerun resumes at the first chunk without a record.
# The week counts from the first failure and restarts after any progress: a run that completed a chunk, or ended with
# 0 or 3. Its start is kept in datafiles/<run folder>/retry_state, which survives the requeues. After RETRY_LIMIT_DAYS (7)
# of failures without progress the job prints a "GIVING UP" block into its log and output.log and ends with exit code 14.
# Stopping: touch datafiles/<run folder>/STOP (in the project root on the cluster). The loop script finishes its current chunk and
# POST health check, deletes the file and ends with exit code 4, and this job ends too, without retrying or requeueing.
# No e-mail: PCSS worker and login nodes cannot send any (tested 2026-10), so that log block and exit code 14
# (sacct -j <jobid>) are the notification.
# Submit from the project root (the job changes into $PROJECT itself, whatever the submit folder is):
#   sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_aqt_loop_with_retries.sbatch backend=PIAST-Q mini    mini benchmark; backend= and the benchmark are required
#   sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_aqt_loop_with_retries.sbatch backend=PIAST-Q full    full benchmark
#   Layout: PIAST-Q reports no calibration, so without layout= / calibration_file= the run uses a RANDOM layout. Add
#   layout=optimized calibration_file=calibration_input_files/PIAST-Q/<file>.json to optimize it, or layout=lowest / layout=15,14,12.
#   Add --force-new-run to start a new run folder even if the latest run of these settings is unfinished (the whole job,
#   with its retries and requeues, then works in that one folder).
# (the arguments go to qbcs_real_submit_aqt_loop.py; without them the job prints that script's usage into its log and ends with exit code 2)
# Retries count towards the PAUSE_AFTER_HOURS of the job, and the job requeues itself once that has passed, also while
# retrying (the loop script gets only the hours that are left). To test with shorter times, submit a copy with smaller
# PAUSE_AFTER_HOURS / RETRY_WAIT_S / RETRY_LIMIT_DAYS (Slurm keeps the submitted copy, also across requeues).
# The Slurm log goes to the submission directory (the project root): aqt_loop_with_retries_<jobid>.out (appended across requeues).
# The loop script's output is also appended to datafiles/<run folder>/output.log.
# scancel on the job also stops the chain, but loses the running chunk.
# PROJECT below is the clone's location on the cluster: change it if the clone is elsewhere.
#SBATCH --job-name=aqt_loop_with_retries
#SBATCH --output=%x_%j.out
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

PROJECT=$HOME/Quantum_Benchmarking_project
export PYTHONIOENCODING=utf-8
PAUSE_AFTER_HOURS=10     # pause at the first chunk boundary after this many hours, then requeue
RETRY_WAIT_S=3600        # wait before each rerun after a failure
RETRY_LIMIT_DAYS=7       # give up after this many days of failures without progress
EXIT_GAVE_UP=14
cd "$PROJECT"

# ── Output helpers: a title between two rules, then aligned "Label:  value" rows (nothing wider than ~90 characters) ──
RULE='────────────────────────────────────────────────────────────'
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
dur()   { if [ $1 -lt 120 ]; then echo "$1 s"; elif [ $1 -lt 7200 ]; then echo "$(( $1 / 60 )) min"; else echo "$(( $1 / 3600 )) h"; fi; }   # seconds as "45 s" / "20 min" / "3 h"
row()   { printf '  %-12s %s\n' "$1:" "$2"; }
describe_exit() {                                      # what an exit code of the loop script means
    case $1 in
        0)  echo "done" ;;
        2)  echo "unusable arguments" ;;
        3)  echo "time limit reached at a chunk boundary" ;;
        4)  echo "stopped by the STOP file" ;;
        10) echo "a health check found the QPU degraded" ;;
        12) echo "a health check could not be submitted (QPU health unknown)" ;;
        13) echo "PIAST-Q did not execute a circuit within 30 min" ;;
        14) echo "gave up after repeated failures" ;;
        *)  echo "error (see the output above)" ;;
    esac
}

PYTHON=$PROJECT/venvs/aqt.venv/bin/python
AQT_DIR=source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q

# No arguments or --help: only the loop script's usage (exit code 2 without arguments, 0 for --help).
# aqt_run_folder.py below would not do: its stdout is captured as the run folder's name.
case " $* " in
    "  "|*" --help "*|*" -h "*)
        "$PYTHON" "$AQT_DIR/qbcs_real_submit_aqt_loop.py" "$@"
        exit $? ;;
esac

# Run folder for these arguments, read from the loop script's Config block, once per job: every run of the loop script
# below (and every requeue) works in it, so the loop script gets it as run_folder= and never --force-new-run.
# A requeued job (restart count > 0) starts with the same arguments: without --force-new-run the default rule picks
# the latest unfinished folder, which is the one this job made at its first start (mkdir -p below; a folder without
# run_info.json counts as unfinished).
LOOP_ARGS=()
for a in "$@"; do [ "$a" != "--force-new-run" ] && LOOP_ARGS+=("$a"); done
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then RESOLVE_ARGS=("${LOOP_ARGS[@]}"); else RESOLVE_ARGS=("$@"); fi
RUN_FOLDER=$("$PYTHON" "$AQT_DIR/PCSS_Quantum_UI/aqt_run_folder.py" "${RESOLVE_ARGS[@]}") || exit 1
LOOP_ARGS+=("run_folder=$RUN_FOLDER")
OUTPUT_LOG=datafiles/$RUN_FOLDER/output.log
STATE_FILE=datafiles/$RUN_FOLDER/retry_state       # first_failure=<epoch s> and failures=<n> of the current series
mkdir -p "datafiles/$RUN_FOLDER"

log() { echo "$*" | tee -a "$OUTPUT_LOG"; }
logrow() { row "$@" | tee -a "$OUTPUT_LOG"; }
# chunks with a complete record (a failed attempt also writes a record, so counting files would not do); this run folder only
chunks_done() { grep -l '"status": "complete"' "datafiles/$RUN_FOLDER/job_records"/job_record_chunk_*.json 2>/dev/null | wc -l; }
state_get() { sed -n "s/^$1=//p" "$STATE_FILE" 2>/dev/null | head -1; }

# A freshly submitted job starts a new series: a retry_state left by an earlier job (e.g. one that gave up)
# would otherwise make the first failure of this one look a week old. A requeued job (restart count > 0) keeps it.
[ "${SLURM_RESTART_COUNT:-0}" -eq 0 ] && rm -f "$STATE_FILE"
JOB_START=$(date +%s)
PAUSE_S=$(awk -v h="$PAUSE_AFTER_HOURS" 'BEGIN{printf "%d", h*3600}')      # PAUSE_AFTER_HOURS may be fractional (tests)
log "$RULE"; log " AQT loop with retries"; log "$RULE"
logrow Job "$SLURM_JOB_ID   (restart ${SLURM_RESTART_COUNT:-0})"
logrow Node "$(hostname)"
logrow Started "$(stamp)"
logrow "Run folder" "datafiles/$RUN_FOLDER"
logrow Arguments "$*"
logrow "Pause after" "$PAUSE_AFTER_HOURS h"
logrow Retry "every $(dur $RETRY_WAIT_S) after a failure, gives up after $RETRY_LIMIT_DAYS days without progress"

requeue() {                              # planned rescheduling: restarts this job with the same ID and arguments
    log "  Requeueing job $SLURM_JOB_ID."
    scontrol requeue "$SLURM_JOB_ID" || exit 1
    sleep 600                            # Slurm stops the job within seconds; exiting first could end it as done
    echo "  ❌ Requeue accepted but the job was not stopped within 10 min." | tee -a "$OUTPUT_LOG" >&2
    exit 1
}

while true; do
    elapsed=$(( $(date +%s) - JOB_START ))
    left_s=$(( PAUSE_S - elapsed ))
    if [ $left_s -le 0 ]; then
        log "  ⏸  Pause time ($PAUSE_AFTER_HOURS h) used up while retrying."
        requeue
    fi
    done_before=$(chunks_done)
    # The loop script counts its time limit from its own start: give it the hours that are left of this job's.
    AQT_PAUSE_AFTER_HOURS=$(awk -v s=$left_s 'BEGIN{printf "%.6f", s/3600}') \
        "$PYTHON" -u "$AQT_DIR/qbcs_real_submit_aqt_loop.py" "${LOOP_ARGS[@]}" 2>&1 | tee -a "$OUTPUT_LOG"
    rc=${PIPESTATUS[0]}                  # the loop script's exit code, not tee's
    log ""; logrow Ended "$(stamp)"; logrow "Exit code" "$rc — $(describe_exit $rc)"

    case $rc in
        0) rm -f "$STATE_FILE"; exit 0 ;;                   # all chunks done
        3) rm -f "$STATE_FILE"; requeue ;;                  # time limit reached at a chunk boundary
        2) exit 2 ;;                                        # unusable arguments: a rerun cannot help
        4) rm -f "$STATE_FILE"; log "  Stopped by the STOP file (datafiles/$RUN_FOLDER/STOP): not retrying, not requeueing."; exit 4 ;;
    esac

    # A failure. Progress (a chunk with a complete record) starts the week again.
    now=$(date +%s)
    if [ "$(chunks_done)" -gt "$done_before" ] || [ -z "$(state_get first_failure)" ]; then
        echo "first_failure=$now" > "$STATE_FILE"
        echo "failures=0" >> "$STATE_FILE"
    fi
    first=$(state_get first_failure)
    failures=$(( $(state_get failures) + 1 ))
    printf 'first_failure=%s\nfailures=%s\n' "$first" "$failures" > "$STATE_FILE"

    if [ $(( now - first )) -ge $(( RETRY_LIMIT_DAYS * 86400 )) ]; then
        log ""; log "$RULE"; log " GIVING UP"; log "$RULE"
        logrow Job "$SLURM_JOB_ID"
        logrow "Run folder" "datafiles/$RUN_FOLDER"
        logrow Time "$(stamp)"
        logrow Failures "$failures since $(date -d @"$first" '+%Y-%m-%d %H:%M'), no progress ($RETRY_LIMIT_DAYS days of retries)"
        logrow "Last exit" "$rc — $(describe_exit $rc)"
        logrow "Next step" "check PIAST-Q's availability with PCSS, then submit this job again"
        exit $EXIT_GAVE_UP
    fi
    wait_s=$(( PAUSE_S - ( $(date +%s) - JOB_START ) ))
    [ $wait_s -gt $RETRY_WAIT_S ] && wait_s=$RETRY_WAIT_S
    log ""
    log "  ⚠️  Loop script failed — exit code $rc"
    log "      $(describe_exit $rc)"
    log "      Failure:      $failures since $(date -d @"$first" '+%Y-%m-%d %H:%M')"
    log "      Rerun at:     $(date -d "+${wait_s} seconds" '+%Y-%m-%d %H:%M')  (in $(dur $wait_s))"
    log "      Gives up on:  $(date -d @$(( first + RETRY_LIMIT_DAYS * 86400 )) '+%Y-%m-%d %H:%M')"
    [ $wait_s -gt 0 ] && sleep $wait_s
done
