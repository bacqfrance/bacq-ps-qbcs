#!/bin/bash
# Run the AQT PIAST-Q benchmark on a worker node, in venvs/aqt.venv, with
# qbcs_real_submit_aqt_loop.py. After PAUSE_AFTER_HOURS (set below) the loop
# script lets the current chunk finish and exits with code 3; this job then
# requeues itself (same job ID and arguments) and the restarted job resumes.
# Dropped connections are retried inside the loop script (qpu/aqt.py: _aqt_retry), and a circuit not executed within 30 min ends it with a warning and
# exit code 13 (run_aqt_loop_with_retries.sbatch, same folder, reruns it hourly). Any other error ends the job with exit code 1; a health check that still can't
# be executed ends it with exit code 12 (QPU health unknown); a health check that ran and found the QPU
# degraded ends it with exit code 10. This script only
# special-cases exit code 3 (pause/requeue) and passes any other code through.
# To stop the job cleanly: touch datafiles/<run folder>/STOP (in the project root on the cluster). The loop script finishes its current
# chunk and POST health check, deletes the file and ends with exit code 4 (Slurm shows FAILED 4:0) instead of starting the next chunk.
# Submit from the project root (the job changes into $PROJECT itself, whatever the submit folder is):
#   sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_aqt_loop.sbatch backend=PIAST-Q mini    mini benchmark; backend= and the benchmark are required
#   sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_aqt_loop.sbatch backend=PIAST-Q full    full benchmark
#   Layout: PIAST-Q reports no calibration, so without layout= / calibration_file= the run uses a RANDOM layout. Add
#   layout=optimized calibration_file=calibration_input_files/PIAST-Q/<file>.json to optimize it, or layout=lowest / layout=15,14,12.
#   Add --force-new-run to start a new run folder even if the latest run of these settings is unfinished (the whole job,
#   with its retries and requeues, then works in that one folder).
# (the arguments go to qbcs_real_submit_aqt_loop.py; without them the job prints that script's usage into its log and ends with exit code 2)
# To test with a shorter limit, submit a copy with a smaller PAUSE_AFTER_HOURS
# (Slurm keeps the submitted copy, also across requeues).
# The Slurm log goes to the submission directory (the project root): aqt_loop_<jobid>.out (appended across requeues).
# The loop script's output is also appended to datafiles/<run folder>/output.log.
# scancel on the job also stops the chain, but loses the running chunk.
# PROJECT below is the clone's location on the cluster: change it if the clone is elsewhere.
#SBATCH --job-name=aqt_loop
#SBATCH --output=%x_%j.out
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

PROJECT=$HOME/Quantum_Benchmarking_project
export PYTHONIOENCODING=utf-8
PAUSE_AFTER_HOURS=10     # pause at the first chunk boundary after this many hours, then requeue
cd "$PROJECT"

# ── Output helpers: a title between two rules, then aligned "Label:  value" rows (nothing wider than ~90 characters) ──
RULE='────────────────────────────────────────────────────────────'
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
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
mkdir -p "datafiles/$RUN_FOLDER"

{
    echo "$RULE"; echo " AQT loop"; echo "$RULE"
    row Job "$SLURM_JOB_ID   (restart ${SLURM_RESTART_COUNT:-0})"
    row Node "$(hostname)"
    row Started "$(stamp)"
    row "Run folder" "datafiles/$RUN_FOLDER"
    row Arguments "$*"
    row "Pause after" "$PAUSE_AFTER_HOURS h"
} | tee -a "$OUTPUT_LOG"
AQT_PAUSE_AFTER_HOURS=$PAUSE_AFTER_HOURS "$PYTHON" -u "$AQT_DIR/qbcs_real_submit_aqt_loop.py" "${LOOP_ARGS[@]}" 2>&1 \
    | tee -a "$OUTPUT_LOG"
rc=${PIPESTATUS[0]}                      # the loop script's exit code, not tee's
{ echo; row Ended "$(stamp)"; row "Exit code" "$rc — $(describe_exit $rc)"; } | tee -a "$OUTPUT_LOG"
if [ $rc -eq 3 ]; then                   # time limit reached at a chunk boundary
    echo "  Requeueing job $SLURM_JOB_ID." | tee -a "$OUTPUT_LOG"
    scontrol requeue "$SLURM_JOB_ID" || exit 1
    sleep 600                            # Slurm stops the job within seconds; exiting first could end it as done
    echo "  ❌ Requeue accepted but the job was not stopped within 10 min." | tee -a "$OUTPUT_LOG" >&2
    exit 1
fi
exit $rc
