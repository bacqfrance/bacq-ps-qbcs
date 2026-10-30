#!/bin/bash
# Run health_check_real_aqt.py on a worker node, in venvs/aqt.venv: one health check on
# PIAST-Q, submitted and collected in one go, into
# datafiles/health-check+reps-1+nshots-100_real-data_aqt-piast-q_lnn-none_service-pcss_spam-none_layout-none_<YYYY-MM-DD>_#<N>/ (a new dated folder per run, unless the latest is unfinished).
# Submit from the project root:  sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_health_check_aqt.sbatch backend=PIAST-Q   (backend= is required; without it the job
# prints health_check_real_aqt.py's usage into its log and ends with exit code 2)
# The Slurm log goes to the submission directory (the project root): health_check_aqt_<jobid>.out
#SBATCH --job-name=health_check_aqt
#SBATCH --output=%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

PROJECT=$HOME/Quantum_Benchmarking_project
export PYTHONIOENCODING=utf-8
cd "$PROJECT"

# ── Output helpers: a title between two rules, then aligned "Label:  value" rows (nothing wider than ~90 characters) ──
RULE='────────────────────────────────────────────────────────────'
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
row()   { printf '  %-12s %s\n' "$1:" "$2"; }
describe_exit() {                                      # what an exit code of health_check_real_aqt.py means
    case $1 in
        0)  echo "done" ;;
        2)  echo "unusable arguments" ;;
        10) echo "a health check found the QPU degraded" ;;
        12) echo "the health check could not be submitted (QPU health unknown)" ;;
        13) echo "PIAST-Q did not execute a circuit within 30 min" ;;
        *)  echo "error (see the output above)" ;;
    esac
}

echo "$RULE"; echo " AQT health check"; echo "$RULE"
row Job "$SLURM_JOB_ID"
row Node "$(hostname)"
row Started "$(stamp)"
row Arguments "$*"
echo
"$PROJECT/venvs/aqt.venv/bin/python" -u \
    source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/health_check_real_aqt.py "$@"
rc=$?
echo; row Ended "$(stamp)"; row "Exit code" "$rc — $(describe_exit $rc)"
exit $rc
