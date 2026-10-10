#!/bin/bash
# Run qbcs_real_simulate_aqt.py on a worker node, in venvs/aqt.venv: the benchmark on the
# local Aer simulator with PIAST-Q's calibration (calibration_file=..., default: the newest file in
# calibration_input_files/PIAST-Q/), no QPU time. Needs the circuit cache for the chosen size
# (for full: run_noiseless_all_to_all.sbatch first).
# Submit from the project root:
#   sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_simulate_aqt.sbatch backend=PIAST-Q mini layout=random    mini benchmark; backend=, the benchmark and layout= are required
#   sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_simulate_aqt.sbatch backend=PIAST-Q full layout=optimized calibration_file=calibration_input_files/PIAST-Q/<file>.json
#   (layout= says which real-submission behaviour to mimic: PIAST-Q reports no calibration, so a submission without a file is random)
# (the arguments go to qbcs_real_simulate_aqt.py; without them the job prints that script's usage into its log and ends with exit code 2)
# The Slurm log goes to the submission directory (the project root): simulate_aqt_<jobid>.out
#SBATCH --job-name=simulate_aqt
#SBATCH --output=%x_%j.out
#SBATCH --time=24:00:00
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
describe_exit() {                                      # what an exit code of the loop script means
    case $1 in
        0)  echo "done" ;;
        2)  echo "unusable arguments" ;;
        10) echo "a health check found the QPU degraded" ;;
        12) echo "a health check could not be submitted (QPU health unknown)" ;;
        13) echo "PIAST-Q did not execute a circuit within 30 min" ;;
        *)  echo "error (see the output above)" ;;
    esac
}

echo "$RULE"; echo " AQT simulation"; echo "$RULE"
row Job "$SLURM_JOB_ID"
row Node "$(hostname)"
row Started "$(stamp)"
row Arguments "$*"
echo
"$PROJECT/venvs/aqt.venv/bin/python" -u \
    source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/qbcs_real_simulate_aqt.py "$@"
rc=$?
echo; row Ended "$(stamp)"; row "Exit code" "$rc — $(describe_exit $rc)"
exit $rc
