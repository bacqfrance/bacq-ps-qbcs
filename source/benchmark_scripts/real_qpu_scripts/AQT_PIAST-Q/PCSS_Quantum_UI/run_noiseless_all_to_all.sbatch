#!/bin/bash
# Run the noiseless all-to-all local simulation on a worker node, in venvs/local_sim.venv.
# The arguments are passed through to benchmark_noiseless_all_to_all.py, so it builds (and simulates)
# whichever benchmark you name. It builds the circuit cache that the AQT runs use:
#   full -> datafiles/circuits/all_to_all_Nt1-3-5-7-9-11-13-15-17-19/
#   mini -> the mini circuit cache (Nt = 1, 3)
# Submit from the project root:  sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_noiseless_all_to_all.sbatch full   (or: mini)
# At least one argument is required, as the script has no default benchmark.
# The Slurm log goes to the submission directory (the project root): noiseless_a2a_<jobid>.out
#SBATCH --job-name=noiseless_a2a
#SBATCH --output=%x_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

if [ $# -eq 0 ]; then
    echo "usage: sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_noiseless_all_to_all.sbatch <benchmark arguments, e.g. full or mini>" >&2
    exit 2
fi

PROJECT=$HOME/Quantum_Benchmarking_project
export PYTHONIOENCODING=utf-8
cd "$PROJECT"

echo "job: $SLURM_JOB_ID  node: $(hostname)  started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "arguments: $*"
"$PROJECT/venvs/local_sim.venv/bin/python" \
    source/benchmark_scripts/local_simulation_scripts/benchmark_noiseless_all_to_all.py "$@"
rc=$?
echo "ended: $(date '+%Y-%m-%d %H:%M:%S')  exit code: $rc"
exit $rc
