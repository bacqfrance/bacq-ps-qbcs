#!/bin/bash
# Run the project's test suite (source/tests/) on a worker node, in venvs/local_sim.venv (test_12 also needs venvs/aqt.venv).
# Submit from the project root:  sbatch source/benchmark_scripts/real_qpu_scripts/AQT_PIAST-Q/PCSS_Quantum_UI/run_pytest.sbatch
# The Slurm log goes to the submission directory (the project root): pytest_qbench_<jobid>.out
# The tests build their caches only in datafiles/tests/circuits/, so it is safe while a benchmark runs.
#SBATCH --job-name=pytest_qbench
#SBATCH --output=%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

PROJECT=$HOME/Quantum_Benchmarking_project
export PYTHONIOENCODING=utf-8
cd "$PROJECT"

echo "job: $SLURM_JOB_ID  node: $(hostname)  started: $(date '+%Y-%m-%d %H:%M:%S')"
"$PROJECT/venvs/local_sim.venv/bin/python" -m pytest source/tests/ -ra
rc=$?
echo "ended: $(date '+%Y-%m-%d %H:%M:%S')  exit code: $rc"
exit $rc
