"""
benchmark_noiseless_all_to_all.py
Noiseless local-simulator benchmark, all-to-all topology. Builds the all-to-all
circuit cache (QPY version 13) that the AQT PIAST-Q, IQM VLQ and IQM Resonance sirius scripts use, then
simulates it. Full benchmark on a laptop-class machine: ~60 min to build the circuits
the first time, then ~15 min to simulate (USER_MANUAL §4.3); mini: ~10 s.
run_local_simulation() prints the time of each stage, so you get a number for your own machine.

Usage (see source/core/runner_support/benchmark_args.py; the benchmark is required — without arguments, or with --help, the usage is printed):
  python benchmark_noiseless_all_to_all.py mini                   mini benchmark: 8 samples, Nt = 1, 3, 200 shots
  python benchmark_noiseless_all_to_all.py full                   full benchmark: 480 samples, Nt = 1…19, 10 shots
  python benchmark_noiseless_all_to_all.py mini nshots_per_circuit=N
  python benchmark_noiseless_all_to_all.py nshots_per_circuit=N n_samples=M n_depths=K   (or nt_list=1,5,9)
The run folder is <benchmark>_sim-noiseless_local-aer_lnn-False_service-local_spam-True_layout-none_<YYYY-MM-DD>_#<N> (source/core/records/run_names.py).
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))   # datafiles/ paths are relative to the project root
from source.core.records.run_names import local_run_name, dated_run_folder
from source.core.runner_support.benchmark_args import make_usage, usage_if_requested   # light (numpy only): the usage is printed before anything else loads
usage_if_requested(sys.argv[1:], make_usage(__file__, "Run the BACQ benchmark on the local noiseless simulator with all-to-all circuits. Builds the all-to-all circuit cache that the AQT PIAST-Q, IQM VLQ and IQM Resonance sirius scripts use.", backend=False))   # no arguments or --help: print the usage and do nothing

from source.core.runner_support.benchmark_args import parse_benchmark_args
from source.core.runner_support.run_local_simulation import run_local_simulation

BENCH      = parse_benchmark_args(sys.argv[1:], layout_args=False)
RUN_FOLDER = dated_run_folder(local_run_name(BENCH.tag, "sim-noiseless", lnn=False))
BACKEND    = "AerSimulator (noiseless)"

def sampler_factory():
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
    return AerSamplerV2()

run_local_simulation(run_folder=RUN_FOLDER, bench=BENCH, lnn=False, backend_label=BACKEND, do_spam=True, sampler_factory=sampler_factory)
