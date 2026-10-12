"""
benchmark_coherent_noise_all_to_all.py
Local-simulator benchmark with coherent CNOT noise, all-to-all topology. Same as
benchmark_noiseless_all_to_all.py, but the circuits carry coherent Pauli noise compiled in
by the RC plugins (INFIDELITY_TOTAL per CNOT, RATIO_CT crosstalk share). The noise is
part of the circuits, so they get their own cache:
datafiles/circuits/<topology>_Nt…_pauli_I5.0e-03_rct5.0e-01/ (common.get_circuit_folder()).
A cache holds the samples of its first build at most: build the largest sample size first.

Usage (see source/core/runner_support/benchmark_args.py; the benchmark is required — without arguments, or with --help, the usage is printed):
  python benchmark_coherent_noise_all_to_all.py mini                   mini benchmark: 8 samples, Nt = 1, 3, 200 shots
  python benchmark_coherent_noise_all_to_all.py full                   full benchmark: 480 samples, Nt = 1…19, 10 shots
  python benchmark_coherent_noise_all_to_all.py mini nshots_per_circuit=N
  python benchmark_coherent_noise_all_to_all.py nshots_per_circuit=N n_samples=M n_depths=K   (or nt_list=1,5,9)
The run folder is <benchmark>_sim-coherent_local-aer_lnn-False_service-local_spam-True_layout-none_<YYYY-MM-DD>_#<N> (source/core/records/run_names.py).
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))   # datafiles/ paths are relative to the project root
from source.core.records.run_names import local_run_name, dated_run_folder
from source.core.runner_support.benchmark_args import make_usage, usage_if_requested   # light (numpy only): the usage is printed before anything else loads
usage_if_requested(sys.argv[1:], make_usage(__file__, "Run the BACQ benchmark on the local simulator with coherent CNOT noise compiled into the circuits (all-to-all topology). The noisy circuits get their own cache: build the largest sample size first.", backend=False))   # no arguments or --help: print the usage and do nothing

from source.core.runner_support.benchmark_args import parse_benchmark_args
from source.core.runner_support.run_local_simulation import run_local_simulation

BENCH      = parse_benchmark_args(sys.argv[1:], layout_args=False)
RUN_FOLDER = dated_run_folder(local_run_name(BENCH.tag, "sim-coherent", lnn=False))
BACKEND    = "AerSimulator (coherent CNOT noise in the circuits)"

INFIDELITY_TOTAL = 0.005    # CNOT infidelity from coherent noise
RATIO_CT         = 0.5      # share of it from crosstalk onto the neighbouring qubit
NOISE_TYPE       = "pauli"

def sampler_factory():
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
    return AerSamplerV2()

run_local_simulation(run_folder=RUN_FOLDER, bench=BENCH, lnn=False, backend_label=BACKEND, do_spam=True, sampler_factory=sampler_factory,
                     extra_qbcs_kwargs=dict(infidelity_total=INFIDELITY_TOTAL, ratio_ct=RATIO_CT, noise_type=NOISE_TYPE),
                     extra_config=[("Coherent noise", f"{NOISE_TYPE}, infidelity {INFIDELITY_TOTAL}, crosstalk ratio {RATIO_CT}")])
