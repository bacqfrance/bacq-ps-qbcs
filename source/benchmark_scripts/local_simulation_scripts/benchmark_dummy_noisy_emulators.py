"""
benchmark_dummy_noisy_emulators.py
Local-simulator benchmarks of the dummy noisy emulators defined in definitions_dummy_noisy_emulators.json (beside this
script): QPUs the benchmark was not run on, each simulated with coherent CNOT noise at its own infidelity
(INFIDELITY_TOTAL = 1 - Fidelity/100), crosstalk ratio (r_ct), noise type and topology (lnn), like
benchmark_coherent_noise_lnn.py / benchmark_coherent_noise_all_to_all.py (same seed, rng_seed = 0). One run per dummy;
a dummy whose latest run is finished is skipped, an unfinished one is resumed. Each noise setting has its own circuit
cache (datafiles/circuits/<topology>_Nt…_<noise>_I…_rct…/): build the largest sample size first.
Each run folder gets dummy_estimates.json, the dummy's pricing and timing inputs, from which the analysis models the
run's QPU time and cost (source/core/analysis/dummy_estimates.py).

Usage (see source/core/runner_support/benchmark_args.py; the benchmark is required — without arguments, or with --help, the usage is printed):
  python3 benchmark_dummy_noisy_emulators.py mini                   mini benchmark: 8 samples, Nt = 1, 3, 200 shots
  python3 benchmark_dummy_noisy_emulators.py full                   full benchmark: 480 samples, Nt = 1…19, 10 shots
  python3 benchmark_dummy_noisy_emulators.py mini nshots_per_circuit=N
  python3 benchmark_dummy_noisy_emulators.py nshots_per_circuit=N n_samples=M n_depths=K   (or nt_list=1,5,9)
The run folders are <benchmark>_sim-dummy_<NAME>_lnn-<bool>_service-local_spam-True_layout-none_<YYYY-MM-DD>_#<N> (source/core/records/run_names.py),
NAME being the dummy's key in the JSON (ibm-marrakesh, ionq-forte, ...).
Exit code 2 if a dummy failed (the others are still run).
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))   # datafiles/ paths are relative to the project root
from source.core.records.run_names import local_run_name
from source.core.runner_support.benchmark_args import make_usage, usage_if_requested   # light (numpy only): the usage is printed before anything else loads
usage_if_requested(sys.argv[1:], make_usage(__file__, "Run the BACQ benchmark on the local simulator for each dummy noisy emulator of definitions_dummy_noisy_emulators.json: coherent CNOT noise at the dummy's infidelity, with the QPU time and cost modelled afterwards. Finished dummies are skipped. Each noise setting has its own circuit cache: build the largest sample size first.", backend=False))   # no arguments or --help: print the usage and do nothing

from pathlib import Path
from source.core.runner_support.benchmark_args import parse_benchmark_args
from source.core.analysis.dummy_estimates import (load_definitions, estimates_record, write_dummy_estimates, dummy_run_folder,
                                                  DEFINITIONS_PATH)

BENCH = parse_benchmark_args(sys.argv[1:], layout_args=False)


def sampler_factory():
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
    return AerSamplerV2()


def run_dummy(name, entry):
    """Simulate (or resume) the dummy `name`; False if it was skipped as finished."""
    from source.core.runner_support.run_local_simulation import run_local_simulation
    est = estimates_record(name, entry)   # refuses a null term before anything is built
    undated = local_run_name(BENCH.tag, "sim-dummy", est["lnn"], device=name)
    run_folder, finished = dummy_run_folder(undated)
    if finished:
        print(f"⏭  {name}: {run_folder} is finished — skipping")
        return False
    print(f"\n=== Dummy {name}: {run_folder} ===")
    write_dummy_estimates(Path("datafiles") / run_folder, name, entry)   # before simulating; an existing file is kept
    run_local_simulation(run_folder=run_folder, bench=BENCH, lnn=est["lnn"], backend_label=f"AerSimulator (coherent CNOT noise, dummy {name})",
                         do_spam=True, sampler_factory=sampler_factory,
                         extra_qbcs_kwargs=dict(infidelity_total=est["infidelity_total"], ratio_ct=est["ratio_ct"], noise_type=est["noise_type"]),
                         extra_config=[("Coherent noise", f"{est['noise_type']}, infidelity {est['infidelity_total']:.6g}, crosstalk ratio {est['ratio_ct']}"),
                                       ("Dummy emulator", name)])
    return True


failed = []
for dummy_name, dummy_entry in load_definitions(DEFINITIONS_PATH).items():
    try:
        run_dummy(dummy_name, dummy_entry)
    except (Exception, SystemExit) as e:   # exit_with_error() exits: go on with the other dummies
        print(f"❌ {dummy_name}: failed — {type(e).__name__}: {e}", file=sys.stderr)
        failed.append(dummy_name)
if failed:
    print(f"❌ {len(failed)} dummy run(s) failed: {', '.join(failed)}", file=sys.stderr)
    sys.exit(2)
