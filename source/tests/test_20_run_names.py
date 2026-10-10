"""
test_20_run_names.py
====================
Unit tests (no simulation, no network) for source/core/records/run_names.py — the one place run-folder,
plot-folder and summary names are built and parsed:

  - every QPU script folder and every data type builds a name that parses back to the same fields
    (real runs, their simulations, standalone health checks, local simulations);
  - the <benchmark> field of the presets and of custom benchmarks (n_depths, an explicit nt_list),
    as source/core/runner_support/benchmark_args.py produces it;
  - subset analyses (+nsamples-N), the summary names (short in the run, full in analysis/summaries/)
    and the collective-plot folder names;
  - malformed names are refused, and an explicit layout has exactly three qubits.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import pytest

from source.core.runner_support import benchmark_args as BA
from source.core.records import run_names as RN


@pytest.mark.parametrize("qpu, backend, device, service", [
    ("IBM",           "ibm_fez",  "ibm-fez",      "ibm-cloud"),
    ("IQM_RESONANCE", "emerald",  "iqm-emerald",  "iqm-resonance"),
    ("IQM_VLQ",       "iqm_vlq",  "iqm-vlq",      "qaas-it4i"),
    ("IQM_EQE1",      "EQE1",     "iqm-eqe1",     "mqss-lrz"),
    ("AQT_PIAST-Q",   "PIAST-Q",  "aqt-piast-q",  "pcss"),
])
def test_qpu_names_round_trip(qpu, backend, device, service):
    real = RN.qpu_run_name(qpu, backend, "full", True, True, "layout-optimized")
    assert real == f"full_real-data_{device}_lnn-True_service-{service}_spam-True_layout-optimized"
    r = RN.parse_run_name(real)
    assert (r.device, r.service, r.data_type, r.is_simulation, str(r)) == (device, service, "real-data", False, real)

    sim = RN.parse_run_name(RN.qpu_run_name(qpu, backend, "mini", False, True, "layout-15-14-21", data_type="sim-qpu-noise"))
    assert (sim.device, sim.service, sim.layout, sim.is_simulation) == (device, "local", "15-14-21", True)

    prefix = RN.qpu_run_prefix(qpu, backend, "full", True, True)
    assert real.startswith(prefix + "_layout-")

    hc = RN.parse_run_name(RN.health_check_run_name(qpu, backend, 1, 100))
    assert (hc.benchmark, hc.lnn, hc.spam, hc.layout, hc.is_health_check) == ("health-check+reps-1+nshots-100", "none", "none", "none", True)


@pytest.mark.parametrize("data_type", [d for d in RN.DATA_TYPES if d != "real-data"])
def test_local_names(data_type):
    name = RN.local_run_name("mini", data_type, lnn=False)
    r = RN.parse_run_name(name)
    assert (r.data_type, r.device, r.service, r.layout, r.is_simulation) == (data_type, "local-aer", "local", "none", True)


def test_entry_data_type():
    for dtype in RN.DATA_TYPES:
        assert RN.entry_data_type(f"{dtype}_ibm-fez") == dtype
        assert RN.entry_data_type(f"{dtype}_ibm-fez_lnn-True_service-local_spam-True_layout-none") == dtype
    with pytest.raises(ValueError):
        RN.entry_data_type("mini_real-data_ibm-fez")


def test_every_data_type_has_a_name():
    for dtype in RN.DATA_TYPES:
        assert RN.parse_run_name(f"mini_{dtype}_ibm-fez_lnn-True_service-local_spam-True_layout-none").data_type == dtype


def test_figure_labels():
    """Every data type has a figure label; simulations are marked as such, sim-fake-fez is called fake-fez."""
    label = lambda name: RN.parse_run_name(name).figure_label
    assert label("mini_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-optimized") == "ibm-fez — real QPU"
    assert label("mini_sim-fake-fez_local-aer_lnn-True_service-local_spam-True_layout-none") == "fake-fez — simulation (FakeFez noise)"
    assert label("mini_sim-qpu-noise_aqt-piast-q_lnn-False_service-local_spam-True_layout-optimized") \
        == "aqt-piast-q — simulation (calibration noise)"
    for dtype in RN.DATA_TYPES:
        r = RN.parse_run_name(f"mini_{dtype}_ibm-fez_lnn-True_service-local_spam-True_layout-none")
        assert ("simulation" in r.figure_label) == r.is_simulation


def test_benchmark_field_from_the_arguments():
    assert BA.parse_benchmark_args(["full"]).tag == "full"
    assert BA.parse_benchmark_args(["mini", "nshots_per_circuit=100"]).tag == "mini+nshots-100"
    assert BA.parse_benchmark_args(["nshots_per_circuit=50", "n_samples=8", "n_depths=3"]).tag == \
        "custom+ndepths-3+nsamples-8+nshots-50"
    assert BA.parse_benchmark_args(["nshots_per_circuit=50", "n_samples=8", "nt_list=1,5,9"]).tag == \
        "custom+nt-1-5-9+nsamples-8+nshots-50"


def test_subsets_summaries_and_collective_folders():
    name = "full_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-optimized"
    assert str(RN.parse_run_name(name).with_samples(48)).startswith("full+nsamples-48_real-data_ibm-fez")
    custom = "custom+ndepths-3+nsamples-8+nshots-50_sim-noiseless_local-aer_lnn-True_service-local_spam-True_layout-none"
    assert RN.parse_run_name(custom).with_samples(2).benchmark == "custom+ndepths-3+nshots-50+nsamples-2"   # replaced, not doubled
    assert RN.run_summary_name() == "summary.json" and RN.run_summary_name(48) == "summary+nsamples-48.json"
    assert RN.subset_suffix() == "" and RN.subset_suffix(48) == "+nsamples-48"   # the suffix of run_data+nsamples-48.json and the other subset outputs
    assert RN.summary_name(name) == f"summary_{name}.json"
    assert RN.summary_name(name, 48) == f"summary_{RN.parse_run_name(name).with_samples(48)}.json"
    assert RN.collective_folder_name("mini", ["real-data", "sim-noiseless"]) == "mini_collective_real-data+sim-noiseless"
    assert RN.parse_run_name(f"old_runs_full/{name}").device == "ibm-fez"   # a path ending in a run name


@pytest.mark.parametrize("bad", [
    "bacq_real_ibm_fez_lnnTrue_spamTrue_full_layout-optimized",                         # the old scheme
    "full_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True",                       # a field missing
    "full_real-data_ibm-fez_lnn-yes_service-ibm-cloud_spam-True_layout-optimized",       # lnn not a bool
    "full_real-data_ibm-fez_lnn-True_service-aws_spam-True_layout-optimized",            # unknown service
    "full_real-QPU_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-optimized",       # unknown data type
    "full_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-15-14",          # two qubits
    "full_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-1-2-3-4",        # four qubits
])
def test_malformed_names_are_refused(bad):
    assert not RN.is_run_name(bad)
    with pytest.raises(ValueError):
        RN.parse_run_name(bad)




def test_service_names():
    # every QPU folder has a service name; the run-name service field maps back to it
    assert [RN.service_name(q) for q in RN.QPUS] == ["IBM_Cloud", "IQM_Resonance", "QAAS_IT4I", "MQSS_LRZ", "PCSS"]
    for qpu, (_, service, name) in RN.QPUS.items():
        assert RN.service_name_of_run_service(service) == name
    assert RN.service_name_of_run_service("local") == "local"
    assert RN.service_name_of_run_service("unknown") is None

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
