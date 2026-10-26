"""
test_20_run_names.py
====================
Unit tests (no simulation, no network) for source/core/records/run_names.py — the one place run-folder,
plot-folder and summary names are built and parsed:

  - every QPU script folder and every data type builds a name that parses back to the same fields
    (real runs, their simulations, standalone health checks, local simulations; the FakeFez simulation with
    its layout field, and an older layout-none one);
  - the <benchmark> field of the presets and of custom benchmarks (n_depths, an explicit nt_list),
    as source/core/runner_support/benchmark_args.py produces it;
  - subset analyses (+nsamples-N), the summary names (short in the run, full in analysis/plots+summaries/)
    and the collective-plot folder names;
  - malformed names are refused, and an explicit layout has exactly three qubits;
  - the date field: dated names round-trip, undated ones still parse, a bad date is refused, the date and run
    number (#N, numeric order, an 8-field name refused) and the full figure labels; dated_run_folder() reuses the
    latest folder only while unfinished and numbers a new run; latest_run_folder() with both date forms;
  - the service names (test_service_names);
  - the run filters of run_post_process.py (RunFilter, make_run_filter, parse_date_filter) and the collective folder name of a filtered set;
  - the collective plots' run labels and run order (collective_plots._labels, run_order), time units
    (_duration_fmt), the run table CSV (_write_run_table) and the linear bar plot (_bar_plot, log=False);
  - the coherent noise of a run read back from its job record (run_post_process.coherent_noise_of_run) and
    the circuit cache folder it names (common.get_circuit_folder).
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import csv
import json

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


@pytest.mark.parametrize("data_type", [d for d in RN.DATA_TYPES if d not in ("real-data", "sim-fake-fez")])   # fake-fez: below
def test_local_names(data_type):
    name = RN.local_run_name("mini", data_type, lnn=False)
    r = RN.parse_run_name(name)
    assert (r.data_type, r.device, r.service, r.layout, r.is_simulation) == (data_type, "local-aer", "local", "none", True)


def test_fake_fez_local_names_carry_a_layout():
    """sim-fake-fez takes a layout field and has no layout-none form: building its name without one is refused."""
    name = RN.local_run_name("mini", "sim-fake-fez", lnn=True, layout_tag=BA.LayoutArgs("random").tag)
    assert name == "mini_sim-fake-fez_local-aer_lnn-True_service-local_spam-True_layout-random"
    r = RN.parse_run_name(name)
    assert (r.data_type, r.device, r.service, r.layout, r.is_simulation) == ("sim-fake-fez", "local-aer", "local", "random", True)
    assert RN.local_run_name("mini", "sim-fake-fez", lnn=True, layout_tag="optimized").endswith("_layout-optimized")
    with pytest.raises(ValueError, match="layout_tag"):
        RN.local_run_name("mini", "sim-fake-fez", lnn=True)


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
    assert RN.collective_plots_dir("mini_collective_real-data").as_posix() == "analysis/plots+summaries/mini_collective_real-data"
    assert RN.collective_plots_dir("x", test=True).as_posix() == "analysis/tests/plots+summaries/x"
    assert RN.plots_dir("tests/test20_x").as_posix() == "analysis/tests/plots+summaries/test20_x"
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



# ── dated run folders ─────────────────────────────────────────────────────────

REAL = "full_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-optimized"


def test_dated_names_round_trip():
    r = RN.parse_run_name(f"{REAL}_2026-10-11_#3")
    assert (r.date, r.run_number, r.undated, str(r), r.layout) == ("2026-10-11", 3, REAL, f"{REAL}_2026-10-11_#3", "optimized")
    assert r.date_label == "11/10/2026 #3"
    assert r.figure_label_full == "ibm-fez — real QPU, layout optimized_by_gates, 11/10/2026 #3"   # an optimized run of before T1/T2 entered the layout score
    assert RN.parse_run_name(f"{REAL}_2026-10-11_#12").run_number == 12
    old = RN.parse_run_name(REAL)                                   # a name without a date still parses
    assert (old.date, old.run_number, old.undated, str(old), old.date_label) == (None, None, REAL, REAL, "date unknown")
    assert old.figure_label_full == "ibm-fez — real QPU, layout optimized, date unknown"
    for bad in (f"{REAL}_11-10-2026_#1", f"{REAL}_2026-10_#1", f"{REAL}_today_#1", f"{REAL}_2026-10-11_x_#1",
                f"{REAL}_2026-10-11",                              # 8 fields: a date without a number
                f"{REAL}_2026-10-11_#0", f"{REAL}_2026-10-11_1", f"{REAL}_2026-10-11_#x"):
        assert not RN.is_run_name(bad)
        with pytest.raises(ValueError):
            RN.parse_run_name(bad)


def _finished_real_run(path, finished):
    """A real run of one chunk whose chunk is complete (or failed)."""
    (path / "job_records").mkdir(parents=True)
    (path / "run_info.json").write_text('{"sample_size": 1}', encoding="utf-8")
    (path / "job_records" / "job_record_chunk_0.json").write_text(
        '{"status": "%s"}' % ("complete" if finished else "failed"), encoding="utf-8")


def test_dated_run_folder_reuses_only_an_unfinished_run(tmp_path, monkeypatch):
    monkeypatch.setattr(RN, "today_utc", lambda: "2026-10-11")
    assert RN.dated_run_folder(REAL, tmp_path) == f"{REAL}_2026-10-11_#1"           # no folder yet: today's #1
    (tmp_path / f"{REAL}_2026-09-01_#1").mkdir()                                      # nothing recorded: unfinished, earlier day
    assert RN.dated_run_folder(REAL, tmp_path) == f"{REAL}_2026-09-01_#1"           # reused
    _finished_real_run(tmp_path / f"{REAL}_2026-10-01_#1", finished=False)
    assert RN.dated_run_folder(REAL, tmp_path) == f"{REAL}_2026-10-01_#1"           # the latest, unfinished: reused
    (tmp_path / f"{REAL}_2026-10-01_#1" / "job_records" / "job_record_chunk_0.json").write_text(
        '{"status": "complete"}', encoding="utf-8")
    assert RN.run_finished(f"{REAL}_2026-10-01_#1", tmp_path)
    assert RN.dated_run_folder(REAL, tmp_path) == f"{REAL}_2026-10-11_#1"           # finished: a new run today
    assert RN.dated_run_folders(REAL, tmp_path) == [f"{REAL}_2026-09-01_#1", f"{REAL}_2026-10-01_#1"]
    _finished_real_run(tmp_path / f"{REAL}_2026-10-11_#1", finished=True)
    assert RN.dated_run_folder(REAL, tmp_path) == f"{REAL}_2026-10-11_#2"           # today's exists and is complete: #2
    _finished_real_run(tmp_path / f"{REAL}_2026-10-11_#2", finished=False)
    assert RN.dated_run_folder(REAL, tmp_path) == f"{REAL}_2026-10-11_#2"           # unfinished: continued
    assert RN.dated_run_folder(REAL, tmp_path, new_run=True) == f"{REAL}_2026-10-11_#3"   # forced: the next number
    (tmp_path / f"{REAL}_notadate").mkdir()                                           # ignored
    (tmp_path / f"{REAL}_2026-10-11").mkdir()                                         # no number: ignored
    assert RN.dated_run_folders(REAL, tmp_path)[-1] == f"{REAL}_2026-10-11_#2"
    assert RN.dated_run_folder(f"tests/{REAL}", tmp_path) == f"tests/{REAL}_2026-10-11_#1"   # a subfolder is kept


def test_run_numbers_sort_numerically(tmp_path):
    for n in (1, 2, 10):
        (tmp_path / f"{REAL}_2026-10-11_#{n}").mkdir()
    (tmp_path / f"{REAL}_2026-10-10_#11").mkdir()
    assert RN.dated_run_folders(REAL, tmp_path) == [f"{REAL}_2026-10-10_#11", f"{REAL}_2026-10-11_#1",
                                                    f"{REAL}_2026-10-11_#2", f"{REAL}_2026-10-11_#10"]   # #10 after #2
    assert RN.latest_run_folder(REAL, tmp_path) == f"{REAL}_2026-10-11_#10"


def test_latest_run_folder(tmp_path):
    assert RN.latest_run_folder(REAL, tmp_path) is None
    for d in ("2026-10-05_#1", "2026-10-05_#2", "2026-09-30_#1"):
        (tmp_path / f"{REAL}_{d}").mkdir()
    assert RN.latest_run_folder(REAL, tmp_path) == f"{REAL}_2026-10-05_#2"
    assert RN.latest_run_folder(REAL, tmp_path, "2026-09-30") == f"{REAL}_2026-09-30_#1"          # a day: its latest run
    assert RN.latest_run_folder(REAL, tmp_path, "2026-10-05") == f"{REAL}_2026-10-05_#2"
    assert RN.latest_run_folder(REAL, tmp_path, "2026-10-05_#1") == f"{REAL}_2026-10-05_#1"       # exactly that run
    assert RN.latest_run_folder(REAL, tmp_path, "2026-10-05_#3") is None
    assert RN.latest_run_folder(REAL, tmp_path, "2026-01-01") is None


def test_collective_plot_labels_and_order():
    from source.core.analysis import collective_plots as CP
    def run(name, layout, date, folder, shots=None, number=1):
        label = RN.RunName("full", "real-data", "x", "y", "True", "local", "True", layout or "none",
                           date, number if date else None).date_label
        return {"name": name, "layout": layout, "date": date, "run_number": number if date else None,
                "date_label": label, "shots": shots, "n_samples": None, "run_folder": folder}
    runs = [run("ibm-fez", "optimized", "2026-10-05", "a"), run("aqt-piast-q", "random", None, "b"),
            run("ibm-fez", "optimized", "2026-09-30", "c"), run("ibm-fez", "lowest", "2026-10-01", "d")]
    assert [r["run_folder"] for r in sorted(runs, key=CP.run_order)] == ["b", "d", "c", "a"]
    assert CP._labels(runs) == ["ibm-fez, layout optimized, 05/10/2026 #1", "aqt-piast-q, layout random, date unknown",
                                "ibm-fez, layout optimized, 30/09/2026 #1", "ibm-fez, layout lowest, 01/10/2026 #1"]
    same_day = [run("ibm-fez", "optimized", "2026-10-05", "x10", number=10), run("ibm-fez", "optimized", "2026-10-05", "x2", number=2)]
    assert [r["run_folder"] for r in sorted(same_day, key=CP.run_order)] == ["x2", "x10"]   # same day: by run number
    twins = [run("ibm-fez", "optimized", "2026-10-05", "a", 100), run("ibm-fez", "optimized", "2026-10-05", "b", 200)]
    assert CP._labels(twins) == ["ibm-fez, layout optimized, 05/10/2026 #1, 100 shots",
                                 "ibm-fez, layout optimized, 05/10/2026 #1, 200 shots"]


def test_duration_fmt_units():
    from source.core.analysis import collective_plots as CP
    assert CP._duration_fmt(42.1) == "42.1 s"
    assert CP._duration_fmt(60) == "60 s"
    assert CP._duration_fmt(60.5) == "1.0 min"
    assert CP._duration_fmt(750) == "12.5 min"
    assert CP._duration_fmt(3600) == "60.0 min"
    assert CP._duration_fmt(3601) == "1.0 h"
    assert CP._duration_fmt(24480) == "6.8 h"
    assert CP._duration_fmt(86400) == "24.0 h"
    assert CP._duration_fmt(90000) == "1.0 d"


def test_collective_run_table(tmp_path):
    from source.core.analysis import collective_plots as CP
    def run(folder, **kw):
        base = dict(run_folder=folder, n_samples=None, sample_size=480, qpu_time_s=151.0, cost_eur=212.084, n_acc=19, n_max=19,
                    emin_nmax=0.04512, best_ems_nmax="NEC+cRC", best_ems_nacc="cRC")
        base.update(kw)
        return base
    runs = [run("full_real-data_a"), run("full_sim_b", n_samples=10, qpu_time_s=None, cost_eur=None,
                                         emin_nmax=float("nan")), run("old_c", n_acc=None, n_max=None, emin_nmax="N/A")]
    path = tmp_path / "collective_summary.csv"
    CP._write_run_table(runs, ["A", "B", "C"], path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len({line.index(";") for line in lines}) == 1                       # columns aligned as text
    rows = [[c.strip() for c in r] for r in csv.reader(lines, delimiter=";")]
    assert rows[0] == ["run", "n_samples", "sample_size", "N_acc", "best_scheme_Nacc", "N_max", "E_min_Nmax", "best_scheme_Nmax",
                       "QPU_time_s", "QPU_time", "effective_cost_EUR"]
    assert [r[0] for r in rows[1:]] == ["A", "B", "C"]
    assert rows[1] == ["A", "", "480", "19", "cRC", "19", "0.04512", "NEC+cRC", "151", "2.5 min", "212.08"]
    assert rows[2][3:] == ["19", "cRC", "19", "", "NEC+cRC", "", "", ""] and rows[2][1] == "10"
    assert rows[3][3] == "" and rows[3][5:7] == ["", ""]
    assert str(tmp_path) not in path.read_text(encoding="utf-8")


def test_bar_plot_linear_keeps_zero(tmp_path):
    from source.core.analysis import collective_plots as CP
    runs = [{"n_acc": 0}, {"n_acc": 7}, {"n_acc": None}]
    path = tmp_path / "n_acc.png"
    CP._bar_plot(runs, ["a", "b", "c"], "n_acc", "N", "N per run", lambda v: f"{v:.0f}", path, log=False)
    assert path.stat().st_size > 0


def test_service_names():
    # every QPU folder has a service name; the run-name service field maps back to it
    assert [RN.service_name(q) for q in RN.QPUS] == ["IBM_Cloud", "IQM_Resonance", "QAAS_IT4I", "MQSS_LRZ", "PCSS"]
    for qpu, (_, service, name) in RN.QPUS.items():
        assert RN.service_name_of_run_service(service) == name
    assert RN.service_name_of_run_service("local") == "local"
    assert RN.service_name_of_run_service("unknown") is None

def test_run_filter():
    # parse_run_name fields drive RunFilter.matches; AND between fields, OR within a list, undated names fail a date filter
    a = RN.parse_run_name("mini_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-optimized_2026-09-10_#1")
    b = RN.parse_run_name("mini_real-data_iqm-garnet_lnn-True_service-iqm-resonance_spam-True_layout-random_2026-10-02_#2")
    c = RN.parse_run_name("mini_sim-noiseless_local-aer_lnn-True_service-local_spam-True_layout-none")
    mk = RN.make_run_filter
    assert RN.RunFilter().is_empty and all(RN.RunFilter().matches(n) for n in (a, b, c))
    assert [n for n in (a, b, c) if mk(date="2026-09-10").matches(n)] == [a]
    assert [n for n in (a, b, c) if mk(date="2026-09-01:2026-10-02").matches(n)] == [a, b]
    assert [n for n in (a, b, c) if mk(date="2026-09-15:").matches(n)] == [b]
    assert [n for n in (a, b, c) if mk(date=":2026-09-15").matches(n)] == [a]
    assert [n for n in (a, b, c) if mk(backends=["ibm-fez", "iqm-garnet"]).matches(n)] == [a, b]
    assert [n for n in (a, b, c) if mk(services=["local"]).matches(n)] == [c]
    assert [n for n in (a, b, c) if mk(layouts=["random", "none"]).matches(n)] == [b, c]
    # any combination
    assert [n for n in (a, b, c) if mk(date="2026-09-01:2026-10-31", layouts=["random"]).matches(n)] == [b]
    assert [n for n in (a, b, c) if mk(backends=["ibm-fez"], layouts=["random"]).matches(n)] == []
    assert [n for n in (a, b, c) if mk(date="2026-10-01:", backends=["iqm-garnet"], services=["iqm-resonance"]).matches(n)] == [b]


def test_run_filter_values_and_names():
    mk = RN.make_run_filter
    assert RN.parse_date_filter("2026-09-01") == ("2026-09-01", "2026-09-01")
    assert RN.parse_date_filter("2026-09-01:") == ("2026-09-01", None)
    for bad in ("", ":", "2026-13-01", "2026-9-1", "2026-09-30:2026-09-01", "x"):
        with pytest.raises(ValueError):
            RN.parse_date_filter(bad)
    with pytest.raises(ValueError):
        mk(backends=["fez"])
    with pytest.raises(ValueError):
        mk(services=["ibm"])
    with pytest.raises(ValueError):
        mk(layouts=["best"])
    # the collective folder tells a filtered set from the whole one
    assert RN.collective_folder_name("mini", ["real-data"]) == "mini_collective_real-data"
    assert RN.collective_folder_name("mini", ["real-data"], RN.RunFilter()) == "mini_collective_real-data"
    assert RN.collective_folder_name("mini", ["real-data"], latest_per_qpu=True) == "mini_collective_real-data_latest-per-qpu"
    assert RN.collective_folder_name("mini", ["real-data"], per_qpu="best-nacc") == "mini_collective_real-data_best-nacc-per-qpu"
    assert (RN.collective_folder_name("mini", ["real-data"], RN.make_run_filter(backends=["ibm-fez"]), per_qpu="best-emin-at-nmax")
            == "mini_collective_real-data_backends-ibm-fez_best-emin-at-nmax-per-qpu")
    f = mk(date="2026-09-01:2026-09-30", backends=["iqm-garnet", "ibm-fez"], layouts=["random"])
    assert RN.collective_folder_name("mini", ["real-data"], f) == (
        "mini_collective_real-data_dates-2026-09-01-to-2026-09-30_backends-ibm-fez+iqm-garnet_layouts-random")
    assert mk(date="2026-09-01").suffix() == "_date-2026-09-01"
    assert mk(date=":2026-09-30").suffix() == "_dates-until-2026-09-30"
    long = mk(backends=[f"ibm-b{i}" for i in range(10)])
    assert len(long.suffix()) < 40 and long.suffix().startswith("_backends-10-")


def test_layout_label():
    em = "full_real-data_iqm-emerald_lnn-True_service-mqss-lrz_spam-True_layout-legacy_2026-09-07_#1"
    eq = "full_real-data_iqm-eqe1_lnn-True_service-mqss-lrz_spam-True_layout-legacy_2026-08-27_#1"
    old = "full_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-optimized_2026-10-19_#1"
    assert RN.layout_label(em) == "optimized_per_chunk_by_2q_gates_IQM"
    assert RN.layout_label(eq) == "legacy"
    assert RN.layout_label(old) == "optimized_by_gates"
    assert RN.layout_label(old.replace("optimized", "random")) == "random"
    assert RN.layout_label(old.replace("2026-10-19", "2026-10-20")) == "optimized"   # no physical_qubits.json to say otherwise
    flt = RN.make_run_filter(layouts=["optimized_by_gates"])
    assert flt.matches(RN.parse_run_name(old)) and not flt.matches(RN.parse_run_name(eq))


def test_pick_best_per_qpu(tmp_path, monkeypatch):
    from source.analysis_scripts import run_post_process as P
    monkeypatch.setattr(P, "PROJECT_ROOT", tmp_path)
    base = "full_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-optimized_2026-10-{d}_#1"
    sim = "full_sim-qpu-noise_ibm-fez_lnn-True_service-local_spam-True_layout-optimized_2026-10-01_#1"
    values = {base.format(d="01"): 5, base.format(d="02"): 9, base.format(d="03"): 9, sim: 50}
    for folder, v in values.items():
        d = tmp_path / "datafiles" / folder / "processed_results"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(f'{{"N_acc": {v}, "E_min_Nmax": {v / 100}}}', encoding="utf-8")
    runs = [(f, None) for f in values]
    assert P.pick_best_per_qpu(runs, "N_acc", True) == [(base.format(d="03"), None), (sim, None)]   # tie: the later run; data types apart
    assert P.pick_best_per_qpu(runs, "E_min_Nmax", False) == [(base.format(d="01"), None), (sim, None)]
    d = tmp_path / "datafiles" / base.format(d="04") / "processed_results"
    d.mkdir(parents=True)
    (d / "summary.json").write_text('{"N_acc": null}', encoding="utf-8")
    assert P.pick_best_per_qpu(runs + [(base.format(d="04"), None)], "N_acc", True)[0][0] == base.format(d="03")


def test_coherent_noise_of_run_names_the_cache(tmp_path):
    from source.analysis_scripts.run_post_process import coherent_noise_of_run
    from source.core.common import get_circuit_folder
    noise = {"noise_type": "pauli", "infidelity_total": 5e-3, "ratio_ct": 0.5}
    (tmp_path / "job_records").mkdir()
    rec = tmp_path / "job_records" / "job_record_chunk_0.json"
    rec.write_text(json.dumps({"CONFIG": {"coherent_noise": noise}}), encoding="utf-8")
    assert coherent_noise_of_run(tmp_path) == noise
    cfg = {"lnn_architecture": False, "coherent_noise": coherent_noise_of_run(tmp_path)}
    assert get_circuit_folder(cfg, top_folder=True).endswith("_pauli_I5.0e-03_rct5.0e-01")
    rec.write_text(json.dumps({"CONFIG": {}}), encoding="utf-8")   # an old or noiseless record
    assert coherent_noise_of_run(tmp_path) is None
    assert coherent_noise_of_run(tmp_path / "no_such_run") is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
