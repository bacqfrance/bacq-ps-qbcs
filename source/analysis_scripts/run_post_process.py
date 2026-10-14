"""
run_post_process.py
Run the full BACQ analysis pipeline for a chosen set of runs (real QPU or local simulator):
  1. merge + analyze   (source.core.qbcs.bacq_ps_qbcs.merge_all_chunks / .analyze_results)
  2. summary json      (source/analysis_scripts/bacq_ps_qbcs_summary_json.py)
  3. observable plots  (source/analysis_scripts/plot_real_qpu_results.py)
  4. AWAE plots        (source/analysis_scripts/plot_awae.py)
  5. health checks     (source/analysis_scripts/plot_health_checks.py)
  6. collective plots  (source/core/analysis/collective_plots.py) — all analyzed runs together:
     AWAE vs N_t of each run's best EMS; QPU, submission and collection time, the
     submission-to-results window and the effective cost per QPU; N_acc and E_min at N_max per
     run; and a run table CSV (N_acc, N_max, E_min, best scheme, QPU time, effective cost) in analysis/summaries/

Stages 2-5 reuse the existing scripts, driven per-run via the QPU_RUN_FOLDER /
QPU_N_SAMPLES env vars; each run's summary json is also copied to analysis/summaries/. Each run's own settings (shots, samples,
depths, topology, SPAM) come from its run folder — source/core/records/run_info.py — not from
constants here, so e.g. mini, full and custom runs are analyzed alike.

Usage (which runs to analyze — see RUN_THESE below; without arguments, or with --help, the usage is printed
and nothing is done):
  python source/analysis_scripts/run_post_process.py mini --include-real-data   the mini benchmark, real-QPU runs
  python source/analysis_scripts/run_post_process.py full --include-real-data   the full benchmark, real-QPU runs
  python source/analysis_scripts/run_post_process.py mini nshots_per_circuit=N --include-real-data
  python source/analysis_scripts/run_post_process.py nshots_per_circuit=N n_samples=M n_depths=K --include-real-data
                                     the runs made with these source/benchmark_scripts/ arguments
                                     (source/core/runner_support/benchmark_args.py): run folders with that <benchmark> field
  python source/analysis_scripts/run_post_process.py nshots_per_circuit=N --include-real-data   every run with N shots per circuit
Which kinds of runs to include — at least one flag is required (without any, nothing
is analyzed and the usage is printed); they combine:
  --include-real-data               real-QPU runs (data type real-data)
  --include-simulations-real        simulations with a real backend's noise (sim-qpu-noise, sim-fake-fez)
  --include-simulations-noiseless   noiseless simulations (sim-noiseless)
  --include-simulations-coherent    simulations with coherent CNOT noise (sim-coherent)
Optional filters narrow the selected runs, for the individual and the collective analysis alike, and combine freely:
  --date=2026-09-01 | --date=2026-09-01:2026-09-30 | --date=2026-09-01: | --date=:2026-09-30   the run's date
  --backends=ibm-fez,iqm-garnet     the run's <provider>-<backend>
  --services=ibm-cloud,pcss         the run's service
  --layouts=optimized,random        the run's qubit layout (optimized, random, lowest, legacy, none, 15-14-21)
A filtered collective set goes to its own folder, e.g. mini_collective_real-data_dates-2026-09-01-to-2026-09-30_backends-ibm-fez.
Add --individual-only (stages 1-5 only) or --collective-only (stage 6 only, from the
runs' existing analysis results); without either flag both are done. The collective plots go to
analysis/plots/<benchmark>_collective_<data types>/, e.g. mini_collective_real-data+sim-noiseless
(source/core/records/run_names.py); with nshots_per_circuit=N alone the benchmark part is any+nshots-N.
Add --nsamples-comparison (full benchmark only) to also analyze each run on 8, 48, 120, 240 and
480 samples (NSAMPLES_COMPARISON; JSONs only, no per-N plots; only the RC samples vary, the SPAM
correction always uses every collected chunk's calibration) and plot the full run's best EMS for
each in analysis/plots/<run folder>/awae_best_ems_nsamples_comparison.png. It is per-run only:
the collective plots are made as usual (from the full runs), and with --collective-only the flag is ignored.

"""

import os
import sys
import json
import re
import shutil
import subprocess
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

from pathlib import Path
from source.core.common import section
from source.core.qbcs import bacq_ps_qbcs
from source.core.records.run_info import load_run_info
from source.core.runner_support.benchmark_args import parse_benchmark_args, usage_if_requested
from source.core.qpu.layout_selection import run_layout_warnings
from source.core.records.health_check_records import load_health_checks
from source.core.analysis.collective_plots import make_collective_plots
from source.core.records.chunk_scope import uncollected_chunks
from source.core.records.timestamps import TIMESTAMP_FILENAME
from source.core.records.health_check_records import format_chunk_range
from source.core.runner_support.venvs import venv_python
from source.core.common import exit_with_error, PROJECT_ROOT
from source.core.records.run_names import (collective_folder_name, collective_plots_dir, collective_table_path, entry_data_type, is_run_name, make_run_filter,
                                   parse_run_name, run_summary_name, subset_suffix, summary_name, summaries_dir)

ANALYSIS_SCRIPTS = PROJECT_ROOT / "source" / "analysis_scripts"   # the post-processing scripts run_script() launches

# ── What to analyze: three modes, chosen on the command line ─────────────────
#   python source/analysis_scripts/run_post_process.py                          the usage (a selection is required)
#   python source/analysis_scripts/run_post_process.py mini                     mini benchmark
#   python source/analysis_scripts/run_post_process.py full                     full benchmark
#   python source/analysis_scripts/run_post_process.py <benchmark arguments>    e.g. mini nshots_per_circuit=100
#   python source/analysis_scripts/run_post_process.py nshots_per_circuit=N     every run with N shots per circuit
# full / mini analyze the lists below. Every run folder is named
#   <benchmark>_<data type>_<provider>-<backend>_lnn-…_service-…_spam-…_layout-…_<YYYY-MM-DD>_#<N>   (source/core/records/run_names.py)
# and a list entry is that name WITHOUT its "<benchmark>_" — the mode supplies the benchmark (full, mini) —
# cut off after any field: "real-data_ibm-fez" matches every IBM Fez run of the benchmark, whatever its
# LNN / SPAM / layout / date; add fields to pick one when there are several. Any other benchmark arguments (as the
# source/benchmark_scripts/ take them — source/core/runner_support/benchmark_args.py) select every run folder whose
# <benchmark> field they give (mini+nshots-100, custom+ndepths-3+nsamples-8+nshots-50, …).
# nshots_per_circuit=N alone selects by shot count only (from each run's run_info.json), whatever the
# benchmark size.
#
# List entries:
# Plain string  = auto-detect: use every chunk actually collected; if that
#                 equals the run's full sample size and it was already analyzed,
#                 skip merge/analyze entirely. Chunks are collected as their jobs
#                 finish, so an incomplete run may have gaps (chunks 0-40 and 45):
#                 it is analyzed as a "+nsamples-<collected>" subset, with the uncollected
#                 chunks and health checks listed inside and warned about on the figures.
# dict(run=…)   the same entry with options:
#   "n_samples"   force-analyze exactly that many chunks, regardless of how many are
#                 collected (must be <= collected) — e.g. preview EQE1's complete
#                 480-chunk run at n_samples=48 without touching its full-run output files.
#   "benchmark"   another <benchmark> field than the mode's, e.g. "full+nshots-100".
#   "layout"      the layout field (optimized, random, lowest, legacy, 15-14-21), when the
#                 entry matches runs with several layouts and only one is wanted (without it, all are analyzed).
#   "date"        the date (YYYY-MM-DD, or YYYY-MM-DD_#N for one run of that day), when only that day is wanted (without it, every
#                 dated run of the entry is analyzed).
#   "run_folder"  an explicit folder relative to datafiles/ (then "run" is not needed),
#                 e.g. "old_runs_full/full_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-lowest_2026-09-21_#1".
# A run's topology (and every other setting) comes from the run folder itself —
#                 source/core/records/run_info.py — so no per-QPU configuration is needed here.
REAL_QPUS = [
     "real-data_ibm-marrakesh",
     "real-data_ibm-kingston",
     "real-data_ibm-fez",
     "real-data_iqm-eqe1",
     "real-data_iqm-emerald",
     "real-data_iqm-garnet",
     "real-data_iqm-sirius",
     "real-data_iqm-vlq",
     "real-data_aqt-piast-q",
]

SIMULATED_QPUS = [   # simulations with a real QPU's calibration (qbcs_real_simulate_*.py), cut off after the backend: every layout
    "sim-qpu-noise_ibm-marrakesh",
    "sim-qpu-noise_ibm-kingston",
    "sim-qpu-noise_ibm-fez",
    "sim-qpu-noise_iqm-eqe1",
    "sim-qpu-noise_iqm-emerald",
    "sim-qpu-noise_iqm-garnet",
    "sim-qpu-noise_iqm-sirius",
    "sim-qpu-noise_iqm-vlq",
    "sim-qpu-noise_aqt-piast-q",
]

LOCAL_SIMULATIONS = [   # source/benchmark_scripts/local_simulation_scripts/
    "sim-noiseless_local-aer_lnn-False",
    "sim-noiseless_local-aer_lnn-True",
    "sim-fake-fez_local-aer_lnn-True",
    "sim-coherent_local-aer_lnn-False",   # coherent CNOT noise (benchmark_coherent_noise_*.py)
    "sim-coherent_local-aer_lnn-True",
]

RUN_THESE = {
    "full": [
        *LOCAL_SIMULATIONS,
        # Simulations with a real QPU's calibration — source/benchmark_scripts/real_qpu_scripts/<QPU>/qbcs_real_simulate_*.py full:
        *SIMULATED_QPUS,
        # Real QPUs:
        *REAL_QPUS,
        # dict(run="real-data_ibm-fez", n_samples=48),
        # dict(run="real-data_aqt-piast-q", n_samples=120),
        # The earlier local full runs, at 1000 shots per circuit:
        # dict(run="sim-noiseless_local-aer_lnn-True", benchmark="full+nshots-1000"),
        # Archived full runs (datafiles/old_runs_full/), located by an explicit run_folder:
        # dict(run_folder="old_runs_full/full_real-data_ibm-fez_lnn-True_service-ibm-cloud_spam-True_layout-lowest_2026-09-21_#1"),
        # dict(run_folder="old_runs_full/full_real-data_iqm-eqe1_lnn-True_service-mqss-lrz_spam-True_layout-legacy_2026-08-27_#1"),
    ],
    "mini": [
        *LOCAL_SIMULATIONS,
        # Simulations with a real QPU's calibration — source/benchmark_scripts/real_qpu_scripts/<QPU>/qbcs_real_simulate_*.py:
        *SIMULATED_QPUS,
        # Real QPUs (source/benchmark_scripts/real_qpu_scripts/, mini):
        *REAL_QPUS,
    ],
}

NSAMPLES_COMPARISON = [8, 48, 120, 240, 480]   # sample counts of --nsamples-comparison

USAGE = """usage: python source/analysis_scripts/run_post_process.py SELECTION INCLUDE... [FILTER...] [--individual-only | --collective-only] [--nsamples-comparison]

  Run the BACQ analysis pipeline (merge + analyze, summary, plots, collective plots) for a chosen set of runs,
  real-QPU and/or simulated. Nothing is analyzed unless at least one INCLUDE flag is given.

  SELECTION — which runs (required):
  mini                   the mini benchmark (RUN_THESE["mini"])
  full                   the full benchmark (RUN_THESE["full"])
  BENCHMARK ARGUMENTS    as the source/benchmark_scripts/ take them (without layout= / calibration_file=), e.g.
                         mini nshots_per_circuit=100 or nshots_per_circuit=50 n_samples=8 n_depths=3:
                         every run folder with that <benchmark> field
  nshots_per_circuit=N   every run in datafiles/ with N shots per circuit (N a positive integer)

  INCLUDE — which kinds of runs to analyze; at least one, they combine (none: nothing is analyzed):
  --include-real-data               real-QPU runs (data type real-data)
  --include-simulations-real        simulations with a real backend's noise (sim-qpu-noise, sim-fake-fez)
  --include-simulations-noiseless   noiseless simulations (sim-noiseless)
  --include-simulations-coherent    simulations with coherent CNOT noise (sim-coherent)

  FILTERS — optional; narrow the selected runs, for the individual and the collective analysis alike. They combine
  (a run must pass every filter given; within one list any value matches):
  --date=D               runs of that day (YYYY-MM-DD), or a range --date=FROM:TO (inclusive; FROM: or :TO leave
                         one end open); runs of a name without a date never match
  --backends=A,B         the <provider>-<backend> of the run's QPU, e.g. ibm-fez,iqm-garnet
  --services=A,B         the service, e.g. ibm-cloud,pcss (local for the local simulations)
  --layouts=A,B          the qubit layout: optimized, random, lowest, legacy, none or three qubits like 15-14-21
  The collective plots of a filtered set go to their own folder (<benchmark>_collective_<data types>_dates-…_backends-…).

  ANALYSIS SCOPE — individual and/or collective analysis and plots:
  --individual-only      only the per-run analysis and plots (stages 1-5)
  --collective-only      only the plots comparing all runs (stage 6), from their existing results
  (neither flag: both)

  --nsamples-comparison  full only: also analyze each run on 8, 48, 120, 240, 480 samples and plot its
                         best EMS per sample count (awae_best_ems_nsamples_comparison.png; only the RC samples vary,
                         the SPAM correction uses every collected chunk); 
                         only for individual analysis (collective plots are unaffected); ignored with --collective-only

  --help                 print this message and exit"""


INCLUDE_FLAGS = {
    "--include-real-data":             "real",
    "--include-simulations-real":      "sim_real",
    "--include-simulations-noiseless": "sim_noiseless",
    "--include-simulations-coherent":  "sim_coherent",
}


FILTER_OPTIONS = ("--date", "--backends", "--services", "--layouts")   # the --<name>=<value> run filters


KIND_DATA_TYPES = {   # run kind → the data types it covers; the first names it in the collective-plot folder
    "real":          ("real-data",),
    "sim_real":      ("sim-qpu-noise", "sim-fake-fez"),
    "sim_noiseless": ("sim-noiseless",),
    "sim_coherent":  ("sim-coherent",),
}


def parse_include(argv):
    """(remaining args, set of run kinds) from the --include-* flags. With none of them,
    prints the usage and exits: nothing is analyzed unless asked for."""
    kinds = {INCLUDE_FLAGS[a] for a in argv if a in INCLUDE_FLAGS}
    if not kinds:
        print(USAGE, file=sys.stderr)
        exit_with_error("Nothing to analyze — pass at least one --include-* flag.")
    return [a for a in argv if a not in INCLUDE_FLAGS], kinds


def _kind_of_data_type(dtype):
    return next(k for k, types in KIND_DATA_TYPES.items() if dtype in types)


def run_kind(run_folder):
    """"real", "sim_real", "sim_noiseless" or "sim_coherent" — the --include-* kind of a run folder."""
    return _kind_of_data_type(parse_run_name(run_folder).data_type)


def entry_kind(cfg):
    """The --include-* kind of a run cfg, from its data type: the first field of a "run" entry, or the
    folder name of a "run_folder" entry. Lets main() skip unselected entries before looking up their folders."""
    if cfg.get("run_folder"):
        dtype = parse_run_name(Path(cfg["run_folder"]).name).data_type
    else:
        dtype = entry_data_type(cfg["run"])
    return _kind_of_data_type(dtype)


def parse_filters(argv):
    """(remaining args, RunFilter) from the --date= / --backends= / --services= / --layouts= options. Prints the
    usage and exits with code 2 on a repeated option or a value no run name can have."""
    values, rest = {}, []
    for a in argv:
        key, eq, val = a.partition("=")
        if eq and key in FILTER_OPTIONS:
            if key in values:
                print(USAGE, file=sys.stderr)
                exit_with_error(f"{key} given twice.")
            values[key] = val
        else:
            rest.append(a)
    try:
        lists = {k: [v for v in values[k].split(",") if v] if k in values else None
                 for k in ("--backends", "--services", "--layouts")}
        for k in ("--backends", "--services", "--layouts"):
            if k in values and not lists[k]:
                raise ValueError(f"{k}= needs at least one value")
        return rest, make_run_filter(values.get("--date"), lists["--backends"], lists["--services"], lists["--layouts"])
    except ValueError as e:
        print(USAGE, file=sys.stderr)
        exit_with_error(str(e))


def parse_scope(argv):
    """(remaining args, do_individual, do_collective) from the --individual-only /
    --collective-only flags. Prints the usage and exits with code 2 if both are given."""
    flags = {a for a in argv if a in ("--individual-only", "--collective-only")}
    if len(flags) > 1:
        print(USAGE, file=sys.stderr)
        exit_with_error("--individual-only and --collective-only exclude each other.")
    rest = [a for a in argv if a not in flags]
    return rest, "--collective-only" not in flags, "--individual-only" not in flags


def parse_mode(argv):
    """The run selection from the command-line arguments: ("preset", "full"|"mini"),
    ("tag", <tag>) for other benchmark arguments, or ("nshots", N) for nshots_per_circuit=N alone.
    Prints the usage and exits with code 2 on anything else."""
    if len(argv) == 1:
        m = re.fullmatch(r"nshots_per_circuit=(\d+)", argv[0])
        if m and int(m.group(1)) > 0:
            return "nshots", int(m.group(1))
    unknown = [a for a in argv if not (a in RUN_THESE or re.fullmatch(r"(nshots_per_circuit|n_samples|n_depths|nt_list)=.+", a))]
    if unknown:
        print(USAGE, file=sys.stderr)
        exit_with_error(f"unknown argument {unknown[0]!r}")
    bench = parse_benchmark_args(argv)
    if bench.tag in RUN_THESE:
        return "preset", bench.tag
    return "tag", bench.tag


def benchmark_runs():
    """Every benchmark run folder in datafiles/ (named as core/records/run_names.py says; no standalone health checks)."""
    return [p.name for p in sorted((PROJECT_ROOT / "datafiles").iterdir())
            if p.is_dir() and is_run_name(p.name) and not parse_run_name(p.name).is_health_check]


def runs_with_tag(tag):
    """Every run folder whose <benchmark> field is `tag`."""
    return [dict(run_folder=f) for f in benchmark_runs() if parse_run_name(f).benchmark == tag]


def runs_with_nshots(nshots):
    """Every run folder in datafiles/ whose run_info.json says nshots shots per circuit."""
    runs = []
    for f in benchmark_runs():
        if not (PROJECT_ROOT / "datafiles" / f / "run_info.json").exists():
            continue
        try:
            if load_run_info(f).nshots == nshots:
                runs.append(dict(run_folder=f))
        except Exception as e:
            print(f"⚠️  {f}: unreadable run_info.json — skipped ({e})")
    return runs


def select_runs(mode, value):
    """(run cfgs, the <benchmark> field their list entries are matched with) for a selection from parse_mode()."""
    if mode == "nshots":
        return runs_with_nshots(value), None
    if mode == "tag":
        return runs_with_tag(value), value
    return [dict(run=e) if isinstance(e, str) else dict(e) for e in RUN_THESE[value]], value


def _date_matches(name, date):
    """Whether the RunName `name` is of `date`: YYYY-MM-DD (any run of that day) or YYYY-MM-DD_#N."""
    return date in (name.date, f"{name.date}_#{name.run_number}")


def find_run_folders(entry, benchmark, layout=None, date=None):
    """The run folders of a RUN_THESE entry: every folder named <benchmark>_<entry>[_<more fields>], with
    layout-<layout> and the date if given (e.g. one per layout and day): YYYY-MM-DD (every run of that day) or
    YYYY-MM-DD_#N (that run). Empty, silently, if the entry has not been run."""
    prefix = f"{benchmark}_{entry}"
    return [f for f in benchmark_runs()
            if (f == prefix or f.startswith(prefix + "_"))
            and (layout is None or parse_run_name(f).layout == layout)
            and (date is None or _date_matches(parse_run_name(f), date))]


def resolve_cfg(cfg, benchmark):
    """The run folders (relative to datafiles/) of a run cfg: a list, empty if there are none."""
    if cfg.get("run_folder"):
        return [cfg["run_folder"]]
    return find_run_folders(cfg["run"], cfg.get("benchmark", benchmark), cfg.get("layout"), cfg.get("date"))


def folder_order(run_folder):
    """Sort key of the runs: (data type, device, benchmark, lnn, layout, date, run number), the last two oldest first — the order
    they are analyzed and listed in."""
    n = parse_run_name(run_folder)
    return (n.data_type, n.device, n.benchmark, n.lnn, n.layout, n.date or "", n.run_number or 0)


def coherent_noise_of_run(run_path):
    """The run's CONFIG["coherent_noise"] (dict with noise_type, infidelity_total, ratio_ct) from its first job
    record, or None for a noiseless run, an old record without the key, or no job record. The circuit cache
    folder (get_circuit_folder()) is named by it, so the rebuilt pipeline object needs it."""
    rec0_path = Path(run_path) / "job_records" / "job_record_chunk_0.json"
    if not rec0_path.exists():
        return None
    return json.loads(rec0_path.read_text(encoding="utf-8"))["CONFIG"].get("coherent_noise")


def merge_and_analyze(run_folder, n_samples_override=None):
    """Stage 1 — reuses bacq_ps_qbcs.merge_all_chunks()/.analyze_results() as-is.

    n_samples_override: if given, force-analyze exactly this many chunks —
    must be <= chunks actually collected/merged — even if more are available.
    This bypasses the "skip if already full" auto-detect behaviour below.

    Local-simulator runs take the same path as real ones: merged +
    analyzed here unless the run's own outputs already exist (the local-simulation
    scripts analyze as they go, so usually they do).

    Returns (ok, n_samples) — ok=False means skipped (no data yet, or an
    invalid override); n_samples=None means the full run's own unsuffixed
    files apply (either just analyzed, or already analyzed)."""
    backend_name = run_folder   # for the messages
    is_sim     = parse_run_name(run_folder).is_simulation
    run_path   = PROJECT_ROOT / "datafiles" / run_folder
    info       = load_run_info(run_folder)

    counts_path = run_path / "raw_results" / "results_counts.json"
    if not is_sim and not (run_path / "job_records" / "job_record_chunk_0.json").exists():
        print(f"⏭  {backend_name}: no job records yet — skipping")
        return False, None
    if not counts_path.exists():
        print(f"⏭  {backend_name}: no collected results yet — skipping")
        return False, None
    n_collected = len(json.loads(counts_path.read_text(encoding="utf-8")))
    sample_size = info.sample_size

    def _analyzed(n):
        suffix = "" if n >= sample_size else subset_suffix(n)
        return (run_path / "processed_results" / f"run_data{suffix}.json").exists()

    if n_samples_override is not None:
        if not (0 < n_samples_override <= n_collected):
            print(f"❌ {backend_name}: requested n_samples={n_samples_override}, but only "
                  f"{n_collected} chunks are collected — skipping", file=sys.stderr)
            return False, None
        target_n = n_samples_override
    else:
        target_n = n_collected
        if n_collected >= sample_size and _analyzed(n_collected):
            print(f"⚠️  {backend_name}: all {sample_size} chunks collected and already analyzed — skipping merge/analyze.")
            return True, None

    noise = coherent_noise_of_run(run_path) or {}   # a coherent-noise run's circuits live in its own cache
    qbcs = bacq_ps_qbcs(
        infidelity_total=noise.get("infidelity_total", 0.0), ratio_ct=noise.get("ratio_ct", 0.0),
        noise_type=noise.get("noise_type", "pauli"),
        sample_size=sample_size, chunk_size=info.chunk_size,
        list_circuit_times=info.list_times, error_mitigation_strategies=info.ems,
        do_SPAM=info.do_spam, nshots=info.nshots, dt=info.dt,
    )
    qbcs.CONFIG["folder_path"]      = str(run_path.resolve())
    qbcs.CONFIG["lnn_architecture"] = info.lnn
    qbcs.TIMESTAMP_FILE             = run_path / TIMESTAMP_FILENAME
    qbcs._use_circuit_LNN           = info.lnn
    qbcs._status                    = "submitted"
    if is_sim:
        qbcs.CONFIG["service"] = "local"   # kept in CONFIG only; nothing reads it back

    # A real run's job records carry the CONFIG it was submitted with — it must agree with what we just rebuilt.
    rec0_path = run_path / "job_records" / "job_record_chunk_0.json"
    if rec0_path.exists():
        rec_cfg = json.loads(rec0_path.read_text(encoding="utf-8"))["CONFIG"]
        assert np.allclose(rec_cfg["T_list"], qbcs.CONFIG["T_list"])
        for k in ("dt", "chunk_size", "shots", "nsites", "g", "SPAM_correction",
                  "nec_measurement_counts", "lnn_architecture", "coherent_noise"):
            assert rec_cfg.get(k) == qbcs.CONFIG.get(k), (
                f"{backend_name}: CONFIG mismatch on '{k}' — run had {rec_cfg.get(k)!r}, "
                f"pipeline has {qbcs.CONFIG[k]!r}"
            )

    missing = uncollected_chunks(run_path, sample_size)
    if missing and n_samples_override is None:
        print(f"⚠️  {backend_name}: {len(missing)} chunk{'s' if len(missing) > 1 else ''} uncollected ({format_chunk_range(missing)}) — "
              f"analyzing the {n_collected} collected ones.")
    merged_path = run_path / "raw_results" / "results_counts_merged.json"
    if not merged_path.exists() or merged_path.stat().st_mtime < counts_path.stat().st_mtime:
        print(f"   {backend_name}: merging {n_collected} chunks ...")   # new chunks were collected since the last merge
        qbcs.merge_all_chunks()

    qbcs.analyze_results(n_samples=target_n)
    suffix = subset_suffix(target_n) if target_n < sample_size else ""
    print(f"✅ {backend_name}: analyzed {target_n}/{sample_size} chunks → run_data{suffix}.json")
    return True, (target_n if target_n < sample_size else None)


def analyzed_data(run_folder, n_samples_override=None):
    """(run_folder, n_samples) to plot for a run, as analyze_run() would return them (n_samples None = the
    full run's unsuffixed files), without analyzing anything — for --collective-only. Only checks that
    raw_results/results_counts.json exists; run_folder None if it does not (no collected results)."""
    backend_name = run_folder
    counts_path = PROJECT_ROOT / "datafiles" / run_folder / "raw_results" / "results_counts.json"
    if not counts_path.exists():
        print(f"⏭  {backend_name}: no collected results — left out of the collective plots")
        return None, None
    n = n_samples_override or len(json.loads(counts_path.read_text(encoding="utf-8")))
    return run_folder, (n if n < load_run_info(run_folder).sample_size else None)


def run_script(script_name, run_folder, n_samples, extra_env=None):
    """Stages 2-5 — run one of the source/analysis_scripts/ post-processing scripts, parameterized via env vars."""
    env = os.environ.copy()
    env.update(extra_env or {})
    env["QPU_RUN_FOLDER"] = run_folder   # the scripts read the run's own settings from it
    env["QPU_N_SAMPLES"]  = "" if n_samples is None else str(n_samples)
    result = subprocess.run(
        [venv_python("local_sim.venv", script_name), str(ANALYSIS_SCRIPTS / script_name)],
        cwd=PROJECT_ROOT, env=env, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        print(f"❌ {script_name} failed for {run_folder} (exit {result.returncode})", file=sys.stderr)
        return False
    return True


def copy_summary(run_folder, n_samples):
    """Copy the run's summary json to analysis/summaries/ (a test run's: analysis/tests/summaries/), where every
    run's summary is collected."""
    src = PROJECT_ROOT / "datafiles" / run_folder / "processed_results" / run_summary_name(n_samples)
    if src.exists():
        dest = PROJECT_ROOT / summaries_dir(run_folder)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / summary_name(run_folder, n_samples))


def analyze_run(run_folder, n_samples_override=None):
    """Stages 1-5 for one run. Returns (ok, run_folder, n_samples): ok is True if every
    stage that ran succeeded; run_folder/n_samples identify the analyzed benchmark data
    (run_folder None if there is none, e.g. a health-check-only run)."""
    backend_name = run_folder   # for the messages
    for msg in run_layout_warnings(PROJECT_ROOT / "datafiles" / run_folder):
        print(f"⚠️  {backend_name}: {msg}")

    # A run recorded (run_info.json) but never submitted — typically aborted because the
    # pre-submission health check found the QPU degraded: only its health checks exist.
    run_path = PROJECT_ROOT / "datafiles" / run_folder
    if not ((run_path / "job_records" / "job_record_chunk_0.json").exists()
            or (run_path / "raw_results" / "results_counts.json").exists()):
        checks = load_health_checks(run_path)
        if not checks:
            print(f"⏭  {backend_name}: no benchmark data and no health checks yet — skipping")
            return False, None, None
        if any(c.get("verdict") == "degraded" for c in checks):
            print(f"⚠️  {backend_name}: the health check found the QPU degraded, so the benchmark was not run "
                  f"— analyzing the health-check data only.")
        else:
            print(f"⚠️  {backend_name}: no benchmark data yet — analyzing the health-check data only.")
        return run_script("plot_health_checks.py", run_folder, None), None, None

    ok, n_samples = merge_and_analyze(run_folder, n_samples_override)
    if not ok:
        return False, None, None
    ok = all([
        run_script("bacq_ps_qbcs_summary_json.py", run_folder, n_samples),
        run_script("plot_real_qpu_results.py",     run_folder, n_samples),
        run_script("plot_awae.py",                 run_folder, n_samples),
        run_script("plot_health_checks.py",        run_folder, n_samples),   # skips runs without health checks
    ])
    copy_summary(run_folder, n_samples)
    return ok, run_folder, n_samples


def nsamples_comparison(run_folder, n_ref):
    """--nsamples-comparison for one already analyzed run: analyze it on each of NSAMPLES_COMPARISON
    samples below the reference (n_ref, None = the full run) — JSONs only — then plot its best EMS
    for every sample count, the reference included. Sample counts above the reference are skipped."""
    backend_name = run_folder   # for the messages
    run_path = PROJECT_ROOT / "datafiles" / run_folder
    info     = load_run_info(run_folder)
    ref_n    = n_ref if n_ref is not None else info.sample_size
    counts_mtime = (run_path / "raw_results" / "results_counts.json").stat().st_mtime
    done = []
    for n in NSAMPLES_COMPARISON:
        if n > ref_n:
            print(f"⚠️  {backend_name}: n_samples={n} > the {ref_n} analyzed samples — left out of the comparison")
            continue
        if n == ref_n:
            continue   # the reference, already analyzed
        files = [run_path / "processed_results" / f"run_data{subset_suffix(n)}.json",
                 run_path / "processed_results" / run_summary_name(n)]
        if all(f.exists() and f.stat().st_mtime > counts_mtime for f in files):
            print(f"   {backend_name}: n_samples={n} already analyzed")
        else:
            ok, _ = merge_and_analyze(run_folder, n)
            if not (ok and run_script("bacq_ps_qbcs_summary_json.py", run_folder, n)):
                continue
            copy_summary(run_folder, n)
        done.append(n)
    return run_script("plot_awae_nsamples_comparison.py", run_folder, n_ref,
                      extra_env={"QPU_NSAMPLES_LIST": ",".join(str(n) for n in done + [ref_n])})


def main():
    usage_if_requested(sys.argv[1:], USAGE)   # no arguments or --help: print the usage and do nothing
    args, kinds = parse_include(sys.argv[1:])
    args, run_filter = parse_filters(args)
    args, do_individual, do_collective = parse_scope(args)
    do_nsamples = "--nsamples-comparison" in args
    args = [a for a in args if a != "--nsamples-comparison"]
    mode, value = parse_mode(args)
    if do_nsamples and (mode, value) != ("preset", "full"):
        print(USAGE, file=sys.stderr)
        exit_with_error("--nsamples-comparison works with the full benchmark only.")
    if do_nsamples and not do_individual:
        print("--nsamples-comparison makes per-run plots only, and --collective-only skips them — flag ignored.")
        do_nsamples = False
    cfgs, benchmark = select_runs(mode, value)
    cfgs    = [c for c in cfgs if entry_kind(c) in kinds]   # only the selected kinds are looked up
    cfg_of  = {f: c for c in cfgs for f in resolve_cfg(c, benchmark)}   # resolved once; an entry matching several folders (layouts) includes all
    folders = sorted((f for f in cfg_of if run_kind(f) in kinds
                      and run_filter.matches(parse_run_name(f))), key=folder_order)   # one filter point: individual and collective alike
    print(f"Including: {', '.join(sorted(kinds))}")
    if not run_filter.is_empty:
        print(f"Filters: {run_filter.describe()}")
    label = {"nshots": f"nshots_per_circuit={value}", "preset": f"{value} benchmark",
             "tag": f"runs with benchmark {value}"}[mode]
    print(f"Mode: {label} — {len(folders)} run(s):")
    for f in folders:
        print(f"  {f}")
    if not folders:
        print("\n⚠️  No run matches the selection and filters — nothing to analyze.")
        if run_filter.backends:
            print("Backends with runs: " + ", ".join(sorted({parse_run_name(f).device for f in benchmark_runs()})))
        return

    analyzed = []   # (run_folder, n_samples) of every run with benchmark data, for the collective plots
    for run_folder in folders:
        n_override = cfg_of[run_folder].get("n_samples")
        if not do_individual:
            run_folder, n_samples = analyzed_data(run_folder, n_override)
            if run_folder is not None:
                analyzed.append((run_folder, n_samples))
            continue
        section(run_folder)
        try:
            _, done_folder, n_samples = analyze_run(run_folder, n_override)
            if done_folder is not None:
                analyzed.append((done_folder, n_samples))
                if do_nsamples:
                    nsamples_comparison(done_folder, n_samples)
        except Exception as e:
            print(f"❌ {run_folder}: merge/analyze failed — {e}", file=sys.stderr)

    if not do_collective:
        print("\n✅ Pipeline complete.")
        return
    section("Collective plots")
    selection = f"any+nshots-{value}" if mode == "nshots" else value
    data_types = [KIND_DATA_TYPES[k][0] for k in KIND_DATA_TYPES if k in kinds]
    try:
        collective_name = collective_folder_name(selection, data_types, run_filter)
        make_collective_plots(analyzed, PROJECT_ROOT / collective_plots_dir(collective_name),
                              PROJECT_ROOT / collective_table_path(collective_name))
    except Exception as e:
        print(f"❌ collective plots failed — {e}", file=sys.stderr)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
