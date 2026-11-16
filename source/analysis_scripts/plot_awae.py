"""
plot_awae.py
Plot AWAE and cumulative AWAE (full-run values, 95% bootstrap interval bars) for all EMS methods of one BACQ run.
Produces 3 figures × 2 subplots each (only the no-SPAM one if the run has do_SPAM=False).
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

from source.core.common import rel_path
from source.core.analysis.mitigation import BASE_EMS_LIST
from source.core.analysis.derived_quantities import EPSILON, best_schemes as summary_best_schemes
from source.core.records.run_info import resolve_run
from source.core.records.run_names import parse_run_name, run_summary_name, plots_dir as plots_dir_of, subset_suffix, plot_names
from source.core.records.circuit_check import load_run_verification, add_circuit_check_banner
from source.core.records.chunk_scope import (load_run_completeness, add_completeness_banner, completeness,
                                             scope_from_env, figure_filter_label)

# ── Config ────────────────────────────────────────────────────────────────────
RUN_FOLDER    = "old_runs_full/full_real-data_iqm-emerald_lnn-True_service-iqm-resonance_spam-True_layout-legacy_2026-09-07_#1"   # standalone: the run (datafiles/<RUN_FOLDER>)
ems_list     = BASE_EMS_LIST

N_SAMPLES    = None # None = full run (unchanged behaviour); int = plot that
                     # subset's run_data+nsamples-N and summary+nsamples-N instead;
                     # scope_from_env() can replace it with a health-check scope (+hc-gt-T)

# ── Optional overrides for source/analysis_scripts/run_post_process.py — standalone behaviour above is unchanged ──
if "QPU_N_SAMPLES" in os.environ:
    _v = os.environ["QPU_N_SAMPLES"]
    N_SAMPLES = None if _v in ("", "None") else int(_v)

RUN_FOLDER, INFO = resolve_run(RUN_FOLDER)   # QPU_RUN_FOLDER wins; the run's own settings win over the constants above
N_SAMPLES    = scope_from_env(RUN_FOLDER, INFO.sample_size, N_SAMPLES)   # --filter-per-health-check: the chunks passing the extra checks
NAME         = parse_run_name(RUN_FOLDER)
NSHOTS, LNN, DO_SPAM = INFO.nshots, INFO.lnn, INFO.do_spam
SUFFIX     = subset_suffix(N_SAMPLES)
PLOT_NAMES = plot_names(LNN, NSHOTS, N_SAMPLES)   # the figure names run_post_process.py looks for

# Determine actual depths from this run's own data, not the hardcoded
# standard 10-depth real-QPU sweep —
# see the same fix in plot_real_qpu_results.py / bacq_ps_qbcs_summary_json.py.
# run_data (no SPAM) always exists regardless of do_SPAM, so it's the one to
# peek at here; run_data_spam has identical depths when it exists too.
with open(f"datafiles/{RUN_FOLDER}/processed_results/run_data{SUFFIX}.json", encoding="utf-8") as _f:
    nt_list = sorted(int(k) for k in json.load(_f)["depths"].keys())

# ── Plot settings ─────────────────────────────────────────────────────────────
colors  = {"bare": "gray",      "NEC": "red",       "RC": "steelblue",
           "NEC+RC": "orange",  "cRC": "darkblue",  "NEC+cRC": "firebrick"}
markers = {"bare": "o", "NEC": "s", "RC": "^",
           "NEC+RC": "v", "cRC": "D", "NEC+cRC": "*"}

plots_dir = plots_dir_of(RUN_FOLDER)
plots_dir.mkdir(parents=True, exist_ok=True)

# ── Executed-vs-submitted circuit check: a red line on every figure if it failed ──
VERIFICATION = load_run_verification(Path("datafiles") / RUN_FOLDER, SUFFIX)
COMPLETENESS = load_run_completeness(Path("datafiles") / RUN_FOLDER, SUFFIX) or completeness(Path("datafiles") / RUN_FOLDER, INFO.sample_size, N_SAMPLES)   # uncollected chunks / health checks: red line on every figure

# ── Output suptitle suffix ────────────────────────────────────────────────────
suffix = f"{NAME.figure_label_full}, LNN={LNN}, shots={NSHOTS}"
if hasattr(N_SAMPLES, "threshold"):
    suffix += figure_filter_label(N_SAMPLES)
elif N_SAMPLES is not None:
    suffix += f", n_samples={N_SAMPLES}"


def _num(v):
    """A stored number, NaN for null."""
    return np.nan if v is None else float(v)


def load_awae(run_folder, use_spam):
    """Load the full-run AWAE and cumulative AWAE per scheme, with their 95% intervals, from run_data[_spam][+nsamples-N].json.
    Returns (awae, cumul, ci, nec_analysis): awae / cumul are {ems: array over nt_list}; ci is {ems: {"awae": [[lo, hi] or None per depth],
    "cumul": the same}} (None: no interval, bootstrap not done); nec_analysis is the file's "nec_analysis" block."""
    fname = f"run_data_spam{SUFFIX}.json" if use_spam else f"run_data{SUFFIX}.json"
    path  = os.path.join("datafiles", run_folder, "processed_results", fname)
    with open(path, encoding="utf-8") as f:
        run_data = json.load(f)
    depths = run_data["depths"]

    awae_Nt       = {ems: [] for ems in ems_list}
    awae_cumul_Nt = {ems: [] for ems in ems_list}
    ci            = {ems: {"awae": [], "cumul": []} for ems in ems_list}

    for nt in nt_list:
        depth_data = depths[str(nt)]
        for ems in ems_list:
            awae_Nt[ems].append(_num(depth_data[ems]["AWAE"]))
            awae_cumul_Nt[ems].append(_num(depth_data[ems]["AWAE_cumul"]))
            ci[ems]["awae"].append(depth_data[ems].get("AWAE_ci95"))
            ci[ems]["cumul"].append(depth_data[ems].get("AWAE_cumul_ci95"))

    for ems in ems_list:
        awae_Nt[ems] = np.array(awae_Nt[ems])
        awae_cumul_Nt[ems] = np.array(awae_cumul_Nt[ems])
    return awae_Nt, awae_cumul_Nt, ci, run_data.get("nec_analysis") or {}


def draw_bars(ax, x, intervals, color, alpha=0.6):
    """95% interval bars (vlines; the full-run value can lie outside its interval, so no errorbar yerr); a null interval draws none."""
    for xi, iv in zip(x, intervals):
        if iv is not None and iv[0] is not None and iv[1] is not None:
            ax.vlines(xi, iv[0], iv[1], color=color, lw=1.5, alpha=alpha)


def plot_awae_subplots(ax_awae, ax_cumul, awae_Nt, awae_cumul_Nt, ci=None,
                       ems_subset=None, label_suffix="", linestyle="-"):
    """Plot AWAE and cumulative AWAE (with their 95% interval bars from ci, when given) on given axes."""
    ems_to_plot = ems_subset if ems_subset else ems_list
    for ems in ems_to_plot:
        label = f"{ems} {label_suffix}".strip()
        ax_awae.plot(nt_list, awae_Nt[ems],
                     color=colors[ems], marker=markers[ems],
                     ls=linestyle, lw=1.5, ms=5, label=label)
        ax_cumul.plot(nt_list, awae_cumul_Nt[ems],
                      color=colors[ems], marker=markers[ems],
                      ls=linestyle, lw=1.5, ms=5, label=label)
        if ci is not None:
            draw_bars(ax_awae, nt_list, ci[ems]["awae"], colors[ems])
            draw_bars(ax_cumul, nt_list, ci[ems]["cumul"], colors[ems])

    for ax in [ax_awae, ax_cumul]:
        ax.set_xlabel("$N_t$")
        ax.set_xticks(nt_list)
        # ax.set_ylim(0, 0.05)
        ax.grid(True, alpha=0.3)

    ax_awae.set_ylabel("AWAE")
    ax_cumul.set_ylabel("Cumulative AWAE")
    ax_awae.set_title("AWAE vs $N_t$")
    ax_cumul.set_title("Cumulative AWAE vs $N_t$")


def add_epsilon_line(ax):
    """Draw the accuracy-threshold line once on an axis (call at most once per axis)."""
    ax.axhline(EPSILON, color="black", ls="--", lw=1, alpha=0.5,
               label=rf"Acc threshold $\varepsilon$={EPSILON}")


# ── Load data ─────────────────────────────────────────────────────────────────
if DO_SPAM:
    awae_spam, cumul_spam, ci_spam, _ = load_awae(RUN_FOLDER, use_spam=True)
else:
    awae_spam = cumul_spam = ci_spam = None   # no run_data_spam.json: the SPAM figures are skipped
    print("  do_SPAM=False: only the no-SPAM figure is produced")
awae_no_spam, cumul_no_spam, ci_no_spam, NEC_INFO = load_awae(RUN_FOLDER, use_spam=False)
BOOTSTRAP_DONE = bool(NEC_INFO.get("bootstrap_done"))
NO_INTERVAL_NOTE = NEC_INFO.get("bootstrap_skipped_because") or "no interval: fewer than 5 samples per input state"

# ── Load the run's summary for best_ems and N_acc ─────────────────────────────────
summary_path = f"datafiles/{RUN_FOLDER}/processed_results/{run_summary_name(N_SAMPLES)}"
with open(summary_path, encoding="utf-8") as f:
    summary = json.load(f)
best_ems      = summary["best_ems_Nmax"]
best_ems_nacc = summary.get("best_ems_Nacc", best_ems)
N_acc         = summary["N_acc"]
N_acc_strict  = summary.get("N_acc_strict")   # None: no bootstrap, so undefined
NACC_STRICT_TXT = "n/a" if N_acc_strict is None else str(N_acc_strict)
print(f"  Best EMS: {best_ems} (N_max), {best_ems_nacc} (N_acc), N_acc: {N_acc}, N_acc_strict: {NACC_STRICT_TXT}")


def add_nacc_lines(ax):
    """The N_acc line (always) and the N_acc_strict line (when it differs); the legend prints both values."""
    ax.axvline(N_acc, color="purple", ls=":", lw=1.5, alpha=0.7, label=f"$N_{{acc}}$={N_acc}")
    if N_acc_strict is not None and N_acc_strict != N_acc:
        ax.axvline(N_acc_strict, color="green", ls="-.", lw=1.5, alpha=0.7, label=f"$N_{{acc,strict}}$={N_acc_strict}")
    else:
        ax.plot([], [], " ", label=f"$N_{{acc,strict}}$={NACC_STRICT_TXT}")   # legend-only entry


def add_interval_note(fig):
    """A note under the figure when the run has no bootstrap intervals."""
    if not BOOTSTRAP_DONE:
        fig.text(0.5, 0.0, NO_INTERVAL_NOTE, ha="center", va="top", fontsize=8, color="firebrick")

# Figure 3 draws each distinct best scheme (N_max's, and N_acc's when it differs) from
# summary_best_schemes(), with the "+SPAM" suffix stripped, from both awae_spam and awae_no_spam.
# the distinct best schemes by bare name ([] when none beats 1)
best_schemes = []
for _tag, _ems in summary_best_schemes(summary):
    _base = _ems.removesuffix("+SPAM")
    if all(_base != b for _, b in best_schemes):
        best_schemes.append((_tag, _base))

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — WITH SPAM: all EMS
# ─────────────────────────────────────────────────────────────────────────────
if DO_SPAM:
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plot_awae_subplots(ax1a, ax1b, awae_spam, cumul_spam, ci_spam)
    add_epsilon_line(ax1b)
    add_nacc_lines(ax1a)
    add_nacc_lines(ax1b)
    ax1a.legend(fontsize=7, ncol=2)
    ax1b.legend(fontsize=7, ncol=2)
    fig1.suptitle(f"AWAE with SPAM correction\n{suffix}", fontweight="bold", fontsize=11)
    plt.tight_layout()
    add_interval_note(fig1); add_circuit_check_banner(fig1, VERIFICATION); add_completeness_banner(fig1, COMPLETENESS)
    fname1 = plots_dir / PLOT_NAMES["awae_spam"]
    plt.savefig(fname1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved → {rel_path(fname1)}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — WITHOUT SPAM: all EMS
# ─────────────────────────────────────────────────────────────────────────────
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
plot_awae_subplots(ax2a, ax2b, awae_no_spam, cumul_no_spam, ci_no_spam)
add_epsilon_line(ax2b)
add_nacc_lines(ax2a)
add_nacc_lines(ax2b)
ax2a.legend(fontsize=7, ncol=2)
ax2b.legend(fontsize=7, ncol=2)
fig2.suptitle(f"AWAE without SPAM correction\n{suffix}", fontweight="bold", fontsize=11)
plt.tight_layout()
add_interval_note(fig2); add_circuit_check_banner(fig2, VERIFICATION); add_completeness_banner(fig2, COMPLETENESS)
fname2 = plots_dir / PLOT_NAMES["awae_no_spam"]
plt.savefig(fname2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved → {rel_path(fname2)}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — BEST EMS: with vs without SPAM
# The best scheme for N_max and the best scheme for N_acc need not be the same,
# so this plots each distinct scheme (up to 3 of them: N_max, N_acc, N_acc_strict) in both its SPAM and
# no-SPAM variant — up to 6 lines total per subplot.
# ─────────────────────────────────────────────────────────────────────────────
if DO_SPAM:
    # best_schemes is empty when no scheme's cumulative AWAE beats 1 (derived_quantities)
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    if not best_schemes:
        msg = "No error-mitigation scheme reaches cumulative AWAE < 1"
        print(f"⚠️  {msg} — saving an empty best-EMS figure")
        for ax in (ax3a, ax3b):
            ax.axis("off")
        fig3.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14, color="firebrick")
    for tag, ems_base in best_schemes:
        plot_awae_subplots(ax3a, ax3b, awae_spam, cumul_spam, ci_spam,
                           ems_subset=[ems_base], label_suffix=f"({tag} best, SPAM)", linestyle="-")
        plot_awae_subplots(ax3a, ax3b, awae_no_spam, cumul_no_spam, ci_no_spam,
                           ems_subset=[ems_base], label_suffix=f"({tag} best, no SPAM)", linestyle="--")
    if best_schemes:
        add_epsilon_line(ax3b)
        add_nacc_lines(ax3a)
        add_nacc_lines(ax3b)
        ax3a.legend(fontsize=8)
        ax3b.legend(fontsize=8)
    title_ems = (" / ".join(f"{ems} ({tag})" for tag, ems in best_schemes) if len(best_schemes) > 1
                 else best_schemes[0][1] if best_schemes else "none")
    fig3.suptitle(f"AWAE — best EMS ({title_ems}): SPAM vs no SPAM\n{suffix}",
                  fontweight="bold", fontsize=11)
    plt.tight_layout()
    add_interval_note(fig3); add_circuit_check_banner(fig3, VERIFICATION); add_completeness_banner(fig3, COMPLETENESS)
    fname3 = plots_dir / PLOT_NAMES["awae_best_ems"]
    plt.savefig(fname3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved → {rel_path(fname3)}")
