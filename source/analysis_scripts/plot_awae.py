"""
plot_awae.py
Plot AWAE and cumulative AWAE for all EMS methods of one BACQ run.
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
from source.core.analysis.derived_quantities import EPSILON, cumulative_awae, best_schemes as summary_best_schemes
from source.core.records.run_info import resolve_run
from source.core.records.run_names import parse_run_name, run_summary_name, plots_dir as plots_dir_of, subset_suffix
from source.core.records.circuit_check import load_run_verification, add_circuit_check_banner
from source.core.records.chunk_scope import load_run_completeness, add_completeness_banner, completeness

# ── Config ────────────────────────────────────────────────────────────────────
RUN_FOLDER    = "old_runs_full/full_real-data_iqm-emerald_lnn-True_service-iqm-resonance_spam-True_layout-legacy_2026-09-07_#1"   # standalone: the run (datafiles/<RUN_FOLDER>)
ems_list     = BASE_EMS_LIST

N_SAMPLES    = None # None = full run (unchanged behaviour); int = plot that
                     # subset's run_data+nsamples-N and summary+nsamples-N instead

# ── Optional overrides for source/analysis_scripts/run_post_process.py — standalone behaviour above is unchanged ──
if "QPU_N_SAMPLES" in os.environ:
    _v = os.environ["QPU_N_SAMPLES"]
    N_SAMPLES = None if _v in ("", "None") else int(_v)

RUN_FOLDER, INFO = resolve_run(RUN_FOLDER)   # QPU_RUN_FOLDER wins; the run's own settings win over the constants above
NAME         = parse_run_name(RUN_FOLDER)
NSHOTS, LNN, DO_SPAM = INFO.nshots, INFO.lnn, INFO.do_spam
SUFFIX     = subset_suffix(N_SAMPLES)

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
if N_SAMPLES is not None:
    suffix += f", n_samples={N_SAMPLES}"


def load_awae(run_folder, use_spam):
    """Load AWAE and cumulative AWAE from run_data[_spam][+nsamples-N].json."""
    fname = f"run_data_spam{SUFFIX}.json" if use_spam else f"run_data{SUFFIX}.json"
    path  = os.path.join("datafiles", run_folder, "processed_results", fname)
    with open(path, encoding="utf-8") as f:
        run_data = json.load(f)
    depths = run_data["depths"]

    awae_Nt       = {ems: [] for ems in ems_list}
    awae_cumul_Nt = {ems: [] for ems in ems_list}
    Z_Nt          = []

    for nt in nt_list:
        depth_data = depths[str(nt)]
        Z_Nt.append(depth_data["Z_exact"])
        for ems in ems_list:
            awae_Nt[ems].append(depth_data[ems]["AWAE"])

    Z_Nt = np.array(Z_Nt)
    for ems in ems_list:
        awae_Nt[ems] = np.array(awae_Nt[ems])
        awae_cumul_Nt[ems] = cumulative_awae(Z_Nt, awae_Nt[ems])
    return awae_Nt, awae_cumul_Nt


def plot_awae_subplots(ax_awae, ax_cumul, awae_Nt, awae_cumul_Nt,
                       ems_subset=None, label_suffix="", linestyle="-"):
    """Plot AWAE and cumulative AWAE on given axes."""
    ems_to_plot = ems_subset if ems_subset else ems_list
    for ems in ems_to_plot:
        label = f"{ems} {label_suffix}".strip()
        ax_awae.plot(nt_list, awae_Nt[ems],
                     color=colors[ems], marker=markers[ems],
                     ls=linestyle, lw=1.5, ms=5, label=label)
        ax_cumul.plot(nt_list, awae_cumul_Nt[ems],
                      color=colors[ems], marker=markers[ems],
                      ls=linestyle, lw=1.5, ms=5, label=label)

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
    awae_spam, cumul_spam = load_awae(RUN_FOLDER, use_spam=True)
else:
    awae_spam = cumul_spam = None   # no run_data_spam.json: the SPAM figures are skipped
    print("  do_SPAM=False: only the no-SPAM figure is produced")
awae_no_spam, cumul_no_spam = load_awae(RUN_FOLDER, use_spam=False)

# ── Load the run's summary for best_ems and N_acc ─────────────────────────────────
summary_path = f"datafiles/{RUN_FOLDER}/processed_results/{run_summary_name(N_SAMPLES)}"
with open(summary_path, encoding="utf-8") as f:
    summary = json.load(f)
best_ems      = summary["best_ems_Nmax"]
best_ems_nacc = summary.get("best_ems_Nacc", best_ems)
N_acc         = summary["N_acc"]
print(f"  Best EMS: {best_ems} (N_max), {best_ems_nacc} (N_acc), N_acc: {N_acc}")

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
    plot_awae_subplots(ax1a, ax1b, awae_spam, cumul_spam)
    add_epsilon_line(ax1b)
    ax1a.axvline(N_acc, color="purple", ls=":", lw=1.5, alpha=0.7, label=f"$N_{{acc}}$={N_acc}")
    ax1b.axvline(N_acc, color="purple", ls=":", lw=1.5, alpha=0.7, label=f"$N_{{acc}}$={N_acc}")
    ax1a.legend(fontsize=7, ncol=2)
    ax1b.legend(fontsize=7, ncol=2)
    fig1.suptitle(f"AWAE with SPAM correction\n{suffix}", fontweight="bold", fontsize=11)
    plt.tight_layout()
    add_circuit_check_banner(fig1, VERIFICATION); add_completeness_banner(fig1, COMPLETENESS)
    fname1 = plots_dir / f"awae_spam{SUFFIX}.png"
    plt.savefig(fname1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved → {rel_path(fname1)}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — WITHOUT SPAM: all EMS
# ─────────────────────────────────────────────────────────────────────────────
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
plot_awae_subplots(ax2a, ax2b, awae_no_spam, cumul_no_spam)
add_epsilon_line(ax2b)
ax2a.axvline(N_acc, color="purple", ls=":", lw=1.5, alpha=0.7, label=f"$N_{{acc}}$={N_acc}")
ax2b.axvline(N_acc, color="purple", ls=":", lw=1.5, alpha=0.7, label=f"$N_{{acc}}$={N_acc}")
ax2a.legend(fontsize=7, ncol=2)
ax2b.legend(fontsize=7, ncol=2)
fig2.suptitle(f"AWAE without SPAM correction\n{suffix}", fontweight="bold", fontsize=11)
plt.tight_layout()
add_circuit_check_banner(fig2, VERIFICATION); add_completeness_banner(fig2, COMPLETENESS)
fname2 = plots_dir / f"awae_no_spam{SUFFIX}.png"
plt.savefig(fname2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved → {rel_path(fname2)}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — BEST EMS: with vs without SPAM
# The best scheme for N_max and the best scheme for N_acc need not be the same,
# so this plots each distinct scheme (up to 2 of them) in both its SPAM and
# no-SPAM variant — up to 4 lines total per subplot.
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
        plot_awae_subplots(ax3a, ax3b, awae_spam, cumul_spam,
                           ems_subset=[ems_base], label_suffix=f"({tag} best, SPAM)", linestyle="-")
        plot_awae_subplots(ax3a, ax3b, awae_no_spam, cumul_no_spam,
                           ems_subset=[ems_base], label_suffix=f"({tag} best, no SPAM)", linestyle="--")
    if best_schemes:
        add_epsilon_line(ax3b)
        ax3a.axvline(N_acc, color="purple", ls=":", lw=1.5, alpha=0.7, label=f"$N_{{acc}}$={N_acc}")
        ax3b.axvline(N_acc, color="purple", ls=":", lw=1.5, alpha=0.7, label=f"$N_{{acc}}$={N_acc}")
        ax3a.legend(fontsize=8)
        ax3b.legend(fontsize=8)
    title_ems = (" / ".join(f"{ems} ({tag})" for tag, ems in best_schemes) if len(best_schemes) > 1
                 else best_schemes[0][1] if best_schemes else "none")
    fig3.suptitle(f"AWAE — best EMS ({title_ems}): SPAM vs no SPAM\n{suffix}",
                  fontweight="bold", fontsize=11)
    plt.tight_layout()
    add_circuit_check_banner(fig3, VERIFICATION); add_completeness_banner(fig3, COMPLETENESS)
    fname3 = plots_dir / f"awae_best_ems_spam_no_spam_comparison{SUFFIX}.png"
    plt.savefig(fname3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved → {rel_path(fname3)}")
