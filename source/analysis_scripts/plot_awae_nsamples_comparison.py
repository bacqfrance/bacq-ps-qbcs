"""
plot_awae_nsamples_comparison.py
AWAE and cumulative AWAE of one run's best EMS, analyzed on different numbers of
RC samples (chunks; the SPAM correction of every analysis uses all collected chunks' calibration) — the n_samples analogue of plot_awae.py's
awae_best_ems_spam_no_spam_comparison figure.

Driven by source/analysis_scripts/run_post_process.py --nsamples-comparison via env vars:
  QPU_RUN_FOLDER      the run, as for the other post-processing scripts
  QPU_N_SAMPLES       the reference analysis ("" = the full run): its best EMS
                      (at N_max and at N_acc) and its N_acc are the ones plotted
  QPU_NSAMPLES_LIST   comma-separated sample counts to compare, e.g. 8,48,120,240,480;
                      each needs its run_data[_spam][+nsamples-N].json already analyzed
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
from source.core.analysis.derived_quantities import EPSILON, cumulative_awae, best_schemes
from source.core.records.run_info import resolve_run
from source.core.records.run_names import parse_run_name, run_summary_name, plots_dir as plots_dir_of, subset_suffix
from source.core.records.circuit_check import load_run_verification, add_circuit_check_banner
from source.core.records.chunk_scope import load_run_completeness, add_completeness_banner, completeness

# ── Config ────────────────────────────────────────────────────────────────────
RUN_FOLDER    = "old_runs_full/full_real-data_iqm-emerald_lnn-True_service-iqm-resonance_spam-True_layout-legacy_2026-09-07_#1"   # standalone: the run (datafiles/<RUN_FOLDER>)
N_SAMPLES     = None                     # reference analysis; None = full run
NSAMPLES_LIST = [8, 48, 120, 240, 480]

# ── Overrides from source/analysis_scripts/run_post_process.py ───────────────────────────────
if "QPU_N_SAMPLES" in os.environ:
    _v = os.environ["QPU_N_SAMPLES"]
    N_SAMPLES = None if _v in ("", "None") else int(_v)
if os.environ.get("QPU_NSAMPLES_LIST"):
    NSAMPLES_LIST = [int(n) for n in os.environ["QPU_NSAMPLES_LIST"].split(",")]

RUN_FOLDER, INFO = resolve_run(RUN_FOLDER)   # QPU_RUN_FOLDER wins; the run's own settings win over the constants above
NAME         = parse_run_name(RUN_FOLDER)
NSHOTS, LNN   = INFO.nshots, INFO.lnn
REF_N         = N_SAMPLES if N_SAMPLES is not None else INFO.sample_size
REF_SUFFIX    = subset_suffix(N_SAMPLES)
NSAMPLES_LIST = sorted(set(n for n in NSAMPLES_LIST if n <= REF_N) | {REF_N})

processed = Path("datafiles") / RUN_FOLDER / "processed_results"
plots_dir = plots_dir_of(RUN_FOLDER)
plots_dir.mkdir(parents=True, exist_ok=True)

VERIFICATION = load_run_verification(Path("datafiles") / RUN_FOLDER, REF_SUFFIX)
COMPLETENESS = load_run_completeness(Path("datafiles") / RUN_FOLDER, REF_SUFFIX) or completeness(Path("datafiles") / RUN_FOLDER, INFO.sample_size, N_SAMPLES)


def suffix_of(n):
    """File suffix of an analysis on n samples: none for the full run."""
    return "" if n >= INFO.sample_size else subset_suffix(n)


def load_awae(n, ems_full):
    """(nt_list, AWAE, cumulative AWAE) of one scheme ("NEC+cRC" or "NEC+cRC+SPAM") analyzed on n samples."""
    use_spam = ems_full.endswith("+SPAM")
    ems      = ems_full.removesuffix("+SPAM")
    fname    = f"run_data{'_spam' if use_spam else ''}{suffix_of(n)}.json"
    with open(processed / fname, encoding="utf-8") as f:
        depths = json.load(f)["depths"]
    nt_list = sorted(int(k) for k in depths)
    Z    = np.array([depths[str(nt)]["Z_exact"] for nt in nt_list])
    awae = np.array([depths[str(nt)][ems]["AWAE"] for nt in nt_list])
    cumul = cumulative_awae(Z, awae)
    return nt_list, awae, cumul


def best_ems_of(n):
    """The best EMS at N_max chosen by the analysis on n samples (None if its summary is missing)."""
    path = processed / run_summary_name(n if n < INFO.sample_size else None)
    return json.loads(path.read_text(encoding="utf-8"))["best_ems_Nmax"] if path.exists() else None


# ── Reference: best EMS (N_max, N_acc) and N_acc of the run's own analysis ─────
with open(processed / run_summary_name(N_SAMPLES), encoding="utf-8") as f:
    summary = json.load(f)
best_ems      = summary["best_ems_Nmax"]
best_ems_nacc = summary.get("best_ems_Nacc", best_ems)
N_acc         = summary["N_acc"]
schemes       = best_schemes(summary)
if not schemes:   # no scheme has a cumulative AWAE below 1 (derived_quantities)
    msg = "No error-mitigation scheme reaches cumulative AWAE < 1"
    print(f"⚠️  {run_summary_name(N_SAMPLES)} names no best EMS: {msg} — saving an empty figure")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14, color="firebrick")
    fig.suptitle(f"AWAE — best EMS: number of samples\n{NAME.figure_label_full}, LNN={LNN}, shots={NSHOTS}",
                 fontweight="bold", fontsize=11)
    add_circuit_check_banner(fig, VERIFICATION); add_completeness_banner(fig, COMPLETENESS)
    fname = plots_dir / "awae_best_ems_nsamples_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved → {rel_path(fname)}")
    sys.exit(0)
print(f"  Best EMS ({REF_N} samples): {best_ems} (N_max), {best_ems_nacc} (N_acc), N_acc: {N_acc}")

# ── Figure ────────────────────────────────────────────────────────────────────
cmap      = plt.get_cmap("viridis")
colors    = {n: cmap(0.9 * i / max(len(NSAMPLES_LIST) - 1, 1)) for i, n in enumerate(reversed(NSAMPLES_LIST))}
linestyle = {"N_max": "-", "N_acc": "--"}
markers   = {"N_max": "o", "N_acc": "s"}

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
nt_ticks = []
for n in NSAMPLES_LIST:
    own_best = best_ems_of(n)
    for tag, ems in schemes:
        try:
            nt_list, awae, cumul = load_awae(n, ems)
        except FileNotFoundError as e:
            print(f"⚠️  n_samples={n}: {e.filename} missing — curve left out")
            continue
        nt_ticks = nt_list
        label = f"n_samples={n}" + (" (reference)" if n == REF_N else "") + (f", {tag} best" if len(schemes) > 1 else "")
        if tag == "N_max" and own_best is not None and own_best != best_ems:
            label += f" [own best: {own_best}]"
        for ax, y in ((ax_a, awae), (ax_b, cumul)):
            ax.plot(nt_list, y, color=colors[n], marker=markers[tag], ls=linestyle[tag],
                    lw=2.2 if n == REF_N else 1.5, ms=5, label=label)

for ax in (ax_a, ax_b):
    ax.axvline(N_acc, color="purple", ls=":", lw=1.5, alpha=0.7, label=f"$N_{{acc}}$={N_acc} ({REF_N} samples)")
    ax.set_xlabel("$N_t$")
    ax.set_xticks(nt_ticks)
    ax.grid(True, alpha=0.3)
ax_b.axhline(EPSILON, color="black", ls="--", lw=1, alpha=0.5, label=rf"Acc threshold $\varepsilon$={EPSILON}")
ax_a.set_ylabel("AWAE");            ax_a.set_title("AWAE vs $N_t$")
ax_b.set_ylabel("Cumulative AWAE"); ax_b.set_title("Cumulative AWAE vs $N_t$")
ax_a.legend(fontsize=7)
ax_b.legend(fontsize=7)

title_ems = schemes[0][1] if len(schemes) == 1 else f"{best_ems} (N_max) / {best_ems_nacc} (N_acc)"
fig.suptitle(f"AWAE — best EMS ({title_ems}): number of samples\n{NAME.figure_label_full}, LNN={LNN}, shots={NSHOTS}",
             fontweight="bold", fontsize=11)
plt.tight_layout()
add_circuit_check_banner(fig, VERIFICATION); add_completeness_banner(fig, COMPLETENESS)
fname = plots_dir / "awae_best_ems_nsamples_comparison.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved → {rel_path(fname)}")
