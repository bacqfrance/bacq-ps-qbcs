"""
plot_real_qpu_results.py
Plot the observables and their maximum errors for one BACQ run (real QPU or
local simulator). Needs only the run folder: no myQLM, no backend connection.
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
from source.core.records.run_info import resolve_run
from source.core.records.run_names import parse_run_name, plots_dir as plots_dir_of, subset_suffix
from source.core.records.circuit_check import load_run_verification, add_circuit_check_banner
from source.core.records.chunk_scope import load_run_completeness, add_completeness_banner, completeness

# ── Config (standalone use; source/analysis_scripts/run_post_process.py overrides it via env vars) ──
RUN_FOLDER    = "old_runs_full/full_real-data_iqm-emerald_lnn-True_service-iqm-resonance_spam-True_layout-legacy_2026-09-07_#1"   # standalone: the run (datafiles/<RUN_FOLDER>)

N_SAMPLES    = None # None = full run (unchanged behaviour); int = plot that subset's run_data+nsamples-N instead

# ── Optional overrides for source/analysis_scripts/run_post_process.py — standalone behaviour above is unchanged ──
if "QPU_N_SAMPLES" in os.environ:
    _v = os.environ["QPU_N_SAMPLES"]
    N_SAMPLES = None if _v in ("", "None") else int(_v)
RUN_FOLDER, INFO = resolve_run(RUN_FOLDER)   # QPU_RUN_FOLDER wins; the run's own settings win over the constants above
NAME         = parse_run_name(RUN_FOLDER)
NSHOTS, LNN, DO_SPAM = INFO.nshots, INFO.lnn, INFO.do_spam
PATH       = f"datafiles/{RUN_FOLDER}/processed_results"
SUFFIX     = subset_suffix(N_SAMPLES)
# Executed-vs-submitted circuit check: a red line on every figure if it failed
VERIFICATION = load_run_verification(f"datafiles/{RUN_FOLDER}", SUFFIX)
COMPLETENESS = load_run_completeness(Path("datafiles") / RUN_FOLDER, SUFFIX) or completeness(Path("datafiles") / RUN_FOLDER, INFO.sample_size, N_SAMPLES)   # uncollected chunks / health checks: red line on every figure
PLOTS_DIR  = plots_dir_of(RUN_FOLDER).as_posix()
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Determine actual depths from this run's own data, not the hardcoded
# standard 10-depth real-QPU sweep —
# a run with different depths (e.g. a reduced local-sim test) would otherwise
# raise KeyError on depths[str(nt)] for any Nt this run never built.
with open(f"{PATH}/run_data{SUFFIX}.json", encoding="utf-8") as _f:
    _depths = json.load(_f)["depths"]
nt_list = sorted(int(k) for k in _depths.keys())

# ── Exact observables: the reference values analyze_results() stored in run_data (from the
# cache build() writes), so plotting needs no myQLM ─────────────────────────────
exact = {nt: np.array(_depths[str(nt)]["perfect_circuit_obs_values"]) for nt in nt_list}

# ── Plot settings ─────────────────────────────────────────────────────────────
obs_labels   = ["X₀", "Y₁", "Z₂", "X₀Y₁", "X₀Z₂", "Y₁Z₂", "X₀Y₁Z₂"]
roles_map    = {"bare": "bare", "NEC": "NEC", "RC": "RC",
                "NEC+RC": "NEC+RC", "cRC": "cRC", "NEC+cRC": "NEC+cRC"}
colors_role  = {"bare": "gray",    "NEC": "red",       "RC": "steelblue",
                "NEC+RC": "orange", "cRC": "darkblue", "NEC+cRC": "firebrick"}
markers_role = {"bare": "o", "NEC": "s", "RC": "^",
                "NEC+RC": "v", "cRC": "D", "NEC+cRC": "*"}


def plot_obs(run_data_path, title_suffix, fname_suffix):
    if not os.path.exists(run_data_path):
        print(f"  ⚠️  Not found: {rel_path(run_data_path)} — skipping.")
        return

    with open(run_data_path, encoding="utf-8") as f:
        ibm_results = json.load(f)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=False)
    axes_flat  = axes.flatten()

    for oi in range(7):
        ax = axes_flat[oi]

        # Exact
        ax.plot(nt_list, [exact[nt][oi] for nt in nt_list],
                "s", lw=2.5, label="exact", zorder=0,
                ms=8.5, alpha=1.0, color="lime")

        # IBM results per role
        for label, display in roles_map.items():
            if label not in ibm_results["depths"][str(nt_list[0])]:
                continue
            vals = [np.array(ibm_results["depths"][str(nt)][label]["obs_values"])[oi]
                    for nt in nt_list]
            ms = 10 if label == "bare" else 6
            ax.plot(nt_list, vals,
                    color=colors_role.get(display, "gray"),
                    marker=markers_role.get(display, "o"),
                    ls="--", lw=1.5, ms=ms, label=display, alpha=0.85)

        ax.set_title(obs_labels[oi], fontsize=10, fontweight="bold")
        ax.set_xlabel("$N_t$")
        ax.set_xticks(nt_list)
        ax.grid(True, alpha=0.3)
        if oi in [0, 4]:
            ax.set_ylabel("Expectation value")

    axes_flat[-1].set_visible(False)
    handles, labels_leg = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower right", fontsize=7,
               ncol=2, bbox_to_anchor=(1.0, 0.05))
    fig.suptitle(
        f"{NAME.figure_label_full} — {title_suffix}\n"
        f"LNN={LNN}, shots={NSHOTS}, {ibm_results.get('nchunks', '?')} chunks",
        fontweight="bold", fontsize=11
    )
    plt.tight_layout()
    add_circuit_check_banner(fig, VERIFICATION); add_completeness_banner(fig, COMPLETENESS)
    plt.savefig(fname_suffix, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Plot saved → {rel_path(fname_suffix)}")


# ── Plot without SPAM correction ──────────────────────────────────────────────
plot_obs(
    run_data_path = f"{PATH}/run_data{SUFFIX}.json",
    title_suffix  = "no SPAM correction",
    fname_suffix  = f"{PLOTS_DIR}/obs_lnn{LNN}_ns{NSHOTS}_no_spam{SUFFIX}.png",
)

# ── Plot with SPAM correction ─────────────────────────────────────────────────
if DO_SPAM:
    plot_obs(
        run_data_path = f"{PATH}/run_data_spam{SUFFIX}.json",
        title_suffix  = "with SPAM correction",
        fname_suffix  = f"{PLOTS_DIR}/obs_lnn{LNN}_ns{NSHOTS}_spam{SUFFIX}.png",
    )


# ── Plot max |obs - exact| per role per Nt — SPAM vs no SPAM ─────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=False)
axes_flat  = axes.flatten()

for ax, (role_key, role_label) in zip(axes_flat, roles_map.items()):
    for fname, label, color, ls in [
        (f"{PATH}/run_data{SUFFIX}.json",      "no SPAM", "steelblue", "-"),
        (f"{PATH}/run_data_spam{SUFFIX}.json", "SPAM",    "tomato",    "--"),
    ]:
        if not os.path.exists(fname):
            continue
        with open(fname, encoding="utf-8") as f:
            ibm_results = json.load(f)
        diffs = []
        for nt in nt_list:
            if role_label not in ibm_results["depths"][str(nt)]:
                diffs.append(np.nan)
                continue
            ibm_obs   = np.array(ibm_results["depths"][str(nt)][role_label]["obs_values"])
            exact_obs = exact[nt]
            diffs.append(np.max(np.abs(ibm_obs - exact_obs)))
        ax.plot(nt_list, diffs, color=color, ls=ls, marker="o",
                lw=1.5, ms=5, label=label)

    ax.set_xlabel("$N_t$")
    ax.set_xticks(nt_list)
    ax.grid(True, alpha=0.3)
    ax.set_title(role_label, fontsize=10, fontweight="bold")
    if ax in axes[:, 0]:
        ax.set_ylabel("max |obs - exact|")


handles, labels_leg = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc="lower right", fontsize=9,
           ncol=2, bbox_to_anchor=(1.0, 0.05))
fig.suptitle(f"{NAME.figure_label_full} — max|obs-exact| per role\n"
             f"LNN={LNN}, shots={NSHOTS}",
             fontweight="bold", fontsize=11)
plt.tight_layout()
fname_plot = f"{PLOTS_DIR}/max_error_lnn{LNN}_ns{NSHOTS}{SUFFIX}.png"
add_circuit_check_banner(fig, VERIFICATION); add_completeness_banner(fig, COMPLETENESS)
plt.savefig(fname_plot, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Error plot saved → {rel_path(fname_plot)}")