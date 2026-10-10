"""
collective_plots.py
Plots comparing all runs analyzed by source/analysis_scripts/run_post_process.py (stage 6):
  awae_best_ems_vs_Nt.png        AWAE vs N_t of each run's best EMS (for N_max and for N_acc: up to two per run)
  cumul_awae_best_ems_vs_Nt.png  the same for the cumulative AWAE (Z_exact-weighted mean up to N_t,
                                 as in source/analysis_scripts/plot_awae.py — the quantity N_acc is defined by)
  awae_and_cumul_awae_best_ems_vs_Nt.png  both side by side
  qpu_time.png              QPU time per run (log scale)
  submission_time.png       submission time per run, summed over jobs (log scale)
  collection_time.png       result collection time per run, summed over jobs (log scale)
  submission_to_results_duration.png  earliest chunk submission to latest chunk result per run (log scale)
  effective_cost_EUR.png    effective cost in EUR per run (log scale)

Reads each run's processed_results/summary.json (a subset: summary+nsamples-N.json) and
run_data[_spam][+nsamples-N].json. Works with both the current summary format and the
legacy one (effective_cost_euros instead of effective_cost_EUR, no computer_name, ...);
a quantity a run's summary lacks is left out of that plot with a warning.
"""

import json
from pathlib import Path

from source.core.common import PROJECT_ROOT, rel_path
from source.core.records.run_names import parse_run_name, run_summary_name, subset_suffix
from source.core.analysis.derived_quantities import EPSILON, cumulative_awae, best_schemes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter



def _find_summary(run_path, n_samples):
    """The run's summary json (core/records/run_names.py: run_summary_name) for the full run (n_samples None) or a subset."""
    p = run_path / "processed_results" / run_summary_name(n_samples)
    return p if p.exists() else None


def _load_run(run_folder, n_samples):
    """The AWAE curves of the run's best EMS (and the label parts and summary the plots need), or None if it has no summary."""
    run_path = PROJECT_ROOT / "datafiles" / run_folder
    summary_path = _find_summary(run_path, n_samples)
    if summary_path is None:
        print(f"⚠️  {run_folder}: no summary json — left out of the collective plots")
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    backend = parse_run_name(run_folder).device
    suffix = subset_suffix(n_samples)

    # Best EMS for N_max and for N_acc; "+SPAM" marks the readout-corrected variant (older
    # summaries name the scheme only — then the uncorrected data is used).
    best = best_schemes(summary)
    run_data = {}
    curves = []
    for tag, ems in best:
        spam = ems.endswith("+SPAM")
        fname = f"run_data{'_spam' if spam else ''}{suffix}.json"
        if fname not in run_data:
            path = run_path / "processed_results" / fname
            run_data[fname] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        data = run_data[fname]
        if data is None:
            print(f"⚠️  {run_folder}: no {fname} — {ems} left out of the AWAE plot")
            continue
        base = ems.removesuffix("+SPAM")
        pts = sorted((int(nt), d[base]["AWAE"], d["Z_exact"]) for nt, d in data["depths"].items()
                     if isinstance(d.get(base), dict) and "AWAE" in d[base])
        if pts:
            nts, awae, z = zip(*pts)
            cumul = list(cumulative_awae(z, awae))   # Z_exact-weighted mean of AWAE over all depths up to N_t
            curves.append((tag, ems, list(nts), dict(awae=list(awae), cumul=cumul)))

    cpu = (summary.get("timing_breakdown") or {}).get("cpu") or {}
    submit_parts = [cpu.get("submit_s"), cpu.get("health_check_submit_s")]
    submit_s = sum(v for v in submit_parts if isinstance(v, (int, float))) if any(isinstance(v, (int, float)) for v in submit_parts) else None

    return dict(
        run_folder = run_folder,
        name       = summary.get("computer_name") or backend,
        shots      = summary.get("shots_per_circuit"),
        n_samples  = n_samples,
        layout     = parse_run_name(run_folder).layout,
        qpu_time_s = summary.get("QPU_time_s"),
        submit_s   = submit_s,   # summed submission time (chunks + health checks), part of CPU_time_s
        collect_s  = summary.get("Collect_time_s"),   # "N/A" for runs collected before it was timed
        window_s   = summary.get("QPU_submission_to_results_duration_s"),   # None for simulations
        cost_eur   = summary.get("effective_cost_EUR", summary.get("effective_cost_euros")),   # legacy key: effective_cost_euros
        curves     = curves,
    )


def _labels(runs):
    """QPU name per run, extended by shots / subset / layout only where it tells runs with the same label apart
    (a field the duplicates share is not appended)."""
    labels = [r["name"] for r in runs]
    for extra in (lambda r: f"{r['shots']} shots" if r["shots"] is not None else None,
                  lambda r: f"n={r['n_samples']}" if r["n_samples"] is not None else None,
                  lambda r: f"layout {r['layout']}" if r["layout"] else None,
                  lambda r: r["run_folder"]):
        dups = {l for l in labels if labels.count(l) > 1}
        if not dups:
            break
        differs = {l for l in dups if len({extra(r) for lab, r in zip(labels, runs) if lab == l}) > 1}
        labels = [f"{l}, {extra(r)}" if l in differs and extra(r) else l for l, r in zip(labels, runs)]
    return labels


def _plain_log_axis(ax):
    """Log-scale y axis labelled with ordinary numbers (1, 10, 100, 0.5, ...) instead of 10^k."""
    ax.set_yscale("log")
    fmt = FuncFormatter(lambda v, _: f"{v:,.10g}")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(fmt)
    lo, hi = ax.get_ylim()
    if hi / lo < 100:   # fewer than ~2 decades: label 2x and 5x too so the axis isn't bare
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.yaxis.set_minor_formatter(fmt)
    else:
        ax.yaxis.set_minor_formatter(NullFormatter())


def _seconds_fmt(v):
    return f"{v:,.0f} s" if v >= 1000 else f"{v:.4g} s"


def _bar_plot(runs, labels, key, ylabel, title, value_fmt, path):
    pts = [(l, r[key]) for l, r in zip(labels, runs) if isinstance(r[key], (int, float)) and r[key] > 0]
    for l, r in zip(labels, runs):
        if not (isinstance(r[key], (int, float)) and r[key] > 0):
            print(f"⚠️  {l}: no positive {key} in its summary — left out of {path.name}")
    if not pts:
        print(f"⏭  {path.name}: no data — skipped")
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(pts) + 2), 5))
    names, vals = zip(*pts)
    bars = ax.bar(range(len(vals)), vals, color="steelblue", width=0.6)
    for b, v in zip(bars, vals):
        ax.annotate(value_fmt(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(min(vals) / 2, max(vals) * 2.5)
    _plain_log_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved → {rel_path(path)}")


AWAE_KINDS = {"awae": "AWAE", "cumul": "Cumulative AWAE"}


def _awae_plot(runs, labels, kinds, path):
    """One panel per kind in kinds ("awae", "cumul"), sharing the y axis and one legend."""
    fig, axes = plt.subplots(1, len(kinds), figsize=(8 * len(kinds) + 7, 6), sharey=True, squeeze=False)
    axes = axes[0]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    all_nt = set()
    for i, (r, l) in enumerate(zip(runs, labels)):
        c = colors[i % len(colors)]
        for tag, ems, nts, vals in r["curves"]:
            for ax, kind in zip(axes, kinds):
                ax.plot(nts, vals[kind], color=c, ls="-" if tag == "N_max" else "--",
                        marker="o" if tag == "N_max" else "s", lw=1.5, ms=4, label=f"{l}: {ems} (best for {tag})")
            all_nt.update(nts)
    if not all_nt:
        plt.close(fig)
        print(f"⏭  {path.name}: no AWAE data — skipped")
        return
    for ax, kind in zip(axes, kinds):
        ax.axhline(EPSILON, color="black", ls=":", lw=1, alpha=0.6, label=rf"accuracy threshold $\varepsilon$={EPSILON}")
        ax.set_xticks(sorted(all_nt))
        ax.set_xlabel("$N_t$")
        ax.set_ylabel(AWAE_KINDS[kind])
        ax.set_title(f"{AWAE_KINDS[kind]} vs $N_t$", fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Best EMS of each run (solid: best at $N_{max}$, dashed: best at $N_{acc}$)", fontweight="bold", fontsize=11)
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved → {rel_path(path)}")


def make_collective_plots(analyzed, out_dir):
    """analyzed: [(run_folder, n_samples or None)]; the plots are written to out_dir."""
    runs = [r for r in (_load_run(f, n) for f, n in analyzed) if r is not None]
    if not runs:
        print("⏭  no analyzed runs — no collective plots")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = _labels(runs)
    _awae_plot(runs, labels, ["awae"], out_dir / "awae_best_ems_vs_Nt.png")
    _awae_plot(runs, labels, ["cumul"], out_dir / "cumul_awae_best_ems_vs_Nt.png")
    _awae_plot(runs, labels, ["awae", "cumul"], out_dir / "awae_and_cumul_awae_best_ems_vs_Nt.png")
    _bar_plot(runs, labels, "qpu_time_s", "QPU time (s)", "QPU time per run",
              _seconds_fmt, out_dir / "qpu_time.png")
    _bar_plot(runs, labels, "submit_s", "Submission time (s)", "Submission time per run (summed over jobs)",
              _seconds_fmt, out_dir / "submission_time.png")
    _bar_plot(runs, labels, "collect_s", "Collection time (s)", "Result collection time per run (summed over jobs)",
              _seconds_fmt, out_dir / "collection_time.png")
    _bar_plot(runs, labels, "window_s", "Duration (s)", "Submission-to-results window per run",
              lambda v: f"{v / 3600:.1f} h" if v >= 3600 else _seconds_fmt(v), out_dir / "submission_to_results_duration.png")
    _bar_plot(runs, labels, "cost_eur", "Effective cost (EUR)", "Effective cost per run (excl. VAT)",
              lambda v: f"€{v:,.2f}", out_dir / "effective_cost_EUR.png")
