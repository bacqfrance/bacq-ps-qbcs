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
  submission_to_results_duration.png  earliest job submission to latest job result per run (log scale)
  submission_to_results_duration_without_dead_time.png  the same window minus long idle gaps (log scale)
  effective_cost_EUR.png    effective cost in EUR per run (log scale)
  n_acc.png                 N_acc per run (linear scale)
  e_min_nmax.png            E_min at N_max per run (linear scale, with the threshold epsilon)
and, in analysis/summaries/ (not with the plots), <benchmark>_collective_<data types>.csv: one row per run with
  run, n_samples, N_acc, N_max, E_min_Nmax, best_scheme_Nmax, QPU_time_s, QPU_time, effective_cost_EUR

Reads each run's processed_results/summary.json (a subset: summary+nsamples-N.json) and
run_data[_spam][+nsamples-N].json. Works with both the current summary format and the
legacy one (effective_cost_euros instead of effective_cost_EUR, computer_name instead of device_label, ...);
a quantity a run's summary lacks is left out of that plot with a warning.
"""

import csv
import json
import math
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
        name       = summary.get("device_label") or summary.get("computer_name") or backend,   # computer_name: summaries saved before the rename
        shots      = summary.get("shots_per_circuit"),
        n_samples  = n_samples,
        layout     = parse_run_name(run_folder).layout,
        date       = parse_run_name(run_folder).date,         # YYYY-MM-DD (None: an undated test run)
        run_number = parse_run_name(run_folder).run_number,   # N of _#N (None: an undated test run)
        date_label = parse_run_name(run_folder).date_label,   # DD/MM/YYYY #N
        qpu_time_s = summary.get("QPU_time_s"),
        submit_s   = submit_s,   # summed submission time (chunks + health checks), part of CPU_time_s
        collect_s  = summary.get("Collect_time_s"),   # "N/A" for runs collected before it was timed
        window_s   = summary.get("QPU_submission_to_results_duration_s"),   # None for simulations without job records
        window_active_s = summary.get("QPU_submission_to_results_duration_without_dead_time_s"),   # None for simulations without job records and older summaries
        n_acc      = summary.get("N_acc"),
        n_max      = summary.get("N_max"),
        emin_nmax  = summary.get("E_min_Nmax"),
        best_ems_nmax = summary.get("best_ems_Nmax"),
        cost_eur   = summary.get("effective_cost_EUR", summary.get("effective_cost_euros")),   # legacy key: effective_cost_euros
        curves     = curves,
    )


def run_order(run):
    """Sort key of the runs in the collective plots: by QPU, then layout, then date and run number (oldest first)."""
    return (run["name"], run["layout"] or "", run["date"] or "", run["run_number"] or 0)


def _labels(runs):
    """QPU name, layout and date (DD/MM/YYYY #N) per run, extended by shots / subset only where it tells runs with
    the same label apart (a field the duplicates share is not appended)."""
    labels = [f"{r['name']}, layout {r['layout']}, {r['date_label']}" for r in runs]
    for extra in (lambda r: f"{r['shots']} shots" if r["shots"] is not None else None,
                  lambda r: f"n={r['n_samples']}" if r["n_samples"] is not None else None,
                  lambda r: r["run_folder"]):
        dups = {l for l in labels if labels.count(l) > 1}
        if not dups:
            break
        differs = {l for l in dups if len({extra(r) for lab, r in zip(labels, runs) if lab == l}) > 1}
        labels = [f"{l}, {extra(r)}" if l in differs and extra(r) else l for l, r in zip(labels, runs)]
    return labels


def _duration_fmt(v):
    """A time in seconds as text: up to 60 s in seconds, up to 3600 s in minutes, up to 86400 s in hours, above in days."""
    if v <= 60:
        return f"{v:.4g} s"
    if v <= 3600:
        return f"{v / 60:.1f} min"
    if v <= 86400:
        return f"{v / 3600:.1f} h"
    return f"{v / 86400:.1f} d"


def _plain_log_axis(ax, tick_fmt=None):
    """Log-scale y axis labelled with ordinary numbers (1, 10, 100, 0.5, ...) instead of 10^k;
    tick_fmt (value -> text) replaces the number format (the time plots use _duration_fmt)."""
    ax.set_yscale("log")
    fmt = FuncFormatter(lambda v, _: tick_fmt(v) if tick_fmt else f"{v:,.10g}")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(fmt)
    lo, hi = ax.get_ylim()
    if hi / lo < 100:   # fewer than ~2 decades: label 2x and 5x too so the axis isn't bare
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.yaxis.set_minor_formatter(fmt)
    else:
        ax.yaxis.set_minor_formatter(NullFormatter())


def _bar_plot(runs, labels, key, ylabel, title, value_fmt, path, tick_fmt=None, log=True):
    """Bar plot of r[key] per run; log=True: log scale, positive values only; log=False: linear axis from 0
    (zero kept), with the threshold line epsilon for E_min."""
    def usable(v):
        return isinstance(v, (int, float)) and not math.isnan(v) and (v > 0 or (not log and v >= 0))
    pts = [(l, r[key]) for l, r in zip(labels, runs) if usable(r[key])]
    for l, r in zip(labels, runs):
        if not usable(r[key]):
            print(f"⚠️  {l}: no {'positive ' if log else ''}{key} in its summary — left out of {path.name}")
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
    if log:
        ax.set_ylim(min(vals) / 2, max(vals) * 2.5)
        _plain_log_axis(ax, tick_fmt)
    elif key == "emin_nmax":
        ax.set_ylim(0, 1.05)
        ax.axhline(EPSILON, color="black", ls="--", lw=1, alpha=0.6, label=rf"accuracy threshold $\varepsilon$={EPSILON}")
        ax.legend(fontsize=8)
    else:
        ax.set_ylim(0, max(max(vals), 1) * 1.15)
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


def _write_run_table(runs, labels, path):
    """The collective run table: one row per run, ";"-separated, each column padded with spaces to a common width
    so the file also reads as a text table (a missing value, None / NaN / "N/A", is an empty cell)."""
    def num(v, fmt):
        return fmt.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v) else ""
    def text(v):
        return v if isinstance(v, str) and v != "N/A" else ""
    rows = [["run", "n_samples", "N_acc", "N_max", "E_min_Nmax", "best_scheme_Nmax",
             "QPU_time_s", "QPU_time", "effective_cost_EUR"]]
    for l, r in zip(labels, runs):
        q = r["qpu_time_s"]
        q_ok = isinstance(q, (int, float)) and not math.isnan(q)
        rows.append([l, "" if r["n_samples"] is None else str(r["n_samples"]),
                     num(r.get("n_acc"), "{:.0f}"), num(r.get("n_max"), "{:.0f}"),
                     num(r.get("emin_nmax"), "{:.4g}"), text(r.get("best_ems_nmax")),
                     num(q, "{:.0f}"), _duration_fmt(round(q)) if q_ok else "",
                     num(r["cost_eur"], "{:.2f}")])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        for row in rows:                        # "a   ; b   ; c": a space on each side of ";", no trailing padding
            cells = [c.ljust(n) for c, n in zip(row, widths)]
            w.writerow([cells[0] + " "] + [" " + c + " " for c in cells[1:-1]] + [" " + row[-1]])
    print(f"✅ Saved → {rel_path(path)}")


def make_collective_plots(analyzed, out_dir, table_path):
    """analyzed: [(run_folder, n_samples or None)]; the plots are written to out_dir, the run table to table_path."""
    runs = sorted((r for r in (_load_run(f, n) for f, n in analyzed) if r is not None), key=run_order)
    if not runs:
        print("⏭  no analyzed runs — no collective plots")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = _labels(runs)
    _awae_plot(runs, labels, ["awae"], out_dir / "awae_best_ems_vs_Nt.png")
    _awae_plot(runs, labels, ["cumul"], out_dir / "cumul_awae_best_ems_vs_Nt.png")
    _awae_plot(runs, labels, ["awae", "cumul"], out_dir / "awae_and_cumul_awae_best_ems_vs_Nt.png")
    _bar_plot(runs, labels, "qpu_time_s", "QPU time", "QPU time per run",
              _duration_fmt, out_dir / "qpu_time.png", _duration_fmt)
    _bar_plot(runs, labels, "submit_s", "Submission time", "Submission time per run (summed over jobs)",
              _duration_fmt, out_dir / "submission_time.png", _duration_fmt)
    _bar_plot(runs, labels, "collect_s", "Collection time", "Result collection time per run (summed over jobs)",
              _duration_fmt, out_dir / "collection_time.png", _duration_fmt)
    _bar_plot(runs, labels, "window_s", "Duration", "Submission-to-results window per run",
              _duration_fmt, out_dir / "submission_to_results_duration.png", _duration_fmt)
    _bar_plot(runs, labels, "window_active_s", "Duration", "Submission-to-results window per run, without dead time",
              _duration_fmt, out_dir / "submission_to_results_duration_without_dead_time.png", _duration_fmt)
    _bar_plot(runs, labels, "cost_eur", "Effective cost (EUR)", "Effective cost per run (excl. VAT)",
              lambda v: f"€{v:,.2f}", out_dir / "effective_cost_EUR.png")
    _bar_plot(runs, labels, "n_acc", "$N_{acc}$", "$N_{acc}$ per run",
              lambda v: f"{v:.0f}", out_dir / "n_acc.png", log=False)
    _bar_plot(runs, labels, "emin_nmax", "$E_{min}$ at $N_{max}$", "$E_{min}$ at $N_{max}$ per run",
              lambda v: f"{v:.3g}", out_dir / "e_min_nmax.png", log=False)
    table_path = Path(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    _write_run_table(runs, labels, table_path)
