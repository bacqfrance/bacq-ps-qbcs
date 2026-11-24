"""
collective_plots.py
Plots comparing all runs analyzed by source/analysis_scripts/run_post_process.py (stage 6):
  awae_best_ems_vs_Nt.png        AWAE vs N_t of each run's best EMS (for N_max, for N_acc and for N_acc_strict: up to three per run,
                                 a scheme listed once), full-run values with 95% bootstrap interval bars (none for a run without a bootstrap)
  cumul_awae_best_ems_vs_Nt.png  the same for the cumulative AWAE (Z_exact-weighted mean up to N_t,
                                 as stored in run_data*.json — the quantity N_acc is defined by)
  awae_and_cumul_awae_best_ems_vs_Nt.png  both side by side
  qpu_time.png              QPU time per run (log scale; a dummy emulator's modelled time is labelled "[estimated]")
  submission_time.png       submission time per run, summed over jobs (log scale)
  collection_time.png       result collection time per run, summed over jobs (log scale)
  submission_to_results_duration.png  earliest job submission to latest job result per run (log scale)
  submission_to_results_duration_without_dead_time.png  the same window minus long idle gaps (log scale)
  effective_cost_EUR.png    effective cost in EUR per run (log scale; a dummy emulator's modelled cost is labelled "[estimated]")
  n_acc.png                 N_acc per run (linear scale)
  n_acc_strict.png          N_acc_strict per run (linear scale; a run without a bootstrap has none and is marked in a note)
  n_acc_bare.png            N_acc_bare per run (the N_acc rule over the unmitigated schemes only; linear scale)
  e_min_nmax.png            E_min at N_max per run (linear scale, with the threshold epsilon and the 95% interval of E_min_Nmax)
  e_bare_nmax.png           E_bare at N_max per run, the lower of bare and bare+SPAM (E_bare_Nmax / E_bare_spam_Nmax; the chosen one
                            is named in the run label), linear scale, with the threshold epsilon and its 95% interval
and, in the same folder, <benchmark>_collective_<data types>[<filter suffix>][_latest-per-qpu|_best-nacc-per-qpu|_best-nacc-strict-per-qpu|_best-nacc-bare-per-qpu|_best-emin-at-nmax-per-qpu][_hc-gt-T].csv (the suffixes only for a filtered set of runs): one row per run with
  run, n_samples (the kept-chunk count of a health-check-filtered run), sample_size, N_acc, best_scheme_Nacc, N_max, E_min_Nmax, best_scheme_Nmax, QPU_time_s, QPU_time, effective_cost_EUR,
  N_acc_strict, best_scheme_Nacc_strict, E_min_Nacc, E_min_Nacc_strict, and the 95% interval ends E_min_Nmax_ci95_lo / _hi, E_min_Nacc_ci95_lo / _hi, E_min_Nacc_strict_ci95_lo / _hi,
  N_acc_bare, E_min_Nacc_bare, E_min_Nacc_bare_ci95_lo / _hi, best_scheme_Nacc_bare,
  E_bare_Nmax (the lower of bare and bare+SPAM), E_bare_Nmax_version ("bare" / "bare+SPAM"), E_bare_Nmax_ci95_lo / _hi
  (the strict and interval cells are empty for a run without a bootstrap)
  and a last column, estimated ("estimated" for a dummy emulator), only when a dummy emulator is among the runs
The figures are redrawn only when the set of runs or one of their summaries changed since the last drawing
(collective_inputs.json in the folder lists them); force=True redraws regardless.

Reads each run's processed_results/summary.json (a subset: summary+nsamples-N.json) and
run_data[_spam][+nsamples-N].json. Works with both the current summary format and the
legacy one (effective_cost_euros instead of effective_cost_EUR, computer_name instead of device_label, ...);
each run's summary json is copied into the plots folder too;
a quantity a run's summary lacks is left out of that plot with a warning.
"""

import csv
import json
import math
import shutil
from pathlib import Path

from source.core.common import PROJECT_ROOT, rel_path
from source.core.records.run_info import load_run_info
from source.core.records.run_names import layout_label, parse_run_name, run_summary_name, subset_suffix, summary_name
from source.core.analysis.derived_quantities import EPSILON, best_schemes

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

    # Best EMS for N_max, N_acc and N_acc_strict; "+SPAM" marks the readout-corrected variant (older
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
        pts = sorted((int(nt), d[base]) for nt, d in data["depths"].items()
                     if isinstance(d.get(base), dict) and "AWAE" in d[base] and "AWAE_cumul" in d[base])
        if pts:
            def col(key, pts=pts):
                return [float("nan") if e.get(key) is None else e[key] for _, e in pts]
            curves.append((tag, ems, [nt for nt, _ in pts],
                           dict(awae=col("AWAE"), cumul=col("AWAE_cumul"),   # full-run values (cumulative: Z_exact-weighted mean up to N_t)
                                awae_ci=[e.get("AWAE_ci95") for _, e in pts], cumul_ci=[e.get("AWAE_cumul_ci95") for _, e in pts])))   # [lo, hi] or None

    cpu = (summary.get("timing_breakdown") or {}).get("cpu") or {}
    submit_parts = [cpu.get("submit_s"), cpu.get("health_check_submit_s")]
    submit_s = sum(v for v in submit_parts if isinstance(v, (int, float))) if any(isinstance(v, (int, float)) for v in submit_parts) else None

    return dict(
        run_folder = run_folder,
        name       = summary.get("device_label") or summary.get("computer_name") or backend,   # computer_name: summaries saved before the rename
        shots      = summary.get("shots_per_circuit"),
        n_samples  = n_samples,   # None, a chunk count, or a chunk_scope.HealthCheckScope (--filter-per-health-check)
        n_chunks   = n_samples if n_samples is None or isinstance(n_samples, int) else len(n_samples),
        sample_size = load_run_info(run_folder).sample_size,   # the chunks of a complete run
        layout     = layout_label(run_folder),   # the layout field, or optimized_by_gates / optimized_per_chunk_by_2q_gates_IQM
        date       = parse_run_name(run_folder).date,         # YYYY-MM-DD (None: an undated test run)
        run_number = parse_run_name(run_folder).run_number,   # N of _#N (None: an undated test run)
        date_label = parse_run_name(run_folder).date_label,   # DD/MM/YYYY #N
        estimated  = parse_run_name(run_folder).is_dummy,   # a dummy emulator: its QPU time and cost are modelled, not measured
        qpu_time_s = summary.get("QPU_time_s"),
        submit_s   = submit_s,   # summed submission time (chunks + health checks), part of CPU_time_s
        collect_s  = summary.get("Collect_time_s"),   # "N/A" for runs collected before it was timed
        window_s   = summary.get("QPU_submission_to_results_duration_s"),   # None for simulations without job records
        window_active_s = summary.get("QPU_submission_to_results_duration_without_dead_time_s"),   # None for simulations without job records and older summaries
        n_acc      = summary.get("N_acc"),
        n_max      = summary.get("N_max"),
        emin_nmax  = summary.get("E_min_Nmax"),
        emin_nmax_ci = summary.get("E_min_Nmax_ci95"),   # [lo, hi] or None (no bootstrap)
        emin_nacc  = summary.get("E_min_Nacc"),
        emin_nacc_ci = summary.get("E_min_Nacc_ci95"),
        n_acc_strict = summary.get("N_acc_strict"),   # None without a bootstrap
        emin_nacc_strict = summary.get("E_min_Nacc_strict"),
        emin_nacc_strict_ci = summary.get("E_min_Nacc_strict_ci95"),
        best_ems_nacc_strict = summary.get("best_ems_Nacc_strict"),
        n_acc_bare = summary.get("N_acc_bare"),   # None for a summary written before it existed
        emin_nacc_bare = summary.get("E_min_Nacc_bare"),
        emin_nacc_bare_ci = summary.get("E_min_Nacc_bare_ci95"),
        best_ems_nacc_bare = summary.get("best_ems_Nacc_bare"),
        best_ems_nmax = summary.get("best_ems_Nmax"),
        **dict(zip(("ebare_nmax", "ebare_nmax_ci", "ebare_nmax_version"), _best_bare_nmax(summary))),
        best_ems_nacc = summary.get("best_ems_Nacc"),
        cost_eur   = summary.get("effective_cost_EUR", summary.get("effective_cost_euros")),   # legacy key: effective_cost_euros
        curves     = curves,
    )


def _best_bare_nmax(summary):
    """(value, 95% interval, version) of the lower of E_bare at N_max without and with SPAM correction
    (E_bare_Nmax / E_bare_spam_Nmax; version "bare" / "bare+SPAM"); (None, None, None) when the summary has neither."""
    found = [(summary[k], summary.get(f"{k}_ci95"), version) for version, k in (("bare", "E_bare_Nmax"), ("bare+SPAM", "E_bare_spam_Nmax"))
             if isinstance(summary.get(k), (int, float)) and not math.isnan(summary[k])]
    return min(found, key=lambda t: t[0]) if found else (None, None, None)


def run_order(run):
    """Sort key of the runs in the collective plots: by device label (QPU name, data type), then layout, then date and run number (oldest first)."""
    return (run["name"], run["layout"] or "", run["date"] or "", run["run_number"] or 0)


def _labels(runs):
    """QPU name, layout and date (DD/MM/YYYY #N) per run, with "(n/N chunks)" on every incomplete run (fewer chunks
    analyzed than the run's sample size; a health-check-filtered run reads "(P011 > T: k/N chunks kept)"), extended by shots only where it tells runs with the same label apart
    (a field the duplicates share is not appended)."""
    labels = [f"{r['name']}, layout {r['layout']}, {r['date_label']}"
              + (f" (P011 > {r['n_samples'].threshold:g}: {r.get('n_chunks', r['n_samples'])}/{r['sample_size']} chunks kept)"
                 if hasattr(r["n_samples"], "threshold")
                 else f" ({r.get('n_chunks', r['n_samples'])}/{r['sample_size']} chunks)" if r["n_samples"] is not None else "") for r in runs]
    for extra in (lambda r: f"{r['shots']} shots" if r["shots"] is not None else None,
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


ESTIMATED = " [estimated]"   # marks a dummy emulator's modelled QPU time and cost in the plots and the run table


def _bar_plot(runs, labels, key, ylabel, title, value_fmt, path, tick_fmt=None, log=True, mark_estimated=False, ci_key=None, missing_note=None):
    """Bar plot of r[key] per run; log=True: log scale, positive values only; log=False: linear axis from 0
    (zero kept), with the threshold line epsilon for E_min. mark_estimated: a dummy emulator's bar is labelled
    "[estimated]" (its value is modelled) and the title says so. ci_key: the run key of a [lo, hi] 95% interval drawn as a line centred on the bar's right
    edge (none for a null interval). missing_note: text added under the plot when a run has no value (it is left out)."""
    if mark_estimated and any(r.get("estimated") for r in runs):
        labels = [l + (ESTIMATED if r.get("estimated") else "") for l, r in zip(labels, runs)]
        title += " (dummy emulators: estimated)"
    def usable(v):
        return isinstance(v, (int, float)) and not math.isnan(v) and (v > 0 or (not log and v >= 0))
    pts = [(l, r[key], r.get(ci_key) if ci_key else None) for l, r in zip(labels, runs) if usable(r[key])]
    for l, r in zip(labels, runs):
        if not usable(r[key]):
            print(f"⚠️  {l}: no {'positive ' if log else ''}{key} in its summary — left out of {path.name}")
    if not pts:
        print(f"⏭  {path.name}: no data — skipped")
        return
    names, vals, cis = zip(*pts)
    # the rotated run labels are long and tight_layout() takes their room from the axes: a taller figure keeps the plot area ~4 in
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(pts) + 2), 4 + min(0.07 * max(len(n) for n in names), 5)))
    bars = ax.bar(range(len(vals)), vals, color="steelblue", width=0.6)
    tops = list(vals)    # where each value label sits: on top of its bar
    highest = list(vals)   # the top of each bar or of its interval line, for the axis range
    for i, ci in enumerate(cis):
        if isinstance(ci, (list, tuple)) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
            # the interval centred on the bar's right edge (bar width 0.6), with short caps, clear of the value label
            x = i + 0.3
            ax.vlines(x, ci[0], ci[1], color="black", lw=1.5, zorder=3)
            ax.hlines(ci, x - 0.04, x + 0.04, color="black", lw=1.5, zorder=3)
            highest[i] = max(highest[i], ci[1])
    if not log and key in ("emin_nmax", "ebare_nmax"):   # a label below the threshold line would sit on it: lift it just above the line
        tops = [max(t, EPSILON) for t in tops]
    for b, v, top in zip(bars, vals, tops):
        # a white box keeps the number readable where the threshold line or a grid line crosses it
        ax.annotate(value_fmt(v), (b.get_x() + b.get_width() / 2, top), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    if log:
        ax.set_ylim(min(vals) / 2, max(highest + tops) * 6)   # room for the value label above the tallest bar or interval
        _plain_log_axis(ax, tick_fmt)
    elif key in ("emin_nmax", "ebare_nmax"):
        ax.set_ylim(0, max(1.05, max(highest + tops) * 1.12))   # room above an interval reaching past 1
        ax.axhline(EPSILON, color="black", ls="--", lw=1, alpha=0.6, zorder=2, label=rf"accuracy threshold $\varepsilon$={EPSILON}")
        ax.legend(fontsize=8)
    else:
        ax.set_ylim(0, max(max(highest + tops), 1) * 1.35)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    if missing_note and len(pts) < len(runs):
        fig.text(0.5, 0.0, missing_note, ha="center", va="top", fontsize=8, color="firebrick")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved → {rel_path(path)}")


AWAE_KINDS = {"awae": "AWAE", "cumul": "Cumulative AWAE"}
CURVE_STYLE = {"N_max": ("-", "o"), "N_acc": ("--", "s"), "N_acc_strict": ("-.", "^")}   # (linestyle, marker) per best-scheme tag


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
                ax.plot(nts, vals[kind], color=c, ls=CURVE_STYLE[tag][0], marker=CURVE_STYLE[tag][1],
                        lw=1.5, ms=4, label=f"{l}: {ems} (best for {tag})")
                for x, iv in zip(nts, vals[kind + "_ci"]):   # 95% interval; the full-run value can lie outside it, so no errorbar yerr
                    if iv is not None and iv[0] is not None and iv[1] is not None:
                        ax.vlines(x, iv[0], iv[1], color=c, lw=1.2, alpha=0.5)
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
    fig.suptitle("Best EMS of each run (solid: best at $N_{max}$, dashed: at $N_{acc}$, dash-dot: at $N_{acc,strict}$; bars: 95% intervals)", fontweight="bold", fontsize=11)
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved → {rel_path(path)}")


def _write_run_table(runs, labels, path):
    """The collective run table: one row per run, ";"-separated, each column padded with spaces to a common width
    so the file also reads as a text table (a missing value, None / NaN / "N/A", is an empty cell). The last column, `estimated`, is added only when a dummy emulator is among the runs."""
    def num(v, fmt):
        return fmt.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v) else ""
    def text(v):
        return v if isinstance(v, str) and v != "N/A" else ""
    def ci_cells(ci):
        ok = isinstance(ci, (list, tuple)) and len(ci) == 2
        return [num(ci[0], "{:.4g}") if ok else "", num(ci[1], "{:.4g}") if ok else ""]
    with_estimated = any(r.get("estimated") for r in runs)   # a column only when a dummy emulator (modelled time and cost) is among the runs
    rows = [["run", "n_samples", "sample_size", "N_acc", "best_scheme_Nacc", "N_max", "E_min_Nmax", "best_scheme_Nmax",
             "QPU_time_s", "QPU_time", "effective_cost_EUR",
             "N_acc_strict", "best_scheme_Nacc_strict", "E_min_Nacc", "E_min_Nacc_strict",
             "E_min_Nmax_ci95_lo", "E_min_Nmax_ci95_hi", "E_min_Nacc_ci95_lo", "E_min_Nacc_ci95_hi",
             "E_min_Nacc_strict_ci95_lo", "E_min_Nacc_strict_ci95_hi",
             "N_acc_bare", "best_scheme_Nacc_bare", "E_min_Nacc_bare", "E_min_Nacc_bare_ci95_lo", "E_min_Nacc_bare_ci95_hi",
             "E_bare_Nmax", "E_bare_Nmax_version", "E_bare_Nmax_ci95_lo", "E_bare_Nmax_ci95_hi"]]
    if with_estimated:
        rows[0].append("estimated")
    for l, r in zip(labels, runs):
        q = r["qpu_time_s"]
        q_ok = isinstance(q, (int, float)) and not math.isnan(q)
        rows.append([l, "" if r["n_samples"] is None else str(r.get("n_chunks", r["n_samples"])), str(r["sample_size"]),
                     num(r.get("n_acc"), "{:.0f}"), text(r.get("best_ems_nacc")), num(r.get("n_max"), "{:.0f}"),
                     num(r.get("emin_nmax"), "{:.4g}"), text(r.get("best_ems_nmax")),
                     num(q, "{:.0f}"), _duration_fmt(round(q)) if q_ok else "",
                     num(r["cost_eur"], "{:.2f}"),
                     num(r.get("n_acc_strict"), "{:.0f}"), text(r.get("best_ems_nacc_strict")),
                     num(r.get("emin_nacc"), "{:.4g}"), num(r.get("emin_nacc_strict"), "{:.4g}")]
                    + ci_cells(r.get("emin_nmax_ci")) + ci_cells(r.get("emin_nacc_ci")) + ci_cells(r.get("emin_nacc_strict_ci"))
                    + [num(r.get("n_acc_bare"), "{:.0f}"), text(r.get("best_ems_nacc_bare")), num(r.get("emin_nacc_bare"), "{:.4g}")]
                    + ci_cells(r.get("emin_nacc_bare_ci"))
                    + [num(r.get("ebare_nmax"), "{:.4g}"), text(r.get("ebare_nmax_version"))] + ci_cells(r.get("ebare_nmax_ci"))
                    + (["estimated" if r.get("estimated") else ""] if with_estimated else []))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        for row in rows:                        # "a   ; b   ; c": a space on each side of ";", no trailing padding
            cells = [c.ljust(n) for c, n in zip(row, widths)]
            w.writerow([cells[0] + " "] + [" " + c + " " for c in cells[1:-1]] + [" " + row[-1]])
    print(f"✅ Saved → {rel_path(path)}")


MANIFEST = "collective_inputs.json"   # the runs (summary name and modification time) and figures of the last drawing


def make_collective_plots(analyzed, out_dir, force=False):
    """analyzed: [(run_folder, n_samples or None)]; the plots, the run table (<out_dir name>.csv) and the runs'
    summary jsons are written to out_dir. The drawing is skipped when MANIFEST there matches the current runs
    and summaries and every figure and the table it lists exists (a change of the plot code is not detected: force=True redraws)."""
    runs = sorted((r for r in (_load_run(f, n) for f, n in analyzed) if r is not None), key=run_order)
    if not runs:
        print("⏭  no analyzed runs — no collective plots")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = _labels(runs)
    for r in runs:   # the summaries behind the plots and the table, next to them
        src = _find_summary(PROJECT_ROOT / "datafiles" / r["run_folder"], r["n_samples"])
        shutil.copy2(src, out_dir / summary_name(r["run_folder"], r["n_samples"]))
    inputs = [[summary_name(r["run_folder"], r["n_samples"]),
               (_find_summary(PROJECT_ROOT / "datafiles" / r["run_folder"], r["n_samples"])).stat().st_mtime_ns] for r in runs]
    manifest_path = out_dir / MANIFEST
    if not force and manifest_path.is_file():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            old = {}
        if old.get("inputs") == inputs and old.get("files") and all((out_dir / f).is_file() for f in old["files"]):
            print(f"⏭  collective plots in {rel_path(out_dir)} are up to date (same runs and summaries; --force-plots redraws)")
            return
    _awae_plot(runs, labels, ["awae"], out_dir / "awae_best_ems_vs_Nt.png")
    _awae_plot(runs, labels, ["cumul"], out_dir / "cumul_awae_best_ems_vs_Nt.png")
    _awae_plot(runs, labels, ["awae", "cumul"], out_dir / "awae_and_cumul_awae_best_ems_vs_Nt.png")
    _bar_plot(runs, labels, "qpu_time_s", "QPU time", "QPU time per run",
              _duration_fmt, out_dir / "qpu_time.png", _duration_fmt, mark_estimated=True)
    _bar_plot(runs, labels, "submit_s", "Submission time", "Submission time per run (summed over jobs)",
              _duration_fmt, out_dir / "submission_time.png", _duration_fmt)
    _bar_plot(runs, labels, "collect_s", "Collection time", "Result collection time per run (summed over jobs)",
              _duration_fmt, out_dir / "collection_time.png", _duration_fmt)
    _bar_plot(runs, labels, "window_s", "Duration", "Submission-to-results window per run",
              _duration_fmt, out_dir / "submission_to_results_duration.png", _duration_fmt)
    _bar_plot(runs, labels, "window_active_s", "Duration", "Submission-to-results window per run, without dead time",
              _duration_fmt, out_dir / "submission_to_results_duration_without_dead_time.png", _duration_fmt)
    _bar_plot(runs, labels, "cost_eur", "Effective cost (EUR)", "Effective cost per run (excl. VAT)",
              lambda v: f"€{v:,.2f}", out_dir / "effective_cost_EUR.png", mark_estimated=True)
    _bar_plot(runs, labels, "n_acc", "$N_{acc}$", "$N_{acc}$ per run",
              lambda v: f"{v:.0f}", out_dir / "n_acc.png", log=False)
    _bar_plot(runs, labels, "n_acc_strict", "$N_{acc,strict}$", "$N_{acc,strict}$ per run",
              lambda v: f"{v:.0f}", out_dir / "n_acc_strict.png", log=False,
              missing_note="runs left out: no $N_{acc,strict}$ (no bootstrap: fewer than 5 samples per input state)")
    _bar_plot(runs, labels, "n_acc_bare", "$N_{acc,bare}$", "$N_{acc,bare}$ per run (unmitigated schemes only)",
              lambda v: f"{v:.0f}", out_dir / "n_acc_bare.png", log=False,
              missing_note="runs left out: no $N_{acc,bare}$ (analysis from before it was added: re-analyze the run)")
    _bar_plot(runs, labels, "emin_nmax", "$E_{min}$ at $N_{max}$", "$E_{min}$ at $N_{max}$ per run",
              lambda v: f"{v:.3f}", out_dir / "e_min_nmax.png", log=False, ci_key="emin_nmax_ci")
    _bar_plot(runs, [f"{l} [{r['ebare_nmax_version']}]" if r.get("ebare_nmax_version") else l for l, r in zip(labels, runs)],
              "ebare_nmax", "$E_{bare}$ at $N_{max}$", "$E_{bare}$ at $N_{max}$ per run (lower of bare and bare+SPAM)",
              lambda v: f"{v:.3f}", out_dir / "e_bare_nmax.png", log=False, ci_key="ebare_nmax_ci",
              missing_note="runs left out: no $E_{bare}$ at $N_{max}$ in the summary (re-analyze the run)")
    _write_run_table(runs, labels, out_dir / f"{out_dir.name}.csv")
    files = sorted(p.name for p in out_dir.iterdir() if p.suffix in (".png", ".csv"))
    manifest_path.write_text(json.dumps({"inputs": inputs, "files": files}, indent=1), encoding="utf-8")
