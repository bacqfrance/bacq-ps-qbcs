"""
plot_health_checks.py
Plot the health checks that accompanied a run, together with the QPU jobs.

  top panel:    P(011) of each collected health check vs. health check number
                (pass threshold marked)
  bottom panel: UTC time bars of the health checks (orange, labelled H<n>) and of
                the QPU jobs (blue, labelled J<n> + the chunk numbers they
                contain), both labelled on the bottom axis, on a common x axis
                ordered by submission time.

Also writes processed_results/health_checks{SUFFIX}.json (SUFFIX is +nsamples-N for a subset analysis; all checks, all jobs with their
chunk lists and time windows).

Inputs: datafiles/<run>/health_checks/*.json (new format only; old files are
skipped), the chunk job records in job_records/ (local simulator runs write them too;
older ones only have chunk_times_UTC_time.json, which is read instead). Health checks and jobs only get a finish time once
qbcs_real_collect_*.py has run; an uncollected health check is drawn as a red X and an
uncollected chunk job as a purple X, both at their submission time (a hollow marker only
for an item with a start time but no end time that is neither of those). A health check that
could not be submitted is a black P at its submission-attempt time; a failed chunk job is a
black bar from its start to its end when the provider reported both, else a black P at its
submission time.
Skips runs without new-format health checks, printing a line.
"""

import os
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
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

from source.core.records.health_check_records import (
    HEALTH_CHECK_THRESHOLD, load_health_checks, collect_chunk_jobs, parse_utc, format_chunk_range,
)

# ── Config (same defaults / env overrides as plot_awae.py) ────────────────────
RUN_FOLDER    = "full_real-data_iqm-emerald_lnn-True_service-iqm-resonance_spam-True_layout-legacy"   # standalone: the run (datafiles/<RUN_FOLDER>)
ZOOM_X       = None    # e.g. (100, 160): show only items with index in this range on the x axis

N_SAMPLES    = None    # None = full run; int = the analysis of that many chunks (file names +nsamples-N)

if "QPU_N_SAMPLES" in os.environ:
    _v = os.environ["QPU_N_SAMPLES"]
    N_SAMPLES = None if _v in ("", "None") else int(_v)
RUN_FOLDER, INFO = resolve_run(RUN_FOLDER)   # QPU_RUN_FOLDER wins; the run's own settings win over the constants above
NAME         = parse_run_name(RUN_FOLDER)
NSHOTS, LNN = INFO.nshots, INFO.lnn
RUN_PATH   = Path("datafiles") / RUN_FOLDER
PLOTS_DIR  = plots_dir_of(RUN_FOLDER)
SUFFIX     = subset_suffix(N_SAMPLES)

COLOR_CHECK = "darkorange"
COLOR_JOB   = "steelblue"
COLOR_MISSING = "red"   # health check submitted but not collected: submission time only
COLOR_FAILED = "black"   # health check could not even be submitted: submission-attempt time only
COLOR_UNCOLLECTED_JOB = "purple"   # chunk job submitted, its chunks not collected: submission time only
MAX_TICKS   = 40


def _iso(dt):
    return None if dt is None else dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _thin(idx_list):
    step = max(1, -(-len(idx_list) // MAX_TICKS))
    return idx_list[::step]


def main():
    checks = load_health_checks(RUN_PATH)
    if not checks:
        print(f"⏭  {RUN_FOLDER}: no health checks in the current format — skipping health check plot")
        return
    jobs = collect_chunk_jobs(RUN_PATH)

    # ── Items on the common axis: (kind, label, submitted, start, end, payload) ──
    items = []
    for c in checks:
        sub   = parse_utc(c.get("submitted_at"))
        start = parse_utc(c.get("scheduled_at")) or sub
        end   = parse_utc(c.get("finished_at"))
        items.append(dict(kind="H", number=c["number"], label=f"H{c['number']}", sub=sub, start=start, end=end, rec=c))
    for j_no, j in enumerate(jobs, start=1):
        items.append(dict(kind="J", number=j_no,
                          label=f"J{j_no}" + ("✗s" if j["submission_failed"] else "✗" if j["failed"] else ""),
                          sub=j["submitted"], start=j["start"], end=j["end"], chunks=j["chunks"],
                          failed=j["failed"]))
    far = datetime.max.replace(tzinfo=timezone.utc)
    items.sort(key=lambda it: (it["sub"] or it["start"] or far, 0 if it["kind"] == "H" else 1, it["number"]))
    for x, it in enumerate(items):
        it["x"] = x
    if ZOOM_X is not None:
        items_view = [it for it in items if ZOOM_X[0] <= it["x"] <= ZOOM_X[1]]
    else:
        items_view = items

    n_items = max(len(items_view), 1)
    lw = 5 if n_items <= 30 else 3 if n_items <= 100 else 1.5

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(10, min(24, 0.25 * n_items + 8)), 8),
                                         gridspec_kw=dict(height_ratios=[1, 1.4]))

    # ── Top: P(011) vs health check ──────────────────────────────────────────
    hcs = [it for it in items_view if it["kind"] == "H"]
    done = [it for it in hcs if it["rec"].get("P011") is not None]
    ax_top.plot([it["x"] for it in done], [it["rec"]["P011"] for it in done],
                "-", color=COLOR_CHECK, lw=1, alpha=0.6, zorder=1)
    for it in done:
        ok = it["rec"]["P011"] > HEALTH_CHECK_THRESHOLD
        ax_top.plot(it["x"], it["rec"]["P011"], "o", color=COLOR_CHECK, mec="black" if ok else "red",
                    mew=1.0 if ok else 2.0, zorder=2)
    ax_top.axhline(HEALTH_CHECK_THRESHOLD, color="black", ls="--", lw=1, alpha=0.6,
                   label=f"threshold = {HEALTH_CHECK_THRESHOLD}")
    ax_top.set_ylim(0, 1.05)
    ax_top.set_ylabel("P(011)")
    n_failed  = sum(1 for it in hcs if it["rec"].get("status") == "failed")
    n_pending = sum(1 for it in hcs if it["rec"].get("P011") is None) - n_failed
    n_failed_jobs = sum(1 for it in items_view if it["kind"] == "J" and it["failed"])
    title_suffix = "  (" + ", ".join(s for s in (
        f"{n_pending} not yet collected" if n_pending else None,
        f"{n_failed} failed" if n_failed else None,
        f"{n_failed_jobs} failed QPU job(s)" if n_failed_jobs else None,
    ) if s) + ")" if (n_pending or n_failed or n_failed_jobs) else ""
    ax_top.set_title(f"Health checks — {NAME.figure_label}" + title_suffix, fontweight="bold")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="lower right", fontsize=8)
    ck = _thin([it["x"] for it in hcs])
    ax_top.set_xticks(ck)
    ax_top.set_xticklabels([next(i["label"] for i in hcs if i["x"] == x) for x in ck], fontsize=7, color=COLOR_CHECK)
    ax_top.tick_params(axis="x", labeltop=True, labelbottom=False, top=True)

    # ── Bottom: UTC time bars ────────────────────────────────────────────────
    for it in items_view:
        color = COLOR_CHECK if it["kind"] == "H" else COLOR_JOB
        s, e = it["start"], it["end"]
        if it["kind"] == "H" and it["rec"].get("status") == "failed":
            # could not even be submitted: only the submission-attempt time is known
            if it["sub"] is not None:
                ax_bot.plot(it["x"], mdates.date2num(it["sub"]), "P", color=COLOR_FAILED, ms=9, zorder=3)
            continue
        if it["kind"] == "H" and it["rec"].get("status") != "complete":
            # submitted, never collected: only its submission time is known
            if it["sub"] is not None:
                ax_bot.plot(it["x"], mdates.date2num(it["sub"]), "X", color=COLOR_MISSING, ms=9, zorder=3)
            continue
        if it["kind"] == "J" and it["failed"]:
            # failed job attempt (archived or still failed): its bar if the provider reported
            # start and end, else a plus at the submission time
            if s is not None and e is not None:
                ax_bot.plot([it["x"], it["x"]], [mdates.date2num(s), mdates.date2num(e)],
                            "-", color=COLOR_FAILED, lw=lw, solid_capstyle="butt")
                ax_bot.plot(it["x"], mdates.date2num(s), "s", color=COLOR_FAILED, ms=3)
            elif (it["sub"] or s) is not None:
                ax_bot.plot(it["x"], mdates.date2num(it["sub"] or s), "P", color=COLOR_FAILED, ms=9, zorder=3)
            continue
        if it["kind"] == "J" and e is None:
            # chunk job submitted, its chunks not collected: purple cross at the submission time
            t = it["sub"] or s
            if t is not None:
                ax_bot.plot(it["x"], mdates.date2num(t), "X", color=COLOR_UNCOLLECTED_JOB, ms=9, zorder=3)
            continue
        if s is None:
            continue
        if e is None:                                           # not collected yet: hollow marker
            ax_bot.plot(it["x"], mdates.date2num(s), "o", mfc="none", mec=color, ms=5)
        else:
            ax_bot.plot([it["x"], it["x"]], [mdates.date2num(s), mdates.date2num(e)],
                        "-", color=color, lw=lw, solid_capstyle="butt")
            ax_bot.plot(it["x"], mdates.date2num(s), "s", color=color, ms=3)   # keeps short bars visible
    ax_bot.yaxis_date()
    ax_bot.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))
    ax_bot.set_ylabel("time (UTC)")
    ax_bot.grid(True, alpha=0.3)
    by_x = {it["x"]: it for it in items_view}
    hk = _thin([it["x"] for it in items_view if it["kind"] == "H"])
    jk = _thin([it["x"] for it in items_view if it["kind"] == "J"])
    bk = sorted(hk + jk)                         # health checks (orange) and jobs (blue) both on the bottom axis
    ax_bot.set_xticks(bk)
    ax_bot.set_xticklabels(
        [by_x[x]["label"] if by_x[x]["kind"] == "H"
         else f"{by_x[x]['label']}\n{format_chunk_range(by_x[x]['chunks'])}" for x in bk],
        fontsize=7, rotation=90 if len(bk) > 12 else 0)
    failed  = lambda it: it["kind"] == "H" and it["rec"].get("status") == "failed"
    missing = lambda it: it["kind"] == "H" and it["rec"].get("status") not in ("complete", "failed")
    for tick, x in zip(ax_bot.get_xticklabels(), bk):
        it = by_x[x]
        tick.set_color(COLOR_FAILED if failed(it) else COLOR_MISSING if missing(it) else
                       COLOR_CHECK if it["kind"] == "H" else
                       COLOR_FAILED if it["failed"] else
                       COLOR_UNCOLLECTED_JOB if it["end"] is None else COLOR_JOB)
    for tick, x in zip(ax_top.get_xticklabels(), ck):
        if failed(by_x[x]):
            tick.set_color(COLOR_FAILED)
        elif missing(by_x[x]):
            tick.set_color(COLOR_MISSING)
    ax_bot.set_xlabel("health check number (orange)   /   QPU job with its chunk numbers (blue)")
    xlim = (min((it["x"] for it in items_view), default=0) - 0.8, max((it["x"] for it in items_view), default=0) + 0.8)
    ax_bot.set_xlim(*xlim)
    ax_top.set_xlim(*xlim)      # panels are aligned by hand (not sharex): each has its own tick labels
    from matplotlib.lines import Line2D
    ax_bot.legend(handles=[Line2D([0], [0], color=COLOR_CHECK, lw=4, label="health check"),
                           Line2D([0], [0], color=COLOR_JOB, lw=4, label="QPU job (chunks)"),
                           Line2D([0], [0], marker="X", color=COLOR_UNCOLLECTED_JOB, ls="",
                                  label="QPU job uncollected (submission time)"),
                           Line2D([0], [0], marker="X", color=COLOR_MISSING, ls="",
                                  label="health check uncollected (submission time)"),
                           Line2D([0], [0], marker="P", color=COLOR_FAILED, ls="",
                                  label="health check could not be submitted (attempt time)")]
                          + ([Line2D([0], [0], color=COLOR_FAILED, lw=4,
                                     label="failed QPU job J✗ (bar, or + at submission); J✗s: submission failed")]
                             if n_failed_jobs else []),
                  loc="upper left", fontsize=8)
    fig.tight_layout()
    add_circuit_check_banner(fig, load_run_verification(RUN_PATH, SUFFIX))
    block = load_run_completeness(RUN_PATH, SUFFIX) or completeness(RUN_PATH, INFO.sample_size, N_SAMPLES)
    add_completeness_banner(fig, block)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = PLOTS_DIR / f"health_checks{SUFFIX}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved → {rel_path(fname)}")

    # ── JSON summary ─────────────────────────────────────────────────────────
    out = RUN_PATH / "processed_results" / f"health_checks{SUFFIX}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "threshold":     HEALTH_CHECK_THRESHOLD,
        "completeness":  block,
        "uncollected_health_checks": [c["number"] for c in checks if c.get("status") != "complete"],
        "failed_health_checks": [c["number"] for c in checks if c.get("status") == "failed"],
        "health_checks": checks,
        "failed_chunk_jobs": [dict(job=it["number"], chunks=it["chunks"], submitted_at=_iso(it["sub"]),
                                   submission_failed=it["label"].endswith("✗s"))
                              for it in items if it["kind"] == "J" and it["failed"]],
        "jobs": [dict(job=it["number"], chunks=it["chunks"], chunk_range=format_chunk_range(it["chunks"]),
                      submitted_at=_iso(it["sub"]), start=_iso(it["start"]), end=_iso(it["end"]),
                      failed=it["failed"])
                 for it in items if it["kind"] == "J"],
    }, indent=2), encoding="utf-8")
    print(f"✅ Saved → {rel_path(out)}")


if __name__ == "__main__":
    main()
