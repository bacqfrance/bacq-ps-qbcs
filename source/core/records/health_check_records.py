"""
health_check_records.py
Light helpers (no qat/qiskit imports) for the health checks that accompany a run:
storage under datafiles/<run_folder>/health_checks/, UTC time helpers (re-exported from job_timing.py), and
reconstruction of QPU jobs (groups of chunks) from the chunk job records.
Used by the pipeline, the runners and the analysis.

File layout (single counter, alongside job_records/ and raw_results/):
    datafiles/<run_folder>/health_checks/health_check_<N>.json
    datafiles/<run_folder>/chunk_times_UTC_time.json      (old local simulator runs only; local
                                                          runs now write job records like real QPUs)
    datafiles/<run_folder>/submit.log                     (append_submit_log(), every submit / simulate script)
"""

import json
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

from source.core.common import read_json_or_none as _read_json, scrub_paths
from source.core.records.job_timing import now_utc_iso as now_utc_iso, parse_utc, iso_utc as to_utc_iso   # noqa: F401  (re-exported)
from source.core.records.job_records import chunks_without_submission, run_relative_path

HEALTH_CHECK_THRESHOLD = 0.5     # pass criterion: P(011) > threshold; must be >= 0.5 (health_check_verdict relies on it)
HEALTH_CHECK_SCHEMA    = 2       # files without schema >= 2 are ignored (old format)
HEALTH_DIRNAME         = "health_checks"
LOCAL_TIMES_FILENAME   = "chunk_times_UTC_time.json"

# Exit codes the real-QPU submit scripts use when a PRE/POST health check doesn't come back
# functional (distinct from 1 = unrelated crash, 2 = argument/usage error elsewhere in this
# project, 3 = AQT's own sbatch pause-and-requeue signal, 4 = the loop script's STOP file, 14 = run_aqt_loop_with_retries.sbatch giving up — none of these are reused).
EXIT_HC_DEGRADED      = 10   # the check ran and the QPU is degraded (P(011) <= threshold)
EXIT_HC_NOT_COLLECTED = 11   # the check was submitted but no result could be collected (unknown health)
EXIT_HC_NOT_SUBMITTED = 12   # the check could not even be submitted/executed (unknown health)
EXIT_AQT_TASK_TIMEOUT = 13   # AQT only (chunk or health check): a circuit's task was not executed within qpu/aqt.py: AQT_TASK_TIMEOUT_S


# ── Health check files ────────────────────────────────────────────────────────
def health_dir(run_path):
    return Path(run_path) / HEALTH_DIRNAME


def health_check_path(run_path, number):
    return health_dir(run_path) / f"health_check_{number}.json"


def load_health_checks(run_path):
    """All new-format health checks of a run, sorted by number. Files of the old
    format (no `schema` >= HEALTH_CHECK_SCHEMA) or unreadable files are skipped."""
    d = health_dir(run_path)
    out = []
    if not d.is_dir():
        return out
    for p in d.glob("health_check_*.json"):
        rec = _read_json(p)
        if isinstance(rec, dict) and rec.get("schema", 0) >= HEALTH_CHECK_SCHEMA and "number" in rec:
            out.append(rec)
    return sorted(out, key=lambda r: r["number"])


def create_health_check(run_path, record):
    """Write `record` as the next health_check_<N>.json. The file is created
    exclusively, so concurrent scripts can't overwrite each other's number.
    Sets record['number'] / record['schema'] and returns the number."""
    d = health_dir(run_path)
    d.mkdir(parents=True, exist_ok=True)
    record["schema"] = HEALTH_CHECK_SCHEMA
    for _ in range(100):
        nums = [int(m.group(1)) for p in d.glob("health_check_*.json")
                if (m := re.fullmatch(r"health_check_(\d+)\.json", p.name))]
        number = max(nums, default=0) + 1
        record["number"] = number
        try:
            with open(health_check_path(run_path, number), "x", encoding="utf-8") as f:
                f.write(json.dumps(record, indent=2))
            return number
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate a health check number in {d}")


def save_health_check(run_path, record):
    """Atomically overwrite an existing health check file."""
    path = health_check_path(run_path, record["number"])
    tmp  = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def health_check_p011(counts):
    """P(011) of a health check's counts (the circuit prepares |011>)."""
    return counts.get("011", 0) / sum(counts.values())


def health_check_verdict(counts):
    """The verdict of a health check, the same for every QPU: "functional" if P(011) is above
    HEALTH_CHECK_THRESHOLD (which is at least 0.5, so
    011 is then also the most frequent outcome), else "degraded"."""
    return "functional" if health_check_p011(counts) > HEALTH_CHECK_THRESHOLD else "degraded"


# ── Chunks / QPU jobs ─────────────────────────────────────────────────────────
def _record_row(rec, nc, key, failed=False):
    submitted = parse_utc(rec.get("submitted_at"))
    sched     = parse_utc(rec.get("scheduled_at"))
    finished  = parse_utc(rec.get("finished_at"))
    start     = sched
    if start is None and finished is not None and rec.get("wall_seconds"):
        start = finished - timedelta(seconds=float(rec["wall_seconds"]))   # AQT
    if start is None:
        start = submitted
    return dict(chunk=int(nc), key=key, submitted=submitted, start=start, end=finished, failed=failed)


def _chunk_rows(run_path):
    """One row per submitted chunk: dict(chunk, key, submitted, start, end, failed), plus one
    row per failed job attempt (archived in job_records/failed_attempts/, or a current record
    still 'failed') with failed=True."""
    from source.core.records.job_records import chunk_record_files, failed_record_files
    run_path = Path(run_path)
    rows = []
    for nc, p in chunk_record_files(run_path / "job_records").items():
        rec = _read_json(p)
        if not isinstance(rec, dict) or rec.get("status") not in ("submitted", "complete"):
            continue
        job_id = rec.get("job_id")
        key    = job_id if job_id not in (None, "", "N/A") else f"chunk_{nc}"
        rows.append(_record_row(rec, nc, key))
    for nc, recs in failed_record_files(run_path).items():
        for i, rec in enumerate(recs, start=1):
            job_id = rec.get("job_id")
            key    = (f"failed:{job_id}" if job_id not in (None, "", "N/A")
                      else f"failed:{rec['submission_id']}" if rec.get("submission_id")
                      else f"failed:chunk_{nc}_{i}")
            row = _record_row(rec, nc, key, failed=True)
            row["submission_failed"] = rec.get("failure_stage") == "submission"
            rows.append(row)
    # Old local simulator runs: chunk_times_UTC_time.json, for the chunks without a record (a run
    # started before local job records existed and resumed after has some of each).
    with_record = {r["chunk"] for r in rows if not r.get("failed")}
    data = _read_json(run_path / LOCAL_TIMES_FILENAME)
    if isinstance(data, dict):
        for k, t in data.items():
            m = re.fullmatch(r"chunk_(\d+)", k)
            if not m or not isinstance(t, dict) or int(m.group(1)) in with_record:
                continue
            s, e = parse_utc(t.get("started_at")), parse_utc(t.get("finished_at"))
            rows.append(dict(chunk=int(m.group(1)), key=k, submitted=s, start=s, end=e))
    return rows


def collect_chunk_jobs(run_path):
    """QPU jobs of a run reconstructed from the chunk records (or from
    chunk_times_UTC_time.json): chunks sharing a job_id form one job; chunks without
    a real job_id (AQT, local simulator) are one job each.

    Returns a list of dicts sorted by submission time:
        chunks (sorted list), submitted / start / end (aware UTC datetimes or None),
        failed (True for a failed job attempt),
        submission_failed (True if a chunk's submission failed).
    `end` is None until the job's results have been collected (or, for a failed job,
    when the provider reported no end time)."""
    groups = {}
    for r in _chunk_rows(run_path):
        groups.setdefault(r["key"], []).append(r)
    jobs = []
    for rows in groups.values():
        starts = [r["start"] for r in rows if r["start"] is not None]
        subs   = [r["submitted"] for r in rows if r["submitted"] is not None]
        ends   = [r["end"] for r in rows]
        jobs.append(dict(
            chunks    = sorted(r["chunk"] for r in rows),
            submitted = min(subs) if subs else None,
            start     = min(starts) if starts else None,
            end       = max(e for e in ends if e is not None) if all(e is not None for e in ends) else None,
            failed    = any(r.get("failed") for r in rows),
            submission_failed = any(r.get("submission_failed") for r in rows),
        ))
    far_future = datetime.max.replace(tzinfo=timezone.utc)
    jobs.sort(key=lambda j: (j["submitted"] or j["start"] or far_future, j["chunks"][0]))
    return jobs




def install_submit_crash_log(run_folder_path, todo):
    """Make an uncaught exception in a submit or local simulation script (network error, failed
    health-check assert, Ctrl+C, ...) append an entry to <run>/submit.log before the traceback is
    printed. The scripts write their normal entry (append_submit_log) only after everything
    succeeded, so a crashed attempt that had already submitted chunks left no trace there. The
    entry lists which of the requested chunks had been submitted (a submitted or complete job record) by then."""
    import sys
    run_path, todo, previous = Path(run_folder_path), list(todo), sys.excepthook

    def _hook(exc_type, exc, tb):
        try:
            if run_path.is_dir():
                still_open = set(chunks_without_submission(run_path, max(todo) + 1 if todo else 0))
                done = [nc for nc in todo if nc not in still_open]
                with open(run_path / "submit.log", "a", encoding="utf-8") as f:
                    f.write(f"Submitted at:     {now_utc_iso()}\n")
                    f.write(f"Run folder:       {run_relative_path(run_path)}\n")
                    f.write(f"INTERRUPTED:      {scrub_paths(f'{exc_type.__name__}: {exc}')[:300]}\n")
                    f.write(f"Chunks submitted before it: {format_chunk_range(done) or 'none'} "
                            f"({len(done)} of the {len(todo)} requested)\n\n")
        except Exception:
            pass   # never hide the real error
        previous(exc_type, exc, tb)

    sys.excepthook = _hook


def append_submit_log(run_folder_path, *, backend, chunks, n_chunks, nshots, layout=None, run_qubits=None,
                      snapshot_path=None, extra_lines=(), post_hc_ok=True, post_hc_numbers=()):
    """Append one submission's entry to <run>/submit.log — the same layout for every real-QPU
    submit script and every local simulation script: time, backend, run folder (relative), the
    chunks of this submission, shots, then (in this order, each when given) the layout summary
    and the run qubits, the latter followed by the calibration summary of snapshot_path
    (default: the run's calibration_snapshot.json; nothing if there is none), then extra_lines (provider-specific),
    the run's health-check numbers and, if the POST check could not be submitted, a line saying
    so. Returns the log's path."""
    run_path = Path(run_folder_path)
    run_lines = [f"Layout:           {layout}"] if layout is not None else []
    if run_qubits is not None:
        from source.core.qpu.qpu_snapshot import SNAPSHOT_FILENAME, snapshot_log_lines
        run_lines += [f"Run qubits:       {run_qubits}",
                      *snapshot_log_lines(snapshot_path or run_path / SNAPSHOT_FILENAME, run_qubits)]
    lines = [f"Submitted at:     {now_utc_iso()}",
             f"Backend:          {backend}",
             f"Run folder:       {run_relative_path(run_path)}",   # relative, forward slashes
             f"Chunks submitted: {format_chunk_range(chunks) or 'none'} ({len(chunks)} of {n_chunks})",
             f"Shots:            {nshots}",
             *run_lines,
             *extra_lines,
             f"Health checks:    {[c['number'] for c in load_health_checks(run_path)]}"]
    if not post_hc_ok:
        lines.append(f"POST health check NOT submitted (attempts {', '.join(f'#{n}' for n in post_hc_numbers)} "
                     f"failed) — chunks are still valid.")
    log_path = run_path / "submit.log"
    run_path.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:   # UTF-8: the log holds µ (T1/T2 in µs)
        f.write("\n".join(lines) + "\n\n")             # blank line between the entries of successive submissions
    return log_path


def health_check_log_line(label, record):
    """'PRE  health check #3: P(011)=0.970 → functional' — a submit.log line for a collected check."""
    if not record or record.get("P011") is None:
        return f"{label} health check: not collected"
    return f"{label} health check #{record.get('number')}: P(011)={record['P011']:.3f} → {record.get('verdict')}"


def print_run_config(run_folder, *, nshots, sample_size, lnn, do_spam, list_times, chunk_size, extra=()):
    """The [Run config] block every submit / simulate script prints. extra: (label, value) pairs."""
    rows = [("Run folder", run_folder), ("Shots", nshots), ("Samples", sample_size), ("LNN", lnn),
            ("SPAM", do_spam), ("Depths", f"{len(list_times)} ({list_times[0]:.1f} → {list_times[-1]:.1f})"),
            ("Chunk size", chunk_size), *extra]
    print("\n[Run config]")
    for label, value in rows:
        print(f"  {label + ':':<17} {value}")


def format_chunk_range(chunks):
    """[0,1,2,3,7] -> '0-3,7'."""
    chunks = sorted(chunks)
    parts, i = [], 0
    while i < len(chunks):
        j = i
        while j + 1 < len(chunks) and chunks[j + 1] == chunks[j] + 1:
            j += 1
        parts.append(str(chunks[i]) if i == j else f"{chunks[i]}-{chunks[j]}")
        i = j + 1
    return ",".join(parts)
