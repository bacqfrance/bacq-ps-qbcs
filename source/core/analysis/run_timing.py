"""
run_timing.py
A run's time and workload from its records and timestamp logs: CPU (transpile, submit; collection is timed separately),
QPU and queue breakdowns, failed attempts, the QPU workload that prices it, and
derive_date_time_from_data() for the summary JSON (source/analysis_scripts/bacq_ps_qbcs_summary_json.py).
"""

import glob
import json
import os
import re
from pathlib import Path

from source.core.common import get_circuit_folder, PROJECT_ROOT, scrub_paths
from source.core.records.chunk_scope import as_chunk_set, chunk_scope
from source.core.records.run_info import load_run_info
from source.core.records.timestamps import _from_timestamp_log, circuit_build_breakdown, TIMESTAMP_FILENAME
from source.core.records.job_timing import parse_utc, MQSS_TIMEZONE
from source.core.records.run_names import PCSS


# The chunk interval, used only by the legacy CPU-time fallback (runs without transpile/submit labels).
# It deliberately ignores the AQT qpu_chunk{nc}_part{p}_* labels: they are nested inside the outer
# qpu_chunk{nc}_start/_finished pair, and their timing is also in raw_results/shot_parts/.
_QPU_CHUNK_LABEL_RE = re.compile(r"^qpu_chunk(\d+)_(start|finished)$")
_SESSION_START_LABEL = "Starting chunk submission."   # a new submission session: no job interval stays open across it


# The interval of a batched IBM or IQM/VLQ job (current).
_BATCH_LABEL_RE = re.compile(r"^chunk(\d+) to chunk(\d+)_(start|end)$")
# Finer CPU labels (written inside a job's qpu_chunk/batch interval):
#   chunk{nc}_transpile_start/_end, spam_transpile_start/_end,
#   chunk{nc}_submit_start/_end, or chunk{a}-{b}_submit_start/_end for a batched IBM job
# and, written when the chunk's results are collected (not CPU time — Collect_time_s):
#   chunk{nc}_collect_start/_end
# These labels are current.
_CPU_LABEL_RE = re.compile(r"^(chunk(\d+)(?:-(\d+))?|spam)_(transpile|submit|collect)_(start|end)$")


def _job_intervals_from_rows(rows, nchunks):
    """Per-job qpu_chunk/batch intervals among chunks [0, nchunks) (or among the chunk
    numbers in nchunks, if it is a collection — source/core/records/chunk_scope.py):
    returns ({chunk: seconds} for single-chunk jobs, [(chunks, seconds)] for batches).

    The logs are append-only, so an attempt killed mid-job leaves a start without its
    finish. A start pairs only with its own finish/end, and only if no other job-level
    start (another chunk or batch, or a new submission session) comes in between —
    nothing but that job's own transpile/submit labels is written inside a job. The
    first complete pair of a chunk counts. AQT's qpu_chunk{nc}_part{p}_* labels are
    ignored by design (nested inside the chunk's pair; informational)."""
    singles, batch_list, current = {}, [], None      # current: (kind, key, start time)
    for lab, t in rows:
        m, mb = _QPU_CHUNK_LABEL_RE.match(lab), _BATCH_LABEL_RE.match(lab)
        if m and m.group(2) == "start":
            current = ("chunk", int(m.group(1)), t)
        elif mb and mb.group(3) == "start":
            current = ("batch", (int(mb.group(1)), int(mb.group(2))), t)
        elif lab == _SESSION_START_LABEL:
            current = None
        elif m and current and current[:2] == ("chunk", int(m.group(1))):
            singles.setdefault(current[1], (t - current[2]).total_seconds())
            current = None
        elif mb and current and current[:2] == ("batch", (int(mb.group(1)), int(mb.group(2)))):
            batch_list.append((current[1], (t - current[2]).total_seconds()))
            current = None
    scope   = as_chunk_set(nchunks)
    singles = {nc: s for nc, s in singles.items() if nc in scope}
    batches, seen = [], set()
    for (a, b), secs in batch_list:
        chunks = (set(range(a, b + 1)) & scope) - set(singles) - seen
        if chunks:                        # skip batches already covered by per-chunk labels
            batches.append((chunks, secs))
            seen |= chunks
    return singles, batches




def _cpu_label_times_from_rows(rows, nchunks):
    """Sums of the finer CPU labels among chunks [0, nchunks). A start pairs with the
    next end of the same name, unless the same start is logged again first (an attempt
    killed mid-way is skipped). Returns dict(transpile_s, spam_transpile_s, submit_s,
    chunks_with_labels, and collect_s / chunks_with_collect from the chunk{nc}_collect
    labels). nchunks may be a collection of chunk numbers (chunk_scope.py)."""
    scope = as_chunk_set(nchunks)
    open_starts, chunks, collected = {}, set(), set()
    out = {"transpile": 0.0, "spam_transpile": 0.0, "submit": 0.0, "collect": 0.0}
    for lab, t in rows:
        m = _CPU_LABEL_RE.match(lab)
        if m is None:
            continue
        name, first, kind, edge = m.group(1), m.group(2), m.group(4), m.group(5)
        key = (name, kind)
        if edge == "start":
            open_starts[key] = t
            continue
        if key not in open_starts:
            continue
        dt = (t - open_starts.pop(key)).total_seconds()
        if name == "spam":
            out["spam_transpile"] += dt
            continue
        a    = int(first)
        last = int(m.group(3)) if m.group(3) else a
        in_job = set(range(a, last + 1))
        if scope is not None:
            in_job &= scope
        if not in_job:
            continue
        out[kind] += dt
        (collected if kind == "collect" else chunks).update(in_job)
    return {"collect_s": out["collect"], "chunks_with_collect": collected,
            "transpile_s": out["transpile"], "spam_transpile_s": out["spam_transpile"],
            "submit_s": out["submit"], "chunks_with_labels": chunks}


def _record_files(run_path, nchunks=None):
    """job_record_chunk_*.json of a run as {chunk: record}, scoped to chunks < nchunks
    (or to the chunk numbers in nchunks, if it is a collection)."""
    from source.core.records.job_records import chunk_record_files
    out = {}
    scope = as_chunk_set(nchunks)
    for nc, p in chunk_record_files(Path(run_path) / "job_records").items():
        if scope is not None and nc not in scope:
            continue
        try:
            out[nc] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return out


def aqt_transpile_rate(exclude=None, datafiles_dir=None):
    """Measured AQT transpilation time per circuit (s), from the chunk records of every AQT run
    under datafiles/ that logged it (records written by the current code; converted old
    records have transpile_s None). Returns (rate, [run folder names used]); rate 0.0 when no
    run has it. Chunks whose transpilation was reused (transpile_reused) are skipped. exclude: a run folder to leave out. datafiles_dir: default <project>/datafiles."""
    total_s, n_circ, runs = 0.0, 0, []
    for jr in sorted(Path(datafiles_dir or PROJECT_ROOT / "datafiles").glob("*/job_records")):
        if exclude is not None and jr.parent.resolve() == Path(exclude).resolve():
            continue
        used = False
        for p in jr.glob("job_record_chunk_*.json"):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            t = rec.get("timing") or {}
            if rec.get("service_name") != PCSS or t.get("transpile_s") is None or t.get("transpile_reused"):
                continue
            total_s += float(t["transpile_s"])
            n_circ  += rec.get("n_circuits") or 0
            used = True
        if used:
            runs.append(jr.parent.name)
    return (total_s / n_circ if n_circ else 0.0), runs


def cpu_time_breakdown(run_path, nchunks, lnn, list_times, dt, total_chunks=None, qiskit_v13_qpy=False):
    """CPU time of a run (s), split into its stages:

      myqlm_build_s, qiskit_conversion_s   circuit build (shared per-topology cache log),
                                           scaled by nchunks/total_chunks for a subset
      transpile_s, spam_transpile_s,       from the run's own timestamp log (UTC file when
      submit_s                             available): chunk{nc}_transpile / spam_transpile /
                                           chunk{nc}_submit labels
      legacy_transpile_submit_s            chunks without those labels (runs before they were
                                           logged): their whole qpu_chunk interval, minus the
                                           AQT record's wall_seconds (queue + QPU, not CPU)
      health_check_transpile_s             the run's health check circuit is transpiled once per run (and
                                           saved): the sum of transpile_s over the variants in
                                           transpiled_circuits/health_check_transpiled.json (a legacy run
                                           without it: the sum over the health check records in scope).
                                           The whole run's when that file exists, also for a subset.
      health_check_submit_s                from the timing blocks of the run's health check records
      transpile_estimated_s                converted old AQT records, whose transpilation was not
                                           logged: aqt_transpile_rate() (measured s per circuit over
                                           the AQT runs that logged it) × their chunk circuits, plus
                                           the SPAM circuits once; 0 if no AQT run logged it. Their
                                           qpu_chunk interval is not used (it holds result handling
                                           and waits, not transpilation); submission is 0 (synchronous).
    """
    from source.core.records.circuit_check import run_health_check_records, health_check_in_scope

    run_path       = Path(run_path)
    # a coherent-noise run's circuits sit in their own cache (get_circuit_folder()); its noise is in the records' CONFIG
    noise = None
    for rec in _record_files(run_path).values():
        noise = (rec.get("CONFIG") or {}).get("coherent_noise")
        break
    circuit_folder = get_circuit_folder({"lnn_architecture": lnn, "T_list": list_times, "dt": dt,
                                         "coherent_noise": noise}, top_folder=True)
    build          = circuit_build_breakdown(lnn, qiskit_v13_qpy=qiskit_v13_qpy, circuit_folder=circuit_folder)
    n_in_scope     = len(as_chunk_set(nchunks))     # nchunks: a count, or the chunk numbers in scope
    subset         = bool(total_chunks) and n_in_scope < total_chunks
    scale          = (n_in_scope / total_chunks) if subset else 1.0
    out = {k: (v * scale if isinstance(v, (int, float)) else v)
           for k, v in build.items() if k != "timestamp_source"}
    out["build_scaled_to_subset"] = scale != 1.0
    out["build_timestamp_source"] = build["timestamp_source"]

    log = run_path / TIMESTAMP_FILENAME
    labels, source = _from_timestamp_log(
        log, lambda rows: ((r := _cpu_label_times_from_rows(rows, nchunks)), bool(r["chunks_with_labels"]))
    )
    labels = labels or {"transpile_s": 0.0, "spam_transpile_s": 0.0, "submit_s": 0.0, "chunks_with_labels": set(),
                        "collect_s": 0.0, "chunks_with_collect": set()}
    def _intervals(rows):
        singles, batches = _job_intervals_from_rows(rows, nchunks)
        covered = set(singles) | {c for chunks, _ in batches for c in chunks}
        return (singles, batches), covered >= as_chunk_set(nchunks)
    intervals, _ = _from_timestamp_log(log, _intervals)
    singles, batches = intervals or ({}, [])
    records = _record_files(run_path, nchunks)

    # Converted old AQT chunks: transpilation estimated from the AQT runs that measured it.
    aqt_old = {nc: r for nc, r in records.items()
               if (r.get("timing") or {}).get("converted_from_old_record") and r.get("service_name") == PCSS
               and r.get("status") == "complete"}
    estimate = None
    if aqt_old:
        rate, rate_runs = aqt_transpile_rate(exclude=run_path)
        n_est = sum(r.get("n_circuits") or 0 for r in aqt_old.values()) + max((r.get("n_spam") or 0 for r in aqt_old.values()), default=0)  # SPAM transpiled once
        estimate = {"transpile_estimated_s": rate * n_est, "rate_s_per_circuit": rate, "n_circuits": n_est,
                    "n_chunks": len(aqt_old), "from_runs": rate_runs,
                    "method": ("measured AQT transpilation time per circuit × circuits" if rate_runs
                               else "no AQT run with measured transpilation: set to 0")}

    legacy, n_legacy = 0.0, 0
    for nc, secs in singles.items():
        if nc in labels["chunks_with_labels"] or nc in aqt_old:
            continue
        rec = records.get(nc) or {}
        wall = rec.get("wall_seconds")                   # AQT: queue + QPU inside the interval, not CPU
        legacy += max(secs - float(wall), 0.0) if wall is not None else secs
        n_legacy += 1
    for chunks, secs in batches:
        if not (chunks & labels["chunks_with_labels"]):
            legacy += secs
            n_legacy += len(chunks)

    hc_transpile = hc_submit = 0.0
    for h in run_health_check_records(run_path):
        t = h.get("timing") or {}
        if not health_check_in_scope(h, nchunks if subset else None):
            continue
        hc_transpile += t.get("transpile_s") or 0.0
        hc_submit    += t.get("submit_s") or 0.0
    try:             # the transpile time of the whole run, once per saved variant (not once per record in scope)
        meta = json.loads((run_path / "transpiled_circuits" / "health_check_transpiled.json").read_text(encoding="utf-8"))
        hc_transpile = sum(float(v.get("transpile_s") or 0.0) for v in meta["variants"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass         # legacy run: the sum over the records stands

    out.update({
        "transpile_s":              labels["transpile_s"],
        "spam_transpile_s":         labels["spam_transpile_s"],
        "submit_s":                 labels["submit_s"],
        "legacy_transpile_submit_s": legacy,
        "n_chunks_legacy_cpu":      n_legacy,
        "health_check_transpile_s": hc_transpile,
        "health_check_submit_s":    hc_submit,
        "transpile_estimated_s":    estimate["transpile_estimated_s"] if estimate else 0.0,
        "transpile_estimate":       estimate,
        "run_timestamp_source":     source,
    })
    parts = ("build_total_s", "transpile_s", "spam_transpile_s", "submit_s", "legacy_transpile_submit_s",
             "health_check_transpile_s", "health_check_submit_s", "transpile_estimated_s")
    out["total_s"] = sum(out[k] or 0.0 for k in parts)
    return out


def collect_time_breakdown(run_path, nchunks=None):
    """Time spent collecting the results (s) — Collect_time_s: from a finished job to its
    saved counts, i.e. fetching the result (the first chunk of a job shared by several
    chunks carries the download), downloading the executed circuits for the circuit check
    and saving counts and records. Waiting for unfinished jobs is not included.

      chunks_s          chunk{nc}_collect_start/_end labels of the run's timestamp log (every
                        backend; AQT and local runs log the handling of the results they
                        receive synchronously)
      health_checks_s   timing.collect_s of the collected health checks

    total_s is None when nothing was measured (runs collected before the collection was
    timed — reported as "N/A", counted as 0 in the total time); n_chunks_without and
    n_health_checks_without count the collected ones without a measurement.
    nchunks: None (every chunk), a count or the chunk numbers in scope (chunk_scope.py)."""
    from source.core.records.circuit_check import run_health_check_records, health_check_in_scope
    run_path = Path(run_path)
    labels, _ = _from_timestamp_log(run_path / TIMESTAMP_FILENAME,
                                    lambda rows: ((r := _cpu_label_times_from_rows(rows, nchunks)),
                                                  bool(r["chunks_with_collect"])))
    records   = {nc: r for nc, r in _record_files(run_path, nchunks).items() if r.get("status") == "complete"}
    timed     = set(labels["chunks_with_collect"]) if labels else set()
    chunks_s  = labels["collect_s"] if labels else 0.0
    n_without = len(set(records) - timed)
    hc_s, n_hc, n_hc_without = 0.0, 0, 0
    for h in run_health_check_records(run_path):
        if h.get("status") != "complete" or not health_check_in_scope(h, nchunks):
            continue
        v = (h.get("timing") or {}).get("collect_s")
        if v is None:
            n_hc_without += 1
        else:
            hc_s += float(v)
            n_hc += 1
    measured = bool(timed) or n_hc > 0
    return {
        "total_s":                 (chunks_s + hc_s) if measured else None,
        "chunks_s":                chunks_s,
        "health_checks_s":         hc_s,
        "n_chunks_timed":          len(timed),
        "n_chunks_without":        n_without,
        "n_health_checks_without": n_hc_without,
    }


def qpu_queue_breakdown(run_path, nchunks=None, service=None):
    """QPU and queue time of a run's chunk jobs and of its accompanying health checks,
    from the records' timing blocks (old records: job_timing.legacy_record_timing()).
    A job shared by several chunks (batched IBM job) is counted once.

    Queue time: jobs queue at the same time (IBM, IQM Resonance, EQE1 accept many jobs at
    once), so their per-job waits overlap and their sum overstates the wait (EQE1: 480 jobs
    submitted at once sum to ~7e6 s). There the queue time is the longest single wait over
    all chunk jobs and health checks. Synchronous services (job_timing.SYNCHRONOUS_SERVICES:
    AQT, the local simulator) run their jobs one after another (no overlap), so there the
    per-job waits are added up. The plain sum is always kept as sum_of_jobs_s.

    Legacy runs (old IBM, IQM Resonance, IQM EQE1 records, job_timing.LEGACY_SERVICES) have
    no timing block and are read through job_timing.legacy_record_timing(); the result then
    says legacy_run=True, as it does when a converted record's QPU time could only come from those old
    fields (qpu_s_source legacy_*). Old AQT records must be converted to the current format first."""
    from source.core.records.job_timing import (service_of_record, SYNCHRONOUS_SERVICES, is_legacy_record,
                                   legacy_record_timing, qpu_execution_s)
    from source.core.records.circuit_check import run_health_check_records, health_check_in_scope
    nchunks  = chunk_scope(run_path, nchunks)           # the first nchunks collected chunks
    n_legacy = 0

    def one(rec, n_circuits, health_check=False):
        nonlocal n_legacy
        t = rec.get("timing")
        if t is not None:                  # current format (a local run has no QPU/queue keys in it)
            if str(t.get("qpu_s_source", "")).startswith("legacy_"):   # converted from the old fields alone
                n_legacy += 1
            return t
        svc = service_of_record(rec, service) or service
        if is_legacy_record(rec, svc):     # legacy run: timing from the fields the old record has
            n_legacy += 1
            return legacy_record_timing(rec, svc, n_circuits, health_check=health_check)
        raise ValueError(f"{run_path}: a {svc or 'job'} record has no timing block — it was written before "
                         f"timing was recorded. Convert the run's records to the current format first.")

    def add(bucket, t, svc):
        for kind in ("qpu", "queue"):
            v, src = t.get(f"{kind}_s"), t.get(f"{kind}_s_source") or "unavailable"
            b = bucket[kind]
            b["sources"][src] = b["sources"].get(src, 0) + 1
            if v is not None:
                b["total_s"] += float(v)
        # QPU execution only (QPU_execution_time_s): a legacy IQM record's qpu_s is already
        # its execution_started -> execution_ended interval
        if t.get("qpu_s_source") == "legacy_interval_iqm":
            v, src = t.get("qpu_s"), "execution_interval"
        else:
            v, src = qpu_execution_s(t, svc)
        b = bucket["qpu_exec"]
        b["sources"][src] = b["sources"].get(src, 0) + 1
        if v is None:
            b["n_missing"] += 1
        else:
            b["total_s"] += float(v)

    empty = lambda: {"qpu": {"total_s": 0.0, "sources": {}}, "queue": {"total_s": 0.0, "sources": {}},
                     "qpu_exec": {"total_s": 0.0, "sources": {}, "n_missing": 0}}
    chunks, hcs = empty(), empty()
    seen, n_jobs = set(), 0
    queues, sequential = {"chunks": [], "health_checks": []}, False
    records = _record_files(run_path, nchunks)
    for nc in sorted(records):
        rec = records[nc]
        if rec.get("status") != "complete":
            continue
        key = rec.get("job_id") if rec.get("job_id") not in (None, "", "N/A") else f"chunk_{nc}"
        if key in seen:
            continue
        seen.add(key)
        n_jobs += 1
        n_circ = (rec.get("n_spam") or 0) + (rec.get("n_circuits") or 0)
        t = one(rec, n_circ)
        add(chunks, t, service_of_record(rec, service) or service)
        sequential |= (service_of_record(rec, service) or service) in SYNCHRONOUS_SERVICES
        if t.get("queue_s") is not None:
            queues["chunks"].append(float(t["queue_s"]))

    n_hc = 0
    subset = nchunks
    for h in run_health_check_records(run_path):
        if h.get("status") != "complete" or not health_check_in_scope(h, subset):
            continue
        n_hc += 1
        t = one(h, 1, health_check=True)
        add(hcs, t, service_of_record(h, service) or service)
        sequential |= (service_of_record(h, service) or service) in SYNCHRONOUS_SERVICES
        if t.get("queue_s") is not None:
            queues["health_checks"].append(float(t["queue_s"]))

    # Overlapping queues (IBM, IQM): the longest single wait; sequential AQT: the sum.
    combine = sum if sequential else (lambda v: max(v, default=0.0))
    method  = "sum over jobs (synchronous service: jobs run one after another)" if sequential else \
              "longest single wait over all chunk jobs and health checks (jobs queue in parallel)"
    for bucket, key in ((chunks, "chunks"), (hcs, "health_checks")):
        bucket["queue"]["sum_of_jobs_s"] = bucket["queue"]["total_s"]
        bucket["queue"]["total_s"]       = combine(queues[key])
    return {
        "chunks":        {**chunks, "n_jobs": n_jobs},
        "health_checks": {**hcs, "n_health_checks": n_hc},
        "qpu_total_s":   chunks["qpu"]["total_s"] + hcs["qpu"]["total_s"],
        # None when any job lacks it (reported as "N/A")
        "qpu_exec_total_s": (None if chunks["qpu_exec"]["n_missing"] + hcs["qpu_exec"]["n_missing"]
                             or n_jobs + n_hc == 0
                             else chunks["qpu_exec"]["total_s"] + hcs["qpu_exec"]["total_s"]),
        "queue_total_s": combine(queues["chunks"] + queues["health_checks"]),
        "queue_method":  method,
        "legacy_run":    n_legacy > 0,
        "n_legacy_records": n_legacy,
    }


def qpu_workload(run_path, nchunks=None):
    """What a run sent to the QPU, for pricing it: jobs, circuits and shots of the
    complete chunk records and accompanying health checks — the same records
    qpu_queue_breakdown() times. Chunks sharing a job_id form one job (IBM packs 48 per
    job, IQM several in the mini benchmark); a chunk without one (AQT) and every health
    check (one circuit) are a job each. A chunk holds n_circuits + n_spam circuits, each
    executed once per entry of its shot_parts (AQT splits more than 200 shots per circuit
    into several executions; records without the field ran once). The circuits and shots
    the failed attempts executed (n_failed_jobs, n_failed_circuits, n_failed_shots, from
    failed_attempts_breakdown: AQT bills the circuits finished before failing, the other services
    each failed job that reached the provider in full) are billed too but are not counted in
    n_jobs / n_circuits / n_shots. n_may_exist_on_provider (submissions that may have run) and
    n_uncollected (jobs still 'submitted': chunks and health checks) are not counted anywhere: their cost is unknown."""
    from source.core.records.circuit_check import run_health_check_records, health_check_in_scope
    nchunks = chunk_scope(run_path, nchunks)            # the first nchunks collected chunks
    jobs = {}                                           # job key -> [circuits executed, shots]
    chunks_in_job = {}                                  # job key -> number of chunks
    circuits_per_chunk = 0
    n_chunks = 0
    n_split_chunks = 0
    records = _record_files(run_path, nchunks)
    key_of = lambda nc, rec: rec.get("job_id") if rec.get("job_id") not in (None, "", "N/A") else f"chunk_{nc}"
    for nc in sorted(records):
        rec = records[nc]
        if rec.get("status") != "complete":
            continue
        n_chunks += 1
        key = key_of(nc, rec)
        n_circ = int(rec.get("n_spam") or 0) + int(rec.get("n_circuits") or 0)
        n_exec = len(rec.get("shot_parts") or [rec["nshots"]])
        n_split_chunks += n_exec > 1
        job = jobs.setdefault(key, [0, 0])
        chunks_in_job[key] = chunks_in_job.get(key, 0) + 1
        circuits_per_chunk = max(circuits_per_chunk, n_circ)
        job[0] += n_circ * n_exec
        job[1] += n_circ * int(rec["nshots"])
    n_chunk_jobs = len(jobs)
    uncollected = {key_of(nc, rec) for nc, rec in records.items() if rec.get("status") == "submitted"}
    for i, h in enumerate(run_health_check_records(run_path)):
        if not health_check_in_scope(h, nchunks):
            continue
        if h.get("status") == "submitted":
            uncollected.add(f"health_check_{i}")
        if h.get("status") != "complete":
            continue
        jobs[f"health_check_{i}"] = [1, int(h["nshots"])]
    fa = failed_attempts_breakdown(run_path, nchunks)
    return {
        "n_chunks":          n_chunks,
        "n_jobs":            len(jobs),
        "n_chunk_jobs":      n_chunk_jobs,
        "n_health_checks":   len(jobs) - n_chunk_jobs,
        "n_circuits":        sum(c for c, _ in jobs.values()),
        "n_shots":           sum(s for _, s in jobs.values()),
        "max_circuits_per_job": max((c for c, _ in jobs.values()), default=0),
        "n_failed_jobs":     fa["n_provider_jobs"],   # failed jobs that reached the provider: not in n_jobs
        "n_failed_circuits": fa["n_circuits"],     # circuits of those jobs (AQT: finished ones): not in n_circuits
        "n_failed_shots":    fa["n_shots"],
        "failed_chunks_per_job": fa["chunks_per_job"],   # chunks in each failed chunk job (not AQT)
        "n_may_exist_on_provider": fa["n_may_exist_on_provider"],   # submissions that may have run (cost unknown)
        "n_uncollected":     len(uncollected),     # submitted, never collected: chunk jobs and health checks (cost unknown)
        "n_split_chunks":    n_split_chunks,       # chunks run as several shot parts (AQT, > 200 shots)
        "chunks_per_job":    list(chunks_in_job.values()),   # chunk jobs only
        "circuits_per_chunk": circuits_per_chunk,  # SPAM + chunk circuits (the largest, if they differ)
    }


def failed_attempts_breakdown(run_path, nchunks=None):
    """What the failed job attempts of a run cost: the records archived in
    job_records/failed_attempts/ (source/core/records/job_records.py) plus current records still 'failed'.
    A failed job shared by several chunks (batched IBM / IQM job) is counted once for its
    QPU, queue, submission and collection time; transpilation is per chunk.

      qpu_s, queue_s, collect_s   not in the run's other totals; totals_incl_failed (aggregate_run_timing)
                                  adds them, and is what the summary reports
      transpile_s, submit_s       from the records, for information only: the run's CPU time
                                  already holds them (every attempt logs its timestamp labels)
      n_circuits, n_shots         circuits (and their shots) the attempts executed before failing:
                                  n_executed_circuits / n_executed_shots of the record, else the
                                  timing block's batch_sizes and batch_shots (old AQT records).
                                  Every other service bills per job: a failed job that reached
                                  the provider (it has a job_id) counts all its chunks' circuits
                                  (n_spam + n_circuits) and their shots; a submission failure none
      n_provider_jobs             the failed jobs that reached the provider (AQT: those that
                                  finished circuits), counting a failed health check as one job
      chunks_per_job              the chunks each failed chunk job of n_provider_jobs held (not AQT)
    queues: the per-job waits (for combining with the successful jobs' queue time)."""
    from source.core.records.job_records import failed_record_files
    from source.core.records.job_timing import service_of_record, SYNCHRONOUS_SERVICES
    out = {"n_attempts": 0, "n_jobs": 0, "n_submission_failures": 0, "n_may_exist_on_provider": 0,
           "chunks": [], "qpu_s": 0.0, "qpu_sources": {}, "queue_s": 0.0,
           "collect_s": 0.0, "transpile_s": 0.0, "submit_s": 0.0, "n_jobs_without_qpu_s": 0,
           "n_circuits": 0, "n_shots": 0, "n_provider_jobs": 0, "chunks_per_job": [],
           "queues": [], "sequential": False, "failures": []}
    seen = set()
    per_job = {}                 # key of a failed job that reached the provider (not AQT) -> [chunks, circuits, shots]
    for nc, recs in failed_record_files(run_path, nchunks).items():
        out["chunks"].append(nc)
        for i, rec in enumerate(recs, start=1):
            out["n_attempts"] += 1
            t = rec.get("timing") or {}
            out["transpile_s"] += float(t.get("transpile_s") or 0.0)
            out["failures"].append({"chunk": nc, "attempt": i, "job_id": rec.get("job_id"),
                                    "submitted_at": rec.get("submitted_at"), "failure": scrub_paths(rec.get("failure")),
                                    "failure_stage": rec.get("failure_stage"),
                                    "may_exist_on_provider": rec.get("may_exist_on_provider")})
            key = rec.get("job_id") if rec.get("job_id") not in (None, "", "N/A") \
                else rec.get("submission_id") or f"chunk_{nc}_{rec.get('submitted_at')}"
            sync = service_of_record(rec) in SYNCHRONOUS_SERVICES
            reached = rec.get("job_id") not in (None, "", "N/A") and rec.get("failure_stage") != "submission"
            if reached and not sync:        # per-job billing: all the chunks' circuits of the job
                n_circ = int(rec.get("n_spam") or 0) + int(rec.get("n_circuits") or 0)
                pj = per_job.setdefault(key, [0, 0, 0])
                pj[0] += 1
                pj[1] += n_circ
                pj[2] += n_circ * int(rec.get("nshots") or 0)
            if key in seen:
                continue
            seen.add(key)
            out["n_jobs"] += 1
            if sync:
                sizes, shots = t.get("batch_sizes") or [], t.get("batch_shots") or []
                n_circ = int(rec["n_executed_circuits"] if rec.get("n_executed_circuits") is not None
                             else sum(sizes))
                out["n_circuits"] += n_circ
                out["n_shots"]    += int(rec["n_executed_shots"] if rec.get("n_executed_shots") is not None
                                         else sum(n * s for n, s in zip(sizes, shots)))
                out["n_provider_jobs"] += n_circ > 0
            if rec.get("failure_stage") == "submission":
                out["n_submission_failures"] += 1
                out["n_may_exist_on_provider"] += bool(rec.get("may_exist_on_provider"))
            out["sequential"] |= service_of_record(rec) in SYNCHRONOUS_SERVICES
            src = t.get("qpu_s_source") or "unavailable"
            out["qpu_sources"][src] = out["qpu_sources"].get(src, 0) + 1
            if t.get("qpu_s") is None:
                out["n_jobs_without_qpu_s"] += 1
            else:
                out["qpu_s"] += float(t["qpu_s"])
            if t.get("queue_s") is not None:
                out["queue_s"] += float(t["queue_s"])
                out["queues"].append(float(t["queue_s"]))
            out["collect_s"] += float(t.get("collect_s") or 0.0)
            out["submit_s"]  += float(t.get("submit_s") or 0.0)
    # Failed health checks that ran (AQT task timeout: failure_stage "execution", with a timing block):
    # their wait is queue time; they have no QPU time. Failed submissions (never executed) are not counted.
    from source.core.records.circuit_check import run_health_check_records, health_check_in_scope
    for h in run_health_check_records(run_path):
        t = h.get("timing")
        if (h.get("status") != "failed" or not health_check_in_scope(h, nchunks)
                or str(h.get("failure") or "").startswith("submission failed")):      # never reached the provider
            continue
        if (h.get("job_id") not in (None, "", "N/A") and h.get("failure_stage") != "submission"
                and service_of_record(h) not in SYNCHRONOUS_SERVICES):   # an async health check that ran: one job, one circuit
            out["n_provider_jobs"] += 1
            out["n_circuits"] += 1
            out["n_shots"]    += int(h.get("nshots") or 0)
        if not t:
            continue
        out["n_attempts"] += 1
        out["n_jobs"] += 1
        out["failures"].append({"chunk": None, "health_check": h.get("number"), "attempt": 1, "job_id": h.get("job_id"),
                                "submitted_at": h.get("submitted_at"), "failure": scrub_paths(h.get("failure")),
                                "failure_stage": h.get("failure_stage"), "may_exist_on_provider": None})
        out["transpile_s"] += float(t.get("transpile_s") or 0.0)
        out["sequential"] |= service_of_record(h) in SYNCHRONOUS_SERVICES
        src = t.get("qpu_s_source") or "unavailable"
        out["qpu_sources"][src] = out["qpu_sources"].get(src, 0) + 1
        if t.get("qpu_s") is None:
            out["n_jobs_without_qpu_s"] += 1
        else:
            out["qpu_s"] += float(t["qpu_s"])
        if t.get("queue_s") is not None:
            out["queue_s"] += float(t["queue_s"])
            out["queues"].append(float(t["queue_s"]))
        out["collect_s"] += float(t.get("collect_s") or 0.0)
        out["submit_s"]  += float(t.get("submit_s") or 0.0)
    for k, c, s in per_job.values():
        out["n_provider_jobs"] += 1
        out["chunks_per_job"].append(k)
        out["n_circuits"] += c
        out["n_shots"]    += s
    return out


def aggregate_run_timing(run_folder, n_samples=None, total_chunks=None, qiskit_v13_qpy=False, verbose=True):
    """CPU / QPU / queue time of a real-QPU (or local) run, health checks included —
    the numbers the run's summary.json reports. run_folder: subfolder of datafiles/
    or a path. n_samples: scope everything to the first n_samples collected chunks (chunk_scope)."""
    run_path = Path(run_folder) if Path(run_folder).is_absolute() else PROJECT_ROOT / "datafiles" / run_folder
    info    = load_run_info(str(run_path))
    total   = total_chunks or info.sample_size
    scope   = chunk_scope(run_path, n_samples)          # the first n_samples collected chunks
    nchunks = scope if scope is not None else total
    cpu = cpu_time_breakdown(run_path, nchunks, info.lnn, info.list_times, info.dt,
                             total_chunks=total, qiskit_v13_qpy=qiskit_v13_qpy)
    qq  = qpu_queue_breakdown(run_path, scope)
    col = collect_time_breakdown(run_path, scope)
    fa  = failed_attempts_breakdown(run_path, scope)
    # Queue time with the failed jobs: the same combination rule as the successful jobs'
    # (sum for synchronous services, the longest single wait for jobs that queue in parallel).
    queue_incl = (qq["queue_total_s"] + fa["queue_s"] if fa["sequential"]
                  else max([qq["queue_total_s"] or 0.0] + fa["queues"]))
    failed = {k: v for k, v in fa.items() if k not in ("queues", "sequential")}
    failed["cpu_included_in_cpu_total"] = True   # transpile_s / submit_s are already in cpu.total_s
    failed["totals_incl_failed"] = {
        "qpu_s":     qq["qpu_total_s"] + fa["qpu_s"],
        "queue_s":   queue_incl,
        "collect_s": (col["total_s"] or 0.0) + fa["collect_s"] if (col["total_s"] is not None or fa["collect_s"]) else None,
    }
    out = {
        "timing_definitions_version": 3,
        "cpu":   cpu,
        "collect": col,
        "qpu":   {"total_s": qq["qpu_total_s"], "chunks_s": qq["chunks"]["qpu"]["total_s"],
                  "health_checks_s": qq["health_checks"]["qpu"]["total_s"],
                  "sources": {"chunks": qq["chunks"]["qpu"]["sources"],
                              "health_checks": qq["health_checks"]["qpu"]["sources"]}},
        "qpu_execution": {"total_s": qq["qpu_exec_total_s"],   # None: not reported for every job
                          "chunks_s": qq["chunks"]["qpu_exec"]["total_s"],
                          "health_checks_s": qq["health_checks"]["qpu_exec"]["total_s"],
                          "n_jobs_without": qq["chunks"]["qpu_exec"]["n_missing"]
                                            + qq["health_checks"]["qpu_exec"]["n_missing"],
                          "sources": {"chunks": qq["chunks"]["qpu_exec"]["sources"],
                                      "health_checks": qq["health_checks"]["qpu_exec"]["sources"]}},
        "queue": {"total_s": qq["queue_total_s"], "chunks_s": qq["chunks"]["queue"]["total_s"],
                  "health_checks_s": qq["health_checks"]["queue"]["total_s"],
                  "method": qq["queue_method"],
                  "sum_of_jobs_s": {"chunks": qq["chunks"]["queue"]["sum_of_jobs_s"],
                                    "health_checks": qq["health_checks"]["queue"]["sum_of_jobs_s"]},
                  "sources": {"chunks": qq["chunks"]["queue"]["sources"],
                              "health_checks": qq["health_checks"]["queue"]["sources"]}},
        "n_jobs":          qq["chunks"]["n_jobs"],
        "n_health_checks": qq["health_checks"]["n_health_checks"],
        "legacy_run":      qq["legacy_run"],
        "failed_attempts": failed,
    }
    if verbose:
        if qq["legacy_run"]:
            print(f"  ⚠️ Legacy run: {qq['n_legacy_records']} record(s) predate the timing block; their QPU and "
                  f"queue times come from the old fields (sources marked 'legacy').")
        c = cpu
        f = lambda v: "n/a" if v is None else f"{v:.1f}s"
        print(f"  CPU time:   {f(c['total_s'])} = build {f(c['build_total_s'])} (myQLM {f(c['myqlm_build_s'])}"
              f" + Qiskit conversion {f(c['qiskit_conversion_s'])}) + transpile {f(c['transpile_s'])}"
              f" + SPAM transpile {f(c['spam_transpile_s'])} + submit {f(c['submit_s'])}"
              f" + legacy transpile/submit {f(c['legacy_transpile_submit_s'])}"
              f" + health checks {f(c['health_check_transpile_s'] + c['health_check_submit_s'])}"
              + (f" + estimated transpile {f(c['transpile_estimated_s'])}" if c.get("transpile_estimate") else ""))
        if c.get("transpile_estimate"):
            e = c["transpile_estimate"]
            print(f"  ⚠️ Transpilation of {e['n_chunks']} converted AQT chunk(s) was not logged: estimated "
                  f"{e['rate_s_per_circuit']:.4f} s/circuit × {e['n_circuits']} circuits "
                  f"({', '.join(e['from_runs']) if e['from_runs'] else 'no AQT run measured it: 0'}).")
        print(f"  QPU time (successful jobs): {f(out['qpu']['total_s'])} (chunks {f(out['qpu']['chunks_s'])}, "
              f"health checks {f(out['qpu']['health_checks_s'])}) — sources {out['qpu']['sources']}")
        print(f"  QPU execution time: {f(out['qpu_execution']['total_s'])} — sources {out['qpu_execution']['sources']}")
        print(f"  Collection (successful jobs): {f(col['total_s'])} (chunks {f(col['chunks_s'])}, health checks {f(col['health_checks_s'])})"
              + (f" — {col['n_chunks_without']} chunk(s) collected before it was timed" if col["n_chunks_without"] else ""))
        print(f"  Queue time (successful jobs): {f(out['queue']['total_s'])} (chunks {f(out['queue']['chunks_s'])}, "
              f"health checks {f(out['queue']['health_checks_s'])}) — sources {out['queue']['sources']}")
        if failed["n_attempts"]:
            ti = failed["totals_incl_failed"]
            print(f"  ⚠️ Failed attempts: {failed['n_attempts']} ({failed['n_jobs']} job(s), chunks "
                  f"{failed['chunks']}; {failed['n_submission_failures']} failed to submit"
                  + (f", {failed['n_may_exist_on_provider']} may exist on the provider" if failed["n_may_exist_on_provider"] else "")
                  + f") — QPU {f(failed['qpu_s'])}, queue {f(failed['queue_s'])}, "
                  f"collection {f(failed['collect_s'])} (their transpile/submit CPU time is already in the CPU time)"
                  + (f"; {failed['n_jobs_without_qpu_s']} job(s) report no QPU time" if failed["n_jobs_without_qpu_s"] else ""))
            print(f"     Including failed attempts (the QPU, queue and collection time of the summary): QPU {f(ti['qpu_s'])}, queue {f(ti['queue_s'])}, "
                  f"collection {f(ti['collect_s'])}")
    return out


def derive_date_time_from_data(run_folder, n_samples=None):
    """Derive the run's date/time window from its job records (n_samples subset scoping below).
    Requires job_records/job_record_chunk_*.json (submitted_at / finished_at): every real-QPU
    run, and local (service="local") runs simulated since local job records were added — older
    local runs have none.

    n_samples: if given, scope the run window to the first n_samples
    collected chunks (chunk_scope) instead of every collected chunk.

    Returns (date_qc_start, date_qc_end, submission_to_results_s): the run window as
    formatted strings (earliest chunk submission, latest chunk finish) and its length
    in seconds (None if either end is unknown)."""

    # naive job-record times are MQSS/EQE1's Europe/Berlin local time
    def _parse(s):
        return parse_utc(s, naive_tz=MQSS_TIMEZONE)

    def _fmt(dt):
        return dt.strftime("%Y-%m-%d, %H:%M") + " UTC" if dt else "N/A"

    _run_dir = PROJECT_ROOT / "datafiles" / run_folder
    _jr_dir = os.path.join(_run_dir, "job_records")
    _files  = sorted(glob.glob(os.path.join(_jr_dir, "job_record_chunk_*.json")),
                     key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
    if not _files:
        raise FileNotFoundError(f"No job_record_chunk_*.json in datafiles/{run_folder}/job_records")
    scope = as_chunk_set(chunk_scope(_run_dir, n_samples))
    if scope is not None:                       # the first n_samples collected chunks
        _files = [f for f in _files if int(f.rsplit("_", 1)[1].split(".")[0]) in scope]
    else:                                       # only chunks with results (uncollected: no finish time)
        _files = [f for f in _files if json.loads(Path(f).read_text(encoding="utf-8")).get("status") == "complete"] or _files
    _recs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in _files]

    subs = [d for d in (_parse(r.get("submitted_at")) for r in _recs) if d]
    fins = [d for d in (_parse(r.get("finished_at"))  for r in _recs) if d]
    start_dt = _fmt(min(subs)) if subs else "N/A"
    end_dt   = _fmt(max(fins)) if fins else "N/A"
    print(f"  Run window: {start_dt}  →  {end_dt}  ({len(_recs)} chunks)")

    duration_s = (max(fins) - min(subs)).total_seconds() if subs and fins else None
    return start_dt, end_dt, duration_s
