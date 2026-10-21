"""
job_timing.py
Per-job timing written into the job records at submit/collect time, and the time
conversions it needs. No myQLM and no module-level qiskit/provider imports, so every
venv can load it — the real-QPU venvs have no myQLM, and the AQT one runs an older Qiskit.

Every chunk record and every health check record gets a `timing` block:

    transpile_s            s spent transpiling this job's own circuits (None when a cached
                           transpilation was reused — transpile_reused=True)
    submit_s               s spent in the provider's submission call (0 for AQT, by definition)
    qpu_s                  QPU time, see QPU_S_SOURCES for what it means per backend
    qpu_s_source           where qpu_s comes from
    queue_s                time between the end of submission and the start of execution
    queue_s_source         where queue_s comes from
    ...                    provider-specific extras (timestamps, billed RU, model values)

QPU time per backend (the definitions agreed for the benchmark):
    IBM             billed time, metrics()["usage"]["qpu_charge_time_seconds"]
    IQM Resonance   billed time, raw GET jobs/{id} -> metrics.billed_seconds
    IQM VLQ         Resonance-equivalent billed time, ceil(compilation + execution) per job
                    from IQM's server timeline (VLQ bills differently — TIMING_NOTES["QAAS_IT4I"])
    IQM EQE1        EQE_QPU_S_PER_CIRCUIT per circuit (the service reports no QPU time)
    AQT PIAST-Q     aqt_qpu_s_per_circuit(shots) per circuit, chunks and health checks alike:
                    3 s up to 10 shots, 15 s at 200 shots (AQT's maximum), linear in between
                    (the service reports no QPU time)
Queue time per backend:
    IBM             first execution span start - job created
    IQM Resonance   compilation_started - created (server timeline)
    IQM VLQ         compilation_started (server) - end of our upload: IT4I's HEAppE queue + IQM's own queue
    IQM EQE1        timestamp_scheduled - timestamp_submitted (server, Europe/Berlin -> UTC)
    AQT PIAST-Q     per backend.run() batch: its submit->result wall time - its QPU time
"""

import math
import re
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from source.core.common import scrub_paths
from source.core.records.run_names import IBM_CLOUD, IQM_RESONANCE, QAAS_IT4I, MQSS_LRZ, PCSS

# Services whose submission call blocks until the results are back, so their jobs run one
# after another: their queue times are added up (the others' overlap — the longest counts).
SYNCHRONOUS_SERVICES = (PCSS, "local")

EQE_QPU_S_PER_CIRCUIT = 0.9     # IQM EQE1 (MQSS): flat QPU time per circuit
# AQT PIAST-Q: QPU time per circuit, linear in the shot count between these two points
AQT_QPU_S_MIN, AQT_SHOTS_AT_MIN = 3.0, 10     # up to 10 shots: 3 s per circuit
AQT_QPU_S_MAX, AQT_MAX_SHOTS    = 15.0, 200   # 200 shots (AQT's maximum): 15 s per circuit


def aqt_shot_parts(nshots, max_shots=AQT_MAX_SHOTS):
    """Split nshots per circuit into the fewest AQT executions of at most max_shots each,
    as evenly as possible (they add up to nshots): 600 -> [200, 200, 200], 401 -> [134, 134, 133]."""
    nshots = int(nshots)
    if nshots <= 0:
        raise ValueError(f"aqt_shot_parts: nshots must be positive, got {nshots}")
    k    = math.ceil(nshots / max_shots)
    q, r = divmod(nshots, k)
    return [q + 1] * r + [q] * (k - r)


def aqt_qpu_s_per_circuit(shots):
    """QPU time (s) of one AQT circuit run with `shots` shots. AQT accepts at most
    AQT_MAX_SHOTS: above that the time is capped at AQT_QPU_S_MAX and a warning printed
    (never raised, so a record of a finished job is always written)."""
    shots = int(shots)
    if shots > AQT_MAX_SHOTS:
        print(f"  ⚠️ AQT job with {shots} shots per circuit, above AQT's maximum of {AQT_MAX_SHOTS}: "
              f"its QPU time is capped at {AQT_QPU_S_MAX:g} s per circuit.")
        return AQT_QPU_S_MAX
    if shots <= AQT_SHOTS_AT_MIN:
        return AQT_QPU_S_MIN
    return AQT_QPU_S_MIN + (AQT_QPU_S_MAX - AQT_QPU_S_MIN) * (shots - AQT_SHOTS_AT_MIN) / (AQT_MAX_SHOTS - AQT_SHOTS_AT_MIN)


MQSS_TIMEZONE = ZoneInfo("Europe/Berlin")   # MQSS/EQE1 timestamps are naive local time there

QPU_S_SOURCES = {
    "billed":                  "provider-reported billed QPU time",
    "model_eqe_per_circuit":   f"{EQE_QPU_S_PER_CIRCUIT} s per circuit (IQM EQE1 reports no QPU time)",
    "model_aqt_per_circuit":   (f"{AQT_QPU_S_MIN:g} s per circuit up to {AQT_SHOTS_AT_MIN} shots, {AQT_QPU_S_MAX:g} s "
                                f"at {AQT_MAX_SHOTS} shots, linear in between (AQT reports no QPU time)"),
    "iqm_equivalent_billed":   "IQM Resonance's billing rule applied to the server timeline: ceil(compilation + execution) per job",
    "legacy_interval_ibm":     "legacy run: scheduled_at -> finished_at interval (the billed time was not recorded)",
    "legacy_interval_iqm":     "legacy run: execution_started -> execution_ended interval (the billed time was not recorded)",
    "unavailable":             "no QPU time available",
    "not executed":            "chunk was never executed (its submission failed): no QPU time",
}

# Dead time of a run: a gap between its merged job intervals (chunk, failed-attempt and health-check jobs) longer
# than this is idle time (a failure noticed hours later, a pause), not the pipeline's own transpile/submit work.
DEAD_TIME_MIN_GAP_S      = 30 * 60
DEAD_TIME_MIN_GAP_S_SYNC = 2 * 3600   # PCSS (AQT, synchronous): its own work between jobs is longer (up to ~6400 s seen)

TIMING_NOTES = {
    IBM_CLOUD: ("QPU time is IBM's billed time (usage.qpu_charge_time_seconds, whole seconds per job). "
            "Queue time runs from job creation to the start of the first execution span; "
            "IBM's set-up after 'running' is therefore counted as queue time."),
    IQM_RESONANCE: ("QPU time is IQM Resonance's billed time (metrics.billed_seconds = "
                      "ceil(compilation_seconds + execution_seconds) per job). Queue time runs from job "
                      "creation to compilation start, so it does not overlap the billed time."),
    MQSS_LRZ: (f"IQM EQE1 (MQSS) reports no QPU or billed time, so QPU time is a flat "
                f"{EQE_QPU_S_PER_CIRCUIT} s per circuit. The measured scheduled -> completed interval "
                f"(an upper bound: it includes compilation and upload) is kept in each job record as "
                f"qpu_s_observed_upper_bound. Queue time is timestamp_scheduled - timestamp_submitted, "
                f"both converted from Europe/Berlin local time to UTC."),
    QAAS_IT4I: ("QPU time is the IQM Resonance equivalent of VLQ's jobs — ceil(compilation + execution) "
                "per job, from IQM's server timeline — so it aligns with the other QPUs. VLQ itself "
                "charges the project allocation (consumpted_resources, kept in each job record) by the "
                "QPU execution time only (qaas' remote_hw_runtime), which is QPU_execution_time_s. "
                "Queue time runs from the end of our upload to IQM's compilation start: the wait in "
                "IT4I's HEAppE job system (it4i_wait_s, upload end -> IQM 'created') plus IQM's own "
                "queue (iqm_queue_s)."),
    PCSS: (f"AQT PIAST-Q reports no QPU or billed time, so QPU time is modelled per circuit "
                    f"from its shot count: {AQT_QPU_S_MIN:g} s up to {AQT_SHOTS_AT_MIN} shots, {AQT_QPU_S_MAX:g} s "
                    f"at {AQT_MAX_SHOTS} shots (the maximum), linear in between — chunks and health checks "
                    f"alike. Submission is synchronous: each batch's "
                    f"submit -> result wall time minus its QPU time is counted as queue time, and "
                    f"submission time is zero by definition (in practice the queue dominates). The "
                    f"gate-level model aqt_circuit_qpu_us is kept in each job record as qpu_s_physics_model."),
}


# QPU_execution_time_s: the time the QPU spent executing the circuits, without compilation,
# set-up or billing rounding — where the provider reports it.
QPU_EXEC_S_SOURCES = {
    "remote_hw_runtime":         "IQM VLQ: qaas remote_hw_runtime (IQM execution_started -> execution_ended)",
    "iqm_execution_seconds":     "IQM Resonance: metrics.execution_seconds of the raw job",
    "execution_interval":        "IQM: execution_started -> execution_ended of the server timeline",
    "ibm_circuits_execution":    "IBM: metrics circuits_execution_time_ns of the job",
    "unavailable":               "not reported by the provider",
}
QPU_EXECUTION_NOTE = ("QPU_execution_time_s is the QPU's execution time alone (no compilation, set-up or "
                      "billing rounding), where the provider reports it: IQM VLQ remote_hw_runtime, IQM "
                      "Resonance metrics.execution_seconds (older records: the execution_started -> "
                      "execution_ended interval), IBM circuits_execution_time_ns. IQM EQE1, AQT PIAST-Q, "
                      "legacy IBM records and simulations report none: then it is 'N/A', as it is when any "
                      "job of the run lacks it.")


def qpu_execution_s(timing, service_name):
    """(QPU execution-only seconds, source) of one job's timing block; (None, 'unavailable')
    when the provider does not report it. See QPU_EXEC_S_SOURCES."""
    t = timing or {}
    if t.get("qpu_exec_s") is not None:
        return float(t["qpu_exec_s"]), t.get("qpu_exec_s_source") or "remote_hw_runtime"
    if service_name == IQM_RESONANCE:
        ex = (t.get("iqm_metrics") or {}).get("execution_seconds")
        if ex is not None:
            return float(ex), "iqm_execution_seconds"
        iv = seconds_between(t.get("execution_start"), t.get("execution_end"))
        if iv is not None and iv >= 0:
            return iv, "execution_interval"
    if service_name == IBM_CLOUD and t.get("circuits_execution_seconds") is not None:
        return float(t["circuits_execution_seconds"]), "ibm_circuits_execution"
    return None, "unavailable"


# ── Time conversion ───────────────────────────────────────────────────────────
_FRACTION = re.compile(r"(\.\d{6})\d+")


def now_utc_iso(precise=False):
    """Current UTC time as ISO string (read directly in UTC, not converted from local time);
    precise=True keeps microseconds."""
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ" if precise else "%Y-%m-%dT%H:%M:%SZ"
    return datetime.now(timezone.utc).strftime(fmt)


def parse_utc(value, naive_tz=None):
    """Datetime / ISO string -> aware UTC datetime. Naive values are taken as UTC, or as local
    time in zone `naive_tz` when given (local_naive_to_utc). Nanosecond precision is trimmed to
    microseconds. None / 'N/A' / unparsable -> None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s or s.upper() == "N/A":
            return None
        s = s.replace("Z", "+00:00").replace("z", "+00:00")
        s = _FRACTION.sub(r"\1", s)                    # ns precision -> us
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
    if dt.tzinfo is None:
        if naive_tz is not None:
            return local_naive_to_utc(dt, naive_tz)
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


parse_any_utc = parse_utc   # older name


def iso_utc(value):
    """Datetime / ISO string -> 'YYYY-MM-DDTHH:MM:SS.ffffffZ' (None-safe; 'N/A' -> None)."""
    dt = parse_utc(value)
    return None if dt is None else dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def local_naive_to_utc(value, tz, not_before=None):
    """Naive local time in zone `tz` -> aware UTC datetime.

    During the repeated hour at the end of daylight-saving time a naive time has two
    readings (fold=0 and fold=1). If `not_before` (aware) is given, the earliest reading
    that is not before it wins — e.g. a job cannot be scheduled before it was submitted.
    Otherwise the first reading (fold=0) is used. Already-aware values are just converted."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s or s.upper() == "N/A":
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)
    readings = sorted({dt.replace(tzinfo=tz, fold=f).astimezone(timezone.utc) for f in (0, 1)})
    if not_before is not None:
        for r in readings:
            if r >= not_before:
                return r
    return dt.replace(tzinfo=tz, fold=0).astimezone(timezone.utc)


def is_ambiguous_local(value, tz):
    """True if a naive local time falls in the repeated hour of a DST change."""
    dt = value if isinstance(value, datetime) else None
    if dt is None:
        try:
            dt = datetime.fromisoformat(str(value).strip())
        except (ValueError, TypeError):
            return False
    if dt.tzinfo is not None:
        return False
    return dt.replace(tzinfo=tz, fold=0).utcoffset() != dt.replace(tzinfo=tz, fold=1).utcoffset()


def seconds_between(start, end):
    """(end - start) in s for aware datetimes / ISO strings; None if either is missing."""
    a, b = parse_utc(start), parse_utc(end)
    if a is None or b is None:
        return None
    return (b - a).total_seconds()


# ── Submit side ───────────────────────────────────────────────────────────────
def submit_timing(transpile_s=None, transpile_reused=False, submit_s=None):
    """The submit-side part of a job's timing block."""
    return {"transpile_s": transpile_s, "transpile_reused": bool(transpile_reused), "submit_s": submit_s}


def merge_timing(record, **fields):
    """Merge fields into record['timing'] (created if missing) and return the block."""
    block = record.get("timing") or {}
    block.update(fields)
    record["timing"] = block
    return block


# ── IBM ───────────────────────────────────────────────────────────────────────
def ibm_collect_timing(job, result=None):
    """Timing of a finished IBM Runtime job. Never raises; missing values are None."""
    out = {"qpu_s": None, "qpu_s_source": "unavailable", "queue_s": None, "queue_s_source": "unavailable"}
    try:
        m = job.metrics() or {}
    except Exception as e:
        out["metrics_error"] = scrub_paths(f"{type(e).__name__}: {e}")
        m = {}
    ts    = m.get("timestamps", {}) or {}
    usage = m.get("usage", {}) or {}
    out["created_at"]  = iso_utc(parse_utc(ts.get("created")))
    out["running_at"]  = iso_utc(parse_utc(ts.get("running")))
    out["finished_at"] = iso_utc(parse_utc(ts.get("finished")))
    exec_ns = m.get("circuits_execution_time_ns")
    out["circuits_execution_seconds"] = exec_ns / 1e9 if exec_ns is not None else None
    billed = usage.get("qpu_charge_time_seconds")
    if billed is None:
        try:
            billed = job.usage()
        except Exception:
            billed = None
    if billed is not None:
        out["qpu_s"], out["qpu_s_source"] = float(billed), "billed"
    if usage.get("unit") == "resource_units":
        out["billed_resource_units"] = usage.get("value")
    try:
        out["usage_estimation_quantum_seconds"] = (job.usage_estimation or {}).get("quantum_seconds")
    except Exception:
        pass

    spans = []
    try:
        result = result if result is not None else job.result()
        for sp in result.metadata["execution"]["execution_spans"]:
            spans.append((parse_utc(sp.start), parse_utc(sp.stop)))
    except Exception:
        pass
    spans = [s for s in spans if s[0] is not None and s[1] is not None]
    if spans:
        out["execution_start"] = iso_utc(min(s[0] for s in spans))
        out["execution_end"]   = iso_utc(max(s[1] for s in spans))
        out["n_execution_spans"] = len(spans)
    out["queue_start_at"], out["queue_end_at"] = out["created_at"], out.get("execution_start")
    queue = seconds_between(out["created_at"], out.get("execution_start"))
    if queue is not None:
        out["queue_s"], out["queue_s_source"] = queue, "execution_span_start - created"
    return out


# ── IQM Resonance ─────────────────────────────────────────────────────────────
def iqm_raw_job(backend, job_id, suffix=""):
    """Raw GET jobs/{id}[/suffix] on IQM Resonance (the SDK drops billing and payload).
    Returns the decoded JSON; raises on failure."""
    import requests
    client = backend.client._iqm_server_client
    r = client._send_request(requests.get, f"jobs/{job_id}{suffix}")
    r.raise_for_status()
    return r.json()


def iqm_resonance_collect_timing(backend, job, job_id):
    """Timing of a finished IQM Resonance job: the full server timeline plus the billed
    metrics from the raw job endpoint. Never raises; missing values are None."""
    out = {"qpu_s": None, "qpu_s_source": "unavailable", "queue_s": None, "queue_s_source": "unavailable"}
    timeline = {}
    try:
        data = job._iqm_job.data
        out["timeline"] = [{"source": t.source, "status": t.status, "timestamp": iso_utc(parse_utc(t.timestamp))}
                           for t in data.timeline]
        for t in out["timeline"]:
            timeline.setdefault(t["status"], t["timestamp"])
        comp = getattr(data, "compilation", None)
        cal  = getattr(comp, "calibration_set_id", None) if comp is not None else None
        out["calibration_set_id"] = str(cal) if cal else None
    except Exception as e:
        out["timeline_error"] = scrub_paths(f"{type(e).__name__}: {e}")

    try:
        raw     = iqm_raw_job(backend, job_id)
        metrics = raw.get("metrics") or {}
        out["iqm_metrics"] = metrics
        if metrics.get("billed_seconds") is not None:
            out["qpu_s"], out["qpu_s_source"] = float(metrics["billed_seconds"]), "billed"
        comp = raw.get("compilation") or {}
        out["execution_runtime_estimate_seconds"] = comp.get("execution_runtime_estimate_seconds")
        if not out.get("calibration_set_id") and comp.get("calibration_set_id"):
            out["calibration_set_id"] = str(comp["calibration_set_id"])
        for t in raw.get("timeline") or []:        # the raw timeline, if the SDK one was unavailable
            if t.get("status") and t.get("timestamp"):
                timeline.setdefault(t["status"], iso_utc(parse_utc(t["timestamp"])))
    except Exception as e:
        out["raw_job_error"] = scrub_paths(f"{type(e).__name__}: {e}")

    out["created_at"]      = timeline.get("created")
    out["execution_start"] = timeline.get("execution_started")
    out["execution_end"]   = timeline.get("execution_ended")
    out["completed_at"]    = timeline.get("completed")
    out["queue_start_at"]  = timeline.get("created")
    out["queue_end_at"]    = timeline.get("compilation_started")
    queue = seconds_between(timeline.get("created"), timeline.get("compilation_started"))
    if queue is not None:
        out["queue_s"], out["queue_s_source"] = queue, "compilation_started - created"
    return out


# ── IQM EQE1 (MQSS) ───────────────────────────────────────────────────────────
def eqe_collect_timing(result, submitted_at_utc, n_circuits):
    """Timing of a finished EQE1 job from the MQSS result's naive Europe/Berlin
    timestamps, converted to UTC. Never raises."""
    raw = {k: (str(getattr(result, k)) if getattr(result, k, None) not in (None, "") else None)
           for k in ("timestamp_submitted", "timestamp_scheduled", "timestamp_completed")}
    ours = parse_utc(submitted_at_utc)
    # The server receives the job a moment before our own submitted_at is written, so
    # allow a little slack when resolving the repeated DST hour.
    slack = None if ours is None else ours.replace(microsecond=0) - timedelta(seconds=60)
    sub   = local_naive_to_utc(raw["timestamp_submitted"], MQSS_TIMEZONE, not_before=slack)
    sched = local_naive_to_utc(raw["timestamp_scheduled"], MQSS_TIMEZONE, not_before=sub or slack)
    comp  = local_naive_to_utc(raw["timestamp_completed"], MQSS_TIMEZONE, not_before=sched)
    out = {
        "raw_server_timestamps": raw,
        "server_submitted_at":   iso_utc(sub),
        "execution_start":       iso_utc(sched),
        "execution_end":         iso_utc(comp),
        "ambiguous_dst":         any(is_ambiguous_local(v, MQSS_TIMEZONE) for v in raw.values() if v),
        "qpu_s":                 EQE_QPU_S_PER_CIRCUIT * n_circuits,
        "qpu_s_source":          "model_eqe_per_circuit",
        "qpu_s_observed_upper_bound": seconds_between(sched, comp),
        "queue_s":               None,
        "queue_s_source":        "unavailable",
        "queue_start_at":        iso_utc(sub or ours),
        "queue_end_at":          iso_utc(sched),
    }
    if sub is not None and sched is not None:
        out["queue_s"], out["queue_s_source"] = (sched - sub).total_seconds(), "scheduled - submitted (server)"
    elif ours is not None and sched is not None:
        out["queue_s"], out["queue_s_source"] = (sched - ours).total_seconds(), "scheduled (server) - submitted_at (local clock)"
    return out


# ── AQT PIAST-Q ───────────────────────────────────────────────────────────────
def aqt_timing(batch_walls, batch_sizes, shots, qpu_s_physics_model=None, batch_shots=None):
    """Timing of a synchronous AQT chunk/health check from its per-batch wall times.
    QPU time: aqt_qpu_s_per_circuit(shots of that batch) per circuit. `shots` is the total
    per circuit; batch_shots (default: `shots` for every batch) the shots each batch ran —
    they differ when a chunk was split into several executions (aqt_shot_parts). Queue time
    of a batch = its wall time - its QPU time (clamped at 0)."""
    if batch_shots is None:
        batch_shots = [int(shots)] * len(batch_sizes)
    qpu    = [aqt_qpu_s_per_circuit(s) * n for s, n in zip(batch_shots, batch_sizes)]
    queues = [w - q for w, q in zip(batch_walls, qpu)]
    per_values = {aqt_qpu_s_per_circuit(s) for s in batch_shots}
    return {
        "submit_s":            0.0,
        "qpu_s":               sum(qpu),
        "qpu_s_source":        "model_aqt_per_circuit",
        "qpu_s_per_circuit":   per_values.pop() if len(per_values) == 1 else None,   # None: batches differ in shots
        "shots":               int(shots),
        "qpu_s_physics_model": qpu_s_physics_model,
        "queue_s":             sum(max(q, 0.0) for q in queues),
        "queue_s_source":      "batch wall time - QPU time",
        "queue_clamped":       any(q < 0 for q in queues),
        "batch_wall_seconds":  [round(w, 3) for w in batch_walls],
        "batch_sizes":         list(batch_sizes),
        "batch_shots":         [int(s) for s in batch_shots],
    }


# ── Records written before the timing block existed ───────────────────────────
# An old AQT record must be converted to the current format before the analysis; old IBM, IQM
# Resonance and IQM EQE1 records without a timing block are read as a legacy run through
# legacy_record_timing().
LEGACY_SERVICES = (IBM_CLOUD, IQM_RESONANCE, MQSS_LRZ)


def is_legacy_record(record, service_name):
    """A completed record without a timing block from a service whose old runs stay legacy."""
    return record.get("status") == "complete" and "timing" not in record and service_name in LEGACY_SERVICES


def legacy_record_timing(record, service_name, n_circuits, health_check=False):
    """Best-effort qpu_s / queue_s for an old record without a `timing` block, from the
    fields it does have. Sources are labelled so they're not mistaken for new values."""
    out = {"qpu_s": None, "qpu_s_source": "unavailable", "queue_s": None, "queue_s_source": "unavailable"}
    if service_name == PCSS:
        # The same rule as new records: wall time of the blocking call - QPU time. Old chunk
        # records kept wall_seconds; old health checks kept submitted_at/finished_at around
        # the blocking call instead.
        wall = record.get("wall_seconds")
        if wall is None:
            wall = seconds_between(record.get("submitted_at"), record.get("finished_at"))
        shots = record.get("nshots") or record.get("shots")
        if shots is None:
            raise ValueError("AQT record without nshots/shots: its QPU time can't be modelled")
        per  = aqt_qpu_s_per_circuit(shots)
        out["qpu_s"], out["qpu_s_source"] = per * n_circuits, "model_aqt_per_circuit"
        out["qpu_s_per_circuit"], out["shots"] = per, int(shots)
        if wall is not None and wall >= 0:
            out["queue_s"], out["queue_s_source"] = max(float(wall) - out["qpu_s"], 0.0), "wall time - QPU time (converted record)"
            out["queue_clamped"] = float(wall) < out["qpu_s"]
        return out
    if service_name == MQSS_LRZ:
        out["qpu_s"], out["qpu_s_source"] = EQE_QPU_S_PER_CIRCUIT * n_circuits, "model_eqe_per_circuit"
        sub   = parse_utc(record.get("submitted_at"))
        sched = record.get("scheduled_at")
        # Chunk records kept MQSS's naive Berlin string; health checks already stored it as (wrong) UTC.
        if sub is not None and sched not in (None, "N/A") and not str(sched).endswith(("Z", "+00:00")):
            sched_utc = local_naive_to_utc(sched, MQSS_TIMEZONE, not_before=sub - timedelta(seconds=60))
            out["queue_start_at"], out["queue_end_at"] = iso_utc(sub), iso_utc(sched_utc)
            out["queue_s"], out["queue_s_source"] = (sched_utc - sub).total_seconds(), "scheduled - submitted_at (legacy)"
        return out
    interval = seconds_between(record.get("scheduled_at"), record.get("finished_at"))
    if interval is not None and interval >= 0:
        out["qpu_s"], out["qpu_s_source"] = interval, ("legacy_interval_ibm" if service_name == IBM_CLOUD
                                                       else "legacy_interval_iqm")
    start = record.get("created_at") if service_name == IBM_CLOUD else record.get("submitted_at")
    out["queue_start_at"], out["queue_end_at"] = start, record.get("scheduled_at")
    queue = seconds_between(start, record.get("scheduled_at"))
    if queue is not None and queue >= 0:
        out["queue_s"] = queue
        out["queue_s_source"] = ("running - created (legacy)" if service_name == IBM_CLOUD
                                 else "execution_started - submitted_at (legacy)")
    return out


def service_of_record(record, backend_hint=None):
    """Which provider wrote a record (old records have no service_name)."""
    s = record.get("service_name")
    if s:
        return s
    name = f"{record.get('backend_name', '')} {backend_hint or ''}".lower()
    if "ibm" in name or "fake" in name:
        return IBM_CLOUD
    if "mqss" in name or "eqe" in name:
        return MQSS_LRZ
    if "aqt" in name or "pcss" in name:
        return PCSS
    if any(q in name for q in ("emerald", "garnet", "sirius", "resonance")):
        return IQM_RESONANCE
    if "vlq" in name:
        return QAAS_IT4I
    return None
