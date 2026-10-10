"""
job_records.py
Light helpers (no qat/qiskit imports) for the chunk job records of a real-QPU run:
which chunks still have to be submitted, the file-name rule for a chunk record, and
the archive of failed job attempts that the post-processing counts as extra cost.

File layout:
    datafiles/<run_folder>/job_records/job_record_chunk_<nc>.json          (current attempt)
    datafiles/<run_folder>/job_records/failed_attempts/
        job_record_chunk_<nc>_failed_<k>.json                              (k-th failed attempt, k = 1, 2, ...)

A failed record is archived just before a resubmission overwrites it, so each attempt's
QPU, queue, submission and collection time stays on disk. The archive is a subfolder, so
every glob over job_records/*.json keeps seeing only the current records.
"""

import json
import re
from pathlib import Path

from source.core.common import read_json_or_none as _read_json

FAILED_DIRNAME  = "failed_attempts"
_CHUNK_RECORD   = re.compile(r"^job_record_chunk_(\d+)\.json$")
_FAILED_RECORD  = re.compile(r"^job_record_chunk_(\d+)_failed_(\d+)\.json$")


# Keys shown first in a written record, so status and failure are seen at a glance; the rest
# keep their order. Readers never depend on key order.
RECORD_HEAD_KEYS = ("chunk_id", "status", "failure", "failure_stage", "may_exist_on_provider",
                    "service_name", "backend_name", "job_id", "submission_id")


def ordered_record(record):
    """record with RECORD_HEAD_KEYS (those present) first."""
    head = {k: record[k] for k in RECORD_HEAD_KEYS if k in record}
    return {**head, **{k: v for k, v in record.items() if k not in head}}


# CONFIG keys that hold this machine's absolute paths: never saved (they leak the username and
# break once the data moves); a reader resolves the run folder against the project root instead.
MACHINE_SPECIFIC_CONFIG_KEYS = ("folder_path", "exp_folder", "exp_folder_nec")


def saved_config(config):
    """The part of a run's CONFIG that goes into a saved record: JSON-writable values only (plain
    types and dicts such as coherent_noise; numpy arrays are dropped), without the
    machine-specific paths (MACHINE_SPECIFIC_CONFIG_KEYS)."""
    return {k: v for k, v in config.items()
            if k not in MACHINE_SPECIFIC_CONFIG_KEYS and (isinstance(v, (str, int, float, bool, list, type(None))) or _json_writable(v))}


def _json_writable(value):
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def run_relative_path(run_path, *parts):
    """'datafiles/<run folder>/<parts...>' with forward slashes — how a record stores a file of
    its run: never an absolute path (it leaks the machine layout and breaks when the folder moves).
    A run in a subfolder of datafiles/ (tests/, old_runs_full/) keeps it: everything after the last
    'datafiles' in the path; a path without one is taken as the run folder's name alone."""
    names = [s for s in str(run_path).replace("\\", "/").split("/") if s not in ("", ".")]
    run   = names[len(names) - names[::-1].index("datafiles"):] if "datafiles" in names else names[-1:]
    return "/".join(["datafiles", *run, *parts])


def write_job_record(path, record):
    """Write a chunk job record, status near the top (ordered_record)."""
    Path(path).write_text(json.dumps(ordered_record(record), indent=2), encoding="utf-8")


def chunk_of_record_file(path):
    """Chunk number of a current record file job_record_chunk_<nc>.json, else None
    (archived attempts, health check records and other files)."""
    m = _CHUNK_RECORD.match(Path(path).name)
    return int(m.group(1)) if m else None


def chunk_record_files(job_records_dir):
    """{chunk: path} of the current chunk records in job_records_dir, sorted by chunk."""
    d = Path(job_records_dir)
    found = {}
    for p in d.glob("job_record_chunk_*.json") if d.is_dir() else []:
        nc = chunk_of_record_file(p)
        if nc is not None:
            found[nc] = p
    return dict(sorted(found.items()))


def chunks_without_submission(run_path, n_chunks):
    """The chunks of [0, n_chunks) whose job record is missing or not 'submitted' /
    'complete' (e.g. 'failed') — what a real-QPU submit script still has to submit. The
    script passes this list to submit(chunks=...) and to its health checks. Needs no built
    bacq_ps_qbcs, so a script can stop before building when it is empty.
    (bacq_ps_qbcs.chunks_to_submit() applies the same rule plus the local simulator's.)"""
    rec_dir = Path(run_path) / "job_records"
    todo = []
    for nc in range(n_chunks):
        rec = _read_json(rec_dir / f"job_record_chunk_{nc}.json")
        if not (isinstance(rec, dict) and rec.get("status") in ("submitted", "complete")):
            todo.append(nc)
    return todo


# ── Failed attempts ───────────────────────────────────────────────────────────
def failed_dir(job_records_dir):
    return Path(job_records_dir) / FAILED_DIRNAME


def _archived(job_records_dir, nc):
    """[(k, path)] of chunk nc's archived failed attempts, sorted by k."""
    d = failed_dir(job_records_dir)
    out = []
    for p in d.glob(f"job_record_chunk_{nc}_failed_*.json") if d.is_dir() else []:
        m = _FAILED_RECORD.match(p.name)
        if m and int(m.group(1)) == nc:
            out.append((int(m.group(2)), p))
    return sorted(out)


def archive_failed_record(job_records_dir, nc):
    """If chunk nc's current record is 'failed', save it (plus archived_at and superseded fields) as
    failed_attempts/job_record_chunk_<nc>_failed_<k>.json (k = next free number) before a
    resubmission overwrites it. Idempotent: a record already archived (same job_id and
    submitted_at) is not archived again. Returns the archive path, or None. Never raises."""
    from source.core.records.health_check_records import now_utc_iso
    try:
        rec = _read_json(Path(job_records_dir) / f"job_record_chunk_{nc}.json")
        if not (isinstance(rec, dict) and rec.get("status") == "failed"):
            return None
        existing = _archived(job_records_dir, nc)
        for _, p in existing:
            old = _read_json(p) or {}
            if (old.get("job_id"), old.get("submitted_at")) == (rec.get("job_id"), rec.get("submitted_at")):
                return p
        k    = existing[-1][0] + 1 if existing else 1
        dest = failed_dir(job_records_dir) / f"job_record_chunk_{nc}_failed_{k}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_job_record(dest, {**rec, "archived_at": now_utc_iso(), "superseded": True})
        print(f"  [Chunk {nc}] Failed attempt archived → {FAILED_DIRNAME}/{dest.name}")
        return dest
    except Exception as e:   # archiving must never block a resubmission
        print(f"  ⚠️  [Chunk {nc}] Could not archive the failed record: {e!r}")
        return None


def failed_record_files(run_path, nchunks=None):
    """Archived failed attempts of a run as {chunk: [record, ...]} (sorted by k), scoped to
    chunks < nchunks (or to the chunk numbers in nchunks, if it is a collection). The current
    record of a chunk that is still 'failed' (not resubmitted yet) is included as its last
    attempt, so a run's failures are counted whether or not it was resubmitted."""
    from source.core.records.chunk_scope import as_chunk_set
    scope = as_chunk_set(nchunks)
    jr  = Path(run_path) / "job_records"
    out = {}
    d   = failed_dir(jr)
    for p in d.glob("job_record_chunk_*_failed_*.json") if d.is_dir() else []:
        m = _FAILED_RECORD.match(p.name)
        rec = _read_json(p) if m else None
        if isinstance(rec, dict):
            out.setdefault(int(m.group(1)), []).append((int(m.group(2)), rec))
    for nc, p in chunk_record_files(jr).items():
        rec = _read_json(p)
        if isinstance(rec, dict) and rec.get("status") == "failed":
            known = {(r.get("job_id"), r.get("submitted_at")) for _, r in out.get(nc, [])}
            if (rec.get("job_id"), rec.get("submitted_at")) not in known:
                out.setdefault(nc, []).append((10**9, rec))
    return {nc: [r for _, r in sorted(v, key=lambda t: t[0])]
            for nc, v in sorted(out.items()) if scope is None or nc in scope}
