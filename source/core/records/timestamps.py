"""
timestamps.py
The timestamp CSV logs a build or a run writes (machine-local time for reading, UTC for every duration),
and the circuit build times read back from them.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path



TIMESTAMP_FILENAME         = "timestamps_local_time.csv"       # naive local (machine) time — for reading only
TIMESTAMP_FILENAME_V13     = "timestamps_qiskit_v13_qpy_local_time.csv"   # same, for all_to_all builds (chunks in QPY version 13)
TIMESTAMP_FILENAME_UTC     = "timestamps_utc.csv"              # the same rows in UTC — every duration is computed from these
TIMESTAMP_FILENAME_V13_UTC = "timestamps_qiskit_v13_qpy_utc.csv"

_LOCAL_FMT = "%Y-%m-%d %H:%M:%S.%f"
_UTC_FMT   = "%Y-%m-%dT%H:%M:%S.%fZ"


def utc_timestamp_path(local_path):
    """UTC companion of a local-time timestamp CSV (timestamps[_qiskit_v13_qpy]_local_time.csv ->
    timestamps[_qiskit_v13_qpy]_utc.csv; any other name gets '_utc' before its suffix)."""
    p = Path(local_path)
    if p.name.endswith("_local_time.csv"):
        return p.with_name(p.name[: -len("_local_time.csv")] + "_utc.csv")
    return p.with_name(p.stem + "_utc" + p.suffix)


def _parse_timestamp(s):
    """A timestamp cell -> datetime: aware UTC for the '..Z' UTC files, naive for the local ones."""
    if s.endswith("Z"):
        return datetime.strptime(s, _UTC_FMT).replace(tzinfo=timezone.utc)
    return datetime.strptime(s, _LOCAL_FMT)


def _read_timestamp_rows(path):
    """Read a timestamp CSV's data rows (label, raw-timestamp-string), skipping the header."""
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.reader(f, delimiter="\t") if len(row) == 2 and row[0] != "label"]


def _sync_circuit_timestamps(circuit_folder, target_filename, other_filename):
    """Append rows present in other_filename but missing (by label) from
    target_filename to the end of target_filename, so a build() call that skips a
    shared stage — because its output already exists from a build of the
    other chunk format (qiskit vs. qiskit_v13) — still has that stage's timestamp
    available for circuit_build_breakdown(). Never touches other_filename, and never
    rewrites target_filename (append only). The Qiskit-conversion rows are not
    copied: each format logs its own conversion.
    Works on either pair (local-time or UTC files); never mixes the two."""
    other_path = os.path.join(circuit_folder, other_filename)
    if not os.path.exists(other_path):
        return

    other_rows = _read_timestamp_rows(other_path)

    target_path = os.path.join(circuit_folder, target_filename)
    target_rows = _read_timestamp_rows(target_path) if os.path.exists(target_path) else []

    target_labels = {row[0] for row in target_rows}
    missing       = [row for row in other_rows
                     if row[0] not in target_labels and not row[0].startswith(_CONVERSION_LABELS)]
    if not missing:
        return

    os.makedirs(circuit_folder, exist_ok=True)
    write_header = not os.path.exists(target_path)
    with open(target_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow(["label", "timestamp"])
        writer.writerows(missing)


_CONVERSION_LABELS = ("Starting Qiskit conversion", "Finished Qiskit conversion")


def sync_circuit_timestamp_logs(circuit_folder, qiskit_v13_qpy):
    """_sync_circuit_timestamps() for both the local-time and the UTC pair of build logs."""
    for qiskit_name, v13_name in ((TIMESTAMP_FILENAME, TIMESTAMP_FILENAME_V13),
                                  (TIMESTAMP_FILENAME_UTC, TIMESTAMP_FILENAME_V13_UTC)):
        target, other = (v13_name, qiskit_name) if qiskit_v13_qpy else (qiskit_name, v13_name)
        _sync_circuit_timestamps(circuit_folder, target, other)


def timestamp_log_rows(local_path):
    """Rows of a timestamp log as (label, datetime), from both of its files:
    {'utc': rows of the UTC file or None, 'local': rows of the local-time file or None}.
    Durations are computed from one file only, never mixing the two — see
    _from_timestamp_log()."""
    out = {}
    for key, path in (("utc", utc_timestamp_path(local_path)), ("local", Path(local_path))):
        out[key] = ([(lab, _parse_timestamp(ts)) for lab, ts in _read_timestamp_rows(path)]
                    if path.exists() else None)
    return out


def _from_timestamp_log(local_path, compute):
    """Run compute(rows) -> (value, complete) on the UTC log; if it is missing or lacks
    labels the calculation needs (complete=False — e.g. a log started before UTC logging
    existed), run it on the local-time log instead (when there is one; otherwise the incomplete UTC
    value is returned). Returns (value, 'utc' | 'local' | None)."""
    logs = timestamp_log_rows(local_path)
    if logs["utc"]:
        value, complete = compute(logs["utc"])
        if complete or not logs["local"]:
            return value, "utc"
    if logs["local"]:
        value, _ = compute(logs["local"])
        return value, "local"
    return None, None


def _build_times_from_rows(rows):
    """(myqlm_s, conversion_s, total_s) of a circuit-build log, or None if it has no build.

    The build is the sum of its stages, not first start to last finish, so days between
    separate build sessions (e.g. the two chunk formats built apart) are
    not counted:
      myQLM:      earliest 'Starting compilation for Nt=' to the first myQLM-interleaving
                  finish after it (or, in logs without that label, the conversion start)
      conversion: the last complete 'Starting Qiskit conversion' -> 'Finished Qiskit
                  conversion' pair in the log (this format's own; the rows are appended,
                  so the last pair is the one that wrote the current chunk files)
    The legacy myQLM-only pipeline never converts: conversion is None there."""
    times = lambda prefix: sorted(t for lab, t in rows if lab.startswith(prefix))
    comps = times("Starting compilation for Nt=")
    if not comps:
        return None
    comp = comps[0]

    conv, open_start = None, None
    for lab, t in rows:                                   # file order: the last complete pair wins
        if lab.startswith("Starting Qiskit conversion"):
            open_start = t
        elif lab.startswith("Finished Qiskit conversion") and open_start is not None:
            conv, open_start = (t - open_start).total_seconds(), None

    ends = [t for t in times(("Finished myQLM interleaving", "Finished interleaving")) if t >= comp]
    if not ends:
        ends = [t for t in times("Starting Qiskit conversion") if t >= comp]
    if not ends:
        return None
    myqlm = (ends[0] - comp).total_seconds()
    return myqlm, conv, myqlm + (conv or 0.0)


def circuit_build_breakdown(lnn, circuit_folder, qiskit_v13_qpy=False):
    """Circuit-build wall-clock split into the myQLM stages (compilation + interleaving)
    and the Qiskit conversion, from the circuit-build log in `circuit_folder` (the cache
    folder from get_circuit_folder()) — the UTC
    file when it covers the build, else the local-time one. Returns a dict with
    myqlm_build_s, qiskit_conversion_s (None for old logs without the conversion
    labels), build_total_s and timestamp_source; the values are None if the log has no build."""
    filename = TIMESTAMP_FILENAME_V13 if qiskit_v13_qpy else TIMESTAMP_FILENAME
    path     = Path(circuit_folder) / filename
    times, source = _from_timestamp_log(path, lambda rows: ((r := _build_times_from_rows(rows)), r is not None))
    myqlm, conv, total = times if times else (None, None, None)
    return {"myqlm_build_s": myqlm, "qiskit_conversion_s": conv, "build_total_s": total,
            "timestamp_source": source}


def timestamp(label, timestamp_file):
    """Log `label` with the current time to timestamp_file (naive machine-local time, for
    reading) and to its UTC companion (utc_timestamp_path(), used for every duration —
    safe across daylight-saving changes). Both rows come from one clock reading."""
    now_utc = datetime.now(timezone.utc)
    now     = now_utc.astimezone().replace(tzinfo=None)
    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {label}")

    timestamp_file = Path(timestamp_file)
    timestamp_file.parent.mkdir(parents=True, exist_ok=True)  # create parent dirs if needed

    for path, value in ((timestamp_file, now.strftime(_LOCAL_FMT)),
                        (utc_timestamp_path(timestamp_file), now_utc.strftime(_UTC_FMT))):
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter='\t')
            if write_header:
                writer.writerow(["label", "timestamp"])
            writer.writerow([label, value])
