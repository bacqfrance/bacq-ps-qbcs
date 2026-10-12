"""
calibration_input.py
The calibration file a run is given (calibration_file=PATH, source/core/runner_support/benchmark_args.py), as a calibration
snapshot (source/core/qpu/qpu_snapshot.py) and an offline backend with its Target.

The file may be in either format:
  - the project's snapshot format (calibration_snapshot.json of a run, calibration_input_files/<provider>/*.json made by
    an earlier conversion): used as it is;
  - the provider's own format — AQT PIAST-Q's piastq_characterisation <date>.json or IQM Euro-Q-Exa's
    calibration_data_Euro-Q-Exa_<date>_active_qubits.json (the files in calibration_input_files/): converted with
    snapshot_from_aqt_json / snapshot_from_iqm_eqe1_json, automatically.

A run keeps its calibration: the snapshot is saved as <run>/calibration_snapshot.json, and a rerun with
the same file reuses it (a different file — another measured_at, or another content when
there is none — is an error — a run never changes calibration halfway).
Plain json + qpu_snapshot: no qiskit and no myQLM at import time.
"""

import json
import re
from pathlib import Path

from source.core.common import PROJECT_ROOT, exit_with_error, rel_path
from source.core.qpu.qpu_snapshot import (SNAPSHOT_FILENAME, snapshot_from_aqt_json, snapshot_from_iqm_eqe1_json,
                                 backend_from_snapshot)

CALIBRATION_DIR = "calibration_input_files"   # <project root>/calibration_input_files/<provider>/: the latest provider files


def detect_format(data):
    """'snapshot' | 'aqt' | 'eqe1' for a loaded calibration JSON; ValueError for anything else."""
    if isinstance(data, dict):
        if "operations" in data and "num_qubits" in data:
            return "snapshot"
        if "vendor" in data and "characterisation" in data.get("vendor", {}):
            return "aqt"
        if "T1" in data and "two_q_fidelity" in data:
            return "eqe1"
    raise ValueError("not a calibration file this project knows: expected a calibration snapshot "
                     "(operations / num_qubits), an AQT PIAST-Q characterisation file (vendor.characterisation) or an "
                     "IQM Euro-Q-Exa calibration file (T1 / two_q_fidelity)")


def source_name(path):
    """The calibration file as it is recorded in results: relative to the project root, never absolute
    (a file outside the project is recorded by its name only)."""
    p = rel_path(path)
    return Path(path).name if Path(p).is_absolute() or re.match(r"^[A-Za-z]:", p) else p


def load_calibration(path):
    """(snapshot dict, format) from a calibration file of either format. A relative `path` is relative
    to the project root (every script runs there). Exits with code 2 (exit_with_error) if the file is missing; ValueError if it is of an unknown format."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        exit_with_error(f"calibration file {path} not found (a relative path is relative to the project root; "
                        f"the latest provider files are in {CALIBRATION_DIR}/)")
    data = json.loads(p.read_text(encoding="utf-8"))
    try:
        fmt = detect_format(data)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    if fmt == "aqt":
        return snapshot_from_aqt_json(data, source_name(p)), fmt
    if fmt == "eqe1":
        return snapshot_from_iqm_eqe1_json(data, source_name(p)), fmt
    return data, fmt


def newest_provider_file(provider_dir):
    """Path of the newest file in calibration_input_files/<provider_dir>/ (the last by name; dated names sort by date)."""
    folder = PROJECT_ROOT / CALIBRATION_DIR / provider_dir
    files  = sorted(p for p in folder.glob("*.json"))
    if not files:
        exit_with_error(f"no calibration file in {rel_path(folder)}/ — pass calibration_file=...")
    return files[-1]


def use_calibration_file(path, run_path, label):
    """Put the calibration file `path` (either format; None is not allowed) in place as the run's snapshot
    <run_path>/calibration_snapshot.json, converting it if it is the provider's own. A rerun reuses the run's
    snapshot if it holds the same calibration (same measured_at, or the same content when a file has none) and exits with an error if it differs.
    Prints what was used. Returns (snapshot dict, offline backend, recorded source name)."""
    try:
        snap, fmt = load_calibration(path)
    except ValueError as e:
        exit_with_error(str(e))
    out = Path(run_path) / SNAPSHOT_FILENAME
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("measured_at") is None or snap.get("measured_at") is None:
            volatile = ("downloaded_at",)
            differs = ({k: v for k, v in old.items() if k not in volatile}
                       != {k: v for k, v in snap.items() if k not in volatile})
        else:
            differs = old["measured_at"] != snap["measured_at"]
        if differs:
            exit_with_error(f"{rel_path(out)} holds the calibration measured {old.get('measured_at') or '(no date)'}, but "
                            f"{path} holds another one (measured {snap.get('measured_at') or '(no date)'}) — "
                            f"a run keeps one calibration. Use the same file, or a new run folder.")
        snap = old
        print(f"  [Calibration] {label}: reusing the run's saved calibration ({rel_path(out)}).")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        print(f"  [Calibration] {label}: {source_name(path)} "
              f"({'already a snapshot' if fmt == 'snapshot' else 'provider format, converted'}), measured "
              f"{snap.get('measured_at')}, {snap['num_qubits']} qubits → {rel_path(out)}")
    if snap.get("excluded_qubits"):
        print(f"    Excluded qubits (incomplete calibration): {snap['excluded_qubits']}")
    print("    ⚠️  This is a calibration FILE, not live backend data.")
    return snap, backend_from_snapshot(snap), source_name(path)


def save_live_calibration(backend, run_folder):
    """Save the calibration a real run is working with — what the QPU's SDK reports now — as
    datafiles/<run_folder>/calibration_snapshot.json (VLQ: also vlq_calibration.json), so the run records the
    calibration its layout was chosen on and its results can be read against it. Done once: a run that already
    has a snapshot (a calibration file, or an earlier session of the same run) keeps it. A QPU that reports no
    calibration data (AQT, EQE1) has nothing to save. Never raises — a missing snapshot must not stop a run."""
    run_path = PROJECT_ROOT / "datafiles" / run_folder
    out = run_path / SNAPSHOT_FILENAME
    if out.exists():
        return None
    try:
        vlq_cal = getattr(backend, "vlq_calibration", None)
        run_path.mkdir(parents=True, exist_ok=True)
        if vlq_cal is not None:
            from source.core.qpu.iqm_vlq import save_vlq_calibration
            save_vlq_calibration(run_path, vlq_cal)
        else:
            from source.core.qpu.qpu_snapshot import save_calibration_snapshot
            save_calibration_snapshot(backend, out)
        print(f"  [Calibration] saved the QPU's current calibration → {rel_path(out)}")
        return out
    except ValueError:
        print("  [Calibration] this QPU reports no calibration data — nothing saved (use calibration_file= to record one).")
    except Exception as e:
        print(f"  ⚠️  [Calibration] could not save the QPU's calibration: {type(e).__name__}: {e}")
    return None


def resolve_layout_or_exit(qbcs, backend, **kwargs):
    """qbcs.resolve_physical_qubits(backend, **kwargs), but a layout that cannot be used (a forced list the QPU
    cannot run, no layout that both a calibration file and the QPU support, …) ends the script with a one-line
    message and exit code 2 — before anything is submitted — instead of a traceback."""
    try:
        return qbcs.resolve_physical_qubits(backend, **kwargs)
    except ValueError as e:
        exit_with_error(f"Layout: {e}")


def _saved_snapshot_backend(backend, run_folder):
    """The offline backend of the run's saved calibration_snapshot.json, which a live run's layout is scored on, or
    None (score on the live `backend`) when there is no usable snapshot or the QPU's circuits are laid out another
    way (VLQ scores on its vlq_calibration.json). Never raises."""
    if getattr(backend, "vlq_calibration", None) is not None:
        return None
    path = PROJECT_ROOT / "datafiles" / run_folder / SNAPSHOT_FILENAME
    try:
        return backend_from_snapshot(path) if path.exists() else None
    except Exception as e:
        print(f"  ⚠️  [Calibration] could not rebuild the backend from {rel_path(path)} "
              f"({type(e).__name__}: {e}) — scoring the layout on the live QPU.")
        return None


def resolve_run_layout(qbcs, backend, run_folder, layout, label=None):
    """What every real-QPU submit script does after build(): print the layout notice, load the calibration file
    (if any) as the run's snapshot — or else save the QPU's own current calibration (save_live_calibration) — and
    fix the run's physical qubits (qbcs.resolve_physical_qubits) with the layout `layout`
    (source/core/runner_support/benchmark_args.py LayoutArgs). The layout is scored on the saved snapshot (an offline
    backend built from it), so the snapshot is the calibration the layout was chosen on; when no snapshot could be
    saved (a QPU that reports no calibration, a failed save) it is scored on the live backend.
    Returns (physical qubits, recorded calibration source)."""
    if layout.notice:
        print(layout.notice)
    layout_backend = source = None
    if layout.calibration_file:
        _, layout_backend, source = use_calibration_file(layout.calibration_file, PROJECT_ROOT / "datafiles" / run_folder,
                                                         label or "layout scoring")
    else:
        save_live_calibration(backend, run_folder)
        layout_backend = _saved_snapshot_backend(backend, run_folder)
    qubits = resolve_layout_or_exit(qbcs, backend, run_folder=run_folder, layout=layout,
                                    layout_backend=layout_backend, calibration_source=source)
    return qubits, source


def layout_summary(layout, qubits=None, source=None):
    """One line for the run config / submit.log: the layout mode, the qubits, the calibration file."""
    return (f"{layout.mode}" + (f" {qubits}" if qubits is not None else "")
            + ((f", scored with {source}" if layout.mode == "optimized" else f", calibration {source}") if source else ""))
