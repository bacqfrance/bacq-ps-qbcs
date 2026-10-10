"""
Run identity: what a run *was* (shots, samples, depths, topology, SPAM), read back from the run itself.

The analysis and plot scripts read a run's settings (shots, samples, depths, ...) from its run_info.json,
which write_run_info() writes to datafiles/<run>/ and load_run_info() reads back. Every run has one except
the very oldest, which predate it.
"""

import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from source.core.common import PROJECT_ROOT
MANIFEST_NAME = "run_info.json"

ALL_EMS       = ["SPAM", "NEC", "RC+NEC", "cRC+NEC"]


@dataclass
class RunInfo:
    run_folder:   str
    nshots:       int
    sample_size:  int
    list_times:   list
    dt:           float
    lnn:          bool
    do_spam:      bool
    chunk_size:   int
    circuits_qiskit_v13_qpy: bool = False   # chunks from chunks_qiskit_v13_qpy/ (all_to_all builds) — see qbcs.build()
    source:       str = field(default="manifest", compare=False)   # kept so older manifests, which store it, still load

    @property
    def ems(self):
        return list(ALL_EMS) if self.do_spam else [e for e in ALL_EMS if e != "SPAM"]

    @property
    def path(self):
        return PROJECT_ROOT / "datafiles" / self.run_folder


def write_run_info(config, run_folder_path, lnn, circuits_qiskit_v13_qpy=False):
    """Called by bacq_ps_qbcs.submit() on every call (and by record_run_info()), once the run folder is known. Writes the manifest
    if missing, leaves it untouched if it already holds the same info, and rewrites it in place if it
    differs. The write itself is atomic (temp file + os.replace()), so a crash mid-write can't leave a
    truncated manifest. Skipping unchanged writes is what avoids the Windows problem: replacing the same
    file on every per-chunk submit() raced with Windows file locks (PermissionError); a short retry
    covers the rare lock that still coincides with a real change."""
    run_folder_path = Path(run_folder_path)
    run_folder_path.mkdir(parents=True, exist_ok=True)
    info = RunInfo(
        run_folder  = run_folder_path.name,
        nshots      = int(config["shots"]),
        sample_size = int(config["samples"]),
        list_times  = [float(t) for t in config["T_list"]],
        dt          = float(config["dt"]),
        lnn         = bool(lnn),
        do_spam     = bool(config.get("SPAM_correction", False)),
        chunk_size  = int(config["chunk_size"]),
        circuits_qiskit_v13_qpy = bool(circuits_qiskit_v13_qpy),
    )
    manifest = run_folder_path / MANIFEST_NAME
    if manifest.exists():
        try:
            if json.loads(manifest.read_text(encoding="utf-8")) == asdict(info):
                return info
        except (json.JSONDecodeError, OSError):
            pass   # unreadable/corrupt manifest — rewrite it below
    tmp = run_folder_path / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(asdict(info), indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, manifest)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))
    return info


def load_run_info(run_folder):
    """RunInfo for datafiles/<run_folder>, from its run_info.json (first written by record_run_info() after build())."""
    manifest = PROJECT_ROOT / "datafiles" / run_folder / MANIFEST_NAME
    if not manifest.exists():
        raise FileNotFoundError(f"datafiles/{run_folder}/{MANIFEST_NAME} not found — a run folder gets one when the run is built; "
                                f"is {run_folder!r} the right folder, and has it been built?")
    d = json.loads(manifest.read_text(encoding="utf-8"))
    d.pop("source", None)
    if "circuits_aqt_format" in d:   # the key's name before the QPY-v13 chunks were named after their format
        d["circuits_qiskit_v13_qpy"] = d.pop("circuits_aqt_format")
    return RunInfo(**d)


def resolve_run(run_folder):
    """(run_folder, RunInfo) for an analysis script. run_post_process.py passes QPU_RUN_FOLDER, which wins;
    standalone, the script's own RUN_FOLDER constant is used. Either way the settings the script then uses
    (nshots, lnn, dt, do_spam, ...) are the run's own, from its run_info.json."""
    run_folder = os.environ.get("QPU_RUN_FOLDER") or run_folder
    return run_folder, load_run_info(run_folder)
