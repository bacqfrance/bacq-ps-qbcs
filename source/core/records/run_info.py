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

from source.core.common import PROJECT_ROOT, exit_with_error
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
    instance:      str = None   # IBM only: the name of the IBM instance that received the run's jobs (never the CRN) and its plan; left out of the manifest when None
    instance_plan: str = None
    scheduling_simulation: bool = None   # QPU-noise simulations only: whether the circuits were transpiled scheduled (idle qubits relax); a run folder must not mix the two (check_scheduling_simulation()); left out of the manifest when None (real runs, noiseless simulations); a noisy simulation without it counts as True
    physical_qubits: list = None   # the qubits every circuit is transpiled onto, as in transpiled_circuits/physical_qubits.json (resolve_physical_qubits()); left out of the manifest when None (not chosen yet, noiseless simulations, legacy per-circuit layouts)
    source:       str = field(default="manifest", compare=False)   # written into every manifest; compare=False only removes it from dataclass equality, write_run_info()'s rewrite check compares the whole dict, so a manifest holding another source is rewritten; load_run_info() drops it

    @property
    def ems(self):
        return list(ALL_EMS) if self.do_spam else [e for e in ALL_EMS if e != "SPAM"]


def _manifest_dict(info):
    d = asdict(info)
    for key in ("instance", "instance_plan", "scheduling_simulation", "physical_qubits"):
        if d[key] is None:
            del d[key]
    return d


def recorded_instance(run_folder, root="datafiles"):
    """(instance name, plan) recorded in root/<run_folder>/run_info.json, (None, None) without a manifest or a record."""
    try:
        d = json.loads((Path(root) / run_folder / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    return d.get("instance"), d.get("instance_plan")


def recorded_physical_qubits(run_folder, root="datafiles"):
    """physical_qubits recorded in root/<run_folder>/run_info.json: None without a manifest or a record."""
    try:
        return json.loads((Path(root) / run_folder / MANIFEST_NAME).read_text(encoding="utf-8")).get("physical_qubits")
    except (OSError, json.JSONDecodeError):
        return None


def recorded_scheduling_simulation(run_folder, root="datafiles"):
    """scheduling_simulation recorded in root/<run_folder>/run_info.json: None without a manifest or a record."""
    try:
        return json.loads((Path(root) / run_folder / MANIFEST_NAME).read_text(encoding="utf-8")).get("scheduling_simulation")
    except (OSError, json.JSONDecodeError):
        return None


def check_scheduling_simulation(run_folder, scheduling_simulation, root="datafiles"):
    """Exit with code 2 (one ❌ line on stderr) if root/<run_folder> already holds a QPU-noise simulation made with the
    other scheduling_simulation= value: its transpiled circuits are reused on a resume, so a run folder never mixes
    scheduled and unscheduled circuits. A run folder with a manifest but no record counts as True; without a manifest
    (a new run) nothing is checked."""
    if not (Path(root) / run_folder / MANIFEST_NAME).exists():
        return
    recorded = recorded_scheduling_simulation(run_folder, root)
    recorded = True if recorded is None else recorded
    if recorded != scheduling_simulation:
        exit_with_error(f"run folder {run_folder} was simulated with scheduling_simulation={str(recorded).lower()}, not "
                        f"{str(scheduling_simulation).lower()}: its transpiled circuits are reused, so continue it with "
                        f"scheduling_simulation={str(recorded).lower()}, or delete the run folder to start over")


def write_run_info(config, run_folder_path, lnn, circuits_qiskit_v13_qpy=False, instance=None, instance_plan=None,
                   scheduling_simulation=None, physical_qubits=None):
    """Called by bacq_ps_qbcs.submit() on every call (and by record_run_info()), once the run folder is known. Writes the manifest
    if missing, leaves it untouched if it already holds the same info, and rewrites it in place if it
    differs. The write itself is atomic (temp file + os.replace()), so a crash mid-write can't leave a
    truncated manifest. Skipping unchanged writes is what avoids the Windows problem: replacing the same
    file on every per-chunk submit() raced with Windows file locks (PermissionError); a short retry
    covers the rare lock that still coincides with a real change.
    instance / instance_plan (IBM): the run's instance; None keeps what the manifest already records, so a later call
    without them (the per-chunk submit()) never drops it.
    scheduling_simulation (QPU-noise simulations): the run's choice, True or False; None (real runs, noiseless simulations)
    records nothing.
    physical_qubits: the run's qubits (resolve_physical_qubits()); None keeps what the manifest already records."""
    run_folder_path = Path(run_folder_path)
    run_folder_path.mkdir(parents=True, exist_ok=True)
    if instance is None:
        instance, instance_plan = recorded_instance(run_folder_path.name, run_folder_path.parent)
    if physical_qubits is None:
        physical_qubits = recorded_physical_qubits(run_folder_path.name, run_folder_path.parent)
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
        instance      = instance,
        instance_plan = instance_plan,
        scheduling_simulation = scheduling_simulation,
        physical_qubits = [int(q) for q in physical_qubits] if physical_qubits is not None else None,
    )
    manifest = run_folder_path / MANIFEST_NAME
    if manifest.exists():
        try:
            if json.loads(manifest.read_text(encoding="utf-8")) == _manifest_dict(info):
                return info
        except (json.JSONDecodeError, OSError):
            pass   # unreadable/corrupt manifest — rewrite it below
    tmp = run_folder_path / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(_manifest_dict(info), indent=2), encoding="utf-8")
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
