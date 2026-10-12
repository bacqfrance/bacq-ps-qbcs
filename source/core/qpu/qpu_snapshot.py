"""
qpu_snapshot.py
Save a real QPU's calibration to JSON and rebuild an offline backend + Aer noise
model from it — used by the source/benchmark_scripts/real_qpu_scripts/*/qbcs_real_simulate_*.py scripts, source/core/runner_support/run_simulation.py,
source/core/qpu/calibration_input.py and source/benchmark_scripts/local_simulation_scripts/benchmark_fake_fez_lnn.py.

A simulated run always works from the saved file (<run>/calibration_snapshot.json),
never from the live backend directly: a resumed run keeps the exact calibration
it started with, needs no connection, and the file records what was simulated.

The snapshot holds everything the backend's qiskit Target reports: qubit count,
native operations and where they're defined, per-locus error and duration, and
per-qubit T1/T2/frequency. Operations qiskit's standard-gate mapping doesn't
know (control flow like if_else, vendor extras) are listed under
"skipped_operations" and left out of the rebuilt Target — the benchmark
circuits never use them.

Two dates: "downloaded_at" is when the snapshot was made, "measured_at" when the provider made
the calibration (UTC, "...Z"; None when the provider does not say). Live: IBM's
backend.properties().last_update_date, IQM Resonance's calibration set end_timestamp, IQM VLQ's
calibration set end_timestamp; a provider's calibration file carries its own date.

Only as good as what the provider reports: IBM, IQM Resonance
(get_backend(use_metrics=True)) and IQM VLQ report full calibration; IQM EQE1 (MQSS) and
AQT report none through their SDK. For those two, a calibration file published
by the provider (the latest ones are in calibration_input_files/) is converted
(snapshot_from_aqt_json / snapshot_from_iqm_eqe1_json) into the same snapshot
format. source/core/qpu/calibration_input.py does that automatically for the scripts'
calibration_file= argument, saves the result as <run>/calibration_snapshot.json
and warns that it is not live backend data.

Uses only qiskit + qiskit-aer, so it works in every backend venv.
"""

import json
import math
from pathlib import Path

from source.core.common import rel_path
from source.core.records.job_timing import iso_utc, now_utc_iso

SNAPSHOT_FILENAME = "calibration_snapshot.json"

NO_CALIBRATION_MESSAGE = """\
❌ Cannot simulate {backend}: its SDK exposes no calibration data.

A device-faithful local simulation needs the backend's per-qubit and per-pair
error rates (and ideally T1/T2 and gate durations), which qiskit-aer turns into
a noise model. This backend does not provide them:
  - IQM EQE1 (via MQSS): the backend reports only its qubits, couplings and
    gate set — every error entry is None.
  - AQT PIAST-Q: no calibration is reported; the provider's offline simulator
    has only a generic noise model (0.3 % / 1 % depolarizing), and AQT assigns
    the physical ions server-side.

Simulated runs download calibration for IBM (source/benchmark_scripts/real_qpu_scripts/IBM/qbcs_real_simulate_ibm.py),
IQM Resonance (source/benchmark_scripts/real_qpu_scripts/IQM_RESONANCE/qbcs_real_simulate_iqm_resonance.py)
and IQM VLQ (a vlq_calibration.json given as calibration_file=..., else downloaded from IT4I's API);
for AQT PIAST-Q and IQM EQE1 they use a calibration file instead (calibration_file=..., default: the
newest in calibration_input_files/<provider>/ — source/core/qpu/calibration_input.py).
"""


def _base_snapshot(backend, measured_at, source_file, num_qubits):
    return {
        "backend":            backend,
        "downloaded_at":      now_utc_iso(),   # when this snapshot was made (here: converted)
        "measured_at":        measured_at,
        "source_file":        source_file,
        "calibration_set_id": None,
        "num_qubits":         num_qubits,
        "dt":                 None,
        "operations":         {},
        "qubit_properties":   None,
        "skipped_operations": [],
        "excluded_qubits":    [],
        "notes":              [],
    }


def snapshot_from_aqt_json(data, source_file):
    """Snapshot dict from an AQT PIAST-Q characterisation JSON (the piastq_characterisation
    <date>.json files: vendor.characterisation, fidelities in %, times in s / µs).

    Native gates as the AQT backend's Target has them: r, rz (virtual, error-free) and rxx
    between every pair (all-to-all), plus measure. The file gives a per-ion r fidelity but
    only a mean rxx fidelity and one readout (SPAM lower bound), T1 and T2 for the whole
    device, so those are applied to every ion/pair. The crosstalk data is kept under
    "crosstalk" for reference but not modelled."""
    char = data["vendor"]["characterisation"]
    n    = int(data["n_qubits"])
    snap = _base_snapshot(f"aqt_{data.get('resource_id', 'piast-q')}",
                          char.get("updated_at") or data["vendor"].get("status_updated_at"),
                          source_file, n)

    f1q    = char["single_qubit_gate_fidelity"]
    t_1q   = char["single_qubit_gate_duration_micros"] * 1e-6
    t_2q   = char["two_qubit_gate_duration_micros"] * 1e-6
    t_ro   = char["readout_time_micros"] * 1e-6
    e_2q   = 1 - char["mean_two_qubit_gate_fidelity"]["value"] / 100
    e_ro   = 1 - char["spam_fidelity_lower_bound"] / 100
    snap["operations"] = {
        "r":       [{"qargs": [q], "error": 1 - f1q[str(q)]["value"] / 100, "duration": t_1q} for q in range(n)],
        "rz":      [{"qargs": [q], "error": 0.0, "duration": 0.0} for q in range(n)],
        "rxx":     [{"qargs": [a, b], "error": e_2q, "duration": t_2q}
                    for a in range(n) for b in range(n) if a != b],
        "measure": [{"qargs": [q], "error": e_ro, "duration": t_ro} for q in range(n)],
    }
    t1, t2 = char["t1_s"]["value"], char["t2_coherence_time_s"]["value"]
    snap["qubit_properties"] = [{"t1": t1, "t2": t2, "frequency": None} for _ in range(n)]
    snap["crosstalk"] = {k: v for k, v in data.get("summary", {}).items() if k.startswith("crosstalk")}
    snap["notes"] = [
        "rxx error: device mean two-qubit fidelity, applied to every pair",
        "measure error: 1 - SPAM fidelity lower bound, applied to every ion",
        "T1/T2: device-wide values, applied to every ion",
        "crosstalk: recorded only, not in the noise model",
    ]
    return snap


def snapshot_from_iqm_eqe1_json(data, source_file):
    """Snapshot dict from an IQM Euro-Q-Exa (EQE1) calibration JSON
    (calibration_data_Euro-Q-Exa_<date>_active_qubits.json: per-qubit lists T1/T2 in µs,
    fidelities as fractions, two_q_fidelity keyed "a,b").

    Native gates as the EQE1 backend's Target has them: r (prx), cz on the listed pairs, and
    measure. T2 is the Ramsey value (the circuits use no dynamical decoupling). The file has
    no gate durations, so none are set: the Aer noise model then has no T1/T2 relaxation
    term — only depolarizing gate errors and readout errors. A qubit with NaN in any field
    used (T1, T2, 1q_fidelity, readout_fidelity) is left out of the Target entirely — no
    r / cz / measure entries — so neither the layout selection nor the transpiler can place
    anything on it; it is listed under "excluded_qubits"."""
    n = len(data["T1"])
    snap = _base_snapshot("iqm_eqe1", data["timestamp"], source_file, n)

    def bad(x):
        return x is None or (isinstance(x, float) and math.isnan(x))

    fields   = ("T1", "T2", "1q_fidelity", "readout_fidelity")
    excluded = sorted(q for q in range(n) if any(bad(data[f][q]) for f in fields))
    ok       = [q for q in range(n) if q not in excluded]
    pairs    = []
    for key, fid in data["two_q_fidelity"].items():
        a, b = (int(x) for x in key.split(","))
        if a in excluded or b in excluded or bad(fid):
            continue
        pairs += [([a, b], 1 - fid), ([b, a], 1 - fid)]   # cz is symmetric: both directions

    snap["operations"] = {
        "r":       [{"qargs": [q], "error": 1 - data["1q_fidelity"][q], "duration": None} for q in ok],
        "cz":      [{"qargs": qa, "error": e, "duration": None} for qa, e in pairs],
        "measure": [{"qargs": [q], "error": 1 - data["readout_fidelity"][q], "duration": None} for q in ok],
    }
    snap["qubit_properties"] = [None if q in excluded else
                                {"t1": data["T1"][q] * 1e-6, "t2": data["T2"][q] * 1e-6, "frequency": None}
                                for q in range(n)]
    snap["excluded_qubits"] = excluded
    snap["notes"] = [
        "no gate durations in the source file: no T1/T2 relaxation in the noise model",
        "T2: Ramsey (not echo)",
        "excluded_qubits: NaN in T1, T2, 1q_fidelity or readout_fidelity — left out of the Target",
    ]
    return snap


def provider_calibration_time(backend):
    """When the provider made the calibration `backend` reports, as a UTC ISO string (job_timing.iso_utc), or None.
    IBM: backend.properties().last_update_date. IQM Resonance: end_timestamp of the calibration set the backend's
    architecture names (calibration_set_id). Never raises: a missing date must not stop a run."""
    try:
        arch   = getattr(backend, "architecture", None)
        cal_id = getattr(arch, "calibration_set_id", None) if arch is not None else None
        if cal_id is not None:
            return iso_utc(backend.client.get_calibration_set(cal_id).end_timestamp)
        return iso_utc(backend.properties().last_update_date)
    except Exception:
        return None


def save_calibration_snapshot(backend, path, extra=None, measured_at=None):
    """Write backend.target's calibration to `path` (JSON). Returns the dict.

    extra: optional dict merged into the top level (e.g. calibration_set_id).
    measured_at: when the provider made the calibration (any ISO string or datetime; saved as UTC);
    None asks the backend (provider_calibration_time()), which gives None when it cannot say.
    Raises ValueError (NO_CALIBRATION_MESSAGE) when the target has no gate errors."""
    target = backend.target
    ops, skipped = {}, []
    from qiskit.circuit.library import get_standard_gate_name_mapping
    known = set(get_standard_gate_name_mapping())
    for name in target.operation_names:
        if name not in known:
            skipped.append(name)
            continue
        entries = []
        for qargs, props in target[name].items():
            entries.append({
                "qargs":    None if qargs is None else list(qargs),
                "error":    None if props is None else props.error,
                "duration": None if props is None else props.duration,
            })
        ops[name] = entries

    qprops = getattr(target, "qubit_properties", None)
    qubits = None
    if qprops is not None:
        qubits = [None if q is None else {"t1": q.t1, "t2": q.t2, "frequency": q.frequency}
                  for q in qprops]

    arch   = getattr(backend, "architecture", None)
    cal_id = getattr(arch, "calibration_set_id", None) if arch is not None else None
    snapshot = {
        "backend":            getattr(backend, "name", str(backend)),
        "downloaded_at":      now_utc_iso(),
        "measured_at":        iso_utc(measured_at) if measured_at is not None else provider_calibration_time(backend),
        "calibration_set_id": None if cal_id is None else str(cal_id),
        "num_qubits":         target.num_qubits,
        "dt":                 target.dt,
        "operations":         ops,
        "qubit_properties":   qubits,
        "skipped_operations": sorted(skipped),
    }
    if extra:
        snapshot.update(extra)

    n_err = sum(1 for es in ops.values() for e in es if e["error"] is not None)
    if n_err == 0:
        raise ValueError(NO_CALIBRATION_MESSAGE.format(backend=snapshot["backend"]))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"  [Snapshot] {snapshot['backend']}: {target.num_qubits} qubits, {n_err} calibrated "
          f"operation entries → {rel_path(path)}")
    return snapshot


def reuse_calibration_snapshot(path):
    """True (after printing a warning) if a snapshot was saved by an earlier run of the script."""
    path = Path(path)
    if not path.exists():
        return False
    snap = load_calibration_snapshot(path)
    print(f"  ⚠️  [Snapshot] Using PRE-SAVED calibration data of {snap.get('backend')} downloaded "
          f"{snap.get('downloaded_at')} ({rel_path(path)}).\n"
          f"     To download fresh calibration data and run with it, remove the folder {rel_path(path.parent)} "
          f"and run the script again.")
    return True


def load_calibration_snapshot(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def snapshot_log_lines(path, run_qubits=None):
    """submit.log lines describing a simulated run's calibration (the snapshot file it used):
    where it came from, the device's median 2-qubit gate error, and — for run_qubits, a chain —
    each pair's 2-qubit error and the mean T1/T2. Never raises; returns [] without a snapshot."""
    try:
        snap = load_calibration_snapshot(path)
    except (OSError, ValueError):
        return []
    lines = [f"Calibration:      {Path(path).name} of {snap.get('backend')}, from {snap.get('downloaded_at')}"
             + (f", calibration set {snap['calibration_set_id']}" if snap.get("calibration_set_id") else "")]
    try:
        errs_2q = {}                                   # (q0, q1) -> (gate, error)
        for gate, entries in (snap.get("operations") or {}).items():
            for e in entries or []:
                if len(e.get("qargs") or []) == 2 and e.get("error") is not None:
                    errs_2q[tuple(e["qargs"])] = (gate, float(e["error"]))
        if errs_2q:
            vals = sorted(err for _, err in errs_2q.values())
            gate = next(iter(errs_2q.values()))[0]
            lines.append(f"{gate.upper()} error (median): {vals[len(vals) // 2]:.3e}")
        if run_qubits:
            pairs = []
            for a, b in zip(run_qubits, run_qubits[1:]):
                g, err = errs_2q.get((a, b)) or errs_2q.get((b, a)) or (None, None)
                pairs.append(f"{(g or '2q').upper()}({a},{b}): " + (f"{err:.3e}" if err is not None else "n/a"))
            if pairs:
                lines.append("   ".join(pairs))
            props = snap.get("qubit_properties") or []
            t1 = [props[q]["t1"] for q in run_qubits if q < len(props) and (props[q] or {}).get("t1")]
            t2 = [props[q]["t2"] for q in run_qubits if q < len(props) and (props[q] or {}).get("t2")]
            lines.append((f"T1_run_qubits: {sum(t1) / len(t1) * 1e6:.1f}µs   " if t1 else "T1_run_qubits: n/a   ")
                         + (f"T2_run_qubits: {sum(t2) / len(t2) * 1e6:.1f}µs" if t2 else "T2_run_qubits: n/a"))
    except Exception as e:                             # a log line must never stop a run
        lines.append(f"(calibration summary unavailable: {type(e).__name__}: {e})")
    return lines


def backend_from_snapshot(snapshot):
    """Offline BackendV2 whose Target is rebuilt from a snapshot dict (or path).
    For transpilation and noise-model construction only — it cannot run circuits."""
    from qiskit.circuit.library import get_standard_gate_name_mapping
    from qiskit.providers import BackendV2, Options
    from qiskit.providers.backend import QubitProperties
    from qiskit.transpiler import InstructionProperties, Target

    if not isinstance(snapshot, dict):
        snapshot = load_calibration_snapshot(snapshot)

    qprops = None
    if snapshot.get("qubit_properties"):
        qprops = [QubitProperties() if q is None else QubitProperties(**q)
                  for q in snapshot["qubit_properties"]]
    target  = Target(num_qubits=snapshot["num_qubits"], dt=snapshot.get("dt"), qubit_properties=qprops)
    mapping = get_standard_gate_name_mapping()
    for name, entries in snapshot["operations"].items():
        props = {}
        for e in entries:
            qargs = None if e["qargs"] is None else tuple(e["qargs"])
            has   = e["error"] is not None or e["duration"] is not None
            props[qargs] = InstructionProperties(error=e["error"], duration=e["duration"]) if has else None
        target.add_instruction(mapping[name], props if props != {None: None} else None, name=name)

    class SnapshotBackend(BackendV2):
        def __init__(self):
            super().__init__(name=f"{snapshot['backend']}_snapshot",
                             description=f"calibration snapshot of {snapshot['backend']} "
                                         f"downloaded {snapshot['downloaded_at']}")

        @property
        def target(self):
            return target

        @property
        def max_circuits(self):
            return None

        @classmethod
        def _default_options(cls):
            return Options()

        def run(self, run_input, **options):
            raise NotImplementedError("SnapshotBackend is for transpilation/noise models only — "
                                      "run circuits through an AerSimulator built from it.")

    return SnapshotBackend()


def noisy_sampler_from_snapshot(snapshot, seed=None):
    """(snapshot_backend, noise_model, AerSamplerV2) for a snapshot dict or path.

    The noise model is qiskit-aer's standard device model built from the Target:
    depolarizing gate errors, thermal relaxation from T1/T2 over each gate's
    duration, and readout errors from `measure`."""
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

    backend     = backend_from_snapshot(snapshot)
    noise_model = NoiseModel.from_backend(backend)
    opts        = dict(backend_options=dict(noise_model=noise_model))
    if seed is not None:
        opts["run_options"] = dict(seed=seed)
    return backend, noise_model, AerSamplerV2(options=opts)
