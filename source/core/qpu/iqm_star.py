"""
iqm_star.py
IQM "star" machines: qubits QB1..QBn around one computational resonator (COMPR1), every two-qubit gate a CZ between a
qubit and the resonator plus MOVE gates. IQM VLQ (IT4I, 24 qubits) and IQM Resonance's Sirius are such machines. What they share,
kept out of the provider modules (iqm_vlq.py reaches VLQ, calibration_input.py / the Resonance scripts reach Sirius):

  qubit_metrics()          per-qubit fidelities, coherence times and gate durations from a raw calibration
  representative_roles()   each virtual qubit's prx / MOVE / CZ counts on the machine (see below)
  select_star_qubits()     the run's physical qubits: every ordered choice scored from those roles
  snapshot_from_calibration()  an effective all-to-all calibration snapshot for the local
                           simulation (source/core/qpu/qpu_snapshot.py format)
  calibration_from_station_control()  Sirius's raw calibration, in the shape of VLQ's vlq_calibration.json

A raw calibration ("cal") is one dict: {source, downloaded_at, calibration_set: {observations: [{dut_field, value,
invalid}], end_timestamp, ...}, metrics: {observations: [...]}, dynamic_architecture: {...}, calibration_set_id}. VLQ's
public API serves it as it is; for Sirius it is built from the observation sets of IQM's server.

Qubit selection. A CX between virtual qubits a and b becomes MOVE(a -> COMPR1), CZ(b, COMPR1),
MOVE(COMPR1 -> a). Which qubit is moved is the transpiler's choice, and it is fixed per
pair (on VLQ, 2026-09-29: the lower virtual index), so each virtual qubit has its own
roles. representative_roles() reads them from the representative circuit transpiled onto
the machine (transpile() runs locally; nothing is submitted): per virtual qubit v,
n_prx(v) native single-qubit gates, n_move(v) MOVE pairs (in and out) and n_cz(v) CZs.
Placing v on physical qubit q scores

    log P(v -> q) = log F_readout(q) + n_prx(v) · log F_prx(q)
                    + n_move(v) · log F_move(q) + n_cz(v) · log F_cz(q)

with F_move the MOVE-pair fidelity of the moved qubit (IQM's randomized benchmarking of the circuit MOVE(q) → MOVE(q) back,
no CZ in between: `irb.circuit.move_ncz_move` with n_interleaved_gates=0; its fixed spectator partner varies with the
qubit) and F_prx the Clifford RB fidelity converted to one native gate, 1 - (1 - F_clifford) / PRX_PER_CLIFFORD. The
fictional CZ of two qubits a (moved) and b is then F_move(a) · F_cz(b); the direct measurement of that circuit
(n_interleaved_gates=1, one partner b per qubit) agrees with it within about 1 percentage point. Every
ordered choice of three distinct usable qubits is scored and the best sum wins (method "star_roles"). A qubit missing any of
the four metrics (a qubit without a MOVE-pair measurement included) is excluded, whatever its role: a gap in its
calibration signals a problem with the qubit.

Gate implementations. A machine may use different implementations of CZ and MOVE on different qubits (Sirius: `tgss` on QB2 and
QB10, `crf_crf` on the others). qubit_metrics() reads the implementation of each locus from the dynamic architecture, else
tries `crf_crf` and then `tgss`.

Idle decay. The fidelities hold the decay during a gate, not a qubit waiting for the others. The resonator serializes
the CZ and MOVE gates, so the circuit lasts  T = Σ_v (2·n_move(v)·t_move + n_cz(v)·t_cz) + max_v n_prx(v)·t_prx
and a qubit idles for T minus its own gate time; each choice also gets  Σ_v log F(t_idle; T1, T2)
(layout_selection.idle_decay_log(), exact for any t_idle). Where a usable qubit lacks T1, T2 or a gate duration
the term is left out for every choice (record field "decoherence_included").

Needs no provider SDK at import time, so it loads in every venv.
"""

import itertools
import math
import re

from source.core.qpu.layout_selection import idle_decay_log
from source.core.records.job_timing import iso_utc

RESONATOR         = "COMPR1"          # the resonator's name when the calibration's architecture does not list one
PRX_PER_CLIFFORD  = 1.875             # average native prx gates per single-qubit Clifford
GATE_IMPLEMENTATIONS = ("crf_crf", "tgss")   # tried in this order where the architecture names none for a locus


# ── Raw calibration ───────────────────────────────────────────────────────────
def _observation_dicts(observations):
    """The observations of an IQM observation set as plain dicts (dut_field, value, unit, invalid)."""
    plain = lambda v: v.tolist() if hasattr(v, "tolist") else v   # numpy arrays / scalars → JSON
    return [{"dut_field": o.dut_field, "value": plain(o.value), "unit": getattr(o, "unit", None),
             "invalid": bool(getattr(o, "invalid", False))} for o in observations]


def _iso_z(ts):
    return None if ts is None else ts.isoformat().replace("+00:00", "Z")


def calibration_from_station_control(server_client, dynamic_architecture, calibration_set_id="default", source=None):
    """Sirius's raw calibration, in the shape of VLQ's vlq_calibration.json (see the module docstring).
    server_client: IQMClient._iqm_server_client (iqm-client's private station-control client); dynamic_architecture:
    IQMClient.get_dynamic_quantum_architecture(); calibration_set_id: a UUID or 'default'. Prefer the id of the
    architecture's calibration set, so the set and its quality metrics are the ones the QPU runs with."""
    cal_set = server_client.get_calibration_set(calibration_set_id)
    quality = server_client.get_calibration_set_quality_metric_set(calibration_set_id)
    from source.core.records.job_timing import now_utc_iso
    dqa = dynamic_architecture.model_dump(mode="json") if hasattr(dynamic_architecture, "model_dump") else dict(dynamic_architecture)
    return {
        "source":              source or f"IQM Resonance calibration set {calibration_set_id}",
        "downloaded_at":       now_utc_iso(),
        "calibration_set":     {"observation_set_id": str(cal_set.observation_set_id),
                                "end_timestamp":      _iso_z(cal_set.end_timestamp),
                                "observations":       _observation_dicts(cal_set.observations)},
        "metrics":             {"observation_set_id": str(quality.observation_set_id),
                                "end_timestamp":      _iso_z(quality.end_timestamp),
                                "observations":       _observation_dicts(quality.observations)},
        "dynamic_architecture": dqa,
        "calibration_set_id":  str(dqa.get("calibration_set_id") or cal_set.observation_set_id),
    }


def is_star_architecture(dynamic_architecture):
    """True for a dynamic quantum architecture with a MOVE gate (an IQM star machine)."""
    gates = getattr(dynamic_architecture, "gates", None)
    if gates is None and isinstance(dynamic_architecture, dict):
        gates = dynamic_architecture.get("gates")
    return bool(gates) and "move" in gates


def attach_star_calibration(backend):
    """For an IQM Resonance backend that is a star machine (Sirius; its dynamic architecture has a MOVE gate): download the
    raw calibration of the architecture's calibration set and attach it as backend.vlq_calibration (the attribute
    the layout choice reads, as on VLQ) and the machine's name as backend.star_name. Returns the calibration, or None
    for any other backend (Emerald, Garnet) or when the download fails (a warning is printed; the caller then
    keeps the Target-based snapshot). Needs iqm-client's private station-control client (IQMClient._iqm_server_client)."""
    try:
        client = backend.client
        dqa    = client.get_dynamic_quantum_architecture()
        if not is_star_architecture(dqa):
            return None
        cal = calibration_from_station_control(client._iqm_server_client, dqa, str(dqa.calibration_set_id),
                                               source=f"IQM Resonance {backend.name} calibration set {dqa.calibration_set_id}")
    except Exception as e:
        print(f"  ⚠️  [Calibration] could not download {getattr(backend, 'name', backend)}'s raw calibration "
              f"({type(e).__name__}: {e}) — using the backend's Target.")
        return None
    backend.vlq_calibration = cal
    backend.star_name       = str(backend.name).capitalize()
    return cal


def _fields(observation_set):
    return {o["dut_field"]: o["value"] for o in observation_set.get("observations", [])
            if not o.get("invalid") and o.get("value") is not None}


def qubit_names(cal):
    names = cal["dynamic_architecture"].get("qubits") or sorted(
        {m.group(0) for f in _fields(cal["metrics"]) for m in [re.search(r"\bQB\d+\b", f)] if m},
        key=lambda n: int(n[2:]))
    return sorted(names, key=lambda n: int(n[2:]))


def physical_index(name):
    """'QB7' -> 6: the backend's qubit index (qaas' backend._idx_to_qb; IQM's sorted qubit names)."""
    return int(name[2:]) - 1


def resonator_name(cal):
    """The calibration's computational resonator (RESONATOR when the architecture lists none)."""
    return (cal["dynamic_architecture"].get("computational_resonators") or [RESONATOR])[0]


def _implementations(cal, gate, qubit, resonator):
    """The implementations of `gate` (cz / move) to try on the locus (qubit, resonator), best first: the ones the dynamic
    architecture lists for that locus (its default among them first), then GATE_IMPLEMENTATIONS."""
    entry = ((cal["dynamic_architecture"].get("gates") or {}).get(gate) or {})
    impls = entry.get("implementations") or {}
    here  = [i for i, v in impls.items() if [qubit, resonator] in [list(l) for l in v.get("loci", [])]]
    if entry.get("default_implementation") in here:
        here.remove(entry["default_implementation"])
        here.insert(0, entry["default_implementation"])
    return here + [i for i in GATE_IMPLEMENTATIONS if i not in here]


def _first(fields, template, implementations):
    """The value of the first of `implementations` that `fields` has under template.format(impl=...), else None."""
    for impl in implementations:
        v = fields.get(template.format(impl=impl))
        if v is not None:
            return v
    return None


def move_pair_fidelities(metrics_fields, resonator):
    """{qubit name: F_move} — the fidelity of the MOVE pair of each moved qubit (`irb.circuit.move_ncz_move` of the locus
    QBa__RES__QBb with n_interleaved_gates=0; the partner QBb is a fixed spectator that varies with the qubit, the lowest
    one is taken if there are several). A qubit without it is absent."""
    pattern = re.compile(rf"^metrics\.irb\.circuit\.move_ncz_move\.(QB\d+)__{re.escape(resonator)}__QB\d+"
                         rf"\.fidelity:n_interleaved_gates=0$")
    out = {}
    for field in sorted(metrics_fields):
        m = pattern.match(field)
        if m:
            out.setdefault(m.group(1), metrics_fields[field])
    return out


def qubit_metrics(cal):
    """{qubit name: {readout, f1q, cz, move, t1, t2, prx_s, cz_s, move_s, measure_s}} —
    fidelities (None when not measured), coherence times and gate durations in s. `move` is F_move, the MOVE-pair
    fidelity of the qubit as the moved one (move_pair_fidelities()); `cz` the CZ fidelity of the qubit with the resonator."""
    m, c = _fields(cal["metrics"]), _fields(cal["calibration_set"])
    res  = resonator_name(cal)
    move = move_pair_fidelities(m, res)
    out = {}
    for q in qubit_names(cal):
        qr = f"{q}__{res}"
        cz_impl, move_impl = _implementations(cal, "cz", q, res), _implementations(cal, "move", q, res)
        out[q] = {
            "readout":   m.get(f"metrics.ssro.measure.constant.{q}.fidelity"),
            "f1q":       m.get(f"metrics.rb.clifford.xy.{q}.fidelity"),
            "cz":        _first(m, "metrics.irb.cz.{impl}." + qr + ".fidelity", cz_impl),
            "move":      move.get(q),
            "t1":        m.get(f"characterization.model.{q}.t1_time"),
            "t2":        m.get(f"characterization.model.{q}.t2_time"),
            "prx_s":     c.get(f"gates.prx.drag_crf.{q}.duration"),
            "cz_s":      _first(c, "gates.cz.{impl}." + qr + ".duration", cz_impl),
            "move_s":    _first(c, "gates.move.{impl}." + qr + ".duration", move_impl),
            "measure_s": c.get(f"gates.measure.constant.{q}.duration"),
        }
    return out


def _log(f):
    return math.log(f) if f is not None and 0 < f <= 1 else None


def prx_fidelity(f_clifford):
    """The Clifford RB fidelity converted to one native prx gate: 1 - (1 - F_clifford) / PRX_PER_CLIFFORD."""
    return None if f_clifford is None else 1 - (1 - f_clifford) / PRX_PER_CLIFFORD


def _qubit_logs(m):
    """{ro, prx, move, cz: log fidelity} of one qubit's qubit_metrics() entry, or None if any is missing or outside
    (0, 1]: such a qubit is used in no role (the one test select_star_qubits() and usable_star_qubits() share)."""
    logs = {"ro": _log(m["readout"]), "prx": _log(prx_fidelity(m["f1q"])), "move": _log(m["move"]), "cz": _log(m["cz"])}
    return None if None in logs.values() else logs


# ── Gate roles and qubit selection ────────────────────────────────────────────
def gate_roles(isa, layout):
    """Per virtual qubit v (on physical qubit layout[v]) of a circuit transpiled onto a star machine:
    {'n_prx': [...], 'n_move': [...] (MOVE pairs, in + out), 'n_cz': [...]}."""
    from source.core.records.circuit_check import physical_index_map   # circuit qubit -> physical index, either layout convention
    phys   = physical_index_map(isa)
    phys2v = {p: v for v, p in enumerate(layout)}
    roles  = {"n_prx": [0] * len(layout), "n_move": [0] * len(layout), "n_cz": [0] * len(layout)}
    key    = {"r": "n_prx", "prx": "n_prx", "move": "n_move", "cz": "n_cz"}
    for ins in isa.data:
        k = key.get(ins.operation.name)
        if k is None:
            continue
        own = [phys2v[phys[q]] for q in ins.qubits if phys[q] in phys2v]
        roles[k][own[0]] += 1
    roles["n_move"] = [m // 2 for m in roles["n_move"]]
    return roles


def offline_backend(cal, name="star_offline"):
    """A transpilation-only star backend built from the calibration's dynamic quantum
    architecture (qubits, resonator, gate loci) with iqm-client's own IQM target — the
    transpiler the provider's backend uses, without logging in. Needs iqm-client (iqm_vlq.venv, iqm_resonance.venv)."""
    from qiskit.providers import Options
    from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
    from iqm.station_control.interface.models.dynamic_quantum_architecture import DynamicQuantumArchitecture

    class OfflineStar(IQMBackendBase):
        max_circuits = None

        @classmethod
        def _default_options(cls):
            return Options()

        def run(self, *args, **kwargs):
            raise NotImplementedError("offline star backend: transpilation only")

    return OfflineStar(DynamicQuantumArchitecture(**cal["dynamic_architecture"]), name=name)


def representative_roles(circuit, cal, backend=None):
    """gate_roles() of `circuit` (raw) transpiled for the star machine on the first qubits with MOVE and CZ data (the roles
    don't depend on which: every qubit reaches the resonator). backend: the live backend,
    or None to build one from the calibration's architecture (offline_backend()). Nothing is
    submitted."""
    from qiskit import transpile
    from source.core.qpu.layout_selection import final_physical_qubits, count_2q_ops
    probe = [physical_index(q) for q, v in qubit_metrics(cal).items()
             if None not in (v["move"], v["cz"])][:circuit.num_qubits]
    isa   = transpile(circuit, backend or offline_backend(cal), optimization_level=1, initial_layout=probe)
    assert final_physical_qubits(isa) == probe, "representative circuit not transpiled onto the probe qubits"
    roles = gate_roles(isa, probe)
    assert sum(roles["n_cz"]) == count_2q_ops(circuit), "every two-qubit gate should become one CZ"
    return roles


def select_star_qubits(roles, cal, n_top=5):
    """The run's physical qubits, given each virtual qubit's gate roles (representative_roles()):
    every ordered choice of distinct physical qubits is scored — see the module docstring.
    Returns a physical_qubits.json record (like layout_selection.select_physical_qubits)."""
    n_virtual = len(roles["n_prx"])   # 3 in the benchmark: every ordered choice of distinct usable qubits is scored
    metrics = qubit_metrics(cal)

    grades, excluded = {}, []         # (physical name, virtual) -> log P
    for q, m in metrics.items():
        logs = _qubit_logs(m)
        if logs is None:              # a calibration gap or an invalid fidelity: the qubit is not used in any role
            excluded.append(q)
            continue
        for v in range(n_virtual):
            need = {"ro": 1, "prx": roles["n_prx"][v], "move": roles["n_move"][v], "cz": roles["n_cz"][v]}
            grades[q, v] = sum(n * logs[k] for k, n in need.items())

    names  = sorted((q for q in metrics if q not in excluded), key=lambda n: int(n[2:]))
    need_t = ("t1", "t2", "prx_s", "cz_s", "move_s")
    decoherence = bool(names) and all(isinstance(metrics[q][k], (int, float)) and metrics[q][k] > 0
                                      for q in names for k in need_t)

    def duration(phys, v):
        """(own gate time of virtual qubit v on phys[v], its prx time) in s."""
        m, prx = metrics[phys[v]], roles["n_prx"][v] * metrics[phys[v]]["prx_s"]
        return prx + 2 * roles["n_move"][v] * m["move_s"] + roles["n_cz"][v] * m["cz_s"], prx

    def score(phys):
        """(log P, circuit duration in s or None) of one ordered choice."""
        log_p = sum(grades[q, v] for v, q in enumerate(phys))
        if not decoherence:
            return log_p, None
        own    = [duration(phys, v) for v in range(n_virtual)]
        t_circ = sum(b - p for b, p in own) + max(p for _, p in own)
        for v, q in enumerate(phys):
            log_p += idle_decay_log(metrics[q]["t1"], metrics[q]["t2"], max(t_circ - own[v][0], 0.0))
        return log_p, t_circ

    scored = []
    for phys in itertools.permutations(names, n_virtual):
        log_p, t_circ = score(phys)
        scored.append((log_p, phys, t_circ))
    scored.sort(key=lambda s: (-s[0], [physical_index(q) for q in s[1]]))
    t_circuit = {phys: t for _, phys, t in scored[:n_top]}
    scored    = [(log_p, phys) for log_p, phys, _ in scored]
    base = dict(gates={"1q": "prx", "2q": "cz with the resonator + move pairs"}, gate_roles=roles,
                prx_per_clifford=PRX_PER_CLIFFORD, calibration_set_id=cal.get("calibration_set_id"),
                calibration_source=cal.get("source"))
    if not scored:
        return dict(base, physical_qubits=None, method="star_roles_no_data", log_success=None,
                    top_candidates=[], n_candidates=0)
    best_log, best = scored[0]
    return dict(
        base,
        physical_qubits = [physical_index(q) for q in best],
        qubit_names     = list(best),
        method          = "star_roles",
        log_success     = best_log,
        decoherence_included = decoherence,
        t_circuit_s     = t_circuit[best] if decoherence else None,
        idle_log        = best_log - sum(grades[q, v] for v, q in enumerate(best)) if decoherence else None,
        top_candidates  = [[[physical_index(q) for q in p], s] for s, p in scored[:n_top]],
        n_candidates    = len(scored),
        excluded_qubits = sorted(excluded, key=lambda n: int(n[2:])),
        chosen_grades   = {q: grades[q, v] for v, q in enumerate(best)},
        chosen_metrics  = {q: metrics[q] for q in best},
    )


def usable_star_qubits(cal):
    """Physical indices of the qubits a layout may use: every qubit of the architecture that has no calibration gap
    or invalid fidelity, as select_star_qubits() decides (all of them reach the resonator, which every layout uses
    through CZ and MOVE), and the excluded ones' names."""
    ok, excluded = [], []
    for q, m in qubit_metrics(cal).items():
        (excluded if _qubit_logs(m) is None else ok).append(q)
    key = lambda n: int(n[2:])
    return sorted(physical_index(q) for q in ok), sorted(excluded, key=key)


def choose_star_layout(mode, roles, cal, qubits=None, seed=None, label="this QPU"):
    """The run's physical qubits on a star machine by layout mode (source/core/runner_support/benchmark_args.py): `optimized` is
    select_star_qubits(); `lowest`, `random` (random.Random(seed), a fresh recorded seed if None) and a forced
    list `qubits` choose among the ordered triples of usable_star_qubits() — the resonator is always part of the
    layout. ValueError (naming `label`) if a forced list is not valid. Returns a physical_qubits.json record."""
    if mode == "optimized":
        rec = select_star_qubits(roles, cal)
        rec["layout_mode"] = mode
        return rec
    n_virtual = len(roles["n_prx"])
    usable, excluded = usable_star_qubits(cal)
    if mode == "list":
        qubits = list(qubits)
        bad = [q for q in qubits if q not in usable]
        if len(qubits) != n_virtual or len(set(qubits)) != n_virtual or bad:
            usable_range = f"{usable[0]}…{usable[-1]}" if usable else "(none)"
            raise ValueError(f"layout {qubits} is not valid on {label}: needs {n_virtual} distinct qubits from the usable ones "
                             f"{usable_range} (without {[physical_index(q) for q in excluded]}); the resonator "
                             f"is always included" + (f"; not usable: {bad}" if bad else ""))
        chosen, method = qubits, "user"
    elif mode == "lowest":
        chosen, method = usable[:n_virtual], "lowest"
    elif mode == "random":
        import random
        import secrets
        seed   = secrets.randbits(32) if seed is None else seed
        chosen, method = random.Random(seed).sample(usable, n_virtual), "random"
    else:
        raise ValueError(f"unknown layout mode {mode!r}")
    n_cand = math.perm(len(usable), n_virtual)
    return dict(physical_qubits=chosen, qubit_names=[f"QB{q + 1}" for q in chosen], method=method, layout_mode=mode,
                log_success=None, top_candidates=[], n_candidates=None if mode == "list" else n_cand, gates=None, gate_roles=roles,
                calibration_set_id=cal.get("calibration_set_id"), calibration_source=cal.get("source"),
                excluded_qubits=excluded, **({"seed": seed} if mode == "random" else {}))


# ── Local simulation: an effective all-to-all snapshot ────────────────────────
def snapshot_from_calibration(cal, backend="VLQ"):
    """A calibration snapshot (source/core/qpu/qpu_snapshot.py format) of a star machine (`backend`: its name) with the resonator
    folded into an effective all-to-all CZ: CZ(a, b) on the device is MOVE(a), CZ(b, COMPR1),
    MOVE back, so its error is 1 - F_cz(b) · F_move(a) (F_move: the MOVE pair, see the module docstring), taking the
    better choice of which qubit is moved, and its duration 2 · MOVE + CZ. Single-qubit gates use the
    Clifford RB fidelity converted to one native prx gate (prx_fidelity()), readout the SSRO fidelity, and T1/T2 give the relaxation. A pair gets
    no CZ entry when neither direction has both MOVE and CZ data (a choice of which qubit is moved lacking either is dropped, the other kept). Qubit index i is QB(i+1)."""
    metrics = qubit_metrics(cal)
    names   = qubit_names(cal)
    err     = lambda f: None if f is None else max(0.0, 1.0 - f)

    ops = {"r": [], "cz": [], "measure": []}
    for q in names:
        v, i = metrics[q], physical_index(q)
        ops["r"].append({"qargs": [i], "error": err(prx_fidelity(v["f1q"])), "duration": v["prx_s"]})
        ops["measure"].append({"qargs": [i], "error": err(v["readout"]), "duration": v["measure_s"]})
    for a, b in itertools.permutations(names, 2):
        options = []
        for moved, target in ((a, b), (b, a)):
            fm, fc = metrics[moved]["move"], metrics[target]["cz"]
            if fm is None or fc is None:
                continue
            dur = None
            if metrics[moved]["move_s"] is not None and metrics[target]["cz_s"] is not None:
                dur = 2 * metrics[moved]["move_s"] + metrics[target]["cz_s"]
            options.append((fc * fm, dur))
        if options:
            f, dur = max(options, key=lambda o: o[0])
            ops["cz"].append({"qargs": [physical_index(a), physical_index(b)], "error": err(f), "duration": dur})

    return {
        "backend":            backend,
        "downloaded_at":      cal.get("downloaded_at"),
        "measured_at":        iso_utc((cal.get("calibration_set") or {}).get("end_timestamp")),   # when the provider made the calibration
        "calibration_set_id": cal.get("calibration_set_id"),
        "calibration_source": cal.get("source"),
        "num_qubits":         len(names),
        "dt":                 None,
        "operations":         ops,
        "qubit_properties":   [{"t1": metrics[q]["t1"], "t2": metrics[q]["t2"], "frequency": None} for q in names],
        "skipped_operations": ["move (folded into the effective CZ)", "prx_12", "cc_prx", "reset_wait"],
        "effective_model":    "resonator folded into an all-to-all CZ: error 1 - F_cz(target)·F_move(moved), "
                              "better choice of which qubit is moved; duration 2·MOVE + CZ",
    }


STAR_CALIBRATION_FILENAME = "star_calibration.json"   # Sirius's raw calibration in a run folder (VLQ's: vlq_calibration.json)


def save_star_calibration(run_dir, cal, backend, filename=STAR_CALIBRATION_FILENAME):
    """Write a star machine's raw calibration into `run_dir` (created if needed) as `filename` and the
    calibration_snapshot.json built from it. Returns the snapshot's path."""
    import json
    from pathlib import Path
    from source.core.qpu.qpu_snapshot import SNAPSHOT_FILENAME
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / filename).write_text(json.dumps(cal), encoding="utf-8")
    snapshot_path = run_dir / SNAPSHOT_FILENAME
    snapshot_path.write_text(json.dumps(snapshot_from_calibration(cal, backend), indent=2), encoding="utf-8")
    return snapshot_path
