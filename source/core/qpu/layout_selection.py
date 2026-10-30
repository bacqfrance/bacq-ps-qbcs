"""
layout_selection.py
Choose which physical qubits a run's circuits (and its SPAM calibration) go on.

Every chunk and the SPAM batch of one run are transpiled with the same
initial_layout, chosen once per run by choose_layout() (select_physical_qubits() for mode "optimized") and stored in
<run>/transpiled_circuits/physical_qubits.json. Without this, qiskit picks a
layout per circuit — SPAM circuits (no 2q gates) always landed on the trivial
layout (0, 1, ...) while the chunk circuits could land elsewhere, and on
backends without error data the choice could even differ within one chunk.

Criterion: for every placement of the circuit's interaction graph onto the
backend's coupling graph that needs no SWAPs, estimate the success probability
of a representative (deepest) circuit on it. A fast prescreen scores every placement,

    log P = Σ_pairs  n_2q(pair)  · log(1 − e_2q(pair))
          + Σ_qubits log(1 − e_readout(q))
          + Σ_qubits n_1q(q)     · log(1 − e_1q(q)),

with n_2q counted on the raw (untranspiled) circuit, where it is exact (transpilation must not change it — see count_2q_ops),
and n_1q from transpiled_1q_counts() (one transpile of the representative circuit onto one SWAP-free placement; virtual-Z rz
not counted: the raw circuit's 1q count would overstate the native gates about threefold). The best N_EXACT_RESCORE placements
are then transpiled exactly as the submission does it (transpile_like_submission()) and scored on the transpiled circuit
(score_transpiled(): every instruction's own error, plus the idle decay of waiting qubits from our own ALAP timing of that
circuit); the highest wins.

Missing data: a qubit or coupler missing a field its QPU normally reports, or with an error of 1 or more (IBM marks dead
qubits with error 1.0), is unusable (REQUIRED_FIELDS, unusable_in_target()) and left out of every layout mode. If the backend
reports no errors at all (e.g. IQM EQE1 via MQSS), every placement scores equally and the lexicographically smallest one
is chosen — not optimised, but deterministic and recorded. If it reports errors for some operations only, placements
touching a missing entry are excluded rather than treated as error-free.

Uses only qiskit, so it works in every backend venv (aqt.venv runs qiskit 1.4).
"""

import itertools
import math
from collections import Counter

# Preferred native gate names, most specific first; the first one present in
# the backend's Target is used for that gate class.
_PREFERRED_2Q = ("cz", "ecr", "cx", "rxx", "rzz", "iswap")
_PREFERRED_1Q = ("r", "sx", "rx", "x", "u")

# Non-gate instructions that can span several qubits.
_NON_GATES = {"barrier", "measure", "reset", "delay"}

N_TOP_CANDIDATES_RECORDED = 5
N_EXACT_RESCORE = 50   # the best placements of the fast prescreen that are transpiled and scored exactly (score_transpiled())

# Strict calibration completeness: what a QPU normally reports for a qubit / coupler. A qubit or coupler missing any of its
# fields (or with an error of 1 or more: IBM marks dead qubits with error 1.0) is unusable and left out of every layout mode.
# Fields: t1 / t2 (qubit_properties), readout (measure error), readout_duration, 1q_error / 1q_duration (every native
# single-qubit gate the QPU calibrates), 2q_error / 2q_duration (the two-qubit gate of a coupler). A field is required only
# where the QPU reports it for some qubit: EQE1 and AQT report no durations per qubit, so their missing durations are no gap.
# The star machines (VLQ, Sirius) are checked on the raw calibration (iqm_star.missing_fields(): the names are those of
# iqm_star.qubit_metrics()); T2 echo and the MOVE-pair IRB with 2 CZs are not required (one is never used, the other does not exist).
REQUIRED_FIELDS = {
    "ibm":   {"qubit": ("t1", "t2", "readout", "1q_error", "1q_duration"), "coupler": ("2q_error", "2q_duration")},
    "iqm":   {"qubit": ("t1", "t2", "readout", "readout_duration", "1q_error", "1q_duration"),
              "coupler": ("2q_error", "2q_duration")},   # IQM Resonance: Garnet, Emerald
    "eqe1":  {"qubit": ("t1", "t2", "readout", "1q_error"), "coupler": ("2q_error",)},
    "aqt":   {"qubit": ("1q_error",), "coupler": ()},
    "star":  {"qubit": ("readout", "f1q", "cz", "move", "move_cz", "move_irb", "t1", "t2",
                        "prx_s", "cz_s", "move_s", "measure_s"), "coupler": ()},
}


# IQM star (VLQ) MOVEs shuttle a qubit's state into the computational resonator and back
# around each CZ: they are transport, not an entangling gate of the circuit.
_NOT_COUNTED_2Q = _NON_GATES | {"move"}


def count_2q_ops(circuit):
    """Number of two-qubit gates in circuit, whatever their name (cz, ecr, rxx, …).
    MOVEs (IQM star architecture) are not counted: a CX becomes one CZ with the
    resonator plus the MOVEs that bring the other qubit's state there and back."""
    return sum(1 for inst in circuit.data
               if len(inst.qubits) == 2 and inst.operation.name not in _NOT_COUNTED_2Q)


def _interaction_counts(circuit):
    """(2q gate count per unordered virtual pair, 1q gate count per virtual qubit)."""
    pairs, singles = Counter(), Counter()
    for inst in circuit.data:
        if inst.operation.name in _NON_GATES:
            continue
        idx = [circuit.find_bit(q).index for q in inst.qubits]
        if len(idx) == 2:
            pairs[tuple(sorted(idx))] += 1
        elif len(idx) == 1:
            singles[idx[0]] += 1
    return pairs, singles


def _pick_op(target, preferred, n_qubits):
    names = set(target.operation_names)
    for name in preferred:
        if name in names:
            return name
    for name in sorted(names):
        op = target.operation_from_name(name)
        if getattr(op, "num_qubits", None) == n_qubits and name not in _NON_GATES:
            return name
    return None


def _errors(target, name):
    """{qargs: error or None} for one operation name; {} if name is None."""
    if name is None:
        return {}
    return {tuple(q): (p.error if p is not None else None)
            for q, p in target[name].items() if q is not None}


def idle_decay_log(t1, t2, t_idle):
    """log of the process fidelity of a qubit left idle for t_idle seconds (T1 relaxation and dephasing, exact for any
    t_idle): F = (1 + 2·exp(−t/T2) + exp(−t/T1)) / 4 — the diagonal of the Pauli transfer matrix (X and Y decay with T2,
    Z with T1). T2 is capped at 2·T1, its physical limit. None if T1 or T2 is missing or not positive."""
    if not t1 or not t2 or t1 <= 0 or t2 <= 0:
        return None
    t2 = min(t2, 2.0 * t1)
    return math.log((1.0 + 2.0 * math.exp(-t_idle / t2) + math.exp(-t_idle / t1)) / 4.0)


def _coherence_times(target):
    """{physical qubit: (T1, T2) in s} of a Target's qubit_properties; {} if it has none."""
    props = getattr(target, "qubit_properties", None)
    if not props:
        return {}
    return {q: (p.t1, p.t2) for q, p in enumerate(props) if p is not None and p.t1 and p.t2}


def _placements(n_virtual, pairs, neighbors, n_physical):
    """Yield every injective virtual→physical map in which each interacting
    virtual pair lands on a coupled physical pair (subgraph monomorphism, DFS)."""
    # Order virtual qubits so each one after the first touches an earlier one when possible.
    adj = {v: set() for v in range(n_virtual)}
    for a, b in pairs:
        adj[a].add(b); adj[b].add(a)
    order, seen = [], set()
    for start in sorted(range(n_virtual), key=lambda v: -len(adj[v])):
        if start in seen:
            continue
        stack = [start]
        while stack:
            v = stack.pop(0)
            if v in seen:
                continue
            seen.add(v); order.append(v)
            stack.extend(sorted(adj[v] - seen))

    assign = {}

    def dfs(i):
        if i == len(order):
            yield tuple(assign[v] for v in range(n_virtual))
            return
        v = order[i]
        placed_nbrs = [assign[u] for u in adj[v] if u in assign]
        if placed_nbrs:
            cands = set.intersection(*(neighbors[p] for p in placed_nbrs))
        else:
            cands = range(n_physical)
        used = set(assign.values())
        for p in sorted(cands):
            if p in used:
                continue
            assign[v] = p
            yield from dfs(i + 1)
            del assign[v]

    yield from dfs(0)


# Single-qubit instructions that cost nothing on the device (virtual Z) or are not gates.
_FREE_1Q = {"rz", "id", "delay"}


def probe_placement(circuit, target):
    """The first SWAP-free placement of `circuit` on `target`'s coupling graph (unscored), to
    transpile onto for transpiled_1q_counts(); None if the target has no coupling map or none fits."""
    name_2q = _pick_op(target, _PREFERRED_2Q, 2)
    err_2q  = _errors(target, name_2q)
    if not err_2q:
        return None
    neighbors = {p: set() for p in range(target.num_qubits)}
    for q in err_2q:
        if len(q) == 2:
            neighbors[q[0]].add(q[1]); neighbors[q[1]].add(q[0])
    pairs, _ = _interaction_counts(circuit)
    return next((list(p) for p in _placements(circuit.num_qubits, pairs, neighbors, target.num_qubits)), None)


def transpile_like_submission(circuit, backend, layout):
    """`circuit` (one circuit or a list) transpiled exactly as a submission does it: transpile(optimization_level=1,
    initial_layout=layout) with no scheduling_method (the backend's own stage plugins run), then drop_identity_final_layout().
    The one call behind the chunk and SPAM transpiles, the health check and the layout grading. Returns the same shape
    (a circuit for a circuit, a list for a list)."""
    from qiskit import transpile
    from qiskit.transpiler import Target
    isa = (transpile(circuit, target=backend, optimization_level=1, initial_layout=layout) if isinstance(backend, Target)
           else transpile(circuit, backend, optimization_level=1, initial_layout=layout))
    if isinstance(isa, list):
        return drop_identity_final_layout(isa)
    return drop_identity_final_layout([isa])[0]


def transpiled_1q_counts(circuit, backend, initial_layout):
    """{virtual qubit: native single-qubit gates} of `circuit` transpiled (locally, nothing is
    submitted) onto `backend` with `initial_layout` — virtual-Z rz and non-gates not counted."""
    from source.core.records.circuit_check import physical_index_map
    isa  = transpile_like_submission(circuit, backend, initial_layout)
    phys = physical_index_map(isa)
    p2v  = {p: v for v, p in enumerate(initial_layout)}
    counts = Counter()
    for inst in isa.data:
        if len(inst.qubits) == 1 and inst.operation.name not in _NON_GATES | _FREE_1Q:
            counts[p2v[phys[inst.qubits[0]]]] += 1
    return counts


_NO_QUBIT_DATA_OPS = {"measure", "delay", "rz", "reset", "barrier", "if_else", "for_loop", "while_loop", "switch_case",
                      "break_loop", "continue_loop", "store"}
_FREE_ERROR_OPS = {"rz", "delay", "barrier", "id"}   # no error of their own when the calibration lists none


def target_provider(target, backend=None):
    """Key of REQUIRED_FIELDS for a Target (and the backend it belongs to, a snapshot backend names its provider)."""
    snap = getattr(backend, "snapshot", None) or {}
    if snap.get("backend") == "iqm_eqe1":
        return "eqe1"
    names = set(target.operation_names)
    if "rxx" in names:
        return "aqt"
    return "ibm" if "sx" in names else "iqm"


def _bad_error(e):
    return e is None or not 0 <= e < 1


def unusable_in_target(target, provider=None, backend=None):
    """(qubits, couplers) of a Target that miss a field the QPU normally reports (REQUIRED_FIELDS), or have an error of 1 or
    more: {qubit: [fields]} and {(a, b) sorted: [fields]}. A field counts only where the Target reports it for some qubit
    (coupler), so a Target without calibration data at all has none missing. A coupler with an unusable end qubit is not listed
    here; its qubits are."""
    fields = REQUIRED_FIELDS[provider or target_provider(target, backend)]
    names  = set(target.operation_names)
    ops_1q = [n for n in sorted(names) if n not in _NO_QUBIT_DATA_OPS
              and any(q is not None and len(q) == 1 for q in target[n])]
    qubits = {q[0] for n in ops_1q + (["measure"] if "measure" in names else []) for q in target[n]
              if q is not None and len(q) == 1}
    bad_q, bad_c = {}, {}

    def flag(store, key, field):
        store.setdefault(key, []).append(field)

    def reported(op, what):   # does any entry of the operation report an error / a duration?
        return any(p is not None and getattr(p, what) is not None for q, p in target[op].items() if q is not None)

    props = getattr(target, "qubit_properties", None) or []
    for field in ("t1", "t2"):
        if field in fields["qubit"] and any(p is not None and getattr(p, field) for p in props):
            for q in qubits:
                p = props[q] if q < len(props) else None
                v = getattr(p, field, None) if p is not None else None
                if not v or v <= 0:
                    flag(bad_q, q, field)
    if "measure" in names:
        for field, what in (("readout", "error"), ("readout_duration", "duration")):
            if field in fields["qubit"] and reported("measure", what):
                for q in qubits:
                    p = target["measure"].get((q,))
                    v = getattr(p, what, None) if p is not None else None
                    if (_bad_error(v) if what == "error" else v is None):
                        flag(bad_q, q, field)
    for field, what in (("1q_error", "error"), ("1q_duration", "duration")):
        if field not in fields["qubit"]:
            continue
        for op in ops_1q:
            if not reported(op, what):
                continue
            for q in qubits:
                p = target[op].get((q,))
                v = getattr(p, what, None) if p is not None else None
                if (_bad_error(v) if what == "error" else v is None):
                    flag(bad_q, q, f"{op} {what}")

    name_2q = _pick_op(target, _PREFERRED_2Q, 2)
    if name_2q is not None and fields["coupler"]:
        for field, what in (("2q_error", "error"), ("2q_duration", "duration")):
            if field not in fields["coupler"] or not reported(name_2q, what):
                continue
            for q, p in target[name_2q].items():
                if q is None or len(q) != 2:
                    continue
                v = getattr(p, what, None) if p is not None else None
                if (_bad_error(v) if what == "error" else v is None):
                    flag(bad_c, tuple(sorted(q)), f"{name_2q} {what}")
    for key in bad_q:
        bad_q[key] = sorted(set(bad_q[key]))
    for key in bad_c:
        bad_c[key] = sorted(set(bad_c[key]))
    return bad_q, bad_c


def target_view(target):
    """The calibration of a Target as score_transpiled() reads it: {"err": {(op, qargs): error}, "dur": {(op, qargs): s},
    "t1t2": {qubit: (T1, T2)}, "dt": dt, "resonator": None}."""
    err, dur = {}, {}
    for name in target.operation_names:
        for q, p in target[name].items():
            if q is None or p is None:
                continue
            if p.error is not None:
                err[name, tuple(q)] = p.error
            if p.duration is not None:
                dur[name, tuple(q)] = p.duration
    return {"err": err, "dur": dur, "t1t2": _coherence_times(target), "dt": target.dt, "resonator": None}


def _lookup(table, name, qargs):
    v = table.get((name, qargs))
    if v is None and len(qargs) == 2:
        v = table.get((name, qargs[::-1]))
    return v


def circuit_timing(tc, view):
    """Our own ALAP timing of a transpiled circuit (the circuit is not changed; nothing is scheduled in the submission):
    every instruction starts as late as its successors allow, with the durations of `view["dur"]`. An instruction occupies all
    its qubits, so a MOVE or CZ with the computational resonator (an ordinary qubit of the circuit, `view["resonator"]`) serializes
    everything that uses the resonator. Barriers take no time but synchronize their qubits; measure and delay take their own.
    Returns None if an instruction has no duration, else {"T": circuit time in s, "active": {physical qubit: time from its
    first gate to the end}, "busy": {physical qubit: time its gates take}}."""
    from source.core.records.circuit_check import physical_index_map
    phys = physical_index_map(tc)
    left, busy = {}, {}   # time from the end of the circuit that a qubit is occupied until (walking backwards)
    for inst in reversed(tc.data):
        op, qs = inst.operation, [phys[q] for q in inst.qubits]
        if op.name == "barrier":
            d = 0.0
        elif op.name == "delay":
            d = op.params[0] * (view["dt"] if op.unit == "dt" else {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[op.unit])
        else:
            d = _lookup(view["dur"], op.name, tuple(qs))
            if d is None and op.name == "rz":
                d = 0.0
            if d is None:
                return None
        start = max(left.get(q, 0.0) for q in qs) + d
        for q in qs:
            left[q] = start
            if op.name != "barrier":
                busy[q] = busy.get(q, 0.0) + d
    return {"T": max(left.values(), default=0.0), "active": left, "busy": busy}


def score_transpiled(tc, view):
    """Exact score of a transpiled circuit `tc` on the calibration `view` (target_view(), iqm_star.star_view()):

      gate_log   Σ over every instruction of log(1 − its own error on its physical qubits): single-qubit gates, two-qubit gates,
                 MOVEs (a star machine's MOVE pair takes the measured MOVE→MOVE IRB: each MOVE carries half its log) and measure
                 (rz and the like have no error of their own)
      idle_log   Σ over the circuit's qubits (not the resonator) of idle_decay_log(T1, T2, t_idle), with
                 t_idle = end of the circuit − the qubit's first gate start − the qubit's busy time (circuit_timing())
      t_circuit_s, counts (tc.count_ops())

    Returns a dict with log_p = gate_log + idle_log (idle_log, t_circuit_s None when a duration or a T1/T2 is missing: the idle
    term is then left out), or None if a gate has no error in the calibration."""
    from source.core.records.circuit_check import physical_index_map
    phys     = physical_index_map(tc)
    gate_log = 0.0
    for inst in tc.data:
        name = inst.operation.name
        if name == "barrier" or name == "delay":
            continue
        qs = tuple(phys[q] for q in inst.qubits)
        e  = _lookup(view["err"], name, qs)
        if e is None:
            if name in _FREE_ERROR_OPS:
                continue
            return None
        if e >= 1:
            return None
        gate_log += math.log(1.0 - e)
    out = dict(gate_log=gate_log, idle_log=None, t_circuit_s=None, counts=dict(tc.count_ops()), log_p=gate_log)
    timing = circuit_timing(tc, view)
    if timing is None:
        return out
    res   = view.get("resonator")
    qubits = [q for q in timing["active"] if q != res]
    if not all(q in view["t1t2"] for q in qubits):
        return out
    idle = sum(idle_decay_log(*view["t1t2"][q], max(timing["active"][q] - timing["busy"][q], 0.0)) for q in qubits)
    return dict(out, idle_log=idle, t_circuit_s=timing["T"], log_p=gate_log + idle)


def _exact_rescore(circuit, backend, view, candidates):
    """[(placement, exact record)] of the candidate placements, transpiled like the submission and scored (score_transpiled());
    a placement that cannot be transpiled or scored is left out."""
    out = []
    for phys in candidates:
        try:
            tc = transpile_like_submission(circuit, backend, list(phys))
        except Exception:
            continue
        rec = score_transpiled(tc, view)
        if rec is not None:
            out.append((phys, rec))
    return out


def select_physical_qubits(circuit, target, singles=None, placement_ok=None, transpile_backend=None, view=None):
    """Pick the physical qubits for `circuit` on a backend described by `target`.
    singles: {virtual qubit: single-qubit gate count} — transpiled_1q_counts(); None counts
    them on the raw circuit (only where the circuit cannot be transpiled onto the backend).
    placement_ok: optional predicate on a placement (tuple of physical qubits); placements it rejects are not
    considered — e.g. placement_filter() of the live QPU when the scoring Target comes from a calibration file.
    transpile_backend: what the circuit is transpiled onto for the exact score (a backend, or a Target); None scores the
    prescreen only. view: the calibration for score_transpiled() (default target_view(target)).

    Returns a dict (JSON-serialisable) with
      physical_qubits  list[int] | None — initial_layout for virtual qubits 0..n-1,
                                          None if no layout should be pinned
      method           "transpiled" | "scored" | "no_error_data" | "no_coupling_map" | "no_swap_free_placement"
                       | "no_scored_placement"
      log_success      float | None     — log P of the chosen placement (exact score for "transpiled")
      top_candidates   [[qubits, log P], …] — best few, for the record
      n_candidates     int (choose_layout() records None for a forced list)
      gates            {"2q": name, "1q": name}
      n_1q             [single-qubit gates per virtual qubit], n_1q_source "transpiled" | "raw circuit" (the prescreen's)
      decoherence_included  bool — whether log_success contains the idle-decay term (see below)
      t_circuit_s      the chosen placement's circuit duration in s (None without the term)
      incomplete       {"qubits": {…}, "couplers": {…}} the fields each unusable qubit / coupler misses (REQUIRED_FIELDS), if any
    and, for "transpiled": prescreen_log_success, idle_log, and per top candidate top_candidate_prescreen (the prescreen
    score), top_candidate_gate_counts (count_ops() of its transpiled circuit) and top_candidate_t_circuit_s.

    Selection. Qubits and couplers missing a field the QPU normally reports, or with an error of 1 or more, are excluded
    (unusable_in_target()). The rest: a fast prescreen scores every placement that needs no SWAP,
        log P = Σ_pairs n_2q · log(1 − e_2q) + Σ_qubits log(1 − e_readout) + Σ_qubits n_1q · log(1 − e_1q),
    with the raw circuit's 2q counts and transpiled_1q_counts() single-qubit counts; the best N_EXACT_RESCORE are then
    transpiled like the submission (transpile_like_submission()) and scored exactly (score_transpiled()): every gate's own error
    on the transpiled circuit, plus the idle decay of a qubit that waits (our ALAP timing, T1/T2 from the Target), which the
    gate and readout errors do not hold. Without durations or T1/T2 the idle term is left out for every placement.
    """
    n_virtual = circuit.num_qubits
    name_2q   = _pick_op(target, _PREFERRED_2Q, 2)
    name_1q   = _pick_op(target, _PREFERRED_1Q, 1)
    err_2q    = _errors(target, name_2q)
    err_ro    = _errors(target, "measure" if "measure" in target.operation_names else None)
    err_1q    = _errors(target, name_1q)
    pairs, raw_singles = _interaction_counts(circuit)
    n_1q_source = "raw circuit" if singles is None else "transpiled"
    singles     = raw_singles if singles is None else singles
    base      = dict(gates={"2q": name_2q, "1q": name_1q}, log_success=None,
                     top_candidates=[], n_candidates=0,
                     decoherence_included=False, t_circuit_s=None,
                     n_1q=[int(singles.get(v, 0)) for v in range(n_virtual)], n_1q_source=n_1q_source)

    if not err_2q:
        # No coupling restriction (e.g. AQT's all-to-all target): nothing to pin.
        return dict(base, physical_qubits=None, method="no_coupling_map")

    bad_q, bad_c = unusable_in_target(target, backend=transpile_backend)
    if bad_q or bad_c:
        base["incomplete"] = {"qubits": {str(q): f for q, f in sorted(bad_q.items())},
                              "couplers": {f"{a}-{b}": f for (a, b), f in sorted(bad_c.items())}}

    neighbors = {p: set() for p in range(target.num_qubits)}
    for q in err_2q:
        if len(q) == 2:
            a, b = q
            if a in bad_q or b in bad_q or tuple(sorted(q)) in bad_c:
                continue
            neighbors[a].add(b); neighbors[b].add(a)

    def pair_err(a, b):
        return err_2q.get((a, b), err_2q.get((b, a)))

    has_data = any(e is not None for e in (*err_2q.values(), *err_ro.values(), *err_1q.values()))

    scored = []
    any_placement = False
    for phys in _placements(n_virtual, pairs, neighbors, target.num_qubits):
        any_placement = True
        if placement_ok is not None and not placement_ok(phys):
            continue
        if not has_data:
            scored.append((phys, 0.0))
            continue
        if any(q in bad_q for q in phys):
            continue   # incomplete calibration of a qubit → excluded
        terms = []
        for (va, vb), n in pairs.items():
            terms.append((pair_err(phys[va], phys[vb]), n))
        for v in range(n_virtual):
            if err_ro:
                terms.append((err_ro.get((phys[v],)), 1))
            if err_1q and singles.get(v):
                terms.append((err_1q.get((phys[v],)), singles[v]))
        if any(e is None or e >= 1 for e, _ in terms):
            continue   # missing (or useless) calibration entry → excluded
        scored.append((phys, sum(n * math.log(1.0 - e) for e, n in terms)))

    if not scored:
        # No SWAP-free placement at all, or some exist but every one was dropped (placement_ok, or an incomplete / unit
        # calibration entry): only the first is a circuit/QPU mismatch that no layout can fix.
        if not any_placement and (bad_q or bad_c):   # would a placement exist if the unusable qubits and couplers were complete?
            full = {p: set() for p in range(target.num_qubits)}
            for q in err_2q:
                if len(q) == 2:
                    full[q[0]].add(q[1]); full[q[1]].add(q[0])
            any_placement = next(_placements(n_virtual, pairs, full, target.num_qubits), None) is not None
        return dict(base, physical_qubits=None,
                    method="no_swap_free_placement" if not any_placement else "no_scored_placement")

    scored.sort(key=lambda s: (-s[1], s[0]))
    n_candidates = len(scored)
    rescored = []
    if has_data and transpile_backend is not None:
        view     = view or target_view(target)
        prescreen = dict(scored)
        rescored = _exact_rescore(circuit, transpile_backend, view, [p for p, _ in scored[:N_EXACT_RESCORE]])
        if rescored:
            # one comparable score: with the idle term only if every rescored placement has it
            with_idle = all(r["idle_log"] is not None for _, r in rescored)
            key = (lambda r: r["log_p"]) if with_idle else (lambda r: r["gate_log"])
            rescored.sort(key=lambda pr: (-key(pr[1]), pr[0]))
            best, best_rec = rescored[0]
            top = rescored[:N_TOP_CANDIDATES_RECORDED]
            return dict(
                base,
                physical_qubits = list(best),
                method          = "transpiled",
                log_success     = key(best_rec),
                prescreen_log_success = prescreen[best],
                top_candidates  = [[list(p), key(r)] for p, r in top],
                top_candidate_prescreen = [prescreen[p] for p, _ in top],
                top_candidate_gate_counts = [r["counts"] for _, r in top],
                top_candidate_t_circuit_s = [r["t_circuit_s"] for _, r in top],
                n_candidates    = n_candidates,
                decoherence_included = with_idle,
                t_circuit_s     = best_rec["t_circuit_s"] if with_idle else None,
                idle_log        = best_rec["idle_log"] if with_idle else None,
            )
    best, best_log = scored[0]
    return dict(
        base,
        physical_qubits = list(best),
        method          = "scored" if has_data else "no_error_data",
        log_success     = best_log if has_data else None,
        top_candidates  = [[list(p), s] for p, s in scored[:N_TOP_CANDIDATES_RECORDED]],
        n_candidates    = n_candidates,
    )


def _usable_qubits_and_neighbors(target):
    """(usable physical qubits, {qubit: coupled qubits}) of a Target.
    Usable = the qubits it lists measure / single-qubit-gate entries for; a Target that lists none per qubit
    (AQT's: global operations) has all of range(num_qubits). Couplings come from the 2q gate's qargs; without any
    (AQT: all-to-all) every pair of usable qubits is coupled. Both are restricted to the usable qubits."""
    names  = set(target.operation_names)
    usable = set()
    for name in ("measure", _pick_op(target, _PREFERRED_1Q, 1)):
        if name in names:
            usable |= {q[0] for q in target[name] if q is not None and len(q) == 1}
    if not usable:
        usable = set(range(target.num_qubits))
    bad_q, bad_c = unusable_in_target(target)
    usable -= set(bad_q)
    err_2q = _errors(target, _pick_op(target, _PREFERRED_2Q, 2))
    edges  = [q for q in err_2q if len(q) == 2 and q[0] in usable and q[1] in usable and tuple(sorted(q)) not in bad_c]
    if edges:
        neighbors = {p: set() for p in usable}
        for a, b in edges:
            neighbors[a].add(b); neighbors[b].add(a)
    else:
        neighbors = {p: usable - {p} for p in usable}
    return usable, neighbors


def placement_filter(circuit, target):
    """Predicate on a placement (tuple of physical qubits, virtual qubit i -> qubits[i]): True if the circuit can run
    SWAP-free there on `target` — the qubits exist and every interacting pair is coupled (what check_forced_layout
    checks, without raising). For restricting a layout chosen on a calibration file's Target to what the live QPU has."""
    usable, neighbors = _usable_qubits_and_neighbors(target)
    pairs, _ = _interaction_counts(circuit)
    return lambda phys: (all(q in usable for q in phys)
                         and all(phys[b] in neighbors[phys[a]] for a, b in pairs))


def _target_couplers(target):
    """The undirected couplers of a Target as a set of sorted pairs (any two-qubit operation)."""
    return {tuple(sorted(q)) for name in target.operation_names for q in (target[name] or {})
            if q is not None and len(q) == 2 and q[0] != q[1]}


def eqe1_live_mapping(snapshot, live_target):
    """{calibration-file qubit: live qubit} for IQM EQE1 (Euro-Q-Exa), whose calibration file numbers qubits 0..53 including
    coupler-less placeholder qubits (which ones changes over time), which the live backend leaves out (it numbers
    the others 0..N-1 in order). The placeholders are the file's qubits with no coupler (snapshot "placeholder_qubits"); the rest
    are mapped in order. The result is then checked against the live coupling graph: the file's couplers under the map must be
    exactly the live couplers. If they are not, an order-preserving subgraph matching of the file's coupler graph onto the live
    one (rustworkx) is tried; the matched file qubits are mapped, the others have no live counterpart and are dropped.
    ValueError (one line) if the graphs cannot be matched: the numbering is then unknown."""
    n_file = snapshot["num_qubits"]
    if snapshot.get("file_couplers") is not None:
        couplers = [tuple(c) for c in snapshot["file_couplers"]]
    else:   # a snapshot saved without the file's coupler list: what its Target has (fidelity-less couplers are missing)
        couplers = sorted({tuple(sorted(e["qargs"])) for e in snapshot["operations"].get("cz", [])})
    in_pair      = {q for c in couplers for q in c}
    placeholders = set(snapshot["placeholder_qubits"]) if snapshot.get("placeholder_qubits") is not None \
                   else {q for q in range(n_file) if q not in in_pair}
    kept     = [q for q in range(n_file) if q not in placeholders]
    mapping  = {q: r for r, q in enumerate(kept)}
    live_set = _target_couplers(live_target)
    n_live   = live_target.num_qubits
    file_set = {tuple(sorted((mapping[a], mapping[b]))) for a, b in couplers if a in mapping and b in mapping}
    if len(kept) == n_live and file_set == live_set:
        return mapping

    import rustworkx as rx
    file_g, live_g = rx.PyGraph(), rx.PyGraph()
    file_g.add_nodes_from(range(n_file))
    live_g.add_nodes_from(range(n_live))
    file_g.add_edges_from_no_data([c for c in couplers if c[0] in mapping and c[1] in mapping])
    live_g.add_edges_from_no_data(sorted(live_set))
    for found in itertools.islice(rx.vf2_mapping(file_g, live_g, subgraph=True, induced=False), 100000):
        pairs = sorted(dict(found).items())
        if len(pairs) == n_live and all(a[1] < b[1] for a, b in zip(pairs, pairs[1:])):   # keeps the qubits' order
            return dict(pairs)
    raise ValueError(f"the calibration file's coupler graph ({len(kept)} qubits, {len(file_set)} couplers without the "
                     f"placeholders) does not match the live backend's ({n_live} qubits, {len(live_set)} couplers): "
                     f"the file's qubit numbering cannot be translated to the live one")


def calibration_to_live(layout_backend, backend):
    """The map from the numbering of the calibration a layout is graded on to the live backend's, for a run graded on a
    calibration file (`layout_backend`, a snapshot backend) and executed on `backend`. IQM EQE1: {file qubit: live qubit}
    (eqe1_live_mapping(); its numbering differs from the file's). Any other QPU numbers its qubits as its calibration does, so
    None; AQT gets a guard instead: PIAST-Q's executed circuits are not returned, so the numbering cannot be checked afterwards,
    and the file must at least list as many ions as the live backend (ValueError otherwise). Also None without a live Target."""
    snap   = getattr(layout_backend, "snapshot", None)
    target = getattr(backend, "target", None)
    if snap is None or target is None or backend is layout_backend:
        return None
    if snap.get("backend") == "iqm_eqe1":
        return eqe1_live_mapping(snap, target)
    if str(snap.get("backend", "")).startswith("aqt") and target.num_qubits != snap["num_qubits"]:
        raise ValueError(f"the calibration file lists {snap['num_qubits']} ions but the live backend has {target.num_qubits}: "
                         f"its ion numbering cannot be trusted (the executed circuits are not returned to check it)")
    return None


def live_qubits_of_record(record, backend, to_live=None, calibration=None):
    """The recorded layout's qubits as numbered by the live `backend` today, for a rerun on another day (hardware identity,
    not the old indices): EQE1's record holds "calibration_qubits" (calibration-file numbering), converted through `to_live`
    (calibration_to_live()); an IQM backend's record holds "qubit_names", converted through the backend's qubit_name_to_index()
    (offline: iqm_star.qubit_index_map() of the raw `calibration`). Returns (qubits, note): note is a ⚠️ line for a record
    that holds neither (made before the names were recorded; its indices are reused as they are), else None. ValueError if a
    recorded qubit no longer exists."""
    old = record["physical_qubits"]
    if to_live is not None:
        cq = record.get("calibration_qubits")
        if cq is None:
            return old, ("  ⚠️  [Layout] the recorded layout has no calibration-file numbering (calibration_qubits): "
                         "its qubit indices are reused as they are.")
        gone = [q for q in cq if q not in to_live]
        if gone:
            raise ValueError(f"recorded qubit(s) {gone} (calibration-file numbering) have no counterpart on the live backend today")
        return [to_live[q] for q in cq], None
    names = record.get("qubit_names")
    if hasattr(backend, "qubit_name_to_index") or calibration is not None:
        if not names:
            return old, ("  ⚠️  [Layout] the recorded layout has no qubit names (qubit_names): "
                         "its qubit indices are reused as they are.")
        if hasattr(backend, "qubit_name_to_index"):
            lookup = backend.qubit_name_to_index
        else:
            from source.core.qpu.iqm_star import qubit_index_map
            idx    = qubit_index_map(calibration)
            lookup = lambda n: idx[n]
        try:
            return [lookup(n) for n in names], None
        except (KeyError, ValueError):
            raise ValueError(f"recorded qubit(s) {names} no longer all exist on the backend") from None
    return old, None


def live_placement_filter(circuit, live_target, to_live):
    """placement_filter() for placements in the calibration file's numbering, checked on the live Target through `to_live`
    ({file qubit: live qubit}): only what both the file and the live backend have is placed (a qubit without a live
    counterpart, or a coupler the live Target lacks, is out)."""
    live_ok = placement_filter(circuit, live_target)
    return lambda phys: all(q in to_live for q in phys) and live_ok(tuple(to_live[q] for q in phys))


def all_placements(circuit, target, placement_ok=None):
    """Every SWAP-free placement of `circuit` on `target`, sorted: tuples of physical qubits for virtual qubits 0..n-1."""
    usable, neighbors = _usable_qubits_and_neighbors(target)
    n_phys  = max(usable) + 1 if usable else 0
    nbrs    = {p: neighbors.get(p, set()) for p in range(n_phys)}
    pairs, _ = _interaction_counts(circuit)
    return sorted(p for p in _placements(circuit.num_qubits, pairs, nbrs, n_phys)
                  if placement_ok is None or placement_ok(p))


def check_forced_layout(circuit, target, qubits):
    """Raise ValueError unless `qubits` (virtual qubit i -> physical qubits[i]) is a valid, SWAP-free layout of
    `circuit` on `target`: right length, distinct, usable qubits, every interacting pair on a coupler."""
    qubits = list(qubits)
    usable, neighbors = _usable_qubits_and_neighbors(target)
    if len(qubits) != circuit.num_qubits:
        raise ValueError(f"layout {qubits} has {len(qubits)} qubits; the circuits use {circuit.num_qubits}")
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"layout {qubits} repeats a qubit")
    bad = [q for q in qubits if q not in usable]
    if bad:
        raise ValueError(f"layout {qubits}: qubit(s) {bad} do not exist or are not usable on this QPU "
                         f"(usable: {min(usable)}…{max(usable)}, {len(usable)} qubits"
                         + (f", missing {sorted(set(range(max(usable) + 1)) - usable)}" if len(usable) <= max(usable) else "") + ")")
    pairs, _ = _interaction_counts(circuit)
    off = [(qubits[a], qubits[b]) for a, b in pairs if qubits[b] not in neighbors[qubits[a]]]
    if off:
        raise ValueError(f"layout {qubits} is incompatible with the QPU's connectivity: the circuit needs "
                         f"{'a coupler' if len(off) == 1 else 'couplers'} between {off}, which {'does' if len(off) == 1 else 'do'} not exist")


def choose_layout(mode, circuit, target, qubits=None, singles=None, seed=None, placement_ok=None):
    """The run's physical qubits by `mode`, as the record select_physical_qubits() returns (plus "layout_mode"):

      optimized  select_physical_qubits(): the placement with the best estimated success probability
      random     one of all SWAP-free placements, drawn with random.Random(seed) (a fresh seed if None; recorded)
      lowest     the lexicographically lowest SWAP-free placement
      list       `qubits`, forced — ValueError if the QPU cannot run them (check_forced_layout)

    `circuit` is the representative raw circuit; `target` the QPU's (or its calibration snapshot's); placement_ok
    restricts the placements considered (see placement_filter())."""
    if mode == "optimized":
        rec = select_physical_qubits(circuit, target, singles, placement_ok)
    elif mode == "list":
        check_forced_layout(circuit, target, qubits)
        rec = dict(physical_qubits=list(qubits), method="user", log_success=None, top_candidates=[],
                   n_candidates=None, gates=None)
    elif mode in ("random", "lowest"):
        cands = all_placements(circuit, target, placement_ok)
        if not cands:
            raise ValueError("no SWAP-free layout of the circuits exists on this QPU's connectivity")
        rec = dict(log_success=None, top_candidates=[], n_candidates=len(cands), gates=None)
        if mode == "lowest":
            rec.update(physical_qubits=list(cands[0]), method="lowest")
        else:
            import random
            import secrets
            seed = secrets.randbits(32) if seed is None else seed
            rec.update(physical_qubits=list(random.Random(seed).choice(cands)), method="random", seed=seed)
    else:
        raise ValueError(f"unknown layout mode {mode!r}")
    rec["layout_mode"] = mode
    return rec


def drop_identity_final_layout(circuits):
    """Remove a final_layout that permutes nothing (every qubit ends where the initial layout put
    it), in place. IQM's star (VLQ) transpilation attaches such a layout, keyed so that QPY
    stores it as a real permutation of the circuit's 25 qubits (24 + the resonator, which the
    input mapping lacks): reloaded, final_physical_qubits() then gives wrong qubits. Without it
    the circuit means the same and round-trips through QPY. Returns the circuits.

    iqm-client 36 on Sirius keys the final layout on the circuit's own qubits (one register of 17, none of them in
    the initial layout) instead of the input qubits: it is the identity when every physical qubit p maps to the
    circuit qubit at index p."""
    for c in circuits:
        lay = getattr(c, "layout", None)
        if lay is None or lay.final_layout is None:
            continue
        init = lay.initial_layout
        placed = init.get_virtual_bits()
        if all(init[v] == p if v in placed else c.find_bit(v).index == p
               for p, v in lay.final_layout.get_physical_bits().items()):
            lay.final_layout = None
    return circuits


def final_physical_qubits(circuit):
    """Physical qubit of each virtual qubit at the end of a transpiled circuit
    (after routing), or None if the transpiler attached no layout (e.g. AQT)."""
    layout = getattr(circuit, "layout", None)
    if layout is None:
        return None
    return list(layout.final_index_layout())


def run_layout_warnings(run_path):
    """Check a run's transpiled circuits for physical-qubit mismatches.

    Looks at chunk 0 (all chunks share the same structure up to single-qubit
    gates) and the SPAM batch in <run_path>/transpiled_circuits/. Returns a
    list of human-readable warnings — empty if everything sits on one set of
    qubits, or if there is nothing transpiled to check."""
    import warnings
    from pathlib import Path
    from qiskit.qpy import load

    tdir = Path(run_path) / "transpiled_circuits"
    chunk_path, spam_path = tdir / "chunk_0_transpiled.qpy", tdir / "spam_transpiled.qpy"
    if not chunk_path.exists():
        return []

    def layouts(path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # QPY written by a slightly newer qiskit
            with open(path, "rb") as f:
                circs = load(f)
        return Counter(tuple(l) if l is not None else None for l in map(final_physical_qubits, circs))

    msgs = []
    chunk = layouts(chunk_path)
    if len(chunk) > 1:
        msgs.append(f"chunk 0's circuits are spread over several physical-qubit sets: {dict(chunk)}")
    if spam_path.exists():
        spam = layouts(spam_path)
        if set(spam) != set(chunk):
            msgs.append(f"SPAM calibration ran on {sorted(spam, key=str)} but the circuits on "
                        f"{sorted(chunk, key=str)} — SPAM-corrected results do not describe the "
                        f"qubits that were used")
    return msgs
