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
of a representative (deepest) circuit on it,

    log P = Σ_pairs  n_2q(pair)  · log(1 − e_2q(pair))
          + Σ_qubits log(1 − e_readout(q))
          + Σ_qubits n_1q(q)     · log(1 − e_1q(q)),

and pick the highest. n_2q is counted on the raw (untranspiled) circuit, where
it is exact (transpilation must not change it — see count_2q_ops). n_1q is the
number of native single-qubit gates the backend runs: transpiled_1q_counts()
transpiles the representative circuit (locally) onto one SWAP-free placement
(probe_placement(); the counts are the same on every placement, since every
qubit has the same native gates) and counts each virtual qubit's single-qubit
gates, leaving out virtual-Z rz (applied in software, no error). The raw
circuit's 1q count (rz, h, rx, ry, …) would overstate them about threefold.

Missing data: if the backend reports no errors at all (e.g. IQM EQE1 via
MQSS), every placement scores equally and the lexicographically smallest
one is chosen — not optimised, but deterministic and recorded. If it reports
errors for some operations only, placements touching a missing entry are
excluded rather than treated as error-free.

Uses only qiskit, so it works in every backend venv (aqt.venv runs qiskit 1.4).
"""

import math
from collections import Counter

# Preferred native gate names, most specific first; the first one present in
# the backend's Target is used for that gate class.
_PREFERRED_2Q = ("cz", "ecr", "cx", "rxx", "rzz", "iswap")
_PREFERRED_1Q = ("r", "sx", "rx", "x", "u")

# Non-gate instructions that can span several qubits.
_NON_GATES = {"barrier", "measure", "reset", "delay"}

N_TOP_CANDIDATES_RECORDED = 5


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


def _durations(target, name):
    """{qargs: duration in s} of one operation, the entries that have one. {} if name is None."""
    if name is None:
        return {}
    out = {}
    for q, p in target[name].items():
        if q is None:
            continue
        if p is not None and p.duration is not None:
            out[tuple(q)] = p.duration
    return out


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


def transpiled_1q_counts(circuit, backend, initial_layout):
    """{virtual qubit: native single-qubit gates} of `circuit` transpiled (locally, nothing is
    submitted) onto `backend` with `initial_layout` — virtual-Z rz and non-gates not counted."""
    from qiskit import transpile
    from source.core.records.circuit_check import physical_index_map
    isa  = transpile(circuit, backend, optimization_level=1, initial_layout=initial_layout)
    phys = physical_index_map(isa)
    p2v  = {p: v for v, p in enumerate(initial_layout)}
    counts = Counter()
    for inst in isa.data:
        if len(inst.qubits) == 1 and inst.operation.name not in _NON_GATES | _FREE_1Q:
            counts[p2v[phys[inst.qubits[0]]]] += 1
    return counts


def select_physical_qubits(circuit, target, singles=None, placement_ok=None):
    """Pick the physical qubits for `circuit` on a backend described by `target`.
    singles: {virtual qubit: single-qubit gate count} — transpiled_1q_counts(); None counts
    them on the raw circuit (only where the circuit cannot be transpiled onto the backend).
    placement_ok: optional predicate on a placement (tuple of physical qubits); placements it rejects are not
    considered — e.g. placement_filter() of the live QPU when the scoring Target comes from a calibration file.

    Returns a dict (JSON-serialisable) with
      physical_qubits  list[int] | None — initial_layout for virtual qubits 0..n-1,
                                          None if no layout should be pinned
      method           "scored" | "no_error_data" | "no_coupling_map" | "no_swap_free_placement"
                       | "no_scored_placement"
      log_success      float | None     — log P of the chosen placement
      top_candidates   [[qubits, log P], …] — best few, for the record
      n_candidates     int (choose_layout() records None for a forced list)
      gates            {"2q": name, "1q": name}
      n_1q             [single-qubit gates per virtual qubit], n_1q_source "transpiled" | "raw circuit"
      decoherence_included  bool — whether log_success contains the idle-decay term (see below)
      t_circuit_s      the chosen placement's estimated circuit duration in s (None without the term)

    Idle decay. The gate and readout errors are measured fidelities, so they already hold the decay during a gate or a
    measurement. What they miss is a qubit waiting while the others work. With T1/T2 (Target.qubit_properties) and gate
    durations, each placement also gets  Σ_qubits log F(t_idle; T1, T2)  (idle_decay_log()), where the 2q gates of the
    3-qubit circuit run one after another, so  T = Σ 2q durations + the longest single-qubit run of one qubit  and
    t_idle(v) = T − (v's own gate time). A placement missing a T1, T2 or duration is dropped from the scored ones; if
    none has the data (no durations at all, or AQT without a calibration file), the term is left out for every
    placement. AQT with a calibration file has per-ion T1/T2 and r/rxx durations, so the term is included (the same
    value on every placement).
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

    neighbors = {p: set() for p in range(target.num_qubits)}
    for q in err_2q:
        if len(q) == 2:
            a, b = q
            neighbors[a].add(b); neighbors[b].add(a)

    def pair_err(a, b):
        return err_2q.get((a, b), err_2q.get((b, a)))

    has_data = any(e is not None for e in (*err_2q.values(), *err_ro.values(), *err_1q.values()))

    coherence = _coherence_times(target)
    dur_1q    = _durations(target, name_1q)
    dur_2q    = _durations(target, name_2q)

    def idle_terms(phys):
        """(Σ log F of the idle qubits, T in s) of one placement; None if a T1, T2 or duration is missing."""
        d1   = [dur_1q.get((phys[v],)) for v in range(n_virtual)]
        d2   = {pair: (dur_2q.get((phys[pair[0]], phys[pair[1]])) or dur_2q.get((phys[pair[1]], phys[pair[0]])))
                for pair in pairs}
        if any(phys[v] not in coherence for v in range(n_virtual)) or any(d is None for d in d2.values()):
            return None
        if any(d1[v] is None for v in range(n_virtual) if singles.get(v)):
            return None
        own_1q = [singles.get(v, 0) * (d1[v] or 0.0) for v in range(n_virtual)]
        busy   = [own_1q[v] + sum(n * d2[pair] for pair, n in pairs.items() if v in pair) for v in range(n_virtual)]
        t_circ = sum(n * d2[pair] for pair, n in pairs.items()) + max(own_1q, default=0.0)
        return sum(idle_decay_log(*coherence[phys[v]], max(t_circ - busy[v], 0.0)) for v in range(n_virtual)), t_circ

    scored = []
    any_placement = False
    for phys in _placements(n_virtual, pairs, neighbors, target.num_qubits):
        any_placement = True
        if placement_ok is not None and not placement_ok(phys):
            continue
        if not has_data:
            scored.append((phys, 0.0))
            continue
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
        # No SWAP-free placement at all, or some exist but every one was dropped (placement_ok, or a missing / unit
        # calibration entry): only the first is a circuit/QPU mismatch that no layout can fix.
        return dict(base, physical_qubits=None,
                    method="no_swap_free_placement" if not any_placement else "no_scored_placement")

    t_circuit = {}
    if has_data and coherence and dur_1q and dur_2q:
        with_idle = []
        for phys, log_p in scored:
            idle = idle_terms(phys)
            if idle is not None:
                with_idle.append((phys, log_p + idle[0]))
                t_circuit[phys] = idle[1]
        if with_idle:
            scored = with_idle

    scored.sort(key=lambda s: (-s[1], s[0]))
    best, best_log = scored[0]
    return dict(
        base,
        physical_qubits = list(best),
        method          = "scored" if has_data else "no_error_data",
        log_success     = best_log if has_data else None,
        top_candidates  = [[list(p), s] for p, s in scored[:N_TOP_CANDIDATES_RECORDED]],
        n_candidates    = len(scored),
        decoherence_included = bool(t_circuit),
        t_circuit_s     = t_circuit.get(best),
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
    err_2q = _errors(target, _pick_op(target, _PREFERRED_2Q, 2))
    edges  = [q for q in err_2q if len(q) == 2 and q[0] in usable and q[1] in usable]
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
