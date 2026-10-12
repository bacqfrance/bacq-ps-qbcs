"""
circuit_check.py
Check that the circuits a provider executed are exactly the ones we submitted, and
summarise those checks for the analysis. No myQLM; qiskit is imported inside functions.

At collect time (source/core/qbcs.py, and source/core/health_check.py for health checks):
    IBM            the server's copy of the job's circuits, job.inputs["pubs"]
    IQM Resonance  the server's copy of the job's circuits, raw GET jobs/{id}/payload
    IQM VLQ        the circuits the server ran, qaas job.get_transpiled_circuits()
    IQM EQE1, AQT  not available — the services do not return the executed circuits
Each chunk / health check record gets
    circuit_check = {status, method, n_submitted, n_executed, n_differences, differences, checked_at}
(match / mismatch); not_available has {status, method (None), reason, checked_at} and
check_failed {status, method, error, checked_at}. status is one of:
    match          every circuit identical, gate by gate (see canonical form below)
    mismatch       the executed circuits differ from the submitted ones
    check_failed   the check itself could not run (e.g. malformed server data)
    not_available  the provider does not return the executed circuits

Canonical form: for each (named) qubit, the ordered list of operations touching it —
(name, angles mod 2π, qubits, measured classical bit). Two circuits are equal if every
qubit sees the same sequence, which is insensitive to the order in which a serializer
lists operations on disjoint qubits. Global phase is ignored.

Analysis (summarize_circuit_checks): a record whose status is not_available (EQE1, AQT,
and old records converted to the current format) is checked instead for
all of the run's transpiled circuits — chunks, the SPAM batch, the health checks — sitting on
the same physical qubits (method 'qubit_consistency'). Legacy runs — old IBM, IQM Resonance
and IQM EQE1 records without any circuit_check (job_timing.LEGACY_SERVICES) — get the same
check under method 'legacy_qubit_consistency', and the summary says legacy_run=True. An old AQT
record that was not converted to the current format stops the analysis.
"""

import json
import math
from collections import Counter
from pathlib import Path

from source.core.common import read_json_or_none as _read_json, scrub_paths
from source.core.records.job_records import chunk_record_files
from source.core.records.job_timing import now_utc_iso

MATCH, MISMATCH, CHECK_FAILED, NOT_AVAILABLE = "match", "mismatch", "check_failed", "not_available"
MAX_DIFFERENCES = 10

# IQM JSON gate names -> Qiskit ISA names (iqm-client's converter: r <-> prx)
IQM_TO_QISKIT = {"prx": "r", "cz": "cz", "measure": "measure", "barrier": "barrier"}
SYMMETRIC_2Q  = {"cz"}
LEGACY_CACHE_FILENAME = "circuit_qubit_check_cache.json"


def _now():
    return now_utc_iso(precise=True)


def _angle(x):
    v = float(x) % (2 * math.pi)
    if abs(v - 2 * math.pi) < 1e-9:
        v = 0.0
    return round(v, 9)


def clbit_key(circuit, clbit):
    """IQM's measurement key for a classical bit: '<reg>_<size>_<reg position>_<index>'."""
    loc = circuit.find_bit(clbit)
    if not loc.registers:
        return f"clbit_{loc.index}"
    reg, idx = loc.registers[0]
    return f"{reg.name}_{reg.size}_{list(circuit.cregs).index(reg)}_{idx}"


def physical_index_map(circuit):
    """Circuit qubit -> physical qubit index, the way the providers serialize it.

    Qiskit's own layout stage leaves physical qubit p at position p of the circuit.
    IQM's transpiler plugin instead keeps the virtual qubits (register 'q' + 'ancilla')
    and records the placement only in circuit.layout.initial_layout; iqm-client's
    serialize_instructions then reads the physical index from there. Same rule here:
    use initial_layout when its registers are exactly the circuit's, else the position."""
    lay = getattr(circuit, "layout", None)
    if lay is not None and lay.initial_layout.get_registers() == set(circuit.qregs):
        return {q: p for p, q in lay.initial_layout.get_physical_bits().items()}
    return {q: circuit.find_bit(q).index for q in circuit.qubits}


# Non-native 1q gates iqm-client converts to prx on the way out (qiskit_to_iqm.py)
_IQM_AS_R = {"x":  lambda p: (math.pi, 0.0),        "rx": lambda p: (p[0], 0.0),
             "y":  lambda p: (math.pi, math.pi / 2), "ry": lambda p: (p[0], math.pi / 2)}


def canonical_qiskit(circuit, qubit_name=None, iqm_native=False):
    """Per-qubit operation sequences of a (transpiled) Qiskit circuit.
    qubit_name: callable physical index -> name (IQM: backend.index_to_qubit_name);
    default str(index). iqm_native: express x/rx/y/ry as r, as iqm-client sends them."""
    name_of = qubit_name or str
    phys    = physical_index_map(circuit)
    seqs = {}
    for ins in circuit.data:
        op   = ins.operation
        name, raw = op.name, list(op.params)
        if iqm_native and name in _IQM_AS_R:
            name, raw = "r", list(_IQM_AS_R[name]([float(x) for x in raw]))
        qs   = [name_of(phys[q]) for q in ins.qubits]
        if name in SYMMETRIC_2Q or name == "barrier":
            qs = sorted(qs)
        extra = clbit_key(circuit, ins.clbits[0]) if name == "measure" and ins.clbits else None
        params = tuple(_angle(p) for p in raw) if name != "barrier" else ()
        token = (name, params, tuple(qs), extra)
        for q in set(qs):
            seqs.setdefault(q, []).append(token)
    return seqs


def canonical_iqm(circuit_json):
    """Per-qubit operation sequences of one circuit of an IQM Resonance job payload."""
    seqs = {}
    for ins in circuit_json["instructions"]:
        name = IQM_TO_QISKIT.get(ins["name"], ins["name"])
        qs   = list(ins["locus"])
        if name in SYMMETRIC_2Q or name == "barrier":
            qs = sorted(qs)
        args = ins.get("args") or {}
        if ins["name"] == "prx":
            params = (_angle(args["angle"]), _angle(args["phase"]))
        else:
            params = ()
        extra = args.get("key") if name == "measure" else None
        token = (name, params, tuple(qs), extra)
        for q in set(qs):
            seqs.setdefault(q, []).append(token)
    return seqs


def _diff_one(i, sub, exe):
    """First difference between two canonical circuits, or None."""
    if set(sub) != set(exe):
        return {"circuit": i, "what": "qubits", "submitted": sorted(sub), "executed": sorted(exe)}
    for q in sorted(sub):
        a, b = sub[q], exe[q]
        for k in range(max(len(a), len(b))):
            ta = a[k] if k < len(a) else None
            tb = b[k] if k < len(b) else None
            if ta != tb:
                return {"circuit": i, "what": "operation", "qubit": q, "position": k,
                        "submitted": list(ta) if ta else None, "executed": list(tb) if tb else None}
    return None


def compare_canonical(submitted, executed, shots_submitted=None, shots_executed=None, method=""):
    """Compare two lists of canonical circuits (same order). Returns a circuit_check dict."""
    diffs = []
    if len(submitted) != len(executed):
        diffs.append({"what": "circuit_count", "submitted": len(submitted), "executed": len(executed)})
    for i, (s, e) in enumerate(zip(submitted, executed)):
        d = _diff_one(i, s, e)
        if d:
            diffs.append(d)
    if shots_submitted is not None and shots_executed is not None:
        bad = sorted({str(s) for s in shots_executed if s is not None and int(s) != int(shots_submitted)})
        if bad:
            diffs.append({"what": "shots", "submitted": shots_submitted, "executed": bad})
    return {
        "status":      MISMATCH if diffs else MATCH,
        "method":      method,
        "n_submitted": len(submitted),
        "n_executed":  len(executed),
        "n_differences": len(diffs),
        "differences": diffs[:MAX_DIFFERENCES],
        "checked_at":  _now(),
    }


def not_available(reason):
    return {"status": NOT_AVAILABLE, "method": None, "reason": reason, "checked_at": _now()}


def check_failed(method, error):
    # no absolute path in the saved text (a FileNotFoundError carries the run path)
    fn = getattr(error, "filename", None) if isinstance(error, OSError) else None
    text = f"{type(error).__name__}: {Path(fn).name if fn else scrub_paths(str(error))}"
    return {"status": CHECK_FAILED, "method": method, "error": text,
            "checked_at": _now()}


def _pub_parts(pub):
    """(circuit, shots) of an IBM pub as stored in job.inputs['pubs'] (tuple or SamplerPub)."""
    if hasattr(pub, "circuit"):
        return pub.circuit, getattr(pub, "shots", None)
    return pub[0], (pub[2] if len(pub) > 2 else None)


def check_ibm(submitted_circuits, pubs, shots):
    """submitted_circuits: the ISA circuits we sent (list); pubs: the server's copy
    (job.inputs['pubs'], possibly sliced to this chunk). Never raises."""
    method = "IBM job.inputs['pubs'] vs submitted ISA circuits"
    try:
        exe, exe_shots = zip(*[_pub_parts(p) for p in pubs]) if pubs else ((), ())
        return compare_canonical([canonical_qiskit(c) for c in submitted_circuits],
                                 [canonical_qiskit(c) for c in exe],
                                 shots, list(exe_shots), method)
    except Exception as e:
        return check_failed(method, e)


def check_iqm(submitted_circuits, payload, qubit_name, shots, circuit_slice=None):
    """submitted_circuits: the ISA circuits we sent; payload: raw GET jobs/{id}/payload
    (or {'payload': ...}); qubit_name: index -> 'QB<n>'. Never raises."""
    method = "IQM GET jobs/{id}/payload vs submitted ISA circuits"
    try:
        body     = payload.get("payload", payload)
        circuits = body["circuits"]
        if circuit_slice is not None:
            circuits = circuits[circuit_slice]
        return compare_canonical([canonical_qiskit(c, qubit_name, iqm_native=True) for c in submitted_circuits],
                                 [canonical_iqm(c) for c in circuits],
                                 shots, [body.get("shots")], method)
    except Exception as e:
        return check_failed(method, e)


def check_vlq(submitted_circuits, executed_circuits, method):
    """submitted_circuits: the ISA circuits we sent; executed_circuits: the circuits VLQ's
    server ran (qaas job.get_transpiled_circuits(), same order). Never raises."""
    try:
        return compare_canonical([canonical_qiskit(c) for c in submitted_circuits],
                                 [canonical_qiskit(c) for c in executed_circuits], method=method)
    except Exception as e:
        return check_failed(method, e)


def load_qpy(path):
    from qiskit.qpy import load
    with open(path, "rb") as f:
        return load(f)


# ── Analysis ──────────────────────────────────────────────────────────────────
def _layout_of_circuits(circuits):
    """Physical qubits of a list of transpiled circuits: a tuple if they all agree,
    'mixed' if they don't, None if the transpiler attached no layout."""
    from source.core.qpu.layout_selection import final_physical_qubits
    sets = {tuple(l) if l is not None else None for l in (final_physical_qubits(c) for c in circuits)}
    if len(sets) == 1:
        return next(iter(sets))
    return "mixed"


class _LegacyLayouts:
    """Physical qubits of the run's transpiled QPY files, cached in
    processed_results/circuit_qubit_check_cache.json (keyed by file size + mtime)."""

    def __init__(self, run_path):
        self.tdir  = Path(run_path) / "transpiled_circuits"
        self.cache_path = Path(run_path) / "processed_results" / LEGACY_CACHE_FILENAME
        self.cache = _read_json(self.cache_path) or {}
        self.dirty = False

    def of(self, filename):
        p = self.tdir / filename
        if not p.exists():
            return "missing"
        st  = p.stat()
        key = f"{st.st_size}:{int(st.st_mtime)}"
        hit = self.cache.get(filename)
        if hit and hit.get("key") == key:
            v = hit["layout"]
        else:
            try:
                v = _layout_of_circuits(load_qpy(p))
            except Exception as e:
                v = f"unreadable: {type(e).__name__}"
            self.cache[filename] = {"key": key, "layout": list(v) if isinstance(v, tuple) else v}
            self.dirty = True
            v = self.cache[filename]["layout"]
        return tuple(v) if isinstance(v, list) else v

    def save(self):
        if self.dirty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=1), encoding="utf-8")


def _hc_layout(rec):
    lay = rec.get("layout")
    if not isinstance(lay, dict) or not lay:
        return None
    try:
        return tuple(int(lay[k]) for k in sorted(lay, key=int))      # {"0": 53, "1": 54, ...}
    except (ValueError, TypeError):
        pass
    try:                                                              # old records: {"<Qubit ...index=0>": 53, ...}
        import re
        lay = {k: v for k, v in lay.items() if '"ancilla"' not in k}  # keep the circuit's own qubits
        idx = {k: int(re.search(r"index=(\d+)", k).group(1)) for k in lay}
        return tuple(int(lay[k]) for k in sorted(lay, key=idx.get))
    except (AttributeError, ValueError, TypeError):
        return None



def run_health_check_records(run_path):
    """All health check records of a run: the numbered accompanying ones
    (health_checks/) and the old standalone ones (job_records/health_check*.json).
    Each is returned with a 'label': its number ('3'), or '<suffix> (standalone)' for
    job_records/health_check_<suffix>.json ('0 (standalone)' for health_check_0.json, 'check (standalone)' for health_check.json)."""
    run_path = Path(run_path)
    out = []
    d = run_path / "health_checks"
    if d.is_dir():
        recs = [r for r in map(_read_json, d.glob("health_check_*.json")) if isinstance(r, dict) and "number" in r]
        out += [{**rec, "label": str(rec["number"])} for rec in sorted(recs, key=lambda r: r["number"])]
    jr = run_path / "job_records"
    if jr.is_dir():
        for p in sorted(jr.glob("health_check*.json")):
            rec = _read_json(p)
            if isinstance(rec, dict):
                suffix = p.stem[len("health_check"):].strip("_")
                out.append({**rec, "label": f"{suffix or 'check'} (standalone)"})
    return out


def health_check_chunks(rec):
    """Chunks a health check accompanies: its `chunks` (the submission it precedes or
    follows, as passed by the submit script). Records written before `chunks` existed:
    their jobs' chunks, or, for a check taken before a submission, the first chunk after
    the ones already on disk. None = unknown (old)."""
    if rec.get("chunks"):
        return list(rec["chunks"])
    jobs = [c for j in rec.get("jobs") or [] for c in j]
    if jobs:
        return jobs
    if "prior_chunks" in rec:
        prior = rec.get("prior_chunks") or []
        return [max(prior) + 1 if prior else 0]
    return None


def health_check_in_scope(rec, n_samples):
    """True if a health check belongs to chunks [0, n_samples) (n_samples None = all) —
    or, with a collection of chunk numbers (source/core/records/chunk_scope.py), to the chunks up to
    the last one in it."""
    from source.core.records.chunk_scope import scope_last_chunk
    if n_samples is None:
        return True
    chunks = health_check_chunks(rec)
    return chunks is None or min(chunks) <= scope_last_chunk(n_samples)


def _summarize(entries):
    """entries: list of (id, status, detail, method). Returns the per-kind summary."""
    by = Counter(s for _, s, _, _ in entries)
    bad = [e for e in entries if e[1] in (MISMATCH, CHECK_FAILED)]
    if not entries:
        status = "no_records"
    elif any(s == MISMATCH for _, s, _, _ in entries):
        status = MISMATCH
    elif any(s == CHECK_FAILED for _, s, _, _ in entries):
        status = CHECK_FAILED
    elif all(s == MATCH for _, s, _, _ in entries):
        status = MATCH
    elif all(s == NOT_AVAILABLE for _, s, _, _ in entries):
        status = NOT_AVAILABLE
    else:
        status = "partial"        # some matched, the rest could not be checked
    return {
        "status":            status,
        "n_records":         len(entries),
        "counts":            dict(by),
        "methods":           sorted({m for *_, m in entries if m}),
        "mismatched_ids":    [e[0] for e in entries if e[1] == MISMATCH],
        "check_failed_ids":  [e[0] for e in entries if e[1] == CHECK_FAILED],
        "details":           {str(e[0]): e[2] for e in bad[:MAX_DIFFERENCES] if e[2]},
    }


def _old_records(run_path):
    """(legacy, unconverted) names of completed records that predate circuit_check: legacy =
    services whose old runs stay legacy (job_timing.LEGACY_SERVICES), unconverted = the rest
    (old AQT records), which must be converted to the current format first."""
    from source.core.records.job_timing import service_of_record, LEGACY_SERVICES
    recs = []
    jr = Path(run_path) / "job_records"
    for p in chunk_record_files(jr).values():
        rec = _read_json(p)
        if isinstance(rec, dict):
            recs.append((p.name, rec))
    recs += [(h["label"], h) for h in run_health_check_records(run_path)]
    old  = [(n, r) for n, r in recs if r.get("status") == "complete" and "circuit_check" not in r]
    hint = next((service_of_record(r) for _, r in recs if service_of_record(r)), None)
    legacy = [n for n, r in old if (service_of_record(r, hint) or hint) in LEGACY_SERVICES]
    return legacy, [n for n, _ in old if n not in legacy]


def _require_current_format(run_path):
    """Stop if the run has old AQT records that were not converted; return the number of
    legacy records (old IBM / IQM Resonance / IQM EQE1), which are read as they are."""
    legacy, old = _old_records(run_path)
    if old:
        raise ValueError(f"{run_path}: {len(old)} AQT record(s) in the old format (e.g. {old[0]}), without "
                         f"timing and circuit_check. Convert them to the current format first.")
    return len(legacy)


def summarize_circuit_checks(run_path, n_samples=None):
    """Summary of the executed-vs-submitted checks of a run, chunks and health checks
    separately. Records whose provider can't return the executed circuits (and converted
    old AQT records) are checked for physical-qubit consistency instead (method
    'qubit_consistency'); records of a legacy run the same way (method
    'legacy_qubit_consistency'). Raises if the run has old AQT records not yet converted.
    n_samples: the first n_samples collected chunks (source/core/records/chunk_scope.py), or a
    collection of chunk numbers; None = every chunk."""
    from source.core.records.chunk_scope import chunk_scope, in_scope
    run_path = Path(run_path)
    n_samples = chunk_scope(run_path, n_samples)
    n_legacy = _require_current_format(run_path)
    legacy   = _LegacyLayouts(run_path)
    pq       = _read_json(run_path / "transpiled_circuits" / "physical_qubits.json") or {}
    expected = tuple(pq["physical_qubits"]) if pq.get("physical_qubits") else None

    records = []
    jr = run_path / "job_records"
    for nc, p in chunk_record_files(jr).items():
        if not in_scope(nc, n_samples):
            continue
        rec = _read_json(p)
        if isinstance(rec, dict) and rec.get("status") == "complete":
            records.append((nc, rec))
    hcs = [h for h in run_health_check_records(run_path)
           if h.get("status") == "complete" and health_check_in_scope(h, n_samples)]

    # Reference qubits for the consistency fallback: physical_qubits.json, else the most
    # common layout of the chunk files that need the fallback.
    needs_fallback = [nc for nc, rec in records
                      if (rec.get("circuit_check") or {}).get("status") in (None, NOT_AVAILABLE)]
    chunk_layouts  = {nc: legacy.of(f"chunk_{nc}_transpiled.qpy") for nc in needs_fallback}
    uses_spam = any(rec.get("n_spam", 0) > 0 for _, rec in records)
    spam_layout = legacy.of("spam_transpiled.qpy") if (uses_spam and needs_fallback) else None
    ref = expected
    if ref is None:
        common = Counter(v for v in chunk_layouts.values() if isinstance(v, tuple)).most_common(1)
        ref = common[0][0] if common else None

    def fallback_chunk(nc, rec):
        lay = chunk_layouts.get(nc)
        if lay == "missing":
            return NOT_AVAILABLE, {"reason": "transpiled circuits not on disk"}
        if isinstance(lay, str) and lay.startswith("unreadable:"):
            return CHECK_FAILED, {"reason": f"transpiled circuit file could not be read ({lay})"}
        if not isinstance(lay, tuple) and lay is not None:
            return MISMATCH, {"reason": f"chunk circuits on inconsistent qubits ({lay})"}
        if lay is None:
            return NOT_AVAILABLE, {"reason": "transpiled circuits carry no layout"}
        if ref is not None and lay != ref:
            return MISMATCH, {"reason": f"chunk on qubits {list(lay)}, run on {list(ref)}"}
        if rec.get("n_spam", 0) > 0 and isinstance(spam_layout, tuple) and spam_layout != lay:
            return MISMATCH, {"reason": f"SPAM circuits on qubits {list(spam_layout)}, chunk on {list(lay)}"}
        if rec.get("n_spam", 0) > 0 and spam_layout == "mixed":
            return MISMATCH, {"reason": "SPAM circuits on inconsistent qubits"}
        return MATCH, None

    chunk_entries = []
    for nc, rec in records:
        cc = rec.get("circuit_check") or {}
        st = cc.get("status")
        if st in (MATCH, MISMATCH, CHECK_FAILED):
            detail = cc.get("differences") or cc.get("error")
            chunk_entries.append((nc, st, detail, "executed_vs_submitted"))
        else:
            fs, detail = fallback_chunk(nc, rec)
            method = "legacy_qubit_consistency" if "circuit_check" not in rec else "qubit_consistency"
            chunk_entries.append((nc, fs, detail, method if fs != NOT_AVAILABLE else None))

    hc_entries = []
    for h in hcs:
        cc = h.get("circuit_check") or {}
        st = cc.get("status")
        if st in (MATCH, MISMATCH, CHECK_FAILED):
            hc_entries.append((h["label"], st, cc.get("differences") or cc.get("error"), "executed_vs_submitted"))
            continue
        lay = _hc_layout(h)
        method = "legacy_qubit_consistency" if "circuit_check" not in h else "qubit_consistency"
        if lay is None:
            hc_entries.append((h["label"], NOT_AVAILABLE, {"reason": "no layout recorded"}, None))
        elif ref is not None and lay != ref:
            hc_entries.append((h["label"], MISMATCH, {"reason": f"health check on qubits {list(lay)}, run on {list(ref)}"},
                               method))
        else:
            hc_entries.append((h["label"], MATCH, None, method))

    legacy.save()
    return {
        "chunks":        _summarize(chunk_entries),
        "health_checks": _summarize(hc_entries),
        "reference_qubits": list(ref) if ref is not None else None,
        "legacy_run":       n_legacy > 0,
    }


# ── Warnings for the terminal and the figures ─────────────────────────────────
def _first_and_others(ids, prefix):
    if not ids:
        return None
    return f"{prefix} {ids[0]}" + (" and others" if len(ids) > 1 else "")


def verification_warning(summary):
    """Short warning line for a figure (or None if nothing is wrong). At most one chunk
    and one health check are named: 'chunk 12 and others; health check 15'."""
    if not summary:
        return None
    parts = []
    for kind, word, key in (("mismatch", "Circuit mismatch", "mismatched_ids"),
                            ("check_failed", "Circuit check failed", "check_failed_ids")):
        items = [_first_and_others((summary.get("chunks") or {}).get(key) or [], "chunk"),
                 _first_and_others((summary.get("health_checks") or {}).get(key) or [], "health check")]
        items = [i for i in items if i]
        if items:
            parts.append(f"⚠ {word}: " + "; ".join(items))
    return "   ".join(parts) if parts else None


def print_verification(summary):
    """Terminal report of a circuit-verification summary."""
    if not summary:
        return
    if summary.get("legacy_run"):
        print("  [Circuit check] Legacy run: the executed circuits were not recorded; checked that all "
              "transpiled circuits sit on the same physical qubits instead.")
    for kind, label in (("chunks", "Chunks"), ("health_checks", "Health checks")):
        s = summary.get(kind) or {}
        line = f"  [Circuit check] {label}: {s.get('status')} ({s.get('counts')})"
        if s.get("mismatched_ids"):
            line += f" — mismatched: {s['mismatched_ids'][:20]}" + (" …" if len(s["mismatched_ids"]) > 20 else "")
        if s.get("check_failed_ids"):
            line += f" — check failed: {s['check_failed_ids'][:20]}"
        print(("  ⚠️ " if s.get("status") in (MISMATCH, CHECK_FAILED) else "") + line)


def add_circuit_check_banner(fig, summary):
    """Red warning line at the bottom of a matplotlib figure if the run's executed
    circuits did not match the submitted ones (or couldn't be checked)."""
    text = verification_warning(summary)
    if text:
        fig.text(0.5, -0.02, text, ha="center", va="top", color="red", fontsize=9, fontweight="bold")
    return text


def load_run_verification(run_path, suffix=""):
    """circuit_verification of a run as written into run_data[_spam]{suffix}.json by the
    analysis; computed directly if no run_data file carries it."""
    pr = Path(run_path) / "processed_results"
    for name in (f"run_data_spam{suffix}.json", f"run_data{suffix}.json"):
        d = _read_json(pr / name)
        if isinstance(d, dict) and "circuit_verification" in d:
            return d["circuit_verification"]
    return summarize_circuit_checks(run_path)
