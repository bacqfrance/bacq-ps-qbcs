"""
dummy_estimates.py
The modelled QPU time and cost of a dummy emulator run (data type sim-dummy; the driver is
source/benchmark_scripts/local_simulation_scripts/benchmark_dummy_noisy_emulators.py).

A dummy is a coherent-noise local simulation of a QPU it was not run on. Its definition (fidelity, pricing model and
constants, timing constants, and the source of each) is in definitions_dummy_noisy_emulators.json beside the driver,
which copies the dummy's entry into the run folder as dummy_estimates.json (inputs only, no absolute paths): the
analysis reads that file, so it needs neither the definitions nor the driver, and works on a moved folder.

Time of one circuit execution (shot):  per_shot_overhead + N1q * t_1q + N2q * t_2q + t_meas
from the gate counts of the logical circuits as the run simulated them (no transpilation; native gate sets and
couplings are ignored): rz is free, every other 1-qubit gate counts as N1q, every 2-qubit gate (cx, cz, rzz, ...) as N2q,
and measure / reset as Nm (measurements). Gates are summed: no parallelism between gates. A run's modelled QPU time is
that time summed over every circuit and shot of the chunk jobs (SPAM circuits included) and of the accompanying health checks.
Cost, by pricing["model"]:
  ibm_payg       modelled seconds / 60 * usd_per_qpu_minute (no rounding up to whole seconds)
  braket         tasks * usd_per_task + shots * usd_per_shot: a chunk job is ceil(circuits / circuits_per_task) tasks,
                 a health check one task; shots counted per circuit
  braket_scaled  the same formula; the per-shot price is the project's scaled estimate for a machine Braket does not offer
  hqc            Quantinuum: per circuit (one job each) HQC = hqc_fixed + C / hqc_divisor * (n1q * N1q + n2q * N2q + nm * Nm)
                 with C the shots of that circuit, summed over all circuits (health checks included) * usd_per_hqc
A null (or missing) term of the timing or pricing block is an error, never 0.

The pure functions (modelled_qpu_time(), modelled_cost()) take the jobs as built by run_jobs(); model_run() does it for a run folder.
"""

import json
from pathlib import Path

from source.core.common import PROJECT_ROOT, get_circuit_folder, get_chunk_folder
from source.core.records.chunk_scope import chunk_scope
from source.core.records.run_names import parse_run_name

ESTIMATES_FILENAME = "dummy_estimates.json"
DEFINITIONS_PATH   = (PROJECT_ROOT / "source" / "benchmark_scripts" / "local_simulation_scripts"
                      / "definitions_dummy_noisy_emulators.json")
LOOKUP_DATE        = "2026-10-26"   # the date of the definitions file (each source string carries its own dates)

TIMING_TERMS = ("t_1q", "t_2q", "t_meas", "per_shot_overhead")
PRICING_TERMS = {   # the constants each pricing model needs (all non-null)
    "ibm_payg":      ("usd_per_qpu_minute",),
    "braket":        ("usd_per_task", "usd_per_shot", "circuits_per_task"),
    "braket_scaled": ("usd_per_task", "usd_per_shot", "circuits_per_task"),
    "hqc":           ("hqc_fixed", "weights", "hqc_divisor", "usd_per_hqc"),
}
HQC_WEIGHTS = ("n1q", "n2q", "nm")
SOURCE_FIELDS = ("source_fidelity", "source_pricing", "source_timing")
FREE_1Q_GATES = ("rz",)                        # 1-qubit gates with no duration
IGNORED_OPS   = ("barrier", "delay")           # directives: no duration, no count


# ── Definitions and the per-run file ──────────────────────────────────────────
def load_definitions(path=DEFINITIONS_PATH):
    """The dummies' definitions: {NAME: entry}."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_entry(name, entry, saved=False):
    """ValueError naming the missing or null term if the entry of dummy `name` cannot be modelled; saved=True: the entry is a
    run's dummy_estimates.json (estimates_record()) rather than a definition."""
    noise_keys = ("infidelity_total", "ratio_ct", "noise_type", "lnn") if saved else ("Fidelity", "r_ct", "noise_type", "lnn")
    for key in ("architecture", *noise_keys, "pricing", "timing", *SOURCE_FIELDS):
        if entry.get(key) is None:
            raise ValueError(f"dummy {name}: '{key}' is missing or null in its {ESTIMATES_FILENAME if saved else 'definition'}")
    pricing, timing = entry["pricing"], entry["timing"]
    for term in TIMING_TERMS:
        if timing.get(term) is None:
            raise ValueError(f"dummy {name}: timing term '{term}' is missing or null (a null term is never taken as 0)")
    model = pricing.get("model")
    if model not in PRICING_TERMS:
        raise ValueError(f"dummy {name}: unknown pricing model {model!r} (one of {', '.join(PRICING_TERMS)})")
    for term in PRICING_TERMS[model]:
        if pricing.get(term) is None:
            raise ValueError(f"dummy {name}: pricing term '{term}' of model {model} is missing or null")
    if model == "hqc":
        for w in HQC_WEIGHTS:
            if (pricing["weights"] or {}).get(w) is None:
                raise ValueError(f"dummy {name}: pricing weight '{w}' of model hqc is missing or null")


def estimates_record(name, entry):
    """What the run folder's dummy_estimates.json holds: the dummy's name and architecture, the noise parameters as the run uses them,
    the pricing and timing blocks and the three source fields of the definitions, and the lookup date."""
    validate_entry(name, entry)
    return {
        "name":             name,
        "architecture":     entry["architecture"],
        "infidelity_total": 1 - float(entry["Fidelity"]) / 100,
        "fidelity_percent": float(entry["Fidelity"]),
        "ratio_ct":         float(entry["r_ct"]),
        "noise_type":       str(entry["noise_type"]).lower(),
        "lnn":              str(entry["lnn"]) == "True",
        "pricing":          entry["pricing"],
        "timing":           entry["timing"],
        **{k: entry[k] for k in SOURCE_FIELDS},
        "lookup_date":      LOOKUP_DATE,
    }


def write_dummy_estimates(run_path, name, entry):
    """Write run_path/dummy_estimates.json once (an existing file is left as it is, so a resumed run keeps the inputs
    it started with). Returns the file's path."""
    path = Path(run_path) / ESTIMATES_FILENAME
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(estimates_record(name, entry), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_dummy_estimates(run_path):
    """The run's dummy_estimates.json, validated; FileNotFoundError / ValueError with the reason otherwise."""
    path = Path(run_path) / ESTIMATES_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path.parent.name} has no {ESTIMATES_FILENAME}: a dummy emulator run needs it for its modelled QPU time and cost")
    record = json.loads(path.read_text(encoding="utf-8"))
    validate_entry(record.get("name", path.parent.name), record, saved=True)
    return record


# ── Gate counts ───────────────────────────────────────────────────────────────
def circuit_op_counts(circuit):
    """(N1q, N2q, Nm) of a Qiskit circuit: rz free; any other 1-qubit gate N1q; 2-qubit gates N2q; measure and reset Nm.
    A circuit without classical registers is measured like the pipeline does (measure_all): num_qubits measurements.
    A gate on 3 or more qubits is a ValueError (the logical circuits hold none)."""
    n1 = n2 = nm = 0
    for inst in circuit.data:
        op = inst.operation
        if op.name in IGNORED_OPS:
            continue
        if op.name in ("measure", "reset"):
            nm += 1
        elif op.num_qubits == 1:
            n1 += op.name not in FREE_1Q_GATES
        elif op.num_qubits == 2:
            n2 += 1
        else:
            raise ValueError(f"gate {op.name!r} acts on {op.num_qubits} qubits: the dummy model counts 1- and 2-qubit gates only")
    if not circuit.cregs:
        nm += circuit.num_qubits
    return n1, n2, nm


# ── The models (pure) ─────────────────────────────────────────────────────────
def circuit_time_s(counts, timing):
    """Seconds of one shot of a circuit with counts = (N1q, N2q, Nm): per_shot_overhead + N1q t_1q + N2q t_2q + t_meas."""
    n1, n2, _ = counts
    return timing["per_shot_overhead"] + n1 * timing["t_1q"] + n2 * timing["t_2q"] + timing["t_meas"]


def modelled_qpu_time(jobs, timing):
    """(total seconds, seconds of the chunk jobs, seconds of the health checks) of jobs
    [{"kind": "chunk" | "health_check", "shots": int, "circuits": [(N1q, N2q, Nm), ...]}]: every circuit runs `shots` times."""
    parts = {"chunk": 0.0, "health_check": 0.0}
    for job in jobs:
        parts[job["kind"]] += sum(job["shots"] * circuit_time_s(c, timing) for c in job["circuits"])
    return parts["chunk"] + parts["health_check"], parts["chunk"], parts["health_check"]


def modelled_cost(jobs, pricing, qpu_time_s):
    """The modelled cost in USD and what it was counted from: {"usd", "model", ...} with n_tasks (braket models) or hqc
    (hqc model); ibm_payg uses qpu_time_s."""
    model = pricing["model"]
    n_circuits = sum(len(j["circuits"]) for j in jobs)
    n_shots    = sum(len(j["circuits"]) * j["shots"] for j in jobs)
    out = {"model": model, "n_jobs": len(jobs), "n_circuits": n_circuits, "n_shots": n_shots}
    if model == "ibm_payg":
        out["usd"] = qpu_time_s / 60 * pricing["usd_per_qpu_minute"]
    elif model in ("braket", "braket_scaled"):
        per_task = int(pricing["circuits_per_task"])
        out["n_tasks"] = sum(-(-len(j["circuits"]) // per_task) for j in jobs)
        out["usd"] = out["n_tasks"] * pricing["usd_per_task"] + n_shots * pricing["usd_per_shot"]
    elif model == "hqc":
        w = pricing["weights"]
        hqc = sum(pricing["hqc_fixed"] + j["shots"] / pricing["hqc_divisor"] * (w["n1q"] * c[0] + w["n2q"] * c[1] + w["nm"] * c[2])
                  for j in jobs for c in j["circuits"])
        out["hqc"] = hqc
        out["usd"] = hqc * pricing["usd_per_hqc"]
    else:
        raise ValueError(f"unknown pricing model {model!r}")
    return out


# ── A run's jobs ──────────────────────────────────────────────────────────────
def _cache_top(estimates, info):
    """The circuit cache folder of the run (the one build() filled), as a Path."""
    config = {"lnn_architecture": info.lnn, "T_list": info.list_times, "dt": info.dt,
              "coherent_noise": {"noise_type": estimates["noise_type"], "infidelity_total": estimates["infidelity_total"],
                                 "ratio_ct": estimates["ratio_ct"]}}
    top = Path(get_circuit_folder(config, top_folder=True))
    return top if top.is_absolute() else PROJECT_ROOT / top


def _qpy_counts(path):
    from source.core.records.circuit_check import load_qpy
    if not Path(path).exists():
        raise FileNotFoundError(f"{path}: the circuit cache file the dummy model counts gates from is missing")
    return [circuit_op_counts(c) for c in load_qpy(path)]


def health_check_counts():
    """(N1q, N2q, Nm) of the health check circuit (3 x, 4 cx, 3 measurements)."""
    import contextlib, io
    from source.core.health_check import build_qpu_health_check_circuit
    with contextlib.redirect_stdout(io.StringIO()):
        return circuit_op_counts(build_qpu_health_check_circuit())


def run_jobs(run_path, info, estimates, n_samples=None):
    """The jobs of a run folder for the models: one per complete chunk record (its SPAM circuits first, then the
    chunk's circuits, all at the record's shots) and one per complete accompanying health check, within the first
    n_samples chunks (None: all), with gate counts loaded from the circuit cache."""
    from source.core.records.circuit_check import run_health_check_records, health_check_in_scope
    from source.core.records.job_records import chunk_record_files
    run_path = Path(run_path)
    nchunks  = chunk_scope(run_path, n_samples)
    scope    = None if nchunks is None else set(nchunks)
    top      = _cache_top(estimates, info)
    kind     = "qiskit_v13" if info.circuits_qiskit_v13_qpy else "qiskit"
    chunk_dir = Path(get_chunk_folder(top, kind))
    spam_counts = None
    jobs = []
    for nc, p in sorted(chunk_record_files(run_path / "job_records").items()):
        if scope is not None and nc not in scope:
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("status") != "complete":
            continue
        circuits = []
        if int(rec.get("n_spam") or 0):
            if spam_counts is None:
                spam_counts = _qpy_counts(chunk_dir / f"spam_{kind}.qpy")
            circuits += spam_counts[:int(rec["n_spam"])]
        circuits += _qpy_counts(chunk_dir / f"chunk_{nc}_{kind}.qpy")[:int(rec["n_circuits"])]
        jobs.append({"kind": "chunk", "shots": int(rec["nshots"]), "circuits": circuits})
    hc = None
    for h in run_health_check_records(run_path):
        if h.get("status") != "complete" or not health_check_in_scope(h, nchunks):
            continue
        hc = hc or health_check_counts()
        jobs.append({"kind": "health_check", "shots": int(h["nshots"]), "circuits": [hc]})
    return jobs


def model_run(run_folder, n_samples=None):
    """The modelled QPU time and cost of the dummy run datafiles/<run_folder> (the first n_samples chunks, None: all):
    {"estimates": the run's dummy_estimates.json, "qpu_time_s", "qpu_time_chunks_s", "qpu_time_health_checks_s",
    "cost": modelled_cost(), "gates": summed gate counts over all executed circuits (once per circuit, not per shot)}."""
    from source.core.records.run_info import load_run_info
    if not parse_run_name(run_folder).is_dummy:
        raise ValueError(f"{run_folder} is not a dummy emulator run (data type sim-dummy)")
    run_path  = PROJECT_ROOT / "datafiles" / run_folder
    estimates = load_dummy_estimates(run_path)
    info      = load_run_info(run_folder)
    jobs      = run_jobs(run_path, info, estimates, n_samples)
    total, chunks_s, hcs_s = modelled_qpu_time(jobs, estimates["timing"])
    return {"estimates": estimates, "qpu_time_s": total, "qpu_time_chunks_s": chunks_s, "qpu_time_health_checks_s": hcs_s,
            "cost": modelled_cost(jobs, estimates["pricing"], total),
            "gates": {k: sum(c[i] for j in jobs for c in j["circuits"]) for i, k in enumerate(("n1q", "n2q", "nm"))}}


# ── The driver's skip / resume rule ───────────────────────────────────────────
def dummy_run_folder(undated, root="datafiles"):
    """(run folder, finished) for a dummy's undated run name: the latest run of those settings and True when it is
    finished (the driver skips it; dated_run_folder() alone would open a new #N run), else the folder to work in
    (the latest unfinished run, resumed, or a new one dated today) and False."""
    from source.core.records.run_names import dated_run_folders, dated_run_folder, run_finished
    existing = dated_run_folders(undated, root)
    if existing and run_finished(existing[-1], root):
        return existing[-1], True
    return dated_run_folder(undated, root), False
