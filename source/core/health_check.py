"""
health_check.py
The QPU health check (USER_MANUAL §7): build its circuit, submit it on any service, collect it, and the
checks the scripts run around a benchmark (require_functional_health_check() before submitting,
submit_post_health_check() after). A standalone check's record is <run>/job_records/health_check[_<suffix>].json; the checks that accompany
a run live in <run>/health_checks/ (records/health_check_records.py).
myQLM is never needed here.
"""

import json
import os
import sys
import time
from pathlib import Path

from source.core.circuits.circuit_drawings import export_circuit_drawings
from source.core.common import exit_with_error, PROJECT_ROOT, rel_path, scrub_paths, section
from source.core.qpu import iqm_vlq as VLQ
from source.core.qpu.aqt import _aqt_run, AQTTaskTimeout
from source.core.qpu.layout_selection import check_forced_layout, transpile_like_submission
from source.core.qpu.qpu_snapshot import schedule_for_simulation
from source.core.qpu.providers import ACCOMPANYING_HEALTH_CHECK_NSHOTS, job_state_of_status
from source.core.qpu.remote_jobs import _is_transient_network_error, _remote_job_state, _warn_circuit_check, HealthCheckSubmitError, print_job_line, SUBMIT_RETRY_BACKOFF_S
from source.core.records import circuit_check as CC
from source.core.records.health_check_records import create_health_check, format_chunk_range, health_check_path, HEALTH_CHECK_THRESHOLD, health_check_p011, health_check_verdict, health_dir, load_health_checks, now_utc_iso, save_health_check
from source.core.records.job_records import write_job_record
from source.core.records.job_timing import aqt_timing, SYNCHRONOUS_SERVICES, eqe_collect_timing, iso_utc, ibm_collect_timing, iqm_raw_job, iqm_resonance_collect_timing, merge_timing, submit_timing
from source.core.records.timestamps import timestamp, TIMESTAMP_FILENAME
from source.core.records.run_names import IBM_CLOUD, IQM_RESONANCE, QAAS_IT4I, MQSS_LRZ, PCSS


def build_qpu_health_check_circuit(reps=1, save_pdf=False):
    """
    Build a 3-qubit QPU health check circuit. Standalone — no qbcs instance needed.

    Structure:
      - X(0), X(1), X(2)
      - For reps repetitions of: CX(0,1), CX(1,2), CX(2,1), CX(1,0)
      - measure all

    At zero noise and reps=1 (the value every script uses): P(011) = 1.0 (bit order q0 q1 q2).
    Pass criterion: P(011) > HEALTH_CHECK_THRESHOLD (health_check_verdict()).
    Total gates (reps=1): 3 X + 4 CNOTs.

    :param reps:     int — number of CNOT block repetitions
    :param save_pdf: bool — save circuit diagram to ./health_check_circuit.pdf
    :returns:        QuantumCircuit
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    qr = QuantumRegister(3, 'q')
    cr = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(qr, cr)

    # ── X block ───────────────────────────────────────────────────────
    for q in [0, 1, 2]:
        qc.x(q)
    qc.barrier()

    # ── CNOT block ────────────────────────────────────────────────────
    for _ in range(reps):
        qc.cx(0, 1); qc.barrier()
        qc.cx(1, 2); qc.barrier()
        qc.cx(2, 1); qc.barrier()
        qc.cx(1, 0); qc.barrier()
    # ── Measure ───────────────────────────────────────────────────────
    qc.measure(qr, cr)

    print(f"  [Health check] Circuit built: {qc.num_qubits} qubits, "
          f"{qc.count_ops().get('cx', 0)} CX gates, "
          f"{qc.count_ops().get('x', 0)} X gates")

    # ── Save circuit diagram ──────────────────────────────────────────
    if save_pdf:
        from qiskit.visualization import circuit_drawer
        fig = circuit_drawer(qc, output="mpl")
        fig.savefig("./health_check_circuit.pdf", bbox_inches="tight")
        print(f"  [Health check] Circuit diagram saved → ./health_check_circuit.pdf")

    return qc

def submit_health_check(run_folder, backend, sampler=None,
                        nshots=ACCOMPANYING_HEALTH_CHECK_NSHOTS, service=None, fake_registry=None,
                        service_name=IBM_CLOUD, reps=1, qubit_set=None, record_suffix=None,
                        accompanying=False, after_submission=False, layout_backend=None, chunks=None,
                        return_number=False, backend_name=None, scheduling_simulation=True):
    """
    Standalone function — no bacq_ps_qbcs instance needed.
    Works for both IBM, IQM and AQT backends (and the local Aer simulator).

    accompanying=False (default): standalone health check — record in
        job_records/health_check[_<record_suffix>].json.
    accompanying=True: health check that accompanies a run — record in
        health_checks/health_check_<N>.json with an auto-incremented single counter
        (record_suffix must be None) and `chunks` required: the chunks the submission
        it accompanies submits (the submit script's chunks_to_submit() list, which may
        have gaps), recorded as `chunks`. Call it before the submission
        (after_submission=False) and after it (after_submission=True) with the same list.
        Nothing is derived from the chunk records on disk.

    :param run_folder:   str — subfolder under datafiles/
    :param backend:      backend object — for transpilation + IQM execution
    :param sampler:      sampler object — for IBM execution (SamplerV2)
    :param nshots:       int — number of shots
    :param service:      QiskitRuntimeService — IBM only, None for IQM/fake
    :param fake_registry:dict — to store job for same-session collection
    :param service_name: str — "IBM_Cloud" | "MQSS_LRZ" | "PCSS" | "IQM_Resonance" | "QAAS_IT4I" | "local"
    :param reps:         int - no. of repetitions of the 4-CNOT circuit block
    :param qubit_set:    list | None — physical qubits for the circuit's qubits 0..2. None (default):
                         the run's own physical qubits if submit() has already recorded them
                         (<run>/transpiled_circuits/physical_qubits.json), else qiskit's choice.
    :param chunks:       list[int] — accompanying=True only (required, non-empty): the chunks of the
                         submission this check precedes or follows
    :param layout_backend: service_name="local" only — the backend to transpile onto, as in
                         submit(layout_backend=...) (e.g. a calibration-snapshot backend, so the
                         health check runs on the run's qubits under the same noise model).
                         None: transpile for a generic AerSimulator (noiseless local simulations).
                         Every health check is transpiled, whatever the service.
    :param return_number: bool — accompanying=True only: also return the health check's own number,
                         so a caller doesn't have to infer it from max(load_health_checks(...)).
    :param backend_name: str | None — the backend's user-facing name (the script's BACKEND_NAME), used in
                         messages only; None: the transpile target's name (layout_backend's for
                         local runs, else the QPU's). The name stored in the record is always the backend object's.
    :param scheduling_simulation: bool — a noisy local simulation's scheduling_simulation= choice: False transpiles the
                         circuit unscheduled (no idle-qubit relaxation), as the run's chunks are. Real QPUs ignore it.
    :returns:            Path — exp_results_folder_path (or (Path, number) if return_number=True)

    For the asynchronous services (IBM_Cloud, MQSS_LRZ, IQM_Resonance, QAAS_IT4I), a submission failure
    with accompanying=True is caught: a "failed" record (job_id=None, status="failed", a
    `failure` message, a timing block with the transpile and submit time) is written under its own health-check number, and HealthCheckSubmitError
    is raised (carrying that number). This keeps every attempt on disk, including failed ones.
    AQT_PIAST-Q with accompanying=True writes the same kind of "failed" record when a circuit's task
    times out (AQTTaskTimeout; failure_stage="execution", the wait as queue_s, the transpile time, no
    QPU time) and re-raises AQTTaskTimeout. Every other failure of the synchronous services (AQT, local)
    and the accompanying=False (standalone) path propagates, and nothing is written.
    """
    exp_results_folder_path = Path(os.path.join(PROJECT_ROOT, "datafiles", run_folder))
    if accompanying:
        assert record_suffix is None, "record_suffix must be None for accompanying health checks (auto-numbered)."
        assert chunks, "an accompanying health check needs chunks=[...] — the chunks of the submission it accompanies."
        chunks = sorted(set(int(c) for c in chunks))
        record_path = None
        health_dir(exp_results_folder_path).mkdir(parents=True, exist_ok=True)
    else:
        fname       = "health_check.json" if record_suffix is None else f"health_check_{record_suffix}.json"
        record_path = exp_results_folder_path / "job_records" / fname
        record_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Idempotency ───────────────────────────────────────────────────────
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            print(f"  [Health check] Already submitted (status={record['status']}) — skipping.")
            return exp_results_folder_path

    # ── Resolve backend ───────────────────────────────────────────────────
    if service_name == IBM_CLOUD:
        real_backend = service.backend(backend) if isinstance(backend, str) else backend
    elif service_name == MQSS_LRZ:
        real_backend = backend
    elif service_name == PCSS:
        real_backend = backend
    elif service_name in (IQM_RESONANCE, QAAS_IT4I):
        real_backend = backend
    elif service_name == "local":
        real_backend = backend          # AerSamplerV2 (noiseless or noisy) — synchronous

    # --- Build the circuit -----------------------------------
    circuit = build_qpu_health_check_circuit(reps=reps)

    # ── Physical qubits: the run's own, once submit() has selected them ────
    # (transpiled_circuits/physical_qubits.json, see bacq_ps_qbcs.resolve_physical_qubits).
    # An explicit qubit_set still wins, with a warning if it differs from the run's.
    layout_record = exp_results_folder_path / "transpiled_circuits" / "physical_qubits.json"
    run_qubits    = json.loads(layout_record.read_text(encoding="utf-8"))["physical_qubits"] if layout_record.exists() else None
    pinnable = service_name != "local" or layout_backend is not None   # a device to pin qubits on
    if qubit_set is None and run_qubits is not None and pinnable:
        qubit_set = run_qubits
        print(f"  [Health check] Using the run's physical qubits {qubit_set}.")
    elif qubit_set is not None and run_qubits is not None and list(qubit_set) != run_qubits:
        print(f"  ⚠️  [Health check] qubit_set={list(qubit_set)} differs from the run's physical qubits {run_qubits}.")

    # ── Transpile (always — same treatment as the run's chunks) ───────────
    from source.core.qpu.layout_selection import final_physical_qubits
    if service_name in (IBM_CLOUD, MQSS_LRZ, PCSS, IQM_RESONANCE, QAAS_IT4I):
        transpile_target = real_backend
    elif service_name == "local":
        if layout_backend is not None:
            transpile_target = layout_backend
        else:
            from qiskit_aer import AerSimulator
            transpile_target = AerSimulator()   # generic, all-to-all — no device qubits to pin
            qubit_set        = None
    else:
        raise ValueError(f"Unknown service_name: {service_name!r}. Must be 'IBM_Cloud', 'MQSS_LRZ', 'PCSS', "
                         f"'IQM_Resonance', 'QAAS_IT4I' or 'local'.")
    # An explicit qubit_set must exist on the QPU and be coupled as the circuit needs: say so up front, in one line,
    # instead of failing after routing. The check reads only the target's qubits and 2q qargs, whatever the
    # provider. IQM VLQ is covered too: its star architecture (every 2q gate goes through the computational
    # resonator) is exposed by the qaas backend as an effective all-to-all CZ target over the 24 circuit qubits
    # (no resonator qubit, no MOVE: a complete graph of 24*23 directed pairs), so every valid set
    # passes; the VLQ-specific layout rule stays in source/core/qpu/iqm_vlq.py: choose_vlq_layout().
    if qubit_set is not None:
        try:
            check_forced_layout(circuit, getattr(transpile_target, "target", transpile_target), qubit_set)
        except ValueError as e:
            exit_with_error(f"health check not submitted: qubit_set={list(qubit_set)} is not usable on "
                            f"{backend_name or getattr(transpile_target, 'name', None) or service_name}: {e}")
    # Every health check of a run is the same circuit on the same qubits: transpile it once,
    # save it (transpiled_circuits/health_check_transpiled.qpy) and submit that saved copy.
    isa, isa_file, transpile_s = _health_check_isa(exp_results_folder_path, circuit, transpile_target,
                                                   qubit_set, reps, scheduling_simulation)
    if qubit_set is not None:
        assert final_physical_qubits(isa) == list(qubit_set), (
            f"health check transpiled onto {final_physical_qubits(isa)}, expected {list(qubit_set)}"
        )
    export_circuit_drawings(exp_results_folder_path, [("health_check", circuit, isa)])

    # final_layout is only set when routing permuted qubits; final_physical_qubits covers both cases
    layout      = final_physical_qubits(isa)
    layout_dict = {str(v): p for v, p in enumerate(layout)} if layout else "N/A"

    print(f"  [Health check] ISA circuit: depth={isa.depth()}, "
          f"gates={dict(isa.count_ops())}, layout={layout}")

    assert len(isa.cregs) > 0, "ISA circuit has no classical registers!"

    # ── Submit ────────────────────────────────────────────────────────────
    submitted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    submitted_precise = now_utc_iso(precise=True)   # accompanying records: microsecond precision (orders them against chunk times)
    synchronous  = service_name in SYNCHRONOUS_SERVICES
    submit_t0    = time.perf_counter()
    asynchronous_service = service_name in (IBM_CLOUD, MQSS_LRZ, IQM_RESONANCE, QAAS_IT4I)
    submit_exc   = None
    if not synchronous:
        try:
            if service_name == IBM_CLOUD:
                assert sampler is not None, "sampler required for IBM submission."
                job = sampler.run([isa], shots=nshots)
            elif service_name == MQSS_LRZ:
                job = real_backend.run([isa], shots=nshots, no_modify=True)
            elif service_name == IQM_RESONANCE:
                job = real_backend.run(isa, shots=nshots)     # async — returns immediately
            elif service_name == QAAS_IT4I:
                job = VLQ.run(real_backend, [isa], "health check", shots=nshots, walltime_limit=VLQ.WALLTIME_LIMIT_S)   # async (HEAppE job)
        except Exception as e:
            submit_exc = e
    else:
        run        = _aqt_run if service_name == PCSS else (lambda b, c, s: b.run(c, shots=s))
        try:
            result = run(real_backend, [isa], nshots).result()
        except AQTTaskTimeout as e:
            if accompanying:
                # A timeout normally means the QPU is unreachable: the wait is queue time, no QPU time.
                number = create_health_check(exp_results_folder_path, {
                    "backend_name":      getattr(real_backend, 'name', str(real_backend)),
                    "layout":            layout_dict,
                    "job_id":            None,
                    "submitted_at":      submitted_precise,
                    "upload_end_at":     None,
                    "nshots":            nshots,
                    "status":            "failed",
                    "result_path":       None,
                    "run_folder":        run_folder,
                    "service_name":      service_name,
                    "isa_circuit_file":  isa_file,
                    "timing":            {**submit_timing(transpile_s=transpile_s, transpile_reused=transpile_s is None,
                                                          submit_s=0.0),
                                          "qpu_s": None, "qpu_s_source": "unavailable",
                                          "queue_s": time.perf_counter() - submit_t0,
                                          "queue_s_source": "wall time until the task timeout"},
                    "scheduled_at":      None,
                    "finished_at":       None,
                    "collected_at":      None,
                    "counts":            None,
                    "P011":              None,
                    "verdict":           None,
                    "threshold":         HEALTH_CHECK_THRESHOLD,
                    "after_submission":  bool(after_submission),
                    "chunks":            chunks,
                    "failure":           scrub_paths(f"execution failed: {e!r}"),
                    "failure_stage":     "execution",
                })
                print(f"  ⚠️  Health check #{number} was not executed on {backend_name or service_name}: {e}")
            raise
        sync_wall  = time.perf_counter() - submit_t0
        if service_name == PCSS:
            counts_raw = result.get_counts()
            counts_raw = counts_raw[0] if isinstance(counts_raw, list) else counts_raw
        else:                                   # Aer SamplerV2 PrimitiveResult
            rdict      = result[0].data
            counts_raw = (rdict.c if hasattr(rdict, "c") else rdict.meas).get_counts()
        counts     = {k[::-1]: v for k, v in counts_raw.items()}
        status, p_011 = print_health_verdict(counts)

    if submit_exc is not None:
        # Asynchronous services, accompanying=True: keep every attempt on disk, including failed
        # ones, as its own numbered health check — then let the caller know via HealthCheckSubmitError.
        # Synchronous services (AQT/local) and standalone (accompanying=False) checks are unchanged:
        # nothing is written and the exception just propagates.
        if accompanying and asynchronous_service:
            fail_record = {
                "backend_name":      getattr(real_backend, 'name', str(real_backend)),
                "layout":            layout_dict,
                "job_id":            None,
                "submitted_at":      submitted_precise,
                "upload_end_at":     None,
                "nshots":            nshots,
                "status":            "failed",
                "result_path":       None,
                "run_folder":        run_folder,
                "service_name":      service_name,
                "isa_circuit_file":  isa_file,
                "timing":            submit_timing(transpile_s=transpile_s, transpile_reused=transpile_s is None,
                                                   submit_s=time.perf_counter() - submit_t0),
                "scheduled_at":      None,
                "finished_at":       None,
                "collected_at":      None,
                "counts":            None,
                "P011":              None,
                "verdict":           None,
                "threshold":         HEALTH_CHECK_THRESHOLD,
                "after_submission":  bool(after_submission),
                "chunks":            chunks,
                "failure":           scrub_paths(f"submission failed: {submit_exc!r}"),
            }
            number  = create_health_check(exp_results_folder_path, fail_record)
            ts_file = exp_results_folder_path / TIMESTAMP_FILENAME
            timestamp(label=f"health_check_{number}_submit_failed", timestamp_file=ts_file)
            print(f"  ⚠️  Health check #{number} could not be submitted on {backend_name or service_name}: {submit_exc}")
            raise HealthCheckSubmitError(number, submit_exc) from submit_exc
        raise submit_exc

    job_id = job.job_id() if not synchronous else "N/A"
    timing = submit_timing(transpile_s=transpile_s, transpile_reused=transpile_s is None,
                           submit_s=None if synchronous else time.perf_counter() - submit_t0)
    if service_name == PCSS:                  # synchronous: submit 0, the rest is queue + QPU
        timing.update(aqt_timing([sync_wall], [1], nshots))
    elif service_name == "local":                      # simulation: the run itself is CPU time
        timing["submit_s"] = sync_wall

    # ── Save job record ───────────────────────────────────────────────────
    record = {
        "backend_name": getattr(real_backend, 'name', None) or type(real_backend).__name__ if service_name == "local"
                        else getattr(real_backend, 'name', str(real_backend)),
        "layout":       layout_dict,
        "job_id":       job_id,
        "submitted_at": submitted_at,
        "upload_end_at": "N/A" if synchronous else now_utc_iso(precise=True),   # synchronous: no separate upload
        "nshots":       nshots,
        "status":       "submitted" if not synchronous else "complete",
        "result_path":  None,
        "run_folder":   run_folder,
        "service_name": service_name,
        "isa_circuit_file": isa_file,
        "timing":       timing,
    }
    if synchronous:
        record["circuit_check"] = CC.not_available(
            "AQT does not return the executed circuits" if service_name == PCSS
            else "local simulator: there is no server copy of the circuits")
        record["P011"]    = p_011
        record["verdict"] = status
        record["counts"]  = counts
        record["status"] = "complete"
        record["finished_at"] =  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if accompanying:
        record.update({
            "scheduled_at": None,
            "finished_at":  now_utc_iso(precise=True) if synchronous else None,
            "submitted_at": submitted_precise,
            "collected_at": now_utc_iso(precise=True) if synchronous else None,
            "counts":       counts if synchronous else None,
            "P011":         p_011 if synchronous else None,
            "verdict":      status if synchronous else None,
            "threshold":    HEALTH_CHECK_THRESHOLD,
            "after_submission": bool(after_submission),
            "chunks":       chunks,
        })
        number = create_health_check(exp_results_folder_path, record)
        print(f"  [Health check] #{number} submitted on {backend_name or record['backend_name']} — job_id: {job_id}; "
              f"{'after' if after_submission else 'before'} chunks {format_chunk_range(chunks)} ({len(chunks)})")
    else:
        number = None
        write_job_record(record_path, record)
        print(f"  [Health check] Submitted on {backend_name or record['backend_name']} — job_id: {job_id}")
        print(f"  [Health check] Record saved → {rel_path(record_path)}")

    # ── Store in fake registry ────────────────────────────────────────────
    if service is None and fake_registry is not None and not synchronous:
        fake_registry[job_id] = job

    if return_number:
        return exp_results_folder_path, number
    return exp_results_folder_path

def _health_check_isa(run_path, circuit, target, qubit_set, reps, scheduling_simulation=True):
    """The run's transpiled health check circuit: loaded from transpiled_circuits/ if it was
    already transpiled for this backend, qubits, reps and scheduling_simulation, else transpiled (timed) and saved.
    Returns (isa, file name, transpile_s — None when the saved copy was reused).

    transpiled_circuits/health_check_transpiled.json lists every saved variant (normally one:
    a run stays on one backend and one set of qubits)."""
    from qiskit.qpy import dump, load

    tdir      = Path(run_path) / "transpiled_circuits"
    meta_path = tdir / "health_check_transpiled.json"
    key       = {"reps": reps, "physical_qubits": list(qubit_set) if qubit_set is not None else None,
                 "backend": getattr(target, "name", None) or type(target).__name__}
    meta      = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"variants": []}
    for v in meta["variants"]:
        if all(v.get(k) == key[k] for k in key) and v.get("scheduled", True) == scheduling_simulation and (tdir / v["file"]).exists():
            with open(tdir / v["file"], "rb") as f:
                isa = load(f)[0]
            print(f"  [Health check] Using the saved transpiled circuit {v['file']}.")
            return isa, v["file"], None

    t0  = time.perf_counter()
    isa = schedule_for_simulation(transpile_like_submission(circuit, target, qubit_set), target, scheduling_simulation)
    transpile_s = time.perf_counter() - t0
    n     = len(meta["variants"])
    fname = "health_check_transpiled.qpy" if n == 0 else f"health_check_transpiled_{n}.qpy"
    tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / fname, "wb") as f:
        dump([isa], f)
    meta["variants"].append({**key, **({} if scheduling_simulation else {"scheduled": False}), "file": fname, "transpile_s": transpile_s,
                             "transpiled_at": now_utc_iso(precise=True)})
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  [Health check] Transpiled and saved → {rel_path(tdir / fname)}")
    return isa, fname, transpile_s


def _check_executed_health_check(run_path, record, service_name, job=None, backend=None):
    """circuit_check for a health check job (see source/core/records/circuit_check.py). Never raises."""
    fname = record.get("isa_circuit_file")
    if service_name in (MQSS_LRZ, PCSS):
        return CC.not_available(f"{service_name} does not return the executed circuits")
    if fname is None:
        return CC.not_available("the submitted circuit was not saved (record predates health_check_transpiled.qpy)")
    try:
        submitted = CC.load_qpy(Path(run_path) / "transpiled_circuits" / fname)
        if service_name == IBM_CLOUD:
            return CC.check_ibm(submitted, job.inputs["pubs"], record["nshots"])
        if service_name == IQM_RESONANCE:
            payload = iqm_raw_job(backend, record["job_id"], "/payload")
            return CC.check_iqm(submitted, payload, backend.index_to_qubit_name, record["nshots"])
        if service_name == QAAS_IT4I:
            return CC.check_vlq(submitted, VLQ.transpiled_circuits(job, record["job_id"]),
                                "VLQ job.get_transpiled_circuits() vs submitted ISA circuit")
    except Exception as e:
        return CC.check_failed(f"{service_name} health check", e)
    return CC.not_available("simulated backend: there is no server copy of the circuit")


def _mark_collect_failed(run_path, number, err):
    """Record a failed collection ATTEMPT on an existing (still 'submitted') health check,
    without changing its status — the job may still finish and be collected later by
    qbcs_real_collect_*.py's number=None path. Purely additive, so every
    check stays recorded."""
    rec_path = health_check_path(run_path, number)
    try:
        record = json.loads(rec_path.read_text(encoding="utf-8"))
        record["last_collect_error"] = scrub_paths(repr(err))[:500]
        save_health_check(run_path, record)
    except Exception as save_exc:
        print(f"  ⚠️  Health check #{number}: could not record the failed collection attempt: {save_exc}")
    ts_file = Path(run_path) / TIMESTAMP_FILENAME
    timestamp(label=f"health_check_{number}_collect_failed", timestamp_file=ts_file)


def require_functional_health_check(run_folder, backend, service_name, collect_service,
                                    sampler=None, service=None, nshots=ACCOMPANYING_HEALTH_CHECK_NSHOTS, reps=1, chunks=None,
                                    after_submission=False, max_attempts=None, backoff_s=None, backend_name=None):
    """Blocking PRE/POST health check with bounded retry on transient network errors, for the
    non-AQT real-QPU submit scripts. Each attempt (submit + collect) is its own numbered health
    check on disk — nothing is nested or overwritten. Exits the process (never returns) unless
    the check comes back functional:
      - degraded QPU:                         EXIT_HC_DEGRADED     (no retry — a real signal)
      - submitted but never collected:        EXIT_HC_NOT_COLLECTED
      - could not even be submitted:          EXIT_HC_NOT_SUBMITTED
    No chunk is ever submitted before this returns a functional record.
    """
    from source.core.records.health_check_records import EXIT_HC_DEGRADED, EXIT_HC_NOT_COLLECTED, EXIT_HC_NOT_SUBMITTED
    run_path = Path(os.path.join(PROJECT_ROOT, "datafiles", run_folder))
    attempts = []   # (number-or-None, outcome-str)
    last_exc = None
    backoff_s = SUBMIT_RETRY_BACKOFF_S if backoff_s is None else backoff_s   # read at call time
    max_attempts = len(backoff_s) if max_attempts is None else max_attempts   # 3 tries: waits 30 s and 60 s only
    for attempt in range(1, max_attempts + 1):
        try:
            _, number = submit_health_check(
                run_folder=run_folder, backend=backend, sampler=sampler, service=service, nshots=nshots,
                service_name=service_name, reps=reps, accompanying=True,
                after_submission=after_submission, chunks=chunks, return_number=True,
                backend_name=backend_name,
            )
        except HealthCheckSubmitError as e:
            attempts.append((e.number, "not submitted"))
            last_exc = e.cause
            if _is_transient_network_error(e.cause) and attempt < max_attempts:
                wait = backoff_s[min(attempt - 1, len(backoff_s) - 1)]
                print(f"  ⚠️  Health check submission attempt {attempt}/{max_attempts} failed (network: {e.cause}) "
                      f"— retrying in {wait}s.")
                time.sleep(wait)
                continue
            break

        try:
            record = _collect_health_check_one(run_path, health_check_path(run_path, number),
                                               collect_service, None, service_name, accompanying=True, backend_name=backend_name)
        except Exception as e:
            _mark_collect_failed(run_path, number, e)
            attempts.append((number, "not collected"))
            last_exc = e
            if _is_transient_network_error(e) and attempt < max_attempts:
                wait = backoff_s[min(attempt - 1, len(backoff_s) - 1)]
                print(f"  ⚠️  Health check #{number} collection attempt {attempt}/{max_attempts} failed "
                      f"(network: {e}) — retrying in {wait}s.")
                time.sleep(wait)
                continue
            break

        if record["verdict"] == "functional":
            print(f"  ✅ QPU functional — check #{number}, P(011)={record['P011']:.3f} — proceeding.")
            return record

        print(f"  ❌ QPU degraded — check #{number}, P(011)={record['P011']:.3f} — aborting!", file=sys.stderr)
        sys.exit(EXIT_HC_DEGRADED)

    summary = ", ".join(f"#{n} {o}" if n is not None else o for n, o in attempts)
    exit_code = EXIT_HC_NOT_COLLECTED if attempts and attempts[-1][1] == "not collected" else EXIT_HC_NOT_SUBMITTED
    print(f"  ❌ QPU health UNKNOWN after {len(attempts)} attempt(s): {summary} (last error: {last_exc}).\n"
          f"     No chunks were submitted. Rerun this script (safe) or, for an uncollected check, run "
          f"qbcs_real_collect_*.py later to record it.", file=sys.stderr)
    sys.exit(exit_code)


def submit_post_health_check(run_folder, backend, service_name, sampler=None, service=None, nshots=ACCOMPANYING_HEALTH_CHECK_NSHOTS, reps=1,
                             chunks=None, max_attempts=None, backoff_s=None, backend_name=None):
    """POST-submission health check for the non-AQT submit scripts: retries only the submission
    (collection happens later, in qbcs_real_collect_*.py, which already handles a collection
    failure there). Never raises or exits — the chunks are already on the QPU and the script must
    still write submit.log. Returns (ok: bool, attempted_numbers: list)."""
    numbers = []
    backoff_s = SUBMIT_RETRY_BACKOFF_S if backoff_s is None else backoff_s   # read at call time
    max_attempts = len(backoff_s) if max_attempts is None else max_attempts
    for attempt in range(1, max_attempts + 1):
        try:
            _, number = submit_health_check(
                run_folder=run_folder, backend=backend, sampler=sampler, service=service, nshots=nshots,
                service_name=service_name, reps=reps, accompanying=True,
                after_submission=True, chunks=chunks, return_number=True,
                backend_name=backend_name,
            )
            numbers.append(number)
            print(f"  ✅ Health check (after submission) submitted → datafiles/{run_folder}/health_checks/")
            return True, numbers
        except HealthCheckSubmitError as e:
            numbers.append(e.number)
            if _is_transient_network_error(e.cause) and attempt < max_attempts:
                wait = backoff_s[min(attempt - 1, len(backoff_s) - 1)]
                print(f"  ⚠️  POST health check submission attempt {attempt}/{max_attempts} failed "
                      f"(network: {e.cause}) — retrying in {wait}s.")
                time.sleep(wait)
                continue
            print(f"  ⚠️  POST health check could NOT be submitted after {attempt} attempt(s): {e.cause}. "
                  f"Chunks were already submitted and remain valid; recorded as failed check(s) "
                  f"{', '.join(f'#{n}' for n in numbers)}.")
            return False, numbers
    return False, numbers


def collect_health_check(exp_results_folder_path, service=None,
                         fake_registry=None, service_name=None, record_suffix=None,
                         accompanying=False, number=None, backend_name=None):
    """
    Standalone function — no bacq_ps_qbcs instance needed.
    Collect results from IBM or IQM health check job and save to disk.

    :param exp_results_folder_path: str/Path — run folder
    :param service:                 QiskitRuntimeService — IBM real QPU, MQSSQiskitAdapter — IQM cross-session retrieval
    :param fake_registry:           dict — for fake/Aer backends (same session)               
    :param service_name:            str - "IBM_Cloud" / "MQSS_LRZ" / "PCSS" / "IQM_Resonance" / "QAAS_IT4I" / "local"
    :param backend_name:            str — the script's BACKEND_NAME, for the progress lines (default: the record's)
    :param accompanying:            bool — collect health checks in health_checks/ (see submit_health_check).
                                    With number=None every finished, uncollected check is collected (files
                                    updated in place); unfinished ones are skipped, failed ones marked failed.
    :param number:                  int — with accompanying=True, collect only this health check (waits for it)
    :returns:                       dict — health check result (accompanying=True: the last collected/loaded
                                    check record, or None if there are none; keys 'P011', 'verdict', ...)
    """
    exp_results_folder_path = Path(exp_results_folder_path)
    if not accompanying:
        fname       = "health_check.json" if record_suffix is None else f"health_check_{record_suffix}.json"
        record_path = exp_results_folder_path / "job_records" / fname
        return _collect_health_check_one(exp_results_folder_path, record_path, service,
                                         fake_registry, service_name, accompanying=False, backend_name=backend_name)

    assert record_suffix is None, "record_suffix must be None for accompanying health checks."
    checks = [c for c in load_health_checks(exp_results_folder_path)
              if number is None or c["number"] == number]
    last = None
    for c in checks:
        # Collecting all checks (number=None) is non-blocking: an unfinished check is left for a later
        # collection, a failed one marked failed. One requested check (a submission's pre-submission
        # check, whose verdict it needs) is waited for.
        if number is None and c.get("status") not in ("complete", "failed") \
                and service_name in (IBM_CLOUD, MQSS_LRZ, IQM_RESONANCE, QAAS_IT4I):
            state, detail = _remote_job_state(service_name, service, c["job_id"], c.get("backend_name"), c.get("nshots"))
            if state == "pending":
                print(f"  [Health check] #{c['number']} (job_id={c['job_id']}) not finished ({detail}) — collect it later.")
                continue
            if state == "failed":
                rec = {k: v for k, v in c.items() if k != "label"}
                rec.update(status="failed", failure=f"job ended {detail}")
                save_health_check(exp_results_folder_path, rec)
                print(f"  ⚠️  Health check #{c['number']} (job_id={c['job_id']}) ended {detail} — marked failed.")
                continue
        if c.get("status") == "failed":
            continue
        try:
            last = _collect_health_check_one(exp_results_folder_path,
                                             health_check_path(exp_results_folder_path, c["number"]),
                                             service, fake_registry, service_name, accompanying=True, backend_name=backend_name,
                                             verb="Waiting for" if number is not None else "Collecting")
        except Exception as e:
            print(f"  ⚠️  Health check #{c['number']} (job_id={c.get('job_id')}) not collected: {e}")
    return last


def _collect_health_check_one(exp_results_folder_path, record_path, service, fake_registry,
                              service_name, accompanying, backend_name=None, verb="Waiting for"):
    if not record_path.exists():
        raise FileNotFoundError(f"No health check job record found in {rel_path(record_path)}")   # an exception, not an exit: collect_health_check() skips one bad record and goes on

    record       = json.loads(record_path.read_text(encoding="utf-8"))
    job_id       = record["job_id"]
    nshots       = record["nshots"]
    label_backend = backend_name or record["backend_name"]   # the script's BACKEND_NAME (the parameter), for the progress lines
    backend_name = record["backend_name"]

    recorded_service = record.get("service_name", IBM_CLOUD)   # old records without the field are IBM's
    if service_name != recorded_service:
        raise ValueError(f"The health check in {rel_path(record_path)} was submitted via {recorded_service}, "
                        f"not {service_name}: collect it with that provider's script.")

    # ── Idempotency ───────────────────────────────────────────────────
    if record["status"] == "complete":
        print(f"  [Health check] Already collected — loading result.")
        return record

    # ── Retrieve result (timed: collect_s, labels health_check_<n>_collect_start/_end) ──
    hc_label  = f"health_check_{record.get('number', record_path.stem.rsplit('_', 1)[-1])}"
    ts_file   = exp_results_folder_path / TIMESTAMP_FILENAME
    timestamp(label=f"{hc_label}_collect_start", timestamp_file=ts_file)
    collect_t0 = time.perf_counter()
    finished_at  = "N/A"
    scheduled_at = "N/A"
    if service_name == MQSS_LRZ:
        # ── IQM EQE path — cross-session via adapter.client ────────────
        assert service is not None, "Pass service=MQSSQiskitAdapter(...) for MQSS_LRZ."
        from mqss_client.job import CircuitJobRequest
        job_type = CircuitJobRequest(
            circuits       = "",
            circuit_format = "qasm",
            resource_name  = backend_name,
            shots          = nshots,
            no_modify      = True,
            queued         = False,
        )
        print_job_line("Health check", label_backend, service_name, job_id, verb)
        result       = service.client.wait_for_job_result(job_id, job_type)
        # MQSS timestamps are naive Europe/Berlin time — converted to UTC (source/core/records/job_timing.py)
        timing       = eqe_collect_timing(result, record.get("submitted_at"), 1)
        finished_at  = timing["execution_end"] or "N/A"
        scheduled_at = timing["execution_start"] or "N/A"
        check        = _check_executed_health_check(exp_results_folder_path, record, service_name)
        counts_raw = result.counts if isinstance(result.counts, dict) else result.counts[0]   # single circuit — index 0
        counts      = {k[::-1]: v for k, v in counts_raw.items()}

    elif service_name == IQM_RESONANCE:
        # `service` is an IQMBackend:
        #   IQMProvider(url, quantum_computer=..., token=...).get_backend()
        assert service is not None, "Pass the IQMBackend as `service=` for IQM_RESONANCE collection."

        print_job_line("Health check", label_backend, service_name, job_id, verb)
        job = service.retrieve_job(job_id)
        job.wait_for_final_state()                       # blocks until COMPLETED / FAILED / CANCELLED
        if not job.done():
            raise RuntimeError(
                f"Health check job {job_id} did not complete: "
                f"status={job.status()} — {job.error_message()}"
            )

        result     = job.result()                        # qiskit Result
        counts_raw = result.get_counts()
        counts_raw = counts_raw[0] if isinstance(counts_raw, list) else counts_raw
        counts     = {k[::-1]: v for k, v in counts_raw.items()}

        # ── Execution window, queue and billed time from the server (timeline + raw job) ──
        timing       = iqm_resonance_collect_timing(service, job_id)
        scheduled_at = timing.get("execution_start") or "N/A"
        finished_at  = timing.get("execution_end") or "N/A"
        if scheduled_at == "N/A" or finished_at == "N/A":      # discover the real step names on first run
            print(f"  ⚠️  [Health check] No execution window in the job's timeline ({len(timing.get('timeline') or [])} events) — timing incomplete.")
        check        = _check_executed_health_check(exp_results_folder_path, record, service_name, backend=service)

    elif service_name == QAAS_IT4I:
        # `service` is the qaas VLQ backend (source/core/qpu/iqm_vlq.py)
        assert service is not None, "Pass the VLQ backend as `service=` for QAAS_IT4I collection."
        print_job_line("Health check", label_backend, service_name, job_id, verb)
        job        = VLQ.retrieve_job(service, job_id)          # logged, time-limited (source/core/qpu/iqm_vlq.py)
        result     = VLQ.wait_for_result(job, job_id)
        counts_raw = result.get_counts()
        counts_raw = counts_raw[0] if isinstance(counts_raw, list) else counts_raw
        counts     = {k[::-1]: v for k, v in counts_raw.items()}
        timing       = VLQ.vlq_collect_timing(job, record.get("upload_end_at") or record.get("submitted_at"))
        scheduled_at = timing.get("execution_start") or "N/A"
        finished_at  = timing.get("execution_end") or "N/A"
        check        = _check_executed_health_check(exp_results_folder_path, record, service_name, job=job)

    elif service_name == IBM_CLOUD:
        # ── IBM real QPU path ─────────────────────────────────────────
        job = service.job(job_id)
        print_job_line("Health check", label_backend, service_name, job_id, verb)
        while True:
            status_str = str(job.status()).upper()
            print(f"  [Health check] Status: {status_str}")
            state = job_state_of_status(status_str)
            if state == "done":
                break
            if state == "failed":
                raise RuntimeError(f"Health check job {job_id} failed: {status_str}")
            time.sleep(30)
        result       = job.result()
        timing       = ibm_collect_timing(job, result)
        scheduled_at = timing.get("execution_start") or timing.get("running_at") or "N/A"   # QPU execution start
        finished_at  = timing.get("finished_at") or "N/A"
        check        = _check_executed_health_check(exp_results_folder_path, record, service_name, job=job)
        rdict      = result[0].data
        if hasattr(rdict, 'meas'):   counts_raw = rdict.meas.get_counts()
        elif hasattr(rdict, 'c'):    counts_raw = rdict.c.get_counts()
        else: raise RuntimeError(f"Unknown register. Available: {list(rdict.__dict__.keys())}")
        counts = {k[::-1]: v for k, v in counts_raw.items()}

    else:
        # ── Fake/Aer path — same session ─────────────────────────────
        assert fake_registry is not None, \
            "Pass fake_registry=qbcs._fake_job_registry for fake backends."
        job = fake_registry.get(job_id)
        if job is None:
            raise RuntimeError(f"Job {job_id} not in fake registry — must be same session.")
        print(f"  [Health check] Fake backend — collecting immediately.")
        timing     = {}
        check      = CC.not_available("simulated backend (fake/Aer): there is no server copy of the circuit")
        result     = job.result()
        rdict      = result[0].data
        if hasattr(rdict, 'meas'):   counts_raw = rdict.meas.get_counts()
        elif hasattr(rdict, 'c'):    counts_raw = rdict.c.get_counts()
        else: raise RuntimeError(f"Unknown register. Available: {list(rdict.__dict__.keys())}")
        counts = {k[::-1]: v for k, v in counts_raw.items()}

    # ── Verdict ───────────────────────────────────────────────────────
    status, p_011 = print_health_verdict(counts)

    # ── Circuit check + timing into the record ─────────────────────────
    _warn_circuit_check("Health check", check)
    record["circuit_check"] = check
    merge_timing(record, **timing, collect_s=round(time.perf_counter() - collect_t0, 3))
    timestamp(label=f"{hc_label}_collect_end", timestamp_file=ts_file)

    # ── Save result ───────────────────────────────────────────────────
    record.update({
        "scheduled_at": iso_utc(scheduled_at),
        "finished_at":  iso_utc(finished_at),
        "collected_at": iso_utc(now_utc_iso()),
        "counts":       counts,
        "P011":         p_011,
        "verdict":      status,
        "threshold":    HEALTH_CHECK_THRESHOLD,
        "status":       "complete",
    })
    if accompanying:
        save_health_check(exp_results_folder_path, record)
        print(f"  [Health check] #{record['number']} updated → {rel_path(record_path)}")
    else:
        write_job_record(record_path, record)   # a standalone check keeps its result in its record
    return record


def top_counts(counts, n=4):
    """The n most frequent outcomes of `counts` as `{'011': 91, ...}`, with `, +k more` after them when cut."""
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    text   = "{" + ", ".join(f"'{k}': {v}" for k, v in ranked[:n]) + "}"
    return text + (f", +{len(ranked) - n} more" if len(ranked) > n else "")


def print_health_verdict(counts):
    """Print a health check's verdict and counts, the same lines for every QPU, and return (verdict, P(011))."""
    status = health_check_verdict(counts)
    p_011  = health_check_p011(counts)
    print(f"\n  {'✅' if status == 'functional' else '❌'} QPU {status} — "
          f"P(011) = {p_011:.3f}  ({counts.get('011', 0)}/{sum(counts.values())} shots)")
    print(f"  Top counts: {top_counts(counts)}")
    return status, p_011


def print_standalone_summary(record, p_011, functional, layout_note="(virtual → physical)", backend_name=None):
    """Print the result block of a standalone health-check script and return its exit code
    (0 functional, EXIT_HC_DEGRADED degraded). `record`: the collected health check (backend_name,
    layout, nshots, counts and the provider's timestamps); the times a provider does not report are left out.
    backend_name: the script's BACKEND_NAME (default: the record's, which for VLQ is a class name)."""
    from source.core.records.health_check_records import EXIT_HC_DEGRADED
    counts = record["counts"]
    section("Health check result")
    print(f"  Backend:      {backend_name or record['backend_name']}")
    print(f"  Qubits:       {record['layout']}   {layout_note}")
    print(f"  Submitted at: {record['submitted_at']}")
    print(f"  Uploaded  at: {record.get('upload_end_at', 'N/A')}")
    if "scheduled_at" in record:
        print(f"  Scheduled at: {record['scheduled_at']}")
    print(f"  Finished  at: {record['finished_at']}")
    if "collected_at" in record:
        print(f"  Collected at: {record['collected_at']}")
    print(f"  Shots:        {record['nshots']}")
    print(f"  P(|011⟩):     {p_011:.4f}")
    print(f"  Status:       {'✅ functional' if functional else '❌ degraded'}")
    print(f"  Top counts:   {top_counts(counts)}")
    return 0 if functional else EXIT_HC_DEGRADED
