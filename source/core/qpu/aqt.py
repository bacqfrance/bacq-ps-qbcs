"""
aqt.py
AQT PIAST-Q specifics shared by the chunk submission (qbcs.py) and the health checks: the QPU time
of a circuit from AQT's gate durations, and a job wrapper that retries the SDK's transient
network errors and gives up on a circuit that is not executed in time (shots over AQT's per-circuit maximum
are split by job_timing.aqt_shot_parts(); the wrapper raises ValueError for them);
and the loop script's own ends: the STOP file and the time limit.
"""

import os
import threading
import time
from pathlib import Path

from tqdm import tqdm

from source.core.common import format_retry
from source.core.records.job_timing import AQT_MAX_SHOTS


# ── AQT / PIAST-Q deterministic QPU-time cost model ───────────────────────────
# Gate durations (µs) + fixed per-circuit overhead, from the timing figures
# AQT/PSNC give for PIAST-Q.
_AQT_GATE_US             = {"r": 31, "rx": 31, "rxx": 508, "rz": 0}   # r and rx are the same 31 µs pulse
_AQT_CIRCUIT_OVERHEAD_US = 4000 + 12000 + 1000   # doppler + sideband cooling + detection

def aqt_circuit_qpu_us(circuit):
    """Deterministic QPU time (µs) for one circuit in the {r, rx, rz, rxx} basis, single shot."""
    counts     = circuit.count_ops()
    disallowed = set(counts) - set(_AQT_GATE_US) - {"barrier", "measure"}
    if disallowed:
        raise ValueError(f"aqt_circuit_qpu_us: unexpected ops {disallowed}")
    return _AQT_CIRCUIT_OVERHEAD_US + sum(_AQT_GATE_US[g] * counts.get(g, 0) for g in _AQT_GATE_US)

def _aqt_run(backend, circuits, shots):
    """backend.run() on AQT, refusing more than AQT_MAX_SHOTS shots per circuit (AQT's limit).
    submit_chunk_aqt() splits larger shot counts into several runs (aqt_shot_parts).
    The returned job's result() retries dropped connections (_AQTResilientJob)."""
    if int(shots) > AQT_MAX_SHOTS:
        raise ValueError(f"AQT accepts at most {AQT_MAX_SHOTS} shots per circuit, got {shots}. "
                         f"Split the shots with job_timing.aqt_shot_parts().")
    return _AQTResilientJob(backend.run(circuits, shots=int(shots)), backend)

# Waits (s) before each retry of one PCSS API call after a connection error; about 19 min
# in all, then the error goes up and ends the loop script.
_AQT_RETRY_DELAYS_S = (5, 15, 30, 60, 120, 300, 600)

def _aqt_retry(what, fn):
    """fn(), retried after network errors (httpx.TransportError: dropped connection,
    refused connect, timeout) instead of letting them end the chunk."""
    import httpx
    for i, delay in enumerate((*_AQT_RETRY_DELAYS_S, None)):
        try:
            out = fn()
            if i:
                print(f"  ✅ [AQT] {what}: reconnected after {i} {'retry' if i == 1 else 'retries'}.", flush=True)
            return out
        except httpx.TransportError as e:
            if delay is None:
                print(f"  ❌ [AQT] {what}: still no connection after {i} retries — giving up, the loop script ends.", flush=True)
                raise
            print(format_retry("AQT", what, e, i + 1, len(_AQT_RETRY_DELAYS_S), delay), flush=True)
            time.sleep(delay)

# One circuit's task must be executed within this long of its submission (the connection retries
# above count towards it). PCSS gives no queue state and its result query polls without end, so a
# task that is never executed (PIAST-Q down, in maintenance or reserved) would block the job forever.
AQT_TASK_TIMEOUT_S = 30 * 60

# ── The loop script's own ends (qbcs_real_submit_aqt_loop.py; the PCSS sbatch files act on these exit codes) ──
STOP_EXIT_CODE  = 4   # `touch datafiles/<run folder>/STOP`: stop at the next chunk boundary
PAUSE_EXIT_CODE = 3   # AQT_PAUSE_AFTER_HOURS has passed: the sbatch file requeues its job

def take_stop_file(run_folder_path):
    """True if <run folder>/STOP exists, deleting it, so a later job of the run is not stopped by it."""
    stop = Path(run_folder_path) / "STOP"
    if not stop.exists():
        return False
    stop.unlink(missing_ok=True)
    return True

def pause_after_hours(environ=os.environ):
    """The loop's time limit in hours: AQT_PAUSE_AFTER_HOURS (set by the PCSS sbatch files), inf when unset."""
    return float(environ.get("AQT_PAUSE_AFTER_HOURS", "inf"))

def pause_due(started, hours, now=None):
    """True once `hours` have passed since `started` (time.perf_counter() values)."""
    return (time.perf_counter() if now is None else now) - started > hours * 3600

class AQTTaskTimeout(RuntimeError):
    """A circuit's PCSS task was not executed within AQT_TASK_TIMEOUT_S."""

def _result_within(what, fn, seconds):
    """fn(), abandoned with AQTTaskTimeout after `seconds`. fn runs in a daemon thread, because the
    PCSS result query cannot be interrupted: after a timeout it keeps polling until the process
    ends, which the caller's error does."""
    box = {}
    def target():
        try:
            box["value"] = fn()
        except BaseException as e:      # handed to the caller below
            box["error"] = e
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise AQTTaskTimeout(f"{what} was not executed within {seconds / 60:g} min of its submission")
    if "error" in box:
        raise box["error"]
    return box["value"]

class _AQTResilientJob:
    """AQTDirectAccessJob whose result() survives connection drops.

    The direct-access job sends each circuit as its own PCSS task, one after another,
    and a network error anywhere ends the whole job: running it again would start over
    at circuit 0. This loop does the same, but retries a failed call where it stopped:
    a failed result query polls the same task again (no QPU time lost), a failed submit
    sends that circuit again. (If a submit reached PCSS but its reply was lost, that
    circuit runs twice; the extra task's result is not used.)"""

    def __init__(self, job, backend):
        self.job, self.backend = job, backend

    def result(self):
        from qiskit.result import Result
        from qiskit_aqt_provider.aqt_job import _partial_qiskit_result_dict
        from qiskit_aqt_provider.api_client.models_direct import JobResultError
        job, opts = self.job, self.job.options
        results   = []
        try:
            for i, circuit in enumerate(tqdm(job.circuits, disable=not opts.with_progress_bar)):
                api_circuit = job.api_submit_payload.payload.circuits[i]
                try:
                    task_id = _aqt_retry(f"circuit {i}: submit", lambda: self.backend.submit(api_circuit))
                except Exception as e:
                    e.aqt_failure_stage = "submission"      # read by submit_chunk_aqt's failed record
                    raise
                try:
                    api_result = _result_within(
                        f"AQT task {task_id} (circuit {i})",
                        lambda: _aqt_retry(f"circuit {i}: result of task {task_id}",
                                           lambda: self.backend.result(task_id, timeout=opts.query_timeout_seconds)),
                        AQT_TASK_TIMEOUT_S)
                except AQTTaskTimeout as e:
                    e.args = (f"{e.args[0]}. PIAST-Q is probably down, in maintenance or reserved, or its queue is "
                              f"very long (the PCSS API reports no queue state), so the run stops here instead of "
                              f"waiting on. Check PIAST-Q's availability with PCSS and rerun later: finished chunks "
                              f"are kept and a rerun resumes.",)
                    e.aqt_failure_stage = "execution"
                    raise
                except Exception as e:
                    e.aqt_failure_stage = "execution"
                    raise
                if isinstance(api_result.payload, JobResultError):
                    err = RuntimeError(f"AQT task {task_id} (circuit {i}) failed: {api_result.payload}")
                    err.aqt_failure_stage = "execution"
                    raise err
                results.append(_partial_qiskit_result_dict(api_result.payload.result, circuit,
                                                           shots=opts.shots, memory=opts.memory))
        except Exception as e:
            e.aqt_circuits_done = len(results)      # circuits finished before the failure: read by submit_chunk_aqt's failed record
            raise
        return Result.from_dict({"backend_name": self.backend.name, "backend_version": self.backend.version,
                                 "qobj_id": id(job.circuits), "job_id": job.job_id(),
                                 "success": True, "results": results})
