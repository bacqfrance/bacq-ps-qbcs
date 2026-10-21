"""
remote_jobs.py
What qbcs.py and health_check.py share about jobs on a provider: their live state, the timing of a
failed job, how a submission error is classified (not connected / answer lost / final) and retried,
the submit errors raised to the scripts, and the warning for an executed-circuit check.
"""

import time

from source.core.common import exit_with_error, scrub_paths
from source.core.qpu import iqm_vlq as VLQ
from source.core.qpu.providers import job_state_of_status
from source.core.records import circuit_check as CC
from source.core.records.job_timing import ibm_collect_timing, iqm_resonance_collect_timing, SYNCHRONOUS_SERVICES
from source.core.records.run_names import IBM_CLOUD, IQM_RESONANCE, QAAS_IT4I, MQSS_LRZ


def print_job_line(tag, backend_name, service_name, job_id, verb="Collecting"):
    """The line printed before fetching a provider's job, the same for every provider:
    `  [<tag>] <verb> <backend> (<service>) job <id> ...` (tag: "Health check", "Chunk 3", ...).
    verb: "Waiting for" where the script blocks until the job may have finished (the pre-submission
    health check, a standalone health check), "Collecting" where the job is known to be finished.
    backend_name is the script's BACKEND_NAME; the job record's own name is the fallback."""
    print(f"  [{tag}] {verb} {backend_name} ({service_name}) job {job_id} ...", flush=True)


def _remote_job_state(service_name, service, job_id, backend_name=None, nshots=None):
    """Non-blocking state of a provider job: ("done" | "pending" | "failed", detail).
    service: IBM QiskitRuntimeService, IQM EQE1 (MQSS_LRZ) MQSSQiskitAdapter, IQM_RESONANCE IQMBackend,
    QAAS_IT4I (IQM VLQ) the qaas backend. A status query that fails counts as pending (tried again next time).
    A service without a query here raises ValueError (a code bug); an MQSS_LRZ record without backend_name ends
    the script through exit_with_error (old records to upgrade). Neither is turned into pending."""
    if service_name in SYNCHRONOUS_SERVICES:   # synchronous: a record exists only once its job has finished
        return "done", service_name
    if service_name not in (MQSS_LRZ, IBM_CLOUD, IQM_RESONANCE, QAAS_IT4I):
        raise ValueError(f"unknown service {service_name!r}: add its status query to _remote_job_state")
    if service_name == MQSS_LRZ and not backend_name:
        exit_with_error(f"job record {job_id} has no backend_name; upgrade the old records")
    try:
        if service_name == MQSS_LRZ:
            from mqss_client.job import CircuitJobRequest
            job_type = CircuitJobRequest(circuits="", circuit_format="qasm", resource_name=backend_name,
                                         shots=nshots or 1, no_modify=True, queued=False)
            status = str(service.client.job_status(job_id, job_type).value).upper()
        elif service_name == IBM_CLOUD:
            status = str(service.job(job_id).status()).upper()
        elif service_name == IQM_RESONANCE:
            status = str(service.retrieve_job(job_id).status()).upper()
        elif service_name == QAAS_IT4I:
            status = str(VLQ.job_status(VLQ.retrieve_job(service, job_id), job_id)).upper()   # logged, time-limited
    except Exception as e:
        return "pending", f"status unavailable ({type(e).__name__}: {e})"
    return job_state_of_status(status), status


def _failed_job_timing(service_name, service, record):
    """Whatever timing the provider still reports for a job that ended in failure (QPU time
    it billed, queue, start/end), for the cost accounting of failed attempts. The query's own
    wall time is stored as collect_s. Never raises; an unavailable value is left out (qpu_s / queue_s are kept even when None)."""
    t0  = time.perf_counter()
    out = {}
    try:
        job_id = record["job_id"]
        if service_name == IBM_CLOUD:
            out = ibm_collect_timing(service.job(job_id))
        elif service_name == IQM_RESONANCE:
            out = iqm_resonance_collect_timing(service, service.retrieve_job(job_id), job_id)
        elif service_name == QAAS_IT4I:
            out = VLQ.vlq_collect_timing(VLQ.retrieve_job(service, job_id),
                                         record.get("upload_end_at") or record.get("submitted_at"))
        # IQM EQE1 (MQSS_LRZ) returns no result object, hence no timestamps, for a failed job.
    except Exception as e:
        out = {**out, "failed_timing_error": scrub_paths(f"{type(e).__name__}: {e}")}
    out = {k: v for k, v in out.items() if v is not None or k in ("qpu_s", "queue_s")}
    out["collect_s"] = round(time.perf_counter() - t0, 3)
    return out


def _warn_circuit_check(label, check):
    """Terminal line for a circuit_check result (source/core/records/circuit_check.py) — a warning, never an error."""
    st = check.get("status")
    if st == CC.MATCH:
        print(f"  [{label}] Executed circuits match the submitted ones ({check.get('n_executed')} circuits).")
    elif st == CC.MISMATCH:
        print(f"  ⚠️  [{label}] EXECUTED CIRCUITS DIFFER FROM THE SUBMITTED ONES "
              f"({check.get('n_differences')} differences, first: {(check.get('differences') or [None])[0]})")
    elif st == CC.CHECK_FAILED:
        print(f"  ⚠️  [{label}] Could not check the executed circuits: {check.get('error')}")


# ── Health check retry/exit-code handling (submit scripts) ────────────────────
# See the QPU readmes: EXIT_HC_DEGRADED / EXIT_HC_NOT_COLLECTED / EXIT_HC_NOT_SUBMITTED
# in source/core/records/health_check_records.py are the exit codes the submit scripts use for these outcomes.

class HealthCheckSubmitError(RuntimeError):
    """Raised by submit_health_check(accompanying=True) when an asynchronous service's submission
    call fails. A 'failed' record is written before this is raised, under `number`."""
    def __init__(self, number, cause):
        super().__init__(f"health check #{number} could not be submitted: {cause}")
        self.number = number
        self.cause  = cause


def _is_transient_network_error(exc):
    """True for a connection/DNS/timeout-shaped error, walking __cause__/__context__. These are
    the only errors require_functional_health_check / submit_post_health_check retry — anything
    else (a degraded/failed job, an assertion, a bad response body) is not a network blip and is
    not retried.
    Unlike the chunk submission (_submit_error_kind), a timeout or reset whose answer was lost is
    retried here too, so a health-check retry may leave a duplicate health-check job (accepted)."""
    transient_types = []
    try:
        import requests.exceptions as _rex
        transient_types += [_rex.ConnectionError, _rex.Timeout]
    except ImportError:
        pass
    try:
        import urllib3.exceptions as _uex
        transient_types += [_uex.NewConnectionError, _uex.MaxRetryError, _uex.NameResolutionError]
    except ImportError:
        pass
    try:
        import httpx
        transient_types += [httpx.TransportError]
    except ImportError:
        pass
    import socket
    transient_types += [socket.gaierror, socket.timeout, ConnectionError, TimeoutError]
    transient_types = tuple(transient_types)

    transient_texts = (
        "getaddrinfo failed", "failed to resolve", "name or service not known",
        "temporary failure in name resolution", "max retries exceeded",
        "connection reset", "connection aborted", "connection refused",
        "read timed out", "nameresolutionerror",
    )

    seen = set()
    e = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, transient_types):
            return True
        if any(t in str(e).lower() for t in transient_texts):
            return True
        e = e.__cause__ or e.__context__
    return False


def _submit_error_kind(exc):
    """How a failed chunk submission failed, walking __cause__/__context__:
      "not_connected"  the request never reached the provider (DNS failure, connection refused,
                       connect timeout): nothing was created, so retrying is safe
      "answer_lost"    the connection broke or timed out after the request may have arrived
                       (read timeout, reset/aborted connection, VLQTimeout): the job may exist
                       on the provider, so a retry could run the chunk twice — not retried
      None             any other error (rejected payload, assertion, bug): not retried
    answer_lost wins when a chain holds both (urllib3 wraps either in MaxRetryError)."""
    lost_types, conn_types = [VLQ.VLQTimeout, ConnectionResetError, ConnectionAbortedError, BrokenPipeError], \
                             [ConnectionRefusedError]
    connect_timeout = ()
    try:
        import requests.exceptions as _rex
        connect_timeout = (_rex.ConnectTimeout,)
        lost_types.append(_rex.ReadTimeout)
    except ImportError:
        pass
    try:
        import urllib3.exceptions as _uex
        conn_types += [_uex.NewConnectionError, _uex.NameResolutionError, _uex.ConnectTimeoutError]
        lost_types += [_uex.ReadTimeoutError, _uex.ProtocolError]
    except ImportError:
        pass
    try:
        import httpx
        conn_types += [httpx.ConnectError, httpx.ConnectTimeout]
        lost_types += [httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.ReadError]
    except ImportError:
        pass
    import socket
    conn_types += [socket.gaierror]
    lost_texts = ("read timed out", "connection reset", "connection aborted", "remotedisconnected",
                  "remote end closed connection", "broken pipe")
    conn_texts = ("getaddrinfo failed", "failed to resolve", "name or service not known",
                  "temporary failure in name resolution", "nameresolutionerror", "connection refused",
                  "failed to establish a new connection", "newconnectionerror", "connect timeout")

    kinds, seen, e = set(), set(), exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        text = str(e).lower()
        if isinstance(e, connect_timeout):
            kinds.add("not_connected")
        elif isinstance(e, tuple(lost_types)) or any(t in text for t in lost_texts):
            kinds.add("answer_lost")
        elif isinstance(e, tuple(conn_types)) or any(t in text for t in conn_texts):
            kinds.add("not_connected")
        e = e.__cause__ or e.__context__
    if "answer_lost" in kinds:
        return "answer_lost"
    return "not_connected" if "not_connected" in kinds else None


SUBMIT_RETRY_BACKOFF_S = (30, 60, 120)   # a chunk submission gets 4 tries, waiting 30, 60, 120 s between them (the 120 s only there);
# the health-check loops use len() tries, so 3: they wait 30 s and 60 s


class ChunkSubmitError(RuntimeError):
    """A chunk submission that did not succeed (after retries, if the error was retryable).
    attempts: [{"at", "s", "error", "kind"}]; kind: see _submit_error_kind.
    submit_s: seconds spent submitting, over all attempts (set by the raiser after construction)."""

    def __init__(self, label, cause, attempts, kind):
        self.cause, self.attempts, self.kind = cause, attempts, kind
        super().__init__(f"{label}: submission failed after {len(attempts)} attempt(s): {cause!r}")
