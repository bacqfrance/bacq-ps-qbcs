"""
providers.py
Constants several QPU scripts share, kept in one place so they cannot drift apart:
the IQM Resonance endpoint, the shots of the health checks that accompany every run, and the
settings of the standalone health check (health_check_real_*.py), whose run folder the job-status
scripts look up, the reading of a provider's job-status text (job_state_of_status), and the
IBM batch context (ibm_batch).
"""

import contextlib

IQM_RESONANCE_URL = "https://resonance.iqm.tech/"

# The health checks before and after every run's chunks (submit, simulate and local benchmark scripts).
ACCOMPANYING_HEALTH_CHECK_NSHOTS = 100

# The standalone health check (health_check_real_*.py): one repetition of the health-check circuit
# (4 CNOTs), 100 shots. check_real_qpu_job_status_*.py finds its folder from the same values.
STANDALONE_HEALTH_CHECK_REPS   = 1
STANDALONE_HEALTH_CHECK_NSHOTS = 100



def job_state_of_status(status):
    """"done" | "failed" | "pending" from a provider's job-status text (an enum's repr or value). A cancelled
    job counts as failed, like an errored one; anything not final (queued, running, validating, …) is pending.
    Collection, the IBM health-check wait and the status report map a status to its job state here;
    job_status_check.classify() only refines the failed states into ERROR vs CANCELLED.
    Known literals: IBM and IQM Resonance statuses as their SDKs report them; EQE1 (MQSS): COMPLETED, PENDING and CANCELLED
    observed live (a PENDING job cancelled with cancel_job() then reads CANCELLED), WAITING and FAILED from the client's
    enum only (not observed). VLQ (qiskit JobStatus via
    HEAppE): DONE, RUNNING, ERROR observed live (a HEAppE "Failed" job reads ERROR). A job cancelled with cancel() reads
    ERROR when queried through a fresh backend object: qaas reports CANCELLED only through the object that issued the cancel
    (qaas/client/backend.py). QUEUED, INITIALIZING and VALIDATING not observed."""
    s = str(status).upper()
    if "DONE" in s or "COMPLETED" in s:
        return "done"
    if any(k in s for k in ("ERROR", "FAIL", "CANCEL")):
        return "failed"
    return "pending"


@contextlib.contextmanager
def ibm_batch(backend, max_time=None):
    """Open an IBM Batch on `backend` (max_time: seconds, None = the service default), yield a SamplerV2 bound
    to it, and close the batch when the block ends, also when it raises. qiskit_ibm_runtime is imported here:
    only the IBM venv has it."""
    from qiskit_ibm_runtime import Batch, SamplerV2
    batch = Batch(backend=backend, max_time=max_time)
    try:
        yield SamplerV2(mode=batch)
    finally:
        batch.close()
