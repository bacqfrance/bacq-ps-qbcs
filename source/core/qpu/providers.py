"""
providers.py
Constants several QPU scripts share, kept in one place so they cannot drift apart:
the IQM Resonance endpoint, the shots of the health checks that accompany every run, and the
settings of the standalone health check (health_check_real_*.py), whose run folder the job-status
scripts look up, the reading of a provider's job-status text (job_state_of_status), and the
IBM batch context (ibm_batch) and the connection to an explicitly chosen IBM instance (connect_ibm).
"""

import contextlib

from source.core.common import exit_with_error

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


IBM_CHANNEL = "ibm_quantum_platform"
IBM_OPEN_PLAN = "open"   # the plan name IBM gives a free Open Plan instance (QiskitRuntimeService.instances()[i]["plan"])


def pick_ibm_instance(instances, name=None):
    """The instance dict (keys "name", "plan", "crn") to use from `instances` (QiskitRuntimeService.instances()):
    the one called `name` (or whose CRN is `name`); without a name, the account's single Open Plan instance.
    Exits with code 2 and a ❌ line listing the instances when there is no such instance, no Open Plan instance
    or more than one (then the user must name one). IBM jobs live in the instance that received them, and the
    QPUs listed differ per instance, so no default is left to the library (which may pick a paid instance)."""
    listing = ", ".join(f"{i['name']} ({i['plan']})" for i in instances) or "none"
    if name is not None:
        found = [i for i in instances if name in (i["name"], i["crn"])]
        if not found:
            exit_with_error(f"IBM instance {name!r} not found on this account — available: {listing}")
        return found[0]
    opens = [i for i in instances if i["plan"] == IBM_OPEN_PLAN]
    if len(opens) != 1:
        exit_with_error(f"the account has {len(opens)} IBM Open Plan instances, so none is the default — name one with "
                        f"instance=NAME; available: {listing}")
    return opens[0]


def connect_ibm(token, instance=None, backend_name=None):
    """(QiskitRuntimeService, instance dict) connected to the explicitly chosen IBM instance: `instance` (a name, see
    pick_ibm_instance()), by default the Open Plan one. Prints the choice and every instance of the account. With `backend_name`, exits with code 2
    unless that QPU is listed on the instance (the list differs per instance). qiskit_ibm_runtime is imported here:
    only the IBM venv has it. One service does it all: with a name the library resolves the instance itself; by default
    it is built with instance="auto" (the library's switch for automatic selection, without its "Instance was not set"
    warning) and plans_preference=["open"], which limits the selection to Open Plan instances. The result is checked
    against the account's instances (pick_ibm_instance()), so a missing or ambiguous Open Plan instance, or a name the
    library resolved differently, still ends with one ❌ line."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    kwargs = dict(channel=IBM_CHANNEL, token=token)
    kwargs.update(instance="auto", plans_preference=[IBM_OPEN_PLAN]) if instance is None else kwargs.update(instance=instance)
    try:
        service = QiskitRuntimeService(**kwargs)
    except Exception:
        # The library refused the instance (unknown name, no Open Plan instance): say so with the account's instances;
        # any other failure (network, token) is re-raised.
        pick_ibm_instance(list(QiskitRuntimeService(channel=IBM_CHANNEL, token=token, instance="auto").instances()), instance)
        raise
    instances = list(service.instances())
    chosen    = pick_ibm_instance(instances, instance)
    if service.active_instance() != chosen["crn"]:
        exit_with_error(f"the IBM service did not activate the instance {chosen['name']!r}")
    available = "\n".join(f"    - {i['name']} ({i['plan']})" for i in instances)
    print(f"\n  Instance: {chosen['name']} ({chosen['plan']})   [chosen for this run]\n  Available instances:\n{available}\n")   # blank lines: set it off from the library's log lines
    if backend_name is not None:
        listed = sorted(b.name for b in service.backends())
        if backend_name not in listed:
            exit_with_error(f"{backend_name} is not listed on the IBM instance {chosen['name']!r} (listed there: "
                            f"{', '.join(listed) or 'none'}); the QPUs differ per instance — pick one with instance=NAME")
    return service, chosen


def ibm_instance_for_runs(requested, run_folders, root="datafiles"):
    """The instance name a script connects with for `run_folders`: the one recorded in their run_info.json (the run's own
    record always wins: an explicit `requested` that differs exits with code 2, so a run is never read or continued on
    another instance), else `requested` (None: connect_ibm()'s default). Runs recorded on different instances
    exit with code 2 (narrow the runs with layout= / date=)."""
    from source.core.records.run_info import recorded_instance
    recorded = {n for n in (recorded_instance(f, root)[0] for f in run_folders) if n}
    if len(recorded) > 1:
        exit_with_error(f"these runs were made on different IBM instances ({', '.join(sorted(recorded))}); "
                        f"narrow them with layout= or date=")
    if recorded:
        (name,) = recorded
        if requested is not None and requested != name:
            exit_with_error(f"the run was made on the IBM instance {name!r}; instance={requested} would not find its jobs")
        return name
    return requested


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
