"""
iqm_vlq.py
IT4Innovations' VLQ (IQM "star24": 24 qubits QB1..QB24 around one computational
resonator COMPR1), reached through LEXIS/HEAppE with IT4I's `qaas` client (package qheappe_client). Everything
VLQ-specific the pipeline needs, kept out of qbcs.py:

  lexis_backend()          the qaas backend, logging in with the saved LEXIS refresh token
  retrieve_job(), job_status(), cancel_job(), wait_for_result()   an existing job, through the backend's QClient
  fetch_calibration()      a calibration set + its quality metrics from IT4I's public API
  qubit_metrics(), representative_roles(), select_vlq_qubits(), snapshot_from_calibration()   from iqm_star.py
  vlq_collect_timing()     QPU / queue time of a finished job (source/core/records/job_timing.py terms)

Qubit selection, the scoring and the effective all-to-all snapshot are shared with IQM Resonance's Sirius and live in
source/core/qpu/iqm_star.py (select_vlq_qubits() is an alias of iqm_star.select_star_qubits(); choose_vlq_layout() and snapshot_from_calibration()
are wrappers that pass the "VLQ" label).

Imports qheappe_client (qaas) / py4lexis only inside lexis_backend(), retrieve_job() and lexis_session(), so the module loads in every venv
(the local simulation uses the calibration functions from local_sim.venv or iqm_vlq.venv).
"""

import json
import math
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from source.core.common import PROJECT_ROOT, exit_with_error, rel_path, scrub_paths
from source.core.qpu.iqm_star import (PRX_PER_CLIFFORD, _fields, _qubit_logs, gate_roles, offline_backend,   # noqa: F401 — re-exported
                                      prx_fidelity, qubit_metrics, qubit_names, representative_roles,
                                      resonator_name, RESONATOR)
from source.core.qpu.iqm_star import choose_star_layout
from source.core.qpu.iqm_star import select_star_qubits as select_vlq_qubits          # noqa: F401
from source.core.qpu.iqm_star import snapshot_from_calibration as _star_snapshot
from source.core.qpu.tokens import token_path
from source.core.records.job_timing import now_utc_iso

TOKEN_FILE        = token_path("iqm_vlq_token")             # tokens/ is git-ignored
LEXIS_LOG_NAME    = "iqm_vlq_authentication_lexis_logs.log"   # py4lexis' own log, in the run folder (git-ignored)
PROJECT_INFO_FILE = token_path("iqm_vlq_project_info")   # PROJECT = … / RESOURCE = … lines (tokens/README.md)
CALIBRATION_API   = "https://extranet.it4i.cz/api/v1/calibration-sets/star24"
STATUS_URL        = "https://quantum.heappe.it4i.cz/vlq-status.json"
BACKEND_NAME      = "iqm_vlq"               # kept for reference; nothing uses it
WALLTIME_LIMIT_S  = 7200                    # HEAppE walltime of one job: the most IT4I accepts (6 h was refused)
CONNECTION_LOG_NAME = "iqm_vlq_connection.log"   # every network call to IT4I / HEAppE, in the run folder

# Time limits (s) of the network calls. The qaas client sets only per-request ones: a HEAppE request that is never
# answered (seen 2026-09-29: JobManagement/CurrentInfoForJob) otherwise waits forever.
TIMEOUTS = {
    "public status":        30,
    "calibration download": 120,
    "LEXIS login":          120,    # silent login with the saved refresh token
    "LEXIS login (browser)": 1800,  # the user approves it in a browser page
    "access token":         60,
    "get_backend":          300,    # QClient login + machine info (1-6 s measured with qaas 1.0.0)
    "retrieve_job":         120,
    "job status":           120,
    "job cancel":           120,
    "result download":      900,    # the job has finished; fetching a 5-chunk job's counts
    "job submission":       3600,   # upload of up to 350 circuits (one submit call). After a timeout the upload may still
                                    # complete: check check_real_qpu_job_status_iqm_vlq.py / LEXIS for a job
                                    # that has no record before submitting again
}


# ── Connection log: every call to IT4I / HEAppE, with a time limit ───────────
class VLQTimeout(RuntimeError):
    """A VLQ / HEAppE call got no answer within its time limit (TIMEOUTS)."""


_connection_log = None     # Path of the run's connection log (set_connection_log); None: terminal only
_pending_events = []       # events before the log was set (e.g. the public status check), written into it then


def set_connection_log(path):
    """Also append the connection events to `path` (the run folder's CONNECTION_LOG_NAME);
    events logged before are written into it first."""
    global _connection_log
    _connection_log = Path(path) if path is not None else None
    if _connection_log is not None:
        _connection_log.parent.mkdir(parents=True, exist_ok=True)
        _append(_pending_events)
        _pending_events.clear()


def _append(lines):
    try:
        with open(_connection_log, "a", encoding="utf-8") as f:
            f.writelines(l + "\n" for l in lines)
    except OSError:
        pass


def conn_log(msg):
    """One timestamped connection event, to the terminal and the run's connection log."""
    now_utc = datetime.now(timezone.utc)
    line = f"[{now_utc_iso(precise=True)}] [VLQ connection] {msg}"      # log file stays UTC
    print(f"  [{now_utc.astimezone().strftime('%Y-%m-%d %H:%M:%S')}] [VLQ connection] {msg}", flush=True)   # terminal: local time
    if _connection_log is not None:
        _append([line])
    else:
        _pending_events.append(line)


def call(stage, fn, detail="", timeout_s=None):
    """Run the network call fn() as a logged stage (START / OK / FAIL / TIMEOUT, with its
    duration). It runs in a daemon thread and is abandoned after timeout_s (default
    TIMEOUTS[stage]); a timeout logs where the call was waiting and raises VLQTimeout."""
    import threading, time, traceback
    timeout_s = timeout_s or TIMEOUTS.get(stage, 300)
    label = f"{stage}{f' ({detail})' if detail else ''}"
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:          # noqa: BLE001 — reported to the caller
            box["error"] = e

    conn_log(f"START   {label} (limit {timeout_s} s)")
    t0 = time.perf_counter()
    th = threading.Thread(target=run, name=f"vlq:{stage}", daemon=True)
    th.start()
    th.join(timeout_s)
    dt = time.perf_counter() - t0
    if th.is_alive():
        frame = sys._current_frames().get(th.ident)
        frames = traceback.extract_stack(frame)[-6:] if frame is not None else []
        where  = "\n".join(f"  {rel_path(fs.filename) if Path(fs.filename).is_relative_to(PROJECT_ROOT) else Path(fs.filename).name}"
                           f":{fs.lineno} in {fs.name}" + (f"\n    {fs.line}" if fs.line else "")
                           for fs in frames) or "(no stack)"
        conn_log(f"TIMEOUT {label}: no answer after {dt:.0f} s — IT4I / HEAppE not responding. "
                 f"Waiting in:\n{where.rstrip()}")
        raise VLQTimeout(f"VLQ {label}: no answer after {dt:.0f} s (IT4I / HEAppE not responding)")
    if "error" in box:
        e = box["error"]
        conn_log(f"FAIL    {label} after {dt:.1f} s: " + scrub_paths(f"{type(e).__name__}: {e}"))
        raise e
    conn_log(f"OK      {label} in {dt:.1f} s")
    return box.get("value")


# ── LEXIS login + backend ─────────────────────────────────────────────────────
def read_project_info(path=PROJECT_INFO_FILE):
    """{'PROJECT': ..., 'RESOURCE': ...} from the 'KEY = VALUE' lines of tokens/iqm_vlq_project_info
    (the LEXIS project and computational resource of the VLQ allocation). A missing file or key ends
    the script with a one-line explanation (exit code 2)."""
    path = Path(path)
    how  = (f"{rel_path(path)} must hold two lines, 'PROJECT = <LEXIS project short name, e.g. EU-XX-YYY>' and "
            f"'RESOURCE = <LEXIS computational resource name>' (see tokens/README.md)")
    if not path.exists():
        exit_with_error(f"No LEXIS project file: {how}.")
    info = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    if not info.get("PROJECT") or not info.get("RESOURCE"):
        exit_with_error(f"Incomplete LEXIS project file: {how}.")
    return info


def lexis_session(token_file=TOKEN_FILE, allow_browser=True, log_file=None):
    """LexisSession from the saved refresh token; a browser login only if there is none or
    LEXIS rejects it. The refresh token rotates on every use, so the newest is saved back.
    log_file: where py4lexis logs (it creates the file at every login, and writes to it only
    for a browser login); default <project root>/LEXIS_LOG_NAME. The scripts pass the run
    folder's."""
    log_file = Path(log_file or PROJECT_ROOT / LEXIS_LOG_NAME)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    set_connection_log(log_file.parent / CONNECTION_LOG_NAME)   # the connection log goes next to it
    from py4lexis.session import LexisSession
    token_file = Path(token_file)
    session = None
    if token_file.exists():
        try:
            session = call("LEXIS login", lambda: LexisSession(
                login_method="token", refresh_token=token_file.read_text(encoding="utf-8").strip(),
                reraise_exceptions=True, suppress_print=True, log_file=str(log_file)), "saved token")
        except VLQTimeout:
            raise
        except Exception as e:
            print(f"  [VLQ] Saved LEXIS token rejected ({type(e).__name__}) — a browser login is needed.")
    if session is None:
        if not allow_browser:
            raise RuntimeError(f"No valid LEXIS token in {token_file} and browser login not allowed.")
        print("  [VLQ] 🔑 LEXIS login needed: approve it in the browser page that opens now "
              "(MyAccessID). The token is then saved and reused.", flush=True)
        session = call("LEXIS login (browser)", lambda: LexisSession(login_method="url", offline_access=True,
                       reraise_exceptions=True, log_file=str(log_file)), "browser")
    token_file.write_text(session.get_refresh_token(), encoding="utf-8")
    return session


def lexis_backend(project=None, resource=None, token_file=TOKEN_FILE, allow_browser=True, log_file=None):
    """The qaas VLQ backend (QBackendIQM, qheappe_client), needed to transpile for VLQ, to submit, and to handle existing
    jobs (retrieve_job() below). project/resource default to tokens/iqm_vlq_project_info.

    qaas names its backend "QBackendIQM" and gives it no backend_name; the pipeline's records, logs and
    is_vlq_backend() expect the machine name (VLQ), so both are set to the name of the QClient's backend."""
    from qheappe_client import QProvider
    info     = read_project_info() if project is None or resource is None else {}
    project  = project or info["PROJECT"]
    resource = resource or info["RESOURCE"]
    session  = lexis_session(token_file, allow_browser, log_file)
    token    = call("access token", session.get_access_token)
    provider = QProvider(project, token)
    backend  = call("get_backend", lambda: provider.get_backend(resource), resource)
    backend.name = backend.backend_name = str(backend._qclient.active_backend or resource)
    return backend


def retrieve_job(backend, job_id):
    """The qaas QJob of an existing HEAppE job, logged and time-limited. It holds only the job id and the backend's
    QClient: status and timing are asked through that client (job_status(), vlq_collect_timing())."""
    from qheappe_client import QJob
    return call("retrieve_job", lambda: QJob(backend, int(job_id)), f"HEAppE job {job_id}")


# HEAppE job states (QClient.get_job_info()["State"]) as the qiskit JobStatus names the pipeline maps (providers.job_state_of_status)
_HEAPPE_STATES = {1: "INITIALIZING", 2: "QUEUED", 4: "QUEUED", 8: "RUNNING", 16: "DONE", 32: "ERROR", 64: "CANCELLED",
                  128: "QUEUED"}


def job_status(job, job_id=""):
    """The job's state from HEAppE, as a JobStatus name (DONE, RUNNING, QUEUED, ERROR, CANCELLED, ...), logged and
    time-limited. qaas' QJob.status() only returns the state it last saw, so HEAppE is asked directly."""
    def ask():
        state = job._qclient.get_job_info(int(job.job_id))["State"]
        return _HEAPPE_STATES.get(state, f"UNKNOWN({state})")
    return call("job status", ask, f"HEAppE job {job_id}")


def cancel_job(job, job_id=""):
    """Ask HEAppE to cancel the job (True when it accepted the request), logged and time-limited. QJob.cancel() is not
    used: in qaas 1.0.0 it never sends the request (it does not await the client's coroutine)."""
    qclient = job._qclient
    return call("job cancel", lambda: qclient._run(qclient.cancel_job(int(job.job_id))), f"HEAppE job {job_id}")


def run(backend, circuits, detail="", walltime_limit=None, **kw):
    """backend.run() - the upload of a job to HEAppE -, logged and time-limited. walltime_limit: HEAppE walltime in s."""
    if walltime_limit is not None:
        kw["walltime_limit_secs"] = walltime_limit
    return call("job submission", lambda: backend.run(circuits, **kw), detail)


NO_EXECUTED_CIRCUITS = "IQM VLQ (qaas 1.0) does not return the executed circuits"   # circuit_check text: HEAppE keeps no payloads


def attach_calibration(backend, calibration_set_id=None):
    """Download the calibration the backend runs with (its architecture's calibration set,
    else the current default) and attach it as backend.vlq_calibration, which
    bacq_ps_qbcs.resolve_physical_qubits() uses to pick the run's qubits. Returns it."""
    live_id = backend_calibration_set_id(backend)
    cal_id  = calibration_set_id or live_id or "default"
    cal     = fetch_calibration(cal_id)
    if live_id and str(cal.get("calibration_set_id")) != live_id:
        # IT4I's public API served another set than the one the backend runs with: grade on the backend's own, or stop
        print(f"  ⚠️  [Calibration] IT4I's calibration set {cal.get('calibration_set_id')} is not the backend's {live_id} "
              f"— downloading the backend's set.")
        try:
            cal = fetch_calibration(live_id)
        except Exception as e:
            exit_with_error(f"cannot fetch the backend's calibration set {live_id} ({type(e).__name__}: {e}); "
                            f"the layout would be graded on another calibration than the QPU runs with")
        if str(cal.get("calibration_set_id")) != live_id:
            exit_with_error(f"IT4I's API returned calibration set {cal.get('calibration_set_id')}, not the backend's {live_id}")
    backend.vlq_calibration = cal
    return cal


def is_vlq_backend(backend):
    """True for the qaas VLQ backend (its backend_name is 'VLQ' or 'VLQ-...')."""
    name = str(getattr(backend, "backend_name", "") or "")
    return name == "VLQ" or name.startswith("VLQ-")


def backend_calibration_set_id(backend):
    arch = getattr(backend, "architecture", None)
    cal  = getattr(arch, "calibration_set_id", None) if arch is not None else None
    return None if cal is None else str(cal)


def qpu_status():
    """IT4I's public VLQ status JSON (QPU_status, QCCW_status, total_status, timestamp)."""
    return _get_json(STATUS_URL, "public status", 15)


# ── Calibration (public API, no authentication) ───────────────────────────────
def _get_json(url, what="calibration download", timeout=60):
    """The JSON at `url`, fetched through call() (logged and time-limited by the label `what`)."""
    def get():
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    return call(what, get, url.rsplit("/star24/", 1)[-1] if what == "calibration download" else "")


def fetch_calibration(calibration_set_id="default"):
    """The calibration set, its quality-metric set and the dynamic architecture, as one dict.
    calibration_set_id: a UUID (e.g. the backend's architecture.calibration_set_id) or 'default'."""
    base = f"{CALIBRATION_API}/{calibration_set_id}"
    cal  = {
        "source":              base,
        "downloaded_at":       now_utc_iso(),
        "calibration_set":     _get_json(base),
        "metrics":             _get_json(f"{base}/metrics"),
        "dynamic_architecture": _get_json(f"{base}/dynamic-quantum-architecture"),
    }
    cal["calibration_set_id"] = cal["dynamic_architecture"].get("calibration_set_id") \
        or cal["calibration_set"].get("observation_set_id")
    return cal


def choose_vlq_layout(mode, roles, cal, qubits=None, seed=None, label="VLQ", rep=None, backend=None):
    """iqm_star.choose_star_layout(), naming `label` (the machine; Sirius passes its own) in its messages."""
    return choose_star_layout(mode, roles, cal, qubits, seed, label=label, rep=rep, backend=backend)


def snapshot_from_calibration(cal):
    """The effective all-to-all calibration snapshot of VLQ (iqm_star.snapshot_from_calibration())."""
    return _star_snapshot(cal, "VLQ")


VLQ_CALIBRATION_FILENAME = "vlq_calibration.json"


def save_vlq_calibration(run_dir, cal):
    """Write a VLQ calibration into `run_dir` (created if needed) as vlq_calibration.json (IT4I's raw data, which
    the layout choice needs) and the calibration_snapshot.json built from it. Returns the snapshot's path."""
    from source.core.qpu.qpu_snapshot import SNAPSHOT_FILENAME
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / VLQ_CALIBRATION_FILENAME).write_text(json.dumps(cal), encoding="utf-8")
    snapshot_path = run_dir / SNAPSHOT_FILENAME
    snapshot_path.write_text(json.dumps(snapshot_from_calibration(cal), indent=2), encoding="utf-8")
    return snapshot_path


def use_run_calibration(run_dir, calibration_file=None, calibration_set="default"):
    """The calibration of a VLQ simulation run, saved in `run_dir` (a Path) as vlq_calibration.json and
    calibration_snapshot.json. From `calibration_file` (a run's vlq_calibration.json, or that run's
    calibration_snapshot.json with its vlq_calibration.json beside it) if given, else
    downloaded once from IT4I's API (`calibration_set`) and reused on a rerun. Problems the user can fix end
    the script through exit_with_error. Prints what was done. Returns (calibration dict, recorded source:
    the calibration file, project-relative or just its name when it lies outside the project; None without a file)."""
    from source.core.qpu.qpu_snapshot import SNAPSHOT_FILENAME, reuse_calibration_snapshot
    snapshot_path = run_dir / SNAPSHOT_FILENAME
    cal_path      = run_dir / VLQ_CALIBRATION_FILENAME
    cal_source    = None
    if calibration_file:
        # VLQ's layout choice needs IT4I's raw calibration (qubits, couplers, metrics), so the file must be a
        # vlq_calibration.json as a VLQ run saves it (datafiles/<run>/vlq_calibration.json).
        src = Path(calibration_file)
        if not src.is_absolute():
            src = PROJECT_ROOT / src   # a relative path is relative to the project root, as in calibration_input.load_calibration
        try:
            file_cal = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            exit_with_error(f"calibration_file {calibration_file}: cannot be read ({e})")
        from source.core.qpu.calibration_input import detect_format
        try:
            is_snapshot = detect_format(file_cal) == "snapshot"
        except ValueError:   # not a format of calibration_input (a raw VLQ file, or something else): checked below
            is_snapshot = False
        if is_snapshot:
            # A snapshot has no coupler data for the layout choice: use the raw file beside it.
            raw = src.parent / VLQ_CALIBRATION_FILENAME
            if not raw.exists():
                exit_with_error(f"calibration_file {calibration_file} is a calibration snapshot, which cannot give VLQ's "
                                f"layout: pass the run's {VLQ_CALIBRATION_FILENAME} (the one in the real run's folder).")
            src = raw
            try:
                file_cal = json.loads(src.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                exit_with_error(f"calibration_file {src}: cannot be read ({e})")
        if not (isinstance(file_cal, dict) and {"dynamic_architecture", "metrics", "calibration_set"} <= file_cal.keys()):
            exit_with_error(f"calibration_file {calibration_file} is not a VLQ calibration: VLQ needs the "
                            f"vlq_calibration.json a VLQ run saves in its folder (datafiles/<run>/vlq_calibration.json).")
        if cal_path.exists() and json.loads(cal_path.read_text(encoding="utf-8")).get("calibration_set_id") != file_cal.get("calibration_set_id"):
            exit_with_error(f"{rel_path(cal_path)} holds another calibration set — a run keeps one calibration. "
                            f"Use the same file, or a new run folder.")
        save_vlq_calibration(run_dir, file_cal)
        from source.core.qpu.calibration_input import source_name
        cal_source = source_name(src)   # project-relative, or just the file name for a file outside the project
        print(f"  [Snapshot] VLQ calibration set {file_cal['calibration_set_id']} from {cal_source} → {rel_path(snapshot_path)}")
    elif not (reuse_calibration_snapshot(snapshot_path) and cal_path.exists()):
        cal = fetch_calibration(calibration_set)
        save_vlq_calibration(run_dir, cal)
        print(f"  [Snapshot] VLQ calibration set {cal['calibration_set_id']} → {rel_path(snapshot_path)}")
    return json.loads(cal_path.read_text(encoding="utf-8")), cal_source


# ── Timing of a finished job ──────────────────────────────────────────────────
def _task_result(job):
    """The HEAppE task result of the job's (single) task, with IQM's server timeline: {status, runtime_ms,
    compilation, timeline: [{source, status, timestamp}], ...}. Asked once per job object; {} when unavailable."""
    cached = getattr(job, "_vlq_task_result", None)
    if cached is not None:
        return cached
    qclient = job._qclient
    try:
        task_id = qclient.get_job_info(int(job.job_id))["Tasks"][0]["Id"]
        res     = qclient._run(qclient.get_task_result(task_id))
        res     = res if isinstance(res, dict) else {}
    except Exception:
        return {}
    job._vlq_task_result = res
    return res


def _iqm_timeline(job):
    """IQM's server timeline of a finished qaas job: [{source, status, timestamp}], or []."""
    from source.core.records.job_timing import iso_utc, parse_utc
    try:
        return [{"source": t["source"], "status": t["status"], "timestamp": iso_utc(parse_utc(t["timestamp"]))}
                for t in _task_result(job).get("timeline", [])]
    except Exception:
        return []


def vlq_collect_timing(job, upload_end_at):
    """Timing of a finished VLQ job (after job.result()). Never raises; missing values are None.

    qpu_s        IQM Resonance-equivalent billed time: ceil(compilation + execution) of the job,
                 from IQM's server timeline (source 'iqm_equivalent_billed')
    qpu_exec_s   QPU execution only: IQM execution_started -> execution_ended
    queue_s      end of our upload -> IQM compilation_started: IT4I's HEAppE job system
                 (it4i_wait_s, upload end -> IQM 'created') plus IQM's own queue
                 (iqm_queue_s, 'created' -> 'compilation_started')."""
    from source.core.records.job_timing import seconds_between, iso_utc, parse_utc
    out = {"qpu_s": None, "qpu_s_source": "unavailable", "queue_s": None, "queue_s_source": "unavailable",
           "qpu_exec_s": None, "qpu_exec_s_source": "unavailable"}
    out["timeline"] = _iqm_timeline(job)
    first = {}
    for t in out["timeline"]:
        first.setdefault(t["status"], t["timestamp"])
    comp_s = seconds_between(first.get("compilation_started"), first.get("compilation_ended"))
    exec_s = seconds_between(first.get("execution_started"), first.get("execution_ended"))
    out["compilation_seconds"], out["execution_seconds"] = comp_s, exec_s
    if comp_s is not None and exec_s is not None:
        out["qpu_s"], out["qpu_s_source"] = float(math.ceil(comp_s + exec_s)), "iqm_equivalent_billed"

    if exec_s is not None:
        out["qpu_exec_s"], out["qpu_exec_s_source"] = exec_s, "execution_interval"

    res  = _task_result(job)
    out["iqm_job_id"]         = str(res["id"]) if res.get("id") else None
    cal  = (res.get("compilation") or {}).get("calibration_set_id")
    out["calibration_set_id"] = str(cal) if cal else None

    out["upload_end_at"]   = iso_utc(parse_utc(upload_end_at))
    out["created_at"]      = first.get("created")
    out["execution_start"] = first.get("execution_started")
    out["execution_end"]   = first.get("execution_ended")
    out["completed_at"]    = first.get("completed") or first.get("ready")
    out["queue_start_at"]  = out["upload_end_at"]
    out["queue_end_at"]    = first.get("compilation_started")
    out["it4i_wait_s"]     = seconds_between(out["upload_end_at"], first.get("created"))
    out["iqm_queue_s"]     = seconds_between(first.get("created"), first.get("compilation_started"))
    queue = seconds_between(out["upload_end_at"], first.get("compilation_started"))
    if queue is not None:
        out["queue_s"], out["queue_s_source"] = max(queue, 0.0), "compilation_started - end of upload (IT4I + IQM queue)"
    return out


def wait_for_result(job, job_id="", timeout_s=None):
    """job.result() of a finished job as a qiskit Result (counts per circuit in submission order), logged and
    time-limited (TIMEOUTS['result download']). Bit strings are requested left to right (key_order_left_to_right=True), the
    order the pipeline's reversal of the counts expects."""
    timeout_s = timeout_s or TIMEOUTS["result download"]
    return call("result download", lambda: job.result(qiskit_legacy=True, key_order_left_to_right=True), f"HEAppE job {job_id}",
                timeout_s=timeout_s + 30)


if __name__ == "__main__":   # quick look: python -m source.core.qpu.iqm_vlq [calibration_set_id]
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    cal = fetch_calibration(sys.argv[1] if len(sys.argv) > 1 else "default")
    for q, v in qubit_metrics(cal).items():
        print(q, {k: (round(x, 5) if isinstance(x, float) else x) for k, x in v.items()})
