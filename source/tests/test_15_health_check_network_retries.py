"""
test_15_health_check_network_retries.py
========================================
Unit tests (no real QPU, no real network — a fake backend/collect-service stands
in for the provider) for the health-check retry/exit-code handling in
source/core/health_check.py (require_functional_health_check, submit_post_health_check) and
source/core/qpu/remote_jobs.py (_is_transient_network_error):

  - _is_transient_network_error() recognizes connection/DNS/timeout-shaped errors
    and nothing else.
  - submit_health_check(accompanying=True) on an asynchronous service (IBM,
    IQM EQE1 (service MQSS_LRZ), IQM Resonance (service IQM_Resonance), IQM VLQ (service QAAS_IT4I)): a submission failure writes its own
    numbered 'failed' record (job_id=None, a `failure` message) and raises
    HealthCheckSubmitError — every attempt stays on disk, per the project
    owner's rule that every check must be recorded.
  - require_functional_health_check() (the PRE-check helper the submit scripts
    call): retries only transient errors, up to a bound; a degraded QPU is
    never retried; it never returns except with a functional record — it exits
    the process otherwise, with a distinct code for each of the three outcomes
    (EXIT_HC_DEGRADED / EXIT_HC_NOT_COLLECTED / EXIT_HC_NOT_SUBMITTED).
  - submit_post_health_check() (the POST-check helper): retries only the
    submission; never raises or exits, since the benchmark chunks are already
    on the QPU by then and submit.log must still be written either way.
  - AQT's synchronous health check (submit_health_check(service_name=
    "PCSS")) is untouched by all of the above: a failure propagates as
    the raw exception (not HealthCheckSubmitError) with no record written,
    except that an AQTTaskTimeout in an accompanying check writes a failed
    record.
  - health_check_verdict() is one rule for every QPU (P(011) above the threshold),
    print_health_verdict() prints the same lines for all, and a synchronous (AQT/local) standalone
    record carries P011 and the verdict.
  - collect_health_check(..., number=None) (the ordinary catch-up collection
    used by qbcs_real_collect_*.py) still skips 'failed' records without
    error, and plot_health_checks.py counts/labels them separately from
    checks that are merely still uncollected.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import json
import shutil
import socket
import subprocess
from source.core.runner_support.venvs import venv_python
from pathlib import Path

import pytest
from source.tests.testing_functions import run_name_for_test
from source.core.records.run_names import plots_dir as run_plots_dir
from qiskit_aer import AerSimulator

from source.core import health_check as HC
from source.core.qpu import remote_jobs as RJ
from source.core.records.health_check_records import (
    load_health_checks, EXIT_HC_DEGRADED, EXIT_HC_NOT_COLLECTED, EXIT_HC_NOT_SUBMITTED,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_path(folder):
    return PROJECT_ROOT / "datafiles" / folder


@pytest.fixture
def run_folder():
    """A throwaway datafiles/<folder> for one test, removed before and after (the code under
    test hardcodes datafiles/<run_folder> relative to the project root, so this can't use
    pytest's tmp_path)."""
    folder = run_name_for_test("test15", "real-data")
    path = _run_path(folder)
    shutil.rmtree(path, ignore_errors=True)
    yield folder
    shutil.rmtree(path, ignore_errors=True)


def _fake_backend(run_fn):
    """A real AerSimulator, so qiskit.transpile() works exactly as in a real health check,
    with .run() swapped out for a fake submission function that raises or returns a fake job."""
    backend = AerSimulator()
    backend.run = lambda *a, **kw: run_fn()
    return backend


class _FakeJob:
    def __init__(self, job_id="job-1"):
        self._id = job_id
    def job_id(self):
        return self._id


def _raise_dns_error():
    raise socket.gaierror("getaddrinfo failed")


# ── _is_transient_network_error ────────────────────────────────────────────────
def test_is_transient_network_error_recognizes_connection_shaped_errors_only():
    assert RJ._is_transient_network_error(socket.gaierror("getaddrinfo failed"))
    assert RJ._is_transient_network_error(ConnectionError("Connection reset by peer"))
    assert RJ._is_transient_network_error(TimeoutError("timed out"))
    assert RJ._is_transient_network_error(RuntimeError("Max retries exceeded with url: ..."))
    assert not RJ._is_transient_network_error(ValueError("not a network problem"))
    assert not RJ._is_transient_network_error(AssertionError("bad state"))
    # walks __cause__: a wrapper exception around a real network error is still transient
    try:
        try:
            raise socket.gaierror("getaddrinfo failed")
        except socket.gaierror as e:
            raise RuntimeError("wrapped") from e
    except RuntimeError as wrapped:
        assert RJ._is_transient_network_error(wrapped)


# ── submit_health_check(accompanying=True): submission failure on an async service ──
def test_submission_failure_writes_a_failed_record_and_raises_health_check_submit_error(run_folder):
    backend = _fake_backend(_raise_dns_error)
    with pytest.raises(RJ.HealthCheckSubmitError) as exc_info:
        HC.submit_health_check(run_folder=run_folder, backend=backend, nshots=100,
                              service_name="MQSS_LRZ", reps=1, accompanying=True, chunks=[0])
    checks = load_health_checks(_run_path(run_folder))
    assert len(checks) == 1
    rec = checks[0]
    assert rec["status"] == "failed" and rec["job_id"] is None and rec["verdict"] is None
    assert "failure" in rec and "getaddrinfo failed" in rec["failure"]
    assert exc_info.value.number == rec["number"]
    t = rec["timing"]                                  # the transpile and submit time of the attempt are kept
    assert t["transpile_s"] is not None and t["transpile_reused"] is False and t["submit_s"] >= 0


def test_synchronous_services_are_not_wrapped_standalone_path_is_not_wrapped(run_folder):
    # AQT/local (synchronous): the raw exception propagates, not HealthCheckSubmitError,
    # and nothing is written — unlike the async services above.
    backend = _fake_backend(_raise_dns_error)
    with pytest.raises(socket.gaierror):
        HC.submit_health_check(run_folder=run_folder, backend=backend, nshots=100,
                              service_name="PCSS", reps=1, accompanying=True, chunks=[0])
    assert load_health_checks(_run_path(run_folder)) == []

    # accompanying=False (standalone): also unwrapped, regardless of service_name.
    shutil.rmtree(_run_path(run_folder), ignore_errors=True)
    backend = _fake_backend(_raise_dns_error)
    with pytest.raises(socket.gaierror):
        HC.submit_health_check(run_folder=run_folder, backend=backend, nshots=100,
                              service_name="MQSS_LRZ", reps=1, accompanying=False)
    assert not (_run_path(run_folder) / "job_records" / "health_check.json").exists()


def test_the_verdict_is_the_same_rule_for_every_qpu(monkeypatch, capsys):
    # functional: P(011) above the threshold (health_check_verdict)
    from source.core.records import health_check_records as HR
    assert HR.health_check_verdict({"011": 51, "010": 49}) == "functional"
    assert HR.health_check_verdict({"011": 50, "010": 50}) == "degraded"            # not above the threshold
    assert HR.health_check_verdict({"011": 30, "000": 20, "010": 50}) == "degraded"
    assert HR.health_check_verdict({"011": 51, "010": 25, "000": 24}) == "functional"
    # the printed lines are the same for every provider
    status, p011 = HC.print_health_verdict({"011": 94, "010": 3, "100": 3})
    out = capsys.readouterr().out
    assert (status, p011) == ("functional", 0.94)
    assert "QPU functional — P(011) = 0.940  (94/100 shots)" in out and "Top counts: {'011': 94" in out


def test_the_synchronous_record_carries_p011_and_the_verdict(run_folder):
    # AQT's (and the simulator's) standalone check answers in the same call: its record has P011 and the verdict
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
    HC.submit_health_check(run_folder=run_folder, backend=AerSamplerV2(seed=0), nshots=100, service_name="local",
                           reps=1, accompanying=False)
    rec = json.loads((_run_path(run_folder) / "job_records" / "health_check.json").read_text(encoding="utf-8"))
    assert rec["verdict"] == "functional" and rec["P011"] > 0.9


def test_collecting_a_standalone_check_keeps_its_result_in_its_record(run_folder):
    # the async standalone check (IBM / IQM scripts) stores counts, P011 and the verdict in its own record,
    # like the accompanying checks; "status" stays the job state ("complete"); there is no processed_results/
    class GoodJob:
        def status(self): return "COMPLETED"
        def wait_for_final_state(self): pass
        def done(self): return True
        def result(self):
            class R:
                def get_counts(self): return {"110": 94, "011": 6}   # reversed to the circuit's bit order on collection
            return R()
    class GoodCollectService:
        def retrieve_job(self, job_id): return GoodJob()

    HC.submit_health_check(run_folder=run_folder, backend=_fake_backend(lambda: _FakeJob("job-standalone")),
                          nshots=100, service_name="IQM_Resonance", reps=1, accompanying=False)
    run_path = _run_path(run_folder)
    rec_path = run_path / "job_records" / "health_check.json"
    assert json.loads(rec_path.read_text(encoding="utf-8"))["status"] == "submitted"
    result = HC.collect_health_check(exp_results_folder_path=run_path, service=GoodCollectService(),
                                    service_name="IQM_Resonance", accompanying=False)
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    assert result == rec
    assert rec["status"] == "complete" and rec["verdict"] == "functional" and rec["P011"] == 0.94
    assert rec["counts"] == {"011": 94, "110": 6} and rec["threshold"] == 0.5 and rec["result_path"] is None
    assert not (run_path / "processed_results").exists()
    assert HC.collect_health_check(exp_results_folder_path=run_path, service=GoodCollectService(),
                                  service_name="IQM_Resonance", accompanying=False) == rec      # already collected


# ── require_functional_health_check(): the PRE-check helper the submit scripts call ──
def test_require_functional_health_check_exits_not_submitted_after_exhausting_retries(run_folder):
    calls = {"n": 0}
    def always_fail():
        calls["n"] += 1
        raise socket.gaierror("getaddrinfo failed")

    with pytest.raises(SystemExit) as exc_info:
        HC.require_functional_health_check(
            run_folder=run_folder, backend=_fake_backend(always_fail), service_name="MQSS_LRZ",
            collect_service=object(), nshots=100, reps=1, chunks=[0],
            max_attempts=2, backoff_s=(0, 0))
    assert exc_info.value.code == EXIT_HC_NOT_SUBMITTED
    assert calls["n"] == 2
    checks = load_health_checks(_run_path(run_folder))
    assert len(checks) == 2 and {c["number"] for c in checks} == {1, 2}   # each attempt its own record
    assert all(c["status"] == "failed" for c in checks)


def test_require_functional_health_check_exits_not_collected_after_exhausting_retries(run_folder):
    submits = {"n": 0}
    def ok_submit():
        submits["n"] += 1
        return _FakeJob(f"job-{submits['n']}")

    class FlakyCollectService:
        def __init__(self):
            self.n = 0
        def retrieve_job(self, job_id):
            self.n += 1
            raise ConnectionError("Connection reset by peer")

    collect_service = FlakyCollectService()
    with pytest.raises(SystemExit) as exc_info:
        HC.require_functional_health_check(
            run_folder=run_folder, backend=_fake_backend(ok_submit), service_name="IQM_Resonance",
            collect_service=collect_service, nshots=100, reps=1, chunks=[0],
            max_attempts=2, backoff_s=(0, 0))
    assert exc_info.value.code == EXIT_HC_NOT_COLLECTED
    assert submits["n"] == 2 and collect_service.n == 2
    checks = load_health_checks(_run_path(run_folder))
    # a collection retry submits a *new* check rather than re-polling the same job_id under a
    # new number (which would make qbcs_real_collect_*.py collect the same job twice) — the old,
    # still-'submitted' record is left for the ordinary collect script to pick up later.
    assert len(checks) == 2 and {c["number"] for c in checks} == {1, 2}
    assert all(c["status"] == "submitted" for c in checks)
    assert all("last_collect_error" in c for c in checks)


def test_require_functional_health_check_degraded_exits_immediately_without_retry(run_folder):
    class DegradedResult:
        def get_counts(self):
            return {"000": 100}   # never '011' -> P(011) = 0, below threshold
    class DegradedJob:
        def wait_for_final_state(self): pass
        def done(self): return True
        def result(self): return DegradedResult()
    class DegradedCollectService:
        def __init__(self):
            self.n = 0
        def retrieve_job(self, job_id):
            self.n += 1
            return DegradedJob()

    collect_service = DegradedCollectService()
    with pytest.raises(SystemExit) as exc_info:
        HC.require_functional_health_check(
            run_folder=run_folder, backend=_fake_backend(lambda: _FakeJob("job-degraded")),
            service_name="IQM_Resonance", collect_service=collect_service,
            nshots=100, reps=1, chunks=[0], max_attempts=5, backoff_s=(0, 0))
    assert exc_info.value.code == EXIT_HC_DEGRADED
    assert collect_service.n == 1   # a real (non-transient) verdict is never retried
    checks = load_health_checks(_run_path(run_folder))
    assert len(checks) == 1 and checks[0]["status"] == "complete" and checks[0]["verdict"] == "degraded"


def test_require_functional_health_check_returns_the_record_when_functional(run_folder):
    class GoodResult:
        def get_counts(self):
            return {"110": 100}   # reversed by health_check.py -> '011', P(011) = 1
    class GoodJob:
        def wait_for_final_state(self): pass
        def done(self): return True
        def result(self): return GoodResult()
    class GoodCollectService:
        def retrieve_job(self, job_id):
            return GoodJob()

    record = HC.require_functional_health_check(
        run_folder=run_folder, backend=_fake_backend(lambda: _FakeJob("job-good")),
        service_name="IQM_Resonance", collect_service=GoodCollectService(),
        nshots=100, reps=1, chunks=[0], max_attempts=3, backoff_s=(0, 0))
    assert record["verdict"] == "functional" and record["P011"] == 1.0


# ── submit_post_health_check(): the POST-check helper — never raises/exits ────────
def test_submit_post_health_check_reports_failure_without_raising_or_exiting(run_folder):
    ok, numbers = HC.submit_post_health_check(
        run_folder=run_folder, backend=_fake_backend(_raise_dns_error), service_name="MQSS_LRZ",
        nshots=100, reps=1, chunks=[0], max_attempts=2, backoff_s=(0, 0))
    assert ok is False and len(numbers) == 2
    checks = load_health_checks(_run_path(run_folder))
    assert len(checks) == 2 and all(c["status"] == "failed" for c in checks)


def test_submit_post_health_check_succeeds_after_a_transient_failure(run_folder):
    attempts = {"n": 0}
    def fail_then_ok():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise socket.gaierror("getaddrinfo failed")
        return _FakeJob("job-post-ok")

    ok, numbers = HC.submit_post_health_check(
        run_folder=run_folder, backend=_fake_backend(fail_then_ok), service_name="MQSS_LRZ",
        nshots=100, reps=1, chunks=[0], max_attempts=3, backoff_s=(0, 0))
    assert ok is True and len(numbers) == 2   # the failed attempt, then the one that succeeded
    checks = load_health_checks(_run_path(run_folder))
    assert checks[0]["status"] == "failed" and checks[1]["status"] == "submitted"


# ── collect_health_check(number=None) and plot_health_checks.py: a mixed run ────
def test_collect_and_plot_treat_failed_records_distinctly_from_uncollected(run_folder):
    # #1: failed at submission. #2: submitted, never collected. #3: submitted then collected.
    try:
        HC.submit_health_check(run_folder=run_folder, backend=_fake_backend(_raise_dns_error),
                              nshots=100, service_name="MQSS_LRZ", reps=1, accompanying=True, chunks=[0])
    except RJ.HealthCheckSubmitError:
        pass
    HC.submit_health_check(run_folder=run_folder, backend=_fake_backend(lambda: _FakeJob("job-uncollected")),
                          nshots=100, service_name="MQSS_LRZ", reps=1, accompanying=True, chunks=[0])

    class GoodResult:
        def get_counts(self):
            return {"110": 100}
    class GoodJob:
        def status(self): return "COMPLETED"          # checked by _remote_job_state (number=None path)
        def wait_for_final_state(self): pass
        def done(self): return True
        def result(self): return GoodResult()
    class GoodCollectService:
        def retrieve_job(self, job_id):
            return GoodJob()

    HC.submit_health_check(run_folder=run_folder, backend=_fake_backend(lambda: _FakeJob("job-good")),
                          nshots=100, service_name="IQM_Resonance", reps=1, accompanying=True, chunks=[0])
    run_path = _run_path(run_folder)
    last_number = max(c["number"] for c in load_health_checks(run_path))

    # number=None (the ordinary qbcs_real_collect_*.py path): must not raise on the failed
    # record (job_id=None) and must leave it alone; only the genuinely-submitted #3 gets collected.
    result = HC.collect_health_check(exp_results_folder_path=run_path, service=GoodCollectService(),
                                    service_name="IQM_Resonance", accompanying=True, number=None)
    assert result["number"] == last_number
    checks = {c["number"]: c for c in load_health_checks(run_path)}
    assert checks[1]["status"] == "failed"
    assert checks[2]["status"] == "submitted"        # untouched: IQM EQE1 (MQSS_LRZ), GoodCollectService doesn't apply to it
    assert checks[last_number]["status"] == "complete"

    # plot_health_checks.py: must run without raising and must separate 'failed' from
    # 'uncollected' in its JSON summary (source/analysis_scripts/plot_health_checks.py).
    from source.core.records.run_info import write_run_info
    write_run_info(config=dict(shots=100, samples=1, T_list=[0.2], dt=0.2, SPAM_correction=True, chunk_size=1),
                   run_folder_path=run_path, lnn=True)
    env = {**os.environ, "QPU_RUN_FOLDER": run_folder, "QPU_N_SAMPLES": ""}
    plots_dir = PROJECT_ROOT / run_plots_dir(run_folder)
    try:
        r = subprocess.run([venv_python("local_sim.venv", "test_15 (plot_health_checks.py)"), str(PROJECT_ROOT / "source" / "analysis_scripts" / "plot_health_checks.py")],
                           cwd=PROJECT_ROOT, env=env, capture_output=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0, r.stdout + r.stderr
        out = json.loads((run_path / "processed_results" / "health_checks.json").read_text(encoding="utf-8"))
        assert out["failed_health_checks"] == [1]
        assert set(out["failed_health_checks"]).issubset(set(out["uncollected_health_checks"]))
        assert 2 in out["uncollected_health_checks"] and 2 not in out["failed_health_checks"]
    finally:
        shutil.rmtree(plots_dir, ignore_errors=True)
