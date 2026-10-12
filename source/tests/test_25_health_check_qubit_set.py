"""
test_25_health_check_qubit_set.py
=================================
Unit tests (no QPU, a fake linear-chain backend) for the check submit_health_check() makes of an explicit
qubit_set before transpiling: a set that is not usable or not coupled as the circuit needs ends the script
with one ❌ line and exit code 2 (exit_with_error), instead of an AssertionError after routing. A run's own
qubits (qubit_set=None) are checked the same way when the run has a physical_qubits.json. IQM VLQ is checked like the rest: the qaas backend's target is an effective
all-to-all CZ graph over its 24 circuit qubits (the resonator is not in it).
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import shutil
from pathlib import Path

import pytest
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator

from source.core import health_check as HC
from source.tests.testing_functions import run_name_for_test

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Reached(Exception):
    """Raised by the fake transpile step: the qubit_set check let the call through."""


@pytest.fixture
def run_folder(monkeypatch):
    folder = run_name_for_test("test25", "real-data")
    path = PROJECT_ROOT / "datafiles" / folder
    shutil.rmtree(path, ignore_errors=True)

    def fake_isa(*args, **kwargs):
        raise _Reached()
    monkeypatch.setattr(HC, "_health_check_isa", fake_isa)
    yield folder
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")
def chain():
    """A 5-qubit linear chain 0-1-2-3-4."""
    return GenericBackendV2(num_qubits=5, coupling_map=[[i, i + 1] for i in range(4)] + [[i + 1, i] for i in range(4)], seed=1)


def _submit(run_folder, backend, qubit_set, service_name="IQM_Resonance"):
    return HC.submit_health_check(run_folder=run_folder, backend=backend, nshots=10, service_name=service_name,
                                  reps=1, qubit_set=qubit_set)


def _exits_with_one_line(run_folder, backend, qubit_set, capsys, **kw):
    with pytest.raises(SystemExit) as e:
        _submit(run_folder, backend, qubit_set, **kw)
    assert e.value.code == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("❌")
    return err[0]


def test_a_coupled_qubit_set_passes_the_check(run_folder, chain):
    with pytest.raises(_Reached):
        _submit(run_folder, chain, [1, 2, 3])


def test_an_uncoupled_qubit_set_exits_with_one_line_before_transpiling(run_folder, chain, capsys):
    line = _exits_with_one_line(run_folder, chain, [0, 2, 4], capsys)
    assert "[0, 2, 4]" in line and "coupler" in line


def test_a_repeated_qubit_exits_with_one_line(run_folder, chain, capsys):
    assert "repeats" in _exits_with_one_line(run_folder, chain, [1, 2, 1], capsys)


def test_an_unusable_qubit_exits_with_one_line(run_folder, chain, capsys):
    assert "not usable" in _exits_with_one_line(run_folder, chain, [2, 3, 9], capsys)


def test_a_qubit_set_of_the_wrong_length_exits_with_one_line(run_folder, chain, capsys):
    _exits_with_one_line(run_folder, chain, [1, 2], capsys)


def test_no_qubit_set_is_not_checked(run_folder, chain):
    with pytest.raises(_Reached):
        _submit(run_folder, chain, None)


@pytest.fixture(scope="module")
def vlq_like():
    """A VLQ-like target: 24 circuit qubits, effective all-to-all CZ (no resonator qubit, no MOVE)."""
    n = 24
    return GenericBackendV2(num_qubits=n, coupling_map=[[a, b] for a in range(n) for b in range(n) if a != b], seed=1)


def test_the_vlq_service_checks_a_valid_set_and_lets_it_through(run_folder, vlq_like):
    with pytest.raises(_Reached):
        _submit(run_folder, vlq_like, [0, 2, 4], service_name="QAAS_IT4I")


def test_the_vlq_service_rejects_a_qubit_that_is_not_on_the_target(run_folder, vlq_like, capsys):
    assert "not usable" in _exits_with_one_line(run_folder, vlq_like, [0, 2, 24], capsys, service_name="QAAS_IT4I")


def test_a_chain_target_is_checked_whatever_the_service_name(run_folder, chain, capsys):
    _exits_with_one_line(run_folder, chain, [0, 2, 4], capsys, service_name="QAAS_IT4I")


def test_a_local_run_without_a_layout_backend_ignores_the_qubit_set(run_folder):
    with pytest.raises(_Reached):
        HC.submit_health_check(run_folder=run_folder, backend=AerSimulator(), nshots=10, service_name="local",
                               reps=1, qubit_set=[0, 2, 4])


def test_the_message_names_the_backend_the_script_gives_not_the_backend_object(run_folder, chain, capsys):
    with pytest.raises(SystemExit):
        HC.submit_health_check(run_folder=run_folder, backend=chain, nshots=10, service_name="QAAS_IT4I",
                               reps=1, qubit_set=[0, 2, 4], backend_name="VLQ")
    line = capsys.readouterr().err
    assert "is not usable on VLQ:" in line
    with pytest.raises(SystemExit):   # without a name: the backend object's own
        _submit(run_folder, chain, [0, 2, 4])
    assert f"is not usable on {chain.name}:" in capsys.readouterr().err
