"""
test_24_ibm_batch_and_optimized_layout.py
=========================================
Unit tests (no simulator run, no QPU, no network, no qiskit_ibm_runtime needed):

  - providers.ibm_batch: opens one Batch on the backend (with max_time when given), yields a SamplerV2 bound to
    that batch, and closes the batch when the block ends, also when the block raises (a fake qiskit_ibm_runtime
    module stands in for the real one);
  - bacq_ps_qbcs.resolve_physical_qubits with layout=optimized: when the QPU has no SWAP-free placement of the
    circuits, the run is refused with a ValueError (resolve_layout_or_exit turns it into one line and exit code 2)
    before anything is recorded; a layout that exists, and a forced list, still resolve;
  - the VLQ calibration source saved with a layout carries no absolute path;
  - the run_local_simulation driver ends cleanly (one line, exit code 2) on an unusable layout, through resolve_layout_or_exit.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import types
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import CZGate, Measure, RGate
from qiskit.transpiler import InstructionProperties, Target

from source.core.qpu.providers import ibm_batch


# ── ibm_batch ────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_runtime(monkeypatch):
    events = []

    class Batch:
        def __init__(self, backend=None, max_time=None):
            self.backend, self.max_time, self.closed = backend, max_time, False
            events.append(self)
        def close(self):
            self.closed = True

    class SamplerV2:
        def __init__(self, mode=None):
            self.mode = mode

    mod = types.ModuleType("qiskit_ibm_runtime")
    mod.Batch, mod.SamplerV2 = Batch, SamplerV2
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", mod)
    return events


def test_ibm_batch_yields_sampler_on_the_batch_and_closes(fake_runtime):
    with ibm_batch("backend-x", max_time=1800) as sampler:
        (batch,) = fake_runtime
        assert sampler.mode is batch and batch.backend == "backend-x" and batch.max_time == 1800
        assert not batch.closed
    assert batch.closed


def test_ibm_batch_default_max_time_is_none(fake_runtime):
    with ibm_batch("b"):
        pass
    assert fake_runtime[0].max_time is None and fake_runtime[0].closed


def test_ibm_batch_closes_when_the_body_raises(fake_runtime):
    with pytest.raises(RuntimeError, match="boom"):
        with ibm_batch("b"):
            raise RuntimeError("boom")
    assert len(fake_runtime) == 1 and fake_runtime[0].closed


# ── layout=optimized on a QPU without a SWAP-free placement ─────────────────

def _chain_target(n=3):
    t = Target(num_qubits=n)
    t.add_instruction(RGate(Parameter("a"), Parameter("b")), {(q,): InstructionProperties(error=1e-3) for q in range(n)}, name="r")
    t.add_instruction(Measure(), {(q,): InstructionProperties(error=1e-2) for q in range(n)})
    t.add_instruction(CZGate(), {e: InstructionProperties(error=1e-2) for a in range(n - 1) for e in ((a, a + 1), (a + 1, a))})
    return t


def _circuit(triangle):
    qc = QuantumCircuit(3)
    qc.cz(0, 1)
    qc.cz(1, 2)
    if triangle:   # a triangle of interactions does not fit a chain without SWAPs
        qc.cz(0, 2)
    return qc


def _resolve(monkeypatch, tmp_path, circuit, layout=None, physical_qubits=None):
    from source.core import qbcs as Q
    from source.core.qpu import layout_selection as LS
    monkeypatch.setattr(Q, "get_run_subfolder", lambda cfg, sub, *a: str(tmp_path / sub))
    monkeypatch.setattr(LS, "transpiled_1q_counts", lambda *a, **k: {})   # no transpilation: the target is not a full backend
    me = SimpleNamespace(CONFIG={}, _load_raw_chunk_qiskit=lambda nc: [circuit])
    backend = SimpleNamespace(name="fake-chain", target=_chain_target())
    from source.core.qpu.calibration_input import resolve_layout_or_exit
    return resolve_layout_or_exit(SimpleNamespace(resolve_physical_qubits=lambda b, **kw: Q.bacq_ps_qbcs.resolve_physical_qubits(me, b, **kw)),
                                  backend, layout=layout, physical_qubits=physical_qubits)


def test_optimized_without_placement_exits_2_with_one_line(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        _resolve(monkeypatch, tmp_path, _circuit(triangle=True), layout=SimpleNamespace(mode="optimized", qubits=None))
    assert e.value.code == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "SWAP" in err[0] and "fake-chain" in err[0] and "layout=optimized" in err[0]
    assert err[0].endswith("Pin the qubits explicitly with layout=a,b,c (three distinct qubits).")
    assert not (tmp_path / "transpiled_circuits" / "physical_qubits.json").exists()   # nothing recorded


def test_optimized_with_placement_resolves(monkeypatch, tmp_path):
    qubits = _resolve(monkeypatch, tmp_path, _circuit(triangle=False), layout=SimpleNamespace(mode="optimized", qubits=None))
    assert sorted(qubits) == [0, 1, 2] and (tmp_path / "transpiled_circuits" / "physical_qubits.json").exists()


def test_forced_list_still_resolves(monkeypatch, tmp_path):
    qubits = _resolve(monkeypatch, tmp_path, _circuit(triangle=False), layout=SimpleNamespace(mode="list", qubits=[2, 1, 0]))
    assert qubits == [2, 1, 0]


# ── the refusal fires only for the true "no SWAP-free placement" case ───────

def test_refusal_not_raised_when_placements_exist_but_all_lack_calibration(monkeypatch, tmp_path):
    """Placements exist but every one is dropped for a missing calibration entry: method no_scored_placement,
    not no_swap_free_placement, so the run is not refused (it stays unpinned, as before the refusal existed)."""
    from source.core.qpu.layout_selection import select_physical_qubits
    t = _chain_target()
    t["cz"][(0, 1)] = InstructionProperties(error=None)   # a missing entry on every coupler the circuit could use
    t["cz"][(1, 0)] = InstructionProperties(error=None)
    t["cz"][(1, 2)] = InstructionProperties(error=None)
    t["cz"][(2, 1)] = InstructionProperties(error=None)
    rec = select_physical_qubits(_circuit(triangle=False), t, {})
    assert rec["physical_qubits"] is None and rec["method"] == "no_scored_placement"
    assert select_physical_qubits(_circuit(triangle=True), _chain_target(), {})["method"] == "no_swap_free_placement"

    from source.core import qbcs as Q
    from source.core.qpu import layout_selection as LS
    monkeypatch.setattr(Q, "get_run_subfolder", lambda cfg, sub, *a: str(tmp_path / sub))
    monkeypatch.setattr(LS, "transpiled_1q_counts", lambda *a, **k: {})
    me = SimpleNamespace(CONFIG={}, _load_raw_chunk_qiskit=lambda nc: [_circuit(triangle=False)])
    backend = SimpleNamespace(name="fake-chain", target=t)
    out = Q.bacq_ps_qbcs.resolve_physical_qubits(me, backend, layout=SimpleNamespace(mode="optimized", qubits=None))
    assert out is None   # unpinned, not refused


def test_refusal_not_raised_with_layout_backend_vlq_cal_or_forced_qubits(monkeypatch, tmp_path):
    from source.core import qbcs as Q
    from source.core.qpu import layout_selection as LS
    monkeypatch.setattr(Q, "get_run_subfolder", lambda cfg, sub, *a: str(tmp_path / sub))
    monkeypatch.setattr(LS, "transpiled_1q_counts", lambda *a, **k: {})
    tri = _circuit(triangle=True)
    me = SimpleNamespace(CONFIG={}, _load_raw_chunk_qiskit=lambda nc: [tri])
    optimized = SimpleNamespace(mode="optimized", qubits=None)
    # layout_backend given: a different (ValueError) message, about the calibration file and the QPU, not the SWAP refusal
    backend = SimpleNamespace(name="fake-chain", target=_chain_target())
    with pytest.raises(ValueError) as e:
        Q.bacq_ps_qbcs.resolve_physical_qubits(me, backend, layout=optimized, layout_backend=backend, calibration_source="x.json")
    assert "calibration file" in str(e.value) and "without SWAPs" not in str(e.value)
    # forced list: no refusal (the list is checked by choose_layout, not by the optimized refusal)
    try:
        Q.bacq_ps_qbcs.resolve_physical_qubits(me, backend, layout=SimpleNamespace(mode="list", qubits=[0, 1, 2]))
    except ValueError as err:
        assert "without SWAPs" not in str(err)
    # other modes
    for mode in ("lowest", "random"):
        try:
            Q.bacq_ps_qbcs.resolve_physical_qubits(me, backend, layout=SimpleNamespace(mode=mode, qubits=None))
        except ValueError as err:
            assert "without SWAPs" not in str(err)


def test_refusal_not_raised_with_vlq_calibration(monkeypatch, tmp_path):
    """A backend carrying vlq_calibration (the VLQ scripts) is laid out by VLQ's own chooser, never by the SWAP refusal."""
    from source.core import qbcs as Q
    from source.core.qpu import layout_selection as LS
    monkeypatch.setattr(Q, "get_run_subfolder", lambda cfg, sub, *a: str(tmp_path / sub))
    monkeypatch.setattr(LS, "transpiled_1q_counts", lambda *a, **k: {})
    seen = {}
    monkeypatch.setattr(Q.VLQ, "representative_roles", lambda rep, cal, be: "roles")
    def fake_choose(mode, roles, cal, forced, **kw):
        seen["called"] = (mode, roles)
        return dict(physical_qubits=None, method="no_swap_free_placement", layout_mode=mode, log_success=None)
    monkeypatch.setattr(Q.VLQ, "choose_vlq_layout", fake_choose)
    me = SimpleNamespace(CONFIG={}, _load_raw_chunk_qiskit=lambda nc: [_circuit(triangle=True)])
    backend = SimpleNamespace(name="fake-vlq", target=_chain_target(), vlq_calibration={"x": 1})
    out = Q.bacq_ps_qbcs.resolve_physical_qubits(me, backend, layout=SimpleNamespace(mode="optimized", qubits=None))
    assert seen["called"] == ("optimized", "roles") and out is None   # reached VLQ's chooser, no ValueError


def test_vlq_calibration_source_has_no_absolute_path(monkeypatch, tmp_path):
    import json
    from source.core.qpu import iqm_vlq
    cal = {"dynamic_architecture": {}, "metrics": {}, "calibration_set": {}, "calibration_set_id": "abc"}
    f = tmp_path / "outside" / "vlq_calibration.json"
    f.parent.mkdir()
    f.write_text(json.dumps(cal), encoding="utf-8")
    monkeypatch.setattr(iqm_vlq, "snapshot_from_calibration", lambda c: {})
    _, source = iqm_vlq.use_run_calibration(tmp_path / "run", calibration_file=str(f))
    assert source == "vlq_calibration.json"


def test_local_simulation_driver_exits_cleanly_on_unusable_layout(monkeypatch, tmp_path, capsys):
    """The driver runs resolve_layout_or_exit on the snapshot backend: an unusable layout is one line, exit code 2."""
    from source.core import qbcs as Q
    from source.core.runner_support import run_local_simulation as R

    class FakeQbcs:
        CONFIG = {}
        def __init__(self, **kw):
            pass
        def build(self, **kw):
            pass
        def resolve_physical_qubits(self, backend, **kw):
            raise ValueError("layout=optimized: the QPU (fake) has no placement of the circuits without SWAPs")

    monkeypatch.setattr(Q, "bacq_ps_qbcs", FakeQbcs)
    bench = SimpleNamespace(sample_size=1, list_times=[1], nshots=10)
    with pytest.raises(SystemExit) as e:
        R.run_local_simulation(run_folder="tests/test24_x", bench=bench, lnn=True, backend_label="fake",
                               noisy_setup=lambda: ("snap.json", SimpleNamespace(name="fake"), object()))
    assert e.value.code == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("❌") and "without SWAPs" in err[0]


def test_local_simulation_driver_calls_resolve_layout_or_exit():
    import inspect
    from source.core.runner_support import run_local_simulation as R
    assert "resolve_layout_or_exit(qbcs, snapshot_backend" in inspect.getsource(R)
