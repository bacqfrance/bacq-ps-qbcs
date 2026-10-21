"""
test_18_layout_args.py
======================
Unit tests (no simulator run, no QPU, no network) for the qubit-layout arguments of the submit / simulate /
collect scripts (source/core/runner_support/benchmark_args.py, source/core/qpu/layout_selection.py, source/core/qpu/calibration_input.py, the VLQ
layouts in source/core/qpu/iqm_vlq.py):

  - parse_layout_args: layout=optimized | random | lowest | 15,14,21 (also [15,14,21]), the default per kind of
    QPU (optimized; random without a calibration file for AQT / EQE1, which report no calibration), the errors
    (layout=optimized without a file for AQT / EQE1, calibration_file with another layout, a repeated qubit, a
    missing layout for an AQT / EQE1 simulation with its explanation) and the run-folder tag layout-<mode>;
    parse_benchmark_args leaves the layout arguments alone; the FakeFez local simulation takes layout= but
    refuses calibration_file=;
  - choose_layout on synthetic targets — a chain with a dead qubit, and AQT's coupling-free target (global
    operations, all-to-all): lowest, random (reproducible from its recorded seed), a forced list and every way
    it can be impossible (wrong length, repeated or missing qubit, no coupler, outside AQT's 20 qubits);
    placement_filter restricts the placements to what a second target supports;
  - the calibration file in either format — the provider's own PIAST-Q / Euro-Q-Exa files of calibration_input_files/ are
    converted automatically, a snapshot is used as it is, anything else is refused — and saved as the run's
    calibration_snapshot.json (a rerun reuses it, a different calibration is refused); recorded sources are never
    absolute paths;
  - find_collect_run_folder: the latest dated folder of a benchmark's layout (or that of date=), the one layout
    on disk when none is named, or an error if there is none or several; find_collect_run_folders: without layout=
    and date= every run of the benchmark with something left to collect (run_has_uncollected), oldest first, or
    an error when nothing is left, and with date= alone that day's run (or #N) of every layout that has one;
    parse_date_arg validates date=YYYY-MM-DD[_#N], parse_run_folder_arg and pop_force_new_run_flag;
  - VLQ: the resonator is always part of the layout — lowest / random / forced choose among the usable qubits;
  - the snapshot of a live run records when the provider made the calibration (measured_at: IBM's last_update_date,
    an IQM calibration set's end_timestamp, VLQ's from vlq_calibration.json), and the layout of a live run is scored on
    the saved snapshot (resolve_run_layout), falling back to the live backend when none could be saved;
  - the scheduling_simulation= argument of the simulate scripts (usage text; the snapshot simulation schedules circuits so idle
    qubits decay) and the layout score's T1/T2 idle-decay term (the exact relaxation fidelity, avoids a short-lived qubit,
    left out without durations or coherence times).
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import CZGate, Measure, RGate, RXXGate
from qiskit.transpiler import InstructionProperties, Target

from source.core.runner_support import benchmark_args as BA
from source.core.qpu import iqm_vlq as V
from source.core.records.run_names import health_check_run_name, qpu_run_name, qpu_run_prefix
from source.core.qpu.calibration_input import (detect_format, load_calibration, newest_provider_file, source_name,
                                      use_calibration_file)
from source.core.qpu.layout_selection import (all_placements, check_forced_layout, choose_layout, placement_filter,
                                     select_physical_qubits)

PROJECT = Path(__file__).resolve().parents[2]
AQT_FILE  = "calibration_input_files/PIAST-Q/piastq_characterisation_2026-09-29.json"
EQE1_FILE = "calibration_input_files/Euro-Q-Exa/calibration_data_Euro-Q-Exa_20260827_active_qubits.json"


def _exits(argv, **kw):
    with pytest.raises(SystemExit) as e:
        BA.parse_layout_args(argv, **kw)
    return e.value.code


def _chain3():
    qc = QuantumCircuit(3)
    qc.cz(0, 1)
    qc.cz(1, 2)
    return qc


def _chain_target(n, edges, dead=()):
    """Target of n qubits with r / measure on the live ones and cz on the edges (both directions)."""
    t = Target(num_qubits=n)
    ok = [q for q in range(n) if q not in dead]
    t.add_instruction(RGate(Parameter("a"), Parameter("b")), {(q,): InstructionProperties(error=1e-3) for q in ok}, name="r")
    t.add_instruction(Measure(), {(q,): InstructionProperties(error=1e-2) for q in ok})
    t.add_instruction(CZGate(), {e: InstructionProperties(error=1e-2) for a, b in edges for e in ((a, b), (b, a))})
    return t


def _aqt_like_target():
    """What the live PIAST-Q backend reports: 20 qubits, global operations (qargs None), no errors."""
    t = Target(num_qubits=20)
    t.add_instruction(RGate(Parameter("a"), Parameter("b")), None, name="r")
    t.add_instruction(RXXGate(Parameter("t")), None, name="rxx")
    t.add_instruction(Measure(), None)
    return t


# ── the arguments ─────────────────────────────────────────────────────────────

def test_layout_defaults_and_values():
    assert BA.parse_layout_args([]).mode == "optimized"
    assert BA.parse_layout_args(["layout=random"]).tag == "layout-random"
    assert BA.parse_layout_args(["layout=lowest"]).tag == "layout-lowest"
    for text in ("15,14,21", "[15,14,21]", "[ 15, 14 ,21 ]"):
        la = BA.parse_layout_args([f"layout={text}"])
        assert (la.mode, la.qubits, la.tag) == ("list", [15, 14, 21], "layout-15-14-21")
    la = BA.parse_layout_args(["layout=optimized", "calibration_file=x.json"])
    assert (la.mode, la.calibration_file, la.tag) == ("optimized", "x.json", "layout-optimized")


def test_no_live_calibration_default_is_random_with_notice():
    la = BA.parse_layout_args([], calibration_required=True)
    assert la.mode == "random" and la.notice and "RANDOM" in la.notice and "calibration_input_files/" in la.notice
    la = BA.parse_layout_args(["calibration_file=x.json"], calibration_required=True)   # a file alone: optimized
    assert (la.mode, la.calibration_file, la.notice) == ("optimized", "x.json", None)
    assert BA.parse_layout_args(["layout=lowest"], calibration_required=True).notice is None


def test_argument_errors(capsys):
    assert _exits(["layout=optimized"], calibration_required=True) == 2   # nothing to score with
    assert "calibration_file" in capsys.readouterr().err
    assert _exits(["layout=random", "calibration_file=x.json"]) == 2      # a file only goes with optimized
    assert _exits(["layout=1,1,2"]) == 2                                  # a qubit twice
    assert _exits(["layout=1,2"]) == 2                                    # the circuits use exactly three qubits
    assert "exactly 3" in capsys.readouterr().err
    assert _exits(["layout=1,2,3,4"]) == 2
    assert _exits(["layout=middle"]) == 2
    assert _exits(["layout=random", "layout=lowest"]) == 2


def test_simulation_requires_layout_with_explanation(capsys):
    assert _exits([], calibration_required=True, layout_required=True, calibration_any_layout=True) == 2
    out = capsys.readouterr().err
    assert "layout is required" in out and "RANDOM layout" in out and "calibration_file" in out
    # in a simulation the file is the noise data, so it goes with any layout; optimized needs none (default file)
    la = BA.parse_layout_args(["layout=random", "calibration_file=x.json"], calibration_required=True,
                              layout_required=True, calibration_any_layout=True)
    assert (la.mode, la.calibration_file) == ("random", "x.json")
    assert BA.parse_layout_args(["layout=optimized"], calibration_required=True, layout_required=True,
                                calibration_any_layout=True).mode == "optimized"


def test_collect_arguments():
    assert BA.parse_layout_args(["backend=x", "mini"], collect=True) is None
    assert BA.parse_layout_args(["layout=lowest"], collect=True).tag == "layout-lowest"
    assert _exits(["calibration_file=x.json"], collect=True) == 2


def test_benchmark_args_ignore_layout_arguments():
    b = BA.parse_benchmark_args(["mini", "layout=15,14,21", "calibration_file=x.json", "backend=B"], extra_keys=("backend",))
    assert b.tag == "mini"


def test_local_simulation_scripts_refuse_layout_arguments():
    for arg in ("layout=random", "calibration_file=x.json"):
        with pytest.raises(SystemExit) as e:
            BA.parse_benchmark_args(["mini", arg], layout_args=False)
        assert e.value.code == 2


def test_usage_texts_mention_layout():
    for form in ("submit_live", "submit_no_live", "simulate", "simulate_required", "collect"):
        text = BA.make_usage(PROJECT / "source/benchmark_scripts/real_qpu_scripts/IQM_EQE1/qbcs_real_submit_iqm_eqe1.py", "x", layout=form)
        assert "layout" in text.lower()
    required = BA.make_usage(PROJECT / "source/benchmark_scripts/real_qpu_scripts/IQM_EQE1/qbcs_real_submit_iqm_eqe1.py", "x", layout="simulate_required")
    assert "RANDOM layout" in required and "(required)" in required
    live = BA.make_usage(PROJECT / "source/benchmark_scripts/real_qpu_scripts/IBM/qbcs_real_submit_ibm.py", "x", layout="submit_live")
    assert "calibration_file=PATH" not in live and "not accepted for a real run" in live


def test_live_calibration_submissions_refuse_a_calibration_file(capsys):
    """IBM, IQM Resonance and IQM VLQ report live calibration: their submit scripts refuse calibration_file=,
    explaining why; the simulate scripts (calibration_any_layout) and AQT / EQE1 submissions still take it."""
    assert _exits(["layout=optimized", "calibration_file=x.json"], live_calibration="IBM") == 2
    out = capsys.readouterr().err
    assert "IBM reports live calibration data" in out and "simulate script" in out
    assert BA.parse_layout_args(["layout=random"], live_calibration="IBM").mode == "random"
    assert BA.parse_layout_args(["layout=random", "calibration_file=x.json"], calibration_any_layout=True).calibration_file == "x.json"
    assert BA.parse_layout_args(["layout=optimized", "calibration_file=x.json"], calibration_required=True).calibration_file == "x.json"


def test_fake_fez_simulation_takes_a_layout_but_no_calibration_file(capsys):
    """benchmark_fake_fez_lnn.py: layout= (default optimized) is accepted, calibration_file= refused with its own usage."""
    script = PROJECT / "source/benchmark_scripts/local_simulation_scripts/benchmark_fake_fez_lnn.py"
    assert BA.parse_benchmark_args(["mini", "layout=random"]).tag == "mini"
    assert BA.parse_layout_args(["mini"], no_calibration_file=True).tag == "layout-optimized"
    assert BA.parse_layout_args(["mini", "layout=15,14,21"], no_calibration_file=True).tag == "layout-15-14-21"
    capsys.readouterr()
    assert _exits(["mini", "layout=random", "calibration_file=x.json"], no_calibration_file=True) == 2
    assert "FakeFez's built-in calibration" in capsys.readouterr().err
    usage = BA.make_usage(script, "x", backend=False, layout="simulate_builtin")
    assert "layout=optimized" in usage and "calibration_file=PATH" not in usage


def test_scheduling_simulation_argument(capsys):
    assert BA.parse_scheduling_arg(["mini", "scheduling_simulation=true"]) is True
    assert BA.parse_scheduling_arg(["scheduling_simulation=false", "layout=random"]) is False
    assert BA.parse_scheduling_arg([]) is True and BA.parse_scheduling_arg(["mini", "layout=random"]) is True   # optional: default true
    for bad in (["scheduling_simulation=yes"], ["scheduling_simulation="], ["scheduling_simulation=True"],
                ["scheduling_simulation=true", "scheduling_simulation=false"]):
        with pytest.raises(SystemExit) as e:
            BA.parse_scheduling_arg(bad)
        assert e.value.code == 2
    assert "scheduling_simulation" in capsys.readouterr().err


def test_simulate_usage_shows_the_scheduling_argument():
    script = PROJECT / "source/benchmark_scripts/real_qpu_scripts/IQM_EQE1/qbcs_real_simulate_iqm_eqe1.py"
    for form in ("simulate", "simulate_builtin", "simulate_required"):
        text = BA.make_usage(script, "x", backend=False, layout=form)
        assert text.splitlines()[0].endswith("[scheduling_simulation=true|false]") and "SCHEDULING" in text
    for form in ("submit_live", "submit_no_live", "collect"):
        assert "scheduling_simulation" not in BA.make_usage(script, "x", layout=form)


# ── choosing the layout ───────────────────────────────────────────────────────

def test_lowest_random_and_forced_on_a_chain_with_a_dead_qubit():
    qc, t = _chain3(), _chain_target(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)], dead=[3])
    lowest = choose_layout("lowest", qc, t)
    assert lowest["physical_qubits"] == [0, 1, 2] and lowest["method"] == "lowest" and lowest["layout_mode"] == "lowest"
    cands = all_placements(qc, t)
    assert lowest["n_candidates"] == len(cands) and all(3 not in c for c in cands)   # the dead qubit is never used
    a = choose_layout("random", qc, t, seed=7)
    assert a["method"] == "random" and a["seed"] == 7 and tuple(a["physical_qubits"]) in cands
    assert choose_layout("random", qc, t, seed=7)["physical_qubits"] == a["physical_qubits"]   # reproducible from the seed
    seeds = {choose_layout("random", qc, t)["seed"] for _ in range(5)}
    assert len(seeds) > 1                                                                      # a fresh seed each time
    forced = choose_layout("list", qc, t, qubits=[2, 1, 5])
    assert forced["physical_qubits"] == [2, 1, 5] and forced["method"] == "user"
    assert choose_layout("optimized", qc, t)["method"] == "scored"


def _timed_chain_target(t1_t2, d_1q=24e-9, d_2q=50e-9):
    """_chain_target(4, …) with the same errors everywhere, a duration on every gate and T1/T2 per qubit (seconds)."""
    from qiskit.providers.backend import QubitProperties
    t = Target(num_qubits=4, qubit_properties=[QubitProperties(t1=a, t2=b) for a, b in t1_t2])
    t.add_instruction(RGate(Parameter("a"), Parameter("b")), {(q,): InstructionProperties(error=1e-3, duration=d_1q) for q in range(4)}, name="r")
    t.add_instruction(Measure(), {(q,): InstructionProperties(error=1e-2) for q in range(4)})
    t.add_instruction(CZGate(), {e: InstructionProperties(error=1e-2, duration=d_2q)
                                 for a, b in [(0, 1), (1, 2), (2, 3)] for e in ((a, b), (b, a))})
    return t


def test_idle_decay_log_is_the_exact_relaxation_fidelity():
    import math
    from source.core.qpu.layout_selection import idle_decay_log
    assert idle_decay_log(50e-6, 30e-6, 0.0) == pytest.approx(0.0)
    f = (1 + 2 * math.exp(-10 / 30) + math.exp(-10 / 50)) / 4
    assert idle_decay_log(50e-6, 30e-6, 10e-6) == pytest.approx(math.log(f))
    assert idle_decay_log(10e-6, 100e-6, 5e-6) == pytest.approx(idle_decay_log(10e-6, 20e-6, 5e-6))   # T2 capped at 2·T1
    assert idle_decay_log(50e-6, 30e-6, 1e-3) > math.log(0.25) - 1e-9                              # bounded, no negative fidelity
    assert idle_decay_log(None, 30e-6, 1e-6) is None and idle_decay_log(50e-6, 0, 1e-6) is None


def test_optimized_layout_avoids_the_short_lived_qubit():
    """Equal gate and readout errors everywhere: only T1/T2 differ, so the chain 3-qubit circuit avoids qubit 0."""
    qc = _chain3()
    long_lived, short = (200e-6, 150e-6), (20e-6, 8e-6)
    rec = select_physical_qubits(qc, _timed_chain_target([short, long_lived, long_lived, long_lived]))
    assert rec["decoherence_included"] and rec["t_circuit_s"] > 0
    assert 0 not in rec["physical_qubits"] and sorted(rec["physical_qubits"]) == [1, 2, 3]
    flat = select_physical_qubits(qc, _timed_chain_target([long_lived] * 4))   # nothing to separate: the lowest placement
    assert flat["physical_qubits"] == [0, 1, 2]


def test_decoherence_term_is_left_out_without_durations_or_coherence_times():
    qc = _chain3()
    no_t = _chain_target(4, [(0, 1), (1, 2), (2, 3)])                          # no qubit_properties, no durations
    rec = select_physical_qubits(qc, no_t)
    assert rec["method"] == "scored" and not rec["decoherence_included"] and rec["t_circuit_s"] is None
    timed_target = _timed_chain_target([(20e-6, 8e-6), (200e-6, 150e-6), (200e-6, 150e-6), (200e-6, 150e-6)])
    assert not select_physical_qubits(qc, _chain_target(4, [(0, 1), (1, 2), (2, 3)]))["decoherence_included"]
    # a Target with T1/T2 but no durations
    bare = _timed_chain_target([(20e-6, 8e-6)] + [(200e-6, 150e-6)] * 3, d_1q=None, d_2q=None)
    assert not select_physical_qubits(qc, bare)["decoherence_included"]
    assert select_physical_qubits(qc, timed_target)["decoherence_included"]


def test_snapshot_simulation_schedules_circuits_so_idle_qubits_decay():
    """A calibration-snapshot backend whose gates all have durations is transpiled scheduled (delays on waiting qubits)
    and Aer relaxes those delays with T1/T2; a snapshot gate without a duration, or a real QPU, stays unscheduled."""
    from qiskit import transpile
    from source.core.qpu.qpu_snapshot import (EQE1_TYPICAL_DURATIONS, backend_from_snapshot, noisy_sampler_from_snapshot,
                                              scheduling_method, snapshot_from_iqm_eqe1_json)
    data = json.loads((PROJECT / EQE1_FILE).read_text(encoding="utf-8"))
    snap = snapshot_from_iqm_eqe1_json(data, EQE1_FILE)
    assert {e["duration"] for e in snap["operations"]["cz"]} == {EQE1_TYPICAL_DURATIONS["cz"]}   # the file lists none
    backend, _, sampler = noisy_sampler_from_snapshot(snap, seed=1)
    assert backend.schedule_idle and scheduling_method(backend) == "alap" and "delay" in backend.target.operation_names
    qc = _chain3()
    qc.id(0)                                       # circuits arrive in the default "dt" unit; the scheduler needs integer delays
    qc.measure_all()
    isa = transpile(qc, backend, optimization_level=1, initial_layout=[0, 1, 2], scheduling_method=scheduling_method(backend))
    assert isa.count_ops().get("delay", 0) > 0
    assert backend.target.dt == 1e-9 and all(isinstance(i.operation.params[0], int) for i in isa.data if i.operation.name == "delay")
    from qiskit.qpy import dump, load
    import io
    buf = io.BytesIO()
    dump(isa, buf)                                 # the run saves the scheduled circuits as QPY (qiskit 2.1 could not read floats)
    buf.seek(0)
    assert load(buf)[0].count_ops() == isa.count_ops()
    # one X, then a wait of several T1: the qubit has relaxed to 0 before the measurement
    idle = QuantumCircuit(1, 1)
    idle.x(0)
    idle.delay(10 * snap["qubit_properties"][0]["t1"], 0, unit="s")
    idle.measure(0, 0)
    counts = sampler.run([transpile(idle, backend, optimization_level=0, initial_layout=[0])], shots=500).result()[0].data.c.get_counts()
    assert counts.get("0", 0) > 0.9 * 500
    no_duration = json.loads(json.dumps(snap))
    no_duration["operations"]["cz"][0]["duration"] = None
    assert scheduling_method(backend_from_snapshot(no_duration)) is None
    no_data = json.loads(json.dumps(snap))                                  # IQM Resonance: a coupler and ops without any data
    no_data["operations"]["cz"][0].update(duration=None, error=None)
    no_data["operations"]["reset"] = [{"qargs": [q], "error": None, "duration": None} for q in (0, 1, 2)]
    assert scheduling_method(backend_from_snapshot(no_data)) == "alap"
    ibm_like = json.loads(json.dumps(snap))                                  # IBM lists a delay without a duration: no gate
    ibm_like["operations"]["delay"] = [{"qargs": [q], "error": None, "duration": None} for q in (0, 1, 2)]
    assert scheduling_method(backend_from_snapshot(ibm_like)) == "alap"
    assert scheduling_method(SimpleNamespace(name="ibm_fez")) is None
    # scheduling_simulation=false: no scheduling method, so no delay in the transpiled circuits
    assert scheduling_method(backend, schedule=False) is None and scheduling_method(backend, schedule=True) == "alap"
    unscheduled = transpile(qc, backend, optimization_level=1, initial_layout=[0, 1, 2],
                            scheduling_method=scheduling_method(backend, schedule=False))
    assert unscheduled.count_ops().get("delay", 0) == 0


@pytest.mark.parametrize("qubits, why", [
    ([0, 1], "has 2 qubits"),                 # wrong count
    ([0, 1, 1], "repeats"),                   # a qubit twice
    ([0, 1, 3], "not usable"),                # the dead qubit
    ([0, 1, 9], "not usable"),                # outside the chip
    ([0, 2, 1], "incompatible"),              # no coupler between 0 and 2
])
def test_impossible_forced_layouts_raise(qubits, why):
    qc, t = _chain3(), _chain_target(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], dead=[3])
    with pytest.raises(ValueError, match=why):
        choose_layout("list", qc, t, qubits=qubits)
    with pytest.raises(ValueError):
        check_forced_layout(qc, t, qubits)


def test_aqt_like_target_all_to_all_within_its_qubits():
    qc = QuantumCircuit(3)
    qc.rxx(0.3, 0, 1); qc.rxx(0.3, 1, 2); qc.rxx(0.3, 0, 2)
    t = _aqt_like_target()
    assert choose_layout("lowest", qc, t)["physical_qubits"] == [0, 1, 2]
    rec = choose_layout("random", qc, t, seed=3)
    assert rec["n_candidates"] == 20 * 19 * 18 and set(rec["physical_qubits"]) <= set(range(20)) and len(set(rec["physical_qubits"])) == 3
    assert choose_layout("list", qc, t, qubits=[19, 0, 7])["physical_qubits"] == [19, 0, 7]
    with pytest.raises(ValueError, match="not usable"):
        choose_layout("list", qc, t, qubits=[0, 1, 20])      # outside the 20 qubits
    assert select_physical_qubits(qc, t)["method"] == "no_coupling_map"   # why optimized needs a calibration file here


def test_placement_filter_restricts_to_the_live_qpu():
    qc = _chain3()
    file_t = _chain_target(8, [(i, i + 1) for i in range(7)])
    live_t = _chain_target(6, [(i, i + 1) for i in range(5)])      # the live QPU lost qubits 6 and 7
    ok = placement_filter(qc, live_t)
    assert ok((1, 2, 3)) and not ok((5, 6, 7)) and not ok((0, 2, 1))
    for mode in ("lowest", "random", "optimized"):
        rec = choose_layout(mode, qc, file_t, placement_ok=ok)
        assert max(rec["physical_qubits"]) < 6
    assert all(max(p) < 6 for p in all_placements(qc, file_t, placement_ok=ok))


# ── calibration files ─────────────────────────────────────────────────────────

def test_provider_files_are_detected_and_converted():
    for rel, fmt, backend, n in ((AQT_FILE, "aqt", "aqt_marten", 20), (EQE1_FILE, "eqe1", "iqm_eqe1", 54)):
        data = json.loads((PROJECT / rel).read_text(encoding="utf-8"))
        assert detect_format(data) == fmt
        snap, got = load_calibration(rel)
        assert got == fmt and snap["backend"] == backend and snap["num_qubits"] == n
        assert detect_format(snap) == "snapshot"
        again, fmt2 = load_calibration(PROJECT / rel)              # an absolute path works too
        assert fmt2 == fmt and again["measured_at"] == snap["measured_at"]
    assert newest_provider_file("PIAST-Q").name.startswith("piastq_characterisation_")
    assert newest_provider_file("Euro-Q-Exa").name.startswith("calibration_data_Euro-Q-Exa_")


def test_unknown_or_missing_calibration_files_are_refused(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a calibration file"):
        load_calibration(bad)
    with pytest.raises(SystemExit):
        load_calibration("calibration_input_files/nothing_here.json")
    assert "calibration_input_files/" in capsys.readouterr().err


def test_calibration_becomes_the_runs_snapshot_and_is_kept(tmp_path, capsys):
    run = tmp_path / "run"
    snap, backend, src = use_calibration_file(AQT_FILE, run, "test")
    assert src == AQT_FILE                                         # the recorded source is project-relative
    assert backend.target.num_qubits == 20 and (run / "calibration_snapshot.json").exists()
    assert "converted" in capsys.readouterr().out
    snap2, _, _ = use_calibration_file(AQT_FILE, run, "test")      # a rerun reuses the saved snapshot
    assert snap2["measured_at"] == snap["measured_at"] and "reusing" in capsys.readouterr().out
    other = json.loads((run / "calibration_snapshot.json").read_text(encoding="utf-8"))
    other["measured_at"] = "2020-01-01T00:00:00Z"                  # the same file again, but the run holds another calibration
    (run / "calibration_snapshot.json").write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(SystemExit):
        use_calibration_file(AQT_FILE, run, "test")
    assert "keeps one calibration" in capsys.readouterr().err
    # files without measured_at are compared by content
    nomt = {k: v for k, v in snap.items() if k != "measured_at"}
    f1, f2 = tmp_path / "nomt1.json", tmp_path / "nomt2.json"
    f1.write_text(json.dumps(nomt), encoding="utf-8")
    f2.write_text(json.dumps({**nomt, "num_qubits": nomt["num_qubits"] + 1}), encoding="utf-8")
    use_calibration_file(f1, tmp_path / "run3", "test")
    use_calibration_file(f1, tmp_path / "run3", "test")                # the same content: reused
    assert "reusing" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        use_calibration_file(f2, tmp_path / "run3", "test")            # other content: refused
    assert "keeps one calibration" in capsys.readouterr().err
    # a snapshot-format file is used as it is
    snap_file = tmp_path / "snap.json"
    snap_file.write_text(json.dumps(snap), encoding="utf-8")
    _, _, src2 = use_calibration_file(snap_file, tmp_path / "run2", "test")
    assert src2 == "snap.json" and not Path(src2).is_absolute()    # outside the project: the name only


def test_layout_on_a_calibration_files_target():
    qc = QuantumCircuit(3)
    qc.rxx(0.3, 0, 1); qc.rxx(0.3, 1, 2); qc.rxx(0.3, 0, 2)
    _, aqt_backend, _ = use_calibration_file(AQT_FILE, PROJECT / "datafiles" / "tests" / "_test18_snapshot_tmp", "test")
    try:
        rec = choose_layout("optimized", qc, aqt_backend.target)
        assert rec["method"] == "scored" and len(rec["physical_qubits"]) == 3 and rec["layout_mode"] == "optimized"
    finally:
        import shutil
        shutil.rmtree(PROJECT / "datafiles" / "tests" / "_test18_snapshot_tmp", ignore_errors=True)
    assert source_name(PROJECT / AQT_FILE) == AQT_FILE


# ── run folders ───────────────────────────────────────────────────────────────

def test_collect_finds_the_latest_dated_run_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(BA, "PROJECT_ROOT", tmp_path)
    base = qpu_run_prefix("IQM_EQE1", "EQE1", "mini", True, True)
    data = tmp_path / "datafiles"
    data.mkdir()
    random = BA.parse_layout_args(["layout=random"], collect=True)
    with pytest.raises(SystemExit):
        BA.find_collect_run_folder(base, random)                    # no folder of that layout yet
    with pytest.raises(SystemExit):
        BA.find_collect_run_folder(base, None)                      # none at all
    for d in ("2026-10-01_#1", "2026-10-05_#2", "2026-10-05_#10"):
        (data / f"{base}_layout-lowest_{d}").mkdir()
    assert BA.find_collect_run_folder(base, None) == f"{base}_layout-lowest_2026-10-05_#10"        # the highest number wins
    assert BA.find_collect_run_folder(base, None, "2026-10-01") == f"{base}_layout-lowest_2026-10-01_#1"  # date= picks a day
    assert BA.find_collect_run_folder(base, None, "2026-10-05") == f"{base}_layout-lowest_2026-10-05_#10"  # its latest run
    assert BA.find_collect_run_folder(base, None, "2026-10-05_#2") == f"{base}_layout-lowest_2026-10-05_#2"  # that run
    with pytest.raises(SystemExit):
        BA.find_collect_run_folder(base, None, "2026-10-02")        # no run of that date
    with pytest.raises(SystemExit):
        BA.find_collect_run_folder(base, None, "2026-10-05_#3")     # no such number
    (data / f"{base}_layout-15-14-21_2026-10-03_#1").mkdir()
    with pytest.raises(SystemExit):
        BA.find_collect_run_folder(base, None)                      # several layouts: name the layout
    assert "name the one with layout=" in capsys.readouterr().err
    la = BA.parse_layout_args(["layout=15,14,21"], collect=True)
    assert BA.find_collect_run_folder(base, la) == f"{base}_layout-15-14-21_2026-10-03_#1"
    lowest = BA.parse_layout_args(["layout=lowest"], collect=True)
    assert BA.find_collect_run_folder(base, lowest, "2026-10-01") == f"{base}_layout-lowest_2026-10-01_#1"


def _fake_run(data, name, statuses=(), health=None, standalone=None):
    """A run folder with chunk job records of the given statuses (and a failed_attempts/ record, which is ignored),
    an accompanying health check of status `health` and a standalone health check record of status `standalone`."""
    jr = data / name / "job_records"
    jr.mkdir(parents=True)
    for i, st in enumerate(statuses):
        (jr / f"job_record_chunk_{i}.json").write_text(json.dumps({"chunk_id": i, "status": st}), encoding="utf-8")
    (jr / "failed_attempts").mkdir()
    (jr / "failed_attempts" / "job_record_chunk_0_failed_1.json").write_text(json.dumps({"status": "submitted"}), encoding="utf-8")
    if health:
        hc = data / name / "health_checks"
        hc.mkdir()
        (hc / "health_check_1.json").write_text(json.dumps({"schema": 2, "number": 1, "status": health}), encoding="utf-8")
    if standalone:
        (jr / "health_check.json").write_text(json.dumps({"status": standalone}), encoding="utf-8")


def test_run_has_uncollected(tmp_path):
    from source.core.records.run_names import run_has_uncollected
    base = qpu_run_prefix("IQM_EQE1", "EQE1", "mini", True, True)
    _fake_run(tmp_path, f"{base}_layout-lowest_2026-10-01_#1", ["complete", "failed"])
    _fake_run(tmp_path, f"{base}_layout-lowest_2026-10-01_#2", ["complete", "submitted"])
    _fake_run(tmp_path, f"{base}_layout-lowest_2026-10-01_#3", ["complete"], health="submitted")
    _fake_run(tmp_path, f"{base}_layout-lowest_2026-10-01_#4", ["complete"], health="failed")
    (tmp_path / f"{base}_layout-lowest_2026-10-01_#5").mkdir()
    assert not run_has_uncollected(f"{base}_layout-lowest_2026-10-01_#1", tmp_path)   # collected / failed only
    assert run_has_uncollected(f"{base}_layout-lowest_2026-10-01_#2", tmp_path)       # a submitted job
    assert run_has_uncollected(f"{base}_layout-lowest_2026-10-01_#3", tmp_path)       # a submitted health check
    assert not run_has_uncollected(f"{base}_layout-lowest_2026-10-01_#4", tmp_path)
    assert not run_has_uncollected(f"{base}_layout-lowest_2026-10-01_#5", tmp_path)   # no records at all
    hc = f"{health_check_run_name('IQM_EQE1', 'EQE1', 1, 100)}_2026-10-01_#1"   # a standalone health check
    _fake_run(tmp_path, hc, standalone="submitted")
    assert run_has_uncollected(hc, tmp_path)
    (tmp_path / hc / "job_records" / "health_check.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    assert not run_has_uncollected(hc, tmp_path)


def test_collect_folders_without_layout_and_date(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(BA, "PROJECT_ROOT", tmp_path)
    base = qpu_run_prefix("IQM_EQE1", "EQE1", "mini", True, True)
    data = tmp_path / "datafiles"
    data.mkdir()
    with pytest.raises(SystemExit) as e:
        BA.find_collect_run_folders(base, None)                     # no folder at all
    assert e.value.code == 2 and "no run folder datafiles/" in capsys.readouterr().err
    low1, low2 = f"{base}_layout-lowest_2026-10-01_#1", f"{base}_layout-lowest_2026-10-05_#10"
    lst, rnd = f"{base}_layout-15-14-21_2026-10-03_#1", f"{base}_layout-random_2026-10-05_#2"
    _fake_run(data, low1, ["complete"])
    _fake_run(data, low2, ["submitted"])
    _fake_run(data, lst, ["submitted", "complete"])
    _fake_run(data, rnd, ["failed"])
    lowest = BA.parse_layout_args(["layout=lowest"], collect=True)
    assert BA.find_collect_run_folders(base, lowest) == [low2]               # layout= : the latest of that layout
    assert BA.find_collect_run_folders(base, lowest, "2026-10-01") == [low1]   # layout= and date=: finished, still named
    assert BA.find_collect_run_folders(base, None) == [lst, low2]            # every unfinished run, oldest first
    _fake_run(data, f"{base}_layout-lowest_2026-10-02_#1", ["submitted"])
    assert BA.find_collect_run_folders(base, None) == [f"{base}_layout-lowest_2026-10-02_#1", lst, low2]
    (data / low2 / "job_records" / "job_record_chunk_0.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (data / lst / "job_records" / "job_record_chunk_0.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (data / f"{base}_layout-lowest_2026-10-02_#1" / "job_records" / "job_record_chunk_0.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8")
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        BA.find_collect_run_folders(base, None)                     # nothing left
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "nothing left to collect" in err and low1 in err and "date=YYYY-MM-DD_#N" in err


def test_collect_folders_with_date_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(BA, "PROJECT_ROOT", tmp_path)
    base = qpu_run_prefix("IQM_EQE1", "EQE1", "mini", True, True)
    data = tmp_path / "datafiles"
    data.mkdir()
    low1, low2 = f"{base}_layout-lowest_2026-10-05_#1", f"{base}_layout-lowest_2026-10-05_#2"
    rnd, lst = f"{base}_layout-random_2026-10-05_#1", f"{base}_layout-15-14-21_2026-10-03_#1"
    for n in (low1, low2, rnd, lst):
        _fake_run(data, n, ["complete"])
    # date=YYYY-MM-DD: that day's latest run of every layout, oldest first by (date, number)
    assert BA.find_collect_run_folders(base, None, "2026-10-05") == [rnd, low2]
    assert BA.find_collect_run_folders(base, None, "2026-10-03") == [lst]           # only one layout has that date
    assert BA.find_collect_run_folders(base, None, "2026-10-05_#1") == [low1, rnd]   # that number, per layout
    assert BA.find_collect_run_folders(base, None, "2026-10-05_#2") == [low2]        # only one layout has #2
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        BA.find_collect_run_folders(base, None, "2026-10-04")
    assert e.value.code == 2
    assert "2026-10-04" in capsys.readouterr().err


def test_date_argument():
    assert BA.parse_date_arg(["mini", "layout=random"]) is None
    assert BA.parse_date_arg(["mini", "date=2026-10-11"]) == "2026-10-11"
    assert BA.parse_date_arg(["mini", "date=2026-10-11_#3"]) == "2026-10-11_#3"
    for bad in (["date=11/10/2026"], ["date=2026-10"], ["date=2026-10-11", "date=2026-10-12"],
                ["date=2026-10-11_#0"], ["date=2026-10-11_3"], ["date=2026-10-11_#x"]):
        with pytest.raises(SystemExit):
            BA.parse_date_arg(bad)


def test_force_new_run_flag():
    assert BA.pop_force_new_run_flag(["mini", "--force-new-run", "layout=random"]) == (True, ["mini", "layout=random"])
    assert BA.pop_force_new_run_flag(["mini"]) == (False, ["mini"])


def test_run_folder_argument():
    undated = qpu_run_name("AQT_PIAST-Q", "PIAST-Q", "mini", True, True, "layout-optimized")
    good = f"{undated}_2026-10-11_#2"
    assert BA.parse_run_folder_arg(["mini", f"run_folder={good}"], undated) == good
    assert BA.parse_run_folder_arg(["mini"], undated) is None
    other = qpu_run_name("AQT_PIAST-Q", "PIAST-Q", "mini", True, True, "layout-random")
    for bad in (f"run_folder={other}_2026-10-11_#1", f"run_folder={undated}", f"run_folder={undated}_2026-10-11",
                "run_folder=nonsense"):
        with pytest.raises(SystemExit):
            BA.parse_run_folder_arg(["mini", bad], undated)
    with pytest.raises(SystemExit):
        BA.parse_run_folder_arg([f"run_folder={good}", f"run_folder={good}"], undated)


# ── VLQ ───────────────────────────────────────────────────────────────────────

VLQ_CAL   = json.loads((PROJECT / "source/tests/iqm_vlq_check_data/iqm_vlq_calibration_fixture.json").read_text(encoding="utf-8"))
VLQ_ROLES = {"n_prx": [96, 153, 210], "n_move": [76, 38, 0], "n_cz": [0, 38, 76]}


def test_vlq_layouts_always_include_the_resonator_and_skip_uncalibrated_qubits():
    usable, excluded = V.usable_vlq_qubits(VLQ_CAL)
    assert excluded and all(V.physical_index(q) not in usable for q in excluded)   # QB12 has no MOVE data
    lowest = V.choose_vlq_layout("lowest", VLQ_ROLES, VLQ_CAL)
    assert lowest["physical_qubits"] == usable[:3] and lowest["method"] == "lowest"
    rnd = V.choose_vlq_layout("random", VLQ_ROLES, VLQ_CAL, seed=11)
    assert rnd["seed"] == 11 and set(rnd["physical_qubits"]) <= set(usable) and len(set(rnd["physical_qubits"])) == 3
    assert V.choose_vlq_layout("random", VLQ_ROLES, VLQ_CAL, seed=11)["physical_qubits"] == rnd["physical_qubits"]
    forced = V.choose_vlq_layout("list", VLQ_ROLES, VLQ_CAL, qubits=[usable[5], usable[0], usable[9]])
    assert forced["physical_qubits"] == [usable[5], usable[0], usable[9]] and forced["method"] == "user"
    bad = V.physical_index(excluded[0])
    with pytest.raises(ValueError, match="not valid on VLQ"):
        V.choose_vlq_layout("list", VLQ_ROLES, VLQ_CAL, qubits=[usable[0], usable[1], bad])
    with pytest.raises(ValueError):
        V.choose_vlq_layout("list", VLQ_ROLES, VLQ_CAL, qubits=[usable[0], usable[0], usable[1]])
    assert V.choose_vlq_layout("optimized", VLQ_ROLES, VLQ_CAL)["method"] == "star_roles"


# ── saving the calibration of a real run, and friendly layout errors ──────────

def test_live_calibration_is_saved_once_and_only_where_reported(tmp_path, monkeypatch, capsys):
    from source.core.qpu import calibration_input as CI
    _, reporting, _ = use_calibration_file(AQT_FILE, tmp_path / "x", "test")     # a backend whose Target has errors
    monkeypatch.setattr(CI, "PROJECT_ROOT", tmp_path)
    out = CI.save_live_calibration(reporting, "run_a")
    assert out == tmp_path / "datafiles" / "run_a" / "calibration_snapshot.json" and out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["num_qubits"] == 20
    first = out.read_text(encoding="utf-8")
    assert CI.save_live_calibration(reporting, "run_a") is None and out.read_text(encoding="utf-8") == first   # a run keeps its first
    silent = SimpleNamespace(name="no_data", target=_aqt_like_target())          # no errors: AQT's / EQE1's live case
    assert CI.save_live_calibration(silent, "run_b") is None
    assert not (tmp_path / "datafiles" / "run_b" / "calibration_snapshot.json").exists()
    assert "reports no calibration data" in capsys.readouterr().out
    broken = SimpleNamespace(name="broken")                                      # never raises
    assert CI.save_live_calibration(broken, "run_c") is None


def _reporting_backend(tmp_path, **attrs):
    """A stub backend (the live QPU) whose Target carries errors — AQT's file converted — plus `attrs`."""
    _, offline, _ = use_calibration_file(AQT_FILE, tmp_path / "stub_src", "test")
    return SimpleNamespace(name="stub_qpu", target=offline.target, **attrs)


def test_snapshot_records_when_the_provider_made_the_calibration(tmp_path):
    from datetime import datetime, timedelta, timezone
    from source.core.qpu.qpu_snapshot import save_calibration_snapshot
    ibm_time = datetime(2026, 10, 12, 19, 25, 25, tzinfo=timezone(timedelta(hours=2)))      # IBM: tz-aware, +02:00
    ibm  = _reporting_backend(tmp_path, properties=lambda: SimpleNamespace(last_update_date=ibm_time))
    snap = save_calibration_snapshot(ibm, tmp_path / "ibm.json")
    assert snap["measured_at"] == "2026-10-12T17:25:25.000000Z"                              # UTC, "Z"
    assert json.loads((tmp_path / "ibm.json").read_text(encoding="utf-8"))["measured_at"] == snap["measured_at"]

    asked = []
    def get_calibration_set(cal_id):
        asked.append(cal_id)
        return SimpleNamespace(end_timestamp=datetime(2026, 10, 12, 5, 45, 6, 25560, tzinfo=timezone.utc))
    iqm = _reporting_backend(tmp_path, architecture=SimpleNamespace(calibration_set_id="set-1"),
                             client=SimpleNamespace(get_calibration_set=get_calibration_set))
    snap = save_calibration_snapshot(iqm, tmp_path / "iqm.json")
    assert asked == ["set-1"] and snap["measured_at"] == "2026-10-12T05:45:06.025560Z"       # the set the backend runs with
    assert snap["calibration_set_id"] == "set-1"

    def refuse():
        raise RuntimeError("no properties")
    assert save_calibration_snapshot(_reporting_backend(tmp_path, properties=refuse), tmp_path / "x.json")["measured_at"] is None
    assert save_calibration_snapshot(_reporting_backend(tmp_path), tmp_path / "y.json")["measured_at"] is None   # silent SDK
    given = save_calibration_snapshot(_reporting_backend(tmp_path), tmp_path / "z.json", measured_at="2026-01-02T03:04:05+01:00")
    assert given["measured_at"] == "2026-01-02T02:04:05.000000Z"


def test_vlq_snapshot_carries_the_calibration_sets_end_timestamp():
    assert V.snapshot_from_calibration(VLQ_CAL)["measured_at"] == "2026-09-28T05:22:21.060904Z"
    assert V.snapshot_from_calibration({k: v for k, v in VLQ_CAL.items() if k != "calibration_set"} | {"calibration_set": {}}
                                       )["measured_at"] is None


def test_a_live_run_scores_its_layout_on_the_saved_snapshot(tmp_path, monkeypatch):
    from source.core.qpu import calibration_input as CI
    live = _reporting_backend(tmp_path)                    # before PROJECT_ROOT moves: it loads a project file
    vlq  = _reporting_backend(tmp_path, vlq_calibration=None)
    monkeypatch.setattr(CI, "PROJECT_ROOT", tmp_path)
    layout = SimpleNamespace(notice=None, calibration_file=None, mode="optimized")

    class Q:
        def resolve_physical_qubits(self, backend, **kw):
            self.kw = kw
            return [0, 1, 2]
    q = Q()
    assert CI.resolve_run_layout(q, live, "run_a", layout)[0] == [0, 1, 2]
    scored_on = q.kw["layout_backend"]
    assert scored_on is not None and scored_on is not live                                   # rebuilt from calibration_snapshot.json
    saved = json.loads((tmp_path / "datafiles" / "run_a" / "calibration_snapshot.json").read_text(encoding="utf-8"))
    assert scored_on.target.num_qubits == saved["num_qubits"] and q.kw["calibration_source"] is None

    # no snapshot could be saved (a QPU that reports no calibration): the live backend scores, as before
    silent = SimpleNamespace(name="no_data", target=_aqt_like_target())
    CI.resolve_run_layout(q, silent, "run_b", layout)
    assert q.kw["layout_backend"] is None
    # VLQ scores on its own vlq_calibration.json, never on the snapshot
    CI.resolve_run_layout(q, vlq, "run_c", layout)
    assert q.kw["layout_backend"] is not None            # vlq_calibration None: not a VLQ backend
    vlq.vlq_calibration = VLQ_CAL
    assert CI._saved_snapshot_backend(vlq, "run_c") is None


def test_unusable_layout_ends_the_script_with_one_line(capsys):
    from source.core.qpu.calibration_input import resolve_layout_or_exit

    class Q:
        def resolve_physical_qubits(self, backend, **kw):
            raise ValueError("layout [0, 2, 1] is incompatible with the QPU's connectivity")

    with pytest.raises(SystemExit) as e:
        resolve_layout_or_exit(Q(), object(), run_folder="r")
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("❌ Layout:") and "Traceback" not in err
