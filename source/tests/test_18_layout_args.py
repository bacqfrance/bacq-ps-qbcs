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
    parse_benchmark_args leaves the layout arguments alone;
  - choose_layout on synthetic targets — a chain with a dead qubit, and AQT's coupling-free target (global
    operations, all-to-all): lowest, random (reproducible from its recorded seed), a forced list and every way
    it can be impossible (wrong length, repeated or missing qubit, no coupler, outside AQT's 20 qubits);
    placement_filter restricts the placements to what a second target supports;
  - the calibration file in either format — the provider's own PIAST-Q / Euro-Q-Exa files of calibration_input_files/ are
    converted automatically, a snapshot is used as it is, anything else is refused — and saved as the run's
    calibration_snapshot.json (a rerun reuses it, a different calibration is refused); recorded sources are never
    absolute paths;
  - find_collect_run_folder: the one folder of a benchmark, or an error if there is none or several;
  - VLQ: the resonator is always part of the layout — lowest / random / forced choose among the usable qubits.
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
from source.core.records.run_names import qpu_run_prefix
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
    assert "calibration_file" in capsys.readouterr().out
    assert _exits(["layout=random", "calibration_file=x.json"]) == 2      # a file only goes with optimized
    assert _exits(["layout=1,1,2"]) == 2                                  # a qubit twice
    assert _exits(["layout=1,2"]) == 2                                    # the circuits use exactly three qubits
    assert "exactly 3" in capsys.readouterr().out
    assert _exits(["layout=1,2,3,4"]) == 2
    assert _exits(["layout=middle"]) == 2
    assert _exits(["layout=random", "layout=lowest"]) == 2


def test_simulation_requires_layout_with_explanation(capsys):
    assert _exits([], calibration_required=True, layout_required=True, calibration_any_layout=True) == 2
    out = capsys.readouterr().out
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
    for form in ("submit", "submit_live", "submit_no_live", "simulate", "simulate_required", "collect"):
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
    out = capsys.readouterr().out
    assert "IBM reports live calibration data" in out and "simulate script" in out
    assert BA.parse_layout_args(["layout=random"], live_calibration="IBM").mode == "random"
    assert BA.parse_layout_args(["layout=random", "calibration_file=x.json"], calibration_any_layout=True).calibration_file == "x.json"
    assert BA.parse_layout_args(["layout=optimized", "calibration_file=x.json"], calibration_required=True).calibration_file == "x.json"


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
    assert newest_provider_file("PIAST-Q").name == "piastq_characterisation_2026-09-29.json"
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

def test_collect_finds_the_one_run_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(BA, "PROJECT_ROOT", tmp_path)
    base = qpu_run_prefix("IQM_EQE1", "EQE1", "mini", True, True)
    (tmp_path / "datafiles").mkdir()
    assert BA.find_collect_run_folder(base, BA.parse_layout_args(["layout=random"], collect=True)) == f"{base}_layout-random"
    with pytest.raises(SystemExit):
        BA.find_collect_run_folder(base, None)                      # none yet
    (tmp_path / "datafiles" / f"{base}_layout-lowest").mkdir()
    assert BA.find_collect_run_folder(base, None) == f"{base}_layout-lowest"
    (tmp_path / "datafiles" / f"{base}_layout-15-14-21").mkdir()
    with pytest.raises(SystemExit):
        BA.find_collect_run_folder(base, None)                      # several: name the layout
    assert "name the one with layout=" in capsys.readouterr().err
    la = BA.parse_layout_args(["layout=15,14,21"], collect=True)
    assert BA.find_collect_run_folder(base, la) == f"{base}_layout-15-14-21"


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
