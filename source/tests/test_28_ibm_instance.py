"""
test_28_ibm_instance.py
=======================
Unit tests (no QPU, no network, no qiskit_ibm_runtime needed) of the explicit IBM instance choice:

  - providers.pick_ibm_instance: the named instance (by name or CRN), else the single Open Plan instance; an unknown name,
    no Open Plan instance or several are refused with one ❌ line and exit code 2;
  - providers.connect_ibm (a fake qiskit_ibm_runtime stands in): one service, limited to Open Plan by default or built for the
    named instance; an unknown name is refused with the instances listed; a QPU the instance does not list is refused;
  - providers.ibm_instance_for_runs: a run's recorded instance wins, a different instance= for it is refused, runs on
    different instances are refused, a run without a record gives the requested one;
  - run_info: the instance fields are written, kept by a later write without them, and left out for other providers;
    scheduling_simulation is recorded and a mismatch is refused;
  - benchmark_args.parse_instance_arg, and the usage text of the IBM scripts.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import json
import types

import pytest

from source.core.qpu import providers as P
from source.core.records.run_info import (write_run_info, load_run_info, recorded_instance, recorded_scheduling_simulation,
                                          check_scheduling_simulation)
from source.core.runner_support import benchmark_args as BA

OPEN = dict(name="open-instance", plan="open", crn="crn:open")
PAID = dict(name="Test - 1 USD limit", plan="pay-as-you-go", crn="crn:paid")
EU   = dict(name="Test EU - 1 USD limit", plan="pay-as-you-go", crn="crn:eu")
CONFIG = dict(shots=10, samples=8, T_list=[0.2], dt=0.2, SPAM_correction=True, chunk_size=6)


def _exits(capsys, fn, *args):
    with pytest.raises(SystemExit) as e:
        fn(*args)
    assert e.value.code == 2
    return capsys.readouterr().err


# ── pick_ibm_instance ────────────────────────────────────────────────────────

def test_default_is_the_open_plan_instance_not_the_first_listed():
    assert P.pick_ibm_instance([PAID, OPEN, EU]) is OPEN


def test_named_instance_by_name_or_crn():
    assert P.pick_ibm_instance([OPEN, PAID], "Test - 1 USD limit") is PAID
    assert P.pick_ibm_instance([OPEN, PAID], "crn:paid") is PAID


def test_unknown_name_lists_the_instances(capsys):
    err = _exits(capsys, P.pick_ibm_instance, [OPEN, PAID], "nope")
    assert err.startswith("❌") and "'nope'" in err and "open-instance (open)" in err and "Test - 1 USD limit (pay-as-you-go)" in err


@pytest.mark.parametrize("instances, count", [([PAID, EU], "0"), ([OPEN, dict(OPEN, name="second", crn="crn:2")], "2")])
def test_no_single_open_plan_instance_needs_a_name(capsys, instances, count):
    err = _exits(capsys, P.pick_ibm_instance, instances)
    assert err.startswith("❌") and f"{count} IBM Open Plan instances" in err and "instance=NAME" in err


# ── connect_ibm ──────────────────────────────────────────────────────────────

@pytest.fixture
def fake_runtime(monkeypatch):
    made = []

    class Service:
        """Like the library: a name or CRN is resolved by the constructor (unknown: an error); without one, "auto" picks the
        first listed instance, or the first of the plans_preference."""
        def __init__(self, channel=None, token=None, instance=None, plans_preference=None):
            self.channel, self.token, self.plans_preference = channel, token, plans_preference
            known = [PAID, OPEN]
            if instance not in (None, "auto"):
                found = [i for i in known if instance in (i["name"], i["crn"])]
                if not found:
                    raise ValueError(f"The instance specified ({instance}) is not a valid instance name.")
                self.instance = found[0]["crn"]
            else:
                pool = [i for i in known if not plans_preference or i["plan"] in plans_preference]
                if not pool:
                    raise ValueError("No matching instances found")
                self.instance = pool[0]["crn"]
            made.append(self)
        def instances(self):
            return [PAID, OPEN]
        def active_instance(self):
            return self.instance
        def backends(self):
            return [types.SimpleNamespace(name=n) for n in ("ibm_fez", "ibm_kingston")]

    mod = types.ModuleType("qiskit_ibm_runtime")
    mod.QiskitRuntimeService = Service
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", mod)
    return made


def test_connect_ibm_uses_the_open_instance_by_default(fake_runtime, capsys):
    service, chosen = P.connect_ibm("tok", None, "ibm_fez")
    assert chosen is OPEN and service.instance == "crn:open" and service.channel == "ibm_quantum_platform"
    assert fake_runtime == [service] and service.plans_preference == ["open"]   # one service (one library log line), limited to Open Plan
    out = capsys.readouterr().out
    assert "Instance: open-instance (open)" in out
    assert "[chosen for this run]" in out
    assert "  Available instances:\n    - Test - 1 USD limit (pay-as-you-go)\n    - open-instance (open)\n" in out
    assert out.startswith("\n") and out.endswith("\n\n")   # set off from the library's log lines


def test_connect_ibm_with_a_named_instance(fake_runtime):
    service, chosen = P.connect_ibm("tok", "Test - 1 USD limit")
    assert chosen is PAID and service.instance == "crn:paid" and fake_runtime == [service]


def test_connect_ibm_unknown_name_lists_the_instances(fake_runtime, capsys):
    err = _exits(capsys, P.connect_ibm, "tok", "nope")
    assert err.startswith("❌") and "'nope'" in err and "open-instance (open)" in err


def test_connect_ibm_refuses_a_qpu_the_instance_does_not_list(fake_runtime, capsys):
    err = _exits(capsys, P.connect_ibm, "tok", None, "ibm_boston")
    assert err.startswith("❌") and "ibm_boston is not listed" in err and "ibm_fez, ibm_kingston" in err and "instance=NAME" in err


# ── a run's recorded instance ────────────────────────────────────────────────

def _run(root, name, instance=None, plan=None):
    write_run_info(CONFIG, root / name, lnn=True, instance=instance, instance_plan=plan)
    return name


def test_recorded_instance_wins_and_a_different_request_is_refused(tmp_path, capsys):
    a = _run(tmp_path, "run_a", "Test - 1 USD limit", "pay-as-you-go")
    assert P.ibm_instance_for_runs(None, [a], tmp_path) == "Test - 1 USD limit"
    assert P.ibm_instance_for_runs("Test - 1 USD limit", [a], tmp_path) == "Test - 1 USD limit"
    err = _exits(capsys, P.ibm_instance_for_runs, "open-instance", [a], tmp_path)
    assert err.startswith("❌") and "'Test - 1 USD limit'" in err and "instance=open-instance" in err


def test_runs_without_a_record_give_the_request(tmp_path):
    assert P.ibm_instance_for_runs(None, ["missing_run"], tmp_path) is None
    assert P.ibm_instance_for_runs("x", [_run(tmp_path, "no_instance")], tmp_path) == "x"


def test_runs_on_different_instances_are_refused(tmp_path, capsys):
    runs = [_run(tmp_path, "r1", "open-instance", "open"), _run(tmp_path, "r2", "Test - 1 USD limit", "pay-as-you-go")]
    err = _exits(capsys, P.ibm_instance_for_runs, None, runs, tmp_path)
    assert err.startswith("❌") and "different IBM instances" in err and "layout=" in err


# ── run_info ─────────────────────────────────────────────────────────────────

def test_instance_is_written_kept_by_a_later_write_and_left_out_for_other_providers(tmp_path, monkeypatch):
    monkeypatch.setattr("source.core.records.run_info.PROJECT_ROOT", tmp_path.parent)
    run = tmp_path / "run_x"
    write_run_info(CONFIG, run, lnn=True, instance="open-instance", instance_plan="open")
    write_run_info(CONFIG, run, lnn=True)                                   # the per-chunk submit() call: no instance given
    assert recorded_instance("run_x", tmp_path) == ("open-instance", "open")
    other = tmp_path / "run_y"
    write_run_info(CONFIG, other, lnn=True)
    assert "instance" not in json.loads((other / "run_info.json").read_text(encoding="utf-8"))
    assert recorded_instance("run_y", tmp_path) == (None, None)
    assert load_run_info(f"../{tmp_path.name}/run_x").instance == "open-instance"   # PROJECT_ROOT/datafiles/../<tmp>/run_x


# ── arguments and usage ──────────────────────────────────────────────────────

def test_parse_instance_arg():
    assert BA.parse_instance_arg(["backend=ibm_fez", "mini"]) is None
    assert BA.parse_instance_arg(["instance=Test - 1 USD limit", "mini"]) == "Test - 1 USD limit"


def test_parse_instance_arg_refuses_twice_or_empty(capsys):
    for argv in (["instance=a", "instance=b"], ["instance= "]):
        assert "❌" in _exits(capsys, BA.parse_instance_arg, argv)


def test_ibm_usage_lists_the_instance_argument():
    script = BA.PROJECT_ROOT / "source/benchmark_scripts/real_qpu_scripts/IBM/qbcs_real_submit_ibm.py"
    usage = BA.make_usage(script, "d", layout="submit_live", instance=True)
    assert "[instance=NAME]" in usage.splitlines()[0] and "instance=NAME" in usage and "Open Plan" in usage
    assert "instance" not in BA.make_usage(script, "d", layout="submit_live").lower()


def test_scheduling_simulation_is_recorded_and_a_mismatch_is_refused(tmp_path, capsys):
    run = tmp_path / "run_s"
    write_run_info(CONFIG, run, lnn=True, scheduling_simulation=False)
    assert recorded_scheduling_simulation("run_s", tmp_path) is False
    check_scheduling_simulation("run_s", False, tmp_path)
    check_scheduling_simulation("no_such_run", True, tmp_path)               # no manifest: a new run, nothing checked
    with pytest.raises(SystemExit) as e:
        check_scheduling_simulation("run_s", True, tmp_path)
    assert e.value.code == 2 and "scheduling_simulation=false" in capsys.readouterr().err
    other = tmp_path / "run_t"
    write_run_info(CONFIG, other, lnn=True)                                  # no record: left out of the manifest, counts as True
    assert "scheduling_simulation" not in json.loads((other / "run_info.json").read_text(encoding="utf-8"))
    check_scheduling_simulation("run_t", True, tmp_path)
    with pytest.raises(SystemExit):
        check_scheduling_simulation("run_t", False, tmp_path)
