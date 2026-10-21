"""
benchmark_args.py
The command-line interface shared by every script in source/benchmark_scripts/ (and by
source/analysis_scripts/run_post_process.py): which benchmark to run, and the tag its run folders carry.

    <script>.py mini | full                                  a preset
    <script>.py mini nshots_per_circuit=N                    a preset with N shots per circuit
    <script>.py nshots_per_circuit=N n_samples=M n_depths=K  custom: Nt = 1, 3, …, 2K-1
    <script>.py nshots_per_circuit=N n_samples=M nt_list=1,5,9   custom depth list

There is no default benchmark: a script called without arguments prints its usage and does nothing
(see usage_if_requested); the benchmark to run must always be named.

The submit and simulate scripts also take the qubit layout (parse_layout_args):

    layout=optimized | random | lowest | 15,14,21       which physical qubits the run uses
    calibration_file=PATH                                the calibration to score layout=optimized with (the simulate
                                                         scripts also take it with any layout: the simulation's noise data)

The simulate scripts that use a QPU's noise (FakeFez, qbcs_real_simulate_*.py) also take the optional
scheduling_simulation=true|false (parse_scheduling_arg(), default true): whether the circuits are scheduled, so that idle qubits relax.

The collect and job-status scripts take layout= too (parse_layout_args(collect=True)), to find the run
folder, which ends in _layout-<mode>_<YYYY-MM-DD>_#<N> (or _layout-15-14-21_<date>_#<N>): the latest one, or that
of date=YYYY-MM-DD (the day's latest run) or date=YYYY-MM-DD_#N (parse_date_arg()). Without layout= and date=
every run of the benchmark with something left to collect is handled (find_collect_run_folders()).

The submit scripts take the word --force-new-run (pop_force_new_run_flag()): start a new run folder even if
the latest run of those settings is unfinished. The EQE1 and VLQ submit scripts also take the word
--retry-health-check (pop_retry_health_check_flag(), not in the usage; their *_with_retries.py wrappers pass it on): resubmit only the health check after
the last chunks of the latest run, which are already submitted.

No qiskit and no myQLM (only numpy): every venv imports it.
"""

import json
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from source.core.common import PROJECT_ROOT, exit_with_error
from source.core.records.run_names import (benchmark_field, layout_field, run_name_with_any_layout, run_name_with_layout,
                                           latest_run_folder, parse_run_name)

DT = 0.2   # Trotter step; evolution time = DT * Nt

PRESETS = {
    "full": dict(nshots=10,  sample_size=480, nt_list=list(range(1, 20, 2))),   # Nt = 1…19
    "mini": dict(nshots=200, sample_size=8,   nt_list=[1, 3]),
}

BENCHMARK_USAGE = """mini                   the mini benchmark: 8 samples, Nt = 1, 3, 200 shots per circuit
  full                   the full benchmark: 480 samples, Nt = 1…19, 10 shots per circuit
  nshots_per_circuit=N   with mini or full: N shots per circuit instead of the preset's (N a positive integer)
  nshots_per_circuit=N n_samples=M n_depths=K
                         a custom benchmark: M samples, Nt = 1, 3, …, 2K-1
  nshots_per_circuit=N n_samples=M nt_list=1,5,9
                         a custom benchmark: M samples, the given increasing odd Nt values
  (A preset's samples and depths cannot be changed. n_samples must be a multiple of 8 for uniform
    SPAM (readout error) calibration across all possible bitstrings on 3 qubits.)"""

LAYOUT_USAGE = """layout=optimized       the qubits with the best calibrated success probability, idle T1/T2 decay included (scorer in source/core/qpu/layout_selection.py; IQM VLQ: source/core/qpu/iqm_vlq.py)
  layout=random          any valid 3-qubit layout on the QPU, drawn once at random (the seed is recorded in the run)
  layout=lowest          the lexicographically lowest valid layout
  layout=15,14,21        exactly these three physical qubits (virtual qubit i -> the i-th number); an error if the QPU cannot run them
  calibration_file=PATH  with layout=optimized: a calibration JSON, either already in the project's snapshot format or
                         as the provider publishes it (converted automatically); PATH is relative to the project root
                         (IQM VLQ: a run's vlq_calibration.json, or that run's calibration_snapshot.json with its vlq_calibration.json beside it)"""

LAYOUT_USAGE_NO_LIVE_CALIBRATION = """  AQT PIAST-Q and IQM EQE1 report no calibration data, so a live run cannot be optimized without a calibration file:
  without calibration_file= the layout is `random` (a notice is printed); layout=optimized requires calibration_file=.
  The latest provider files are in calibration_input_files/."""

LAYOUT_REQUIRED_EXPLANATION = """AQT PIAST-Q and IQM EQE1 report no calibration data through their SDKs, so a real submission uses a RANDOM layout unless a
calibration_file= is given. A simulation must therefore be told which layout to mimic — name it explicitly:
  layout=random                              what a real submission without a calibration file does
  layout=optimized calibration_file=calibration_input_files/<provider>/<file>.json    what a real submission with that file does
  layout=lowest, or layout=15,14,21          a fixed layout"""

LIVE_CALIBRATION_EXPLANATION = ("calibration_file= is not accepted here: {qpu} reports live calibration data, and a real run "
    "always uses it (saved in the run folder as calibration_snapshot.json), so the layout matches the device as it is "
    "now. To simulate with an older calibration, give calibration_file= to the simulate script instead.")

SCHEDULING_USAGE = """scheduling_simulation=true | false   (optional, default true)
                         true (default): the circuits are scheduled, so idle qubits relax through T1/T2 as on the QPU (slower: about 4 times the
                         time of false on the full benchmark); false: no scheduling, so idle-qubit relaxation is left out (faster)"""

HELP_USAGE = "  --help                 print this message and exit"

USAGE = f"""usage: <script>.py BENCHMARK

  BENCHMARK — which benchmark (required):
  {BENCHMARK_USAGE}

{HELP_USAGE}"""   # fallback of the argument errors of a script that has not set its own usage

_active_usage = None   # the running script's own usage, set by usage_if_requested(); what its argument errors print


INSTANCE_USAGE = """instance=NAME          the IBM Quantum Platform instance (default: the account's Open Plan instance, which is free).
                         The QPUs listed differ per instance, and a job is found only through the instance that received it,
                         so a run's recorded instance is always used: a different instance=NAME for an existing run is an error."""


def make_usage(script_file, description, backend=True, benchmark=True, health_check_flag=False, layout=None, instance=False):
    """The uniform usage text of a script in source/benchmark_scripts/: the command line, the description, then
    the sections that apply — BACKEND (names from the folder's backends.json), BENCHMARK, for the
    job-status scripts (health_check_flag) HEALTH CHECK, and LAYOUT. `layout` selects its form:
      None             no layout section
      "submit_live"    layout= only (default optimized): the QPU's live calibration, no calibration_file= (IBM, Resonance, VLQ)
      "submit_no_live" the same for QPUs without live calibration (AQT, EQE1): random unless a file is given
      "simulate"       layout= (default optimized) and calibration_file=, which is the simulation's noise data under any layout
      "simulate_builtin"  layout= (default optimized), no calibration_file=: the simulated QPU's calibration is bundled (FakeFez)
      "simulate_required"  layout= is required (simulating AQT, EQE1); calibration_file= defaults to the newest in calibration_input_files/
      The three simulate forms also show the optional [scheduling_simulation=true|false] (parse_scheduling_arg()).
      "collect"       layout= and date= are optional: they name the run folder; without them every run of the
                       benchmark with something left to collect is handled
    `instance` adds the optional instance=NAME of the IBM scripts."""
    script = Path(script_file).resolve()
    path = script.relative_to(Path(__file__).resolve().parents[3]).as_posix()
    bench = "BENCHMARK" if not health_check_flag else "(BENCHMARK | health_check)"
    lay_syn = {"submit_live": "[LAYOUT]", "submit_no_live": "[LAYOUT]", "simulate": "[LAYOUT]", "simulate_builtin": "[LAYOUT]", "simulate_required": "LAYOUT", "collect": "[layout=...]"}.get(layout, "")
    scheduling = layout in ("simulate", "simulate_builtin", "simulate_required")   # the QPU-noise simulations
    synopsis = " ".join(filter(None, ["backend=NAME" if backend else "", "[instance=NAME]" if instance else "", bench if benchmark else "", lay_syn,
                                      "[scheduling_simulation=true|false]" if scheduling else ""]))
    blocks = [f"usage: python {path} {synopsis}",
              textwrap.fill(description, width=110, initial_indent="  ", subsequent_indent="  ")]
    if backend:
        backends = json.loads((script.parent / "backends.json").read_text(encoding="utf-8"))["backends"]
        blocks.append("  BACKEND — the QPU to use (required):\n"
                      f"  backend=NAME           one of: {', '.join(backends)}")
    if instance:
        blocks.append(f"  INSTANCE — the IBM instance (optional):\n  {INSTANCE_USAGE}")
    if benchmark:
        what = " to run (required):" if not health_check_flag else "'s run to check (required, unless health_check is given):"
        blocks.append(f"  BENCHMARK — which benchmark{what}\n  {BENCHMARK_USAGE}")
    if health_check_flag:
        blocks.append("  HEALTH CHECK — the standalone health check instead of a benchmark run:\n"
                      "  health_check           check the backend's standalone health-check job; no BENCHMARK is needed")
    if layout == "collect":
        blocks.append("  LAYOUT — which run folder (…_layout-<mode>); optional: without layout= and date=, every run of the\n"
                      "  benchmark with something left to collect is handled, all layouts and dates:\n"
                      "  layout=optimized | random | lowest | 15,14,21\n\n"
                      "  DATE — which run of these settings (the folder's _<YYYY-MM-DD>_#<N>); optional, default the latest:\n"
                      "  date=YYYY-MM-DD        the latest run of that day\n"
                      "  date=YYYY-MM-DD_#N     that run\n"
                      "  date= alone covers every layout with a run of that date; with layout=, only that layout")
    elif layout:
        head = {"submit_live": "(optional, default optimized)",
                "submit_no_live": "(optional; random unless calibration_file= is given)",
                "simulate": "(optional, default optimized)", "simulate_builtin": "(optional, default optimized)",
                "simulate_required": "(required)"}[layout]
        text = f"  LAYOUT — the physical qubits the run uses {head}:\n  {LAYOUT_USAGE}"
        if layout == "submit_live":   # live calibration only: no calibration_file= line
            text = text[:text.index("\n  calibration_file=PATH")] + (
                "\n  (The layout is scored with the QPU's live calibration; calibration_file= is not accepted for a real run —"
                "\n  give it to the simulate script to simulate with an older calibration.)")
        if layout == "simulate_builtin":   # a bundled calibration (FakeFez): no calibration_file= line
            text = text[:text.index("\n  calibration_file=PATH")] + (
                "\n  (The layout is scored with the simulated QPU's built-in calibration; calibration_file= is not accepted.)")
        if layout in ("simulate", "simulate_required"):
            text += ("\n  (In a simulation calibration_file= is also the noise data of the simulated QPU, whatever the layout; "
                     + ("without it the newest file in calibration_input_files/ is used.)" if layout == "simulate_required"
                        else "without it the QPU's calibration is downloaded.)"))
        if layout in ("submit_no_live", "simulate_required"):
            text += "\n\n" + (LAYOUT_USAGE_NO_LIVE_CALIBRATION if layout == "submit_no_live"
                              else textwrap.indent(LAYOUT_REQUIRED_EXPLANATION, "  "))
        blocks.append(text)
        if scheduling:
            blocks.append(f"  SCHEDULING — whether idle qubits relax in the simulation (optional, default true):\n  {SCHEDULING_USAGE}")
    if layout in ("submit_live", "submit_no_live"):
        blocks.append("  RUN FOLDER — by default the latest unfinished run of these settings is continued:\n"
                      "  --force-new-run        start a new run folder even if the latest run of these settings is unfinished")
    blocks.append(HELP_USAGE)
    return "\n\n".join(blocks)


def usage_if_requested(argv, usage):
    """Print `usage` and exit — doing nothing else — when the script is called without arguments (to stderr,
    exit code 2: a batch job that was given nothing to do must not look successful) or with --help / -h
    (to stdout, exit code 0). Otherwise remember it as the usage that the argument errors print."""
    global _active_usage
    _active_usage = usage
    if "--help" in argv or "-h" in argv:
        print(usage)
        sys.exit(0)
    if not argv:
        print(usage, file=sys.stderr)
        sys.exit(2)


@dataclass
class BenchmarkArgs:
    nshots:      int
    sample_size: int
    nt_list:     list
    tag:         str    # the <benchmark> field of every run-folder name: full, mini+nshots-100, custom+ndepths-3+nsamples-8+nshots-50, … (core/records/run_names.py)
    preset:      str = None

    @property
    def list_times(self):
        """Evolution times. For Nt = 1, 3, 5, … these are exactly the values of the former
        np.arange(0.2, T_max, 0.4) constants, so existing runs and caches keep matching."""
        k = len(self.nt_list)
        if self.nt_list == list(range(1, 2 * k, 2)):
            return [float(t) for t in np.arange(DT, DT * 2 * k, 2 * DT)[:k]]
        return [DT * nt for nt in self.nt_list]

    @property
    def n_depths(self):
        return len(self.nt_list)


FORCE_NEW_RUN_WORD = "--force-new-run"   # bare command-line word of the submit scripts: start a new run folder


def pop_force_new_run_flag(argv):
    """(whether `--force-new-run` is in argv, argv without it) — to be called before the benchmark/backend
    argument parsers, which do not know the word."""
    return FORCE_NEW_RUN_WORD in argv, [a for a in argv if a != FORCE_NEW_RUN_WORD]


RETRY_HEALTH_CHECK_WORD = "--retry-health-check"   # bare word of the EQE1 / VLQ submit scripts, given by their *_with_retries.py


def pop_retry_health_check_flag(argv):
    """(whether `--retry-health-check` is in argv, argv without it) — like pop_force_new_run_flag()."""
    return RETRY_HEALTH_CHECK_WORD in argv, [a for a in argv if a != RETRY_HEALTH_CHECK_WORD]


def _fail(msg):
    print(f"❌ {msg}\n\n{_active_usage or USAGE}", file=sys.stderr)
    sys.exit(2)


def _positive_int(key, value):
    if not re.fullmatch(r"\d+", value) or int(value) <= 0:
        _fail(f"{key} must be a positive integer, got {value!r}")
    return int(value)


def parse_benchmark_args(argv, extra_keys=(), required=True, layout_args=True):
    """BenchmarkArgs from the command-line arguments (sys.argv[1:]). Prints the usage and exits
    with code 2 on anything invalid. `extra_keys`: other key=value arguments the caller handles
    itself (e.g. "backend"); they are skipped here. The benchmark has no default: without any (and
    required=True) it is an error; with required=False the result is None. --help / -h prints the
    usage and exits with code 0. `layout_args`: also accept (and skip) layout= and calibration_file=,
    which parse_layout_args() handles; the noiseless and coherent-noise local scripts pass False and reject them."""
    if "--help" in argv or "-h" in argv:
        print(_active_usage or USAGE)
        sys.exit(0)
    layout_keys = ("layout", "calibration_file") if layout_args else ()
    presets = [a for a in argv if a in PRESETS]
    kv = {}
    for a in argv:
        if a in PRESETS:
            continue
        m = re.fullmatch(r"(\w+)=(.+)", a)
        if not m or m.group(1) not in ("nshots_per_circuit", "n_samples", "n_depths", "nt_list", *layout_keys, *extra_keys):
            _fail(f"unknown argument {a!r}")
        if m.group(1) in kv:
            _fail(f"{m.group(1)} given twice")
        kv[m.group(1)] = m.group(2)
    for k in (*extra_keys, *layout_keys):   # parse_layout_args() handles these
        kv.pop(k, None)
    if len(presets) > 1:
        _fail("give at most one of full / mini")

    nshots = _positive_int("nshots_per_circuit", kv["nshots_per_circuit"]) if "nshots_per_circuit" in kv else None

    if not presets and not kv:
        if not required:
            return None
        _fail("no benchmark given — name the benchmark: mini, full, or a custom one")

    if presets:
        name = presets[0]
        if {"n_samples", "n_depths", "nt_list"} & kv.keys():
            _fail(f"the {name} preset's samples and depths cannot be changed — only nshots_per_circuit")
        p = PRESETS[name]
        tag = benchmark_field(preset=name, nshots=nshots)
        return BenchmarkArgs(nshots or p["nshots"], p["sample_size"], list(p["nt_list"]), tag, name)

    if nshots is None or "n_samples" not in kv or ("n_depths" in kv) == ("nt_list" in kv):
        _fail("a custom benchmark needs nshots_per_circuit, n_samples and exactly one of n_depths / nt_list")
    n_samples = _positive_int("n_samples", kv["n_samples"])
    if n_samples % 8:
        _fail(f"n_samples must be a multiple of 8 (every input bitstring of the 3 qubits equally often), got {n_samples}")
    if "n_depths" in kv:
        k = _positive_int("n_depths", kv["n_depths"])
        return BenchmarkArgs(nshots, n_samples, list(range(1, 2 * k, 2)),
                             benchmark_field(nshots=nshots, n_samples=n_samples, n_depths=k))
    try:
        nt_list = [int(x) for x in kv["nt_list"].split(",")]
    except ValueError:
        _fail(f"nt_list must be comma-separated odd integers, got {kv['nt_list']!r}")
    if any(nt <= 0 or nt % 2 == 0 for nt in nt_list) or nt_list != sorted(set(nt_list)):
        _fail(f"nt_list must be increasing, distinct, positive odd integers, got {kv['nt_list']!r}")
    return BenchmarkArgs(nshots, n_samples, nt_list, benchmark_field(nshots=nshots, n_samples=n_samples, nt_list=nt_list))


def parse_scheduling_arg(argv):
    """The optional scheduling_simulation=true|false argument of the simulate scripts that use a QPU's noise (FakeFez,
    qbcs_real_simulate_*.py), as a bool: whether the circuits are transpiled scheduled (scheduling_method="alap"), so that
    idle qubits relax through T1/T2 (true), or not (false: idle-qubit relaxation is left out; much faster). Missing: True. Any
    other value, or a duplicate: the usage is printed and the script exits with code 2. Those scripts pass "scheduling_simulation"
    in parse_benchmark_args(extra_keys=...)."""
    values = [a.split("=", 1)[1] for a in argv if a.startswith("scheduling_simulation=")]
    if len(values) > 1:
        _fail("scheduling_simulation given twice")
    if not values:
        return True
    if values[0] not in ("true", "false"):
        _fail(f"scheduling_simulation must be true or false, got {values[0]!r}")
    return values[0] == "true"


def parse_backend_arg(argv, default, allowed=None, others_ok=False):
    """The backend=NAME argument; `default` without it, or — with default=None — a usage line and
    exit code 2 (the argument is then required). `allowed`: the only accepted names (e.g.
    ["PIAST-Q"]). Any other argument prints a usage line and exits with code 2, unless
    others_ok (the benchmark scripts, whose other arguments parse_benchmark_args() checks)."""
    backend = default
    usage = ("usage: <script>.py backend=NAME" if default is None
             else f"usage: <script>.py [backend=NAME]   (default: {default})")
    if allowed is not None:
        usage += f"   (available: {', '.join(sorted(allowed))})"
    usage = _active_usage or usage
    for a in argv:
        m = re.fullmatch(r"backend=(.+)", a)
        if m:
            backend = m.group(1)
        elif not others_ok:
            print(f"❌ unknown argument {a!r}\n{usage}", file=sys.stderr)
            sys.exit(2)
    if backend is None:
        print(f"❌ no backend given — the QPU to run on must be named explicitly\n{usage}", file=sys.stderr)
        sys.exit(2)
    if allowed is not None:
        require_known_backend(backend, allowed)
    return backend


def parse_backend_config(argv, script_dir, others_ok=True):
    """The required backend=NAME argument, checked against <script_dir>/backends.json — the table
    of the QPUs a provider folder's scripts may run on, with each one's settings ("lnn": whether it
    needs linear-nearest-neighbour circuits rather than all-to-all ones). Returns (name, settings).
    There is no default: the QPU must always be named."""
    table = json.loads((Path(script_dir) / "backends.json").read_text(encoding="utf-8"))
    name = parse_backend_arg(argv, None, allowed=table["backends"], others_ok=others_ok)
    return name, table["backends"][name]


def require_known_backend(name, available, where="this provider"):
    """Exit with code 2 unless `name` is one of `available` — call it before anything is submitted,
    so a mistyped device name never reaches a provider that might pick a default device instead."""
    available = sorted(available)
    if name not in available:
        print(f"❌ unknown backend {name!r} for {where} — available: {', '.join(available) or 'none'}", file=sys.stderr)
        sys.exit(2)


# ── Qubit layout ───────────────────────────────────────────────────────────────

LAYOUT_MODES    = ("optimized", "random", "lowest")
LAYOUT_N_QUBITS = 3   # the benchmark circuits act on exactly three qubits


@dataclass
class LayoutArgs:
    mode:             str            # optimized | random | lowest | list
    qubits:           list = None    # mode == "list": the physical qubits, virtual qubit i -> qubits[i]
    calibration_file: str  = None    # the calibration to score mode == "optimized" with (as given; relative to the project root); a simulate script accepts it with any mode (the simulation's noise data)
    notice:           str  = None    # printed by the script when the mode was chosen by default

    @property
    def tag(self):
        """The last part of the run-folder name: layout-optimized, layout-random, layout-lowest, layout-15-14-21."""
        return layout_field(self.mode, self.qubits)


def _layout_value(value):
    """(mode, qubits) from a layout= value: a mode name, or physical qubits as 15,14,21 / [15,14,21]."""
    if value in LAYOUT_MODES:
        return value, None
    m = re.fullmatch(r"\[?\s*(\d+(?:\s*,\s*\d+)*)\s*\]?", value)
    if not m:
        _fail(f"layout must be one of {', '.join(LAYOUT_MODES)} or a list of physical qubits like 15,14,21, got {value!r}")
    qubits = [int(x) for x in m.group(1).split(",")]
    if len(set(qubits)) != len(qubits):
        _fail(f"layout lists a qubit twice: {qubits}")
    if len(qubits) != LAYOUT_N_QUBITS:
        _fail(f"layout must list exactly {LAYOUT_N_QUBITS} physical qubits (the benchmark circuits use {LAYOUT_N_QUBITS}), got {len(qubits)}: {qubits}")
    return "list", qubits


def parse_layout_args(argv, calibration_required=False, layout_required=False, collect=False, calibration_any_layout=False,
                      live_calibration=None, no_calibration_file=False):
    """The layout= and calibration_file= arguments (sys.argv[1:]); prints the usage and exits with code 2
    on anything invalid. Returns a LayoutArgs.

      no_calibration_file   a simulation with a bundled calibration (benchmark_fake_fez_lnn.py): calibration_file= is refused.

      live_calibration      a real submission to a QPU that reports its calibration (its name, e.g. "IBM"): calibration_file=
                            is refused with an explanation — the run always uses the live calibration.

      calibration_required  the QPU reports no calibration (AQT, EQE1): layout=optimized needs calibration_file=, and
                            without either argument the layout is `random`, with a notice.
      layout_required       layout= must be given (simulating AQT / EQE1); its absence prints the explanation.
      calibration_any_layout  simulation scripts: calibration_file= is the simulated QPU's noise data, so it is allowed
                            with any layout (it only drives the scoring for layout=optimized).
      collect               for the collect / job-status scripts: layout= only, to name the run folder; the result is
                            None when it is absent (find_collect_run_folders() then handles every layout).

    Otherwise (a QPU with live calibration, a simulation, or a submission to AQT / EQE1 with a calibration_file=) the
    default is layout=optimized. A relative calibration_file is relative to the
    project root, where every script runs. See source/core/qpu/layout_selection.py and source/core/qpu/calibration_input.py."""
    kv = {}
    for a in argv:
        m = re.fullmatch(r"(layout|calibration_file)=(.+)", a)
        if m:
            if m.group(1) in kv:
                _fail(f"{m.group(1)} given twice")
            kv[m.group(1)] = m.group(2)
    if collect:
        if "calibration_file" in kv:
            _fail("calibration_file is not needed here — the layout only names the run folder")
        return LayoutArgs(*_layout_value(kv["layout"])) if "layout" in kv else None

    cal = kv.get("calibration_file")
    if no_calibration_file and cal is not None:
        _fail("calibration_file= is not accepted here: this simulation uses FakeFez's built-in calibration")
    if live_calibration and cal is not None:
        _fail(LIVE_CALIBRATION_EXPLANATION.format(qpu=live_calibration))
    if "layout" not in kv:
        if layout_required:
            _fail("layout is required for this script.\n\n" + LAYOUT_REQUIRED_EXPLANATION)
        if calibration_required and cal is None:
            return LayoutArgs("random", notice=(
                "ℹ️  layout not given and no calibration_file: this QPU reports no calibration data, so the layout is "
                "RANDOM. To optimize the layout instead, add calibration_file=calibration_input_files/<provider>/<file>.json "
                "(the latest provider files are in calibration_input_files/)."))
        mode, qubits = "optimized", None
    else:
        mode, qubits = _layout_value(kv["layout"])
    if cal is not None and mode != "optimized" and not calibration_any_layout:
        _fail(f"calibration_file only goes with layout=optimized (the layout is {mode})")
    if mode == "optimized" and calibration_required and cal is None and not calibration_any_layout:
        _fail("layout=optimized needs calibration_file=...: this QPU reports no calibration data, so there is nothing to "
              "score the layouts with. The latest provider files are in calibration_input_files/; or use layout=random.")
    return LayoutArgs(mode, qubits, cal)


def parse_date_arg(argv):
    """The date=YYYY-MM-DD or date=YYYY-MM-DD_#N argument of a collect / job-status script (None without one):
    which dated run folder to read (the day's latest run, or that run). Those scripts pass "date" in parse_benchmark_args(extra_keys=...)."""
    dates = [a.split("=", 1)[1] for a in argv if a.startswith("date=")]
    if len(dates) > 1:
        _fail("date given twice")
    if dates and not re.fullmatch(r"\d{4}-\d{2}-\d{2}(_#[1-9]\d*)?", dates[0]):
        _fail(f"date must be YYYY-MM-DD or YYYY-MM-DD_#N, got {dates[0]!r}")
    return dates[0] if dates else None


def parse_instance_arg(argv):
    """The instance=NAME argument of the IBM scripts (None without one): the IBM Quantum Platform instance to use
    (source/core/qpu/providers.py: connect_ibm(); default the account's Open Plan instance). The IBM scripts pass
    "instance" in parse_benchmark_args(extra_keys=...). A run's own recorded instance wins (ibm_instance_for_runs())."""
    names = [a.split("=", 1)[1] for a in argv if a.startswith("instance=")]
    if len(names) > 1:
        _fail("instance given twice")
    if names and not names[0].strip():
        _fail("instance is empty")
    return names[0] if names else None


def parse_run_folder_arg(argv, undated):
    """The optional run_folder=<name> argument of the AQT loop script (None without one): pins the run folder
    (the sbatch files resolve it once per job). It must be a dated run name with the settings `undated`
    (qpu_run_name()), else the usage is printed and the script exits with code 2."""
    values = [a.split("=", 1)[1] for a in argv if a.startswith("run_folder=")]
    if len(values) > 1:
        _fail("run_folder given twice")
    if not values:
        return None
    try:
        name = parse_run_name(values[0])
    except ValueError as e:
        _fail(f"run_folder: {e}")
    if not name.date or str(name) != values[0] or name.undated != undated:
        _fail(f"run_folder {values[0]!r} is not a dated run folder of these settings ({undated}_<YYYY-MM-DD>_#<N>)")
    return values[0]


def find_collect_run_folder(base, layout, date=None):
    """The run folder of a collect / job-status script: `base` (the name up to and including the benchmark
    tag) + _layout-<mode> + _<YYYY-MM-DD>_#<N>: the latest such folder, or that of `date` (YYYY-MM-DD or YYYY-MM-DD_#N). With layout None, the
    one layout on disk — an error (exit code 2) if there is none or several (then name the layout); the collect and
    job-status scripts call find_collect_run_folders() instead, which handles every layout.
    Run from the project root, as every script does."""
    root = PROJECT_ROOT / "datafiles"
    if layout is None:
        layouts = sorted({parse_run_name(p.name).undated for p in root.glob(run_name_with_any_layout(base))
                          if p.is_dir() and parse_run_name(p.name).date})
        if len(layouts) != 1:
            exit_with_error(f"no run folder datafiles/{run_name_with_any_layout(base)}" if not layouts else
                            f"run folders of several layouts match: {', '.join(layouts)} — name the one with "
                            "layout=optimized | random | lowest | 15,14,21")
        undated = layouts[0]
    else:
        undated = run_name_with_layout(base, layout.tag)
    found = latest_run_folder(undated, root, date)
    if found is None:
        exit_with_error(f"no run folder datafiles/{undated}_{date or '<YYYY-MM-DD>_#<N>'}")
    return found


def find_collect_run_folders(base, layout, date=None):
    """The run folders (names, oldest first) a collect / job-status script handles. With `layout` given (with or
    without `date`): [find_collect_run_folder(base, layout, date)] (a finished run can be collected again). With
    `date` alone: the run of that date (YYYY-MM-DD: its latest, YYYY-MM-DD_#N: that number) of every layout on disk
    that has one, finished or not; an error (exit code 2) if none does. With neither: every dated folder of the
    benchmark `base`, whatever its layout, that run_has_uncollected() — a job or health check left to collect; an
    error (exit code 2) if there is none, naming what was found.
    Run from the project root, as every script does."""
    if layout is not None:
        return [find_collect_run_folder(base, layout, date)]
    from source.core.records.run_names import run_has_uncollected
    root = PROJECT_ROOT / "datafiles"
    pattern = run_name_with_any_layout(base)
    if date is not None:
        layouts = sorted({parse_run_name(p.name).undated for p in root.glob(pattern)
                          if p.is_dir() and parse_run_name(p.name).date})
        folders = [f for f in (latest_run_folder(u, root, date) for u in layouts) if f is not None]
        if not folders:
            exit_with_error(f"no run folder of date {date} in datafiles/{pattern}"
                            + (f" (layouts found: {', '.join(layouts)})" if layouts else ""))
        return sorted(folders, key=lambda n: ((r := parse_run_name(n)).date, r.run_number or 0))
    found = []
    for p in root.glob(pattern):
        if p.is_dir() and (name := parse_run_name(p.name)).date:
            found.append(((name.date, name.run_number or 0), p.name))
    folders = [n for _, n in sorted(found)]
    if not folders:
        exit_with_error(f"no run folder datafiles/{pattern}")
    todo = [f for f in folders if run_has_uncollected(f, root)]
    if not todo:
        exit_with_error(f"nothing left to collect in {', '.join(folders)}; name one with date=YYYY-MM-DD_#N or "
                        "layout=optimized | random | lowest | 15,14,21")
    return todo
