"""
common.py
myQLM-free helpers shared by the pipeline: run/circuit-cache folder paths,
NEC target bitstrings, path scrubbing (scrub_paths), exit_with_error, rel_path,
the print helpers (section, print_done_block, print_run_status_banner, print_chunk_saved, format_retry),
read_json_or_none, require_myqlm and the circuit-cache config check. Split out of main.py (which re-exports them) so that
qbcs.py and the modules it uses can be imported without
loading myQLM — the real-QPU venvs only need these, never the circuit builders.
"""

import json
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LAUNCHER = "py" if sys.platform == "win32" else "python3"   # how the docs say to start a script, in the activated venv (usage lines print it)

# Every circuit cache lives under circuits_root(). The test suite (source/tests/conftest.py,
# and only it, besides test_19, which sets it with monkeypatch to test circuits_root() itself)
# sets BACQ_CIRCUITS_ROOT so that its caches go to datafiles/tests/circuits/
# and never touch the ones real runs use.
CIRCUITS_ROOT_ENV     = "BACQ_CIRCUITS_ROOT"
DEFAULT_CIRCUITS_ROOT = "datafiles/circuits"


def circuits_root():
    """Folder (relative to the project root, forward slashes) holding every circuit cache:
    datafiles/circuits, or $BACQ_CIRCUITS_ROOT when the test suite set it."""
    return os.environ.get(CIRCUITS_ROOT_ENV) or DEFAULT_CIRCUITS_ROOT


def scrub_paths(text):
    """`text` without absolute paths of this machine: the project folder removed, the home folder
    written as `~` (both slash forms). For error messages saved into records. None stays None."""
    if text is None:
        return None
    text = str(text)
    for root, repl in ((PROJECT_ROOT, ""), (Path.home(), "~")):
        for form in (str(root), root.as_posix()):
            text = text.replace(form + "/", repl and repl + "/").replace(form + "\\", repl and repl + "\\").replace(form, repl)
    return text


def exit_with_error(message):
    """End the script with `❌ message` on stderr and exit code 2 — for a problem the user can fix
    (a missing token or file, an unusable argument), instead of a traceback. Every such exit in
    the suite goes through here, so they all look and behave the same."""
    print(f"❌ {message}", file=sys.stderr)
    sys.exit(2)


def rel_path(path):
    """`path` for printing: relative to the project root, with forward slashes
    (e.g. datafiles/<run folder>/job_records). Run output is saved too (AQT's
    output.log, Slurm logs), so it must not hold absolute paths. A path outside
    the project is returned unchanged."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except (ValueError, OSError):
        return str(path)


RULE_WIDTH = 60


def section(title):
    """Print a section header — a blank line, then `── title ───…` filled to RULE_WIDTH — so that the
    status check, the health-check summary and the post-processing all separate their blocks the same way."""
    head = f"── {title} "
    print(f"\n{head}{'─' * max(RULE_WIDTH - len(head), 3)}")


def print_done_block(title, rows):
    """Print the closing block of a script: `✅ title`, then one aligned `   Label:  value` row per
    (label, value) in `rows`. Rows whose value is None are left out."""
    rows = [(label, value) for label, value in rows if value is not None]
    width = max((len(label) for label, _ in rows), default=0) + 1
    print(f"\n✅ {title}")
    for label, value in rows:
        print(f"   {(label + ':').ljust(width)} {value}")


def print_run_status_banner(title, lines, complete):
    """Print the verdict on a run at the end of a collection, between two ruled lines so it stands out in a long
    log: `✅ title` for a complete run, `⚠️  title` for an incomplete one, then one indented line per entry of `lines`."""
    rule = "═" * 72
    print(f"\n{rule}\n{'✅' if complete else '⚠️ '} {title}")
    for line in lines:
        print(f"   {line}")
    print(rule)


def print_chunk_saved(nc, n_spam, n_counts, n_fid):
    """One line for what a chunk's collection (or simulation) saved: its count and fidelity keys,
    and the SPAM input states when SPAM correction is on."""
    spam = f", SPAM counts for {n_spam} input states" if n_spam else ""
    print(f"  [Chunk {nc}] Saved {n_counts} count keys, {n_fid} fidelity keys{spam}.")


def format_retry(label, what, exc, attempt, total, delay, *, noun="retry", reason="connection error"):
    """The notice printed before waiting to try a provider call again — a short headline, then the
    exception on its own indented line (cut to 80 characters; the full text stays in the records):

      ⚠️  [AQT] submitting circuits: connection error — retry 3/7 in 60 s  (14:03:12)
          ConnectError: All connection attempts failed
    """
    import time
    text = " ".join(str(exc).split())
    text = text if len(text) <= 80 else text[:77] + "..."
    return (f"  ⚠️  [{label}] {what}: {reason} — {noun} {attempt}/{total} in {delay} s  ({time.strftime('%H:%M:%S')})\n"
            f"      {type(exc).__name__}{': ' + text if text else ''}")


def require_myqlm(what):
    """Fail with a clear message, instead of an ImportError deep inside, when a
    step needs myQLM in an environment without it (the real-QPU venvs)."""
    try:
        import qat  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"{what} needs myQLM, which this environment doesn't have. Run build() "
            f"in venvs/local_sim.venv first, with the same settings, so that "
            f"everything this step needs is in the shared circuit cache "
            f"(datafiles/circuits/<topology>_Nt<steps>/)."
        ) from e


def sample_index_to_input_bitstring(sample_index, nsamples, nqubits):
    assert nsamples % (2**nqubits) == 0, \
        "nsamples must be a multiple of 2**nqubits for even bitstring coverage"
    state = sample_index % (2**nqubits)
    return format(state, f"0{nqubits}b")

def get_target_bitstring(circuit_key, n_input_states, nqubits, classical_nec_map):
    if n_input_states == 1:
        zero_bitstring = "0" * nqubits
        return classical_nec_map[zero_bitstring]
    tail = circuit_key.rsplit("_", 1)[-1]
    assert tail.isdigit(), f"Circuit key {circuit_key!r} doesn't end in a plain integer sample index"
    sample_index = int(tail)
    input_bitstring = sample_index_to_input_bitstring(sample_index, n_input_states, nqubits)
    return classical_nec_map[input_bitstring]


def get_exp_folder(config, top_folder=False):
    if not top_folder:
        return f"{config['folder_path']}/{config['exp_label']}_circuit_twirls_ns{config['samples']}_ntrotter{config['Nt']}_seed{config['rng_seed']}"
    else: 
        return f"{config['folder_path']}"

def get_exp_file(config, filename, top_folder=False):
    return os.path.join(get_exp_folder(config, top_folder=top_folder), filename)

# Each depth list gets its own circuit cache, named by its Trotter steps (Nt = T/dt), e.g.
# datafiles/circuits/all_to_all_Nt1-3 for the mini benchmark, since chunks hold one circuit
# per (role, depth) and can't be shared between depth lists. The full benchmark's 10 depths
# give datafiles/circuits/<topology>_Nt1-3-5-7-9-11-13-15-17-19.
def get_circuit_folder(config, top_folder=False):
    if "lnn_architecture" not in config or config["lnn_architecture"] is None:
        assert False, "No information about LNN architecture"

    topology = "linear_nearest_neighbor" if config["lnn_architecture"] else "all_to_all"
    if config.get("T_list") is not None:
        steps = [int(round(t / config["dt"])) for t in config["T_list"]]
        topology += "_Nt" + "-".join(map(str, steps))
    noise = config.get("coherent_noise")
    if noise:   # noisy circuits must not share a cache with the noiseless ones
        topology += f"_{noise['noise_type']}_I{noise['infidelity_total']:.1e}_rct{noise['ratio_ct']:.1e}"

    folder = os.path.join(circuits_root(), topology)
    return folder if top_folder else os.path.join(folder, "compiled_from_plugins")

CACHE_CONFIG_FILENAME = "cache_config.json"

# Version of the code that generates the circuits and exact values build() caches. Raise it
# whenever a change alters what build() produces (rc_functions, circuit_functions, chunking/
# interleaving, SPAM circuits, the QPY conversion, the exact observables) — not for refactors
# that leave the output unchanged. A cache built by another version is refused, never reused.
CIRCUIT_CODE_VERSION = "1.0"


def circuit_cache_config(config):
    """The settings that determine a circuit cache's contents. The folder name
    only carries the topology and Trotter-step list (get_circuit_folder()), so
    the rest is recorded in the folder and checked by check_circuit_cache_config()."""
    return {
        "lnn_architecture": bool(config["lnn_architecture"]),
        "trotter_steps":    [int(round(t / config["dt"])) for t in config["T_list"]],
        "dt":               round(float(config["dt"]), 10),
        "samples":          int(config["samples"]),
        "rng_seed":         config["rng_seed"],
        "circuit_code_version": CIRCUIT_CODE_VERSION,
        **({"coherent_noise": config["coherent_noise"]} if config.get("coherent_noise") else {}),
    }


def check_circuit_cache_config(config):
    """Record this run's cache settings in its circuit cache, or refuse to reuse
    a cache built with different ones (a different rng_seed or dt, or more
    samples than it holds, would mix in other circuits under the same folder
    name), or by another CIRCUIT_CODE_VERSION. A folder without the file
    (a new cache) gets it written. A cache is never overwritten: build() only adds
    the files it lacks."""
    circ_top = Path(get_circuit_folder(config, top_folder=True))
    path     = circ_top / CACHE_CONFIG_FILENAME
    wanted   = circuit_cache_config(config)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        # A smaller run reads the cache's first chunks, so only more samples than cached is a mismatch.
        diff = {k: (cached.get(k), v) for k, v in wanted.items()
                if (cached.get(k) < v if k == "samples" else cached.get(k) != v)}
        if diff:
            details = ", ".join(f"{k}: cache {c!r} vs run {r!r}" for k, (c, r) in diff.items())
            if "circuit_code_version" in diff:
                details += (f" — the cache was built by circuit code version {cached.get('circuit_code_version') or 'unknown (older than versioning)'}, "
                            f"this code is version {CIRCUIT_CODE_VERSION}, so its circuits may differ")
            raise ValueError(
                f"The circuit cache {rel_path(circ_top)} was built with different settings ({details}). "
                f"Back up or delete that folder, then rerun to rebuild it "
                f"(or run with the cache's settings).")
        return
    circ_top.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(wanted, f, indent=2)


def get_chunk_folder(circ_top, kind):
    """
    Subfolder of a circuit-cache top folder (get_circuit_folder(..., top_folder=True))
    where interleaved chunk files of a given format live. Created on demand.

    :param circ_top: str/Path — the circuit-cache top folder.
    :param kind:     one of "myqlm", "qiskit" (LNN chunks), "qiskit_v13" (all_to_all chunks, QPY version 13).
    """
    subfolder_by_kind = {
        "myqlm":      "chunks_myqlm",
        "qiskit":     "chunks_qiskit",
        "qiskit_v13": "chunks_qiskit_v13_qpy",
    }
    assert kind in subfolder_by_kind, f"Unknown chunk kind {kind!r} — expected one of {list(subfolder_by_kind)}."
    path = os.path.join(str(circ_top), subfolder_by_kind[kind])
    os.makedirs(path, exist_ok=True)
    return path

def migrate_flat_chunk_files(circ_top):
    """
    One-time migration for circuit-cache folders built before interleaved
    chunk files were split into chunks_myqlm/ and chunks_qiskit/
    subfolders: copies (never moves/deletes) any
    old-style flat chunk files found directly under circ_top into the
    matching new subfolder. Safe to call on every build — skips files
    already present in the destination, and does nothing once a folder
    has no more flat-style leftovers.
    """
    circ_top = Path(circ_top)
    if not circ_top.is_dir():
        return
    glob_pattern_by_kind = {
        "myqlm":  "batch_jobs_interleaved_chunk*.batch",
        "qiskit": "chunk_*_qiskit.qpy",
    }
    for kind, pattern in glob_pattern_by_kind.items():
        flat_files = list(circ_top.glob(pattern))
        if not flat_files:
            continue
        dest_folder = Path(get_chunk_folder(circ_top, kind))
        for src in flat_files:
            dest = dest_folder / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
                print(f"  [migrate] Copied {rel_path(src)} → {rel_path(dest)}")

def get_run_subfolder(config, subfolder, filename=None):
    """
    subfolder: one of 'job_records', 'transpiled_circuits', 'raw_results', 'processed_results'
    filename:  optional — if provided, returns full path to file; otherwise returns folder path
    """
    base = get_exp_folder(config, top_folder=True)
    path = os.path.join(base, subfolder)
    os.makedirs(path, exist_ok=True)
    if filename is not None:
        return os.path.join(path, filename)
    return path


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def read_json_or_none(path):
    """The JSON content of path, or None if the file is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
