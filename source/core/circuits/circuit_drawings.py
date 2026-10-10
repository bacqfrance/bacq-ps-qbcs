"""
circuit_drawings.py
Circuit drawings of a run, so whoever runs a benchmark sees what is submitted: the first circuit of
every error-mitigation role (and of SPAM, and the health check), before and after transpilation,
saved as PDFs in datafiles/<run>/circuit_drawings/. Nothing is drawn in the terminal.

Called from source/core/qbcs.py and source/core/health_check.py wherever circuits are transpiled or run. Existing drawings are kept, so a
run draws each circuit once. A drawing that fails only prints a warning: it never stops a run.
Qiskit + matplotlib only (no myQLM), so it works in every venv.
"""

from pathlib import Path
from source.core.common import rel_path

FOLDER = "circuit_drawings"

# a chunk's roles (bacq_ps_qbcs.roles) → the error-mitigation scheme each one is run for
ROLE_LABELS = {
    "raw":     "bare",
    "raw_nec": "NEC",
    "rc":      "RC",
    "rc_nec":  "RC+NEC",
    "crc":     "cRC",
    "crc_nec": "cRC+NEC",
}


def chunk_items(roles, raw_circuits, isa_circuits=None):
    """(name, before, after) for the first circuit of every role: a chunk starts with one circuit per
    role at the first depth (local index = depth index * len(roles) + role index)."""
    return [(f"{ri + 1:02d}_{ROLE_LABELS.get(role, role)}", raw_circuits[ri],
             isa_circuits[ri] if isa_circuits is not None else None)
            for ri, role in enumerate(roles)]


def export_circuit_drawings(run_path, items):
    """Save <name>_before_transpilation.pdf and, when an ISA circuit is given,
    <name>_after_transpilation.pdf for every (name, before, after) in items; files that already
    exist are kept. Without an ISA circuit (noiseless local simulation, which runs the circuits
    as they are) only the "before" drawing is saved."""
    try:
        from matplotlib.figure import Figure   # no pyplot: no GUI backend, safe outside the main thread
        from qiskit.visualization import circuit_drawer
        folder = Path(run_path) / FOLDER
        saved  = 0
        for name, before, after in items:
            for stage, circuit in (("before", before), ("after", after)):
                pdf = folder / f"{name}_{stage}_transpilation.pdf"
                if circuit is None or pdf.exists():
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                fig = Figure()
                circuit_drawer(circuit, output="mpl", fold=40, idle_wires=False, ax=fig.add_subplot())
                fig.savefig(pdf, bbox_inches="tight")
                saved += 1
        if saved:
            print(f"  [Circuits] Saved {saved} circuit drawing(s) → {rel_path(folder)}")
    except Exception as e:   # a drawing must never stop a (paid) run
        print(f"  ⚠️  [Circuits] Could not save the circuit drawings: {type(e).__name__}: {e}")
