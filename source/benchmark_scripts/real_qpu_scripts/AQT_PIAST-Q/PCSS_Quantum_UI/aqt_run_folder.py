"""
aqt_run_folder.py
The run folder (datafiles/<name>) that qbcs_real_submit_aqt_loop.py uses for given
benchmark arguments, without starting that script (which connects to PIAST-Q at once).

  - As a script: prints the name. run_aqt_loop.sbatch and run_aqt_loop_with_retries.sbatch (this folder) use it to get the
    run folder name (they build the output.log path from it).
        python3 aqt_run_folder.py [benchmark arguments]
  - As a module: compute_run_folder() (importable).
"""

import os
import sys
from pathlib import Path

LOOP_SCRIPT = Path(__file__).resolve().parent.parent / "qbcs_real_submit_aqt_loop.py"   # in AQT_PIAST-Q/


def compute_run_folder():
    """
    Read RUN_FOLDER out of qbcs_real_submit_aqt_loop.py's own Config block
    without executing the rest of that script. This keeps the two in sync
    automatically if NSHOTS/LNN/DO_SPAM/BACKEND_NAME change — nothing is
    duplicated here. The Config block parses sys.argv[1:], the same arguments
    the loop script is launched with (--force-new-run included: it makes a new
    folder, so the sbatch files call this once per job).
    """
    text = LOOP_SCRIPT.read_text(encoding="utf-8")
    try:
        start = text.index("# ── Config")
        end   = text.index("# ── Connect to PIAST-Q")
    except ValueError:
        raise RuntimeError(
            "Could not find the '# ── Config' / '# ── Connect to PIAST-Q' section "
            "markers in qbcs_real_submit_aqt_loop.py — has it been restructured? "
            "Update compute_run_folder() in aqt_run_folder.py accordingly."
        )
    usage_src = "\n".join(ln for ln in text.split("\n")   # the loop script's usage, so that an argument error prints it
                          if ln.startswith(("from source.core.runner_support.benchmark_args import make_usage", "usage_if_requested(")))
    config_src = ("import sys\nimport numpy as np\nfrom pathlib import Path\n"
                  "from source.core.runner_support.benchmark_args import parse_benchmark_args, parse_backend_config, parse_layout_args, parse_run_folder_arg, pop_force_new_run_flag\n"
                  "from source.core.records.run_names import qpu_run_name, service_name, dated_run_folder\n"
                  + usage_src + "\n" + text[start:end])
    namespace = {"__file__": str(LOOP_SCRIPT)}   # the Config block finds backends.json next to the loop script
    exec(compile(config_src, str(LOOP_SCRIPT), "exec"), namespace)
    return namespace["RUN_FOLDER"]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))   # the project root, for the source.* imports
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))   # datafiles/ paths are relative to the project root
    print(compute_run_folder())
