"""
test_19_circuit_cache_and_venvs.py
==================================
Unit tests (no myQLM build, no network) for where circuit caches live and when they
are reused, and for finding a venv's interpreter:

  - circuits_root() is datafiles/circuits for real runs and $BACQ_CIRCUITS_ROOT when set;
    the test session sets it to datafiles/tests/circuits (conftest.py), and cache folders
    are <root>/<topology>_Nt<steps>.
  - No test added anything to datafiles/circuits/ (the tests' caches are in datafiles/tests/circuits/): a cache path
    built without circuits_root() would land there.
  - check_circuit_cache_config() records CIRCUIT_CODE_VERSION in a new cache, reuses a
    cache of the same version, and refuses one of another version or without a version,
    naming both versions.
  - venv_python_path() gives Scripts/python.exe on Windows and bin/python elsewhere.
  - A missing token (read_token), a missing or incomplete LEXIS project file (iqm_vlq.read_project_info)
    end the script the same way (common.exit_with_error): one ❌ line on stderr, exit code 2
    (an unusable layout does too; test_18_layout_args.py covers that).
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import json
from pathlib import Path

import pytest

from source.core import common as C
from source.core.runner_support.venvs import PROJECT_ROOT, venv_python_path
from source.tests.testing_functions import TEST_CIRCUITS_ROOT, rm_glob


def _config(**over):
    cfg = {"lnn_architecture": False, "T_list": [0.2, 0.6], "dt": 0.2, "samples": 4, "rng_seed": 7}
    cfg.update(over)
    return cfg


def test_circuits_root_default_and_override(monkeypatch):
    monkeypatch.delenv(C.CIRCUITS_ROOT_ENV, raising=False)
    assert C.circuits_root() == "datafiles/circuits"
    assert Path(C.get_circuit_folder(_config(), top_folder=True)) == Path("datafiles/circuits/all_to_all_Nt1-3")
    assert Path(C.get_circuit_folder(_config(lnn_architecture=True), top_folder=True)) == \
        Path("datafiles/circuits/linear_nearest_neighbor_Nt1-3")
    monkeypatch.setenv(C.CIRCUITS_ROOT_ENV, "datafiles/tests/circuits")
    assert Path(C.get_circuit_folder(_config(), top_folder=True)) == Path("datafiles/tests/circuits/all_to_all_Nt1-3")


def test_session_uses_test_cache_folder():
    assert os.environ.get(C.CIRCUITS_ROOT_ENV) == TEST_CIRCUITS_ROOT == "datafiles/tests/circuits"


def test_tests_added_nothing_outside_their_cache_folder():
    CIRCUITS_AT_START = set(json.loads(os.environ["BACQ_TEST_CIRCUITS_AT_START"]))   # recorded by conftest.py
    real = Path("datafiles/circuits")
    now  = set(os.listdir(real)) if real.is_dir() else set()
    assert now - CIRCUITS_AT_START - {"tests"} == set(), \
        "a test wrote a circuit cache into datafiles/circuits/ instead of datafiles/tests/circuits/ (a path not built with circuits_root())"


@pytest.fixture
def version_root(monkeypatch):
    root = f"{TEST_CIRCUITS_ROOT}/version_check"
    rm_glob(root)
    monkeypatch.setenv(C.CIRCUITS_ROOT_ENV, root)
    yield Path(root)
    rm_glob(root)


def test_new_cache_records_code_version(version_root):
    C.check_circuit_cache_config(_config())
    saved = json.loads((version_root / "all_to_all_Nt1-3" / C.CACHE_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert saved["circuit_code_version"] == C.CIRCUIT_CODE_VERSION
    C.check_circuit_cache_config(_config())   # same version and settings: reused, no error


@pytest.mark.parametrize("cached_version", ["0.9", None])
def test_other_or_missing_code_version_is_refused(version_root, cached_version):
    C.check_circuit_cache_config(_config())
    path  = version_root / "all_to_all_Nt1-3" / C.CACHE_CONFIG_FILENAME
    saved = json.loads(path.read_text(encoding="utf-8"))
    if cached_version is None:
        del saved["circuit_code_version"]
    else:
        saved["circuit_code_version"] = cached_version
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError) as e:
        C.check_circuit_cache_config(_config())
    msg = str(e.value)
    assert "circuit code version" in msg and C.CIRCUIT_CODE_VERSION in msg
    assert (cached_version or "older than versioning") in msg


def test_venv_python_path_per_os():
    assert venv_python_path("x.venv", "nt") == PROJECT_ROOT / "venvs" / "x.venv" / "Scripts" / "python.exe"
    assert venv_python_path("x.venv", "posix") == PROJECT_ROOT / "venvs" / "x.venv" / "bin" / "python"


def test_exit_with_error(capsys):
    with pytest.raises(SystemExit) as e:
        C.exit_with_error("something the user can fix")
    assert e.value.code == 2
    assert capsys.readouterr().err == "❌ something the user can fix\n"


def test_missing_token_and_lexis_project_file_exit_2(tmp_path, monkeypatch, capsys):
    from source.core.qpu import tokens as T
    from source.core.qpu import iqm_vlq as VLQ
    monkeypatch.setattr(T, "TOKENS_DIR", tmp_path)
    with pytest.raises(SystemExit) as e:
        T.read_token("nonexistent_token")
    assert e.value.code == 2 and "tokens/nonexistent_token does not exist" in capsys.readouterr().err
    (tmp_path / "x_token").write_text("  abc123\n", encoding="utf-8")
    assert T.read_token("x_token") == "abc123"                         # stripped

    with pytest.raises(SystemExit) as e:
        VLQ.read_project_info(tmp_path / "iqm_vlq_project_info")
    assert e.value.code == 2 and "No LEXIS project file" in capsys.readouterr().err
    (tmp_path / "iqm_vlq_project_info").write_text("PROJECT = p1\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        VLQ.read_project_info(tmp_path / "iqm_vlq_project_info")
    assert e.value.code == 2 and "Incomplete LEXIS project file" in capsys.readouterr().err
    (tmp_path / "iqm_vlq_project_info").write_text("PROJECT = p1\nRESOURCE = r 1\n", encoding="utf-8")
    assert VLQ.read_project_info(tmp_path / "iqm_vlq_project_info") == {"PROJECT": "p1", "RESOURCE": "r 1"}


def test_scrub_paths_removes_project_and_home_folders():
    """scrub_paths drops the project folder and writes the home folder as ~, in both slash forms."""
    root, home = C.PROJECT_ROOT, Path.home()
    text = f"boom in {root / 'source' / 'x.py'} and {root.as_posix()}/source/y.py, cfg {home.as_posix()}/.cfg"
    out = C.scrub_paths(text)
    assert str(root) not in out and root.as_posix() not in out and str(home) not in out.replace("~", "")
    assert "source/y.py" in out and "~/.cfg" in out
    assert C.scrub_paths(None) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
