"""
test_21_docs_consistency.py
===========================
Unit tests (no simulation or network) that the user docs still match the repository:

  - no TAB bytes (a pasted Windows path like tokens\\ibm_token once became tokens<TAB>oken_ibm) and no
    mention of internal_use/ (maintainer-only) in the user docs;
  - every `source/…` path a doc names in backticks exists;
  - every relative Markdown link points to an existing file, and every in-page "#anchor" link of
    every user doc to an existing heading;
  - every "USER_MANUAL §N" / "main USER_MANUAL §N.M" reference names a section USER_MANUAL.md has, and every
    "readme_<qpu>.md §N" reference a section of that QPU readme; no doc or requirements file refers to a
    "README §N" (README.md has no numbered sections);
  - no doc names a token file as tokens/token_<provider> (they are tokens/<provider>_token);
  - every QPU readme has the same seven top-level sections, in order, and the shared paragraphs
    (SHARED_PHRASES, ASYNC_SHARED_PHRASES) in the same words.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
USER_DOCS = sorted([ROOT / "README.md", ROOT / "USER_MANUAL.md", ROOT / "QUICK_START_Windows.md", ROOT / "QUICK_START_Linux.md",
                    ROOT / "GUIDE_FOR_ADDING_NEW_BACKENDS.md", ROOT / "tokens" / "README.md",
                    *(ROOT / "source" / "benchmark_scripts" / "real_qpu_scripts").glob("*/readme_*.md")])
IDS = [d.relative_to(ROOT).as_posix() for d in USER_DOCS]


def _text(doc):
    return doc.read_text(encoding="utf-8")


def _slug(heading):
    """GitHub's anchor of a heading: lower case, punctuation dropped, spaces → '-'."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


@pytest.mark.parametrize("doc", USER_DOCS, ids=IDS)
def test_no_tabs_and_no_internal_use(doc):
    text = _text(doc)
    assert "\t" not in text, f"{doc.name}: TAB byte at line {text[:text.index(chr(9))].count(chr(10)) + 1}"
    assert "internal_use" not in text, f"{doc.name} mentions internal_use/ (maintainer-only)"


@pytest.mark.parametrize("doc", USER_DOCS, ids=IDS)
def test_named_source_paths_exist(doc):
    missing = []
    for m in re.finditer(r"`(source/[^`\s]+)`", _text(doc)):
        path = m.group(1).split(":")[0].rstrip(".,;)")
        if any(c in path for c in "<>*…{}") or "/<" in path:
            continue                                     # a pattern, not a path
        if not (ROOT / path).exists():
            missing.append(path)
    assert not missing, f"{doc.name} names paths that do not exist: {sorted(set(missing))}"


@pytest.mark.parametrize("doc", USER_DOCS, ids=IDS)
def test_links_resolve(doc):
    text = _text(doc)
    anchors = {_slug(h) for h in re.findall(r"^#{1,6} (.+)$", text, re.M)}
    bad = []
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if re.match(r"[a-z]+://|mailto:", target):
            continue
        if target.startswith("#"):
            if target[1:] not in anchors:
                bad.append(target)
        elif not (doc.parent / target.split("#")[0]).exists():
            bad.append(target)
    assert not bad, f"{doc.name} has broken links: {bad}"


REQUIREMENTS = sorted([ROOT / "venvs" / "requirements_local_sim.txt",
                       *(ROOT / "source" / "benchmark_scripts" / "real_qpu_scripts").glob("*/requirements_*.txt")])


def test_user_manual_section_references_exist():
    manual   = _text(ROOT / "USER_MANUAL.md")
    sections = set(re.findall(r"^#{2,4} (\d+(?:\.\d+)*)\.? ", manual, re.M))
    bad = []
    for doc in USER_DOCS + REQUIREMENTS:
        for m in re.finditer(r"USER_MANUAL(?:\.md)?(?:'s)?`? §(\d+(?:\.\d+)*)", _text(doc)):
            if m.group(1) not in sections:
                bad.append(f"{doc.name}: §{m.group(1)}")
    assert not bad, f"references to USER_MANUAL sections that do not exist: {bad}"


def test_no_readme_section_references():
    bad = [f"{doc.name}: {m.group(0)}" for doc in USER_DOCS + REQUIREMENTS
           for m in re.finditer(r"README(?:\.md)?(?:'s)?`? §\d+(?:\.\d+)*", _text(doc))]
    assert not bad, f"README.md has no numbered sections — refer to USER_MANUAL §N instead: {bad}"


@pytest.mark.parametrize("doc", USER_DOCS, ids=IDS)
def test_token_file_names(doc):
    assert "tokens/token_" not in _text(doc), f"{doc.name}: token files are tokens/<provider>_token (tokens/README.md)"


QPU_READMES  = sorted((ROOT / "source" / "benchmark_scripts" / "real_qpu_scripts").glob("*/readme_*.md"))
QPU_SECTIONS = ["Prerequisites", "Connecting", "Config knobs", "Workflow", "Timing, billing and circuit check",
                "Outputs & post-processing", "Known issues"]


@pytest.mark.parametrize("doc", QPU_READMES, ids=[d.name for d in QPU_READMES])
def test_qpu_readmes_share_the_section_layout(doc):
    numbered = re.findall(r"^## (\d+)\. (.+)$", _text(doc), re.M)
    assert [int(n) for n, _ in numbered] == list(range(1, len(QPU_SECTIONS) + 1)), f"{doc.name}: {numbered}"
    for (_, title), expected in zip(numbered, QPU_SECTIONS):
        assert title.startswith(expected), f"{doc.name}: section '{title}', expected '{expected}…'"


# Shared behaviour is described in the same words in every QPU readme (compared with whitespace collapsed).
SHARED_PHRASES = [
    "Called without arguments, a script prints its usage and does nothing.",
    "**Build it in one pass**",
    "Simulating locally with this QPU's calibration",
    "--include-simulations-real",
    "like the other QPUs' health-check scripts",
    "**Circuit drawings.**",
    "`submit.log` (appended, never overwritten) holds one entry per submission, the same in every QPU folder "
    "(`append_submit_log()`): the time, backend, run folder, chunks and shots, the layout, the run qubits with the "
    "calibration summary of `calibration_snapshot.json`, and the run's health-check numbers",
    "**Post-processing.** `python source/analysis_scripts/run_post_process.py mini --include-real-data` (or `full`) "
    "merges, analyzes, summarizes and plots",
    "`run_post_process.py` runs the other scripts in `source/analysis_scripts/`, passing each run to them",
]
ASYNC_SHARED_PHRASES = [   # every QPU but AQT, which submits synchronously and has no collect script
    "An aborted run stays on record — `run_info.json` is written before the PRE check",
    "**A network blip during the PRE/POST health check is retried**",
    "Health checks are collected by the collect script; the submit script skips checks and submission if all chunks "
    "are already submitted.",
]


@pytest.mark.parametrize("doc", QPU_READMES, ids=[d.name for d in QPU_READMES])
def test_qpu_readmes_share_the_shared_paragraphs(doc):
    text    = " ".join(_text(doc).split())
    phrases = SHARED_PHRASES + (ASYNC_SHARED_PHRASES if doc.name != "readme_aqt.md" else [])
    missing = [p for p in phrases if p not in text]
    assert not missing, f"{doc.name} lacks the shared wording: {missing}"


def test_qpu_readme_section_references_exist():
    sections = {d.name: set(re.findall(r"^#{2,4} (\d+(?:\.\d+)*)\.? ", _text(d), re.M)) for d in QPU_READMES}
    sources  = USER_DOCS + [*(ROOT / "source" / "core").rglob("*.py")]
    bad = []
    for doc in sources:
        for name, num in re.findall(r"(readme_[A-Za-z_]+\.md)`? §(\d+(?:\.\d+)*)", _text(doc)):
            if name in sections and num not in sections[name]:
                bad.append(f"{doc.name}: {name} §{num}")
    assert not bad, f"references to QPU readme sections that do not exist: {bad}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
