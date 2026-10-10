"""
test_22_output_format.py
========================
Unit tests (no simulation, no network) for the shared output helpers in source/core/common.py and
source/core/health_check.py, so that the scripts print their headers, closing blocks and retry
notices the same way:

  - section(): a blank line, then a title filled with rules to RULE_WIDTH;
  - print_done_block(): aligned `Label:  value` rows under a ✅ line, rows without a value left out;
  - print_chunk_saved(): one line, with the SPAM input states only when SPAM correction is on;
  - format_retry(): a short headline (label, what, attempt, wait) and the exception on its own line, cut to 80 characters;
  - top_counts(): the most frequent outcomes of a health check, with the number left out.
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # the project root, for the source.* imports
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # datafiles/ paths are relative to the project root

from source.core.common import RULE_WIDTH, format_retry, print_chunk_saved, print_done_block, section
from source.core.health_check import top_counts


def test_section_is_rule_wide(capsys):
    section("Summary")
    lines = capsys.readouterr().out.split("\n")
    assert lines[0] == "" and lines[1].startswith("── Summary ─") and len(lines[1]) == RULE_WIDTH


def test_done_block_aligns_rows_and_skips_none(capsys):
    print_done_block("3 chunk(s) simulated: 0-2", [("Run folder", "datafiles/x"), ("Log saved", None), ("Job records", "datafiles/x/job_records/")])
    assert capsys.readouterr().out.split("\n")[1:4] == ["✅ 3 chunk(s) simulated: 0-2",
                                                        "   Run folder:  datafiles/x",
                                                        "   Job records: datafiles/x/job_records/"]


def test_chunk_saved_line(capsys):
    print_chunk_saved(4, 8, 30, 30)
    print_chunk_saved(5, 0, 30, 30)
    out = capsys.readouterr().out.splitlines()
    assert out == ["  [Chunk 4] Saved 30 count keys, 30 fidelity keys, SPAM counts for 8 input states.",
                   "  [Chunk 5] Saved 30 count keys, 30 fidelity keys."]


def test_retry_notice_is_two_short_lines():
    text = format_retry("AQT", "submitting circuits", ConnectionError("x" * 300), 3, 7, 60)
    head, detail = text.split("\n")
    assert head.startswith("  ⚠️  [AQT] submitting circuits: connection error — retry 3/7 in 60 s  (")
    assert detail.startswith("      ConnectionError: x") and detail.endswith("...") and len(detail) < 120


def test_top_counts_cuts_and_counts_the_rest():
    counts = {"011": 90, "010": 5, "001": 3, "000": 1, "111": 1, "110": 0}
    assert top_counts(counts) == "{'011': 90, '010': 5, '001': 3, '000': 1}, +2 more"
    assert top_counts({"011": 100}) == "{'011': 100}"
