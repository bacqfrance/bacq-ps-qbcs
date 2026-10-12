"""
chunk_scope.py
Which chunks an analysis covers, and what a run is still missing.

Results are collected as their jobs finish, not in chunk order, so a run can have gaps:
chunks 0-40 and 45 collected, 41-44 still running. An analysis of n chunks covers the
first n *collected* chunks (numerically) — for a run without gaps that is chunks
[0, n). chunk_scope() resolves n against the collected chunks into a set of chunk numbers.
Everything that scopes an analysis (timing, workload, circuit checks, SPAM matrix, run
window) takes such a scope: None (every chunk), an int n (as_chunk_set() reads it as
chunks [0, n); chunk_scope() as the first n collected chunks) or an explicit collection
of chunk numbers.

Light module: no qat / qiskit imports.
"""

import json
from pathlib import Path

from source.core.records.health_check_records import load_health_checks, format_chunk_range


def collected_chunks(run_path):
    """Sorted chunk numbers in raw_results/results_counts.json ([] if there is none)."""
    path = Path(run_path) / "raw_results" / "results_counts.json"
    try:
        keys = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return sorted(int(k.split("_", 1)[1]) for k in keys if k.startswith("chunk_"))


def chunk_scope(run_path, n_samples):
    """The chunk numbers an analysis of n_samples chunks covers: the first n_samples
    collected ones. None for n_samples None (every chunk). A run without collected
    results (e.g. test fixtures with records only) scopes to [0, n_samples)."""
    if n_samples is None:
        return None
    if not isinstance(n_samples, int):
        return frozenset(n_samples)
    done = collected_chunks(run_path)
    return frozenset(done[:n_samples]) if done else frozenset(range(n_samples))


def as_chunk_set(scope):
    """None -> None (everything); int n -> {0..n-1}; a collection -> a frozenset."""
    if scope is None:
        return None
    if isinstance(scope, int):
        return frozenset(range(scope))
    return frozenset(scope)


def in_scope(nc, scope):
    s = as_chunk_set(scope)
    return s is None or nc in s


def scope_last_chunk(scope):
    """The highest chunk number in scope (None: unbounded; -1: empty)."""
    s = as_chunk_set(scope)
    return None if s is None else max(s, default=-1)


def uncollected_chunks(run_path, sample_size):
    """Chunks of [0, sample_size) without collected results."""
    done = set(collected_chunks(run_path))
    return [nc for nc in range(sample_size) if nc not in done]


def uncollected_health_checks(run_path):
    """Numbers of the accompanying health checks submitted but not collected."""
    return [c["number"] for c in load_health_checks(run_path) if c.get("status") != "complete"]


def completeness(run_path, sample_size, n_samples=None):
    """What an analysis of n_samples chunks covers and what the run is missing — the
    'completeness' block of run_data*.json and the summary json."""
    done   = collected_chunks(run_path)
    scope  = as_chunk_set(chunk_scope(run_path, n_samples)) if n_samples is not None \
             else frozenset(done or range(sample_size))
    missing, missing_hc = uncollected_chunks(run_path, sample_size), uncollected_health_checks(run_path)
    return {
        "n_chunks_total":            sample_size,
        "n_chunks_collected":        len(done),
        "n_chunks_analyzed":         len(scope),
        "chunks_analyzed":           format_chunk_range(scope),
        "uncollected_chunks":        missing,
        "uncollected_chunks_range":  format_chunk_range(missing),
        "uncollected_health_checks": missing_hc,
        "complete":                  not missing and not missing_hc,
    }


def completeness_warning(block):
    """One line for a figure or the terminal, or None when nothing is missing:
    '⚠ Incomplete run (40 of 44 chunks analyzed): chunks 40-43 are uncollected; health check 3 is uncollected.'"""
    if not block or block.get("complete", True):
        return None
    parts = []
    if chunks := block.get("uncollected_chunks"):
        many = len(chunks) > 1
        parts.append(f"chunk{'s' if many else ''} {block['uncollected_chunks_range'].replace(',', ', ')} "
                     f"{'are' if many else 'is'} uncollected")
    if checks := block.get("uncollected_health_checks"):
        many = len(checks) > 1
        parts.append(f"health check{'s' if many else ''} {', '.join(map(str, checks))} "
                     f"{'are' if many else 'is'} uncollected")
    return f"⚠ Incomplete run ({block['n_chunks_analyzed']} of {block['n_chunks_total']} chunks analyzed): " \
           + "; ".join(parts) + "."


def add_completeness_banner(fig, block, y=-0.06):
    """Red warning line under a matplotlib figure for an incomplete run (below the
    circuit check banner, which sits at y=-0.02)."""
    text = completeness_warning(block)
    if text:
        fig.text(0.5, y, text, ha="center", va="top", color="red", fontsize=9, fontweight="bold")
    return text


def load_run_completeness(run_path, suffix=""):
    """The completeness block written into run_data[_spam]{suffix}.json, or None."""
    pr = Path(run_path) / "processed_results"
    for name in (f"run_data_spam{suffix}.json", f"run_data{suffix}.json"):
        try:
            d = json.loads((pr / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and "completeness" in d:
            return d["completeness"]
    return None
