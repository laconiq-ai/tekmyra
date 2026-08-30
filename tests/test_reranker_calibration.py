"""Calibration gate for RerankerChunkBaseline held-out results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_FILE = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "reranker_held_out"
    / "reranker_chunk_results.json"
)


def _load_results() -> dict:
    if not RESULT_FILE.exists():
        pytest.skip(
            f"{RESULT_FILE.name} absent — it is produced by "
            "scripts/run_reranker_held_out.py, which is not published, and "
            "tekmyra-reference-artifacts.tar.gz does not carry it. Expected "
            "on every clone, bundle or no bundle; nothing in the README rests "
            "on these calibration checks."
        )
    with open(RESULT_FILE) as fh:
        return json.load(fh)


def test_summary_in_tolerance_rate_is_one():
    data = _load_results()
    summary = data.get("summary", {})
    assert summary.get("in_tolerance_rate") == 1.0, summary


def test_every_row_passes_quality_gates():
    data = _load_results()
    rows = data.get("results", [])
    assert rows, "expected at least one held-out result"
    for row in rows:
        assert row["protected_disposition_valid"] is True, row
        assert row["introduced_duplicate_count"] == 0, row
        assert row["required_answer_fact_recall"] == 1.0, row
