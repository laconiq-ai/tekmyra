"""Tests for WP-7 Chunk D downstream quality artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results" / "wp7_fast_track"
QUALITY_PATH = RESULTS_DIR / "downstream_quality.json"
LATENCY_PATH = RESULTS_DIR / "latency_fallback.json"

FORBIDDEN_FIELDS = {"original", "compressed", "content", "text"}
QUALITY_TOLERANCE = 0.15
REQUIRED_QUALITY_SUMMARY_KEYS = {
    "raw_correct_rate",
    "compressed_correct_rate",
    "avg_semantic_similarity",
    "avg_critical_detail_retention",
    "compression_ratio",
    "quality_tolerance",
    "quality_degradation",
    "quality_degradation_documented",
    "degradation_note",
}


@pytest.fixture(scope="module")
def quality_report() -> dict:
    if not QUALITY_PATH.exists():
        pytest.skip(
            f"{QUALITY_PATH.name} absent — it is produced by "
            "benchmarks/wp7_fast_track_downstream_eval.py, which is not "
            "published, and tekmyra-reference-artifacts.tar.gz does not carry "
            "it. Expected on every clone, bundle or no bundle; nothing in the "
            "README rests on these gates."
        )
    with open(QUALITY_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def latency_report() -> dict:
    if not LATENCY_PATH.exists():
        pytest.skip(
            f"{LATENCY_PATH.name} absent — it is produced by "
            "benchmarks/wp7_fast_track_downstream_eval.py, which is not "
            "published, and tekmyra-reference-artifacts.tar.gz does not carry "
            "it. Expected on every clone, bundle or no bundle; nothing in the "
            "README rests on these gates."
        )
    with open(LATENCY_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _forbidden_keys_present(obj: object, path: str = "") -> list[str]:
    """Return dotted paths of any forbidden keys found in a JSON-serializable tree."""
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            current = f"{path}.{key}" if path else str(key)
            if key in FORBIDDEN_FIELDS:
                found.append(current)
            found.extend(_forbidden_keys_present(value, current))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(_forbidden_keys_present(value, f"{path}[{index}]"))
    return found


def test_quality_artifact_exists_and_has_required_keys(quality_report: dict) -> None:
    assert "metadata" in quality_report
    assert "summary" in quality_report
    assert "per_fixture" in quality_report
    summary = quality_report["summary"]
    assert REQUIRED_QUALITY_SUMMARY_KEYS.issubset(summary.keys())


def test_quality_metadata_documents_evaluation_method(quality_report: dict) -> None:
    meta = quality_report["metadata"]
    assert meta.get("evaluation_method") == "deterministic_stub_oracle"
    assert "cost_verification_note" in meta
    assert meta.get("cost_verified") is False


def test_quality_no_raw_content_fields(quality_report: dict) -> None:
    forbidden = _forbidden_keys_present(quality_report)
    assert not forbidden, f"Forbidden raw-content keys found: {forbidden}"


def test_compressed_quality_within_tolerance_or_documented(quality_report: dict) -> None:
    summary = quality_report["summary"]
    raw_rate = float(summary["raw_correct_rate"])
    compressed_rate = float(summary["compressed_correct_rate"])
    tolerance = float(summary["quality_tolerance"])
    degradation = raw_rate - compressed_rate

    if degradation <= tolerance:
        return

    assert summary.get("quality_degradation_documented") is True, (
        "Compressed quality materially lower than raw without documentation"
    )
    assert summary.get("degradation_note")
    assert isinstance(summary["degradation_note"], str) and len(summary["degradation_note"]) > 0


def test_latency_fallback_artifact_has_required_keys(latency_report: dict) -> None:
    assert "metadata" in latency_report
    assert "latency_fallback" in latency_report
    lf = latency_report["latency_fallback"]
    assert "overall" in lf
    assert "per_route_class" in lf
    overall = lf["overall"]
    required_overall = {
        "fixture_count",
        "fallback_event_count",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_avg_ms",
    }
    assert required_overall.issubset(overall.keys())


def test_latency_percentiles_and_fallback_are_non_negative(latency_report: dict) -> None:
    lf = latency_report["latency_fallback"]
    for scope, metrics in [("overall", lf["overall"])] + list(lf["per_route_class"].items()):
        assert metrics["fallback_event_count"] >= 0, (
            f"Negative fallback count in {scope}"
        )
        assert metrics["latency_p50_ms"] >= 0, f"Negative p50 in {scope}"
        assert metrics["latency_p95_ms"] >= 0, f"Negative p95 in {scope}"
        assert metrics["latency_avg_ms"] >= 0, f"Negative avg latency in {scope}"


def test_latency_fallback_no_raw_content_fields(latency_report: dict) -> None:
    forbidden = _forbidden_keys_present(latency_report)
    assert not forbidden, f"Forbidden raw-content keys found: {forbidden}"


def test_kimchi_v2_baseline_reused_when_available(quality_report: dict) -> None:
    if "kimchi_v2_baseline" not in quality_report:
        pytest.skip("Kimchi v2 baseline not present; live-model evidence unavailable")
    baseline = quality_report["kimchi_v2_baseline"]
    assert "models" in baseline
    assert "aggregate_raw_correct_rate" in baseline
    assert "aggregate_compressed_correct_rate" in baseline
    for model in baseline["models"]:
        assert "raw_correct_rate" in model
        assert "compressed_correct_rate" in model
