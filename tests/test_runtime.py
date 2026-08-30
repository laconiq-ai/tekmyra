"""Integration tests for the deterministic end-to-end pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

import tekmyra.runtime as runtime
from tekmyra.baselines import run_baseline
from tekmyra.contracts import RouteClass
from tekmyra.fixtures import load_fixtures
from tekmyra.runtime import TerminalVerificationError, run_deterministic_pipeline
from tekmyra.verifier import VerifierResult


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "fixtures" / "synthetic"


def test_deterministic_pipeline_runs_on_all_fixtures(run_or_refusal):
    fixtures = load_fixtures(FIXTURE_DIR)
    for fixture in fixtures:
        response = run_or_refusal(lambda fx=fixture: run_deterministic_pipeline(fx))
        if response is None:
            continue  # open core refused, and the refusal itself was asserted
        assert response.compressed_segments
        assert 0.0 < response.global_metrics.compression_ratio <= 1.0
        assert response.global_metrics.verifier_pass is True
        explanation = response.explanations[0]
        assert "rules_fired" in explanation


def test_json_fixture_routes_correctly():
    fixtures = load_fixtures(FIXTURE_DIR)
    fx = next(f for f in fixtures if f.fixture_id == "json-000")
    response = run_deterministic_pipeline(fx)
    explanation = response.explanations[0]
    assert explanation["route_after_override"] == RouteClass.DETERMINISTIC_JSON_COMPRESSOR.value
    assert "R_JSON_VALID" in explanation["rules_fired"]


def test_fixture_tenant_is_propagated_to_runtime_envelope():
    fixtures = load_fixtures(FIXTURE_DIR)
    fixture = fixtures[0].model_copy(update={"tenant_id": "tenant-a"})
    envelope = runtime.fixture_to_envelope(fixture)
    assert envelope.tenant_id == "tenant-a"


def test_second_verifier_failure_is_terminal(monkeypatch):
    fixtures = load_fixtures(FIXTURE_DIR)
    fixture = next(f for f in fixtures if f.fixture_id == "json-000")
    calls = 0

    def _always_fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return VerifierResult(passed=False)

    monkeypatch.setattr(runtime, "verify", _always_fail)
    with pytest.raises(
        TerminalVerificationError,
        match="verification_failed_after_fallback",
    ):
        run_deterministic_pipeline(fixture)
    assert calls == 2


def test_pii_fixture_forces_redaction(run_or_refusal):
    fixtures = load_fixtures(FIXTURE_DIR)
    fx = next(f for f in fixtures if f.fixture_id == "pii-000")
    response = run_or_refusal(lambda: run_deterministic_pipeline(fx))
    if response is None:
        return  # open core refused, and the refusal itself was asserted
    explanation = response.explanations[0]
    assert explanation["route_after_override"] == RouteClass.PII_REDACTION_BEFORE_COMPRESSION.value
    assert "SAFETY_OVERRIDE" in explanation["rules_fired"]
    assert response.global_metrics.fail_safe_triggered is True


def test_deterministic_baseline_is_registered():
    """Registration, which is what the name says.

    This asserted ``compression_ratio == 1.0`` until 2026-08-09. That was never
    a property of the deterministic baseline — it runs the full pipeline, and
    the no-op control is a different baseline entirely (``no_compression`` /
    ``passthrough``, covered separately). The 1.0 held only because
    ``fixtures[0]`` is a code fixture and the code route was returning its
    input unchanged, so the assertion was pinning a defect in place and would
    have failed the moment that route started working. It did.

    Replaced with the invariant actually worth holding here: the baseline is
    registered, it runs, and it returns a well-formed ratio.
    """
    fixtures = load_fixtures(FIXTURE_DIR)
    fx = fixtures[0]
    response = run_baseline("deterministic", fx)
    ratio = response.global_metrics.compression_ratio
    assert 0.0 < ratio <= 1.0, f"not a well-formed compression ratio: {ratio}"
    assert response.explanations


def test_route_precision_on_obvious_structural_classes():
    """Verify deterministic precision >= 0.98 on obvious structural classes.

    A structural fixture is counted correct when the router either emits the
    expected codec route or falls back to a safer route (passthrough or
    reversible retrieval) after recognizing the structural class.
    """
    fixtures = load_fixtures(FIXTURE_DIR)
    structural_types = {"json", "code", "diff", "log", "stack"}
    structural = [f for f in fixtures if f.content_type in structural_types]
    correct = 0
    for fx in structural:
        response = run_deterministic_pipeline(fx)
        explanation = response.explanations[0]
        predicted = explanation["route_after_override"]
        expected = fx.gold_output.expected_route.value
        rules_fired = explanation["rules_fired"]
        structural_rule_fired = any(
            rule in rules_fired
            for rule in (
                "R_JSON_VALID", "R_DIFF_SIGNATURE", "R_STACKTRACE", "R_LOG_DENSE",
                "R_CODE_AST_VALID", "R_SHORT_SAFE",
            )
        )
        safety_override = (
            "SAFETY_OVERRIDE" in rules_fired
            or "SAFETY_OVERRIDE_LEARNED_LOSSY" in rules_fired
        )
        # Correct if exact match, recognized+safe fallback, or safety override.
        safe_fallback = predicted in {"passthrough", "reversible_retrieval_mode"}
        if predicted == expected or (structural_rule_fired and safe_fallback) or safety_override:
            correct += 1
    precision = correct / len(structural)
    assert precision >= 0.98, f"Structural precision {precision:.2f} below 0.98"


def test_no_high_risk_lossy_routes(run_or_refusal):
    """High-risk fixtures must not be routed to lossy compressors."""
    fixtures = load_fixtures(FIXTURE_DIR)
    lossy_routes = {
        RouteClass.PROSE_TOKEN_COMPRESSOR.value,
        RouteClass.SUMMARIZER.value,
        RouteClass.EXTRACTIVE_KEYPOINT_COMPRESSOR.value,
    }
    high_risk = [f for f in fixtures if f.risk_tier.value in ("high", "critical")]
    for fx in high_risk:
        response = run_or_refusal(lambda f=fx: run_deterministic_pipeline(f))
        if response is None:
            continue  # open core refused, and the refusal itself was asserted
        explanation = response.explanations[0]
        assert explanation["route_after_override"] not in lossy_routes
