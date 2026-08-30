"""Tests for the deterministic protected-span detector."""

from __future__ import annotations

from tekmyra.contracts import RiskClass, RouteClass
from tekmyra.fixtures import Fixture, FixtureGoldOutput, FixtureProvenance, ProtectedSpanGold
from tekmyra.protected_spans import build_fact_index, detect_protected_spans, policy_for_risk


def test_detects_pii():
    text = "Contact alice@example.com or 555-123-4567. SSN: 123-45-6789."
    spans = detect_protected_spans(text)
    types = {s.span_type for s in spans}
    assert "email" in types
    assert "phone" in types
    assert "ssn" in types


def test_detects_financial_and_dates():
    text = "Invoice total $12,345.67 due 2024-12-31. Server IP 192.168.1.1"
    spans = detect_protected_spans(text)
    types = {s.span_type for s in spans}
    assert "amount" in types
    assert "date" in types
    assert "ip_address" in types


def test_detects_policy_and_citations():
    text = "Per POL-123 and [Smith2024], the control CTRL-99 applies."
    spans = detect_protected_spans(text)
    types = {s.span_type for s in spans}
    assert "policy_id" in types
    assert "citation" in types


def test_overlapping_spans_prefer_longer():
    text = "Reach me at 555-123-4567 anytime."
    spans = detect_protected_spans(text)
    # Phone pattern should match once; SSN pattern (123-45-6789) should not match.
    assert len(spans) == 1
    assert spans[0].span_type == "phone"


def test_policy_for_risk_escalates():
    span = detect_protected_spans("ssn 123-45-6789")[0]
    assert policy_for_risk(span, RiskClass.LOW) == "anchorize"
    assert policy_for_risk(span, RiskClass.HIGH) == "redact"
    redact_span = detect_protected_spans("card 4111-1111-1111-1111")[0]
    redact_span.policy = "redact"  # type: ignore[assignment]
    assert policy_for_risk(redact_span, RiskClass.CRITICAL) == "human_review"


def test_strict_phone_rejects_bare_10_digit_token():
    text = "Account 9988776655 and phone 555-123-4567."
    spans = detect_protected_spans(text)
    types = {s.span_type for s in spans}
    assert "phone" in types
    assert not any(s.span_type == "phone" and s.text == "9988776655" for s in spans)


def test_phone_variants_detected():
    variants = [
        "555-123-4567",
        "(555) 123-4567",
        "555.123.4567",
        "+1 555-123-4567",
    ]
    for variant in variants:
        spans = detect_protected_spans(f"call {variant}")
        assert any(s.span_type == "phone" for s in spans), f"missed {variant}"


def test_detected_spans_emit_stable_ids():
    text = "SSN 123-45-6789 and amount $1,000."
    spans = detect_protected_spans(text)
    for span in spans:
        assert span.span_id, f"{span.span_type} missing span_id"
        assert span.span_id.startswith("span:")
        assert span.fact_id, f"{span.span_type} missing fact_id"
        assert span.fact_id.startswith("fact:")


def test_build_fact_index_uses_longest_text_for_fact_id():
    fixture = Fixture(
        schema_version="2.1.0",
        fixture_id="fact-index-test",
        content_type="prose",
        risk_tier="low",
        content="Account ABC123 details.",
        route_labels=[RouteClass.PASSTHROUGH],
        protected_spans=[
            ProtectedSpanGold(
                span_type="account_id",
                start=8,
                end=14,
                text="ABC123",
                policy="preserve",
                fact_id="fact:account",
            ),
        ],
        gold_output=FixtureGoldOutput(
            route_intent=RouteClass.PASSTHROUGH,
            allowed_outcomes=[RouteClass.PASSTHROUGH],
            expected_route=RouteClass.PASSTHROUGH,
            expected_risk="low",
            must_preserve_spans=[
                ProtectedSpanGold(
                    span_type="account_id",
                    start=8,
                    end=14,
                    text="ABC123-X",
                    policy="preserve",
                    fact_id="fact:account",
                ),
            ],
        ),
        provenance=FixtureProvenance(
            source="unit_test",
            license="proprietary_laconiq_mock",
            generated_by="test_protected_spans",
        ),
    )
    index = build_fact_index(fixture)
    assert index.get("fact:account") == "ABC123-X"
