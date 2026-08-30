"""Tests for the authoritative safety gate."""

from __future__ import annotations


from tekmyra.contracts import PolicyTier, RiskClass, RouteClass, SourceChannel
from tekmyra.prober import probe_text
from tekmyra.safety_gate import SafetyGate


def gate() -> SafetyGate:
    return SafetyGate()


def test_low_risk_benign():
    text = "The quick brown fox jumps over the lazy dog."
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    decision = gate().classify(features)
    assert decision.risk_class == RiskClass.LOW
    assert decision.forced_route is None


def test_pii_forces_redaction():
    text = "Customer Jane Doe, SSN: 123-45-6789, phone: 555-019-1234"
    features = probe_text(text, SourceChannel.CHAT, PolicyTier.STANDARD)
    decision = gate().classify(features)
    assert decision.risk_class == RiskClass.HIGH
    assert decision.forced_route == RouteClass.PII_REDACTION_BEFORE_COMPRESSION


def test_legal_forces_safe_mode():
    text = "This contract is binding and may result in litigation."
    features = probe_text(text, SourceChannel.EMAIL, PolicyTier.STANDARD)
    decision = gate().classify(features)
    assert decision.risk_class == RiskClass.HIGH
    assert decision.forced_route == RouteClass.LEGAL_SECURITY_SAFE_MODE


def test_governance_forces_safe_compression():
    text = "Control ID: SOC2-CC7.2 | Status: Open | Finding: access review missing."
    features = probe_text(text, SourceChannel.TICKET, PolicyTier.STANDARD)
    decision = gate().classify(features)
    assert decision.risk_class == RiskClass.HIGH
    assert decision.forced_route == RouteClass.GOVERNANCE_SAFE_COMPRESSION


def test_critical_policy_tier_rejects():
    text = "Some harmless text."
    features = probe_text(text, SourceChannel.API, PolicyTier.CRITICAL)
    decision = gate().classify(features)
    assert decision.risk_class == RiskClass.CRITICAL
    assert decision.forced_route == RouteClass.REJECT_OR_HUMAN_REVIEW


def test_empty_payload_rejects():
    features = probe_text("", SourceChannel.API, PolicyTier.STANDARD)
    decision = gate().classify(features)
    assert decision.risk_class == RiskClass.CRITICAL
    assert decision.forced_route == RouteClass.REJECT_OR_HUMAN_REVIEW


def test_financial_deadline_is_medium():
    text = "Invoice total: $1,234.56 due by 2026-07-20."
    features = probe_text(text, SourceChannel.EMAIL, PolicyTier.STANDARD)
    decision = gate().classify(features)
    assert decision.risk_class == RiskClass.MEDIUM
    assert decision.forced_route is None


def test_public_policy_reference_does_not_force_high_risk_route():
    text = (
        "According to policy manual, the procedure requires two-factor "
        "approval and quarterly review."
    )
    features = probe_text(
        text,
        SourceChannel.API,
        PolicyTier.STANDARD,
        content_type="rag",
    )
    decision = gate().classify(features)
    assert features.governance_keyword_count > 0
    assert decision.risk_class == RiskClass.LOW
    assert decision.forced_route is None


def test_incident_response_topic_is_not_itself_a_secret():
    text = "Document excerpt: incident response plan. Citation: [2]"
    features = probe_text(
        text,
        SourceChannel.API,
        PolicyTier.STANDARD,
        content_type="rag",
    )
    decision = gate().classify(features)
    assert features.security_keyword_count > 0
    assert decision.risk_class == RiskClass.LOW
    assert decision.forced_route is None


def test_security_plus_pii_is_critical():
    text = "Breach report: attacker exfiltrated SSN 123-45-6789 and 987-65-4321."
    features = probe_text(text, SourceChannel.TICKET, PolicyTier.STANDARD)
    decision = gate().classify(features)
    assert decision.risk_class == RiskClass.CRITICAL
    assert decision.forced_route == RouteClass.REJECT_OR_HUMAN_REVIEW


def test_many_pii_is_critical():
    text = "SSNs: 123-45-6789, 987-65-4321, 111-22-3333, 444-55-6666, 777-88-9999."
    features = probe_text(text, SourceChannel.TICKET, PolicyTier.STANDARD)
    decision = gate().classify(features)
    assert decision.risk_class == RiskClass.CRITICAL
    assert decision.forced_route == RouteClass.REJECT_OR_HUMAN_REVIEW


def test_negated_security_and_legal_terms_do_not_escalate() -> None:
    text = (
        "Routine contract handoff. There is no secret, breach, regulated "
        "finding, emergency, or active compromise."
    )
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)

    decision = gate().classify(features)

    assert decision.risk_class == RiskClass.LOW
    assert decision.forced_route is None


def test_reported_security_claim_refuted_by_evidence_does_not_escalate() -> None:
    text = (
        "A draft alert claimed a credential had been exposed. Validation disproved "
        "that sentence: no access changed and no customer record moved."
    )
    features = probe_text(text, SourceChannel.TICKET, PolicyTier.STANDARD)

    decision = gate().classify(features)

    assert decision.risk_class == RiskClass.LOW
    assert decision.forced_route is None


def test_contained_control_finding_is_high_not_critical() -> None:
    text = (
        "CONTAINED CONTROL FINDING CTRL-BE-100. The control weakness is verified "
        "and consequential. The tenant is isolated and there is no active "
        "compromise. A designated officer must decide before correction."
    )
    features = probe_text(
        text, SourceChannel.TICKET, PolicyTier.STANDARD, content_type="governance"
    )

    decision = gate().classify(features)

    assert decision.risk_class == RiskClass.HIGH
    assert decision.forced_route == RouteClass.REJECT_OR_HUMAN_REVIEW


def test_confirmed_active_incident_is_critical() -> None:
    text = (
        "ACTIVE CONTROL FAILURE CTRL-BE-100. Evidence confirms an active "
        "compromise, confirmed data exfiltration, and continuing customer impact."
    )
    features = probe_text(
        text, SourceChannel.TICKET, PolicyTier.STANDARD, content_type="governance"
    )

    decision = gate().classify(features)

    assert decision.risk_class == RiskClass.CRITICAL
    assert decision.forced_route == RouteClass.REJECT_OR_HUMAN_REVIEW


def test_confirmed_exfiltration_paraphrase_is_critical() -> None:
    text = (
        "Packet captures confirm data exfiltration and continuing customer impact. "
        "Affected systems must be isolated and emergency authorization is required."
    )
    features = probe_text(
        text, SourceChannel.TICKET, PolicyTier.STANDARD, content_type="governance"
    )

    decision = gate().classify(features)

    assert decision.risk_class == RiskClass.CRITICAL
    assert decision.forced_route == RouteClass.REJECT_OR_HUMAN_REVIEW


def test_stateful_and_unknown_formats_are_medium_without_keyword_alarm() -> None:
    for content_type in ("agent_trace", "mixed", "unknown"):
        text = "Stable operational record with exact field ordering."
        if content_type == "unknown":
            text = f"UNCLASSIFIED-FORMAT :: {text}"
        features = probe_text(
            text,
            SourceChannel.AGENT,
            PolicyTier.STANDARD,
            content_type=content_type,
        )
        decision = gate().classify(features)
        assert decision.risk_class == RiskClass.MEDIUM
        assert decision.forced_route is None
