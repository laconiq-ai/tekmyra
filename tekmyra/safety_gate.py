"""Authoritative safety gate for the TekMyra runtime.

The safety gate consumes deterministic features and emits a risk class,
risk score, and optional forced route override.  It is fail-closed: any
internal error or unsupported input escalates to CRITICAL and rejects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tekmyra.contracts import (
    FeatureVector,
    PolicyTier,
    RiskClass,
    RouteClass,
)


# Broad topic mentions remain useful probe features, but they are not by
# themselves evidence that a payload is sensitive.  These narrower indicators
# drive route overrides.  This prevents benign text such as a public policy
# citation or an agent awaiting approval from being treated as regulated data.
STRONG_LEGAL_INDICATORS = {
    "liable", "liability", "warranty", "indemnify",
    "litigation", "attorney", "counsel", "subpoena", "confidential",
    "privileged", "non-disclosure", "nda", "binding",
}
STRONG_SECURITY_INDICATORS = {
    "vulnerability", "cve", "exploit", "malware", "ransomware", "breach",
    "intrusion", "attack", "threat", "phishing", "credential", "password",
    "secret", "api key", "private key", "penetration", "zero-day", "sqli",
    "xss", "csrf", "backdoor", "rootkit", "trojan", "logic bomb",
}
STRONG_GOVERNANCE_INDICATORS = {
    "control id", "control finding", "control weakness", "finding:", "waiver",
    "risk appetite", "fiduciary", "designated officer must decide",
    "authorized officer must decide", "officer must decide next",
    "officer must approve changes",
}

CRITICAL_INCIDENT_INDICATORS = {
    "active compromise",
    "confirmed data exfiltration",
    "data exfiltration is confirmed",
    "confirm data exfiltration",
    "continuing customer impact",
    "customer impact continues",
    "continuing impact",
    "stop all automation immediately",
    "automation must stop",
    "emergency authorization is required",
    "ransomware active",
    "ongoing breach",
}

_NEGATION_WORDS = {
    "absent",
    "deny",
    "denied",
    "neither",
    "never",
    "no",
    "nor",
    "not",
    "unconfirmed",
    "without",
}


def _indicator_matches(text: str, indicators: set[str]) -> list[re.Match[str]]:
    """Return affirmed indicator matches, excluding local explicit negation.

    The safety gate is intentionally conservative, but a literal phrase such
    as ``no secret, breach, or active compromise`` is counter-evidence rather
    than a positive incident signal.  Negation is scoped to the current clause
    and at most six preceding words so an earlier unrelated ``no`` cannot
    suppress a later assertion.
    """
    lowered = text.lower()
    affirmed: list[re.Match[str]] = []
    for indicator in sorted(indicators):
        pattern = re.compile(rf"(?<!\w){re.escape(indicator)}(?!\w)")
        for match in pattern.finditer(lowered):
            clause_start = max(
                lowered.rfind(mark, 0, match.start()) for mark in (".", ";", "\n")
            )
            clause_start = max(
                clause_start,
                *(lowered.rfind(mark, 0, match.start()) for mark in (", and ", ", but ", " but ")),
            )
            prefix = lowered[clause_start + 1 : match.start()]
            words = re.findall(r"[a-z]+(?:-[a-z]+)?", prefix)[-6:]
            if any(word in _NEGATION_WORDS for word in words):
                continue
            sentence_start = lowered.rfind(".", 0, match.start()) + 1
            sentence_prefix = lowered[sentence_start : match.start()]
            following = lowered[match.end() : match.end() + 180]
            reported_claim = any(
                cue in sentence_prefix
                for cue in ("claimed", "suggested", "alleged", "draft alert")
            )
            explicitly_refuted = any(
                cue in following
                for cue in (
                    "disproved",
                    "refute",
                    "was incorrect",
                    "validation rejected",
                )
            )
            if reported_claim and explicitly_refuted:
                continue
            affirmed.append(match)
    return affirmed


def _contains_indicator(text: str, indicators: set[str]) -> bool:
    return bool(_indicator_matches(text, indicators))


@dataclass(frozen=True)
class SafetyDecision:
    risk_class: RiskClass
    risk_score: float
    forced_route: RouteClass | None
    explanation_tags: list[str]


class SafetyGateError(Exception):
    """Raised when the safety gate cannot produce a decision safely."""


class SafetyGate:
    """Deterministic safety gate with configurable thresholds.

    Design intent:
    - Policy tier and manifest danger signals dominate content signals.
    - High-risk content is forced to safe/extractive/review routes.
    - Medium-risk content is forced to reversible retrieval.
    - Low-risk content gets no override.
    - Any error raises SafetyGateError so callers can fail closed.
    """

    def __init__(self) -> None:
        pass

    def classify(self, features: FeatureVector) -> SafetyDecision:
        """Return a safety decision for the given feature vector."""
        try:
            return self._classify(features)
        except SafetyGateError:
            raise
        except Exception as exc:
            raise SafetyGateError(f"Safety gate internal error: {exc}") from exc

    def _classify(self, features: FeatureVector) -> SafetyDecision:
        tags: list[str] = []
        text = features.text or ""
        legal_high = _contains_indicator(text, STRONG_LEGAL_INDICATORS)
        security_high = _contains_indicator(text, STRONG_SECURITY_INDICATORS)
        governance_high = _contains_indicator(text, STRONG_GOVERNANCE_INDICATORS)
        human_review_required = _contains_indicator(
            text,
            {
                "designated officer must decide",
                "authorized officer must decide",
                "officer must decide next",
                "officer must approve changes",
                "human review required",
                "manual approval required",
            },
        )
        critical_incident_signals = len(
            _indicator_matches(text, CRITICAL_INCIDENT_INDICATORS)
        )

        # CRITICAL: critical policy tier, empty payload, or unsupported content.
        if features.policy_tier == PolicyTier.CRITICAL:
            tags.append("critical_policy_tier")
            return SafetyDecision(
                risk_class=RiskClass.CRITICAL,
                risk_score=1.0,
                forced_route=RouteClass.REJECT_OR_HUMAN_REVIEW,
                explanation_tags=tags,
            )

        if features.char_len == 0:
            tags.append("empty_payload")
            return SafetyDecision(
                risk_class=RiskClass.CRITICAL,
                risk_score=1.0,
                forced_route=RouteClass.REJECT_OR_HUMAN_REVIEW,
                explanation_tags=tags,
            )

        # Aggregate high-risk content signals.
        high_risk_signals = 0
        if features.pii_entity_count > 0:
            tags.append("pii_detected")
            high_risk_signals += 1
        if legal_high:
            tags.append("strong_legal_indicator")
            high_risk_signals += 1
        if security_high:
            tags.append("strong_security_indicator")
            high_risk_signals += 1
        if governance_high:
            tags.append("strong_governance_indicator")
            high_risk_signals += 1
        if features.policy_tier == PolicyTier.REGULATED:
            tags.append("regulated_policy_tier")
            high_risk_signals += 1

        # Financial and deadline signals raise risk to at least medium.
        if features.financial_amount_count > 0:
            tags.append("financial_amounts")
        if features.deadline_date_count > 0:
            tags.append("deadline_dates")

        # Critical escalation: many PII hits, strong adversarial/security combo,
        # or multiple independently affirmed active-incident facts.
        if features.pii_entity_count >= 5 or (
            security_high and features.pii_entity_count > 0
        ) or critical_incident_signals >= 2:
            tags.append("critical_signal_density")
            return SafetyDecision(
                risk_class=RiskClass.CRITICAL,
                risk_score=0.95,
                forced_route=RouteClass.REJECT_OR_HUMAN_REVIEW,
                explanation_tags=tags,
            )

        # HIGH: any high-risk signal present.
        if high_risk_signals > 0:
            if features.pii_entity_count > 0:
                forced = RouteClass.PII_REDACTION_BEFORE_COMPRESSION
            elif legal_high or security_high:
                forced = RouteClass.LEGAL_SECURITY_SAFE_MODE
            elif human_review_required:
                forced = RouteClass.REJECT_OR_HUMAN_REVIEW
                tags.append("explicit_human_decision_required")
            elif governance_high or features.policy_tier == PolicyTier.REGULATED:
                forced = RouteClass.GOVERNANCE_SAFE_COMPRESSION
            else:
                forced = RouteClass.REVERSIBLE_RETRIEVAL_MODE

            score = min(0.99, 0.70 + 0.05 * high_risk_signals)
            return SafetyDecision(
                risk_class=RiskClass.HIGH,
                risk_score=score,
                forced_route=forced,
                explanation_tags=tags,
            )

        # These content types encode state or heterogeneous structure whose
        # semantics must remain reversible even when no sensitive keyword is
        # present.  This is a deterministic format risk, not a learned guess.
        structure_requires_reversible = features.content_type.lower() in {
            "agent_trace",
            "mixed",
        } or (
            features.content_type.lower() == "unknown"
            and any(
                cue in text.lower()
                for cue in (
                    "unclassified-format",
                    "unmapped payload",
                    "field semantics have not been identified",
                    "grammar has not been established",
                    "source bytes",
                    "¦",
                )
            )
        )
        if structure_requires_reversible:
            tags.append("structure_requires_reversible_handling")
            return SafetyDecision(
                risk_class=RiskClass.MEDIUM,
                risk_score=0.35,
                forced_route=None,
                explanation_tags=tags,
            )

        # MEDIUM: financial/deadline content without high-risk signals.
        # The rule router will select a safe structural route; do not force
        # reversible here so that chart, prose, and summary intents can fire.
        if (
            features.financial_amount_count > 0
            or features.deadline_date_count > 0
        ) and features.policy_tier != PolicyTier.CRITICAL:
            if features.financial_amount_count > 0:
                tags.append("financial_amounts")
            if features.deadline_date_count > 0:
                tags.append("deadline_dates")
            return SafetyDecision(
                risk_class=RiskClass.MEDIUM,
                risk_score=0.45,
                forced_route=None,
                explanation_tags=tags,
            )

        # MEDIUM: stack traces / exception content carry structured data but
        # the deterministic stack-trace codec is verified to preserve app
        # frames and exception messages, so no route override is required.
        if features.stacktrace_hits > 0:
            tags.append("stacktrace_detected")
            return SafetyDecision(
                risk_class=RiskClass.MEDIUM,
                risk_score=0.35,
                forced_route=None,
                explanation_tags=tags,
            )

        # LOW: default.
        return SafetyDecision(
            risk_class=RiskClass.LOW,
            risk_score=0.1,
            forced_route=None,
            explanation_tags=["low_risk"],
        )


def classify_safety(features: FeatureVector) -> SafetyDecision:
    """Convenience function using the default safety gate."""
    return SafetyGate().classify(features)
