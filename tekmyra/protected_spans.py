"""Protected-span detector for the TekMyra runtime.

Detects must-preserve spans (PII, operational IDs, business fields,
governance/legal/security markers, RAG citations, agent state markers) and
assigns a locked-span policy: preserve, redact, anchorize, or human_review.

All detection is deterministic regex/heuristic based. No learned models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from tekmyra.contracts import ProtectedSpan, RiskClass
from tekmyra.content_hash import content_hash as _content_hash
from tekmyra.fixtures import Fixture


# --- Patterns ---------------------------------------------------------------

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Phone number variants.  Bare 10-digit tokens (e.g. account numbers) are
# filtered out in post-processing.
_PHONE_PATTERNS = [
    re.compile(r"\b\+?1[-.\s]\(?\d{3}\)?[-.\s]?\d{3}[-.\s]\d{4}\b"),
    re.compile(r"\b\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
]
_BARE_DIGITS_RE = re.compile(r"\+?\d+")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# Financial amounts: $1,234.56, USD 1234.56, etc.
_AMOUNT_RE = re.compile(
    r"\$\s*[\d,]+(?:\.\d{2})?|"
    r"\b\d{1,3}(?:,\d{3})+\.\d{2}\b|"
    r"\b(?:USD|EUR|GBP|JPY|CAD|AUD)\s*[\d,]+(?:\.\d{2})?"
)

# ISO-8601-ish timestamps and simple dates.
_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
)

# File paths with extensions or common prefixes.
_FILE_PATH_RE = re.compile(
    r"\b(?:/[^/\s]+)+\.[a-zA-Z0-9]+\b|"
    r"\b[a-zA-Z]:\\[^\\\s]+\.[a-zA-Z0-9]+\b"
)

# Line-number references: "line 42" or ":42".
_LINE_NUMBER_RE = re.compile(r"\bline\s+\d+\b|:\d+\b")

# Policy / control IDs: POL-123, CTRL-456, etc.
_POLICY_ID_RE = re.compile(r"\b(?:POL|CTRL|SEC|GOV|AUDIT|FINDING)-\d+\b", re.I)

# RAG citations / source IDs: [1], [Smith2024], SOURCE:abc123
_CITATION_RE = re.compile(r"\[\d+\]|\[[A-Za-z]+\d{4}\]|\bSOURCE:\S+", re.I)

# Agent state markers: TODO, BLOCKER, APPROVED, etc.
_AGENT_STATE_RE = re.compile(
    r"\b(?:TODO|BLOCKER|APPROVED|REJECTED|PENDING|FOLLOW-UP|ACTION ITEM):?\b",
    re.I,
)

# URL references.
_URL_RE = re.compile(r"https?://\S+")


@dataclass(frozen=True)
class DetectorConfig:
    """Policy assignment by span type and optional risk override."""

    pii_policy: Literal["preserve", "redact", "anchorize", "human_review"] = "anchorize"
    amount_policy: Literal["preserve", "redact", "anchorize", "human_review"] = "preserve"
    date_policy: Literal["preserve", "redact", "anchorize", "human_review"] = "preserve"
    path_policy: Literal["preserve", "redact", "anchorize", "human_review"] = "preserve"
    line_number_policy: Literal["preserve", "redact", "anchorize", "human_review"] = "preserve"
    policy_id_policy: Literal["preserve", "redact", "anchorize", "human_review"] = "preserve"
    # PRESERVE, not anchorize. PII is protected for SECRECY, so every
    # occurrence must be replaced. Citations are protected for PROVENANCE —
    # the content is not secret and traceability is the point, so anchorizing
    # them hid information the downstream model needs for no secrecy benefit.
    # A citation containing PII still gets PII treatment from the detector.
    citation_policy: Literal["preserve", "redact", "anchorize", "human_review"] = "preserve"
    agent_state_policy: Literal["preserve", "redact", "anchorize", "human_review"] = "preserve"
    url_policy: Literal["preserve", "redact", "anchorize", "human_review"] = "preserve"
    # If True, reject bare 10-digit tokens that have no separators or
    # punctuation (e.g. account numbers).  Recommended for production.
    strict_phone: bool = True


# Map span type to detector name.
_DETECTOR = "deterministic_regex"


def _stable_fact_id(span_type: str, text: str) -> str:
    """Return a stable fact ID derived from the span type and canonical text."""
    return f"fact:{span_type}:{_content_hash(text)}"


def _stable_span_id(span_type: str, start: int, end: int) -> str:
    """Return a stable span ID derived from the span type and document offsets."""
    return f"span:{span_type}:{start}:{end}"


def _make_span(
    span_type: str,
    text: str,
    start: int,
    end: int,
    policy: Literal["preserve", "redact", "anchorize", "human_review"],
) -> ProtectedSpan:
    return ProtectedSpan(
        span_type=span_type,
        start=start,
        end=end,
        text=text,
        policy=policy,
        detector=_DETECTOR,
        confidence=0.9,
        span_id=_stable_span_id(span_type, start, end),
        fact_id=_stable_fact_id(span_type, text),
    )


def detect_protected_spans(text: str, config: DetectorConfig | None = None) -> list[ProtectedSpan]:
    """Return all protected spans found in ``text``.

    Spans are returned in document order and non-overlapping where possible.
    When two spans overlap, the longer span wins.
    """
    config = config or DetectorConfig()

    candidates: list[ProtectedSpan] = []

    def add(pattern: re.Pattern[str], span_type: str, policy: Literal["preserve", "redact", "anchorize", "human_review"]) -> None:
        for match in pattern.finditer(text):
            candidates.append(_make_span(span_type, match.group(), match.start(), match.end(), policy))

    add(_SSN_RE, "ssn", config.pii_policy)
    for phone_pattern in _PHONE_PATTERNS:
        add(phone_pattern, "phone", config.pii_policy)
    if config.strict_phone:
        # Drop phone matches that are nothing but digits (e.g. 9988776655).
        candidates = [
            s for s in candidates
            if s.span_type != "phone" or not _BARE_DIGITS_RE.fullmatch(s.text)
        ]
    add(_EMAIL_RE, "email", config.pii_policy)
    add(_CREDIT_CARD_RE, "credit_card", config.pii_policy)
    add(_IP_RE, "ip_address", config.pii_policy)
    add(_AMOUNT_RE, "amount", config.amount_policy)
    add(_DATE_RE, "date", config.date_policy)
    add(_FILE_PATH_RE, "file_path", config.path_policy)
    add(_LINE_NUMBER_RE, "line_number", config.line_number_policy)
    add(_POLICY_ID_RE, "policy_id", config.policy_id_policy)
    add(_CITATION_RE, "citation", config.citation_policy)
    add(_AGENT_STATE_RE, "agent_state", config.agent_state_policy)
    add(_URL_RE, "url", config.url_policy)

    if not candidates:
        return []

    # Sort by start, then prefer longer spans on overlap.
    candidates.sort(key=lambda s: (s.start, -(s.end - s.start)))
    merged: list[ProtectedSpan] = []
    for span in candidates:
        if not merged or span.start >= merged[-1].end:
            merged.append(span)
        elif span.end > merged[-1].end:
            # Overlapping; extend the last span if the new one is longer overall.
            last = merged[-1]
            if (span.end - span.start) > (last.end - last.start):
                merged[-1] = span
    return merged


def build_fact_index(fixture: Fixture) -> dict[str, str]:
    """Build a fact_id -> canonical text index from a fixture.

    The index is built from both the fixture-level protected spans and the
    gold-output must-preserve spans. When the same fact_id maps to different
    texts, the longest text wins so that canonical lookup preserves the full
    fact.
    """
    index: dict[str, str] = {}
    sources: list[Any] = []
    sources.extend(fixture.protected_spans or [])
    sources.extend(fixture.gold_output.must_preserve_spans or [])
    for span in sources:
        fact_id = getattr(span, "fact_id", None)
        text = getattr(span, "text", None)
        if not fact_id or not text:
            continue
        existing = index.get(fact_id)
        if existing is None or len(text) > len(existing):
            index[fact_id] = text
    return index


def policy_for_risk(span: ProtectedSpan, risk_class: RiskClass) -> str:
    """Return the effective policy for a span given the artifact risk class.

    Higher risk tightens the policy by escalating one level per tier:
    - LOW/MEDIUM: preserve -> preserve, anchorize -> anchorize, redact -> redact.
    - HIGH: preserve -> anchorize, anchorize -> redact, redact -> human_review.
    - CRITICAL: preserve -> redact, anchorize -> human_review, redact -> human_review.
    """
    policy = span.policy
    if risk_class == RiskClass.HIGH:
        if policy == "preserve":
            return "anchorize"
        if policy == "anchorize":
            return "redact"
        if policy == "redact":
            return "human_review"
    if risk_class == RiskClass.CRITICAL:
        if policy == "preserve":
            return "redact"
        if policy in ("anchorize", "redact"):
            return "human_review"
    return policy
