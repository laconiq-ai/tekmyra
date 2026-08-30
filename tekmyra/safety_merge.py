"""Safety merge: apply locked-span policies before compression.

Detects protected spans, resolves effective policy given artifact risk, and
transforms the input text:
- ``preserve`` — keep the span text unchanged.
- ``redact`` — replace with ``[REDACTED:<type>]``.
- ``anchorize`` — store the span in the anchor store and replace with the
  generated ``[CTX_REF:...]`` ID.
- ``human_review`` — signal that the artifact must not be compressed
  automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

from tekmyra.anchor_interface import AnchorStoreProtocol as AnchorStore
from tekmyra.contracts import ProtectedSpan, RiskClass
from tekmyra.fixtures import ProtectedSpanGold
from tekmyra.protected_spans import detect_protected_spans, policy_for_risk


@dataclass(frozen=True)
class SafetyMergeResult:
    transformed_text: str
    anchor_ids: list[str]
    anchor_records: list[dict]
    redacted_count: int
    anchorized_count: int
    preserved_count: int
    human_review_count: int
    human_review_required: bool
    locked_span_count: int
    locked_spans: list[ProtectedSpan]


# PINNED in-band anchor handle format. Every reader of the emitted artifact must
# know this token: verifier._MARKER_RE (one definition, used by both the entity
# check and the numeric strip), rule_router's retrieval detection, and
# block_segmenter's retrieval markers. Bump this only together with all of them,
# or an old artifact is silently readable as a new one.
#
# ORDERING IS ASYMMETRIC AND LOAD-BEARING: teach the READERS first, then change
# the emitter. Regex-first matches a format not yet produced and breaks nothing;
# splice-first fails every anchorized artifact at once.
#
# The ordinal is base-26 LETTERS, not digits, as defence in depth: a digit form
# contributes an INTRODUCED numeric and fails numeric_preserved if the strip is
# ever missed. Letters make that failure mode unreachable rather than merely
# handled. A version bump appends a letter (ANCHORB), never a digit.
_ANCHOR_HANDLE_TOKEN = "ANCHOR"


def _ordinal_letters(index: int) -> str:
    """Render an ordinal in base-26 letters so a handle carries no digit."""
    out = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        out = chr(ord("a") + rem) + out
    return out


def _redaction_marker(span: ProtectedSpan) -> str:
    return f"[REDACTED:{span.span_type}]"


def _gold_to_protected(gold: ProtectedSpanGold) -> ProtectedSpan:
    return ProtectedSpan(
        span_type=gold.span_type,
        start=gold.start,
        end=gold.end,
        text=gold.text,
        policy=gold.policy,
        detector="fixture_gold",
        confidence=1.0,
        span_id=gold.span_id,
        fact_id=gold.fact_id,
    )


def _dedupe_spans(spans: list[ProtectedSpan]) -> list[ProtectedSpan]:
    """Return spans with overlaps removed; fixture gold wins exact ties."""
    if not spans:
        return []
    sorted_spans = sorted(
        spans,
        key=lambda span: (
            span.start,
            -(span.end - span.start),
            0 if span.detector == "fixture_gold" else 1,
        ),
    )
    merged: list[ProtectedSpan] = []
    for span in sorted_spans:
        if not merged or span.start >= merged[-1].end:
            merged.append(span)
        elif span.end > merged[-1].end:
            if (span.end - span.start) > (merged[-1].end - merged[-1].start):
                merged[-1] = span
    return merged


def apply_safety_merge(
    text: str,
    risk_class: RiskClass,
    tenant_id: str,
    source_id: str,
    anchor_store: AnchorStore,
    additional_spans: list[ProtectedSpanGold] | None = None,
    doc_id: str | None = None,
    chunk_id: str | None = None,
) -> SafetyMergeResult:
    """Apply locked-span policies and return transformed text + anchor IDs.

    Spans are processed in reverse order so replacements do not invalidate
    offsets of earlier spans. ``additional_spans`` are merged with detector
    output; fixture gold labels are treated as ground-truth locked spans.
    """
    detected = detect_protected_spans(text)
    additional = [_gold_to_protected(g) for g in (additional_spans or [])]
    spans = _dedupe_spans(detected + additional)

    anchor_ids: list[str] = []
    anchor_records: list[dict] = []
    redacted_count = 0
    anchorized_count = 0
    preserved_count = 0
    human_review_count = 0

    transformed = text
    _handles_by_text: dict[str, str] = {}
    # Process from end to start so earlier offsets stay valid.
    for span in reversed(spans):
        effective_policy = policy_for_risk(span, risk_class)
        if effective_policy == "human_review":
            human_review_count += 1
            continue

        span_text = text[span.start : span.end]
        if effective_policy == "preserve":
            preserved_count += 1
            continue

        if effective_policy == "redact":
            marker = _redaction_marker(span)
            transformed = transformed[: span.start] + marker + transformed[span.end :]
            redacted_count += 1
            continue

        if effective_policy == "anchorize":
            # One record and one handle per distinct span TEXT in this
            # artifact. Anchorizing the same value 130 times minted 130
            # records and 130 handles for one secret. Pure token win, no
            # semantic change: the same value resolves to the same anchor.
            existing = _handles_by_text.get(span_text)
            if existing is not None:
                transformed = (
                    transformed[: span.start] + existing + transformed[span.end :]
                )
                anchorized_count += 1
                continue
            record = anchor_store.create(
                tenant_id=tenant_id,
                source_id=source_id,
                original_text=span_text,
                offset_start=span.start,
                offset_end=span.end,
                sensitivity=risk_class,
                doc_id=doc_id,
                chunk_id=chunk_id,
            )
            # THE OUTBOUND TEXT CARRIES A SHORT OPAQUE HANDLE, NOT THE ANCHOR ID.
            #
            # Splicing `record.anchor_id` put
            #   [CTX_REF:<tenant_id>:<doc_id>:<chunk_id>:<64-hex-hash>]
            # directly into the prompt sent to a third-party model. For a
            # PII-masking product that is a disclosure regression on our own
            # terms: we remove the customer's protected span and substitute
            # structural metadata naming WHO THEY ARE and WHICH DOCUMENT it came
            # from. An artifact-scoped handle discloses neither.
            #
            # It is also ~57 tokens against ~5. Measured on long_context_v1, one
            # incident-report fixture emitted 79 anchors consuming 4,483 tokens —
            # 69.7% of its entire output — and the pipeline returned 2.53x the
            # input tokens while reporting compression_ratio 1.0000.
            #
            # The full id stays in `anchor_ids` and `anchor_records`, which never
            # enter the outbound text. Verification and tenant isolation resolve
            # through the store against the STORED tenant, so shortening the
            # in-band form cannot weaken either: the id is a key, not a credential.
            handle = f"[{_ANCHOR_HANDLE_TOKEN}:{_ordinal_letters(len(anchor_ids))}]"
            _handles_by_text[span_text] = handle
            anchor_ids.append(record.anchor_id)
            anchor_records.append(
                {
                    "handle": handle,
                    "anchor_id": record.anchor_id,
                    "segment_id": source_id,
                    "offset_start": record.offset_start,
                    "offset_end": record.offset_end,
                    "recovery_ref": record.anchor_id,
                    "content_hash": record.content_hash,
                    "content_type": span.span_type,
                }
            )
            transformed = (
                transformed[: span.start] + handle + transformed[span.end :]
            )
            anchorized_count += 1
            continue

        # Unknown policy: fail closed by preserving.
        preserved_count += 1

    return SafetyMergeResult(
        transformed_text=transformed,
        anchor_ids=list(reversed(anchor_ids)),
        anchor_records=list(reversed(anchor_records)),
        redacted_count=redacted_count,
        anchorized_count=anchorized_count,
        preserved_count=preserved_count,
        human_review_count=human_review_count,
        human_review_required=human_review_count > 0,
        locked_span_count=len(spans),
        # Report the EFFECTIVE policy — what this merge actually did to each
        # span — not the span's declared policy. Risk-tier escalation can
        # rewrite redact->human_review and leave the text untouched; a
        # consumer reading the declared label to infer what happened to the
        # text (the verifier's retention guard does) would be misled
        # (adversarial QA rounds 3-5).
        locked_spans=[
            span.model_copy(update={"policy": policy_for_risk(span, risk_class)})
            for span in spans
        ],
    )
