"""Post-compression verifier v1.

Fails closed: any unverifiable or mutated content flags the segment so the
runtime can fall back to a safer route.  All checks are deterministic and
content-free telemetry-friendly where possible.

Protected-span verification is identity-driven: each locked span is tracked by
its stable ``span_id`` and ``fact_id`` rather than by substring count alone.
A span is considered correctly disposed when it appears exactly once as a
verbatim substring, an approved redaction marker, or a resolvable anchor.
"""

from __future__ import annotations

import bisect
import hmac
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from tekmyra.anchor_interface import AnchorStoreProtocol as AnchorStore
from tekmyra.codecs.base import CodecResult
from tekmyra.contracts import FeatureVector, RouteClass
from tekmyra.entity_fidelity import (
    ENTITY_RE,
    MIN_OCCURRENCES,
    RETENTION_FACTOR,
    content_tokens,
    retention_floor,
    sentence_initial,
    truncate_at_sentence_end,
)
from tekmyra.numeric_fidelity import (
    extract_numerics,
    extract_numerics_in_spans,
)
from tekmyra.sentence_boundaries import ABBREVIATIONS as _ABBREVIATIONS

# The numeric literal matcher now lives in tekmyra.numeric_fidelity so
# producers and this verifier share one definition. A local copy here would be a
# second definition free to drift from the one codecs compress against.

# THE ENTITY CONTRACT NOW LIVES IN A SHARED LEAF. A producer that re-expressed
# this definition with its own regex would drift from the check it must satisfy
# — the same reason numeric_fidelity exists. The compressor's featuriser reads
# the SAME mention definition and the SAME floor from entity_fidelity, so the
# share it preserves and the share this check demands cannot disagree.
# How far back to keep scanning for an overlapping span before giving up.
# Locked spans are short; a span starting this far before an entity cannot
# reach it, and the bound keeps the scan linear on span-dense documents.
_MAX_SPAN_LOOKBACK = 4096
_ENTITY_MIN_OCCURRENCES = MIN_OCCURRENCES
_ENTITY_RETENTION_FACTOR = RETENTION_FACTOR
_ENTITY_RE = ENTITY_RE
_truncate_at_sentence_end = truncate_at_sentence_end
_entity_content_tokens = content_tokens
_sentence_initial = sentence_initial
# ANCHOR is the short in-band anchor handle. It MUST be stripped here, and
# its ordinal is base-26 LETTERS rather than digits as defence in depth: a
# digit form like "[ANC1:7]" contributes a numeric "7" that was never in the
# source and fails numeric_preserved on an INTRODUCED numeric if this strip
# is ever missed. Measured. Letters make that failure mode unreachable.
_MARKER_RE = re.compile(r"\[(?:REDACTED:[^\]]*|CTX_REF:[^\]]*|ANCHOR:[^\]]*)\]")

# WHY a span was transformed, not merely THAT it was — the whole of
# "motive-aware".
#
# A policy that REMOVES text does so for SECRECY: the value must not reach the
# model, so its absence from the output is COMPLIANCE. A policy that KEEPS text
# does so for PROVENANCE: the value is not secret and traceability is the point,
# so its absence is LOSS. One is the product working; the other is a defect.
# Length alone cannot tell them apart, which is why a length-only guard had to
# exempt itself from exactly the cases it existed to judge.
#
# The effective policy is the motive carrier, and correctly so: risk escalation
# rewrites it, and when it does the motive genuinely changes. A citation that is
# preserve-by-default becomes anchorize at HIGH risk because the operator chose
# secrecy over provenance for that tier — reading the motive off the effective
# policy follows that choice instead of contradicting it. Reading it off the
# span TYPE would not.
_SECRECY_POLICIES = frozenset({"redact", "anchorize"})


def _secrecy_removed_chars(
    locked_spans: list[dict[str, Any]], span_texts: list[str]
) -> int:
    """Characters the safety merge removed from the original for secrecy.

    Spans arrive deduplicated and non-overlapping from ``apply_safety_merge``,
    so their lengths sum without double counting. Handle deduplication does not
    change this: reusing one handle for a repeated value still replaces every
    occurrence, and every occurrence is still its own span.
    """
    return sum(
        len(text)
        for span, text in zip(locked_spans, span_texts)
        if str(span.get("policy", "preserve")) in _SECRECY_POLICIES
    )


def _check_repeated_entity_retention(
    original: str,
    output: str,
    locked_spans: list[dict[str, Any]],
    route: RouteClass,
) -> VerificationCheck:
    """Fail when a frequently repeated entity is near-eliminated.

    The presence-based checks see a span that survives ONCE and are
    satisfied; a name mentioned 52 times surviving 4 times is invisible to
    them (the Slice-4 repetition-collapse defect). This check compares
    per-entity frequency against the output's own length ratio — repetition
    must compress at roughly the document's rate, not disappear. Fails
    closed into the existing fallback machinery.
    """
    if route == RouteClass.DIFF_COMPRESSOR:
        # Diff compression drops unchanged context lines by design.
        #
        # MEASURED, not assumed — the argument that retired the length exemption
        # applies to this one too, and an exemption whose premise nobody has
        # checked is a hole with a comment on it. All 20 diff fixtures in the
        # synthetic corpus were run through the real pipeline and their entity
        # check re-evaluated with this exemption removed: 0 of 20 would fail.
        # So it is currently costing nothing, which is an argument for deleting
        # it rather than for keeping it. Left in place deliberately — unlike the
        # length exemption it is NAMED in the result, so a reader can see it
        # fired; and no corpus we own exercises diff compression hard enough to
        # evidence the case it was written for.
        return VerificationCheck(
            "repeated_entity_retention", True, "diff_route_exempt"
        )
    span_texts = [str(span.get("text", "")) for span in locked_spans]
    cleaned_output = _MARKER_RE.sub("", output)

    # THE RATIO IS MEASURED ON THE MATERIAL THAT WAS SUPPOSED TO SURVIVE.
    #
    # A raw len(output)/len(original) charges the compressor for both halves of
    # secrecy masking: the original's secret text is gone from the numerator
    # (correctly removed) while still inflating the denominator, and the
    # replacement handles inflate the numerator while representing no content.
    # On short, secret-dense content the two together pushed the ratio above
    # 1.0, and the guard exempted itself precisely where masking was heaviest.
    #
    # Correcting both sides — strip the handles from the output, discount the
    # secrecy-removed text from the original — leaves a ratio that describes
    # what compression did to the PROVENANCE material, which is the only
    # material this check has an opinion about.
    #
    # STRIPPED FROM BOTH SIDES. Stripping the numerator alone was asymmetric:
    # a source document that already contains a literal "[REDACTED:…]" — which
    # 20 of our 1,461 fixtures do — had those characters removed from the output
    # and still counted in the original, depressing the ratio and with it the
    # floor. That is the FAIL-OPEN direction for a guard, which is why it is
    # fixed rather than noted. Markers the merge emitted exist only in the
    # output and are still stripped from the numerator alone, which is correct:
    # they represent no content on either side.
    cleaned_original = _MARKER_RE.sub("", original)
    free_original_chars = max(
        1, len(cleaned_original) - _secrecy_removed_chars(locked_spans, span_texts)
    )
    length_ratio = len(cleaned_output) / free_original_chars
    # NO LENGTH EXEMPTION (ruling 4). This check used to return True whenever
    # the raw output was longer than the input, which is the condition anchor
    # bloat guaranteed — so on anchorized content it never ran at all, and the
    # content loss it was written to catch went unreported for its whole life.
    #
    # Deleting it required ruling 3 first. Output-longer-than-input has two
    # unrelated causes, and only the motive tells them apart: secrecy masking
    # that legitimately expands short content, versus a compressor that grew
    # the text while losing entities from it. Against the raw ratio, deletion
    # alone failed 22 tests that were the product working as designed. Against
    # the corrected ratio above, the expansion caused by masking is already
    # discounted, so what remains for this check to judge is only the second.
    #
    # ONE FUNCTION, BOTH SIDES — see entity_fidelity.retention_floor.
    floor = retention_floor(length_ratio)
    # WHICH OCCURRENCES WERE DESTROYED BY MASKING — read from OFFSETS, not from
    # substring arithmetic.
    #
    # The first version credited removal by counting the entity inside each
    # masked span's text. That only ever sees containment in ONE direction, and
    # the common case runs the other way: the span is "INC-2026-07-23-001" and
    # the entity is "Incident INC-2026-07-23-001". No span equals the entity, so
    # the credit was zero, the entity read as eliminated 20->0, and the artifact
    # was REFUSED for masking its own incident ID correctly. Three fixtures on
    # long_context_v1 were refused by this check alone — the exact category
    # confusion ruling 3 exists to remove, reintroduced by the mechanism meant
    # to implement it.
    #
    # An entity occurrence that OVERLAPS a secrecy-masked span cannot survive
    # verbatim and must not be expected to. Offsets decide that exactly, in
    # either direction of containment and for partial overlaps too.
    secrecy_bounds = sorted(
        (int(span["start"]), int(span["end"]))
        for span in locked_spans
        if str(span.get("policy", "preserve")) in _SECRECY_POLICIES
        and span.get("start") is not None
        and span.get("end") is not None
    )
    _starts = [start for start, _ in secrecy_bounds]

    def _masked(start: int, end: int) -> bool:
        """True when [start, end) touches any secrecy-removed span."""
        index = bisect.bisect_right(_starts, end - 1)
        # Scan back over spans starting at or before this occurrence ends; the
        # first one ending after it starts is an overlap.
        for span_start, span_end in reversed(secrecy_bounds[:index]):
            if span_end > start:
                return True
            if span_end <= start and span_start < start - _MAX_SPAN_LOOKBACK:
                break
        return False

    counts: Counter[str] = Counter()
    masked_counts: Counter[str] = Counter()
    for match in _ENTITY_RE.finditer(original):
        entity = _truncate_at_sentence_end(match.group(0))
        content = _entity_content_tokens(entity.split())
        if content == 0:
            continue
        if content == 1 and _sentence_initial(original, match.start()):
            # Capitalization explained by sentence position proves nothing.
            continue
        counts[entity] += 1
        if secrecy_bounds and _masked(match.start(), match.start() + len(entity)):
            masked_counts[entity] += 1
    # Only spans whose policy removed text (redaction, anchorization) reduce the
    # retention obligation. The policy field here is the EFFECTIVE policy
    # reported by apply_safety_merge — what actually happened to the span, with
    # risk-tier escalation already applied — so it is authoritative.
    # Reconstructing removal from the output text instead is unsound: substring
    # and marker-counting schemes were each defeated by adversarial QA (rounds
    # 3-5: escalation staleness, bare-name survivors voiding real credit,
    # cross-entity marker borrowing). Preserve/human_review spans never subtract.
    #
    # Collapsed to DISTINCT span texts, carrying each one's multiplicity, and
    # hoisted out of the entity loop. Identical arithmetic — the same texts are
    # still counted the same number of times — but the loop below was O(entities
    # x spans) and ran once per entity. That cost was invisible while the length
    # exemption returned early on exactly the anchorized documents where span
    # counts are highest; deleting the exemption made it live, on the longest
    # documents we have. Anchor deduplication is what makes the collapse pay:
    # one secret repeated 130 times is 130 spans but a single distinct text.
    secrecy_texts: Counter[str] = Counter(
        text
        for span, text in zip(locked_spans, span_texts)
        if str(span.get("policy", "preserve")) in _SECRECY_POLICIES and text
    )
    violations: list[str] = []
    for entity, original_count in counts.items():
        # Offsets when we have them; the text-based credit remains for callers
        # that supply spans without positions (older fixtures, direct tests).
        # It only ever catches entity-inside-span, which is why it cannot be
        # the primary any more.
        if secrecy_bounds:
            removed_by_policy = masked_counts[entity]
        else:
            removed_by_policy = sum(
                occurrences * text.count(entity)
                for text, occurrences in secrecy_texts.items()
            )
        free_count = original_count - removed_by_policy
        if free_count < _ENTITY_MIN_OCCURRENCES:
            continue
        retention = cleaned_output.count(entity) / free_count
        if retention < floor:
            violations.append(f"{entity}: {original_count}->" f"{cleaned_output.count(entity)}")
    if not violations:
        return VerificationCheck("repeated_entity_retention", True)
    detail = f"{len(violations)} repeated entit(y/ies) near-eliminated: " + "; ".join(
        sorted(violations)[:5]
    )
    return VerificationCheck("repeated_entity_retention", False, detail)


# The numeric contract lives in numeric_fidelity so producers can honour exactly
# what this module enforces. These names are kept as the local spelling used
# throughout the checks below.
_extract_numerics = extract_numerics
_extract_numerics_in_spans = extract_numerics_in_spans


def _ast_parse_safe(text: str, language: str | None) -> tuple[bool, str | None]:
    """Best-effort AST parse for Python; returns (success, error)."""
    if language == "python":
        try:
            compile(text, "<verify>", "exec")
            return True, None
        except SyntaxError as exc:
            return False, f"python_syntax_error: {exc}"
        except Exception as exc:
            return False, f"python_compile_error: {exc}"
    return True, None


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    passed: bool
    detail: str | None = None


@dataclass(frozen=True)
class VerifierResult:
    passed: bool
    checks: list[VerificationCheck] = field(default_factory=list)
    numeric_mutations: int = 0
    locked_spans_missing: int = 0
    protected_disposition_valid: bool = False
    introduced_duplicate_count: int = 0


def _span_key(span: dict[str, Any]) -> str:
    """Return a stable identity key for a locked span.

    Falls back from ``fact_id`` to ``span_id`` to the verbatim text so older
    fixtures without IDs continue to work during the transition.
    """
    return str(
        span.get("fact_id")
        or span.get("span_id")
        or span.get("text")
        or ""
    )


def _count_exact_span_dispositions(
    output: str,
    span: dict[str, Any],
    anchor_ids: list[str],
    anchor_store: AnchorStore | None,
    handles_by_anchor: dict[str, str] | None = None,
) -> int:
    """Count identity-specific verbatim and anchor representations.

    ONE ANCHOR CAN DISPOSE MANY SPANS, and counting it once was a false-refusal
    factory. Anchor deduplication mints a single record per distinct span TEXT
    while ``locked_spans`` stays PER SITE, so a value appearing 19 times is 19
    locked spans and 1 anchor. Crediting +1 per resolving anchor left 18 of them
    reading as MISSING — content that was masked correctly, present in the
    output as 19 handles, and reported as lost.

    Measured on long_context_v1: every one of the 7 refusals satisfied
    locked_missing == spans - distinct_texts EXACTLY (79 spans, 8 distinct, 8
    anchors, all 8 resolving, 79 handles in the output, 71 "missing"). It is the
    same category confusion ruling 3 removed from the entity check — "removed
    for secrecy" scored as "lost" — surviving in a second check that ruling
    never touched.

    The credit is the HANDLE COUNT IN THE OUTPUT, not the span count. That keeps
    the duplicate and drop guards intact: a compressor that deletes anchored
    content emits fewer handles and the missing spans are still reported. The
    handle is read from the merge's own records rather than reconstructed from
    the ordinal convention, so the two modules do not encode it twice.
    """
    span_text = span.get("text") or ""
    count = 0

    if span_text:
        count += output.count(span_text)

    if span_text and anchor_store is not None:
        for aid in anchor_ids:
            if not _anchor_resolves_to_span(aid, span_text, anchor_store):
                continue
            handle = (handles_by_anchor or {}).get(aid)
            # Without a handle mapping the caller is pre-dedup or a test; one
            # occurrence is the historical meaning and stays the fallback.
            count += output.count(handle) if handle else 1

    return count


def _anchor_resolves_to_span(
    anchor_id: str,
    span_text: str,
    anchor_store: AnchorStore,
) -> bool:
    """Verify anchor identity without assuming an unsalted hash format."""
    parsed = anchor_id.split(":")
    if len(parsed) < 2:
        return False
    record = anchor_store.get(anchor_id, parsed[1])
    return record is not None and hmac.compare_digest(record.original_text, span_text)


def _anchor_resolves(anchor_id: str, anchor_store: AnchorStore) -> bool:
    # The original tenant is encoded in the anchor ID; for verification we
    # simply check existence. Tenant enforcement is handled at retrieval time.
    parsed = anchor_id.split(":")
    if len(parsed) >= 2:
        tenant_id = parsed[1]
        return anchor_store.get(anchor_id, tenant_id) is not None
    return False


def _check_protected_disposition(
    original: str,
    output: str,
    locked_spans: list[dict[str, Any]],
    anchor_ids: list[str],
    anchor_store: AnchorStore | None,
    fact_index: dict[str, str] | None,
    route: RouteClass | None = None,
    handles_by_anchor: dict[str, str] | None = None,
) -> tuple[VerificationCheck, VerificationCheck | None, int, int, bool]:
    """Return locked-span disposition check, optional fact check, and counts.

    The disposition check passes when every locked span is represented exactly
    once.  The optional fact check verifies that any ``fact_id`` present in a
    locked span resolves to canonical text in the output or in a sanctioned
    disposition form.

    Diff routes legitimately render the same path twice (e.g.,
    ``a/src/config.yaml`` and ``b/src/config.yaml``), so duplicate span text
    is not treated as a defect for those routes.
    """
    locked_missing = 0
    duplicates = 0
    diff_route = route == RouteClass.DIFF_COMPRESSOR

    # Allocate identity-specific verbatim/anchor occurrences one-to-one.  A
    # repeated protected value may legitimately correspond to multiple locked
    # spans, while one locked span emitted twice remains a duplicate.
    disposed: set[int] = set()
    by_text: dict[str, list[int]] = {}
    for index, span in enumerate(locked_spans):
        text = span.get("text") or ""
        if text:
            by_text.setdefault(text, []).append(index)

    for text, indices in by_text.items():
        count = _count_exact_span_dispositions(
            output, locked_spans[indices[0]], anchor_ids, anchor_store,
            handles_by_anchor,
        )
        allocated = min(count, len(indices))
        disposed.update(indices[:allocated])
        allowed_occurrences = max(len(indices), original.count(text))
        if count > allowed_occurrences and not diff_route:
            duplicates += count - allowed_occurrences

    # Generic redaction markers do not encode which same-type entity they
    # replaced.  Allocate them one-to-one only among otherwise undisposed spans
    # of that type; this avoids treating two distinct emails as duplicates.
    remaining_by_type: dict[str, list[int]] = {}
    for index, span in enumerate(locked_spans):
        if index in disposed:
            continue
        span_type = str(span.get("span_type", "span"))
        remaining_by_type.setdefault(span_type, []).append(index)

    for span_type, indices in remaining_by_type.items():
        marker_count = output.count(f"[REDACTED:{span_type}]")
        allocated = min(marker_count, len(indices))
        disposed.update(indices[:allocated])
        if marker_count > len(indices) and not diff_route:
            duplicates += marker_count - len(indices)

    locked_missing = len(locked_spans) - len(disposed)
    all_valid = locked_missing == 0 and duplicates == 0

    disposition_check = VerificationCheck(
        "locked_spans_preserved",
        all_valid and locked_missing == 0 and duplicates == 0,
        f"missing={locked_missing} duplicates={duplicates}",
    )

    fact_check: VerificationCheck | None = None
    if fact_index:
        missing_facts = []
        for fact_id, canonical in fact_index.items():
            if not canonical:
                continue
            if canonical not in output:
                # An allocated sanctioned redaction/anchor disposition also
                # satisfies the fact without conflating same-type entities.
                found = any(
                    index in disposed and span.get("fact_id") == fact_id
                    for index, span in enumerate(locked_spans)
                )
                if not found:
                    missing_facts.append(fact_id)
        fact_check = VerificationCheck(
            "facts_preserved",
            not missing_facts,
            f"missing_facts={missing_facts}" if missing_facts else None,
        )
        if missing_facts:
            all_valid = False

    return disposition_check, fact_check, locked_missing, duplicates, all_valid


def verify(
    original: str,
    codec_result: CodecResult,
    route: RouteClass,
    features: FeatureVector,
    locked_spans: list[dict[str, Any]],
    anchor_ids: list[str] | None = None,
    anchor_store: AnchorStore | None = None,
    fact_index: dict[str, str] | None = None,
    anchor_records: list[dict[str, Any]] | None = None,
) -> VerifierResult:
    """Verify that a codec result safely preserves the original content.

    Args:
        original: Original uncompressed text.
        codec_result: Result from the codec.
        route: Route that was applied.
        features: Deterministic feature vector.
        locked_spans: Span gold records (must contain ``text`` key; ``span_id``
            and ``fact_id`` are used when present).
        anchor_ids: Anchor IDs that should resolve if dropped from output.
        anchor_store: Store used to resolve anchor IDs.
        fact_index: Optional fact_id -> canonical text index for fact-level
            verification.
        anchor_records: Optional merge records carrying ``handle`` and
            ``anchor_id``. Supplies the handle each anchor was spliced as, so
            one deduplicated anchor disposes every span it actually masked
            rather than exactly one of them.

    Returns:
        VerifierResult with pass/fail and per-check details.
    """
    checks: list[VerificationCheck] = []
    anchor_ids = anchor_ids or []

    # Codec must report success; otherwise fail closed.
    if not codec_result.success:
        checks.append(VerificationCheck("codec_success", False, codec_result.error))
        return VerifierResult(passed=False, checks=checks)
    checks.append(VerificationCheck("codec_success", True))

    output = codec_result.output

    # JSON validity check for JSON route.
    if route == RouteClass.DETERMINISTIC_JSON_COMPRESSOR:
        try:
            parsed = json.loads(output)
            if not isinstance(parsed, (dict, list)):
                checks.append(VerificationCheck("json_valid", False, "json_root_not_object_or_array"))
            else:
                checks.append(VerificationCheck("json_valid", True))
        except Exception as exc:
            checks.append(VerificationCheck("json_valid", False, f"json_parse_error: {exc}"))
    else:
        checks.append(VerificationCheck("json_valid", True, "not_applicable"))

    # Code corruption check for code route.
    if route == RouteClass.CODE_COMPRESSOR and features.ast_parse_success:
        ok, err = _ast_parse_safe(output, features.code_language)
        if ok:
            checks.append(VerificationCheck("code_corruption", True))
        else:
            checks.append(VerificationCheck("code_corruption", False, err))
    else:
        checks.append(VerificationCheck("code_corruption", True, "not_applicable"))

    # Locked-span preservation by stable identity.
    disposition_check, fact_check, locked_missing, duplicates, disposition_valid = (
        _check_protected_disposition(
            original,
            output,
            locked_spans,
            anchor_ids,
            anchor_store,
            fact_index,
            route=route,
            handles_by_anchor={
                str(rec.get("anchor_id")): str(rec.get("handle"))
                for rec in (anchor_records or [])
                if rec.get("anchor_id") and rec.get("handle")
            },
        )
    )
    checks.append(disposition_check)
    if fact_check is not None:
        checks.append(fact_check)

    # Anchor resolution sanity check.
    bad_anchors = [aid for aid in anchor_ids if anchor_store is None or not _anchor_resolves(aid, anchor_store)]
    if not bad_anchors:
        checks.append(VerificationCheck("anchors_resolvable", True))
    else:
        checks.append(
            VerificationCheck(
                "anchors_resolvable",
                False,
                f"{len(bad_anchors)} anchor(s) do not resolve",
            )
        )

    # Numeric mutation check.
    # Diff compression intentionally drops unchanged context lines, which may
    # contain numeric values; comparing numeric sets across the whole original
    # would therefore produce false positives.  For diff routes we trust the
    # hunk headers and changed-line preservation and skip this check.
    #
    # Safety-merge redaction/anchorization intentionally removes numeric
    # values inside locked spans.  Only those span-internal numerics are exempt
    # from the numeric comparison; numerics outside locked spans must still
    # match.  Strip markers from the output so anchor IDs do not pollute the
    # numeric set.
    if route == RouteClass.DIFF_COMPRESSOR:
        checks.append(VerificationCheck("numeric_preserved", True, "diff_route_exempt"))
        numeric_mutations = 0
    else:
        original_numerics = _extract_numerics(original)
        locked_span_numerics = _extract_numerics_in_spans(original, locked_spans)
        required_numerics = original_numerics - locked_span_numerics

        # Single definition, deliberately. This was an inline duplicate of
        # _MARKER_RE: two copies of one safety pattern, where updating the
        # named constant and missing this one would strip anchors on one
        # path and not the other and read as an intermittent failure.
        cleaned_output = _MARKER_RE.sub("", output)
        output_numerics = _extract_numerics(cleaned_output) - locked_span_numerics

        # Ignore false positives from format changes (e.g., JSON whitespace removal).
        # A numeric mutation means a numeric value present in the original is absent
        # or a new numeric value appeared in the output.
        missing = required_numerics - output_numerics
        extra = output_numerics - required_numerics
        numeric_mutations = len(missing) + len(extra)
        if numeric_mutations == 0:
            checks.append(VerificationCheck("numeric_preserved", True))
        else:
            checks.append(
                VerificationCheck(
                    "numeric_preserved",
                    False,
                    f"missing={len(missing)} extra={len(extra)}",
                )
            )

    checks.append(
        _check_repeated_entity_retention(original, output, locked_spans, route)
    )

    passed = all(check.passed for check in checks)
    return VerifierResult(
        passed=passed,
        checks=checks,
        numeric_mutations=numeric_mutations,
        locked_spans_missing=locked_missing,
        protected_disposition_valid=disposition_valid and locked_missing == 0 and duplicates == 0,
        introduced_duplicate_count=duplicates,
    )
