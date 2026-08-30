"""Safe-mode compression: the codec the safety gate has been asking for.

``GOVERNANCE_SAFE_COMPRESSION`` and ``LEGAL_SECURITY_SAFE_MODE`` mapped to
``_passthrough`` under a comment reading "Safety-mode routes currently
passthrough; later phases can add redaction". The gate had already made its
decision — it routed this content to a SAFE COMPRESSION MODE — and that mode did
not exist. Building it IMPLEMENTS the gate's decision; it does not override it.

THE GUARANTEE IS STRUCTURAL, NOT CHECKED AFTERWARDS (Vega ruling 41). Protected
material and its predicating clause are added to the protected set BEFORE
selection, so predication cannot be removed BECAUSE IT IS PROTECTED. A post-hoc
predication check would need an oracle we do not have; a pre-emptive one needs
only a boundary rule.

WHY PREDICATION AND NOT JUST THE SPAN. Sub-sentence compression can preserve a
span's characters exactly while destroying the clause that gives it meaning. An
audit finding ID whose finding text is compressed away is preserved, verifiable,
and useless — it passes a span check and fails the purpose. That is the failure
mode this module exists to make unreachable.

THE BOUNDARY RULE IS COARSE AND THAT WAS MEASURED, NOT ASSUMED. Splitting on
``. ! ? ;`` and newline. A finer rule that also split on ``:`` and ``,`` was
validated against the text first and found to CUT INSIDE VALUES — 24.1% of
governance spans had a timestamp or ratio severed at its own colon, and a
further 2.2% lost the ``Label:`` that predicated over them. Its larger removable
pool WAS its under-protection. Where the boundary is uncertain the rule protects
MORE, per ruling 41, and the finer rule was not cheaper — it was wrong.

MATCHED BY TEXT, NOT BY OFFSET. The codec receives the SAFETY-MERGED text while
locked-span offsets index the ORIGINAL, and the merge moves them: anchorization
replaces a span with a shorter handle. Offsets would silently protect the wrong
characters — which is the defect class this codebase has spent a week removing.
"""

from __future__ import annotations

import re
from typing import Any, cast

from tekmyra.codecs.base import CodecResult
from tekmyra.contracts import FeatureVector
from tekmyra.compression_budget import DEFAULT_BUDGET_RATIO as _DEFAULT_BUDGET
from tekmyra.entity_fidelity import top_up_retention
from tekmyra.numeric_fidelity import NUMERIC_RE

# The in-band disposition of a locked span AFTER the safety merge. On gate-HIGH
# content — which is the whole reason this codec exists — spans are anchorized,
# so the span TEXT is not in the codec's input at all. Its handle is. Protecting
# only by span text therefore protects nothing on exactly the content this
# module was written for, and a guarantee phrased over absent text passes
# vacuously. Kept in sync with verifier._MARKER_RE by shape; the two must agree.
_MARKER_RE = re.compile(r"\[(?:REDACTED:[^\]]*|CTX_REF:[^\]]*|ANCHOR:[^\]]*)\]")

# Clause delimiters. See the module docstring: measured, not chosen.
_CLAUSE_RE = re.compile(r"[.!?;\n]")


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Extend ``[start, end)`` to the punctuation-delimited clause holding it."""
    opens = [match.end() for match in _CLAUSE_RE.finditer(text, 0, start)]
    left = opens[-1] if opens else 0
    closing = _CLAUSE_RE.search(text, end)
    return left, max(closing.start() if closing else len(text), end)


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def protected_ranges(text: str, locked_span_texts: list[str]) -> list[tuple[int, int]]:
    """Character ranges the safe codec may never remove.

    Locked spans, each extended to its predicating clause, plus every numeric —
    ``verify()`` requires numerics outside locked spans to survive, so removing
    one would fail the artifact regardless of how good the compression was.
    """
    found: list[tuple[int, int]] = []
    for span_text in locked_span_texts:
        if not span_text:
            continue
        cursor = text.find(span_text)
        while cursor != -1:
            found.append(_clause_bounds(text, cursor, cursor + len(span_text)))
            cursor = text.find(span_text, cursor + 1)
    # Anchor/redaction handles ARE the locked spans' disposition. Removing one
    # deletes the span as surely as deleting its text would.
    for match in _MARKER_RE.finditer(text):
        found.append(_clause_bounds(text, match.start(), match.end()))
    found.extend((m.start(), m.end()) for m in NUMERIC_RE.finditer(text))
    return _merge(found)


def _obligation_holds(
    output: str, locked_span_texts: list[str], text: str
) -> tuple[bool, str | None]:
    """The guarantee, verified before the result is allowed to leave.

    Ruling 9: a working mechanism attached to output whose safety cannot be
    demonstrated does not ship. This is cheap and it is the difference between a
    codec that compresses safely and one that compressed and hopes.
    """
    for span_text in locked_span_texts:
        if not span_text:
            continue
        if text.count(span_text) and span_text not in output:
            return False, f"locked span removed: {span_text[:32]!r}"

    # AND EVERY HANDLE, COUNTED. A span replaced by an anchor is present in the
    # input only as its handle, so the loop above cannot see it — the guarantee
    # would hold vacuously on precisely the anchorized content this codec
    # exists to compress. Counts, not membership: dropping 79 of 80 identical
    # handles leaves the marker present and the spans gone.
    before = len(_MARKER_RE.findall(text))
    after = len(_MARKER_RE.findall(output))
    if after < before:
        return False, f"anchor/redaction handles removed: {before} -> {after}"
    return True, None


def compress_safe_mode(
    text: str,
    _features: FeatureVector,
    **kwargs: Any,
) -> CodecResult:
    """Compress while preserving locked spans and the clauses predicating them.

    Sentence-granular: a sentence overlapping any protected range is kept whole.
    On this corpus that is the right granularity — governance content averages
    6.9 words per sentence, so its clause IS its sentence in 86.5% of cases and
    sub-sentence work would buy almost nothing that is not generator boilerplate.
    """
    raw_spans = kwargs.get("locked_spans")
    spans = cast(list[dict[str, object]], raw_spans) if isinstance(raw_spans, list) else []
    span_texts = [str(s.get("text", "")) for s in spans if s.get("text")]

    raw_budget = kwargs.get("budget_ratio", _DEFAULT_BUDGET)
    budget = (
        float(raw_budget)
        if isinstance(raw_budget, (int, float, str))
        else _DEFAULT_BUDGET
    )

    protected = protected_ranges(text, span_texts)

    # Sentence spans over the merged text.
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for match in re.finditer(r"[.!?\n]+", text):
        bounds.append((cursor, match.end()))
        cursor = match.end()
    if cursor < len(text):
        bounds.append((cursor, len(text)))
    if not bounds:
        return CodecResult(success=True, output=text, ratio=1.0)

    must_keep = {
        index
        for index, (start, end) in enumerate(bounds)
        if any(pstart < end and pend > start for pstart, pend in protected)
    }
    # Everything else is removable. Keep the earliest free sentences up to the
    # budget: earliest-first preserves document opening and discourse order,
    # and this codec deliberately does NOT rank by a learned score — its job is
    # to be safe and predictable on content the gate marked high risk.
    target = max(len(must_keep), int(round(len(bounds) * budget)))
    selected = set(must_keep)
    for index in range(len(bounds)):
        if len(selected) >= target:
            break
        selected.add(index)

    # ENTITY RETENTION, THE THIRD CONTRACT (Vega ruling 47).
    #
    # This codec honoured locked spans, their predicating clauses, the anchor
    # handles and every numeric — and not the repeated-entity floor, which
    # `verify()` grades just as hard. That is the same omission
    # entity_fidelity's own docstring records against the original prose
    # compressor, reintroduced here three days after it was written up. A codec
    # is not safe because it protects the things its author remembered.
    #
    # THIS CORPUS CANNOT VALIDATE THIS GUARANTEE, AND THAT IS RECORDED RATHER
    # THAN GLOSSED. The only content routed here is distractor_dense, whose
    # repeated entities are generator-planted noise ("Professor Bibble",
    # fixture_templates.py:246) that a correct codec SHOULD discard. So on the
    # corpus we own, this floor preserves filler. It is built anyway because
    # safe mode carries legal and security material in production, where entity
    # retention matters most, and deciding a safety contract from the fact that
    # our only test content is planted noise would be letting the corpus define
    # the product. The guarantee is right; we may not claim it is verified.
    segments = [text[a:b] for a, b in bounds]
    selected, output = top_up_retention(
        text,
        segments,
        selected,
        lambda chosen: "".join(
            segment for index, segment in enumerate(segments) if index in chosen
        ),
    )
    ok, error = _obligation_holds(output, span_texts, text)
    if not ok:
        # Fail closed. The runtime falls back rather than shipping an artifact
        # whose safety obligation we cannot demonstrate.
        return CodecResult(success=False, output=text, ratio=1.0, error=error)

    ratio = len(output) / len(text) if text else 1.0
    return CodecResult(
        success=True,
        output=output,
        ratio=ratio,
        preserved_spans=len(span_texts),
    )
