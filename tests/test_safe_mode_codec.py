"""The safe-mode codec's guarantees, including the one it was missing.

``compress_safe_mode`` shipped at c2b9e45 with no test file at all. It protected
locked-span texts, their predicating clauses, the anchor/redaction handles and
every numeric — and NOT the repeated-entity retention floor, which ``verify()``
grades exactly as hard. Measured on long_context_v1 at budget 0.30, three of the
six fixtures routed here failed ``repeated_entity_retention`` and fell back to
passthrough, banking nothing after paying for the attempt.

That is the same omission ``entity_fidelity``'s own docstring records against
the original prose compressor, reintroduced in the codec written three days
after it. A codec is not safe because it protects the things its author
remembered, which is the argument for testing each contract by name rather than
testing that the codec "works".

WHAT THIS FILE CANNOT DO, STATED SO NOBODY INFERS OTHERWISE (Vega ruling 47).
The only corpus content routed to safe mode is ``distractor_dense``, whose
repeated entities are generator-planted noise a correct codec SHOULD discard.
So the fixtures we own cannot demonstrate that the entity floor protects
anything worth protecting here — they can only demonstrate that it holds. The
synthetic cases below carry real repeated entities precisely because the corpus
does not. The guarantee is right; it is not corpus-verified, and we may not
claim it is.
"""

from __future__ import annotations

from tekmyra.codecs.safe_mode import compress_safe_mode, protected_ranges
from tekmyra.entity_fidelity import retention_floor, retention_shortfalls

# Built through the SAME helper the codec-selector tests use. Hand-rolling a
# second minimal FeatureVector here would be one more copy of a definition that
# already exists, and a schema change would then leave this file asserting
# against a shape nothing else uses. The codec ignores this argument entirely
# (it takes it as ``_features``); it is supplied only to satisfy the signature.
from tests.test_codecs_selector import _features


def _document() -> str:
    """A document with a genuinely load-bearing repeated entity."""
    body = []
    for i in range(30):
        body.append(f"Platform Reliability Team reviewed segment {i} of the audit.")
        body.append("Unrelated filler prose that carries no obligation whatsoever.")
        body.append("More filler that a compressor is entirely free to discard.")
    return " ".join(body)


def _compress(text: str, spans: list[dict[str, object]], budget: float):
    return compress_safe_mode(
        text, _features(), locked_spans=spans, budget_ratio=budget
    )


def test_the_entity_floor_is_honoured_at_a_tight_budget() -> None:
    """The contract that was missing, asserted directly."""
    text = _document()
    result = _compress(text, [], 0.2)
    assert result.success, result.error
    assert retention_shortfalls(text, result.output) == {}, (
        "safe mode emitted an output that under-retains a repeated entity — "
        "verify() would reject this and the fixture would fall back to "
        "passthrough, losing the whole saving rather than part of it"
    )


def test_it_still_compresses_while_honouring_the_floor() -> None:
    """A floor that defeats compression is not a fix.

    The remedy must not become 'satisfy the check by not compressing' — the
    failure mode entity_fidelity's docstring calls out as defeating the thing
    it protects.
    """
    text = _document()
    result = _compress(text, [], 0.2)
    assert result.ratio < 0.9, f"no meaningful compression happened: ratio {result.ratio}"


def test_locked_spans_survive_the_top_up() -> None:
    """The pre-existing guarantee must not regress behind the new one."""
    text = _document() + " Incident INC-2026-07-23-001 remains open."
    spans = [{"text": "INC-2026-07-23-001", "policy": "preserve"}]
    result = _compress(text, spans, 0.2)
    assert result.success, result.error
    assert "INC-2026-07-23-001" in result.output


def test_anchor_handles_are_counted_not_merely_present() -> None:
    """Dropping 79 of 80 identical handles leaves the marker present.

    The guarantee is over COUNTS for exactly this reason, and the top-up must
    not disturb it.
    """
    text = " ".join(f"[ANCHOR:a{i}] Platform Reliability Team acted." for i in range(30))
    result = _compress(text, [], 0.2)
    assert result.success, result.error
    assert result.output.count("[ANCHOR:") == text.count("[ANCHOR:")


def test_numerics_outside_locked_spans_still_survive() -> None:
    """verify() requires them, so removing one fails the artifact regardless."""
    text = _document() + " The error budget fell to 17 percent in Q3."
    result = _compress(text, [], 0.2)
    assert result.success, result.error
    assert "17" in result.output


def test_protected_ranges_cover_handles_and_numerics() -> None:
    """The structural guarantee, checked at its own level.

    Protecting by span TEXT alone passes vacuously on anchorized content, where
    the span is not in the codec's input at all — only its handle is.
    """
    text = "Alpha [ANCHOR:x1] beta 4242 gamma."
    ranges = protected_ranges(text, [])
    covered = "".join(text[a:b] for a, b in ranges)
    assert "[ANCHOR:x1]" in covered
    assert "4242" in covered


def test_a_document_with_no_repeated_entity_is_left_to_the_budget() -> None:
    """The floor must be inert where it has no obligation, not a tax on everything.

    DELIBERATELY FREE OF DIGITS. An earlier version of this case numbered its
    sentences, and every one was then pinned by the numeric guard — the codec
    returned ratio 1.0 and the test read as the entity floor refusing to
    compress. It was the numeric contract working correctly. A negative case has
    to exclude the OTHER protections or it cannot say anything about this one.
    """
    filler = [
        "the quiet meadow shifted under a pale sky",
        "a distant kettle whistled in the empty kitchen",
        "the lantern guttered and went out without ceremony",
        "someone had left the gate unlatched again",
        "rain moved across the roof in long slow passes",
        "the corridor smelled faintly of wet paper",
    ]
    text = ". ".join(filler * 10) + "."
    tight = _compress(text, [], 0.2)
    assert tight.success, tight.error
    assert tight.ratio < 0.5, (
        "the entity floor inflated an output that carries no repeated entity"
    )


def test_the_floor_tracks_the_output_not_the_budget() -> None:
    """Producer and check agree because both read the achieved ratio."""
    text = _document()
    result = _compress(text, [], 0.2)
    achieved = len(result.output) / len(text)
    floor = retention_floor(achieved)
    kept = result.output.count("Platform Reliability Team")
    total = text.count("Platform Reliability Team")
    assert kept / total >= floor, (
        f"retention {kept}/{total} is below the floor {floor:.4f} implied by the "
        f"output's own length ratio {achieved:.4f}"
    )
