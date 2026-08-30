"""Sentence segmentation with original-offset tracking."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tekmyra.sentence_boundaries import ABBREVIATIONS as _ABBREVIATIONS

_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?\n])\s+")
# Final one- or two-token run ending in a period, e.g. "Dr." or "et al.".
# The token must be whitespace-delimited (or start the fragment): without
# that boundary, ordinal endings match abbreviation entries ("1st." ends
# with "st") and sentences wrongly merge — adversarial QA, 2026-08-02.
# Allowing start-of-fragment is a deliberate, one-directional improvement
# over the baseline splitter's strict (?<=\s), which fails to protect an
# abbreviation at position 0 ("Dr. Smith..." opening a document).
_FINAL_TOKEN_RE = re.compile(r"(?:^|(?<=\s))([A-Za-z]{1,7})\.\s*$")
_FINAL_TWO_TOKEN_RE = re.compile(r"(?:^|(?<=\s))([A-Za-z]{1,7}\s+[A-Za-z]{1,7})\.\s*$")


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int
    index: int


def _ends_with_abbreviation(fragment: str) -> bool:
    """True when the fragment's trailing period belongs to an abbreviation.

    "Dr. Aris Thorne adjusted the lens." must stay ONE sentence: splitting
    after "Dr." leaves a bare-title fragment whose repetitions the selector
    then treats as a duplicate family (the Slice-4 repetition-collapse
    defect). The abbreviation set is shared with the baseline splitter so
    the two segmenters cannot drift apart.
    """
    two = _FINAL_TWO_TOKEN_RE.search(fragment)
    if two is not None and two.group(1).lower() in _ABBREVIATIONS:
        return True
    one = _FINAL_TOKEN_RE.search(fragment)
    return one is not None and one.group(1).lower() in _ABBREVIATIONS


def segment_sentences(text: str) -> list[Sentence]:
    """Split text into sentences and record original character offsets."""
    if not text:
        return []

    breaks: list[tuple[int, int]] = []
    for match in _SENTENCE_BREAK_RE.finditer(text):
        # The lookbehind character is the last char before the whitespace.
        # Only a '.' boundary can belong to an abbreviation; '!', '?', and
        # newline boundaries always split.
        if text[match.start() - 1] == "." and _ends_with_abbreviation(
            text[: match.start()]
        ):
            continue
        breaks.append((match.start(), match.end()))

    sentences: list[Sentence] = []
    prev_end = 0
    for boundary_start, _boundary_end in [*breaks, (len(text), len(text))]:
        segment = text[prev_end:boundary_start]
        segment_start = prev_end
        prev_end = _boundary_end
        stripped = segment.strip()
        if not stripped:
            continue
        lead = len(segment) - len(segment.lstrip())
        start = segment_start + lead
        end = start + len(stripped)
        sentences.append(
            Sentence(
                text=text[start:end],
                start=start,
                end=end,
                index=len(sentences),
            )
        )
    return sentences
