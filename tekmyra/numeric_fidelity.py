"""The numeric-preservation contract, shared by producers and the verifier.

``verify()`` requires every numeric outside a locked span to survive
compression. A codec therefore has to honour the same definition of "numeric"
that the check applies — and a codec that re-expressed that definition with its
own regex would drift from the check it must satisfy, which is the defect class
this module exists to remove.

It deliberately has NO imports from the rest of the package. The obvious
alternative — importing the helpers straight from ``verifier`` — pulls the
verifier and, transitively, ``anchor_store`` into the compressor's import
surface, which the training-provenance contract measures and which puts an
anchorization dependency inside a component intended for the open core. A leaf
module keeps one definition without buying that coupling.
"""

from __future__ import annotations

import re
from typing import Any

# Signed integers and decimals, including scientific notation. Note this does
# not span a trailing '%': "12%" yields "12", and "2026-08-15" yields
# {"2026", "-08", "-15"}. Producers must match this exactly, quirks included.
NUMERIC_RE = re.compile(r"-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")


def extract_numerics(text: str) -> set[str]:
    """Return the set of numeric literals appearing in ``text``."""
    return set(NUMERIC_RE.findall(text))


def extract_numerics_in_spans(text: str, spans: list[dict[str, Any]]) -> set[str]:
    """Return numeric values that fall inside locked spans in ``text``.

    MATCHED IN PLACE, NOT ON ISOLATED SLICES — the two disagree at span
    boundaries, and the disagreement is a false-refusal factory.

    ``NUMERIC_RE`` begins with an optional ``-`` that can consume a character
    immediately BEFORE the span. In "INC-2026-07-23-001" the date detector locks
    the slice [2026-07-23], which in isolation yields {"2026", "-07", "-23"};
    the same digits scanned in the full document yield "-2026", because the
    regex reaches back into "INC-". The exempt set then lacked "-2026" by one
    character, the check required a value whose surrounding text had been
    correctly masked, and three long_context_v1 fixtures were refused for it.

    Scanning the original once and exempting any match that OVERLAPS a locked
    span removes the boundary entirely: the numeric is judged where it actually
    occurs. The slice fallback stays for spans that carry no offsets.
    """
    bounds = [
        (int(span["start"]), int(span["end"]))
        for span in spans
        if span.get("start") is not None and span.get("end") is not None
    ]
    numerics: set[str] = set()

    if bounds:
        bounds.sort()
        for match in NUMERIC_RE.finditer(text):
            start, end = match.start(), match.end()
            for span_start, span_end in bounds:
                if span_start >= end:
                    break
                if span_end > start:
                    numerics.add(match.group())
                    break

    # Spans without offsets cannot be located, so fall back to their declared
    # text. Applied only to those spans; the located ones are already handled.
    for span in spans:
        if span.get("start") is None or span.get("end") is None:
            numerics.update(extract_numerics(span.get("text") or ""))
    return numerics
