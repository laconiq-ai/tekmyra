"""Token counting — the unit the product is actually sold in.

Characters are not that unit, and the two can disagree in SIGN. The anchor-bloat
defect is the worked example: the emitted artifact was shorter than the input in
characters while costing 7.06% MORE tokens, because a handle that reads as a few
characters can tokenize into dozens. A compression figure reported in characters
would have called that a win.

REFUSES RATHER THAN APPROXIMATES. If the tokenizer is unavailable this raises,
and every caller records the refusal instead of substituting a character count or
a words-times-1.3 estimate. A number in the wrong unit that looks like the right
one is worse than no number: it is the same defect class as a ratio measuring the
codec instead of the emitted artifact, which is how the corpus finding started.

A leaf by construction: it imports nothing from this package, so either half of
the open-core seam may use it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# Pinned. The encoding decides the number, so it belongs in provenance beside the
# figure it produced — a token delta measured under a different encoding is not
# comparable to one measured under this, and a silently changed default would
# move published numbers with no code change to point at.
ENCODING_NAME = "cl100k_base"


class TokenizerUnavailable(RuntimeError):
    """The pinned tokenizer could not be loaded, so no token figure exists."""


@lru_cache(maxsize=1)
def _encoder() -> Any:
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TokenizerUnavailable(
            f"tiktoken is not installed, so tokens cannot be counted under "
            f"{ENCODING_NAME}. Install the 'measure' extra. Refusing to "
            "substitute a character or word-based estimate: a token figure in "
            "the wrong unit reads as a measurement."
        ) from exc
    try:
        return tiktoken.get_encoding(ENCODING_NAME)
    except Exception as exc:  # pragma: no cover - network/cache dependent
        raise TokenizerUnavailable(
            f"tiktoken could not load the {ENCODING_NAME} encoding: {exc}"
        ) from exc


def tokenizer_available() -> tuple[bool, str | None]:
    """Return (available, reason_if_not) without raising."""
    try:
        _encoder()
    except TokenizerUnavailable as exc:
        return False, str(exc)
    return True, None


def count_tokens(text: str) -> int:
    """Return the token count of ``text`` under the pinned encoding."""
    return len(_encoder().encode(text, disallowed_special=()))
