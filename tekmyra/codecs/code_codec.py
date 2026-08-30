"""Deterministic code compressor.

When ``strip_comments`` is enabled and the language is supported, comments are
removed while preserving line numbers and string/character literals — but only
those comments whose removal cannot lose a fact.

THE NUMERIC RULE (Vega ruling 72). ``verify()`` enforces ``numeric_preserved``:
every numeric literal outside a locked span must survive compression. Comments
carry numerics, so stripping them indiscriminately fails that check — measured,
a four-line sample compressed to ratio 0.478 and was rejected with
``missing=1``. The question "is a numeric in a comment protected?" has no single
answer, because the code has two cases:

    x = base * 1.12   # add 12% margin      <- the 12 restates 1.12; the
                                               artifact still carries the fact
    # upstream rate limit is 500/min        <- the 500 exists nowhere else;
                                               deleting the comment deletes it

So the rule is not a policy about comments but a test applied to each one:
**a comment may be stripped iff every numeric literal in it also appears in the
text that will remain.**

Judged against the RESIDUE, not "appears somewhere else" — and that distinction
is load-bearing. If two comments each carry ``500`` and nothing else does, each
is individually redundant against the other, and an "appears elsewhere" test
would strip both and lose it. The residue here is the MAXIMAL strip, every
comment removed, so any numeric found in it is guaranteed to survive whatever
we later decide to keep. That makes the test order-independent.

This is not a heuristic approximating ``numeric_preserved``. ``extract_numerics``
returns a set of literal strings, so ``"12"`` and ``"1.12"`` are distinct
members — which is exactly the mechanism of that ``missing=1``. The predicate
below IS the check's own set-difference, evaluated before the strip instead of
after it. A comment passing it cannot fail the check, by the check's definition.
Nothing about the contract is weakened; the verifier still runs afterwards.
"""

from __future__ import annotations

import re
import tokenize
from collections.abc import Callable
from io import BytesIO

from tekmyra.codecs.base import CodecResult
from tekmyra.numeric_fidelity import extract_numerics


# Very lightweight per-language comment removal.
_C_STYLE_COMMENT_RE: re.Pattern[str] = re.compile(r"//.*$|/\*[\s\S]*?\*/", re.M)

_C_STYLE_LANGUAGES: frozenset[str] = frozenset(
    {"javascript", "typescript", "java", "go", "rust"}
)

# Quote characters that indicate string/character/template literals in the
# supported C-style languages.  We fail closed: if comments are present and
# the source contains any of these, we refuse to strip comments rather than
# risk corrupting string or regex literals.
_STRING_LITERAL_HINTS: frozenset[str] = frozenset({"'", '"', "`"})


def _numerics_survive_without(residue: str) -> Callable[[str], bool]:
    """Ruling 72(d): may this comment go, given what will remain?

    ``residue`` must be the MAXIMAL strip — the source with every comment
    removed. Anything numeric found there survives no matter which comments are
    subsequently kept, so the predicate is order-independent and two comments
    sharing a numeric cannot excuse each other.
    """
    surviving = extract_numerics(residue)

    def should_strip(comment: str) -> bool:
        return extract_numerics(comment) <= surviving

    return should_strip


def _python_strip_comments(
    text: str, should_strip: Callable[[str], bool] | None = None
) -> tuple[str, bool]:
    """Strip Python comments using the stdlib tokenizer to protect strings.

    Returns ``(output, validated)``.  ``validated`` is ``False`` when the
    reconstructed source does not preserve the original line count, which
    indicates the token stream could not be safely reconstructed.

    ``should_strip`` decides per comment; ``None`` strips every one, which is
    how the residue for that decision is computed in the first place.
    """
    # THE TOKENIZE LOOP IS INSIDE THE TRY, and it was not before.
    #
    # ``tokenize.tokenize`` raises on any input that is not valid Python —
    # IndentationError, TokenError, SyntaxError — and this loop sat OUTSIDE the
    # guard, so those propagated out of the codec and out of the runtime as an
    # unhandled exception. Not a refusal: a crash.
    #
    # It was unreachable while ``strip_comments`` defaulted to False, because
    # ``compress_code`` returned before ever tokenizing. Enabling the lever for
    # ruling 72 made a latent crash live, and no corpus we own could show it:
    # the first document that ever reached this path with non-Python content
    # was a Congressional Record page in the public-domain corpus, which the
    # router classified as code. English prose is not Python, and it took real
    # text to produce that pairing.
    #
    # A codec must fail closed. ``(text, False)`` is that contract — the caller
    # returns success=False and the runtime falls back to a safer route.
    source = text.encode("utf-8")
    tokens = []
    try:
        for tok in tokenize.tokenize(BytesIO(source).readline):
            if tok.type == tokenize.COMMENT and (
                should_strip is None or should_strip(tok.string)
            ):
                continue
            tokens.append(tok)
        reconstructed_raw = tokenize.untokenize(tokens)
        if isinstance(reconstructed_raw, bytes):
            reconstructed = reconstructed_raw.decode("utf-8")
        else:
            reconstructed = reconstructed_raw
    except Exception:
        return text, False

    # Preserve trailing newline semantics and strip trailing whitespace left
    # by removed inline/whole-line comments.
    ends_with_newline = text.endswith("\n")
    cleaned = "\n".join(line.rstrip() for line in reconstructed.splitlines())
    if ends_with_newline and not cleaned.endswith("\n"):
        cleaned += "\n"

    if len(cleaned.splitlines()) != len(text.splitlines()):
        return text, False
    return cleaned, True


def _has_string_literal_hints(text: str) -> bool:
    """Return True if the text contains characters that likely delimit strings."""
    return any(ch in text for ch in _STRING_LITERAL_HINTS)


def _c_style_strip_comments(
    text: str, language: str, should_strip: Callable[[str], bool] | None = None
) -> tuple[str, bool]:
    """Strip C-style comments, failing closed if string literals are present.

    Returns ``(output, validated)``.  When the source contains quotes or
    backticks and also contains comments, we return the original text and
    ``validated=False`` to avoid silently corrupting string/regex literals.

    ``should_strip`` decides per comment; ``None`` strips every one.
    """
    has_comments = _C_STYLE_COMMENT_RE.search(text) is not None
    if has_comments and _has_string_literal_hints(text):
        return text, False

    def _replace(match: re.Match[str]) -> str:
        comment = match.group(0)
        if should_strip is None or should_strip(comment):
            return ""
        return comment

    cleaned = _C_STYLE_COMMENT_RE.sub(_replace, text)
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    return cleaned, True


def compress_code(
    text: str,
    language: str | None = None,
    strip_comments: bool = False,
) -> CodecResult:
    """Compress code by normalizing whitespace or stripping comments.

    With ``strip_comments=False`` the codec is lossless and compresses
    nothing — it returns its input. When stripping is requested, Python source
    is reconstructed from tokenizer output, C-style-family sources are only
    processed when no string/character/template literal delimiters are present,
    and in both cases a comment is removed only if every numeric literal in it
    also survives in the residue. See the module docstring for why that test,
    and not a policy about comments, is the correct shape.

    Args:
        text: Source code text.
        language: Detected language (e.g., ``python``, ``javascript``).
        strip_comments: If True, remove comments that carry no numeric the
            residue would lose. False returns the input verbatim.

    Returns:
        CodecResult with compressed code.  If stripping cannot be validated
        safely, ``success`` is ``False`` and the original text is returned
        so the runtime can fall back to a safer route.
    """
    if not text:
        return CodecResult(success=True, output="", ratio=1.0)

    if not strip_comments:
        return CodecResult(success=True, output=text, ratio=1.0)

    language = (language or "").lower()
    if language == "python":
        # Pass one computes the residue: every comment gone. Pass two keeps the
        # comments whose numerics would not survive it (ruling 72(d)).
        residue, validated = _python_strip_comments(text)
        if validated:
            compressed, validated = _python_strip_comments(
                text, _numerics_survive_without(residue)
            )
        if not validated:
            return CodecResult(
                success=False,
                output=text,
                ratio=1.0,
                error="python_comment_reconstruction_failed",
            )
    elif language in _C_STYLE_LANGUAGES:
        residue, validated = _c_style_strip_comments(text, language)
        if validated:
            compressed, validated = _c_style_strip_comments(
                text, language, _numerics_survive_without(residue)
            )
        if not validated:
            return CodecResult(
                success=False,
                output=text,
                ratio=1.0,
                error=f"{language}_string_literal_detected_comments_not_stripped",
            )
    else:
        # Unknown language: fail closed and keep original rather than corrupt.
        return CodecResult(
            success=True,
            output=text,
            ratio=1.0,
        )

    ratio = max(1, len(compressed)) / max(1, len(text))
    return CodecResult(success=True, output=compressed, ratio=ratio)
