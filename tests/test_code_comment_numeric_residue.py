"""A comment may be stripped only if the numerics in it survive without it.

Vega ruling 72. The code compressor's only compression lever is comment
removal, and it was disabled — in a parameter default, recorded nowhere —
because stripping comments loses the numerics they carry and ``verify()``
correctly rejected the result. Measured before the fix: a four-line sample
compressed to ratio 0.478 and failed ``numeric_preserved`` with ``missing=1``.
With the lever off, the route returned its input unchanged on every code
fixture while reporting success.

The question "is a numeric in a comment protected?" is malformed, because the
code has two cases and any single answer is wrong for one of them:

    x = base * 1.12   # add 12% margin      restatement
    # upstream rate limit is 500/min        the only copy

So the rule is a test per comment, not a policy about comments.

WHY THE RESIDUE AND NOT "APPEARS ELSEWHERE". If two comments each carry ``500``
and nothing else does, each is redundant *against the other*, and an
"appears elsewhere" test strips both and loses the fact. Judging against the
residue — the source with EVERY comment removed — is order-independent and
cannot be defeated that way. ``test_two_comments_sharing_a_numeric_both_survive``
is that case, and it is the one the ruling exists to get right.

THIS IS NOT A HEURISTIC APPROXIMATING THE CHECK. ``extract_numerics`` returns a
set of literal strings, so the predicate here IS ``numeric_preserved``'s own set
difference, evaluated before the strip instead of after it. The consequence is
visible in ``test_a_restatement_is_kept_because_the_strings_differ``: ``"12"``
is not a member of ``{"1.12"}``, so that comment is KEPT even though a human
reads the fact as preserved. Conservative, contract-exact, and deliberately not
second-guessed by this codec.
"""

from __future__ import annotations

import pytest

from tekmyra import verifier
from tekmyra.codecs.code_codec import compress_code
from tekmyra.codecs.selector import apply_codec
from tekmyra.contracts import PolicyTier, RouteClass, SourceChannel
from tekmyra.numeric_fidelity import extract_numerics
from tekmyra.prober import probe_text


def _verify(original: str, result) -> object:
    features = probe_text(original, SourceChannel.API, PolicyTier.STANDARD)
    return verifier.verify(
        original, result, RouteClass.CODE_COMPRESSOR, features, locked_spans=[]
    )


def test_a_comment_with_no_numerics_is_stripped() -> None:
    """The plain win: prose comments carry no facts the contract protects."""
    source = "# leading explanation of the algorithm\ndef total(rows):\n    return rows\n"
    result = compress_code(source, language="python", strip_comments=True)
    assert result.success
    assert "leading explanation" not in (result.output or "")
    assert len(result.output or "") < len(source), "nothing was compressed"


def test_a_comment_holding_the_only_copy_of_a_numeric_is_kept() -> None:
    """The case that made stripping unsafe in the first place."""
    source = (
        "# upstream rate limit is 500/min per RFC 6585\n"
        "def call(client):\n"
        '    return client.get("/v1/items")\n'
    )
    result = compress_code(source, language="python", strip_comments=True)
    assert result.success
    assert "500" in (result.output or ""), "the only copy of 500 was deleted"
    assert "6585" in (result.output or "")


def test_two_comments_sharing_a_numeric_both_survive() -> None:
    """Order-independence — the reason the test is against the RESIDUE.

    Each comment is individually redundant against the other. An
    "appears elsewhere" rule strips whichever it examines first, then finds the
    second also redundant, and the fact is gone. Judging both against the
    source with EVERY comment removed makes that impossible.
    """
    source = "# retry budget is 500\n# and the ceiling is also 500\ndef go():\n    return None\n"
    result = compress_code(source, language="python", strip_comments=True)
    assert result.success
    assert (result.output or "").count("500") >= 1, (
        "both comments were stripped because each looked redundant against the "
        "other — this is exactly what judging against the residue prevents"
    )
    assert _verify(source, result).passed


def test_a_restatement_is_kept_because_the_strings_differ() -> None:
    """The ruling's own worked example, run through the ruling's mechanism.

    A reader sees `# add 12% margin` beside `* 1.12` and calls the fact
    preserved. ``numeric_preserved`` does not: it compares literal strings, and
    ``"12"`` is not a member of ``{"1.12"}``. The codec therefore KEEPS this
    comment. That is the check's own semantics, not a limitation of the strip —
    if this ever changes, it must change in ``numeric_fidelity`` where the
    contract lives, and both sides move together.
    """
    source = "def total(rows):\n    # add 12% margin\n    return rows * 1.12\n"
    assert extract_numerics("# add 12% margin") == {"12"}
    assert "12" not in extract_numerics("return rows * 1.12")

    result = compress_code(source, language="python", strip_comments=True)
    assert "12%" in (result.output or ""), (
        "the comment was stripped, so the codec is applying looser semantics "
        "than the check it must satisfy"
    )


def test_the_strip_is_verified_by_the_real_verifier_not_by_this_test() -> None:
    """The predicate is the check's own, so the check must confirm it."""
    source = (
        "# tuning notes, no figures here\n"
        "def total(rows):\n"
        "    # margin is 1.12 exactly\n"
        "    return sum(rows) * 1.12\n"
    )
    result = compress_code(source, language="python", strip_comments=True)
    assert result.success
    assert len(result.output or "") < len(source), "nothing stripped, so nothing proved"
    assert _verify(source, result).passed, "the strip produced output the verifier rejects"


def test_c_style_sources_get_the_same_test() -> None:
    """The rule is the contract's, not Python's."""
    source = "// timeout is 30 seconds\nfunction go() { return 1; }\n"
    result = compress_code(source, language="go", strip_comments=True)
    assert result.success
    # Quote-free, so stripping is permitted at all; 30 appears only in the
    # comment, so it must be retained regardless.
    assert "30" in (result.output or "")


def test_string_literals_still_fail_closed() -> None:
    """The pre-existing safety must not be traded away for the new capability."""
    source = '// note\nfunction go() { return "text with // inside"; }\n'
    result = compress_code(source, language="javascript", strip_comments=True)
    assert result.success is False
    assert "string_literal_detected" in (result.error or "")


def test_the_production_route_now_compresses_code() -> None:
    """The wiring, not just the codec — the defect was at the call site.

    ``_code_route`` hardcoded ``strip_comments=False``, so every test of the
    codec could pass while the route did nothing. This asserts the route.
    """
    source = (
        "# module docstring substitute, purely explanatory\n"
        "def total(rows):\n"
        "    # sums the rows\n"
        "    return sum(rows)\n"
    )
    features = probe_text(source, SourceChannel.API, PolicyTier.STANDARD)
    result = apply_codec(source, RouteClass.CODE_COMPRESSOR, features)
    assert result.success
    assert len(result.output or "") < len(source), (
        "the code route still returns its input unchanged — the codec was fixed "
        "but the call site was not"
    )


@pytest.mark.parametrize(
    "language", ["javascript", "typescript", "java", "go", "rust", "python"]
)
def test_no_language_loses_a_numeric_it_alone_carries(language: str) -> None:
    """Swept across every supported language, since the rule is contract-wide."""
    marker = "// " if language != "python" else "# "
    source = f"{marker}threshold is 4271\nx = 1\n"
    result = compress_code(source, language=language, strip_comments=True)
    if not result.success:
        return  # fail-closed is an acceptable outcome; losing the number is not
    assert "4271" in (result.output or ""), f"{language} dropped its only numeric"
