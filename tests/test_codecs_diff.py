"""Tests for the deterministic diff codec."""

from __future__ import annotations

from tekmyra.codecs.diff_codec import compress_diff


def test_diff_trims_context():
    text = (
        "--- a/file.py\n"
        "+++ b/file.py\n"
        "@@ -1,5 +1,5 @@\n"
        " unchanged 1\n"
        " unchanged 2\n"
        "-old line\n"
        "+new line\n"
        " unchanged 3\n"
        " unchanged 4\n"
    )
    result = compress_diff(text, context_keep=1)
    assert result.success
    assert result.ratio < 1.0
    assert "--- a/file.py" in result.output
    assert "+new line" in result.output
    assert "unchanged 1" in result.output


def test_non_diff_passthrough():
    text = "just some prose\nwith no diff markers\n"
    result = compress_diff(text)
    assert result.success
    assert result.output == text


def test_empty_diff():
    result = compress_diff("")
    assert result.success
    assert result.output == ""
