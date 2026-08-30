"""Tests for the deterministic log codec."""

from __future__ import annotations

from tekmyra.codecs.log_codec import compress_logs


def test_short_log_passthrough():
    text = "2024-01-01T00:00:00Z INFO hello\n"
    result = compress_logs(text)
    assert result.success
    assert result.output == text
    assert result.ratio == 1.0


def test_repeated_log_lines_folded():
    lines = ["2024-01-01T00:00:00Z INFO request succeeded"] * 5
    text = "\n".join(lines)
    result = compress_logs(text, max_repetition_fold=3, context_lines=0)
    assert result.success
    assert result.ratio < 1.0
    # One representative line is kept, and the remaining four are summarized.
    assert result.output.startswith("2024-01-01T00:00:00Z INFO request succeeded")
    assert "(... 4 identical log lines ...)" in result.output


def test_distinct_templates_preserved():
    text = (
        "2024-01-01T00:00:00Z INFO start\n"
        "2024-01-01T00:00:01Z ERROR fail\n"
        "2024-01-01T00:00:02Z INFO done\n"
    )
    result = compress_logs(text)
    assert result.success
    assert result.ratio == 1.0
    assert "ERROR fail" in result.output


def test_severity_change_prevents_folding():
    lines = ["2024-01-01T00:00:00Z INFO request succeeded"] * 3
    lines.append("2024-01-01T00:00:03Z ERROR request succeeded")
    lines.extend(["2024-01-01T00:00:04Z INFO request succeeded"] * 3)
    text = "\n".join(lines)
    result = compress_logs(text, max_repetition_fold=3, context_lines=0)
    assert result.success
    assert "ERROR request succeeded" in result.output
    # The ERROR line must remain separate from the INFO groups.
    out_lines = result.output.splitlines()
    assert out_lines.count("2024-01-01T00:00:03Z ERROR request succeeded") == 1


def test_context_lines_kept_around_fold():
    lines = ["2024-01-01T00:00:00Z INFO request succeeded"] * 6
    text = "\n".join(lines)
    result = compress_logs(text, max_repetition_fold=3, context_lines=2)
    assert result.success
    assert result.ratio < 1.0
    # First two and last two instances should be preserved around the summary.
    assert result.output.startswith("2024-01-01T00:00:00Z INFO request succeeded")
    assert "(... 2 identical log lines ...)" in result.output


def test_empty_log():
    result = compress_logs("")
    assert result.success
    assert result.output == ""
