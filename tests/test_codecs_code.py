"""Tests for the deterministic code codec."""

from __future__ import annotations

from tekmyra.codecs.code_codec import compress_code


def test_code_passthrough_by_default():
    text = "# comment\nx = 1\n"
    result = compress_code(text, language="python", strip_comments=False)
    assert result.success
    assert result.output == text


def test_python_comments_stripped():
    text = "# comment\nx = 1  # inline\n"
    result = compress_code(text, language="python", strip_comments=True)
    assert result.success
    assert "# comment" not in result.output
    assert "x = 1" in result.output


def test_python_hash_inside_string_preserved():
    text = 'x = "# not a comment"  # real comment\n'
    result = compress_code(text, language="python", strip_comments=True)
    assert result.success
    assert '# not a comment' in result.output
    assert 'real comment' not in result.output


def test_python_comments_preserve_blank_lines():
    text = "# a\n# b\nx = 1\n"
    result = compress_code(text, language="python", strip_comments=True)
    assert result.success
    assert len(result.output.splitlines()) == len(text.splitlines())
    assert "x = 1" in result.output


def test_c_style_with_string_literal_fails_closed():
    text = 'const msg = "hello"; // greeting\n'
    result = compress_code(text, language="javascript", strip_comments=True)
    assert not result.success
    assert result.output == text
    assert result.ratio == 1.0


def test_c_style_without_strings_strips_comments():
    text = "const x = 1; // greeting\nconst y = 2;\n"
    result = compress_code(text, language="javascript", strip_comments=True)
    assert result.success
    assert "// greeting" not in result.output
    assert "const x = 1" in result.output


def test_c_style_block_comment_fails_closed_with_strings():
    text = 'String s = "hello"; /* old */\n'
    result = compress_code(text, language="java", strip_comments=True)
    assert not result.success
    assert result.output == text


def test_unknown_language_passthrough():
    text = "// unknown language\n"
    result = compress_code(text, language="brainfuck", strip_comments=True)
    assert result.success
    assert result.output == text


def test_empty_code():
    result = compress_code("")
    assert result.success
    assert result.output == ""
