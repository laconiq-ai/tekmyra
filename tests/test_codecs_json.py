"""Tests for the deterministic JSON codec."""

from __future__ import annotations

import json


from tekmyra.codecs.json_codec import compress_json


def test_json_minifies_pretty_printed():
    original = '{\n  "a": 1,\n  "b": [\n    2,\n    3\n  ]\n}\n'
    result = compress_json(original)
    assert result.success
    expected = '{"a":1,"b":[2,3]}'
    assert result.output == expected
    assert result.ratio < 1.0


def test_json_invalid_fails_closed():
    original = '{"a": 1, "b":}'
    result = compress_json(original)
    assert not result.success
    assert result.output == original
    assert "json_parse_failed" in (result.error or "")


def test_json_preserves_numeric_precision():
    original = '{"pi": 3.141592653589793, "big": 123456789012345}'
    result = compress_json(original)
    assert result.success
    parsed = json.loads(result.output)
    assert parsed["pi"] == 3.141592653589793
    assert parsed["big"] == 123456789012345


def test_json_passthrough_empty():
    result = compress_json("  ")
    assert result.success
    assert result.output == ""
    assert result.ratio == 1.0


def test_json_preserves_unicode():
    original = '{"emoji": "🚀", "text": "héllo"}'
    result = compress_json(original)
    assert result.success
    parsed = json.loads(result.output)
    assert parsed["emoji"] == "🚀"
    assert parsed["text"] == "héllo"
