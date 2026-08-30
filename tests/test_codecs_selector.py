"""Tests for the codec selector."""

from __future__ import annotations

from tekmyra.codecs import apply_codec
from tekmyra.codecs.selector import select_codec
from tekmyra.contracts import FeatureVector, PolicyTier, RouteClass, SourceChannel


def _features(stacktrace_hits: int = 0) -> FeatureVector:
    return FeatureVector(
        token_len=10,
        char_len=50,
        line_count=1,
        byte_len=50,
        char_entropy=3.0,
        json_parse_success=False,
        json_depth=0,
        markdown_density=0.0,
        code_language=None,
        ast_parse_success=False,
        diff_markers=0,
        log_timestamp_density=0.0,
        log_repetition_ratio=0.0,
        stacktrace_hits=stacktrace_hits,
        table_density=0.0,
        url_density=0.0,
        pii_entity_count=0,
        pii_entity_types=[],
        pii_max_confidence=0.0,
        legal_keyword_count=0,
        security_keyword_count=0,
        governance_keyword_count=0,
        financial_amount_count=0,
        deadline_date_count=0,
        embedded_text_ratio=0.0,
        ocr_confidence=0.0,
        image_hint="none",
        source_channel=SourceChannel.BATCH,
        policy_tier=PolicyTier.STANDARD,
    )


def test_json_route_selected():
    codec = select_codec(RouteClass.DETERMINISTIC_JSON_COMPRESSOR)
    assert codec is not None


def test_unknown_route_passthrough():
    result = apply_codec("hello", RouteClass.PROSE_TOKEN_COMPRESSOR, _features())
    assert result.success
    assert result.output == "hello"


def test_json_apply_compresses():
    text = '{\n  "a": 1\n}'
    result = apply_codec(text, RouteClass.DETERMINISTIC_JSON_COMPRESSOR, _features())
    assert result.success
    assert result.output == '{"a":1}'


def test_log_route_delegates_to_stacktrace_codec():
    text = (
        "Traceback (most recent call last):\n"
        '  File "/usr/local/lib/python3.11/site-packages/flask/app.py", line 1000\n'
        '  File "/usr/local/lib/python3.11/site-packages/flask/app.py", line 1001\n'
        '  File "/usr/local/lib/python3.11/site-packages/flask/app.py", line 1002\n'
        '  File "/app/main.py", line 42\n'
        "ValueError: bad\n"
    )
    result = apply_codec(text, RouteClass.LOG_COMPRESSOR, _features(stacktrace_hits=5))
    assert result.success
    assert result.ratio < 1.0
    assert "(... library frames omitted ...)" in result.output
    assert "/app/main.py" in result.output
    assert "ValueError: bad" in result.output
