"""Tests for the deterministic parser/prober."""

from __future__ import annotations


from tekmyra.contracts import PolicyTier, SourceChannel
from tekmyra.prober import probe_text


def test_probe_json():
    text = '{"id": 1, "items": [{"name": "a"}, {"name": "b"}]}'
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    assert features.json_parse_success is True
    assert features.json_depth == 3
    assert features.token_len > 0
    assert features.char_entropy > 0


def test_probe_code_python():
    text = "def hello():\n    return 'world'\n"
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    assert features.code_language == "python"
    assert features.ast_parse_success is True


def test_probe_diff():
    text = "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n- old\n+ new\n"
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    assert features.diff_markers >= 2


def test_probe_log():
    text = (
        "2026-07-06T00:00:01Z INFO host-01 request-1001 Starting job\n"
        "2026-07-06T00:00:02Z INFO host-01 request-1001 Done\n"
    )
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    assert features.log_timestamp_density >= 0.5


def test_probe_stacktrace():
    text = (
        "Traceback (most recent call last):\n"
        '  File "/app/db.py", line 91, in query\n'
        "    cursor.execute('SELECT *')\n"
        "DBTimeoutError: query exceeded 5000ms\n"
    )
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    assert features.stacktrace_hits >= 2


def test_probe_pii():
    text = "Customer SSN: 123-45-6789, phone: 555-019-1234"
    features = probe_text(text, SourceChannel.CHAT, PolicyTier.STANDARD)
    assert features.pii_entity_count > 0
    assert "ssn" in features.pii_entity_types or "phone" in features.pii_entity_types


def test_probe_legal_keywords():
    text = "This contract is binding and subject to litigation and attorney review."
    features = probe_text(text, SourceChannel.EMAIL, PolicyTier.STANDARD)
    assert features.legal_keyword_count >= 3


def test_probe_keywords_do_not_match_inside_larger_words():
    text = "The secretary issued a new tokenization guide."
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    assert features.security_keyword_count == 0


def test_probe_empty_fail_closed():
    features = probe_text("", SourceChannel.API, PolicyTier.STANDARD)
    assert features.char_len == 0
    assert features.pii_entity_count == 0


def test_probe_table_density():
    text = "| col1 | col2 |\n|------|------|\n| a    | b    |\n"
    features = probe_text(text, SourceChannel.UPLOAD, PolicyTier.STANDARD)
    assert features.table_density > 0
