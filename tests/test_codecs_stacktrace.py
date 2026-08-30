"""Tests for the deterministic stack-trace codec."""

from __future__ import annotations

from tekmyra.codecs.stacktrace_codec import compress_stacktrace


def test_library_frames_folded():
    text = (
        "Traceback (most recent call last):\n"
        '  File "/app/main.py", line 42, in handler\n'
        "    raise ValueError('bad')\n"
        '  File "/usr/local/lib/python3.11/site-packages/flask/app.py", line 1000, in run\n'
        '  File "/usr/local/lib/python3.11/site-packages/flask/app.py", line 1001, in dispatch\n'
        '  File "/usr/local/lib/python3.11/site-packages/flask/app.py", line 1002, in full_dispatch\n'
        "ValueError: bad\n"
    )
    result = compress_stacktrace(text)
    assert result.success
    assert result.ratio < 1.0
    assert "(... library frames omitted ...)" in result.output
    assert "/app/main.py" in result.output
    assert "ValueError: bad" in result.output


def test_app_frames_preserved():
    text = (
        "Traceback (most recent call last):\n"
        '  File "/app/main.py", line 42, in handler\n'
        '  File "/app/service.py", line 10, in helper\n'
        "ValueError: bad\n"
    )
    result = compress_stacktrace(text)
    assert result.success
    assert "main.py" in result.output
    assert "service.py" in result.output


def test_no_trace_passthrough():
    text = "Just a plain message\n"
    result = compress_stacktrace(text)
    assert result.success
    assert result.output.strip() == text.strip()
