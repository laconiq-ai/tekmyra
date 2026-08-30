"""Deterministic structure codecs for the TekMyra context-compression runtime."""

from __future__ import annotations

from tekmyra.codecs.base import CodecResult
from tekmyra.codecs.code_codec import compress_code
from tekmyra.codecs.diff_codec import compress_diff
from tekmyra.codecs.json_codec import compress_json
from tekmyra.codecs.log_codec import compress_logs
from tekmyra.codecs.selector import apply_codec, select_codec
from tekmyra.codecs.stacktrace_codec import compress_stacktrace

__all__ = [
    "CodecResult",
    "apply_codec",
    "compress_code",
    "compress_diff",
    "compress_json",
    "compress_logs",
    "compress_stacktrace",
    "select_codec",
]
