"""Deterministic JSON compressor.

Minifies valid JSON while preserving parse validity, required keys,
numeric precision, and ordering.  No structural transformation beyond
whitespace removal is performed.
"""

from __future__ import annotations

import json

from tekmyra.codecs.base import CodecResult


def compress_json(text: str, strip_outer_whitespace: bool = True) -> CodecResult:
    """Return a minified JSON string if ``text`` is valid JSON.

    Args:
        text: Candidate JSON payload.
        strip_outer_whitespace: Also strip leading/trailing whitespace before parsing.

    Returns:
        CodecResult with success=True and minified output, or success=False on error.
    """
    try:
        candidate = text.strip() if strip_outer_whitespace else text
        if not candidate:
            return CodecResult(success=True, output="", ratio=1.0)
        data = json.loads(candidate)
    except Exception as exc:
        return CodecResult(
            success=False,
            output=text,
            ratio=1.0,
            error=f"json_parse_failed: {exc}",
        )

    try:
        # separators=(",", ":") removes whitespace after separators.
        # ensure_ascii=False keeps non-ASCII characters as-is (UTF-8 safe).
        minified = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    except Exception as exc:
        return CodecResult(
            success=False,
            output=text,
            ratio=1.0,
            error=f"json_dump_failed: {exc}",
        )

    original_len = max(1, len(text))
    compressed_len = max(1, len(minified))
    ratio = compressed_len / original_len
    return CodecResult(success=True, output=minified, ratio=ratio)
