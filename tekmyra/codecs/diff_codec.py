"""Deterministic diff compressor.

Trims context lines around hunks while preserving file paths, hunk headers,
and every changed (``+`` / ``-``) line.
"""

from __future__ import annotations

import re

from tekmyra.codecs.base import CodecResult


_HUNK_RE = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@")
_DIFF_HEADER_RE = re.compile(r"^(---|\+\+\+|diff\s+--git|index\s+\S+)")


def compress_diff(text: str, context_keep: int = 2) -> CodecResult:
    """Compress a unified diff by trimming context lines around hunks.

    Args:
        text: Unified diff text.
        context_keep: Number of unchanged context lines to retain on each side
            of a changed line.  ``0`` keeps only hunk headers and changed lines.

    Returns:
        CodecResult with the trimmed diff.
    """
    lines = text.splitlines()
    if not lines:
        return CodecResult(success=True, output="", ratio=1.0)

    # If it doesn't look like a diff, passthrough rather than risk corruption.
    marker_hits = sum(1 for line in lines if _HUNK_RE.match(line) or _DIFF_HEADER_RE.match(line))
    if marker_hits < 2:
        return CodecResult(success=True, output=text, ratio=1.0)

    out_lines: list[str] = []
    context_buffer: list[str] = []
    in_hunk = False

    def flush_context() -> None:
        if context_buffer:
            keep = context_buffer[:context_keep]
            if len(context_buffer) > len(keep):
                keep.append("(... context omitted ...)")
            out_lines.extend(keep)
        # Do not reset context_buffer here; callers manage it.

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if _DIFF_HEADER_RE.match(line):
            in_hunk = False
            flush_context()
            context_buffer = []
            out_lines.append(line)
            continue

        if _HUNK_RE.match(line):
            in_hunk = True
            flush_context()
            context_buffer = []
            out_lines.append(line)
            continue

        if not in_hunk:
            out_lines.append(line)
            continue

        stripped = line.lstrip()
        if stripped.startswith("+") or stripped.startswith("-"):
            flush_context()
            context_buffer = []
            out_lines.append(line)
        elif stripped.startswith(" "):
            context_buffer.append(line)
        else:
            # Metadata inside hunk (e.g., ``\ No newline at end of file``).
            flush_context()
            context_buffer = []
            out_lines.append(line)

    # Trailing context may remain.
    if context_buffer:
        keep = context_buffer[:context_keep]
        if len(context_buffer) > len(keep):
            keep.append("(... context omitted ...)")
        out_lines.extend(keep)

    compressed = "\n".join(out_lines)
    ratio = max(1, len(compressed)) / max(1, len(text))
    return CodecResult(success=True, output=compressed, ratio=ratio)
