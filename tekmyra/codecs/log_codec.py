"""Deterministic log compressor.

Folds repeated lines, deduplicates timestamp/request-id templates, and
preserves severity, timestamps, and correlation IDs.
"""

from __future__ import annotations

import re
from collections import Counter

from tekmyra.codecs.base import CodecResult


# ISO-8601-like timestamps and severity markers.
_TS_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_SEVERITY_RE = re.compile(r"\b(?:INFO|WARN|WARNING|ERROR|FATAL|DEBUG|TRACE|CRITICAL)\b", re.I)
_CORR_ID_RE = re.compile(
    r"\b(?:request|trace|span|correlation|req|txn)[-_]?(?:id)?[:=]?\s*\S+\b",
    re.I,
)


def _template_key(line: str) -> str:
    """Normalize a log line so repeated templates collapse together.

    Timestamps and correlation IDs are normalized, but severity is preserved
    so that ``INFO request succeeded`` and ``ERROR request succeeded`` are
    not folded into each other.
    """
    key = _TS_RE.sub("<TS>", line)
    key = _CORR_ID_RE.sub("<ID>", key)
    return key.rstrip()


def compress_logs(
    text: str,
    max_repetition_fold: int = 3,
    context_lines: int = 2,
) -> CodecResult:
    """Compress timestamped logs by folding repeated templates.

    Args:
        text: Raw log text.
        max_repetition_fold: Minimum identical-template occurrences required
            before folding into a summary line.
        context_lines: Number of unique lines to keep around each folded group.

    Returns:
        CodecResult with folded output.
    """
    lines = text.splitlines()
    if not lines:
        return CodecResult(success=True, output="", ratio=1.0)

    # Short or low-density logs: passthrough is safer than aggressive folding.
    if len(lines) <= 4:
        return CodecResult(success=True, output=text, ratio=1.0)

    template_counts: Counter[str] = Counter(_template_key(line) for line in lines)
    frequent = {t for t, c in template_counts.items() if c >= max_repetition_fold and t}

    if not frequent:
        return CodecResult(success=True, output=text, ratio=1.0)

    out_lines: list[str] = []
    current_group: list[str] = []
    current_template: str | None = None

    def flush_group() -> None:
        if not current_group or current_template is None:
            return
        if current_template not in frequent:
            out_lines.extend(current_group)
            return

        # Keep context_lines occurrences on each side of a folded group and
        # summarize the identical lines in the middle.  When context_lines is
        # zero we still preserve one representative line so that severity,
        # timestamps, and correlation IDs remain visible.
        if context_lines > 0:
            head = current_group[:context_lines]
            tail = current_group[-context_lines:]
        else:
            head = [current_group[0]]
            tail = []
        omitted = len(current_group) - len(head) - len(tail)
        if omitted <= 0:
            out_lines.extend(current_group)
            return

        out_lines.extend(head)
        out_lines.append(f"(... {omitted} identical log lines ...)")
        out_lines.extend(tail)

    for line in lines:
        tpl = _template_key(line)
        if tpl != current_template:
            flush_group()
            current_group = [line]
            current_template = tpl
        else:
            current_group.append(line)
    flush_group()

    compressed = "\n".join(out_lines)
    ratio = max(1, len(compressed)) / max(1, len(text))
    return CodecResult(success=True, output=compressed, ratio=ratio)
