"""Deterministic stack-trace compressor.

Folds library frames while preserving the exception class/message and
application frames (file path + line number).
"""

from __future__ import annotations

import re

from tekmyra.codecs.base import CodecResult


# Frame lines in Python / Java / JS style traces.
_FRAME_RE = re.compile(
    r"^\s*(?:File\s+\"([^\"]+)\",\s+line\s+(\d+)|at\s+(\S+)\s+\(([^)]+)\)|(\S+)\.(\S+)\(([^)]+)\))",
    re.M,
)
_EXCEPTION_RE = re.compile(r"^[\w.]*(?:Error|Exception|Failure|Fault):\s*", re.I)


_LIBRARY_FRAME_MARKERS = (
    "/usr/lib/",
    "/usr/local/lib/",
    "/lib/python",
    "/site-packages/",
    "/node_modules/",
    "/.venv/",
    "/venv/",
    "/env/lib/",
    "/dist-packages/",
)


def _is_library_frame(line: str) -> bool:
    """Return True if the frame clearly belongs to a third-party library."""
    lower = line.lower()
    return any(marker in lower for marker in _LIBRARY_FRAME_MARKERS)


def compress_stacktrace(text: str, library_fold_threshold: int = 3) -> CodecResult:
    """Compress a stack trace by folding consecutive library frames.

    Args:
        text: Raw stack trace text.
        library_fold_threshold: Minimum consecutive library frames required
            before folding them into a summary.

    Returns:
        CodecResult with the compressed trace.
    """
    lines = text.splitlines()
    if not lines:
        return CodecResult(success=True, output="", ratio=1.0)

    out_lines: list[str] = []
    library_buffer: list[str] = []
    app_buffer: list[str] = []

    def flush_library() -> None:
        nonlocal library_buffer
        if not library_buffer:
            return
        if len(library_buffer) >= library_fold_threshold:
            out_lines.append("(... library frames omitted ...)")
        else:
            out_lines.extend(library_buffer)
        library_buffer = []

    def flush_app() -> None:
        nonlocal app_buffer
        out_lines.extend(app_buffer)
        app_buffer = []

    for line in lines:
        # Always preserve the exception message line(s).
        if _EXCEPTION_RE.search(line) or line.strip().startswith("Traceback") or line.strip().startswith("Caused by"):
            flush_library()
            flush_app()
            out_lines.append(line)
            continue

        is_frame = bool(_FRAME_RE.search(line))
        if not is_frame:
            flush_library()
            flush_app()
            out_lines.append(line)
            continue

        if _is_library_frame(line):
            flush_app()
            library_buffer.append(line)
        else:
            flush_library()
            app_buffer.append(line)

    flush_library()
    flush_app()

    compressed = "\n".join(out_lines)
    ratio = max(1, len(compressed)) / max(1, len(text))
    return CodecResult(success=True, output=compressed, ratio=ratio)
