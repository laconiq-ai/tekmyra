"""Shared forbidden release-ready phrase list and regex.

Both ``scripts/wp7_fast_track_decision_record.py`` and
``tests/test_fast_track_consolidation.py`` import from this module so the
canonical list and regex stay in sync.
"""

from __future__ import annotations

import re

FORBIDDEN_RELEASE_PHRASES = [
    "green",
    "ready for release",
    "production-ready",
    "release consideration",
]

FORBIDDEN_RELEASE_RE = re.compile(
    r"\bgreen\b|ready for release|release[-\s]?ready|production[-\s]?ready|release consideration",
    re.IGNORECASE,
)


def contains_forbidden_release_language(text: str) -> bool:
    """Return True if *text* contains any forbidden release-ready phrase."""
    return FORBIDDEN_RELEASE_RE.search(text) is not None
