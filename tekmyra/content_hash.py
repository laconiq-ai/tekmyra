"""Salted content hashing, shared by the open and commercial halves.

Extracted from ``anchor_store`` because it carries no anchor semantics: it is a
sha256 of text with an optional salt, and both ``protected_spans`` (open) and
the anchor store (commercial) need it. Leaving it in ``anchor_store`` made
``protected_spans`` — a module the open core requires — transitively depend on
the anchorization implementation for three lines of hashing.

This is the same leaf-contract pattern as ``numeric_fidelity``: when two halves
of a seam need one definition, the definition moves to a leaf that neither owns,
rather than one half importing the other.
"""

from __future__ import annotations

import hashlib


def content_hash(text: str, salt: str | None = None) -> str:
    """Return the sha256 of ``text``, optionally salted.

    The salt is mixed in so that hashes are not portable across deployments.
    Callers that are not deriving a deployment-scoped identity pass no salt.
    """
    payload = text if salt is None else f"{salt}:{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
