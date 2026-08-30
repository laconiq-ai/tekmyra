"""The anchor-store contract, and the open build's implementation of it.

THE SEAM. Anchorization is the commercial half; everything that compresses and
verifies is the open half. The compression path touches the anchor store through
exactly two methods — ``create`` and ``get`` — so that is the whole contract, and
this module is its SINGLE definition. The commercial ``AnchorStore`` satisfies it
structurally; :class:`NonResolvingAnchorStore` is the open default. There is
never a second copy of this contract in another repository: two copies of a
safety contract diverge silently, and they diverge on the safety path.

THE OPEN BUILD ANCHORIZES BUT CAN NEVER RESOLVE. That asymmetry is the design,
and both halves of it were measured rather than reasoned:

* ``create`` MUST NOT raise. ``safety_merge`` calls it whenever a protected span
  carries the ``anchorize`` policy — 192 times across the 301-fixture synthetic
  corpus, because ``pii_policy`` and ``citation_policy`` both default to it.
  That call happens BEFORE any verification runs, so raising there converts
  clean verification refusals into unhandled exceptions. A crash is not a
  refusal: it yields no verdict, no preservation report, and nothing a caller
  can act on. Fail-messy is not fail-closed, and it reads as safe because the
  run does stop.

* ``get`` ALWAYS returns ``None``, including for ids this store minted itself.
  This is what makes the open build STRICTER than the commercial one. The open
  half cannot ask "was this dropped protected span recoverable via an anchor?"
  — that question needs the closed store — so it must treat the drop as
  unrecoverable and refuse. Measured on the live PII route: the commercial build
  completes 30 of 30 rows; the open build refuses 4, exactly the rows that route
  to ``PII_REDACTION_BEFORE_COMPRESSION``. The verdicts differ, which is the
  result. Identical verdicts would mean nothing is being enforced.

NAMING IS LOAD-BEARING HERE. An earlier draft was called ``NullAnchorStore``,
and "null" reads as empty-but-functional — which produced, in good faith, an
in-memory store that RESOLVED what it held and was therefore indistinguishable
from the commercial one while being advertised as stricter. No published number
would ever have revealed it, because the benchmark corpus never enters this
branch. The name must describe the one thing withheld.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from tekmyra.contracts import AnchorRecord, RiskClass


@runtime_checkable
class AnchorStoreProtocol(Protocol):
    """The two methods the compression path requires of an anchor store."""

    def create(
        self,
        tenant_id: str,
        source_id: str,
        original_text: str,
        offset_start: int,
        offset_end: int,
        sensitivity: RiskClass = RiskClass.LOW,
        doc_id: str | None = None,
        chunk_id: str | None = None,
        expiry: datetime | None = None,
    ) -> AnchorRecord:
        """Mint an anchor for a protected span and return its record."""
        ...

    def get(self, anchor_id: str, requesting_tenant_id: str) -> AnchorRecord | None:
        """Resolve an anchor id, or return None when it cannot be resolved."""
        ...


class NonResolvingAnchorStore:
    """Open-build anchor store: mints anchors, resolves nothing.

    Not a stub and not a disabled store — a store that *cannot* resolve, which
    is a stronger position than the commercial one rather than a weaker one.
    See the module docstring for why ``create`` must not raise.
    """

    def __init__(self, hash_salt: str | None = None) -> None:
        self._hash_salt = hash_salt
        self._minted = 0

    def create(
        self,
        tenant_id: str,
        source_id: str,
        original_text: str,
        offset_start: int,
        offset_end: int,
        sensitivity: RiskClass = RiskClass.LOW,
        doc_id: str | None = None,
        chunk_id: str | None = None,
        expiry: datetime | None = None,
    ) -> AnchorRecord:
        from tekmyra.content_hash import content_hash

        self._minted += 1
        resolved_doc = doc_id or "_default_doc"
        resolved_chunk = chunk_id or "_default_chunk"
        span_hash = content_hash(original_text, self._hash_salt)
        return AnchorRecord(
            anchor_id=(
                f"[CTX_REF:{tenant_id}:{resolved_doc}:{resolved_chunk}:{span_hash}]"
            ),
            tenant_id=tenant_id,
            source_id=source_id,
            doc_id=resolved_doc,
            chunk_id=resolved_chunk,
            content_hash=span_hash,
            original_text=original_text,
            offset_start=offset_start,
            offset_end=offset_end,
            sensitivity=sensitivity,
        )

    def get(self, anchor_id: str, requesting_tenant_id: str) -> AnchorRecord | None:
        """Always None. The open build cannot vouch for recoverability."""
        return None


def default_anchor_store(db_path: str = ":memory:") -> AnchorStoreProtocol:
    """Return the commercial store when it is present, else the open one.

    This selects on whether the commercial module SHIPS, which is what makes one
    codebase produce both builds. It is deliberately not a silent fallback in the
    dangerous direction: the substitute refuses strictly more than the store it
    replaces, so a build that expected resolution and did not get it fails
    loudly at verification rather than quietly permitting a dropped span.

    Callers that must know which half they are running on read
    ``type(store).__name__``; the compressor benchmark records it.
    """
    try:
        from tekmyra.anchor_store import AnchorStore
    except ImportError:
        return NonResolvingAnchorStore()
    return AnchorStore(db_path)
