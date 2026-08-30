"""Weak labeler for sentence keep/drop decisions."""

from __future__ import annotations

from tekmyra.entity_fidelity import RETENTION_FACTOR
from tekmyra.learned_compressor.features import SentenceFeatures


def label_keep(feature: SentenceFeatures) -> int:
    """Return 1 if the sentence should be kept, 0 if it may be dropped.

    Locked spans, citations, agent-state markers, numerics, and opening/closing
    sentences are treated as weak keep signals.
    """
    if feature.has_locked_span:
        return 1
    if feature.has_citation or feature.has_agent_state or feature.has_numeric:
        return 1
    # THE RETENTION CONTRACT, PROPORTIONALLY.
    #
    # Keep a sentence carrying a frequently-repeated entity only while it is
    # within the first RETENTION_FACTOR share of that entity's mentions. NOT a
    # blanket keep: labelling all 110 sentences of a 110-mention entity would
    # teach the model to satisfy the retention check BY NOT COMPRESSING, which
    # is a remedy that defeats the thing it protects.
    #
    # RETENTION_FACTOR is the same constant entity_fidelity.retention_floor is
    # built from, so the share taught here and the share demanded by the
    # verifier are one number rather than two that agree today.
    if (
        feature.entity_mention_rank is not None
        and feature.entity_mention_rank < RETENTION_FACTOR
    ):
        return 1
    # Keep first and last sentences as cheap discourse anchors.
    if feature.position <= 0.05 or feature.position >= 0.95:
        return 1
    return 0
