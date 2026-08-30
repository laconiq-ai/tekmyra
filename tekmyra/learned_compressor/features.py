"""Sentence-level feature extraction for the learned compressor."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from tekmyra.contracts import FeatureVector
from tekmyra.entity_fidelity import MIN_OCCURRENCES, RETENTION_FACTOR, count_entities
from tekmyra.learned_compressor.segmenter import Sentence


_NUMERIC_RE = re.compile(r"-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")
_CITATION_RE = re.compile(r"\[\d+\]|\[[A-Za-z]+\d{4}\]|\bSOURCE:\S+", re.I)
_AGENT_STATE_RE = re.compile(
    r"\b(?:TODO|BLOCKER|APPROVED|REJECTED|PENDING|FOLLOW-UP|ACTION ITEM):?\b", re.I
)


@dataclass(frozen=True)
class SentenceFeatures:
    sentence: Sentence
    length: int
    word_count: int
    position: float  # 0.0 = first, 1.0 = last
    has_numeric: bool
    has_citation: bool
    has_agent_state: bool
    has_locked_span: bool
    # THE RETENTION CONTRACT, MADE VISIBLE TO THE MODEL.
    #
    # None when the sentence mentions no frequently-repeated entity. Otherwise
    # this mention's rank among all mentions of the most-repeated entity it
    # carries, normalised to [0, 1) in document order — 0.0 for the first
    # mention, approaching 1.0 for the last.
    #
    # A RANK, NOT A FLAG, AND THAT IS THE WHOLE DESIGN. A boolean
    # "mentions a repeated entity" would let the labeler keep all 110 sentences
    # carrying a 110-mention entity, so the compressor would satisfy the
    # retention check BY NOT COMPRESSING — a remedy that defeats the thing it
    # protects, which this codebase has now done three times. A rank lets the
    # labeler keep a PROPORTION, and the proportion is read from the same
    # constant the verifier's floor is built on.
    entity_mention_rank: float | None = None


def compute_entity_mention_ranks(sentences: list[Sentence]) -> dict[int, float]:
    """Rank each sentence among the mentions of the repeated entity it carries.

    Computed once per DOCUMENT because repetition is a document property — a
    sentence cannot know it mentions the 87th of 110 occurrences by looking at
    itself. Returns sentence index -> rank in [0, 1); absent means the sentence
    carries no frequently-repeated entity.

    A sentence mentioning several repeated entities takes its LOWEST rank, so
    being an early mention of any one of them is enough to be worth keeping.
    Mention definition and the repetition threshold come from entity_fidelity,
    which is the same code the verifier counts with.
    """
    per_sentence = [count_entities(sentence.text) for sentence in sentences]
    totals: Counter[str] = Counter()
    for counts in per_sentence:
        totals.update(counts)

    ranks: dict[int, float] = {}
    for entity, total in totals.items():
        if total < MIN_OCCURRENCES:
            continue
        seen = 0
        for index, counts in enumerate(per_sentence):
            occurrences = counts.get(entity, 0)
            if not occurrences:
                continue
            rank = seen / total
            seen += occurrences
            if index not in ranks or rank < ranks[index]:
                ranks[index] = rank
    return ranks


def compute_entity_sentence_members(
    sentences: list[Sentence],
) -> dict[str, list[tuple[int, int]]]:
    """Per repeated entity, the sentences mentioning it: (index, mention_count).

    PER ENTITY, NOT PER SENTENCE, and that distinction is the difference
    between this working and not. A per-sentence scalar rank — "how early is
    this sentence among the mentions of the entity it is earliest for" —
    protects a sentence on behalf of ONE entity, so an entity whose every
    mention lands in a sentence that is LATE for some other entity receives no
    protection at all. Measured: forcing 26 sentences by min-rank left
    "Platform Reliability Team" at 20 -> 2, unchanged, because none of the 26
    were its sentences.

    Retention is a promise made to each entity separately, so it has to be
    allocated to each entity separately.
    """
    members: dict[str, list[tuple[int, int]]] = {}
    for sentence in sentences:
        for entity, count in count_entities(sentence.text).items():
            members.setdefault(entity, []).append((sentence.index, count))
    return {
        entity: rows
        for entity, rows in members.items()
        if sum(count for _, count in rows) >= MIN_OCCURRENCES
    }


def extract_sentence_features(
    sentence: Sentence,
    feature_vector: FeatureVector,
    locked_span_texts: set[str],
    total_sentences: int,
    entity_mention_ranks: dict[int, float] | None = None,
) -> SentenceFeatures:
    text = sentence.text
    return SentenceFeatures(
        sentence=sentence,
        length=len(text),
        word_count=len(text.split()),
        position=sentence.index / max(1, total_sentences - 1),
        has_numeric=_NUMERIC_RE.search(text) is not None,
        has_citation=_CITATION_RE.search(text) is not None,
        has_agent_state=_AGENT_STATE_RE.search(text) is not None,
        has_locked_span=any(span in text for span in locked_span_texts),
        entity_mention_rank=(entity_mention_ranks or {}).get(sentence.index),
    )
