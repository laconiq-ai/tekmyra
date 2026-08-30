"""Local evaluation for raw-vs-compressed downstream model answers."""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, Callable

from pydantic import BaseModel

from tekmyra.fixtures import Fixture


def _tokenize(text: str) -> list[str]:
    """Tokenize text using a simple whitespace split."""
    return text.lower().split()


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _token_f1(a: str, b: str) -> float:
    """Token-overlap F1 between two strings."""
    tokens_a = _token_set(a)
    tokens_b = _token_set(b)
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = tokens_a & tokens_b
    if not intersection:
        return 0.0
    precision = len(intersection) / len(tokens_a)
    recall = len(intersection) / len(tokens_b)
    return 2 * precision * recall / (precision + recall)


def _answer_length_delta(raw_answer: str, compressed_answer: str) -> float:
    """Relative token length change from raw to compressed answer."""
    raw_len = max(1, len(_tokenize(raw_answer)))
    compressed_len = len(_tokenize(compressed_answer))
    return (compressed_len - raw_len) / raw_len


def _signals_match(expected: Any, actual: Any) -> bool:
    """Return whether the extracted *actual* signal matches *expected*.

    Supports sets (subset match), lists (all items present), and direct equality.
    """
    if isinstance(expected, set):
        if not expected:
            return True
        if isinstance(actual, set):
            return expected.issubset(actual)
        if isinstance(actual, (list, tuple)):
            return expected.issubset(set(actual))
        return False
    if isinstance(expected, (list, tuple)):
        if not expected:
            return True
        if isinstance(actual, (list, tuple)):
            return set(expected).issubset(set(actual))
        if isinstance(actual, set):
            return set(expected).issubset(actual)
        return False
    return expected == actual


def _date_variants(text: str) -> list[str]:
    """Return alternate renderings of an ISO date span.

    Handles ``YYYY-MM-DD`` -> ``Month D, YYYY`` and ``Mon D, YYYY`` style
    reformats, so a span of ``2026-08-15`` matches either ``2026-08-15``,
    ``August 15, 2026``, or ``Aug 15, 2026`` in the answer. Non-date text
    returns the single normalized form.
    """
    text = text.strip()
    variants: list[str] = [text]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(text, fmt)
        except ValueError:
            continue
        for out_fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"):
            try:
                rendered = dt.strftime(out_fmt).replace(" 0", " ")
            except ValueError:
                continue
            if rendered not in variants:
                variants.append(rendered)
        break
    return variants


def _number_variants(text: str) -> list[str]:
    """Return alternate renderings of a numeric span (percent, currency).

    ``12%`` also matches ``12 percent``; ``$1,234.50`` also matches
    ``1234.50``. Non-numeric text is returned unchanged.
    """
    variants = [text]
    stripped = text.strip()
    if stripped.endswith("%") and stripped[:-1].isdigit():
        variants.append(stripped[:-1] + " percent")
        variants.append(stripped[:-1] + " pct")
    if stripped.startswith("$"):
        bare = stripped.lstrip("$").replace(",", "")
        if bare not in variants:
            variants.append(bare)
    return variants


def _span_variants(span_text: str) -> list[str]:
    """Combine date, number, and case variants for a protected-span text."""
    variants: list[str] = []
    for variant in _date_variants(span_text):
        for num in _number_variants(variant):
            low = num.lower()
            if low not in variants:
                variants.append(low)
            if num not in variants:
                variants.append(num)
    return variants or [span_text.lower()]


def _protected_disposition_valid(compressed_answer: str, fixture: Fixture) -> bool:
    """Return True when every protected span/fact is present in the answer.

    Verification is identity-driven: a span is satisfied by its stable
    ``fact_id`` (looked up in ``fixture.protected_fact_index``) or by its
    verbatim ``span_id``/text.  Formatting variants (date reformatting,
    percent notation, currency stripping, case) are accepted.
    """
    answer_lower = compressed_answer.lower()
    fact_index = fixture.protected_fact_index or {}

    spans = []
    spans.extend(fixture.protected_spans or [])
    spans.extend(fixture.gold_output.must_preserve_spans or [])

    if not spans and not fixture.gold_output.answer_required_fact_ids:
        return True

    for span in spans:
        fact_id = getattr(span, "fact_id", None)
        canonical = fact_index.get(fact_id) if fact_id else None
        texts = [canonical] if canonical else []
        span_text = getattr(span, "text", None)
        if span_text and span_text not in texts:
            texts.append(span_text)

        found = False
        for text in texts:
            if not text:
                continue
            for variant in _span_variants(text):
                if variant in answer_lower:
                    found = True
                    break
            if found:
                break
        if not found:
            return False

    return True


class FactRecallResult(BaseModel):
    required_ids: list[str]
    recalled_ids: list[str]
    recall: float


def _required_answer_fact_recall(compressed_answer: str, fixture: Fixture) -> FactRecallResult:
    """Recall of ``gold_output.answer_required_fact_ids`` in the model answer.

    Each required fact_id is resolved to canonical text via
    ``fixture.protected_fact_index`` and then searched in the answer.  The
    returned recall is ``recalled_ids / required_ids`` and is always in
    ``[0.0, 1.0]``.
    """
    required_ids = fixture.gold_output.answer_required_fact_ids or []
    fact_index = fixture.protected_fact_index or {}
    answer_lower = compressed_answer.lower()

    recalled_ids: list[str] = []
    for fact_id in required_ids:
        canonical = fact_index.get(fact_id)
        if not canonical:
            # Fact ID is required but not indexed; treat as missing.
            continue
        for variant in _span_variants(canonical):
            if variant in answer_lower:
                recalled_ids.append(fact_id)
                break

    recall = len(recalled_ids) / len(required_ids) if required_ids else 1.0
    return FactRecallResult(
        required_ids=list(required_ids),
        recalled_ids=recalled_ids,
        recall=recall,
    )


def _critical_detail_retention(compressed_answer: str, fixture: Fixture) -> float:
    """Deprecated: fraction of protected-span texts present in the answer.

    Kept for one transition cycle as a backward-compatible alias. New code
    should use :func:`_protected_disposition_valid` and
    :func:`_required_answer_fact_recall`.
    """
    warnings.warn(
        "_critical_detail_retention is deprecated; use _protected_disposition_valid",
        DeprecationWarning,
        stacklevel=2,
    )
    if not fixture.protected_spans:
        return 1.0
    compressed_lower = compressed_answer.lower()
    retained = 0
    for span in fixture.protected_spans:
        variants = _span_variants(span.text)
        if any(variant in compressed_lower for variant in variants):
            retained += 1
    return retained / len(fixture.protected_spans)


def evaluate_pair(
    raw_answer: str,
    compressed_answer: str,
    fixture: Fixture,
    extractor: Callable[[str], Any],
) -> dict[str, Any]:
    """Evaluate a raw-vs-compressed downstream answer pair.

    Parameters
    ----------
    raw_answer:
        The model answer produced from the raw (uncompressed) content.
    compressed_answer:
        The model answer produced from the deterministic compressed content.
    fixture:
        The benchmark fixture, used for protected-span disposition and
        required-answer-fact recall.
    extractor:
        Callable returned by :func:`tekmyra.downstream_tasks.build_task`
        that extracts the expected signal from a text.

    Returns
    -------
    dict
        Evaluation row with keys ``raw_correct``, ``compressed_correct``,
        ``protected_disposition_valid``, ``required_answer_fact_recall``,
        ``semantic_similarity``, ``answer_length_delta``, and the deprecated
        alias ``critical_detail_retention``.
    """
    expected_signal = extractor(fixture.content)
    raw_signal = extractor(raw_answer)
    compressed_signal = extractor(compressed_answer)

    disposition_valid = _protected_disposition_valid(compressed_answer, fixture)
    fact_recall = _required_answer_fact_recall(compressed_answer, fixture)

    return {
        "raw_correct": _signals_match(expected_signal, raw_signal),
        "compressed_correct": _signals_match(expected_signal, compressed_signal),
        "protected_disposition_valid": disposition_valid,
        "required_answer_fact_recall": fact_recall.recall,
        "semantic_similarity": _token_f1(raw_answer, compressed_answer),
        "answer_length_delta": _answer_length_delta(raw_answer, compressed_answer),
        "critical_detail_retention": 1.0 if disposition_valid else 0.0,
    }
