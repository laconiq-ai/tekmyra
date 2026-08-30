"""Repeated families must never be annihilated wholesale.

CTRL-207. ``select_sentences`` allocated proportional floors largest-first
against a capped budget, so once ``_FAMILY_FLOOR_BUDGET_SHARE`` was exhausted by
the biggest families every remaining family got zero floor and survived only if
the model's ranking happened to pick it. Because ``_family_key`` is exact text,
identical sentences earn identical scores -- so a family below that cut-line is
all-or-nothing: it survives whole or disappears whole.

Measured on a real long_prose fixture (20 families, 913 sentences,
budget_ratio 0.8) only the 6 largest families cleared the cap; whole families of
73+ identical sentences were observed going to zero while near-identical
siblings survived untouched.

Raising the cap does not fix it -- at a 1.0 share a family still lands on zero.
These tests therefore pin the allocation SHAPE (survival before
proportionality), not the cap size.
"""

from __future__ import annotations

import math

from tekmyra.learned_compressor.model import KeepDropScores
from tekmyra.learned_compressor.segmenter import Sentence
from tekmyra.learned_compressor.selector import (
    _FAMILY_FLOOR_BUDGET_SHARE,
    _REPEAT_FAMILY_MIN,
    _family_key,
    select_sentences,
)


class _Feat:
    """Minimal stand-in: the selector only reads ``has_locked_span``."""

    def __init__(self, has_locked_span: bool = False) -> None:
        self.has_locked_span = has_locked_span


# Family sizes measured from the real fixture behind pair-007 of the Slice-4
# relabel bundle: ten "plain" families and ten smaller near-identical siblings.
_REAL_FAMILY_SIZES = [86, 82, 80, 78, 78, 78, 77, 75, 75, 73] + [
    21, 15, 15, 14, 14, 13, 12, 12, 9, 6
]


def _build(sizes: list[int]) -> tuple[list[Sentence], list[_Feat], list[KeepDropScores]]:
    """Build a corpus whose ranking is HOSTILE to the smaller families.

    Scores descend with family size, so the model's ranking alone would keep the
    large families and drop every small one. That is what makes the floor
    load-bearing: without it the small families die, which is exactly the
    production failure.
    """
    sentences: list[Sentence] = []
    scores: list[KeepDropScores] = []
    for fam, size in enumerate(sizes):
        text = f"Family {fam} sentence content."
        # Descending keep_prob by family index; identical within a family,
        # which is the real behaviour for identical text.
        prob = 0.99 - (fam * 0.9 / max(1, len(sizes)))
        for _ in range(size):
            sentences.append(Sentence(text=text, index=len(sentences), start=0, end=len(text)))
            scores.append(KeepDropScores(keep_prob=prob, drop_prob=1.0 - prob))
    feats = [_Feat() for _ in sentences]
    return sentences, feats, scores


def _survivors_by_family(kept: list[Sentence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in kept:
        counts[_family_key(s.text)] = counts.get(_family_key(s.text), 0) + 1
    return counts


def test_no_repeated_family_is_annihilated_on_the_real_fixture_shape() -> None:
    """The defect, on the exact family shape that produced it in production."""
    sentences, feats, scores = _build(_REAL_FAMILY_SIZES)
    kept = select_sentences(sentences, feats, scores, budget_ratio=0.8)
    survivors = _survivors_by_family(kept)

    expected_families = {_family_key(f"Family {i} sentence content.") for i in
                         range(len(_REAL_FAMILY_SIZES))}
    annihilated = sorted(
        f for f in expected_families if survivors.get(f, 0) == 0
    )
    assert annihilated == [], (
        f"{len(annihilated)} repeated families were wiped out entirely: {annihilated}"
    )


def test_survival_pass_precedes_proportional_pass() -> None:
    """Every family gets its first slot before any family gets its second.

    This is the property that distinguishes the fix from merely enlarging the
    cap. Uses a budget deliberately tight enough that a largest-first
    proportional allocation would exhaust the cap before reaching the small
    families.
    """
    sizes = [200, 180, 160, 8, 6, 4]
    sentences, feats, scores = _build(sizes)
    kept = select_sentences(sentences, feats, scores, budget_ratio=0.5)
    survivors = _survivors_by_family(kept)

    for i in range(len(sizes)):
        key = _family_key(f"Family {i} sentence content.")
        assert survivors.get(key, 0) >= 1, (
            f"family {i} (size {sizes[i]}) was annihilated despite an unspent floor budget"
        )


def test_budget_is_still_respected() -> None:
    """Survival must not become a licence to blow the size target.

    The floor is capped in aggregate; guaranteeing one slot per family must not
    push the output past the budget the caller asked for.
    """
    sizes = _REAL_FAMILY_SIZES
    sentences, feats, scores = _build(sizes)
    for budget_ratio in (0.3, 0.5, 0.8):
        kept = select_sentences(sentences, feats, scores, budget_ratio=budget_ratio)
        target = max(1, int(round(len(sentences) * budget_ratio)))
        assert len(kept) <= target, (
            f"budget_ratio={budget_ratio}: kept {len(kept)} of {len(sentences)}, "
            f"target was {target}"
        )


def test_many_tiny_families_do_not_starve_the_ranking() -> None:
    """Pathological input: more repeated families than the cap can seat.

    Preserved behaviour from adversarial QA 2026-08-02 -- floors are capped in
    aggregate so duplicated filler can never crowd out the model's ranking. With
    hundreds of tiny families the survival pass must degrade gracefully against
    the cap rather than consuming the whole budget.
    """
    sizes = [2] * 400
    sentences, feats, scores = _build(sizes)
    kept = select_sentences(sentences, feats, scores, budget_ratio=0.5)
    target = max(1, int(round(len(sentences) * 0.5)))
    assert len(kept) <= target


def test_survival_boundary_is_asserted_not_assumed() -> None:
    """Pin WHERE the survival guarantee stops holding. Vega's ruling, 2026-08-03.

    The survival pass seats one slot per repeated family out of ``floor_cap``, so
    it can only be annihilation-free while

        family_count <= floor_cap  ==  mean_family_size >= 1 / (ratio * share)

    At ratio 0.5 / share 0.5 that means a mean family size of 4. Below it the
    pigeonhole bites and exactly ``family_count - floor_cap`` families are
    annihilated -- the fix cannot help, and no cap size fixes it either.

    This replaces an assertion that read as coverage and was not: the previous
    check on the [2]*400 shape was ``len(kept) - floor_cap >= 0``, i.e.
    ``400 - 200 >= 0``, which cannot fail for ANY implementation. The boundary
    was being discovered rather than asserted. Real corpora sit far inside the
    safe region (the long_context fixtures average 45.7), which is exactly why
    the guarantee reads as unconditional there and must be pinned here instead.

    THE FORMULA IS PINNED, NOT THE NUMBERS. An earlier draft asserted the literal
    counts (200, 25, 0, 0) measured at share 0.5. Those are share-dependent: at
    share 0.65 the SAME allocator yields 140 on [2]*400, so a literal pin fails
    when only the cap moves and cannot distinguish that from the allocator
    changing -- which is the single distinction this test exists to make. It
    therefore derives the expectation from ``_FAMILY_FLOOR_BUDGET_SHARE`` in
    force. Move the share and it still passes; change the allocation shape and it
    fails.
    """
    for size in (2, 3, 4, 5):
        count = 400 if size == 2 else 100
        sizes = [size] * count
        sentences, feats, scores = _build(sizes)
        kept = select_sentences(sentences, feats, scores, budget_ratio=0.5)

        target = max(1, int(round(len(sentences) * 0.5)))
        floor_cap = max(0, int(target * _FAMILY_FLOOR_BUDGET_SHARE))
        # One survival slot per family out of floor_cap; the excess is exactly
        # what pigeonhole denies. No allocation ORDER changes this count.
        predicted = max(0, count - floor_cap)

        survivors = _survivors_by_family(kept)
        annihilated = sum(
            1 for i in range(count)
            if survivors.get(_family_key(f"Family {i} sentence content."), 0) == 0
        )
        assert annihilated == predicted, (
            f"size {size} ({count} families, budget_ratio 0.5, share "
            f"{_FAMILY_FLOOR_BUDGET_SHARE}, floor_cap {floor_cap}): the boundary "
            f"formula max(0, families - floor_cap) predicts {predicted} "
            f"annihilated, measured {annihilated}. The allocation shape has "
            f"changed -- re-derive the boundary, do not re-pin the number."
        )

    # The boundary must also BITE and RELENT across it, or the formula above is
    # being satisfied vacuously by a corpus that never crosses the line.
    strict, safe = [3] * 100, [5] * 100
    for sizes, must_annihilate in ((strict, True), (safe, False)):
        sentences, feats, scores = _build(sizes)
        kept = select_sentences(sentences, feats, scores, budget_ratio=0.5)
        survivors = _survivors_by_family(kept)
        annihilated = sum(
            1 for i in range(len(sizes))
            if survivors.get(_family_key(f"Family {i} sentence content."), 0) == 0
        )
        assert (annihilated > 0) == must_annihilate, (
            f"sizes {sizes[0]}x{len(sizes)}: expected "
            f"{'annihilation' if must_annihilate else 'none'}, got {annihilated}"
        )


def test_single_occurrence_sentences_are_not_treated_as_families() -> None:
    """_REPEAT_FAMILY_MIN still gates the floor; unique text rides the ranking."""
    assert _REPEAT_FAMILY_MIN == 2
    sizes = [50, 50, 1, 1, 1]
    sentences, feats, scores = _build(sizes)
    kept = select_sentences(sentences, feats, scores, budget_ratio=0.5)
    # The two repeated families must survive; singletons are the ranking's call.
    for i in (0, 1):
        key = _family_key(f"Family {i} sentence content.")
        assert _survivors_by_family(kept).get(key, 0) >= 1


def test_negative_control_the_survival_pass_is_load_bearing() -> None:
    """Prove the guard is what passes the tests, not the corpus being easy.

    Re-implements select_sentences WITHOUT the survival pass -- the exact
    pre-fix allocation -- and asserts it annihilates families on the same input.
    If this ever stops failing, the tests above have gone vacuous and the fix is
    no longer doing any work.
    """
    from collections import defaultdict

    def _pre_fix_select(sentences, features, scores, budget_ratio):
        must_keep: set[int] = set()
        ranked: list[tuple[float, int]] = []
        for i, (feat, score) in enumerate(zip(features, scores)):
            if feat.has_locked_span:
                must_keep.add(i)
            ranked.append((score.keep_prob, i))
        total_budget = max(len(must_keep), 1, int(round(len(sentences) * budget_ratio)))
        families: dict[str, list[int]] = defaultdict(list)
        for i, sentence in enumerate(sentences):
            families[_family_key(sentence.text)].append(i)
        available = total_budget - len(must_keep)
        floor_cap = max(0, int(available * _FAMILY_FLOOR_BUDGET_SHARE))
        floor_spent = 0
        # The original: full proportional floors, largest-first, until the cap.
        for members in sorted(families.values(), key=len, reverse=True):
            if len(members) < _REPEAT_FAMILY_MIN or floor_spent >= floor_cap:
                continue
            floor = max(1, math.ceil(len(members) * budget_ratio))
            for idx in members[:floor]:
                if floor_spent >= floor_cap:
                    break
                if idx not in must_keep:
                    must_keep.add(idx)
                    floor_spent += 1
        ranked.sort(key=lambda x: x[0], reverse=True)
        selected = set(must_keep)
        for _prob, idx in ranked:
            if len(selected) >= total_budget:
                break
            if idx not in selected:
                selected.add(idx)
        return [sentences[i] for i in sorted(selected)]

    sentences, feats, scores = _build(_REAL_FAMILY_SIZES)
    pre_fix_kept = _pre_fix_select(sentences, feats, scores, 0.8)
    pre_fix_survivors = _survivors_by_family(pre_fix_kept)
    pre_fix_annihilated = [
        i for i in range(len(_REAL_FAMILY_SIZES))
        if pre_fix_survivors.get(_family_key(f"Family {i} sentence content."), 0) == 0
    ]
    assert pre_fix_annihilated, (
        "the pre-fix allocation annihilated NOTHING on this corpus -- the tests "
        "above would pass without the fix, so they prove nothing"
    )

    # And the real implementation must survive the same input.
    fixed_kept = select_sentences(sentences, feats, scores, budget_ratio=0.8)
    fixed_survivors = _survivors_by_family(fixed_kept)
    fixed_annihilated = [
        i for i in range(len(_REAL_FAMILY_SIZES))
        if fixed_survivors.get(_family_key(f"Family {i} sentence content."), 0) == 0
    ]
    assert fixed_annihilated == [], (
        f"fix failed on the very corpus the control proves is hostile: {fixed_annihilated}"
    )
