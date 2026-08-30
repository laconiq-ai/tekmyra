"""Budgeted sentence selector that preserves locked spans."""

from __future__ import annotations

import math
from collections import defaultdict

from tekmyra.entity_fidelity import retention_floor
from tekmyra.learned_compressor.features import SentenceFeatures
from tekmyra.learned_compressor.model import KeepDropScores
from tekmyra.learned_compressor.segmenter import Sentence

# A sentence text occurring this many times or more forms a "repeated
# family". Identical texts earn identical model scores, so a hard rank
# cutoff otherwise drops the entire family in one block once it falls
# below the budget line (the Slice-4 repetition-collapse defect: 52
# identical mentions surviving as 4). Applies to every content type.
_REPEAT_FAMILY_MIN = 2
# Family floors may claim at most this share of the total budget, so
# repetition survives proportionally WITHOUT floors becoming the budget:
# unbounded floors let many small duplicate families blow the size target
# by orders of magnitude, or let one huge boilerplate family starve
# high-scoring unique content (adversarial QA, 2026-08-02). The model's
# ranking always keeps at least the other half of the budget.
_FAMILY_FLOOR_BUDGET_SHARE = 0.5


# How far the budget may be loosened, and in what steps, when locked spans
# saturate it. See `_tightest_satisfiable_budget`.
_BACKOFF_STEP = 0.05


def _family_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def _allocate_entity_floor(
    entity_sentence_members: dict[str, list[tuple[int, int]]],
    must_keep: set[int],
    floor_cap: int,
    entity_floor: float,
    spent: int = 0,
) -> tuple[int, list[str]]:
    """Grant each entity its retention floor, earliest mentions first.

    Mutates ``must_keep``. Returns ``(spent, unmet)`` where ``unmet`` names the
    entities that could not reach their floor within ``floor_cap``.

    ONE DEFINITION, USED TWICE. The backoff search below has to ask "would the
    floor be satisfiable at budget b?", and the only honest way to answer is to
    run the real allocation on a trial copy. A second model of the arithmetic
    would be a second thing to drift — which is the defect this package has
    spent a week removing, and it would drift in the direction of claiming
    satisfiable when it is not.
    """
    unmet: list[str] = []
    for entity, entity_members in sorted(
        entity_sentence_members.items(),
        key=lambda kv: sum(c for _, c in kv[1]),
        reverse=True,
    ):
        total_mentions = sum(count for _, count in entity_members)
        needed = math.ceil(total_mentions * entity_floor)
        kept_mentions = sum(
            count for idx, count in entity_members if idx in must_keep
        )
        for idx, count in entity_members:
            if spent >= floor_cap or kept_mentions >= needed:
                break
            if idx in must_keep:
                continue
            must_keep.add(idx)
            spent += 1
            kept_mentions += count
        if kept_mentions < needed:
            unmet.append(entity)
    return spent, unmet


def _tightest_satisfiable_budget(
    n_sentences: int,
    locked: set[int],
    entity_sentence_members: dict[str, list[tuple[int, int]]],
    requested: float,
) -> tuple[float, int]:
    """The tightest budget at or above ``requested`` whose entity floor can be granted.

    WHY THE BUDGET MOVES INSTEAD OF THE CAP (Vega ruling 47). ``floor_cap``
    derives from ``total_budget - len(locked)``, so a document whose locked
    spans already fill the budget leaves the entity floor nothing to spend:
    measured on lc-issue_threads-eligible-000, 122 of 222 sentences carry locked
    spans, ``available`` is 0, and an entity floor demanding 10 sentences is
    granted 0. The output then near-eliminates a repeated entity, fails
    ``repeated_entity_retention``, and the whole fixture falls back to
    passthrough at ZERO saving.

    Enlarging ``floor_cap`` or ``total_budget`` in place would compress less
    than asked while still reporting the asked-for budget — a silent lie about
    the target. Backing the budget off instead is the same degradation stated
    honestly: this fixture cannot reach the requested budget while honouring a
    contract ``verify()`` grades, so it compresses to the tightest budget it
    CAN honour and reports that number. Graceful degradation beats a hard
    failure that yields 0%, and the achieved budget travels with the result so
    the report cannot imply otherwise.

    Returns ``(budget, steps)``; ``steps`` is 0 when the request was already
    satisfiable, which is the overwhelmingly common case.
    """
    if not entity_sentence_members:
        return requested, 0
    budget = requested
    steps = 0
    while budget <= 1.0:
        total_budget = max(len(locked), 1, int(round(n_sentences * budget)))
        floor_cap = max(0, int((total_budget - len(locked)) * _FAMILY_FLOOR_BUDGET_SHARE))
        _, unmet = _allocate_entity_floor(
            entity_sentence_members,
            set(locked),  # trial copy: the search must not commit selections
            floor_cap,
            retention_floor(budget),
        )
        if not unmet:
            return budget, steps
        budget = round(budget + _BACKOFF_STEP, 4)
        steps += 1
    # Nothing in [requested, 1.0] satisfies it. Return the request unchanged and
    # let the floor do what it can: a fixture whose entity floor is unsatisfiable
    # even at budget 1.0 has more obligation than document, and silently
    # emitting the whole input as "compressed" would be the worse answer.
    return requested, steps


def select_sentences(
    sentences: list[Sentence],
    features: list[SentenceFeatures],
    scores: list[KeepDropScores],
    budget_ratio: float,
    entity_sentence_members: dict[str, list[tuple[int, int]]] | None = None,
    diagnostics: dict[str, object] | None = None,
) -> list[Sentence]:
    """Return the sentences to keep, always including locked spans.

    Args:
        sentences: Ordered sentence segments.
        features: Feature records aligned with sentences.
        scores: Keep/drop scores aligned with sentences.
        budget_ratio: Target ratio of sentences to keep (0.0 - 1.0).

    Returns:
        Ordered list of sentences selected for the compressed output.
    """
    if not sentences:
        return []
    entity_sentence_members = entity_sentence_members or {}

    # Always keep locked spans and sentences the model strongly wants.
    must_keep: set[int] = set()
    ranked: list[tuple[float, int]] = []
    for i, (feat, score) in enumerate(zip(features, scores)):
        if feat.has_locked_span:
            must_keep.add(i)
        ranked.append((score.keep_prob, i))

    # BACK THE BUDGET OFF UNTIL THE ENTITY FLOOR IS GRANTABLE (Vega ruling 47).
    #
    # Done before any budget arithmetic below, because every figure downstream —
    # total_budget, floor_cap, both floors, the family share — has to be
    # computed at the budget actually being honoured. Reported through
    # diagnostics so the achieved budget travels with the number instead of the
    # requested one standing in for it.
    requested_budget = budget_ratio
    budget_ratio, backoff_steps = _tightest_satisfiable_budget(
        len(sentences), must_keep, entity_sentence_members, budget_ratio
    )

    # Determine how many sentences we can keep in total. Locked spans may
    # exceed the ratio target (correctness wins); everything else fits
    # inside it.
    total_budget = max(len(must_keep), 1, int(round(len(sentences) * budget_ratio)))

    available = total_budget - len(must_keep)
    # Round DOWN: ceil would hand floors 100% of a budget of 1 (a normal
    # TL;DR/headline setting), letting duplicated filler beat the single
    # top-ranked sentence (adversarial QA round 2). At tiny budgets the
    # model's ranking owns the whole budget.
    floor_cap = max(0, int(available * _FAMILY_FLOOR_BUDGET_SHARE))
    floor_spent = 0

    # ENTITY FLOOR — the contract the verifier actually grades against.
    #
    # The family floor above protects repeated SENTENCES, keyed on exact text.
    # It cannot see a repeated ENTITY spread across 110 DISTINCT sentences: no
    # two are identical, so no family forms and nothing protects them. That is
    # why "Platform Reliability Team" went 110 -> 2 at budget 0.8 while the
    # family machinery reported itself working.
    #
    # ALLOCATED PER ENTITY, EARLIEST MENTIONS FIRST, AND ACCUMULATED IN
    # MENTIONS RATHER THAN SENTENCES. Retention is a promise made to each
    # entity separately: a first attempt that force-kept sentences by a
    # per-sentence rank left the failing entity untouched, because none of the
    # sentences it appeared in were the ones that rank favoured. And the
    # verifier counts MENTIONS in the output, not sentences, so a sentence
    # carrying three mentions discharges three.
    #
    # THE FLOOR IS entity_fidelity.retention_floor, the same function the
    # verifier grades with. The inputs differ by necessity — the producer knows
    # its BUDGET, the verifier measures the ACHIEVED ratio — but the arithmetic
    # is one definition, so the two cannot drift the way the numeric contract
    # did before it got its shared leaf.
    entity_floor = retention_floor(budget_ratio)
    # A STARVED FLOOR MUST SAY SO RATHER THAN SAY NOTHING (Vega ruling 32).
    #
    # This pass was silently inert in its first form: it shared floor_cap with
    # the family floor, which had already spent it, so it ran with zero budget
    # and changed nothing while the suite stayed green. "Ran and found nothing"
    # and "never ran" were indistinguishable. They are recorded separately now.
    # DEMAND IS IN MENTIONS AND THE CAP IS IN SENTENCES — they are not
    # comparable and must not be subtracted from one another. One sentence can
    # discharge several mentions, so the sentence cost of a floor is only known
    # by walking it. `entity_floor_unmet` below is the field that actually
    # answers "was the obligation met", and it is what the backoff drives to
    # zero; the two older counters are kept because they are what distinguished
    # "ran and found nothing" from "never ran" (ruling 32).
    entity_demand = sum(
        math.ceil(sum(c for _, c in rows) * entity_floor)
        for rows in entity_sentence_members.values()
    )
    entity_spent_at_start = floor_spent

    floor_spent, unmet_entities = _allocate_entity_floor(
        entity_sentence_members, must_keep, floor_cap, entity_floor, floor_spent
    )

    if diagnostics is not None:
        granted = floor_spent - entity_spent_at_start
        diagnostics["entity_floor_demand"] = entity_demand
        diagnostics["entity_floor_granted"] = granted
        # A SATURATION FLAG COMPARES WHAT WAS DELIVERED AGAINST WHAT WAS STILL
        # NEEDED — NEVER AGAINST WHAT WAS ORIGINALLY DEMANDED (Vega ruling 51).
        #
        # `entity_demand` is GROSS: the mentions every entity's floor asks for,
        # before subtracting what `must_keep` already carries. Measuring
        # starvation as `granted == 0 and demand > 0` therefore fires whenever
        # prior coverage happened to satisfy the floor — which is a SUCCESS.
        # Measured at request 0.30: four fixtures reported starved with
        # demand=10 and granted=0 while no entity was unmet, because locked
        # sentences already carried those ten mentions.
        #
        # `unmet_entities` is the field that already knows, so the flag is
        # derived from it rather than recomputed from a quantity that cannot
        # answer the question.
        diagnostics["entity_floor_starved"] = bool(unmet_entities)
        # THE OLD SIGNAL, RETAINED ON PURPOSE FOR ONE RELEASE. Deleting a
        # misfiring guard in the commit that fixes it destroys the evidence it
        # ever misfired, and this one is worth keeping visible: it was CORRECT
        # when ruling 32 built it, and the budget backoff changed the world
        # around it. A guard does not have to be broken to stop working — it can
        # simply outlive the condition it keys on, and a green suite cannot see
        # that happen.
        diagnostics["entity_floor_starved_gross_legacy"] = (
            bool(entity_demand) and granted == 0
        )
        # NAMED, NOT COUNTED ONLY: an unsatisfiable floor is the one condition
        # the backoff could not resolve, so the entities it failed on are the
        # evidence a reader needs rather than a bare number.
        diagnostics["entity_floor_unmet"] = list(unmet_entities)
        diagnostics["floor_cap"] = floor_cap
        diagnostics["budget_requested"] = requested_budget
        diagnostics["budget_achieved"] = budget_ratio
        diagnostics["budget_backoff_steps"] = backoff_steps

    # ORDERED BEFORE THE FAMILY FLOOR, AND THE ORDER IS THE FIX.
    #
    # Both floors draw on one `floor_cap`. Family floors ask for
    # ceil(len * budget_ratio) per family, which on a real fixture (15 families,
    # sizes 23/20/18/18/17...) demands far more than the whole cap, so the
    # entity pass ran with nothing left and was silently inert — measured: the
    # failing entity stayed at 20 -> 2 with the pass in place.
    #
    # ENTITY RETENTION IS THE CONTRACT verify() ENFORCES. Family retention is a
    # heuristic about near-duplicate sentences that no check grades. The
    # obligation is allocated first and the heuristic takes what remains.

    # Repetition floor: repeated families retain a proportional share of
    # their members (earliest occurrences first), so repetition compresses
    # at the document's own rate instead of being eliminated wholesale.
    # Repetition IS signal in narrative content. Floors are capped in
    # aggregate (largest families first) so they never crowd out the
    # model's ranking or inflate the output past the budget.
    families: dict[str, list[int]] = defaultdict(list)
    for i, sentence in enumerate(sentences):
        families[_family_key(sentence.text)].append(i)

    repeated = sorted(
        (m for m in families.values() if len(m) >= _REPEAT_FAMILY_MIN),
        key=len,
        reverse=True,
    )

    # Pass 1 -- SURVIVAL. One slot for every repeated family before ANY family
    # takes a second.
    #
    # Allocating full proportional floors largest-first exhausts floor_cap on
    # the few biggest families, leaving every family below that cut-line with
    # zero floor -- surviving only if the model's ranking happens to pick it.
    # And because _family_key is exact text, identical sentences earn identical
    # scores, so a family below the cut is ALL-OR-NOTHING: it survives whole or
    # is annihilated whole. Measured on a real long_prose fixture (20 families,
    # 913 sentences, budget_ratio 0.8): only the 6 largest families cleared the
    # cap; 14 families -- including plain families of 77, 75, 75 and 73
    # sentences -- got nothing, and whole families were observed going 73 -> 0
    # while near-identical siblings survived untouched.
    #
    # Raising floor_cap does NOT fix this: even at a 1.0 share (floors
    # consuming the entire budget, which would starve unique content outright)
    # a family still lands on zero. The allocation SHAPE has to change, not its
    # size. Losing every instance of a distinct fact is a categorically worse
    # failure than keeping fewer copies of it, so survival is allocated before
    # proportionality.
    for members in repeated:
        if floor_spent >= floor_cap:
            break
        for idx in members:
            if idx not in must_keep:
                must_keep.add(idx)
                floor_spent += 1
                break

    # Pass 2 -- PROPORTIONALITY. Top families up toward their proportional
    # share, largest first, with whatever cap remains. Repetition still
    # compresses at the document's own rate; it simply can no longer take a
    # family to zero while the cap is unspent.
    for members in repeated:
        if floor_spent >= floor_cap:
            break
        floor = max(1, math.ceil(len(members) * budget_ratio))
        kept = sum(1 for idx in members if idx in must_keep)
        for idx in members:
            if floor_spent >= floor_cap or kept >= floor:
                break
            if idx not in must_keep:
                must_keep.add(idx)
                floor_spent += 1
                kept += 1

    # Sort by keep probability descending, excluding must-keeps.
    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = set(must_keep)
    for prob, idx in ranked:
        if len(selected) >= total_budget:
            break
        if idx not in selected:
            selected.add(idx)

    # Return in original order.
    return [sentences[i] for i in sorted(selected)]
