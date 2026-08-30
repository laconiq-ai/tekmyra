"""The prose compressor must honour the retention floor it is GRADED against.

Two separate defects, both measured at c2b9e45 on long_context_v1 at budget
0.30, both ending the same way: a fixture compressed correctly, failed
``repeated_entity_retention``, and fell all the way back to passthrough — paying
the full cost of compression and banking none of it.

STARVATION. ``floor_cap`` derives from ``total_budget - locked_sentences``, and
``total_budget`` is raised to hold the locked spans when they overflow the
ratio. So a document whose locked spans saturate its budget leaves the entity
floor nothing to spend: lc-issue_threads-eligible-000 has 122 of 222 sentences
carrying locked spans, ``available`` 0, floor_cap 0, a floor demanding 10
sentences granted 0. The ruling-32 diagnostic reported ``starved=True``
correctly for a week and nobody read it.

DRIFT. The selector allocates against ``retention_floor(budget_ratio)`` — the
budget it AIMED at — while ``verify()`` grades ``retention_floor(achieved)``.
Every codec that undershoots its budget is therefore graded to a higher floor
than it allocated for. Measured: lc-issue_threads-calibration-001 delivered
"Platform Reliability Team" at retention 0.200 against a graded floor of
0.2093, and lost a 58% saving over a single mention.

The comment above the selector's floor asserted these "cannot drift" because
the arithmetic is one definition. The arithmetic was; the INPUTS were two. This
file is that assertion converted into a test, which is the only form of it that
keeps being true.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

from tekmyra.entity_fidelity import (
    RETENTION_FACTOR,
    count_entities,
    retention_floor,
    retention_shortfalls,
)
from tekmyra.hybrid_router import HybridRouter
from tekmyra.learned_compressor.selector import (
    _allocate_entity_floor,
    _tightest_satisfiable_budget,
)
from tekmyra.anchor_interface import default_anchor_store
from tekmyra.fixtures import load_fixtures
from tekmyra.runtime import run_hybrid_pipeline

CORPUS = Path("benchmarks/fixtures/long_context_v1")
BUDGET = 0.30

# THE PER-FIXTURE PIPELINE TESTS BELOW PIN THE RESOLVING-ANCHOR-STORE
# (COMMERCIAL) BUILD'S OUTCOMES. ``anchor_interface.make_anchor_store`` is the
# documented build selector: when ``tekmyra.anchor_store`` does not ship, the
# ``NonResolvingAnchorStore`` substitute refuses STRICTLY MORE, deliberately —
# so on the open build a handful of prose fixtures that the commercial build
# compresses instead fail verification and fall back, which is those fixtures
# being REFUSED, not this suite finding a bug. Asserting "no fixture falls
# back" here would fail a healthy open build for behaving exactly as its
# refusal accounting (26 of 40 reached, 14 refused — reproduced by the
# benchmark) says it does. Skip on the open build, with which build stated;
# never weaken the assertion where the commercial store is present.
_OPEN_BUILD = find_spec("tekmyra.anchor_store") is None
requires_resolving_anchor_store = pytest.mark.skipif(
    _OPEN_BUILD,
    reason=(
        "this test pins per-fixture outcomes of the resolving-anchor-store "
        "(commercial) build; this tree runs the open build, whose "
        "NonResolvingAnchorStore refuses strictly more by design (see "
        "tekmyra/anchor_interface.py). The open build's refusal accounting is "
        "verified by the benchmark reproduction and test_refusal_denominator."
    ),
)


def test_shortfalls_are_measured_against_the_achieved_ratio_not_a_target() -> None:
    """The whole point: the floor follows the output, not the intention."""
    original = ("Platform Reliability Team owns this. " * 20) + ("Filler prose. " * 200)
    # An output that keeps a tenth of the document keeps a low floor...
    lean = "Platform Reliability Team owns this. " * 2
    # ...while one that keeps most of it is held to a much higher one.
    fat = original[: int(len(original) * 0.9)]

    assert retention_floor(len(lean) / len(original)) < retention_floor(
        len(fat) / len(original)
    ), "a longer output must be held to a HIGHER retention floor, or the check is gameable"


def test_a_shortfall_reports_how_many_mentions_are_missing() -> None:
    """Not a boolean: the producer has to know how much to restore."""
    original = "Platform Reliability Team responded. " * 20 + "Other prose here. " * 20
    output = "Platform Reliability Team responded. " + "Other prose here. " * 18
    shortfalls = retention_shortfalls(original, output)
    assert "Platform Reliability Team" in shortfalls
    assert shortfalls["Platform Reliability Team"] >= 1


def test_no_shortfall_when_retention_tracks_the_document() -> None:
    """Proportional, not categorical — keeping a fair share must not fire."""
    original = "Platform Reliability Team responded. " * 20
    output = "Platform Reliability Team responded. " * 20
    assert retention_shortfalls(original, output) == {}


def test_entities_below_the_threshold_carry_no_obligation() -> None:
    """Below MIN_OCCURRENCES the presence-based machinery is the right tool."""
    original = "Aris Thorne spoke. " * 3 + "Filler. " * 100
    assert retention_shortfalls(original, "Filler. " * 50) == {}


def test_the_backoff_loosens_the_budget_until_the_floor_can_be_granted() -> None:
    """The starvation fix, exercised on the arithmetic that starved.

    250 sentences, 150 of them locked. At budget 0.30 total_budget is raised to
    150 to hold the locks, `available` is 0, and floor_cap is 0 — so an entity
    floor of any size is granted nothing.
    """
    locked = set(range(150))
    members = {"Platform Reliability Team": [(i, 1) for i in range(150, 200)]}

    budget, steps = _tightest_satisfiable_budget(250, locked, members, BUDGET)
    assert budget > BUDGET, "budget did not back off despite a floor that cannot be granted"
    assert steps > 0

    # And at the budget it returned, the floor really is grantable.
    total_budget = max(len(locked), 1, int(round(250 * budget)))
    floor_cap = max(0, int((total_budget - len(locked)) * 0.5))
    _, unmet = _allocate_entity_floor(
        members, set(locked), floor_cap, retention_floor(budget)
    )
    assert unmet == [], f"backoff returned {budget} but the floor is still unmet"


def test_the_backoff_does_nothing_when_the_request_is_already_satisfiable() -> None:
    """It must be inert on the common case, not quietly inflate every output."""
    members = {"Aris Thorne": [(i, 1) for i in range(10)]}
    budget, steps = _tightest_satisfiable_budget(900, set(range(4)), members, BUDGET)
    assert budget == BUDGET
    assert steps == 0


def test_backoff_is_bounded_and_returns_the_request_when_unsatisfiable() -> None:
    """An obligation larger than the document must not spin or over-promise."""
    # Every sentence locked: available is 0 at EVERY budget, so no budget helps.
    locked = set(range(100))
    members = {"Platform Reliability Team": [(i, 1) for i in range(100)]}
    budget, _ = _tightest_satisfiable_budget(100, locked, members, BUDGET)
    assert budget == BUDGET, "an unsatisfiable floor must not silently raise the budget"


def test_starvation_is_measured_against_what_was_still_needed() -> None:
    """A saturation flag must not fire on a success (Vega ruling 51).

    ``entity_floor_demand`` is GROSS — the mentions every floor asks for, before
    subtracting what ``must_keep`` already carries. Measuring starvation as
    "granted nothing while demand was non-zero" therefore fires whenever prior
    coverage happened to satisfy the floor, which is the opposite of starvation.
    Measured at request 0.30: four fixtures reported starved with demand 10 and
    granted 0 while no entity was unmet.

    Constructed here so the condition is reproducible without the corpus: every
    sentence carrying the entity is already locked, so the floor is fully
    covered before the allocator runs and it correctly grants nothing.
    """
    from tekmyra.learned_compressor.selector import _allocate_entity_floor

    members = {"Platform Reliability Team": [(i, 1) for i in range(20)]}
    must_keep = set(range(20))  # every mention already carried
    spent, unmet = _allocate_entity_floor(members, must_keep, floor_cap=0, entity_floor=0.5)

    assert spent == 0, "nothing needed to be bought"
    assert unmet == [], (
        "the obligation was met by prior coverage — reporting this as starvation "
        "would flag a success, which is what the gross-demand comparison did"
    )


@pytest.mark.skipif(not CORPUS.is_dir(), reason="long-context corpus not present")
@requires_resolving_anchor_store
def test_the_starved_flag_no_longer_fires_where_nothing_is_unmet() -> None:
    """The misfire, pinned on the real corpus rather than described.

    The legacy signal is retained deliberately for one release — deleting a
    misfiring guard in the commit that fixes it destroys the evidence it ever
    misfired. This asserts that the two now DISAGREE, which is the whole point:
    if they ever agree again on this corpus, the legacy field has stopped
    demonstrating anything and can go.
    """
    router = HybridRouter(passthrough_token_budget=0)
    store = default_anchor_store()
    disagreements = 0
    for fixture in load_fixtures(CORPUS):
        response = run_hybrid_pipeline(
            fixture,
            router=router,
            anchor_store=store,
            budget_ratio=BUDGET,
            use_expected_route=False,
        )
        explanation = response.explanations[0] if response.explanations else {}
        if explanation.get("entity_floor_starved") is None:
            continue
        assert explanation["entity_floor_starved"] == bool(
            explanation.get("entity_floor_unmet")
        ), "the corrected flag must be derived from unmet, not recomputed"
        if explanation["entity_floor_starved"] != explanation.get(
            "entity_floor_starved_gross_legacy"
        ):
            disagreements += 1
    assert disagreements > 0, (
        "the legacy gross-demand flag no longer disagrees with the corrected "
        "one anywhere on this corpus, so it has stopped evidencing the misfire "
        "and should be removed"
    )


@pytest.mark.skipif(not CORPUS.is_dir(), reason="long-context corpus not present")
@requires_resolving_anchor_store
def test_no_prose_fixture_falls_back_on_entity_retention() -> None:
    """The regression this whole file exists to hold.

    Before the backoff and the top-up, three prose fixtures on this corpus
    compressed correctly and were rejected — lc-issue_threads-calibration-001,
    -eligible-000 and -primary_5k_10k-000 — each trading a real saving for zero.
    A fallback here means a producer is again failing a contract it is graded on.
    """
    router = HybridRouter(passthrough_token_budget=0)
    store = default_anchor_store()
    offenders: list[str] = []
    for fixture in load_fixtures(CORPUS):
        response = run_hybrid_pipeline(
            fixture,
            router=router,
            anchor_store=store,
            budget_ratio=BUDGET,
            use_expected_route=False,
        )
        explanation = response.explanations[0] if response.explanations else {}
        if explanation.get("attempted_route") != "prose_token_compressor":
            continue
        if explanation.get("fell_back"):
            offenders.append(
                f"{fixture.fixture_id}: {explanation.get('fallback_failed_checks')}"
            )
    assert not offenders, (
        "prose fixtures fell back rather than compressing: "
        + "; ".join(offenders)
        + ". Each one costs its FULL saving, not a marginal one."
    )


@pytest.mark.skipif(not CORPUS.is_dir(), reason="long-context corpus not present")
def test_the_shared_constant_is_the_only_definition_of_the_share() -> None:
    """One constant, both sides — the property entity_fidelity exists to hold."""
    from tekmyra import verifier

    assert verifier.retention_floor is retention_floor
    assert RETENTION_FACTOR == 0.5
    assert retention_floor(0.4) == pytest.approx(0.2)
    assert retention_floor(3.0) == 1.0, "a floor above full retention is unclearable"


def test_counting_agrees_with_the_check_on_a_repeated_entity() -> None:
    """Producer and check must agree on WHAT is being counted, not just how."""
    text = "Platform Reliability Team paged. " * 12
    assert count_entities(text)["Platform Reliability Team"] == 12
