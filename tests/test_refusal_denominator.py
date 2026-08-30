"""Refusing must never raise a reported rate.

Vega ruling 7. The corpus totals summed over completed rows only, so a refused
fixture's tokens left the denominator entirely and REFUSING PAID. The open-core
build scored +6.09% against the commercial build's +5.15% while saving FEWER
tokens (8,849 against 8,873), purely by refusing twice as many fixtures. No
intent was required; the arithmetic did it unprompted, to both builds.

The general rule this enforces: A METRIC WHOSE DENOMINATOR IS SET BY THE
SYSTEM'S OWN REFUSALS IS GAMEABLE BY FAILING. Any rate we publish must have a
denominator the system cannot choose.

These tests exist because the refusal site already carried a comment saying
"record it so the denominator stays honest" while the code did the opposite. A
comment asserting an invariant nobody tested is load-bearing in the wrong
direction: a reviewer auditing that exact question finds an assurance at the
exact site and moves on. An unaccompanied bug gets caught by the next careful
reader; a bug wearing a correct comment survives that reader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tekmyra.benchmark_compressor import run_compressor_benchmark
from tekmyra.runtime import TerminalVerificationError

FIXTURES = Path("benchmarks/fixtures/long_context_v1")
pytestmark = pytest.mark.skipif(
    not FIXTURES.is_dir(), reason="long_context_v1 corpus not present"
)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return run_compressor_benchmark(fixture_dir=FIXTURES)


def _requires_tokens(report: dict[str, Any]) -> None:
    if not report["tokens_measured"]:
        pytest.skip(f"tokenizer unavailable: {report['tokens_withheld_reason']}")


def test_refused_fixtures_are_counted_in_the_corpus_denominator(
    report: dict[str, Any],
) -> None:
    """Every fixture's tokens must appear, whether or not it produced output."""
    _requires_tokens(report)
    if not report["failures"]:
        pytest.skip("no refusals in this corpus — nothing to weigh")

    completed = sum(r["input_tokens"] for r in report["rows"])
    refused = sum(f["input_tokens"] for f in report["failures"])
    assert report["corpus_input_tokens"] == completed + refused, (
        "the corpus denominator excludes refused fixtures, so refusing shrinks "
        "the denominator and inflates the rate"
    )


def test_a_refusal_contributes_exactly_zero_saving(report: dict[str, Any]) -> None:
    """Refused input and output must be equal — zero saving, not absent."""
    _requires_tokens(report)
    if not report["failures"]:
        pytest.skip("no refusals in this corpus")

    refused = sum(f["input_tokens"] for f in report["failures"])
    completed_in = sum(r["input_tokens"] for r in report["rows"])
    completed_out = sum(r["output_tokens"] for r in report["rows"])
    saved = report["corpus_input_tokens"] - report["corpus_output_tokens"]
    assert saved == completed_in - completed_out, (
        "refusals moved the numerator; they must contribute zero saving, so the "
        "tokens saved must equal what the COMPLETED fixtures saved"
    )
    assert report["corpus_output_tokens"] == completed_out + refused


def test_every_refusal_carries_a_token_count(report: dict[str, Any]) -> None:
    """Uncapturable at the summation — it has to be recorded at the refusal."""
    _requires_tokens(report)
    missing = [f["fixture_id"] for f in report["failures"] if f.get("input_tokens") is None]
    assert not missing, (
        f"refusal(s) recorded with no token count: {missing}. The summation "
        "cannot recover it later, so the denominator silently loses them."
    )


# Fixtures whose EMITTED ARTIFACT costs more tokens than the input, with the
# excess. A SHRINK-ONLY ratchet: this may go down, never up.
#
# Found by the test below rather than by inspection. It is the residue of the
# anchor-bloat defect that once cost 7.06% corpus-wide: a handle can still be a
# token or two more expensive than the short span it replaces, and on a fixture
# with almost nothing else to compress that surfaces as a net loss.
#
# It matters beyond its size because it is the ONLY thing that keeps the
# refusal invariant from holding absolutely. Refusing a fixture that was
# LOSING tokens does raise the corpus rate — correctly, arithmetically, and
# unavoidably once refusals count as zero saving. So every entry here is a hole
# in "refusing is never rewarded", and the honest fix is to empty this list,
# not to soften the invariant.
KNOWN_TOKEN_LOSING_FIXTURES = {"lc-distractor_dense-eligible-001": 2}


def test_refusing_a_fixture_that_saved_tokens_can_never_raise_the_rate(
    report: dict[str, Any],
) -> None:
    """The gameability property, checked by construction.

    Converts each completed fixture into a refusal and asserts the reported rate
    does not improve. Restricted to fixtures that were contributing a saving:
    for a fixture that COSTS tokens, refusing it genuinely does help, and that
    is a defect in the fixture's handling rather than in this denominator —
    tracked separately by the ratchet below.
    """
    _requires_tokens(report)
    rows = [r for r in report["rows"] if r["output_tokens"] <= r["input_tokens"]]
    if not rows:
        pytest.skip("no token-saving fixtures to convert")

    total_in = report["corpus_input_tokens"]
    total_out = report["corpus_output_tokens"]
    baseline = (total_in - total_out) / total_in * 100.0

    worsened: list[str] = []
    for row in rows:
        # Refusing this fixture: its output becomes equal to its input.
        simulated_out = total_out - row["output_tokens"] + row["input_tokens"]
        rate = (total_in - simulated_out) / total_in * 100.0
        if rate > baseline + 1e-9:
            worsened.append(f"{row['fixture_id']} -> {rate:.4f}% vs {baseline:.4f}%")

    assert not worsened, (
        "refusing a fixture RAISED the reported reduction, which means the "
        "metric rewards failing:\n  " + "\n  ".join(worsened[:5])
    )


def test_the_set_of_token_losing_fixtures_does_not_grow(
    report: dict[str, Any],
) -> None:
    """Shrink-only. Each entry is a hole in the refusal invariant."""
    _requires_tokens(report)
    observed = {
        r["fixture_id"]: r["output_tokens"] - r["input_tokens"]
        for r in report["rows"]
        if r["output_tokens"] > r["input_tokens"]
    }
    new = sorted(set(observed) - set(KNOWN_TOKEN_LOSING_FIXTURES))
    assert not new, (
        f"fixture(s) now emit MORE tokens than they consumed: {new}. The "
        "pipeline is charging the caller for the privilege, and each one lets "
        "refusing raise the corpus rate."
    )
    worse = {
        fid: (cost, KNOWN_TOKEN_LOSING_FIXTURES[fid])
        for fid, cost in observed.items()
        if cost > KNOWN_TOKEN_LOSING_FIXTURES.get(fid, 0)
    }
    assert not worse, f"token loss grew (now, baseline): {worse}"


def test_per_route_totals_also_weigh_their_refusals(report: dict[str, Any]) -> None:
    """A route that refuses everything must not report a clean figure."""
    _requires_tokens(report)
    for name, stats in report["by_route"].items():
        route_rows = [r for r in report["rows"] if str(r["route"]) == name]
        route_failures = [f for f in report["failures"] if str(f.get("route")) == name]
        if not route_failures:
            continue
        refused = sum(f["input_tokens"] for f in route_failures)
        expected_in = sum(r["input_tokens"] for r in route_rows) + refused
        assert stats["input_tokens"] == expected_in, (
            f"{name}: per-route denominator drops its {len(route_failures)} "
            "refusal(s), so the route's rate is measured over the work it "
            "happened to finish"
        )
        assert stats["token_delta"] == sum(
            r["input_tokens"] - r["output_tokens"] for r in route_rows
        ), f"{name}: refusals contributed a non-zero saving"


def test_the_terminal_error_still_carries_its_attribution() -> None:
    """Guards the field the refusal record depends on."""
    exc = TerminalVerificationError("boom")
    assert exc.route is None and exc.failed_checks == ()


def test_the_denominator_holds_when_refusals_are_forced(
    monkeypatch: pytest.MonkeyPatch,
    report: dict[str, Any],
) -> None:
    """The invariant, exercised without depending on the corpus refusing.

    THIS TEST EXISTS BECAUSE THE OTHERS STOPPED RUNNING. Once the false-refusal
    classes were fixed, long_context_v1 refused nothing, and every test above
    began skipping on "no refusals in this corpus" — a guard that reports
    success by not executing, which is the exact shape this whole file was
    written to distrust. A corpus that happens to refuse is not a test fixture;
    forcing the refusal is.

    Fails every fixture whose id ends in an even digit, then checks the
    arithmetic the real code performs.

    THE EXPECTED COUNT IS forced ∪ natural, NOT len(forced). On the resolving
    -anchor-store build this corpus refuses nothing and the two are equal —
    which is how ``== len(forced)`` survived here. On the open build the
    NonResolvingAnchorStore refuses strictly more by design, so fixtures the
    forcing hook never touched refuse on their own and the shipped suite
    reported 30 != 24 against a healthy tree. The unforced module ``report``
    supplies the natural refusal set; the union is exact because an even-digit
    fixture is forced before the real pipeline can refuse it.
    """
    from tekmyra import benchmark_compressor as BC

    real_pipeline = BC.run_hybrid_pipeline
    forced: set[str] = set()

    def sometimes_refuse(fixture, **kwargs):
        if fixture.fixture_id.endswith(("0", "2", "4", "6", "8")):
            forced.add(fixture.fixture_id)
            raise TerminalVerificationError(
                "forced", failed_checks=("numeric_preserved",)
            )
        return real_pipeline(fixture, **kwargs)

    monkeypatch.setattr(BC, "run_hybrid_pipeline", sometimes_refuse)
    result = run_compressor_benchmark(fixture_dir=FIXTURES)
    _requires_tokens(result)

    natural = {f["fixture_id"] for f in report["failures"]}
    assert forced, "no fixture was forced to refuse — the test proves nothing"
    assert result["refused_fixtures"] == len(forced | natural)

    refused_tokens = sum(f["input_tokens"] for f in result["failures"])
    assert refused_tokens > 0, "forced refusals contributed nothing to the denominator"
    assert result["corpus_input_tokens"] == (
        sum(r["input_tokens"] for r in result["rows"]) + refused_tokens
    )
    # Zero saving from the refusals: the delta is exactly what completed.
    assert (result["corpus_input_tokens"] - result["corpus_output_tokens"]) == sum(
        r["input_tokens"] - r["output_tokens"] for r in result["rows"]
    )


def test_forcing_refusals_never_beats_refusing_nothing(
    monkeypatch: pytest.MonkeyPatch, report: dict[str, Any]
) -> None:
    """The property that matters, end to end: failing must not pay.

    Compares the corpus rate with every token-saving fixture refused against the
    rate with none refused. If refusing ever wins, the metric rewards failure.
    """
    _requires_tokens(report)
    from tekmyra import benchmark_compressor as BC

    savers = {
        r["fixture_id"]
        for r in report["rows"]
        if r["output_tokens"] < r["input_tokens"]
    }
    if not savers:
        pytest.skip("no token-saving fixtures in this corpus")

    real_pipeline = BC.run_hybrid_pipeline

    def refuse_savers(fixture, **kwargs):
        if fixture.fixture_id in savers:
            raise TerminalVerificationError("forced")
        return real_pipeline(fixture, **kwargs)

    monkeypatch.setattr(BC, "run_hybrid_pipeline", refuse_savers)
    refused_run = run_compressor_benchmark(fixture_dir=FIXTURES)

    assert refused_run["corpus_input_tokens"] == report["corpus_input_tokens"], (
        "the denominator changed when fixtures were refused — it must be a "
        "property of the corpus, not of how much work the system completed"
    )
    assert refused_run["corpus_token_reduction_pct"] <= (
        report["corpus_token_reduction_pct"] + 1e-9
    ), (
        f"refusing every token-saving fixture reported "
        f"{refused_run['corpus_token_reduction_pct']:+.4f}% against "
        f"{report['corpus_token_reduction_pct']:+.4f}% for refusing none — "
        "failing pays"
    )
