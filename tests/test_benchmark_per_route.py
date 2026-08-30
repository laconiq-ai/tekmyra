"""A corpus mean must be reported beside the distribution it averages.

The defect these prevent is not a crash and not a wrong number. It is a report
that is arithmetically correct and still misleads: "+5.15% tokens saved" over a
corpus where two routes did all the work and four were flat reads as a uniform
property of the product. The mean cannot say which, and a reader has no way to
tell the two worlds apart from the aggregate alone.

Three things must survive refactoring:

* every route that produced a row appears, with its own token delta;
* abstention — routing to REVERSIBLE_RETRIEVAL_MODE rather than compressing —
  is labelled as such and not left to read as a route that compresses badly;
* refusals are attributable to the route that refused, so the refusal count is
  a column in the table rather than a total floating beside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tekmyra.benchmark_compressor import (
    print_compressor_report,
    run_compressor_benchmark,
)
from tekmyra.contracts import RouteClass
from tekmyra.runtime import TerminalVerificationError
from tekmyra.token_measure import (
    ENCODING_NAME,
    TokenizerUnavailable,
    count_tokens,
    tokenizer_available,
)

FIXTURES = Path("benchmarks/fixtures/synthetic")
pytestmark = pytest.mark.skipif(
    not FIXTURES.is_dir(), reason="synthetic fixtures not present"
)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return run_compressor_benchmark(fixture_dir=FIXTURES)


def test_every_route_that_produced_a_row_appears_in_the_breakdown(
    report: dict[str, Any],
) -> None:
    by_route = report["by_route"]
    assert by_route, "no per-route breakdown was produced"

    routes_in_rows = {str(r["route"]) for r in report["rows"] if r["route"] is not None}
    assert routes_in_rows <= set(by_route), (
        "route(s) produced rows but are missing from by_route: "
        f"{sorted(routes_in_rows - set(by_route))} — the mean would then average "
        "over rows the breakdown never shows"
    )


def test_the_breakdown_accounts_for_every_completed_and_refused_fixture(
    report: dict[str, Any],
) -> None:
    """No row may fall between the table and the totals."""
    by_route = report["by_route"]
    completed = sum(s["completed_fixtures"] for s in by_route.values())
    assert completed == report["completed_fixtures"], (
        f"per-route completed sums to {completed} but the report claims "
        f"{report['completed_fixtures']}"
    )

    attributed = sum(s["refused_fixtures"] for s in by_route.values())
    unattributed = sum(1 for f in report["failures"] if not f.get("route"))
    assert attributed + unattributed == report["refused_fixtures"], (
        "refusals are neither attributed to a route nor counted as "
        "unattributable — some refusal has gone missing from the table"
    )


def test_abstention_is_labelled_not_averaged_away(report: dict[str, Any]) -> None:
    """REVERSIBLE_RETRIEVAL_MODE is the pipeline declining, and must say so."""
    by_route = report["by_route"]
    flagged = [name for name, s in by_route.items() if s["is_abstention_route"]]
    assert flagged == [RouteClass.REVERSIBLE_RETRIEVAL_MODE.value] or flagged == [], (
        f"exactly the abstention route may carry the abstention flag, got {flagged}"
    )

    expected = sum(
        1
        for r in report["rows"]
        if str(r["route"]) == RouteClass.REVERSIBLE_RETRIEVAL_MODE.value
    )
    assert report["abstention_rows"] == expected


def test_token_totals_are_summed_over_tokens_not_averaged_over_fixtures(
    report: dict[str, Any],
) -> None:
    """A 200-token fixture must not weigh the same as a 6,000-token one."""
    if not report["tokens_measured"]:
        pytest.skip(f"tokenizer unavailable: {report['tokens_withheld_reason']}")

    assert report["token_encoding"] == ENCODING_NAME
    rows = [r for r in report["rows"] if r["input_tokens"] is not None]
    assert rows, "tokens were reported as measured but no row carries a count"
    assert report["corpus_input_tokens"] == sum(r["input_tokens"] for r in rows)
    assert report["corpus_output_tokens"] == sum(r["output_tokens"] for r in rows)

    for name, stats in report["by_route"].items():
        route_rows = [r for r in rows if str(r["route"]) == name]
        assert stats["input_tokens"] == sum(r["input_tokens"] for r in route_rows), name
        assert stats["token_delta"] == stats["input_tokens"] - stats["output_tokens"]


def test_a_withheld_token_figure_carries_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent tokenizer must withhold and explain, never report zero.

    A zero token delta and an unmeasurable one are the same value and opposite
    claims. This is the same contract the headline ratio already honours.
    """
    monkeypatch.setattr(
        "tekmyra.benchmark_compressor.tokenizer_available",
        lambda: (False, "tiktoken absent (simulated)"),
    )
    result = run_compressor_benchmark(fixture_dir=FIXTURES)

    assert result["tokens_measured"] is False
    assert result["tokens_withheld_reason"] == "tiktoken absent (simulated)"
    assert result["token_encoding"] is None
    assert result["corpus_input_tokens"] is None
    assert result["corpus_token_reduction_pct"] is None
    for name, stats in result["by_route"].items():
        assert stats["token_delta"] is None, f"{name} reported a delta with no tokenizer"
        assert stats["token_reduction_pct"] is None

    # And it must still print rather than raising on the None columns.
    print_compressor_report(result)


def test_the_breakdown_prints_before_the_headline(report: dict[str, Any], capsys) -> None:
    """A reader who meets the aggregate first has already formed the impression."""
    print_compressor_report(report)
    out = capsys.readouterr().out
    assert "## Per route" in out
    assert out.index("## Per route") < out.index("## Result"), (
        "the distribution must precede the mean it corrects"
    )


def test_a_refusal_carries_the_route_and_the_checks_that_failed() -> None:
    """Without these a refusal cannot be attributed, only tallied."""
    exc = TerminalVerificationError(
        "verification_failed_after_fallback",
        route=RouteClass.PROSE_TOKEN_COMPRESSOR,
        fallback_route=RouteClass.REVERSIBLE_RETRIEVAL_MODE,
        failed_checks=("numeric_preserved", "repeated_entity_retention"),
    )
    assert str(exc) == "verification_failed_after_fallback"
    assert exc.route is RouteClass.PROSE_TOKEN_COMPRESSOR
    assert exc.fallback_route is RouteClass.REVERSIBLE_RETRIEVAL_MODE
    assert exc.failed_checks == ("numeric_preserved", "repeated_entity_retention")


def test_the_bare_message_form_still_works() -> None:
    """Callers that only pass a message must keep working."""
    exc = TerminalVerificationError("boom")
    assert str(exc) == "boom"
    assert exc.route is None
    assert exc.failed_checks == ()


def test_tokens_are_counted_not_estimated() -> None:
    """The unit must be tokens under the pinned encoding, never a proxy."""
    available, reason = tokenizer_available()
    if not available:
        pytest.skip(f"tokenizer unavailable: {reason}")
    # A handle is short in characters and expensive in tokens — the exact
    # disagreement that made a character-based figure report the anchor-bloat
    # regression as a win.
    handle = "[ANCHOR:a]"
    assert count_tokens(handle) > 1
    assert count_tokens("") == 0


def test_the_tokenizer_refusal_is_an_exception_not_a_fallback() -> None:
    """TokenizerUnavailable must exist and be an error, not a sentinel value."""
    assert issubclass(TokenizerUnavailable, Exception)
