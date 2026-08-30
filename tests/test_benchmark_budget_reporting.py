"""The budget a run REQUESTED is not the budget it ACHIEVED, and the report says both.

Since the producer backs its budget off per fixture (Vega ruling 47), the
``budget_ratio`` in a report header is a REQUEST. Measured on long_context_v1:
requesting 0.30 achieves a median output token ratio of 0.343 and NOT ONE of the
32 compressed fixtures reaches the request; requesting 0.05 achieves a median of
0.236. A reader given "budget 0.05, corpus 61.15%" would reasonably conclude we
compressed to five percent — a conclusion four times better than the truth.

That is the provenance defect this benchmark has already fixed three times: the
eligibility filter that reported on survivors without reporting exclusions, the
refusals that left the denominator, and the router that was handed the answer.
Ruling 49 caught it about to reappear underneath the best number we have.

TWO FIGURES, NOT ONE, AND THE DIFFERENCE IS LOAD-BEARING. ``budget_achieved`` is
the SENTENCE-COUNT target the selector honoured; ``output_ratio_achieved`` is
the TOKEN share of the document that survived. They diverge because sentences
differ in length and because locked spans, numerics and the entity top-up are
kept whole whatever the target. Reporting either alone answers a question
nobody asked.
"""

from __future__ import annotations

import pytest

from tekmyra.benchmark_compressor import (
    _COMPRESSING_ROUTES,
    _budget_achieved_summary,
    _output_ratio_summary,
)
from tekmyra.codecs.selector import (
    SAFE_TERMINAL_ROUTES,
    UNIMPLEMENTED_CODEC_ROUTES,
)
from tekmyra.contracts import RouteClass


def _row(**kw: object) -> dict:
    base = {
        "attempted_route": RouteClass.PROSE_TOKEN_COMPRESSOR.value,
        "input_tokens": 1000,
        "output_tokens": 400,
        "budget_achieved": 0.3,
    }
    base.update(kw)
    return base


def test_both_summaries_are_reported_and_they_are_different_quantities() -> None:
    """A run where the sentence target was met but the token ratio was not."""
    rows = [_row(budget_achieved=0.3, input_tokens=1000, output_tokens=400)]
    budget = _budget_achieved_summary(rows, 0.3)
    ratio = _output_ratio_summary(rows, 0.3)

    assert budget["rows_reaching_request"] == 1, "the sentence target was honoured"
    assert ratio["rows_reaching_request"] == 0, (
        "the output kept 40% of tokens against a 30% request — reporting only "
        "the budget would hide that entirely"
    )
    assert budget["median"] != ratio["median"]


def test_an_absent_achieved_budget_is_withheld_rather_than_zeroed() -> None:
    """0.0 would read as 'achieved a budget of zero' — the best possible lie."""
    summary = _budget_achieved_summary([_row(budget_achieved=None)], 0.3)
    assert summary["rows_reporting"] == 0
    assert "withheld_reason" in summary
    assert "min" not in summary and "median" not in summary


def test_no_compressing_row_is_withheld_rather_than_zeroed() -> None:
    """Same rule one level over: an empty scope is not a measurement of zero."""
    summary = _output_ratio_summary(
        [_row(attempted_route=RouteClass.REVERSIBLE_RETRIEVAL_MODE.value)], 0.3
    )
    assert summary["rows_reporting"] == 0
    assert "withheld_reason" in summary


def test_passthrough_rows_do_not_dilute_the_achieved_ratio() -> None:
    """A route that declined to compress is a coverage fact, not a ratio of 1.0.

    Mixing them in would drag the achieved ratio toward 1.0 and make the codec
    look weaker than it is — the mirror of the survivor bias, which made it look
    stronger.
    """
    rows = [
        _row(input_tokens=1000, output_tokens=300),
        _row(
            attempted_route=RouteClass.GOVERNANCE_SAFE_COMPRESSION.value,
            input_tokens=1000,
            output_tokens=1000,
        ),
    ]
    summary = _output_ratio_summary(rows, 0.3)
    assert summary["rows_reporting"] == 1
    assert summary["median"] == 0.3


def test_the_median_is_a_real_median_on_an_even_count() -> None:
    """Off-by-one here would misreport every even-sized run."""
    rows = [
        _row(input_tokens=100, output_tokens=20),
        _row(input_tokens=100, output_tokens=40),
    ]
    assert _output_ratio_summary(rows, 0.3)["median"] == pytest.approx(0.30)


def test_compressing_routes_are_derived_from_the_codec_map() -> None:
    """A literal list here would have gone stale the day safe mode was built.

    LEGAL_SECURITY_SAFE_MODE moved from the passthrough no-op to a real codec at
    c2b9e45. A hardcoded pair would have kept reporting it as non-compressing
    while it was doing 67% of the work on its route.
    """
    assert RouteClass.LEGAL_SECURITY_SAFE_MODE.value in _COMPRESSING_ROUTES
    assert RouteClass.PROSE_TOKEN_COMPRESSOR.value in _COMPRESSING_ROUTES
    assert RouteClass.REVERSIBLE_RETRIEVAL_MODE.value not in _COMPRESSING_ROUTES
    assert RouteClass.GOVERNANCE_SAFE_COMPRESSION.value not in _COMPRESSING_ROUTES
    for route in SAFE_TERMINAL_ROUTES | UNIMPLEMENTED_CODEC_ROUTES:
        assert route.value not in _COMPRESSING_ROUTES
