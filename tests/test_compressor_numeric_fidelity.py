"""The prose compressor must not drop figures out of prose.

Measured defect this guards (2026-08-07): prose-000 compressed to ratio 0.656 by
deleting "The quarterly review showed a 12% increase in recurring revenue." —
losing the figure entirely, while locked-span preservation still reported a
clean 68/68. Every safety indicator read green while the output silently lost a
business fact, and the pipeline then died at the verifier rather than degrading.

Sentence selection is score-based and has no notion of the verifier's
numeric_preserved contract, so nothing but this guard keeps the two aligned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tekmyra.fixtures import load_fixtures
from tekmyra.hybrid_router import HybridRouter
from tekmyra.numeric_fidelity import (
    extract_numerics,
    extract_numerics_in_spans,
)
from tekmyra.runtime import TerminalVerificationError, run_hybrid_pipeline

FIXTURES = Path("benchmarks/fixtures/synthetic")


def test_numeric_contract_has_exactly_one_definition() -> None:
    """The verifier must enforce the same helpers producers compress against.

    Two copies of this regex that agree today are free to drift tomorrow, and
    the drift would be invisible until a codec started losing figures again.
    """
    from tekmyra import verifier

    assert verifier._extract_numerics is extract_numerics
    assert verifier._extract_numerics_in_spans is extract_numerics_in_spans
    assert not hasattr(verifier, "_NUMERIC_RE")


def test_numeric_extraction_matches_the_documented_quirks() -> None:
    """Producers must match the check exactly, including where it is surprising."""
    assert extract_numerics("a 12% rise") == {"12"}
    assert extract_numerics("2026-08-15") == {"2026", "-08", "-15"}
    assert extract_numerics("no digits here") == set()


def test_locked_span_numerics_are_excluded_from_the_requirement() -> None:
    text = "Board meets 2026-08-15. Revenue rose 12 percent."
    spans = [{"text": "2026-08-15", "start": 12, "end": 22}]
    locked = extract_numerics_in_spans(text, spans)
    required = extract_numerics(text) - locked
    assert "12" in required
    assert "2026" not in required


def test_compressor_itself_preserves_numerics_under_an_aggressive_budget() -> None:
    """The guard must hold AT THE COMPRESSOR, not via a downstream rescue.

    Asserting only end-to-end is not sufficient and was measured to be useless:
    with this repair disabled the pipeline still emits every numeric, because the
    prose route's fallback (REVERSIBLE_RETRIEVAL_MODE) passes the ORIGINAL text
    through untouched. That rescue makes a lossy compressor and a faithful one
    indistinguishable from outside. This exercises compress() directly so the
    fallback cannot launder the result.
    """
    from tekmyra.contracts import PolicyTier, SourceChannel
    from tekmyra.learned_compressor import load_compressor
    from tekmyra.prober import probe_text

    text = (
        "The quarterly review showed a 12% increase in recurring revenue. "
        "Customer churn remained below 2% for the third consecutive quarter. "
        "The team held its usual weekly sync and discussed the agenda. "
        "Several colleagues mentioned that the office coffee has improved. "
        "The board will meet on 2026-08-15 to approve the budget."
    )
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    compressor = load_compressor()

    # Aggressive budget: force the selector to drop sentences.
    result = compressor.compress(text, features, locked_spans=[], budget_ratio=0.2)

    assert result.success
    required = extract_numerics(text)
    emitted = extract_numerics(result.output)
    assert required <= emitted, (
        f"compressor dropped numerics {sorted(required - emitted)} "
        f"at ratio {result.ratio:.3f}"
    )


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="synthetic fixtures not present")
def test_prose_fixtures_complete_on_the_compressor_route_with_numerics_intact() -> None:
    """End to end: prose completes THROUGH THE COMPRESSOR, figures intact.

    The route assertion is load-bearing. Without it this passes even when the
    compressor loses a figure, because the fallback would have rescued the text.
    """
    fixtures = [f for f in load_fixtures(FIXTURES) if f.content_type == "prose"]
    assert fixtures, "expected prose fixtures in the synthetic corpus"

    router = HybridRouter(passthrough_token_budget=0)
    for fixture in fixtures:
        try:
            response = run_hybrid_pipeline(fixture, router=router)
        except TerminalVerificationError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{fixture.fixture_id} went terminal: {exc}")

        route = (response.explanations[0] if response.explanations else {}).get(
            "route_after_override"
        )
        assert route == "prose_token_compressor", (
            f"{fixture.fixture_id} did not complete on the compressor route "
            f"(got {route!r}) — a fallback rescue hides compressor data loss"
        )

        spans = [
            {"text": s.text, "start": s.start, "end": s.end}
            for s in (fixture.protected_spans or [])
        ]
        locked = extract_numerics_in_spans(fixture.content, spans)
        required = extract_numerics(fixture.content) - locked
        emitted = extract_numerics(
            " ".join(seg.compressed_content for seg in response.compressed_segments)
        )

        assert required <= emitted | locked, (
            f"{fixture.fixture_id} lost numerics "
            f"{sorted(required - (emitted | locked))} at ratio "
            f"{response.global_metrics.compression_ratio:.3f}"
        )
