"""End-to-end tests for the hybrid learned router baseline."""

from __future__ import annotations

from pathlib import Path


from tekmyra.baselines import run_baseline
from tekmyra.contracts import RouteClass
from tekmyra.fixtures import load_fixtures


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "fixtures" / "synthetic"


LOSSY_ROUTES = {
    RouteClass.PROSE_TOKEN_COMPRESSOR.value,
    RouteClass.SUMMARIZER.value,
    RouteClass.EXTRACTIVE_KEYPOINT_COMPRESSOR.value,
}


def test_learned_baseline_runs_on_all_fixtures(run_or_refusal):
    fixtures = load_fixtures(FIXTURE_DIR)
    for fx in fixtures:
        response = run_or_refusal(lambda f=fx: run_baseline("learned", f))
        if response is None:
            continue  # open core refused, and the refusal itself was asserted
        assert response.compressed_segments
        assert 0.0 < response.global_metrics.compression_ratio <= 1.0
        assert response.global_metrics.verifier_pass is True


def test_learned_pipeline_no_high_risk_lossy(run_or_refusal):
    fixtures = load_fixtures(FIXTURE_DIR)
    for fx in fixtures:
        if fx.risk_tier.value in ("high", "critical"):
            response = run_or_refusal(lambda f=fx: run_baseline("learned", f))
            if response is None:
                continue  # open core refused, and the refusal itself was asserted
            route = response.explanations[0]["route_after_override"]
            assert route not in LOSSY_ROUTES


def test_learned_baseline_is_registered():
    from tekmyra.baselines import BASELINES

    assert "learned" in BASELINES
