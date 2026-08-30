"""Parametrized real-runtime tests over synthetic fixtures.

These tests assert the same route-intent safety gates over the content-free
runtime metrics artifact produced by ``benchmarks/wp7_real_runtime_eval.py``.
"""

from __future__ import annotations

import json
import random
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest

from tekmyra.contracts import RouteClass
from tekmyra.fixtures import (
    Fixture,
    ProtectedSpanGold,
    load_fixtures,
)
from tekmyra.hybrid_router import HybridRouter
from tekmyra.runtime import run_deterministic_pipeline, run_hybrid_pipeline
from tekmyra.safety_gate import SafetyGate


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "fixtures" / "synthetic"
METRICS_PATH = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "results"
    / "wp7_fast_track"
    / "real_runtime_synthetic_metrics.json"
)
# Was an eighth restatement of the same five values, outside the package
# where the literal guard was looking. Imported now.
from tekmyra.pii_vocabulary import DETECTION_TYPES as PII_SPAN_TYPES
SAMPLE_SIZE = 10

# TWO GATES, EACH ON EXACTLY WHAT IT GUARDS. This file shipped once with
# neither and reported 6 failed / 9 errors against a healthy payload; it was
# then pulled from staging wholesale, which threw away the seven per-fixture
# PII-leakage and locked-span-recall runs that pass everywhere — the only
# end-to-end assertions of those properties a cloner had. Gate the pieces on
# their actual dependencies instead of deleting the file:
#
# 1. The metrics-gate classes read a results artifact produced by
#    ``benchmarks/wp7_real_runtime_eval.py``, an evaluator that is not
#    published. Without the file they must SKIP with that stated, not error
#    in setup.
# 2. Three of the seeded sample fixtures verify only on the resolving-anchor-
#    store (commercial) build; the open build's NonResolvingAnchorStore
#    refuses strictly more by design (see tekmyra/anchor_interface.py). Those
#    three skip per-fixture on the open build; the rest run end-to-end
#    everywhere, and ``_sample_fixtures`` guarantees at least one of them is
#    a PII fixture, so the leakage and recall branch executes on both builds
#    rather than being claimed by a message and skipped by the draw.
_OPEN_BUILD = find_spec("tekmyra.anchor_store") is None
COMMERCIAL_ONLY_SAMPLES = frozenset({"gov-002", "pii-002", "rebal-legal-0132"})
OPEN_BUILD_PII_SAMPLE = "rebal-pii-0154"

requires_runtime_metrics = pytest.mark.skipif(
    not METRICS_PATH.exists(),
    reason=(
        "benchmarks/results/wp7_fast_track/real_runtime_synthetic_metrics.json "
        "absent — it is produced by benchmarks/wp7_real_runtime_eval.py, which "
        "is not published. The per-fixture pipeline safety assertions in this "
        "module still run without it."
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        pytest.fail(
            f"Missing artifact: {path}. Run "
            f"`python benchmarks/wp7_real_runtime_eval.py --seed 42` first."
        )
    with open(path) as fh:
        return json.load(fh)


def _is_pii_fixture(fixture: Fixture) -> bool:
    """Identify PII fixtures by expected route, tags, or protected spans."""
    if fixture.gold_output.expected_route == RouteClass.PII_REDACTION_BEFORE_COMPRESSION:
        return True
    if any(tag.lower().startswith("pii") for tag in fixture.tags):
        return True
    spans: list[ProtectedSpanGold] = []
    spans.extend(fixture.gold_output.must_preserve_spans or [])
    spans.extend(fixture.protected_spans or [])
    return any(span.span_type in PII_SPAN_TYPES for span in spans)


def _pii_leakage_count(fixture: Fixture, compressed_text: str) -> int:
    leaked = 0
    spans: list[Any] = []
    spans.extend(fixture.gold_output.must_preserve_spans or [])
    spans.extend(fixture.protected_spans or [])
    seen: set[str] = set()
    for span in spans:
        if span.span_type not in PII_SPAN_TYPES or not span.text:
            continue
        key = f"{span.span_type}:{span.text}"
        if key in seen:
            continue
        seen.add(key)
        if span.text in compressed_text:
            leaked += 1
    return leaked


def _locked_span_recall(response: Any) -> float:
    detected = response.preservation_report.locked_spans_detected
    preserved = response.preservation_report.locked_spans_preserved
    if detected == 0:
        return 1.0
    return preserved / detected


def _final_records(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the final pipeline record for each fixture."""
    hybrid_available = bool(metrics["metadata"].get("hybrid_router_available"))
    pipeline = "hybrid" if hybrid_available else "deterministic"
    return [r for r in metrics["per_fixture"] if r["pipeline"] == pipeline]


@pytest.fixture(scope="module")
def hybrid_router_info():
    """Attempt to load the hybrid router once per module."""
    try:
        router = HybridRouter()
        return {"available": True, "router": router, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "router": None, "error": f"{type(exc).__name__}: {exc}"}


def _sample_fixtures() -> list[Fixture]:
    fixtures = load_fixtures(FIXTURE_DIR)
    fixtures.sort(key=lambda f: f.fixture_id)
    random.seed(42)
    sample_size = min(SAMPLE_SIZE, len(fixtures))
    sample = random.sample(fixtures, sample_size)
    # THE PII BRANCH MUST EXECUTE ON BOTH BUILDS, WHICH THE SEED ALONE DOES
    # NOT GUARANTEE. Measured on the review pass: the seeded ten contained
    # exactly two PII fixtures and BOTH were commercial-only, so on the open
    # build — the only build a cloner has — the leakage and recall assertions
    # below never ran while the skip message said they did. A sample is a
    # coverage instrument; if it must cover a branch, that is a constraint on
    # the sample, not a hope about the seed.
    #
    # THE PINNED FIXTURE IS MEASURED, NOT THE FIRST SORTED. Probed against the
    # published bundle on the open build: nearly every PII fixture refuses
    # there — the PII routes lean on the anchor store, and refusing them is
    # the open build's documented posture — and exactly three completed both
    # pipelines (missing-reject_or_human_review-000, rebal-pii-0154,
    # rebal-pii-0155). Appending "the first PII fixture" would have appended
    # gov-000, which refuses, and shipped a red test as the fix for a dormant
    # one. If a future artifact changes this fixture's outcome, this test
    # failing IS the signal that the open build's PII behaviour moved.
    if not any(
        _is_pii_fixture(f) and f.fixture_id not in COMMERCIAL_ONLY_SAMPLES
        for f in sample
    ):
        sampled_ids = {f.fixture_id for f in sample}
        if OPEN_BUILD_PII_SAMPLE not in sampled_ids:
            sample.append(next(
                f for f in fixtures if f.fixture_id == OPEN_BUILD_PII_SAMPLE
            ))
    return sample


SAMPLE_FIXTURES = _sample_fixtures()


@pytest.mark.parametrize("fixture", SAMPLE_FIXTURES, ids=lambda f: f.fixture_id)
def test_real_runtime_pipeline_on_sample(fixture: Fixture, hybrid_router_info: dict[str, Any]):
    """Run deterministic + hybrid pipelines on a fixture sample and assert safety."""
    if _OPEN_BUILD and fixture.fixture_id in COMMERCIAL_ONLY_SAMPLES:
        pytest.skip(
            f"{fixture.fixture_id} verifies only with the resolving anchor "
            "store (commercial build); the open build's NonResolvingAnchorStore "
            "refuses it by design — a refusal, not a defect. The remaining "
            "sample fixtures still run end-to-end on both builds, and the "
            "sample deterministically includes at least one PII fixture that "
            "does, so the leakage and recall assertions execute here too."
        )
    deterministic_response = run_deterministic_pipeline(fixture)

    assert deterministic_response.compressed_segments, f"{fixture.fixture_id}: no compressed segments"
    assert deterministic_response.global_metrics.verifier_pass is True, (
        f"{fixture.fixture_id}: deterministic verifier_pass is False"
    )

    if _is_pii_fixture(fixture):
        det_text = deterministic_response.compressed_segments[0].compressed_content
        assert _locked_span_recall(deterministic_response) == 1.0, (
            f"{fixture.fixture_id}: deterministic locked-span recall < 1.0"
        )
        assert _pii_leakage_count(fixture, det_text) == 0, (
            f"{fixture.fixture_id}: deterministic PII leakage detected"
        )

    if hybrid_router_info["available"]:
        hybrid_response = run_hybrid_pipeline(
            fixture,
            router=hybrid_router_info["router"],
            safety_gate=SafetyGate(),
        )
        assert hybrid_response.compressed_segments, f"{fixture.fixture_id}: hybrid produced no segments"
        if _is_pii_fixture(fixture):
            hyb_text = hybrid_response.compressed_segments[0].compressed_content
            assert _locked_span_recall(hybrid_response) == 1.0, (
                f"{fixture.fixture_id}: hybrid locked-span recall < 1.0"
            )
            assert _pii_leakage_count(fixture, hyb_text) == 0, (
                f"{fixture.fixture_id}: hybrid PII leakage detected"
            )
    else:
        pytest.skip(f"HybridRouter unavailable: {hybrid_router_info['error']}")


@requires_runtime_metrics
class TestRealRuntimeRouteIntentGates:
    """Gate assertions over the runtime metrics artifact."""

    @pytest.fixture(scope="class")
    def metrics(self):
        return _load_json(METRICS_PATH)

    def test_pii_allowed_outcome_rate_is_one(self, metrics: dict[str, Any]):
        records = [r for r in _final_records(metrics) if r["is_pii"]]
        assert records, "No PII fixtures in runtime metrics"
        bad = [r for r in records if not r["allowed_outcome_match"]]
        assert not bad, (
            f"{len(bad)}/{len(records)} PII fixtures routed outside allowed_outcomes: "
            f"{[(r['fixture_id'], r['predicted_route'], r['allowed_outcomes']) for r in bad]}"
        )

    def test_pii_zero_lossy_routes(self, metrics: dict[str, Any]):
        records = [r for r in _final_records(metrics) if r["is_pii"]]
        lossy = [r for r in records if r["lossy_final_route"]]
        assert not lossy, (
            f"{len(lossy)}/{len(records)} PII fixtures used lossy routes: "
            f"{[(r['fixture_id'], r['predicted_route']) for r in lossy]}"
        )

    def test_pii_zero_leakage(self, metrics: dict[str, Any]):
        records = [r for r in _final_records(metrics) if r["is_pii"]]
        leaks = [r for r in records if r["pii_leakage_count"] > 0]
        assert not leaks, f"{len(leaks)} PII fixture(s) leaked PII"

    def test_high_critical_zero_lossy_routes(self, metrics: dict[str, Any]):
        records = [r for r in _final_records(metrics) if r["is_high_critical"] and not r["is_pii"] and not r["is_ambiguous"]]
        assert records, "No high/critical fixtures in runtime metrics"
        lossy = [r for r in records if r["lossy_final_route"]]
        assert not lossy, (
            f"{len(lossy)}/{len(records)} high/critical fixtures used lossy routes: "
            f"{[(r['fixture_id'], r['predicted_route']) for r in lossy]}"
        )

    def test_high_critical_locked_span_recall_is_one(self, metrics: dict[str, Any]):
        records = [r for r in _final_records(metrics) if r["is_high_critical"] and not r["is_pii"] and not r["is_ambiguous"]]
        bad = [r for r in records if r["locked_span_recall"] != pytest.approx(1.0)]
        assert not bad, f"{len(bad)}/{len(records)} high/critical fixtures have locked-span recall < 1.0"

    def test_high_critical_risk_recall_at_least_95(self, metrics: dict[str, Any]):
        agg = metrics["aggregates"]["hybrid"]
        num = agg["high_risk_recall_numerator"]
        den = agg["high_risk_recall_denominator"]
        assert den > 0, "No high/critical fixtures in runtime metrics"
        recall = num / den
        assert recall >= 0.95, f"High/critical risk recall {recall:.4f} < 0.95"

    def test_normal_non_ambiguous_route_intent_accuracy_at_least_90(self, metrics: dict[str, Any]):
        agg = metrics["aggregates"]["hybrid"]
        count = agg["normal_non_ambiguous_count"]
        correct = agg["normal_non_ambiguous_correct_count"]
        assert count > 0, "No normal non-ambiguous fixtures in runtime metrics"
        accuracy = correct / count
        assert accuracy >= 0.90, (
            f"Normal non-ambiguous route-intent accuracy {accuracy:.4f} < 0.90 "
            f"({correct}/{count})"
        )

    def test_ambiguous_allowed_outcome_rate_is_one(self, metrics: dict[str, Any]):
        records = [r for r in _final_records(metrics) if r["is_ambiguous"]]
        if not records:
            pytest.skip("No ambiguous residue fixtures in synthetic runtime metrics")
        bad = [r for r in records if not r["allowed_outcome_match"]]
        assert not bad, (
            f"{len(bad)}/{len(records)} ambiguous fixtures routed outside allowed_outcomes: "
            f"{[(r['fixture_id'], r['predicted_route'], r['allowed_outcomes']) for r in bad]}"
        )

    def test_ambiguous_not_in_normal_accuracy(self, metrics: dict[str, Any]):
        records = _final_records(metrics)
        ambiguous_ids = {r["fixture_id"] for r in records if r["is_ambiguous"]}
        normal_ids = {r["fixture_id"] for r in records if r["is_normal_non_ambiguous"]}
        assert not (ambiguous_ids & normal_ids), "Ambiguous fixtures found in normal population"


@requires_runtime_metrics
class TestRealRuntimeNegativeCases:
    """Prove the runtime gate assertions are not vacuous."""

    @pytest.mark.parametrize(
        "mutation,expected_substring",
        [
            ("pii_disallowed", "allowed_outcomes"),
            ("high_critical_lossy", "high/critical fixtures"),
            ("normal_low_accuracy", "route-intent accuracy"),
        ],
    )
    def test_gate_fails_on_broken_metrics(self, mutation: str, expected_substring: str):
        metrics = _load_json(METRICS_PATH)
        records = _final_records(metrics)

        if mutation == "pii_disallowed":
            pii = [r for r in records if r["is_pii"]]
            assert pii, "Need at least one PII record for negative test"
            pii[0]["allowed_outcomes"] = ["reject_or_human_review"]
            pii[0]["allowed_outcome_match"] = False
        elif mutation == "high_critical_lossy":
            hc = [r for r in records if r["is_high_critical"] and not r["is_pii"] and not r["is_ambiguous"]]
            assert hc, "Need at least one high/critical record for negative test"
            hc[0]["predicted_route"] = "summarizer"
            hc[0]["lossy_final_route"] = True
        elif mutation == "normal_low_accuracy":
            normal = [r for r in records if r["is_normal_non_ambiguous"]]
            assert normal, "Need at least one normal record for negative test"
            for r in normal:
                r["route_intent_match"] = False
        else:
            raise ValueError(mutation)

        with pytest.raises(AssertionError) as exc_info:
            if mutation == "pii_disallowed":
                pii = [r for r in records if r["is_pii"]]
                bad = [r for r in pii if not r["allowed_outcome_match"]]
                assert not bad, (
                    f"{len(bad)}/{len(pii)} PII fixtures routed outside allowed_outcomes"
                )
            elif mutation == "high_critical_lossy":
                hc = [r for r in records if r["is_high_critical"] and not r["is_pii"] and not r["is_ambiguous"]]
                lossy = [r for r in hc if r["lossy_final_route"]]
                assert not lossy, (
                    f"{len(lossy)}/{len(hc)} high/critical fixtures used lossy routes"
                )
            elif mutation == "normal_low_accuracy":
                normal = [r for r in records if r["is_normal_non_ambiguous"]]
                correct = sum(r["route_intent_match"] for r in normal)
                accuracy = correct / len(normal)
                assert accuracy >= 0.90, (
                    f"normal non-ambiguous route-intent accuracy {accuracy:.4f} < 0.90"
                )

        assert expected_substring in str(exc_info.value)
