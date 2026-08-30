"""Tests for the hybrid rule + learned router."""

from __future__ import annotations


from tekmyra.contracts import LearnedDecision, PolicyTier, RiskClass, RouteClass, SourceChannel
from tekmyra.hybrid_router import HybridRouter
from tekmyra.prober import probe_text
from tekmyra.safety_gate import SafetyGate


class _FakeLearnedRouter:
    """Deterministic learned-router stand-in for unit tests."""

    def __init__(self, predicted_route: RouteClass | None, confidence: float, abstained: bool = False) -> None:
        self.predicted_route = predicted_route
        self.confidence = confidence
        self.abstained = abstained

    def route(self, artifact_id: str, text: str, numeric: list[float]) -> LearnedDecision:
        if self.abstained:
            return LearnedDecision(
                artifact_id=artifact_id,
                abstained=True,
                abstention_reason="test_abstained",
                confidence_calibrated=0.0,
                model_version="fake",
            )
        route_value = self.predicted_route.value if self.predicted_route else RouteClass.REVERSIBLE_RETRIEVAL_MODE.value
        return LearnedDecision(
            artifact_id=artifact_id,
            route=self.predicted_route,
            abstained=False,
            confidence_calibrated=self.confidence,
            route_probabilities={route_value: self.confidence},
            risk_class=RiskClass.LOW,
            model_version="fake",
        )


def test_deterministic_json_still_routes_to_json():
    text = '{"a": 1}'
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    router = HybridRouter()
    decision = router.route("a", features, safety)
    assert decision.route_after_override == RouteClass.DETERMINISTIC_JSON_COMPRESSOR
    assert decision.confidence_calibrated == 1.0
    assert "LEARNED" not in decision.rules_fired


def test_ambiguous_prose_abstains_without_model(tmp_path):
    text = "Some arbitrary text that does not match any deterministic rule."
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    from tekmyra.learned_router.config import LearnedRouterConfig

    router = HybridRouter(
        passthrough_token_budget=5,
        learned_config=LearnedRouterConfig(model_artifact_dir=tmp_path),
    )
    decision = router.route("a", features, safety, text=text)
    assert decision.abstained is True
    assert "LEARNED_ABSTAINED" in decision.rules_fired


def test_safety_override_skips_learned_for_critical():
    text = "harmless"
    features = probe_text(text, SourceChannel.API, PolicyTier.CRITICAL)
    safety = SafetyGate().classify(features)
    router = HybridRouter()
    decision = router.route("a", features, safety, text=text)
    assert decision.route_after_override == RouteClass.REJECT_OR_HUMAN_REVIEW
    assert "LEARNED" not in decision.rules_fired


def test_learned_risk_cannot_de_escalate():
    text = "Customer SSN: 123-45-6789"
    features = probe_text(text, SourceChannel.CHAT, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    router = HybridRouter()
    decision = router.route("a", features, safety, text=text)
    # Safety gate forces PII redaction; learned must not override to prose.
    assert decision.route_after_override in {
        RouteClass.PII_REDACTION_BEFORE_COMPRESSION,
        RouteClass.REJECT_OR_HUMAN_REVIEW,
    }


def test_expected_route_match_uses_rule_decision():
    text = '{"a": 1}'
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    router = HybridRouter()
    decision = router.route(
        "a", features, safety, expected_route=RouteClass.DETERMINISTIC_JSON_COMPRESSOR
    )
    assert decision.route_after_override == RouteClass.DETERMINISTIC_JSON_COMPRESSOR
    assert decision.confidence_calibrated == 1.0
    assert "LEARNED" not in decision.rules_fired


def test_expected_route_mismatch_invokes_learned_when_confident():
    text = "This is a short but clearly prose test sentence."
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    router = HybridRouter()
    router.learned_router = _FakeLearnedRouter(RouteClass.PROSE_TOKEN_COMPRESSOR, 0.9)
    decision = router.route(
        "a", features, safety, text=text, expected_route=RouteClass.PROSE_TOKEN_COMPRESSOR
    )
    assert decision.route_after_override == RouteClass.PROSE_TOKEN_COMPRESSOR
    assert "LEARNED" in decision.rules_fired
    assert decision.confidence_calibrated == 0.9


def test_expected_route_mismatch_falls_back_on_low_confidence():
    text = "This is a short but clearly prose test sentence."
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    router = HybridRouter()
    router.learned_router = _FakeLearnedRouter(RouteClass.PROSE_TOKEN_COMPRESSOR, 0.3)
    decision = router.route(
        "a", features, safety, text=text, expected_route=RouteClass.PROSE_TOKEN_COMPRESSOR
    )
    # Rule router fires R_SHORT_SAFE -> passthrough; learned is too uncertain.
    assert decision.route_after_override == RouteClass.PASSTHROUGH
    assert decision.confidence_calibrated == 1.0
    assert "LEARNED" not in decision.rules_fired
    assert "LEARNED_LOW_CONFIDENCE" in decision.rules_fired


def test_route_intent_separates_short_bypass_from_codec_classification():
    text = "This short input is declared prose."
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    features = features.model_copy(update={"content_type": "prose"})
    safety = SafetyGate().classify(features)
    router = HybridRouter()

    execution = router.route("a", features, safety)
    intent = router.route_intent(
        "a",
        features,
        safety,
        expected_route=RouteClass.PROSE_TOKEN_COMPRESSOR,
    )

    assert execution.route_after_override == RouteClass.PASSTHROUGH
    assert intent.route_before_override == RouteClass.PROSE_TOKEN_COMPRESSOR
