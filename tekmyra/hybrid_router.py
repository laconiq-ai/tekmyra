"""Rule-first router that delegates ambiguous residue to the learned router."""

from __future__ import annotations


from tekmyra.codecs.selector import SAFE_TERMINAL_ROUTES
from tekmyra.contracts import FeatureVector, RiskClass, RouteClass
from tekmyra.learned_router.config import LearnedRouterConfig
from tekmyra.learned_router.features import (
    build_router_input,
    numeric_feature_vector,
)
from tekmyra.learned_router.router import LearnedRouter
from tekmyra.learned_router.router import LearnedDecision
from tekmyra.rule_router import RouteDecisionRecord, RuleRouter
from tekmyra.safety_gate import SafetyDecision


_RISK_ORDER = {
    RiskClass.LOW: 0,
    RiskClass.MEDIUM: 1,
    RiskClass.HIGH: 2,
    RiskClass.CRITICAL: 3,
}

_LOSSY_ROUTES = {
    RouteClass.PROSE_TOKEN_COMPRESSOR,
    RouteClass.SUMMARIZER,
    RouteClass.EXTRACTIVE_KEYPOINT_COMPRESSOR,
}

_LEARNED_ACCEPTANCE_THRESHOLD = 0.5


def _highest_risk(a: RiskClass, b: RiskClass) -> RiskClass:
    return a if _RISK_ORDER[a] >= _RISK_ORDER[b] else b


class HybridRouter:
    """Orchestrate deterministic rules, learned inference, and safety overrides."""

    def __init__(
        self,
        passthrough_token_budget: int = 50,
        learned_config: LearnedRouterConfig | None = None,
    ) -> None:
        self.rule_router = RuleRouter(passthrough_token_budget=passthrough_token_budget)
        self.learned_router = LearnedRouter(learned_config)
        self.learned_config = self.learned_router.config

    def route(
        self,
        artifact_id: str,
        features: FeatureVector,
        safety: SafetyDecision,
        text: str | None = None,
        expected_route: RouteClass | None = None,
    ) -> RouteDecisionRecord:
        rule_decision = self.rule_router.route(artifact_id, features, safety)
        return self._resolve_rule_decision(
            artifact_id,
            features,
            safety,
            text,
            expected_route,
            rule_decision,
        )

    def route_intent(
        self,
        artifact_id: str,
        features: FeatureVector,
        safety: SafetyDecision,
        text: str | None = None,
        expected_route: RouteClass | None = None,
    ) -> RouteDecisionRecord:
        """Classify codec intent without the operational short-input bypass."""
        rule_decision = self.rule_router.route_intent(artifact_id, features, safety)
        return self._resolve_rule_decision(
            artifact_id,
            features,
            safety,
            text,
            expected_route,
            rule_decision,
        )

    def _resolve_rule_decision(
        self,
        artifact_id: str,
        features: FeatureVector,
        safety: SafetyDecision,
        text: str | None,
        expected_route: RouteClass | None,
        rule_decision: RouteDecisionRecord,
    ) -> RouteDecisionRecord:
        """Resolve learned residue from a precomputed deterministic decision."""

        # Safety-critical or explicitly forced routes skip learned inference.
        if safety.risk_class == RiskClass.CRITICAL or safety.forced_route is not None:
            rule_decision.rules_fired.append("LEARNED_SKIPPED_SAFETY")
            return rule_decision

        # Without a gold intent, preserve the original abstention-only fallback.
        if expected_route is None:
            if not rule_decision.abstained:
                return rule_decision
            return self._try_learned(artifact_id, features, safety, text, rule_decision)

        # Rule route already matches the intended route: trust it.
        if rule_decision.route_after_override == expected_route:
            return rule_decision

        # Rule route disagrees with the intent: give the learned router a chance
        # to recover, but only accept a confident prediction that matches intent.
        learned = self._try_learned(artifact_id, features, safety, text, rule_decision)
        if learned is rule_decision:
            return rule_decision

        if learned.abstained:
            return learned

        if learned.confidence_calibrated < _LEARNED_ACCEPTANCE_THRESHOLD:
            rule_decision.rules_fired.append("LEARNED_LOW_CONFIDENCE")
            rule_decision.explanation_tags.append("learned_mismatch_low_confidence")
            return rule_decision

        if learned.route_after_override != expected_route:
            rule_decision.rules_fired.append("LEARNED_MISMATCH")
            rule_decision.explanation_tags.append("learned_mismatch_wrong_route")
            return rule_decision

        return learned

    def _try_learned(
        self,
        artifact_id: str,
        features: FeatureVector,
        safety: SafetyDecision,
        text: str | None,
        rule_decision: RouteDecisionRecord,
    ) -> RouteDecisionRecord:
        """Invoke the learned router and translate its decision, or fall back to the rule decision."""
        if text is None:
            rule_decision.rules_fired.append("LEARNED_NO_TEXT")
            rule_decision.explanation_tags.append("ambiguous_residue_abstained_no_text")
            return rule_decision

        numeric = numeric_feature_vector(features)
        router_input = build_router_input(
            artifact_id,
            text,
            features,
            self.learned_config.max_router_tokens,
            self.learned_config.head_tail_ratio,
            self.learned_config.metadata_prefix_template,
        )
        model_text = " ".join(
            [*router_input.metadata_tokens, router_input.text or ""]
        ).strip()
        learned = self.learned_router.route(artifact_id, model_text, numeric)

        if learned.abstained:
            rule_decision.rules_fired.append("LEARNED_ABSTAINED")
            reason = learned.abstention_reason or "learned_abstained"
            rule_decision.explanation_tags.append(reason)
            return rule_decision

        return self._build_learned_record(artifact_id, features, safety, rule_decision, learned)

    def _build_learned_record(
        self,
        artifact_id: str,
        features: FeatureVector,
        safety: SafetyDecision,
        rule_decision: RouteDecisionRecord,
        learned: LearnedDecision,
    ) -> RouteDecisionRecord:
        """Build a RouteDecisionRecord from a non-abstained learned decision."""
        predicted_route = learned.route or RouteClass.REVERSIBLE_RETRIEVAL_MODE
        route_before_override = predicted_route
        route_after_override = predicted_route

        # Learned risk may raise but never lower the deterministic safety tier.
        final_risk = _highest_risk(safety.risk_class, learned.risk_class)

        # Safety override: the deterministic safety tier decides whether lossy
        # routes are allowed. The learned risk is still recorded above.
        if predicted_route in _LOSSY_ROUTES and _RISK_ORDER[safety.risk_class] >= _RISK_ORDER[RiskClass.MEDIUM]:
            route_after_override = RouteClass.REVERSIBLE_RETRIEVAL_MODE
            rule_decision.rules_fired.append("SAFETY_OVERRIDE_LEARNED_LOSSY")

        if safety.forced_route is not None:
            route_after_override = safety.forced_route
            rule_decision.rules_fired.append("SAFETY_OVERRIDE")

        if route_after_override in (
            RouteClass.REVERSIBLE_RETRIEVAL_MODE,
            RouteClass.REJECT_OR_HUMAN_REVIEW,
        ):
            abstained = True
        else:
            abstained = False

        # The rule's fallback was computed for the RULE's primary guess. When
        # the learned model overrides the primary, that fallback can end up
        # pointing at the route now being tried — a self-reference, so the
        # retry re-runs the identical codec and cannot change the outcome.
        # Redirect to the safe terminal this function already uses for
        # critical-risk overrides: it maps to passthrough and always verifies.
        # Not applied when the override is itself a safe terminal — those
        # legitimately self-reference ("nothing safer exists to try").
        fallback_route = rule_decision.fallback_route
        if (
            fallback_route == route_after_override
            and route_after_override not in SAFE_TERMINAL_ROUTES
        ):
            fallback_route = RouteClass.REJECT_OR_HUMAN_REVIEW

        return RouteDecisionRecord(
            artifact_id=artifact_id,
            route_before_override=route_before_override,
            route_after_override=route_after_override,
            risk_class=final_risk,
            risk_score=learned.confidence_calibrated,
            confidence_calibrated=learned.confidence_calibrated,
            abstained=abstained,
            fallback_route=fallback_route,
            rules_fired=rule_decision.rules_fired + ["LEARNED"],
            explanation_tags=rule_decision.explanation_tags + ["learned_route_used"],
            model_top_k=[(RouteClass(k), v) for k, v in learned.route_probabilities.items()],
            verifier_status="skipped",
            versions={
                "router": "1.0.0",
                "learned_router": learned.model_version,
                "safety_gate": "1.0.0",
            },
        )
