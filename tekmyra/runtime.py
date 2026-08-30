"""Deterministic and hybrid end-to-end runtime pipeline.

Stages: ingest envelope -> prober -> safety gate -> router (rule or hybrid) ->
safety merge -> codec -> verifier -> output assembly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from tekmyra.anchor_interface import (
    AnchorStoreProtocol as AnchorStore,
    default_anchor_store,
)
from tekmyra.contracts import (
    AnchorSummary,
    ArtifactEnvelope,
    ArtifactMetadata,
    CompressionEconomics,
    CompressionMetrics,
    CompressionResponse,
    CompressionSegment,
    FallbackLevel,
    FeatureVector,
    LatencyBreakdown,
    ObservedCompressionResult,
    PolicyTier,
    PreservationReport,
    RiskClass,
    RouteClass,
    SourceChannel,
    TrustTier,
)
from tekmyra.fixtures import Fixture
from tekmyra.protected_spans import build_fact_index
from tekmyra.codecs import apply_codec
from tekmyra.codecs.selector import SAFE_TERMINAL_ROUTES
from tekmyra.hybrid_router import HybridRouter
from tekmyra.prober import probe
from tekmyra.rule_router import RouteDecisionRecord, RuleRouter
from tekmyra.safety_gate import SafetyGate, SafetyDecision
from tekmyra.safety_merge import apply_safety_merge
from tekmyra.verifier import verify


# THE COMPRESSION TARGET — the share of content the compressor aims to KEEP.
#
# IMPORTED, NOT RESTATED, and the history is the argument. This constant used
# to be a bare 0.8 defined here, under a comment claiming it was "named here so
# the layers cannot drift". It drifted anyway: on 2026-08-09, landing Hari's
# ruling of 0.15 found the same value written at EIGHT sites across runtime,
# both codec selectors and the compressor. Changing any one of them would have
# looked like landing the ruling while the other seven silently enforced the
# old point.
#
# Naming a value in one layer does not make it the definition; only the other
# layers importing it does. The definition lives in ``compression_budget``, a
# LEAF that imports nothing from the package — the compressor was the obvious
# home and produced a real circular import, which is the same reason
# ``numeric_fidelity`` and ``entity_fidelity`` are leaves.
from tekmyra.compression_budget import (
    DEFAULT_BUDGET_RATIO as _DEFAULT_BUDGET_RATIO,
)


class TerminalVerificationError(RuntimeError):
    """Raised when both the selected route and its safe fallback fail verification.

    Carries WHICH route refused and WHICH checks failed. Without them a refusal
    is a bare tally that cannot be attributed to a route, so a per-route report
    would show compression rates beside a refusal count floating free of the
    table — the same defect as a mean without its distribution, one level down.
    The message stays the first positional argument so existing ``str(exc)``
    handling is unchanged.
    """

    def __init__(
        self,
        message: str,
        *,
        route: RouteClass | None = None,
        fallback_route: RouteClass | None = None,
        failed_checks: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.route = route
        self.fallback_route = fallback_route
        self.failed_checks = tuple(failed_checks)


def fixture_to_envelope(fixture: Fixture) -> ArtifactEnvelope:
    """Build a synthetic ArtifactEnvelope from a benchmark fixture."""
    policy_tier_map = {
        RiskClass.LOW: PolicyTier.STANDARD,
        RiskClass.MEDIUM: PolicyTier.STANDARD,
        RiskClass.HIGH: PolicyTier.REGULATED,
        RiskClass.CRITICAL: PolicyTier.CRITICAL,
    }
    return ArtifactEnvelope(
        artifact_id=fixture.fixture_id,
        tenant_id=fixture.tenant_id,
        source_channel=SourceChannel.BATCH,
        mime_type=f"text/{fixture.content_type}",
        filename=f"{fixture.fixture_id}.txt",
        text_content=fixture.content,
        metadata=ArtifactMetadata(
            created_at=datetime.now(timezone.utc),
            user_id=None,
            trust_tier=TrustTier.HIGH,
            policy_tier=policy_tier_map.get(fixture.risk_tier, PolicyTier.STANDARD),
        ),
        bytes_=None,
    )


def _route_decision(
    router: RuleRouter | HybridRouter,
    fixture: Fixture,
    features: FeatureVector,
    safety: SafetyDecision,
) -> RouteDecisionRecord:
    """Call the router with the right signature for its type."""
    if isinstance(router, HybridRouter):
        return router.route(
            fixture.fixture_id,
            features,
            safety,
            text=fixture.content,
            expected_route=fixture.gold_output.route_intent,
        )
    return router.route(fixture.fixture_id, features, safety)


def _run_pipeline(
    fixture: Fixture,
    router: RuleRouter | HybridRouter,
    safety_gate: SafetyGate,
    anchor_store: AnchorStore,
    *,
    envelope: ArtifactEnvelope | None = None,
    use_expected_route: bool = True,
    budget_ratio: float = _DEFAULT_BUDGET_RATIO,
) -> CompressionResponse:
    """Shared end-to-end pipeline used by deterministic and hybrid runners."""
    envelope = envelope or fixture_to_envelope(fixture)
    if envelope.tenant_id != fixture.tenant_id:
        raise ValueError("tenant_context_mismatch")
    features = probe(envelope)
    safety = safety_gate.classify(features)
    if isinstance(router, HybridRouter) and not use_expected_route:
        decision = router.route(
            fixture.fixture_id,
            features,
            safety,
            text=fixture.content,
            expected_route=None,
        )
    else:
        decision = _route_decision(router, fixture, features, safety)

    segment_id = f"{fixture.fixture_id}-seg-0"
    merge_result = apply_safety_merge(
        text=fixture.content,
        risk_class=safety.risk_class,
        tenant_id=envelope.tenant_id,
        source_id=segment_id,
        anchor_store=anchor_store,
        additional_spans=fixture.protected_spans,
        doc_id=fixture.fixture_id,
        chunk_id="0",
    )

    fallback_route: RouteClass | None
    if merge_result.human_review_required:
        active_route = RouteClass.REJECT_OR_HUMAN_REVIEW
        fallback_route = RouteClass.REJECT_OR_HUMAN_REVIEW
    else:
        active_route = decision.route_after_override
        fallback_route = decision.fallback_route
        # Chokepoint guard: a fallback equal to the primary makes the retry
        # below (see the `not verifier_result.passed` branch) re-run the same
        # codec on the same input — a second codec+verify cycle that cannot
        # change the verdict. Guarding here covers any future decision source
        # without requiring it to remember.
        #
        # Deliberately NOT applied when the primary is already a safe
        # terminal: R_MIXED_SAFE, R_AGENT_TRACE_SAFE and R_SHORT_SAFE
        # legitimately return PASSTHROUGH -> PASSTHROUGH, meaning "nothing
        # safer exists to try". Redirecting those would rewrite the recorded
        # route for benign content without changing a single output byte.
        if (
            fallback_route is not None
            and fallback_route == active_route
            and active_route not in SAFE_TERMINAL_ROUTES
        ):
            fallback_route = RouteClass.REJECT_OR_HUMAN_REVIEW

    locked_spans = [span.model_dump() for span in merge_result.locked_spans]

    # Inject fixture-level protected_spans (must-preserve) so the codec never
    # drops them, even if safety_merge replaced them with anchor IDs.
    fixture_protected: list[dict[str, object]] = [
        {
            "text": span.text,
            "span_type": span.span_type,
            "policy": span.policy,
            "detector": span.detector,
            "confidence": span.confidence,
            "span_id": span.span_id,
            "fact_id": span.fact_id,
        }
        for span in fixture.protected_spans
        if span.text
    ]

    fact_index = build_fact_index(fixture)

    codec_result = apply_codec(
        merge_result.transformed_text,
        active_route,
        features,
        locked_spans=locked_spans,
        safety_tier=safety.risk_class,
        protected_spans=fixture_protected,
        budget_ratio=budget_ratio,
    )
    learned_compressor_invoked = codec_result.learned_compressor_invoked

    verifier_result = verify(
        fixture.content,
        codec_result,
        active_route,
        features,
        locked_spans,
        anchor_ids=merge_result.anchor_ids,
        anchor_store=anchor_store,
        fact_index=fact_index,
        anchor_records=merge_result.anchor_records,
    )

    # ROUTED HERE, OR FELL BACK HERE? The row could not tell them apart.
    #
    # `route_after_override` is OVERWRITTEN with the fallback below, so a
    # fixture that was sent to reversible retrieval and a fixture that tried the
    # compressor and failed into it recorded IDENTICALLY. Two different facts
    # with opposite remedies — one is a routing decision, the other a
    # verification failure — and Vega and I each read the same rows and reached
    # confidently opposite conclusions about the largest population in the
    # corpus. Neither of us could have been shown wrong by the artifact.
    attempted_route = active_route
    fallback_checks: tuple[str, ...] = ()
    if not verifier_result.passed:
        fallback_checks = tuple(
            check.name for check in verifier_result.checks if not check.passed
        )

    if not verifier_result.passed and fallback_route is not None:
        codec_result = apply_codec(
            merge_result.transformed_text,
            fallback_route,
            features,
            locked_spans=locked_spans,
            safety_tier=safety.risk_class,
            protected_spans=fixture_protected,
            budget_ratio=budget_ratio,
        )
        learned_compressor_invoked = (
            learned_compressor_invoked
            or codec_result.learned_compressor_invoked
        )
        active_route = fallback_route
        verifier_result = verify(
            fixture.content,
            codec_result,
            active_route,
            features,
            locked_spans,
            anchor_ids=merge_result.anchor_ids,
            anchor_store=anchor_store,
            fact_index=fact_index,
            anchor_records=merge_result.anchor_records,
        )

    if not verifier_result.passed:
        # A failed fallback is terminal. Never construct or return a response
        # containing output which the verifier rejected.
        raise TerminalVerificationError(
            "verification_failed_after_fallback",
            route=active_route,
            fallback_route=fallback_route,
            failed_checks=tuple(
                check.name for check in verifier_result.checks if not check.passed
            ),
        )

    # Always reflect the final active_route on the decision, whether or not a
    # fallback retry happened.
    decision.route_after_override = active_route

    compressed_content = codec_result.output
    ratio = codec_result.ratio

    if merge_result.human_review_required:
        compressed_content = f"[[HUMAN_REVIEW_REQUIRED]] {segment_id}"

    tokens = features.token_len
    output_tokens = max(1, int(tokens * ratio))

    mode_map: dict[
        RouteClass,
        Literal["passthrough", "deterministic", "learned", "hybrid", "reversible", "human_review"],
    ] = {
        RouteClass.PASSTHROUGH: "passthrough",
        RouteClass.DETERMINISTIC_JSON_COMPRESSOR: "deterministic",
        RouteClass.CODE_COMPRESSOR: "deterministic",
        RouteClass.LOG_COMPRESSOR: "deterministic",
        RouteClass.DIFF_COMPRESSOR: "deterministic",
        RouteClass.PROSE_TOKEN_COMPRESSOR: "deterministic",
        RouteClass.PII_REDACTION_BEFORE_COMPRESSION: "deterministic",
        RouteClass.GOVERNANCE_SAFE_COMPRESSION: "deterministic",
        RouteClass.LEGAL_SECURITY_SAFE_MODE: "deterministic",
        RouteClass.REVERSIBLE_RETRIEVAL_MODE: "reversible",
        RouteClass.REJECT_OR_HUMAN_REVIEW: "human_review",
    }

    fallback_map = {
        RouteClass.REVERSIBLE_RETRIEVAL_MODE: FallbackLevel.DETERMINISTIC_ONLY_COMPRESSION,
        RouteClass.REJECT_OR_HUMAN_REVIEW: FallbackLevel.REJECT_OR_HUMAN_REVIEW,
        RouteClass.PASSTHROUGH: FallbackLevel.BYPASS_ORIGINAL,
    }

    segment = CompressionSegment(
        segment_id=segment_id,
        original_content=fixture.content,
        compressed_content=compressed_content,
        content_type=fixture.content_type,
        mode=(
            "learned"
            if learned_compressor_invoked
            else mode_map.get(active_route, "deterministic")
        ),
        fallback_level=fallback_map.get(active_route, FallbackLevel.NORMAL_COMPRESSION),
        locked_span_count=merge_result.locked_span_count,
        anchor_ids=merge_result.anchor_ids,
        compression_ratio=ratio,
    )

    versions = {"router": "1.0.0", "safety_gate": "1.0.0", "prober": "1.0.0"}
    versions.update(decision.versions)

    anchor_map = [
        AnchorSummary(
            anchor_id=rec["anchor_id"],
            segment_id=rec["segment_id"],
            offset_start=rec["offset_start"],
            offset_end=rec["offset_end"],
            recovery_ref=rec["recovery_ref"],
            content_hash=rec["content_hash"],
            content_type=rec["content_type"],
        )
        for rec in merge_result.anchor_records
    ]

    return CompressionResponse(
        request_id=f"req-{fixture.fixture_id}",
        compressed_segments=[segment],
        global_metrics=CompressionMetrics(
            input_tokens=tokens,
            output_tokens=output_tokens,
            compression_ratio=ratio,
            latency_ms=LatencyBreakdown(detect=0.0, route=0.0, model=0.0, verify=0.0, anchor=0.0, total=0.0),
            verifier_pass=verifier_result.passed,
            fail_safe_triggered=(
                decision.abstained
                or safety.forced_route is not None
                or decision.route_after_override == RouteClass.REJECT_OR_HUMAN_REVIEW
                or active_route == RouteClass.REJECT_OR_HUMAN_REVIEW
                or not verifier_result.passed
            ),
        ),
        preservation_report=PreservationReport(
            locked_spans_detected=merge_result.locked_span_count,
            locked_spans_preserved=merge_result.locked_span_count - verifier_result.locked_spans_missing,
            citations_preserved=sum(1 for rec in merge_result.anchor_records if rec["content_type"] == "citation"),
            numeric_mutations=verifier_result.numeric_mutations,
            policy_control_id_losses=0,
        ),
        observed_result=ObservedCompressionResult(
            observed_eligibility_decision=True,
            selected_route=active_route,
            learned_compressor_invoked=learned_compressor_invoked,
            economics=CompressionEconomics(),
            protected_disposition_valid=verifier_result.protected_disposition_valid,
            introduced_duplicate_count=verifier_result.introduced_duplicate_count,
            required_answer_fact_recall=0.0,
        ),
        anchor_map=anchor_map,
        explanations=[
            {
                "rules_fired": decision.rules_fired,
                "explanation_tags": decision.explanation_tags,
                "route_after_override": decision.route_after_override.value,
                "fallback_route": decision.fallback_route.value if decision.fallback_route else None,
                "risk_class": decision.risk_class.value,
                "risk_score": decision.risk_score,
                "abstained": decision.abstained,
                "model_top_k": [(r.value, s) for r, s in decision.model_top_k],
                # DID THE CODEC ACTUALLY TRANSFORM ANYTHING?
                #
                # The routes in `SAFE_TERMINAL_ROUTES` share one `_passthrough`
                # codec that returns its input verbatim, and the prose codec
                # ALSO returns its input untouched on HIGH/CRITICAL tiers.
                # (This read "six routes" until 2026-08-12; the answer is five.
                # The count moved when LEGAL_SECURITY_SAFE_MODE was built and
                # three copies of the sentence did not move with it.
                # `SAFE_TERMINAL_ROUTES` is DERIVED from the codec map, so
                # naming it cannot go stale the way a written number does.)
                # So a route name does not
                # tell you whether compression happened, and a per-route token
                # delta is not evidence that it did: the safety merge shortens
                # text on its own, and that delta lands on whatever route the
                # fixture happened to take. Measured on long_context_v1, two
                # routes reported non-zero token reductions while their codec
                # was a no-op on every fixture — the movement was anchorization,
                # credited to compression.
                #
                # Comparing the codec's own input and output is the only thing
                # that separates the two, and it is knowable exactly here.
                # Both apply_codec call sites (initial and fallback) are fed
                # merge_result.transformed_text, so it is the codec's input on
                # either path.
                "codec_transformed": (
                    codec_result.output != merge_result.transformed_text
                ),
                # See the attempted_route comment above. `route_after_override`
                # reports where the fixture ENDED; these report how it got
                # there, which is the difference between a routing problem and
                # a verification problem.
                "attempted_route": attempted_route.value,
                "realised_route": active_route.value,
                "fell_back": attempted_route != active_route,
                "fallback_failed_checks": list(fallback_checks),
                # THE BUDGET ASKED FOR IS NOT THE BUDGET ACHIEVED, AND THE ROW
                # HAS TO SAY SO (Vega ruling 47, enforced by ruling 49).
                #
                # The producer backs its budget off per fixture when locked
                # spans saturate it, so "budget 0.05" describes a REQUEST. On
                # long_context_v1 not one fixture reaches the request at either
                # 0.30 or 0.05 — a request of 0.05 achieves a median of 0.236.
                # A reader given only the requested figure would conclude we
                # compressed to five percent. Reporting the request alone is
                # the same provenance defect this pipeline has now fixed three
                # times, and it would be sitting under the best number we have.
                "budget_requested": codec_result.budget_requested,
                "budget_achieved": codec_result.budget_achieved,
                # AND WHETHER THE FLOOR IS STILL BINDING. Zero fallbacks means
                # the entity check never fails, and a check that cannot fail is
                # indistinguishable from an absent one UNLESS it is binding
                # preventively rather than by rejection. These fields are what
                # tell those apart, so they travel with the row rather than
                # being recoverable only by instrumenting the selector.
                **{
                    key: (codec_result.selection_diagnostics or {}).get(key)
                    for key in (
                        "budget_backoff_steps",
                        "entity_floor_demand",
                        "entity_floor_granted",
                        "entity_floor_starved",
                        "entity_floor_starved_gross_legacy",
                        "entity_floor_unmet",
                        "floor_cap",
                    )
                },
            }
        ],
    )


def run_deterministic_pipeline(
    fixture: Fixture,
    router: RuleRouter | None = None,
    safety_gate: SafetyGate | None = None,
    anchor_store: AnchorStore | None = None,
) -> CompressionResponse:
    """Run the Phase 3 deterministic pipeline on a fixture."""
    router = router or RuleRouter()
    safety_gate = safety_gate or SafetyGate()
    anchor_store = anchor_store or default_anchor_store()
    return _run_pipeline(fixture, router, safety_gate, anchor_store)


def run_hybrid_pipeline(
    fixture: Fixture,
    router: HybridRouter | None = None,
    safety_gate: SafetyGate | None = None,
    anchor_store: AnchorStore | None = None,
    budget_ratio: float = _DEFAULT_BUDGET_RATIO,
    use_expected_route: bool = True,
) -> CompressionResponse:
    """Run the Phase 4 hybrid pipeline on a fixture.

    ``budget_ratio`` is the compression target — the share of content the
    compressor aims to KEEP. It was previously unreachable from here, so every
    harness we own measured at a default nobody had chosen and no published
    figure could name the setting it was taken at.

    ``use_expected_route`` decides whether the router is TOLD THE ANSWER. When
    True the pipeline hands it ``fixture.gold_output.route_intent``, a label the
    production path never has. The switch existed in ``_run_pipeline`` and was
    unreachable from here — the same shape as ``budget_ratio``, and with a
    larger consequence, because it means every routing measurement taken
    through this entry point was taken with the gold label in play.
    """
    router = router or HybridRouter()
    safety_gate = safety_gate or SafetyGate()
    anchor_store = anchor_store or default_anchor_store()
    return _run_pipeline(
        fixture,
        router,
        safety_gate,
        anchor_store,
        use_expected_route=use_expected_route,
        budget_ratio=budget_ratio,
    )


def run_live_pipeline(
    fixture: Fixture,
    envelope: ArtifactEnvelope,
    router: HybridRouter | None = None,
    safety_gate: SafetyGate | None = None,
    anchor_store: AnchorStore | None = None,
    budget_ratio: float = _DEFAULT_BUDGET_RATIO,
) -> CompressionResponse:
    """Run live traffic without benchmark gold-route leakage.

    The caller supplies the real tenant, channel and policy in ``envelope``.
    ``fixture`` carries only the text and protected spans required by the
    existing safety/verifier pipeline; its gold route is deliberately ignored.
    """
    router = router or HybridRouter()
    safety_gate = safety_gate or SafetyGate()
    anchor_store = anchor_store or default_anchor_store()
    return _run_pipeline(
        fixture,
        router,
        safety_gate,
        anchor_store,
        envelope=envelope,
        use_expected_route=False,
        budget_ratio=budget_ratio,
    )
