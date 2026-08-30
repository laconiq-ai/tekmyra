"""Fallback routes must never self-reference.

A fallback equal to the primary route makes the runtime's retry re-run the
identical codec on identical input: a second codec+verify cycle that cannot
change the verdict, and silent degradation for any route whose fallback ends
up pointing at itself.

The defect is in ``hybrid_router.py``'s ``_build_learned_record``, which
carried the rule's fallback through unchanged when the learned model overrode
the primary route, so an override landing on the rule's old fallback
self-referenced.

Measured correction to the ruling's premise: the rule table is NOT free of
self-reference. 3 of its 16 literal branches (R_MIXED_SAFE,
R_AGENT_TRACE_SAFE, R_SHORT_SAFE) return PASSTHROUGH -> PASSTHROUGH. Those
are legitimate — passthrough never transforms content, so nothing safer
exists to try — which is why the guards below exempt safe terminals rather
than rewriting the recorded route for benign content. The property worth
enforcing is therefore not "fallback != primary" but "a self-referential
fallback only ever on a route that cannot fail by transforming".

Ruling: Vega, ``laconiq_docs/Vega_CTRL205_Fallback_Self_Reference_Ruling_20260803.md``.
"""

from __future__ import annotations

import inspect
from dataclasses import replace

from tekmyra import rule_router as rule_router_module
from tekmyra.codecs.selector import SAFE_TERMINAL_ROUTES
from tekmyra.contracts import RiskClass, RouteClass


def _require_a_real_compressor() -> None:
    """Skip when the learned compressor artifact is absent, and SAY SO.

    ``load_compressor`` resolves ``_ARTIFACT_DIR = Path(".cache/compressor/1.0.0")``
    — a RELATIVE path into a gitignored directory. Where the artifact exists the
    prose route compresses and can fail verification; where it does not,
    ``ProseCompressor(None)`` returns a fail-closed passthrough at ratio 1.0, and
    ``_check_repeated_entity_retention`` short-circuits on ``length_ratio >= 1.0``.
    Verification then passes, the retry never fires, and the test below has no
    subject.

    This is an EXPLICIT skip rather than a silent pass because the condition is
    worth seeing on every run: an environment without the artifact is not
    exercising the learned compressor at all, so a green suite there says nothing
    about compression behaviour.
    """
    import pytest

    from tekmyra.learned_compressor import load_compressor

    if load_compressor()._model is None:
        pytest.skip(
            "learned compressor artifact absent (.cache/compressor/1.0.0) — the "
            "prose route is a fail-closed passthrough here, so primary "
            "verification cannot fail and the retry this test measures cannot "
            "fire. This environment does not exercise the learned compressor."
        )


def _a_long_prose_fixture():
    """A realistic long-prose fixture. Its dirtiness is INJECTED, not intrinsic.

    THE NAME AND DOCSTRING USED TO CLAIM OTHERWISE and by 2026-08-08 the claim
    was false. It read "this one fails repeated_entity_retention on the prose
    route, so the retry genuinely fires" — true when written, and then the
    caller was rewritten to FORCE the first verdict to fail (see the comment at
    the call site, which explains the CI break that motivated it), and the
    producer-side entity floor at 187e787/9b03bcc made this fixture verify
    cleanly at every budget. Measured: primary verification passes with no
    failed checks at 0.8 and at 0.3.

    So the test is NOT vacuous — the forcing guarantees the retry fires — but a
    reader who found this helper would have believed the fixture supplied the
    dirtiness, and would have "repaired" the wrong thing when it next changed.
    A helper named for a property it no longer has is the same defect class as a
    comment asserting an invariant nobody tests, and this codebase has removed
    that one three times this week.

    Requires the learned compressor artifact; see
    ``_require_a_real_compressor``.
    """
    from pathlib import Path

    from tekmyra.fixtures import load_fixtures

    root = Path(__file__).resolve().parent.parent
    for fixture in load_fixtures(root / "benchmarks/fixtures/long_context_v1"):
        if fixture.fixture_id == "lc-long_prose-primary_10k_plus-000":
            return fixture
    raise AssertionError("expected long-prose fixture not found")


def _rule_table_pairs() -> list[tuple[int, str, str]]:
    """Extract (lineno, primary, fallback) for every literal rule-table branch.

    ``_classify_by_content`` returns ``(route, rule_name, fallback_route)``, so
    the RouteClass members are the first and last elements of the tuple.
    """
    import ast
    import textwrap

    source = textwrap.dedent(inspect.getsource(rule_router_module.RuleRouter._classify_by_content))
    pairs: list[tuple[int, str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Tuple):
            continue
        names = [
            element.attr
            for element in node.value.elts
            if isinstance(element, ast.Attribute)
            and isinstance(element.value, ast.Name)
            and element.value.id == "RouteClass"
        ]
        if len(names) >= 2:
            pairs.append((node.lineno, names[0], names[-1]))
    return pairs


def test_rule_table_self_reference_only_ever_on_a_safe_terminal() -> None:
    """Pin the rule table's self-reference profile.

    Three branches (R_MIXED_SAFE, R_AGENT_TRACE_SAFE, R_SHORT_SAFE) return
    PASSTHROUGH -> PASSTHROUGH, which is legitimate: passthrough never
    transforms content, so there is nothing safer to fall back to. What must
    never appear is a self-referential *compressing* route, where the retry
    re-runs a real codec that already failed.
    """
    pairs = _rule_table_pairs()
    assert pairs, "failed to parse any rule-table branches — test is vacuous"

    safe = {r.name for r in SAFE_TERMINAL_ROUTES}
    offenders = [
        f"line {lineno}: RouteClass.{primary} -> itself"
        for lineno, primary, fallback in pairs
        if primary == fallback and primary not in safe
    ]
    assert offenders == [], (
        "rule-table branches that self-reference a transforming route: "
        + "; ".join(offenders)
    )


def test_rule_table_self_reference_count_is_pinned() -> None:
    """Lock the known-good exceptions so a new one has to be deliberate."""
    pairs = _rule_table_pairs()
    self_ref = {primary for _, primary, fallback in pairs if primary == fallback}
    assert self_ref == {"PASSTHROUGH"}, (
        f"unexpected self-referential routes in the rule table: {sorted(self_ref)}"
    )


def test_learned_override_never_leaves_a_self_referential_fallback() -> None:
    """The actual defect, exercised — not asserted against source text.

    Drives the real _build_learned_record with a rule decision whose fallback
    equals the route the learned model predicts. A source-string check would
    pass against a guard disabled with `and False`; this will not.
    """
    from tekmyra.contracts import LearnedDecision, PolicyTier, SourceChannel
    from tekmyra.hybrid_router import HybridRouter
    from tekmyra.prober import probe_text
    from tekmyra.rule_router import RouteDecisionRecord
    from tekmyra.safety_gate import SafetyDecision

    router = HybridRouter()
    # Rule guessed JSON with a PROSE_TOKEN_COMPRESSOR fallback; the learned
    # model then overrides the primary to exactly that fallback.
    rule_decision = RouteDecisionRecord(
        artifact_id="a",
        route_before_override=RouteClass.DETERMINISTIC_JSON_COMPRESSOR,
        route_after_override=RouteClass.DETERMINISTIC_JSON_COMPRESSOR,
        risk_class=RiskClass.LOW,
        risk_score=0.1,
        confidence_calibrated=0.9,
        abstained=False,
        fallback_route=RouteClass.PROSE_TOKEN_COMPRESSOR,
        rules_fired=["R_JSON_VALID"],
        explanation_tags=[],
        model_top_k=[],
        verifier_status="skipped",
        versions={},
    )
    learned = LearnedDecision(
        artifact_id="a",
        route=RouteClass.PROSE_TOKEN_COMPRESSOR,
        risk_class=RiskClass.LOW,
        confidence_calibrated=0.9,
        abstained=False,
        model_version="test",
    )
    safety = SafetyDecision(
        risk_class=RiskClass.LOW, risk_score=0.1, forced_route=None, explanation_tags=[]
    )
    features = probe_text("word " * 400, SourceChannel.API, PolicyTier.STANDARD, "long_prose")

    record = router._build_learned_record(
        "a", features, safety, rule_decision, learned
    )
    assert record.route_after_override == RouteClass.PROSE_TOKEN_COMPRESSOR
    assert record.fallback_route != record.route_after_override, (
        "learned override left a self-referential fallback"
    )
    assert record.fallback_route == RouteClass.REJECT_OR_HUMAN_REVIEW


def test_learned_override_self_reference_property_holds_for_every_route() -> None:
    """The STRUCTURAL property, over the whole route space — not one instance.

    CTRL-205 is a latent defect found by code inspection: there are no known
    triggering inputs. So the guarantee worth pinning is universal — for every
    route the learned model could predict, a rule fallback equal to it must
    never survive onto the record. Exhaustive over RouteClass rather than
    example-based, because an example-based test proves nothing about the
    cases nobody thought of.
    """
    from tekmyra.contracts import LearnedDecision, PolicyTier, SourceChannel
    from tekmyra.hybrid_router import HybridRouter
    from tekmyra.prober import probe_text
    from tekmyra.rule_router import RouteDecisionRecord
    from tekmyra.safety_gate import SafetyDecision

    router = HybridRouter()
    features = probe_text("word " * 400, SourceChannel.API, PolicyTier.STANDARD, "long_prose")
    safety = SafetyDecision(
        risk_class=RiskClass.LOW, risk_score=0.1, forced_route=None, explanation_tags=[]
    )

    checked = 0
    for predicted in RouteClass:
        # Worst case: the rule's fallback is exactly what the model predicts.
        rule_decision = RouteDecisionRecord(
            artifact_id="a",
            route_before_override=RouteClass.PASSTHROUGH,
            route_after_override=RouteClass.PASSTHROUGH,
            risk_class=RiskClass.LOW,
            risk_score=0.1,
            confidence_calibrated=0.9,
            abstained=False,
            fallback_route=predicted,
            rules_fired=[],
            explanation_tags=[],
            model_top_k=[],
            verifier_status="skipped",
            versions={},
        )
        learned = LearnedDecision(
            artifact_id="a",
            route=predicted,
            risk_class=RiskClass.LOW,
            confidence_calibrated=0.9,
            abstained=False,
            model_version="test",
        )
        record = router._build_learned_record(
            "a", features, safety, rule_decision, learned
        )
        checked += 1
        if record.fallback_route == record.route_after_override:
            # Only legitimate when the route cannot fail by transforming.
            assert record.route_after_override in SAFE_TERMINAL_ROUTES, (
                f"predicted={predicted.value}: learned override left a "
                f"self-referential fallback on a transforming route "
                f"({record.route_after_override.value})"
            )

    assert checked == len(RouteClass), "did not cover every route — test is partial"


def test_runtime_chokepoint_redirects_a_self_referential_fallback() -> None:
    """Guard covers future producers — exercised through the real pipeline.

    Uses a stub router that emits a self-referential COMPRESSING route, then
    asserts the pipeline does not spend a second futile codec call on it.
    """
    from tekmyra import runtime as runtime_module
    from tekmyra.rule_router import RuleRouter

    calls: list[str] = []
    real_apply = runtime_module.apply_codec

    def counting_apply(text, route, features, **kwargs):
        calls.append(route.value)
        return real_apply(text, route, features, **kwargs)

    class SelfReferentialRouter(RuleRouter):
        def route(self, artifact_id, features, safety):  # type: ignore[override]
            decision = super().route(artifact_id, features, safety)
            decision.route_after_override = RouteClass.PROSE_TOKEN_COMPRESSOR
            decision.fallback_route = RouteClass.PROSE_TOKEN_COMPRESSOR
            return decision

    fixture = _a_long_prose_fixture()

    # Force the PRIMARY verification to fail, rather than relying on a fixture
    # the compressor happens to be bad at.
    #
    # The subject here is the chokepoint — when the primary fails and the
    # fallback points at itself, does the runtime redirect — not the verifier's
    # opinion of any particular fixture. Depending on a fixture that "fails
    # repeated_entity_retention" coupled this test to the COMPRESSOR MODEL's
    # competence, and the model is retrained per environment
    # (scripts/prepare_test_artifacts.py) so that competence is not a constant.
    # It broke in CI for exactly that reason: the freshly-trained model
    # compresses lc-long_prose-primary_10k_plus-000 cleanly at ratio 0.815, the
    # retry never fires, and the test loses its subject through no fault of the
    # code under test.
    #
    # Forcing the first verdict removes the coupling without weakening anything:
    # the retry is guaranteed to fire, and every assertion below still measures
    # the guard. A fixture-dependent precondition would have to be repaired on
    # every improvement to the compressor, and one day someone would repair it
    # by weakening the assertion instead.
    real_verify = runtime_module.verify
    verify_calls: list[int] = []

    def failing_first_verify(*args, **kwargs):
        result = real_verify(*args, **kwargs)
        verify_calls.append(1)
        if len(verify_calls) == 1:
            return replace(result, passed=False)
        return result

    original = runtime_module.apply_codec
    runtime_module.apply_codec = counting_apply
    runtime_module.verify = failing_first_verify
    try:
        try:
            runtime_module.run_deterministic_pipeline(
                fixture, router=SelfReferentialRouter()
            )
        except runtime_module.TerminalVerificationError:
            pass
    finally:
        runtime_module.apply_codec = original
        runtime_module.verify = real_verify

    assert calls, "pipeline never invoked a codec — test is vacuous"
    assert verify_calls, "verify was never called — the forced failure never applied"
    assert len(calls) >= 2, (
        "the retry never fired even with the primary verdict forced to fail — "
        f"the runtime is not retrying at all (codec calls: {calls})"
    )
    assert calls.count(RouteClass.PROSE_TOKEN_COMPRESSOR.value) == 1, (
        "the retry re-ran the same compressing codec: guard did not fire "
        f"(codec calls: {calls})"
    )


def test_reject_or_human_review_is_a_safe_terminal() -> None:
    """The redirect target must always verify, or the guard just moves the bug."""
    from tekmyra.codecs.selector import (
        UNIMPLEMENTED_CODEC_ROUTES,
        _CODEC_MAP,
        apply_codec,
    )
    from tekmyra.contracts import PolicyTier, SourceChannel
    from tekmyra.prober import probe_text
    from tekmyra.verifier import verify

    assert RouteClass.REJECT_OR_HUMAN_REVIEW in _CODEC_MAP
    assert RouteClass.REJECT_OR_HUMAN_REVIEW not in UNIMPLEMENTED_CODEC_ROUTES

    text = "Some ordinary prose that nothing should reject. " * 50
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD, "long_prose")
    result = apply_codec(text, RouteClass.REJECT_OR_HUMAN_REVIEW, features)
    assert result.success is True
    verdict = verify(text, result, RouteClass.REJECT_OR_HUMAN_REVIEW, features, [])
    assert verdict.passed is True, "the safe terminal must always verify"
