"""Tests for the deterministic rule router."""

from __future__ import annotations


from tekmyra.contracts import PolicyTier, RiskClass, RouteClass, SourceChannel
from tekmyra.prober import probe_text
from tekmyra.rule_router import RuleRouter
from tekmyra.safety_gate import SafetyGate


def router() -> RuleRouter:
    return RuleRouter()


def test_json_routes_to_json_compressor():
    text = '{"id": 1, "name": "test"}'
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    decision = router().route("fx-1", features, safety)
    assert decision.route_after_override == RouteClass.DETERMINISTIC_JSON_COMPRESSOR
    assert "R_JSON_VALID" in decision.rules_fired
    assert decision.confidence_calibrated == 1.0


def test_diff_routes_to_diff_compressor():
    text = "--- a/file.py\n+++ b/file.py\n@@ -1,2 +1,2 @@\n- old\n+ new\n"
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    decision = router().route("fx-2", features, safety)
    assert decision.route_after_override == RouteClass.DIFF_COMPRESSOR
    assert "R_DIFF_SIGNATURE" in decision.rules_fired


def test_code_routes_to_code_compressor():
    text = "def hello():\n    return 'world'\n"
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    decision = router().route("fx-3", features, safety)
    assert decision.route_after_override == RouteClass.CODE_COMPRESSOR
    assert "R_CODE_AST_VALID" in decision.rules_fired


def test_log_routes_to_log_compressor():
    text = (
        "2026-07-06T00:00:01Z INFO host-01 request-1001 Starting job\n"
        "2026-07-06T00:00:02Z INFO host-01 request-1001 Done\n"
    )
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    decision = router().route("fx-4", features, safety)
    assert decision.route_after_override == RouteClass.LOG_COMPRESSOR
    assert "R_LOG_DENSE" in decision.rules_fired


def test_stacktrace_routes_to_log_compressor_for_stacktrace_codec():
    text = (
        "Traceback (most recent call last):\n"
        '  File "/usr/local/lib/python3.11/site-packages/flask/app.py", line 1000\n'
        '  File "/app/db.py", line 91, in query\n'
        "    cursor.execute('SELECT *')\n"
        "DBTimeoutError: query exceeded 5000ms\n"
    )
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    decision = router().route("fx-5", features, safety)
    # Stack traces are medium risk but no longer forced to reversible because
    # the deterministic stack-trace codec is verified.
    assert safety.risk_class == RiskClass.MEDIUM
    assert safety.forced_route is None
    assert "R_STACKTRACE" in decision.rules_fired
    assert decision.route_after_override == RouteClass.LOG_COMPRESSOR


def test_safety_override_wins():
    text = "Customer SSN: 123-45-6789"
    features = probe_text(text, SourceChannel.CHAT, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    decision = router().route("fx-6", features, safety)
    assert decision.route_after_override == RouteClass.PII_REDACTION_BEFORE_COMPRESSION
    assert "SAFETY_OVERRIDE" in decision.rules_fired


def test_short_safe_passthrough():
    text = "Short note."
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    decision = router().route("fx-7", features, safety)
    assert decision.route_after_override == RouteClass.PASSTHROUGH
    assert "R_SHORT_SAFE" in decision.rules_fired


def test_route_intent_is_separate_from_short_input_bypass():
    text = "Short prose note."
    features = probe_text(
        text,
        SourceChannel.API,
        PolicyTier.STANDARD,
        content_type="prose",
    )
    safety = SafetyGate().classify(features)

    operational = router().route("fx-intent", features, safety)
    intent = router().route_intent("fx-intent", features, safety)

    assert operational.route_before_override == RouteClass.PASSTHROUGH
    assert intent.route_before_override == RouteClass.PROSE_TOKEN_COMPRESSOR
    assert intent.route_after_override == RouteClass.PROSE_TOKEN_COMPRESSOR


def test_mixed_payload_preserves_whole_document():
    text = (
        "Incident summary.\n"
        "2026-07-06T00:00:01Z ERROR request failed\n"
        '{"status": "retrying"}'
    )
    features = probe_text(
        text,
        SourceChannel.API,
        PolicyTier.STANDARD,
        content_type="mixed",
    )
    safety = SafetyGate().classify(features)
    decision = router().route("fx-mixed", features, safety)

    assert decision.route_before_override == RouteClass.PASSTHROUGH
    assert "R_MIXED_SAFE" in decision.rules_fired


def test_ambiguous_abstains():
    # Use a tiny passthrough budget so this short text cannot trigger R_SHORT_SAFE.
    custom_router = RuleRouter(passthrough_token_budget=5)
    text = "Some arbitrary text that does not match any deterministic rule."
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    decision = custom_router.route("fx-8", features, safety)
    assert decision.route_after_override == RouteClass.REVERSIBLE_RETRIEVAL_MODE
    assert decision.abstained is True
    assert "R_UNRESOLVED_AMBIGUOUS" in decision.rules_fired


def test_deterministic_routes_have_confidence_one():
    text = '{"id": 1}'
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    safety = SafetyGate().classify(features)
    decision = router().route("fx-9", features, safety)
    assert decision.confidence_calibrated == 1.0


def test_critical_safety_rejects():
    text = " harmless "
    features = probe_text(text, SourceChannel.API, PolicyTier.CRITICAL)
    safety = SafetyGate().classify(features)
    decision = router().route("fx-10", features, safety)
    assert decision.route_after_override == RouteClass.REJECT_OR_HUMAN_REVIEW
    assert decision.risk_class == RiskClass.CRITICAL
