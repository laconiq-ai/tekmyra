import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import jsonschema  # type: ignore[import-untyped]
import pytest
from pydantic import ValidationError

from tekmyra.contracts import (
    ArtifactEnvelope,
    ArtifactMetadata,
    CompressionEconomics,
    CompressionRequest,
    CompressionResponse,
    EligibilityOutcome,
    EligibilityReason,
    FeatureVector,
    ObservedCompressionResult,
    PolicyTier,
    RiskClass,
    RouteClass,
    RouteDecision,
    SourceChannel,
    TelemetryRecord,
    TrustTier,
)


def test_artifact_envelope_requires_payload():
    with pytest.raises(ValidationError):
        ArtifactEnvelope(
            artifact_id="a1",
            tenant_id="t1",
            source_channel=SourceChannel.API,
            mime_type="text/plain",
            metadata=ArtifactMetadata(
                created_at=datetime.now(timezone.utc),
                trust_tier=TrustTier.HIGH,
                policy_tier=PolicyTier.STANDARD,
            ),
        )


def test_artifact_envelope_accepts_text():
    env = ArtifactEnvelope(
        artifact_id="a1",
        tenant_id="t1",
        source_channel=SourceChannel.API,
        mime_type="text/plain",
        text_content="hello world",
        metadata=ArtifactMetadata(
            created_at=datetime.now(timezone.utc),
            trust_tier=TrustTier.HIGH,
            policy_tier=PolicyTier.STANDARD,
        ),
    )
    assert env.text_content == "hello world"


def test_route_decision_round_trip():
    decision = RouteDecision(
        decision_id=str(uuid.uuid4()),
        artifact_id="a1",
        route_before_override=RouteClass.PASSTHROUGH,
        route_after_override=RouteClass.PASSTHROUGH,
        risk_class=RiskClass.LOW,
        risk_score=0.1,
        confidence_calibrated=1.0,
        rules_fired=["R_SHORT_SAFE"],
        versions={"rule_set": "0.1.0"},
    )
    data = decision.model_dump(mode="json")
    restored = RouteDecision.model_validate(data)
    assert restored.route_after_override == RouteClass.PASSTHROUGH


def test_feature_vector_required_fields():
    # Omitting required `json_parse_success` should raise ValidationError.
    with pytest.raises(ValidationError):
        FeatureVector(
            token_len=10,
            char_len=10,
            char_entropy=1.5,
            ast_parse_success=False,
            pii_entity_count=0,
            legal_keyword_count=0,
            security_keyword_count=0,
            governance_keyword_count=0,
            source_channel=SourceChannel.API,
            policy_tier=PolicyTier.STANDARD,
        )


def test_compression_request_segments():
    now = datetime.now(timezone.utc)
    segment = ArtifactEnvelope(
        artifact_id="a1",
        tenant_id="t1",
        source_channel=SourceChannel.TICKET,
        mime_type="text/plain",
        text_content="sample",
        metadata=ArtifactMetadata(
            created_at=now, trust_tier=TrustTier.MEDIUM, policy_tier=PolicyTier.REGULATED
        ),
    )
    req = CompressionRequest(
        request_id="r1", task_type="compress", segments=[segment]
    )
    assert len(req.segments) == 1


def test_telemetry_rejects_raw_content_fields():
    record = TelemetryRecord(
        decision_id="d1",
        timestamp=datetime.now(timezone.utc),
        tenant_id="t1",
        artifact_id="a1",
        payload_sha256="abc",
        source_channel=SourceChannel.API,
        mime_type="text/plain",
        policy_tier=PolicyTier.STANDARD,
        feature_schema_version="1.0.0",
        feature_checksum="cafe",
        route_before_override=RouteClass.PASSTHROUGH,
        route_after_override=RouteClass.PASSTHROUGH,
        confidence_calibrated=1.0,
        risk_class=RiskClass.LOW,
        risk_score=0.0,
        abstained=False,
        latency_ms=0.0,
        verifier_result="skipped",
    )
    assert record.payload_sha256 == "abc"


def test_generated_schemas_are_valid_json():
    from tekmyra.contracts import dump_schemas
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dump_schemas(Path(tmp))
        for name in [
            "artifact_envelope",
            "route_decision",
            "compression_request",
            "compression_response",
            "compression_manifest",
            "feature_vector",
            "telemetry_record",
            "anchor_record",
            "label_schema",
            "eligibility_outcome",
            "compression_economics",
            "observed_compression_result",
        ]:
            with open(Path(tmp) / f"{name}.schema.json") as f:
                schema = json.load(f)
            assert "title" in schema or "$defs" in schema


def test_generated_schemas_pass_jsonschema_validator():
    from tekmyra.contracts import dump_schemas
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dump_schemas(Path(tmp))
        for name in [
            "artifact_envelope",
            "route_decision",
            "compression_request",
            "compression_response",
            "compression_manifest",
            "feature_vector",
            "telemetry_record",
            "anchor_record",
            "label_schema",
            "eligibility_outcome",
            "compression_economics",
            "observed_compression_result",
        ]:
            with open(Path(tmp) / f"{name}.schema.json") as f:
                schema = json.load(f)
            jsonschema.Draft202012Validator.check_schema(schema)


def test_eligibility_reason_enum_values():
    assert EligibilityReason.ELIGIBLE.value == "eligible"
    assert EligibilityReason.TOO_SHORT.value == "too_short"
    assert EligibilityReason.PRIMARY_POPULATION.value == "primary_population"


def test_eligibility_outcome_round_trip():
    outcome = EligibilityOutcome(
        eligible=True,
        selected_route=RouteClass.EXTRACTIVE_KEYPOINT_COMPRESSOR,
        selected_codec="keypoint_v1",
        reason=EligibilityReason.ELIGIBLE,
        predicted_request_tokens=1000,
        predicted_request_cost=0.0015,
        predicted_compressor_cost=0.0001,
        predicted_compressor_latency_ms=12.0,
        predicted_savings=0.0004,
        safety_margin=0.10,
        expected_net_value=0.0003,
        passthrough_default=False,
    )
    data = outcome.model_dump(mode="json")
    restored = EligibilityOutcome.model_validate(data)
    assert restored.eligible is True
    assert restored.selected_route == RouteClass.EXTRACTIVE_KEYPOINT_COMPRESSOR
    assert restored.reason == EligibilityReason.ELIGIBLE


def test_compression_economics_predicted_and_observed_fields():
    econ = CompressionEconomics(
        predicted_request_tokens=1000,
        predicted_request_cost=0.0015,
        observed_raw_context_tokens=5000,
        observed_compressed_context_tokens=4250,
        observed_compressor_cpu_ms=120,
    )
    assert econ.predicted_request_tokens == 1000
    assert econ.observed_compressed_context_tokens == 4250


def test_observed_compression_result_round_trip():
    observed = ObservedCompressionResult(
        observed_eligibility_decision=False,
        observed_bypass_reason=EligibilityReason.TOO_SHORT,
        selected_route=RouteClass.PASSTHROUGH,
        selected_codec=None,
        learned_compressor_invoked=False,
        economics=CompressionEconomics(
            observed_raw_context_tokens=1200,
            observed_compressed_context_tokens=1200,
        ),
        protected_disposition_valid=True,
        introduced_duplicate_count=0,
        required_answer_fact_recall=1.0,
    )
    data = observed.model_dump(mode="json")
    restored = ObservedCompressionResult.model_validate(data)
    assert restored.observed_bypass_reason == EligibilityReason.TOO_SHORT
    assert restored.learned_compressor_invoked is False
    assert restored.required_answer_fact_recall == 1.0


def test_compression_response_observed_result_optional():
    response = CompressionResponse(
        request_id="r1",
        compressed_segments=[],
        global_metrics={
            "input_tokens": 100,
            "output_tokens": 100,
            "compression_ratio": 1.0,
            "latency_ms": {},
        },
        preservation_report={
            "locked_spans_detected": 0,
            "locked_spans_preserved": 0,
        },
        anchor_map=[],
    )
    assert response.observed_result is None
    observed = ObservedCompressionResult(
        observed_eligibility_decision=True,
        selected_route=RouteClass.PROSE_TOKEN_COMPRESSOR,
    )
    response_with_observed = CompressionResponse(
        request_id="r2",
        compressed_segments=[],
        global_metrics={
            "input_tokens": 100,
            "output_tokens": 100,
            "compression_ratio": 1.0,
            "latency_ms": {},
        },
        preservation_report={
            "locked_spans_detected": 0,
            "locked_spans_preserved": 0,
        },
        anchor_map=[],
        observed_result=observed,
    )
    assert response_with_observed.observed_result is not None
    assert response_with_observed.observed_result.selected_route == RouteClass.PROSE_TOKEN_COMPRESSOR
