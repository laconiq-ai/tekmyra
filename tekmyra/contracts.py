"""Runtime contracts as Pydantic models and generated JSON schemas."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tekmyra.resources import resource_dir

CONTRACTS_DIR = resource_dir("contracts")


class RouteClass(str, Enum):
    PASSTHROUGH = "passthrough"
    DETERMINISTIC_JSON_COMPRESSOR = "deterministic_json_compressor"
    CODE_COMPRESSOR = "code_compressor"
    LOG_COMPRESSOR = "log_compressor"
    DIFF_COMPRESSOR = "diff_compressor"
    PROSE_TOKEN_COMPRESSOR = "prose_token_compressor"
    SUMMARIZER = "summarizer"
    EXTRACTIVE_KEYPOINT_COMPRESSOR = "extractive_keypoint_compressor"
    OCR_FIRST = "ocr_first"
    IMAGE_OPTIMIZER = "image_optimizer"
    CHART_TABLE_EXTRACTION = "chart_table_extraction"
    PII_REDACTION_BEFORE_COMPRESSION = "pii_redaction_before_compression"
    GOVERNANCE_SAFE_COMPRESSION = "governance_safe_compression"
    LEGAL_SECURITY_SAFE_MODE = "legal_security_safe_mode"
    REVERSIBLE_RETRIEVAL_MODE = "reversible_retrieval_mode"
    REJECT_OR_HUMAN_REVIEW = "reject_or_human_review"


class RiskClass(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EligibilityReason(str, Enum):
    TOO_SHORT = "too_short"
    CALIBRATION_BAND = "calibration_band"
    HIGH_RISK = "high_risk"
    PII = "pii"
    STRUCTURE_GATE_FAIL = "structure_gate_fail"
    SAFETY_MARGIN_FAIL = "safety_margin_fail"
    PROTECTED_SPAN_CONFLICT = "protected_span_conflict"
    ELIGIBLE = "eligible"
    PRIMARY_POPULATION = "primary_population"


class FallbackLevel(str, Enum):
    NORMAL_COMPRESSION = "normal_compression"
    RELAX_BUDGET = "relax_budget"
    BLOCK_ONLY_COMPRESSION = "block_only_compression"
    DETERMINISTIC_ONLY_COMPRESSION = "deterministic_only_compression"
    BYPASS_ORIGINAL = "bypass_original"
    REJECT_OR_HUMAN_REVIEW = "reject_or_human_review"


class SourceChannel(str, Enum):
    API = "api"
    CHAT = "chat"
    EMAIL = "email"
    TICKET = "ticket"
    UPLOAD = "upload"
    AGENT = "agent"
    BATCH = "batch"


class TrustTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PolicyTier(str, Enum):
    STANDARD = "standard"
    REGULATED = "regulated"
    CRITICAL = "critical"


class CompressorVariant(str, Enum):
    PYTORCH = "pytorch"
    ONNX_V1 = "onnx_v1"
    ONNX_V2 = "onnx_v2"
    ONNX_STATIC_INT8 = "onnx_static_int8"


class ModalityLabel(str, Enum):
    PROSE = "prose"
    JSON = "json"
    CODE = "code"
    LOG = "log"
    STACKTRACE = "stacktrace"
    DIFF = "diff"
    TABLE = "table"
    IMAGE = "image"
    RAG = "rag"
    AGENT_TRACE = "agent_trace"
    GOVERNANCE = "governance"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ArtifactMetadata(BaseModel):
    created_at: datetime
    user_id: str | None = None
    trust_tier: TrustTier
    policy_tier: PolicyTier

    model_config = ConfigDict(extra="allow")


class ArtifactEnvelope(BaseModel):
    artifact_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    source_channel: SourceChannel
    mime_type: str = Field(..., min_length=1)
    filename: str | None = None
    bytes_: str | None = Field(None, alias="bytes")
    text_content: str | None = None
    metadata: ArtifactMetadata

    @model_validator(mode="after")
    def at_least_one_payload(self):
        if self.bytes_ is None and self.text_content is None:
            raise ValueError("Either 'bytes' or 'text_content' must be provided")
        return self

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class FeatureVector(BaseModel):
    feature_schema_version: Literal["1.0.0"] = "1.0.0"
    token_len: int = Field(..., ge=0)
    char_len: int = Field(..., ge=0)
    line_count: int = Field(0, ge=0)
    byte_len: int = Field(0, ge=0)
    char_entropy: float = Field(..., ge=0)
    json_parse_success: bool
    json_depth: int = Field(0, ge=0)
    markdown_density: float = Field(0.0, ge=0)
    code_language: str | None = None
    ast_parse_success: bool = False
    diff_markers: int = Field(0, ge=0)
    log_timestamp_density: float = Field(0.0, ge=0)
    log_repetition_ratio: float = Field(0.0, ge=0)
    stacktrace_hits: int = Field(0, ge=0)
    table_density: float = Field(0.0, ge=0)
    bullet_line_ratio: float = Field(0.0, ge=0, le=1)
    image_keyword_count: int = Field(0, ge=0)
    ocr_keyword_count: int = Field(0, ge=0)
    summarizer_keyword_count: int = Field(0, ge=0)
    prose_sentence_density: float = Field(0.0, ge=0)
    url_density: float = Field(0.0, ge=0)
    pii_entity_count: int = Field(..., ge=0)
    pii_entity_types: list[str] = Field(default_factory=list)
    pii_max_confidence: float = Field(0.0, ge=0, le=1)
    legal_keyword_count: int = Field(..., ge=0)
    security_keyword_count: int = Field(..., ge=0)
    governance_keyword_count: int = Field(..., ge=0)
    financial_amount_count: int = Field(0, ge=0)
    deadline_date_count: int = Field(0, ge=0)
    embedded_text_ratio: float = Field(0.0, ge=0, le=1)
    ocr_confidence: float = Field(0.0, ge=0, le=1)
    image_hint: Literal[
        "none", "photo", "screenshot", "chart", "table", "document_scan", "unknown"
    ] = "none"
    source_channel: SourceChannel
    policy_tier: PolicyTier
    content_type: str = "unknown"
    text: str | None = None

    model_config = ConfigDict(extra="forbid")


class RouteDecision(BaseModel):
    decision_id: str = Field(..., min_length=1)
    artifact_id: str = Field(..., min_length=1)
    route_before_override: RouteClass
    route_after_override: RouteClass
    risk_class: RiskClass
    risk_score: float = Field(..., ge=0, le=1)
    confidence_calibrated: float = Field(..., ge=0, le=1)
    abstained: bool = False
    fallback_route: RouteClass | None = None
    rules_fired: list[str] = Field(default_factory=list)
    explanation_tags: list[str] = Field(default_factory=list)
    model_top_k: list[tuple[RouteClass, float]] = Field(default_factory=list)
    verifier_status: Literal["pass", "fail", "skipped"] = "skipped"
    mixed_subroutes: list[dict[str, Any]] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class EligibilityOutcome(BaseModel):
    eligible: bool
    selected_route: RouteClass
    selected_codec: str | None = None
    reason: EligibilityReason
    predicted_request_tokens: int = Field(..., ge=0)
    predicted_request_cost: float = Field(..., ge=0)
    predicted_compressor_cost: float = Field(..., ge=0)
    predicted_compressor_latency_ms: float = Field(..., ge=0)
    predicted_savings: float
    safety_margin: float = Field(..., ge=0)
    expected_net_value: float
    passthrough_default: bool = False

    model_config = ConfigDict(extra="forbid")


class CompressionEconomics(BaseModel):
    # predicted (ex ante)
    predicted_request_tokens: int | None = None
    predicted_request_cost: float | None = None
    predicted_compressor_cost: float | None = None
    predicted_compressor_latency_ms: float | None = None
    predicted_savings: float | None = None
    predicted_safety_margin: float | None = None
    # observed (ex post)
    observed_raw_context_tokens: int | None = Field(None, ge=0)
    observed_compressed_context_tokens: int | None = Field(None, ge=0)
    observed_raw_request_tokens: int | None = Field(None, ge=0)
    observed_compressed_request_tokens: int | None = Field(None, ge=0)
    observed_raw_output_tokens: int | None = Field(None, ge=0)
    observed_compressed_output_tokens: int | None = Field(None, ge=0)
    observed_raw_cost_usd: float | None = Field(None, ge=0)
    observed_compressed_cost_usd: float | None = Field(None, ge=0)
    observed_compressor_cpu_ms: int | None = Field(None, ge=0)
    observed_model_raw_latency_ms: float | None = Field(None, ge=0)
    observed_model_compressed_latency_ms: float | None = Field(None, ge=0)

    model_config = ConfigDict(extra="forbid")


class ObservedCompressionResult(BaseModel):
    observed_eligibility_decision: bool
    observed_bypass_reason: EligibilityReason | None = None
    selected_route: RouteClass | None = None
    selected_codec: str | None = None
    learned_compressor_invoked: bool = False
    economics: CompressionEconomics = Field(default_factory=CompressionEconomics)
    protected_disposition_valid: bool = False
    introduced_duplicate_count: int = Field(0, ge=0)
    required_answer_fact_recall: float = Field(0.0, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class RouterInput(BaseModel):
    artifact_id: str = Field(..., min_length=1)
    text: str | None = None
    feature_vector: FeatureVector | None = None
    metadata_tokens: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def has_some_input(self):
        if not self.text and self.feature_vector is None:
            raise ValueError("RouterInput needs text or feature_vector")
        return self

    model_config = ConfigDict(extra="forbid")


class LearnedDecision(BaseModel):
    artifact_id: str = Field(..., min_length=1)
    route: RouteClass | None = None
    route_probabilities: dict[str, float] = Field(default_factory=dict)
    risk_class: RiskClass = RiskClass.LOW
    risk_probabilities: dict[str, float] = Field(default_factory=dict)
    modalities: list[ModalityLabel] = Field(default_factory=list)
    modality_scores: dict[str, float] = Field(default_factory=dict)
    confidence_calibrated: float = Field(0.0, ge=0, le=1)
    abstained: bool = True
    abstention_reason: str | None = None
    model_version: str = "none"
    latency_ms: float = 0.0

    model_config = ConfigDict(extra="forbid")


class CompressionSegment(BaseModel):
    segment_id: str
    original_content: str
    compressed_content: str
    content_type: str
    mode: Literal["passthrough", "deterministic", "learned", "hybrid", "reversible", "human_review"]
    fallback_level: FallbackLevel
    locked_span_count: int = Field(0, ge=0)
    anchor_ids: list[str] = Field(default_factory=list)
    compression_ratio: float = Field(..., ge=0)

    model_config = ConfigDict(extra="forbid")


class LatencyBreakdown(BaseModel):
    detect: float = 0.0
    route: float = 0.0
    model: float = 0.0
    verify: float = 0.0
    anchor: float = 0.0
    total: float = 0.0

    model_config = ConfigDict(extra="forbid")


class CompressionMetrics(BaseModel):
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    compression_ratio: float = Field(..., ge=0)
    latency_ms: LatencyBreakdown
    verifier_pass: bool = False
    fail_safe_triggered: bool = False

    model_config = ConfigDict(extra="forbid")


class PreservationReport(BaseModel):
    locked_spans_detected: int = Field(..., ge=0)
    locked_spans_preserved: int = Field(..., ge=0)
    citations_preserved: int = Field(0, ge=0)
    numeric_mutations: int = Field(0, ge=0)
    policy_control_id_losses: int = Field(0, ge=0)

    model_config = ConfigDict(extra="forbid")


class AnchorSummary(BaseModel):
    anchor_id: str
    segment_id: str
    offset_start: int = Field(..., ge=0)
    offset_end: int = Field(..., ge=0)
    recovery_ref: str
    content_hash: str
    content_type: str

    model_config = ConfigDict(extra="forbid")


class CompressionResponse(BaseModel):
    request_id: str
    compressed_segments: list[CompressionSegment]
    global_metrics: CompressionMetrics
    preservation_report: PreservationReport
    anchor_map: list[AnchorSummary]
    observed_result: ObservedCompressionResult | None = None
    explanations: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class CompressionManifestBlock(BaseModel):
    block_id: str
    block_type: str
    route_class: RouteClass
    confidence_calibrated: float = Field(..., ge=0, le=1)
    risk_class: RiskClass
    output_ref: str
    anchor_ids: list[str] = Field(default_factory=list)
    verifier_status: Literal["pass", "fail", "skipped"] = "skipped"

    model_config = ConfigDict(extra="forbid")


class CompressionManifest(BaseModel):
    artifact_id: str
    is_mixed: bool
    blocks: list[CompressionManifestBlock]
    assembly_order: list[str]
    global_fallback: FallbackLevel

    model_config = ConfigDict(extra="forbid")


class CompressionRequest(BaseModel):
    request_id: str
    task_type: str
    query: str | None = None
    max_output_tokens: int | None = Field(None, ge=1)
    safety_mode: Literal["standard", "strict", "audit"] = "standard"
    segments: list[ArtifactEnvelope]
    recovery_options: dict[str, Any] = Field(default_factory=dict)
    explanation_options: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class AnchorRecord(BaseModel):
    anchor_id: str
    tenant_id: str
    source_id: str
    doc_id: str
    chunk_id: str
    content_hash: str
    original_text: str
    offset_start: int = Field(..., ge=0)
    offset_end: int = Field(..., ge=0)
    sensitivity: RiskClass
    expiry: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(extra="forbid")


class ProtectedSpan(BaseModel):
    span_type: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    text: str
    policy: Literal["preserve", "redact", "anchorize", "human_review"]
    detector: str
    confidence: float = Field(0.0, ge=0, le=1)
    span_id: str | None = None
    fact_id: str | None = None
    # WHAT KIND OF DATA THIS IS — orthogonal to span_type (HOW to handle it) and
    # to policy (WHAT WAS DONE to it). Declared because both models are
    # extra="forbid", so it cannot be added at a construction site.
    #
    # None MEANS NOT CATEGORISED. IT DOES NOT MEAN "NO PII".
    #
    # That distinction is the whole risk in this field. The PII leakage check
    # has read False forever because it compares a handling label against a
    # category list; a nullable category that consumers read as "nothing here"
    # would be a THIRD independent way for the same check to look alive while
    # asserting nothing. Any consumer must treat None as "cannot assert" and
    # fail closed, never as evidence of absence.
    #
    # Deliberately str, not an enum: the value set is the taxonomy ruling and is
    # not this field's to decide. NOTHING POPULATES IT YET — every construction
    # site is held, because filling it in before the vocabulary exists is how a
    # further vocabulary gets born. `label` must not be repurposed for it
    # either; on the block path `label` already carries a live 13-value category
    # vocabulary from block_segmenter, and overloading it again would bury one
    # defect underneath another.
    data_category: str | None = None

    model_config = ConfigDict(extra="forbid")


class BlockSpan(BaseModel):
    """A protected span local to a segmented block."""

    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    label: str
    # "fact" was missing while 160 fixture spans carry it. It could not surface
    # as a validation error because hierarchical_compressor hardcoded this field
    # to "must-preserve" instead of passing the real value — the flattening hid
    # the gap that the flattening created. Admitting an existing measured value,
    # not inventing a taxonomy: the data-category dimension is a separate,
    # unresolved question and does not belong in this field.
    span_type: Literal["anchor", "locked", "must-preserve", "fact"]
    detector: str
    confidence: float = Field(0.0, ge=0, le=1)
    # See ProtectedSpan.data_category. None MEANS NOT CATEGORISED, NOT "no PII".
    #
    # Note the trap specific to this model: `label` on the block-segmenter path
    # already holds real data categories (ssn, email, phone, credit_card,
    # api_key, citation, timestamp, amount, file_path, line_number). This field
    # is NOT a rename of that and must not be populated by copying it — the two
    # would then be a fifth and sixth vocabulary for the same idea, disagreeing
    # by construction. Reconciling them is the taxonomy ruling.
    data_category: str | None = None

    model_config = ConfigDict(extra="forbid")


class Block(BaseModel):
    """Segmented content block with deterministic routing metadata."""

    type: str
    content: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    route_label: str
    risk_tier: RiskClass
    protected_spans: list[BlockSpan] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class LabelRecord(BaseModel):
    sample_id: str
    content_ref: str
    content_type: str
    risk_mode: RiskClass
    protected_spans: list[ProtectedSpan]
    keep_labels: list[bool] = Field(default_factory=list)
    importance_scores: list[float] = Field(default_factory=list)
    must_preserve: list[bool] = Field(default_factory=list)
    retrieval_anchor: list[bool] = Field(default_factory=list)
    budget_class: str = "moderate"
    downstream_score: float | None = None
    uncertainty: bool = False
    label_source: str
    reviewer: str | None = None

    model_config = ConfigDict(extra="forbid")


class TelemetryRecord(BaseModel):
    decision_id: str
    timestamp: datetime
    tenant_id: str
    artifact_id: str
    payload_sha256: str
    source_channel: SourceChannel
    mime_type: str
    policy_tier: PolicyTier
    modality_summary: str = ""
    parser_outcomes: dict[str, Any] = Field(default_factory=dict)
    feature_schema_version: str
    feature_checksum: str
    rules_fired: list[str] = Field(default_factory=list)
    route_before_override: RouteClass
    route_after_override: RouteClass
    confidence_calibrated: float = Field(..., ge=0, le=1)
    risk_class: RiskClass
    risk_score: float = Field(..., ge=0, le=1)
    abstained: bool
    latency_ms: float = Field(..., ge=0)
    verifier_result: Literal["pass", "fail", "skipped"]
    reviewer_result: str | None = None
    block_subroutes: list[dict[str, Any]] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "artifact_envelope": ArtifactEnvelope,
    "compression_request": CompressionRequest,
    "compression_response": CompressionResponse,
    "compression_manifest": CompressionManifest,
    "route_decision": RouteDecision,
    "feature_vector": FeatureVector,
    "telemetry_record": TelemetryRecord,
    "anchor_record": AnchorRecord,
    "label_schema": LabelRecord,
    "eligibility_outcome": EligibilityOutcome,
    "compression_economics": CompressionEconomics,
    "observed_compression_result": ObservedCompressionResult,
}

ENUM_FILES: dict[str, list[str] | list[Enum]] = {
    "route_classes": [r.value for r in RouteClass],
    "risk_classes": [r.value for r in RiskClass],
    "fallback_levels": [f.value for f in FallbackLevel],
    "eligibility_reasons": [r.value for r in EligibilityReason],
}


def dump_schemas(output_dir: Path | None = None) -> None:
    output_dir = output_dir or CONTRACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMA_MODELS.items():
        schema = model.model_json_schema()
        with open(output_dir / f"{name}.schema.json", "w") as f:
            json.dump(schema, f, indent=2)
            f.write("\n")
    for name, values in ENUM_FILES.items():
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump([v.value if isinstance(v, Enum) else v for v in values], f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    dump_schemas()
