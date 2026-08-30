from __future__ import annotations

from tekmyra.contracts import FeatureVector, RouterInput
from tekmyra.learned_router.tokenizer_window import head_tail_window

_NUMERIC_FEATURES = [
    "token_len", "char_len", "line_count", "byte_len", "char_entropy",
    "json_depth", "markdown_density", "diff_markers", "log_timestamp_density",
    "log_repetition_ratio", "stacktrace_hits", "table_density", "url_density",
    "pii_entity_count", "pii_max_confidence", "legal_keyword_count",
    "security_keyword_count", "governance_keyword_count", "financial_amount_count",
    "deadline_date_count", "embedded_text_ratio", "ocr_confidence",
]


def _metadata_tokens(features: FeatureVector, template: str) -> list[str]:
    rendered = template.format(
        mime=features.source_channel.value,
        channel=features.source_channel.value,
        policy=features.policy_tier.value,
    )
    return [t.strip() for t in rendered.split() if t.strip()]


def numeric_feature_vector(features: FeatureVector) -> list[float]:
    values: list[float] = []
    for name in _NUMERIC_FEATURES:
        v = getattr(features, name, 0)
        values.append(float(v) if v is not None else 0.0)
    return values


def build_router_input(
    artifact_id: str,
    text: str,
    features: FeatureVector,
    max_tokens: int = 512,
    head_tail_ratio: float = 0.5,
    metadata_template: str = "[MIME={mime}] [CHANNEL={channel}] [POLICY={policy}]",
) -> RouterInput:
    window = head_tail_window(text, max_tokens, head_tail_ratio)
    meta = _metadata_tokens(features, metadata_template)
    return RouterInput(
        artifact_id=artifact_id,
        text=window,
        feature_vector=features,
        metadata_tokens=meta,
    )
