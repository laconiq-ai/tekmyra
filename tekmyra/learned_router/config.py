from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:  # import-time cycle: artifact_paths is imported where used
    from tekmyra import artifact_paths


class LearnedRouterConfig(BaseModel):
    name: str = "router_small"
    version: str = "1.0.0"
    max_router_tokens: int = Field(512, ge=8, le=2048)
    head_tail_ratio: float = Field(0.5, ge=0.1, le=0.9)
    metadata_prefix_template: str = "[MIME={mime}] [CHANNEL={channel}] [POLICY={policy}]"
    route_classes: list[str] = Field(default=[
        "passthrough",
        "prose_token_compressor",
        "reversible_retrieval_mode",
        "reject_or_human_review",
    ])
    risk_classes: list[str] = Field(default=["low", "medium", "high", "critical"])
    modalities: list[str] = Field(default=[
        "prose",
        "rag",
        "agent_trace",
        "governance",
        "mixed",
        "unknown",
    ])
    confidence_threshold_low: float = Field(0.70, ge=0, le=1)
    confidence_threshold_medium: float = Field(0.75, ge=0, le=1)
    confidence_threshold_high: float = Field(0.85, ge=0, le=1)
    confidence_threshold_critical: float = Field(0.95, ge=0, le=1)
    min_margin_for_abstention: float = Field(0.10, ge=0, le=1)
    temperature: float = Field(1.0, ge=0.01, le=10.0)
    calibration_method: Literal["temperature", "isotonic", "none"] = "temperature"
    # RELATIVE BY DECLARATION, RESOLVED BY `resolved_artifact_dir`. Leaving the
    # bare relative path to be resolved against the working directory by each
    # consumer is the same defect the compressor carried: run from anywhere but
    # the tree root and the artifact silently is not found. The field stays
    # relative because it is a declared default that must round-trip to YAML;
    # the resolution happens in one place below.
    model_artifact_dir: Path = Path(".cache/router")
    onnx_file_name: str = "router.onnx"
    sklearn_file_name: str = "router.joblib"
    fallback_on_error: Literal["abstain", "metadata"] = "abstain"
    use_onnx_if_available: bool = True

    @field_validator("route_classes", "risk_classes", "modalities", mode="before")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("label list must not be empty")
        return v

    def resolved_artifact_dir(self) -> "artifact_paths.Resolution":
        """Where this config's versioned artifact directory actually is.

        THE SINGLE DEFINITION. Three call sites previously each computed
        ``Path(config.model_artifact_dir) / config.version`` and resolved it
        against their own working directory. Three copies of a path rule
        diverge silently, and they diverge on the path that decides whether a
        published number describes a trained model or a no-op.

        Returns the whole search, not just the answer, so a caller reporting a
        miss can name every location tried instead of inventing one.
        """
        from tekmyra import artifact_paths as _paths

        return _paths.resolve(Path(self.model_artifact_dir) / self.version)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LearnedRouterConfig":
        with open(path) as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)
