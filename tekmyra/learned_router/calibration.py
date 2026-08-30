from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tekmyra.contracts import RiskClass
from tekmyra.learned_router.config import LearnedRouterConfig


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def scale(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exps = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)


@dataclass(frozen=True)
class AbstentionResult:
    abstain: bool
    reason: str | None
    confidence: float
    margin: float


class AbstentionPolicy:
    def __init__(
        self,
        thresholds: dict[RiskClass, float] | None = None,
        min_margin: float = 0.10,
    ) -> None:
        default_thresholds = {
            RiskClass.LOW: 0.70,
            RiskClass.MEDIUM: 0.75,
            RiskClass.HIGH: 0.85,
            RiskClass.CRITICAL: 0.95,
        }
        self.thresholds = dict(thresholds) if thresholds else default_thresholds
        self.min_margin = min_margin

    @classmethod
    def from_config(cls, config: LearnedRouterConfig) -> "AbstentionPolicy":
        thresholds = {
            RiskClass.LOW: config.confidence_threshold_low,
            RiskClass.MEDIUM: config.confidence_threshold_medium,
            RiskClass.HIGH: config.confidence_threshold_high,
            RiskClass.CRITICAL: config.confidence_threshold_critical,
        }
        return cls(thresholds=thresholds, min_margin=config.min_margin_for_abstention)

    def decide(
        self,
        top_class: str,
        top_prob: float,
        second_prob: float,
        risk_class: RiskClass,
    ) -> AbstentionResult:
        margin = top_prob - second_prob
        threshold = self.thresholds.get(risk_class, 0.85)
        confidence = top_prob
        if confidence < threshold:
            return AbstentionResult(
                abstain=True,
                reason=f"confidence {confidence:.3f} below {risk_class.value} threshold {threshold}",
                confidence=confidence,
                margin=margin,
            )
        if margin < self.min_margin:
            return AbstentionResult(
                abstain=True,
                reason=f"margin {margin:.3f} below minimum {self.min_margin}",
                confidence=confidence,
                margin=margin,
            )
        return AbstentionResult(abstain=False, reason=None, confidence=confidence, margin=margin)
