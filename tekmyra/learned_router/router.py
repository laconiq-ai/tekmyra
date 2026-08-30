"""Public learned-router inference API.

Fails closed: any error or missing artifact returns an abstained decision.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from tekmyra.contracts import (
    LearnedDecision,
    RouteClass,
)
from tekmyra.learned_router.calibration import AbstentionPolicy, TemperatureScaler
from tekmyra.learned_router.config import LearnedRouterConfig
from tekmyra.learned_router.heads import ModalityHead, RiskHead, RouteHead
from tekmyra.learned_router.model import ModelManager
from tekmyra.resources import resource_path


class LearnedRouter:
    """Thin, stateless inference wrapper around a versioned model artifact."""

    def __init__(self, config: LearnedRouterConfig | None = None) -> None:
        base_config = config or LearnedRouterConfig.from_yaml(
            resource_path("configs", "model_router_small.yaml")
        )
        self.config = ModelManager.resolve_config(base_config)
        self._manager = ModelManager(self.config)
        self._model = self._manager.load()
        self._abstention = AbstentionPolicy.from_config(self.config)
        self._temperature = TemperatureScaler(self.config.temperature)

    @property
    def model_loaded(self) -> bool:
        """Whether the exact configured learned artifact loaded successfully."""
        return self._model is not None

    @property
    def model_version(self) -> str:
        """Return the configured model version only when a model is loaded."""
        return self.config.version if self.model_loaded else "none"

    def route(
        self,
        artifact_id: str,
        text: str,
        numeric: list[float],
    ) -> LearnedDecision:
        start = time.perf_counter()
        if self._model is None:
            return LearnedDecision(
                artifact_id=artifact_id,
                abstained=True,
                abstention_reason="no_model_artifact",
                model_version="none",
                confidence_calibrated=0.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            raw = self._model.predict([text], [numeric])
        except Exception as exc:
            return LearnedDecision(
                artifact_id=artifact_id,
                abstained=True,
                abstention_reason=f"inference_error:{type(exc).__name__}",
                model_version=self.config.version,
                confidence_calibrated=0.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        route_pred = RouteHead.predict(raw.route_probs)
        risk_pred = RiskHead.predict(raw.risk_probs)
        modality_pred = ModalityHead.predict(raw.modality_scores)

        # Calibrate route probabilities using temperature scaling.
        logits = np.log(np.clip(list(raw.route_probs.values()), 1e-10, 1.0))
        calibrated = self._temperature.scale(logits.reshape(1, -1))[0]
        label_list = list(raw.route_probs.keys())
        calibrated_probs = {label_list[i]: float(calibrated[i]) for i in range(len(label_list))}
        sorted_probs = sorted(calibrated_probs.items(), key=lambda x: x[1], reverse=True)
        top_class, top_prob = sorted_probs[0]
        second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0

        abstention = self._abstention.decide(top_class, top_prob, second_prob, risk_pred.risk_class)

        shared: dict[str, Any] = {
            "route_probabilities": calibrated_probs,
            "risk_class": risk_pred.risk_class,
            "risk_probabilities": raw.risk_probs,
            "modalities": modality_pred.modalities,
            "modality_scores": raw.modality_scores,
            "confidence_calibrated": abstention.confidence,
            "model_version": raw.model_version,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }

        if abstention.abstain:
            return LearnedDecision(
                artifact_id=artifact_id,
                route=route_pred.route,
                abstained=True,
                abstention_reason=abstention.reason,
                route_probabilities=shared["route_probabilities"],
                risk_class=shared["risk_class"],
                risk_probabilities=shared["risk_probabilities"],
                modalities=shared["modalities"],
                modality_scores=shared["modality_scores"],
                confidence_calibrated=shared["confidence_calibrated"],
                model_version=shared["model_version"],
                latency_ms=shared["latency_ms"],
            )

        return LearnedDecision(
            artifact_id=artifact_id,
            route=RouteClass(top_class),
            abstained=False,
            abstention_reason=None,
            route_probabilities=shared["route_probabilities"],
            risk_class=shared["risk_class"],
            risk_probabilities=shared["risk_probabilities"],
            modalities=shared["modalities"],
            modality_scores=shared["modality_scores"],
            confidence_calibrated=shared["confidence_calibrated"],
            model_version=shared["model_version"],
            latency_ms=shared["latency_ms"],
        )
