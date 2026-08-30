from __future__ import annotations

from tekmyra.contracts import ModalityLabel, RiskClass, RouteClass


class RouteHead:
    _AMBIGUOUS = [
        RouteClass.PASSTHROUGH,
        RouteClass.PROSE_TOKEN_COMPRESSOR,
        RouteClass.REVERSIBLE_RETRIEVAL_MODE,
        RouteClass.REJECT_OR_HUMAN_REVIEW,
    ]

    @classmethod
    def allowed_routes(cls) -> list[RouteClass]:
        return list(cls._AMBIGUOUS)

    @classmethod
    def from_label_map(cls, labels: list[str]) -> list[RouteClass]:
        allowed = {r.value for r in cls._AMBIGUOUS}
        return [RouteClass(label) for label in labels if label in allowed]

    @classmethod
    def predict(cls, probabilities: dict[str, float]) -> "RoutePrediction":
        if not probabilities:
            return RoutePrediction(route=None, probabilities={})
        top_label = max(probabilities, key=lambda k: probabilities[k])
        route = RouteClass(top_label) if top_label in {r.value for r in cls._AMBIGUOUS} else None
        return RoutePrediction(route=route, probabilities=probabilities)


class RoutePrediction:
    def __init__(self, route: RouteClass | None, probabilities: dict[str, float]) -> None:
        self.route = route
        self.probabilities = probabilities


class RiskHead:
    _ORDER = [RiskClass.LOW, RiskClass.MEDIUM, RiskClass.HIGH, RiskClass.CRITICAL]

    @classmethod
    def predict(cls, probabilities: dict[str, float]) -> "RiskPrediction":
        if not probabilities:
            return RiskPrediction(risk_class=RiskClass.LOW, probabilities={})
        best = max(probabilities, key=lambda k: probabilities[k])
        return RiskPrediction(risk_class=RiskClass(best), probabilities=probabilities)


class RiskPrediction:
    def __init__(self, risk_class: RiskClass, probabilities: dict[str, float]) -> None:
        self.risk_class = risk_class
        self.probabilities = probabilities


class ModalityHead:
    @classmethod
    def predict(cls, probabilities: dict[str, float], threshold: float = 0.5) -> "ModalityPrediction":
        modalities = [ModalityLabel(k) for k, v in probabilities.items() if v >= threshold]
        return ModalityPrediction(modalities=modalities, scores=probabilities)


class ModalityPrediction:
    def __init__(self, modalities: list[ModalityLabel], scores: dict[str, float]) -> None:
        self.modalities = modalities
        self.scores = scores
