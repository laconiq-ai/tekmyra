from types import SimpleNamespace

import numpy as np
import pytest

from tekmyra.contracts import RiskClass
from tekmyra.learned_router.calibration import AbstentionPolicy, TemperatureScaler
from tekmyra.contracts import LearnedDecision
from tekmyra.learned_router.config import LearnedRouterConfig
from tekmyra.learned_router.heads import RouteHead
from tekmyra.learned_router.model import (
    ModelManager,
    SklearnRouterModel,
    _assert_classifier_converged,
)
from tekmyra.learned_router.router import LearnedRouter


def test_route_head_maps_probabilities_to_enum():
    classes = [r.value for r in RouteHead.allowed_routes()]
    probs = {c: 0.0 for c in classes}
    probs[classes[0]] = 1.0
    pred = RouteHead.predict(probs)
    assert pred.route == RouteHead.allowed_routes()[0]


def test_temperature_scaler_leaves_uniform():
    ts = TemperatureScaler(1.0)
    logits = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    scaled = ts.scale(logits)
    np.testing.assert_allclose(scaled[0], 1.0 / 3.0, atol=0.01)


def test_abstention_policy_abstains_low_confidence():
    thresholds = {risk: 0.9 for risk in RiskClass}
    policy = AbstentionPolicy(thresholds=thresholds, min_margin=0.05)
    decision = policy.decide(
        top_class="a",
        top_prob=0.6,
        second_prob=0.55,
        risk_class=RiskClass.LOW,
    )
    assert decision.abstain is True


def test_classifier_convergence_guard_accepts_completed_fit():
    classifier = SimpleNamespace(n_iter_=np.array([41]), max_iter=5000)
    _assert_classifier_converged(classifier, head="route")


def test_classifier_convergence_guard_rejects_iteration_cap():
    classifier = SimpleNamespace(n_iter_=np.array([5000]), max_iter=5000)
    with pytest.raises(RuntimeError, match="did not converge"):
        _assert_classifier_converged(classifier, head="route")


def test_sklearn_model_predicts_shape():
    cfg = LearnedRouterConfig(
        route_classes=["passthrough", "prose_token_compressor"],
        risk_classes=["low", "medium"],
        modalities=["prose"],
    )
    texts = ["hello world", "foo bar baz"]
    numeric = [[10.0] * 22, [5.0] * 22]
    model = SklearnRouterModel(cfg)
    model.fit(texts, numeric, routes=["passthrough", "prose_token_compressor"])
    out = model.predict(["hello"], [[1.0] * 22])
    assert set(out.route_probs.keys()) == set(cfg.route_classes)
    assert set(out.risk_probs.keys()) == set(cfg.risk_classes)


def test_model_manager_abstains_when_no_artifact(tmp_path):
    cfg = LearnedRouterConfig(
        route_classes=["passthrough"],
        risk_classes=["low"],
        modalities=["prose"],
        model_artifact_dir=tmp_path,
    )
    mgr = ModelManager(cfg)
    model = mgr.load()
    assert model is None


def test_learned_router_abstains_without_model(tmp_path):
    cfg = LearnedRouterConfig(
        route_classes=["passthrough", "prose_token_compressor"],
        risk_classes=["low", "medium"],
        modalities=["prose"],
        model_artifact_dir=tmp_path,
    )
    router = LearnedRouter(cfg)
    decision = router.route("a", text="hello", numeric=[1.0] * 22)
    assert isinstance(decision, LearnedDecision)
    assert decision.abstained is True


def test_learned_router_routes_when_model_present(tmp_path):
    cfg = LearnedRouterConfig(
        route_classes=["passthrough", "prose_token_compressor"],
        risk_classes=["low", "medium"],
        modalities=["prose"],
        model_artifact_dir=tmp_path,
        confidence_threshold_low=0.0,
        min_margin_for_abstention=0.0,
    )
    model = SklearnRouterModel(cfg)
    model.fit(
        texts=["hello world", "foo bar baz"],
        numeric=[[1.0] * 22, [2.0] * 22],
        routes=["passthrough", "prose_token_compressor"],
        risks=["low", "medium"],
        modalities=[["prose"], ["prose"]],
    )
    model.save(tmp_path / cfg.version)

    router = LearnedRouter(cfg)
    decision = router.route("a", text="hello world", numeric=[1.0] * 22)
    assert isinstance(decision, LearnedDecision)
    assert decision.route is not None
    assert decision.abstained is False
    assert decision.confidence_calibrated > 0
