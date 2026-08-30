import pytest

from tekmyra.contracts import (
    FeatureVector,
    LearnedDecision,
    ModalityLabel,
    PolicyTier,
    RouteClass,
    RouterInput,
    SourceChannel,
)
from tekmyra.learned_router.config import LearnedRouterConfig
from tekmyra.learned_router.features import build_router_input, numeric_feature_vector
from tekmyra.prober import probe_text


def test_router_input_requires_text_or_features():
    with pytest.raises(ValueError):
        RouterInput(artifact_id="a", text=None, feature_vector=None)


def test_router_input_accepts_text_only():
    inp = RouterInput(artifact_id="a", text="hello")
    assert inp.text == "hello"
    assert inp.feature_vector is None


def test_router_input_accepts_feature_vector_only():
    fv = FeatureVector(
        token_len=10,
        char_len=50,
        char_entropy=3.0,
        json_parse_success=False,
        pii_entity_count=0,
        legal_keyword_count=0,
        security_keyword_count=0,
        governance_keyword_count=0,
        source_channel=SourceChannel.API,
        policy_tier=PolicyTier.STANDARD,
    )
    inp = RouterInput(artifact_id="a", feature_vector=fv)
    assert inp.feature_vector is fv
    assert inp.text is None


def test_router_input_rejects_empty_artifact_id():
    with pytest.raises(ValueError):
        RouterInput(artifact_id="", text="hello")


def test_router_input_rejects_extra_fields():
    with pytest.raises(ValueError):
        RouterInput(artifact_id="a", text="hello", unknown_field=123)


def test_learned_decision_defaults_to_abstain():
    d = LearnedDecision(artifact_id="a")
    assert d.abstained is True
    assert d.route is None


def test_learned_decision_accepts_route_assignment():
    d = LearnedDecision(artifact_id="a", route=RouteClass.PASSTHROUGH)
    assert d.route == RouteClass.PASSTHROUGH


def test_learned_decision_rejects_extra_fields():
    with pytest.raises(ValueError):
        LearnedDecision(artifact_id="a", extra_field=True)


def test_learned_decision_modalities_are_modality_labels():
    d = LearnedDecision(
        artifact_id="a", modalities=[ModalityLabel.PROSE, ModalityLabel.CODE]
    )
    assert d.modalities == [ModalityLabel.PROSE, ModalityLabel.CODE]


def test_config_loads_default_yaml():
    cfg = LearnedRouterConfig.from_yaml("configs/model_router_small.yaml")
    assert cfg.max_router_tokens > 0
    assert cfg.route_classes


def test_config_default_instantiation():
    cfg = LearnedRouterConfig(
        route_classes=["a"],
        risk_classes=["b"],
        modalities=["c"],
    )
    assert cfg.max_router_tokens == 512
    assert cfg.temperature == 1.0


def test_config_rejects_empty_route_classes():
    with pytest.raises(ValueError):
        LearnedRouterConfig(route_classes=[], risk_classes=["b"], modalities=["c"])


def test_config_rejects_empty_risk_classes():
    with pytest.raises(ValueError):
        LearnedRouterConfig(route_classes=["a"], risk_classes=[], modalities=["c"])


def test_config_rejects_empty_modalities():
    with pytest.raises(ValueError):
        LearnedRouterConfig(route_classes=["a"], risk_classes=["b"], modalities=[])


def test_config_validates_max_router_tokens_bounds():
    with pytest.raises(ValueError):
        LearnedRouterConfig(
            route_classes=["a"],
            risk_classes=["b"],
            modalities=["c"],
            max_router_tokens=1,
        )


def test_config_validates_temperature_bounds():
    with pytest.raises(ValueError):
        LearnedRouterConfig(
            route_classes=["a"],
            risk_classes=["b"],
            modalities=["c"],
            temperature=0.0,
        )


def test_config_round_trip_model_validate():
    cfg = LearnedRouterConfig.from_yaml("configs/model_router_small.yaml")
    data = cfg.model_dump(mode="json")
    cfg2 = LearnedRouterConfig.model_validate(data)
    assert cfg2.name == cfg.name
    assert cfg2.route_classes == cfg.route_classes


def test_head_tail_window_respects_token_budget():
    text = "word " * 2000
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    inp = build_router_input("a", text, features, max_tokens=64, head_tail_ratio=0.5)
    tokens = inp.text.split()
    assert len(tokens) <= 64 + 2
    assert "..." in inp.text or len(tokens) < 100


def test_metadata_tokens_include_channel_and_policy():
    text = "Some ambiguous prose."
    features = probe_text(text, SourceChannel.TICKET, PolicyTier.REGULATED)
    inp = build_router_input("a", text, features)
    assert any("ticket" in t for t in inp.metadata_tokens)
    assert any("regulated" in t for t in inp.metadata_tokens)


def test_numeric_feature_vector_length_matches_schema():
    text = "hello world"
    features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
    numeric = numeric_feature_vector(features)
    assert len(numeric) > 0
    assert all(isinstance(v, float) for v in numeric)
