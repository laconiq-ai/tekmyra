from pathlib import Path

import pytest
from pydantic import ValidationError

from tekmyra.contracts import EligibilityReason, RiskClass
from tekmyra.fixtures import Fixture, load_fixtures


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "fixtures" / "synthetic"


def _minimal_fixture(**overrides):
    data = {
        "fixture_id": "test-000",
        "schema_version": "2.1.0",
        "content_type": "prose",
        "risk_tier": "low",
        "content": "test content",
        "route_labels": ["passthrough"],
        "protected_spans": [],
        "gold_output": {
            "route_intent": "passthrough",
            "allowed_outcomes": ["passthrough"],
            "expected_route": "passthrough",
            "expected_risk": "low",
        },
        "provenance": {
            "source": "test",
            "license": "CC0-1.0",
            "generated_by": "test_fixtures",
        },
        "tags": [],
    }
    data.update(overrides)
    return data


def test_load_all_fixtures():
    fixtures = load_fixtures(FIXTURE_DIR)
    assert len(fixtures) >= 50  # WP-0 expanded synthetic dir to 86 fixtures
    for fx in fixtures:
        assert fx.fixture_id
        assert fx.content.strip()
        assert fx.schema_version in ("2.0.0", "2.1.0")
        assert fx.gold_output.allowed_outcomes
        assert fx.gold_output.route_intent in fx.gold_output.allowed_outcomes


def test_fixture_risk_tiers_are_valid():
    fixtures = load_fixtures(FIXTURE_DIR)
    tiers = {f.risk_tier.value for f in fixtures}
    assert tiers <= {"low", "medium", "high", "critical"}


def test_high_risk_fixtures_have_protected_spans():
    fixtures = load_fixtures(FIXTURE_DIR)
    for fx in fixtures:
        if fx.risk_tier.value in ("high", "critical"):
            assert fx.protected_spans, f"{fx.fixture_id} lacks protected spans"


def test_ambiguous_fixtures_load():
    path = FIXTURE_DIR.parent / "learned_train"
    if path.exists():
        fixtures = load_fixtures(path)
        assert len(fixtures) >= 50
        for fx in fixtures:
            assert fx.fixture_id.startswith("ambig-")
            assert fx.gold_output.expected_route
            assert fx.schema_version == "2.1.0"
            assert fx.gold_output.route_intent in fx.gold_output.allowed_outcomes


def test_fixture_blocks_and_adjudication_loaded():
    fixtures = load_fixtures(FIXTURE_DIR)
    for fx in fixtures:
        assert isinstance(fx.adjudicated, bool)
        if fx.adjudicated:
            assert fx.adjudicated_at is not None
        for block in fx.blocks:
            assert block.type
            assert block.route_label
            assert block.risk_tier in RiskClass


def test_legacy_fixture_without_blocks_defaults():
    legacy = {
        "fixture_id": "legacy-000",
        "schema_version": "2.0.0",
        "content_type": "prose",
        "risk_tier": "low",
        "content": "legacy content",
        "route_labels": ["passthrough"],
        "protected_spans": [],
        "gold_output": {
            "route_intent": "passthrough",
            "allowed_outcomes": ["passthrough"],
            "expected_route": "passthrough",
            "expected_risk": "low",
        },
        "provenance": {
            "source": "test",
            "license": "CC0-1.0",
            "generated_by": "test",
        },
        "tags": [],
    }
    fx = Fixture.model_validate(legacy)
    assert fx.blocks == []
    assert fx.adjudicated is False
    assert fx.adjudicated_at is None
    assert fx.provenance.adjudicated_by is None


@pytest.mark.parametrize(
    "field,mutation",
    [
        ("missing route_intent", lambda d: d["gold_output"].pop("route_intent")),
        (
            "allowed_outcomes missing intent",
            lambda d: d["gold_output"].update(
                {"route_intent": "passthrough", "allowed_outcomes": ["code_compressor"]}
            ),
        ),
        (
            "conflicting flags",
            lambda d: d["gold_output"].update(
                {"passthrough": True, "redaction_only": True}
            ),
        ),
        (
            "compressible false without flag",
            lambda d: d["gold_output"].update(
                {"compressible": False, "passthrough": False, "redaction_only": False}
            ),
        ),
    ],
)
def test_route_intent_contract_validation_failures(field, mutation):
    data = _minimal_fixture()
    mutation(data)
    with pytest.raises(ValidationError):
        Fixture.model_validate(data)


def test_pii_fixture_defaults_to_non_compressible():
    data = _minimal_fixture(
        risk_tier="high",
        route_labels=["pii_redaction_before_compression"],
        gold_output={
            "route_intent": "pii_redaction_before_compression",
            "allowed_outcomes": [
                "pii_redaction_before_compression",
                "reject_or_human_review",
            ],
            "expected_route": "pii_redaction_before_compression",
            "expected_risk": "high",
            "must_preserve_spans": [
                {
                    "span_type": "email",  # was "pii_email" — a convention no fixture uses
                    "start": 0,
                    "end": 5,
                    "text": "a@b.c",
                    "policy": "preserve",
                    "detector": "gold",
                    "confidence": 1.0,
                }
            ],
        },
    )
    fx = Fixture.model_validate(data)
    assert fx.gold_output.compressible is False
    assert fx.gold_output.redaction_only is True


def test_legacy_fixture_without_schema_version_defaults_to_2_1_0():
    data = _minimal_fixture()
    data.pop("schema_version")
    fx = Fixture.model_validate(data)
    assert fx.schema_version == "2.1.0"


def test_invalid_schema_version_fails_validation():
    data = _minimal_fixture()
    data["schema_version"] = "1.0.0"
    with pytest.raises(ValidationError):
        Fixture.model_validate(data)


def test_fixture_gold_output_extra_fields_forbidden():
    data = _minimal_fixture()
    data["gold_output"]["unknown_field"] = "x"
    with pytest.raises(ValidationError):
        Fixture.model_validate(data)


def test_fixture_2_1_0_round_trip_exposes_new_gold_fields():
    data = _minimal_fixture(
        schema_version="2.1.0",
        gold_output={
            "route_intent": "passthrough",
            "allowed_outcomes": ["passthrough"],
            "expected_route": "passthrough",
            "expected_risk": "low",
            "expected_compression_eligibility": False,
            "expected_allowed_bypass_reasons": ["too_short"],
            "query_task": "Summarize the provided text.",
            "answer_required_fact_ids": ["fact-001"],
            "allowed_safety_overrides": ["allow_passthrough_short"],
            "must_preserve_spans": [
                {
                    "span_type": "fact",
                    "start": 0,
                    "end": 11,
                    "text": "test content",
                    "span_id": "span-001",
                    "fact_id": "fact-001",
                }
            ],
        },
        protected_fact_index={"fact-001": "test content"},
    )
    fx = Fixture.model_validate(data)
    assert fx.schema_version == "2.1.0"
    assert fx.gold_output.expected_compression_eligibility is False
    assert fx.gold_output.expected_allowed_bypass_reasons == [EligibilityReason.TOO_SHORT]
    assert fx.gold_output.query_task == "Summarize the provided text."
    assert fx.gold_output.answer_required_fact_ids == ["fact-001"]
    assert fx.gold_output.allowed_safety_overrides == ["allow_passthrough_short"]
    assert fx.gold_output.must_preserve_spans[0].span_id == "span-001"
    assert fx.gold_output.must_preserve_spans[0].fact_id == "fact-001"
    assert fx.protected_fact_index == {"fact-001": "test content"}
    dumped = fx.model_dump(mode="json")
    restored = Fixture.model_validate(dumped)
    assert restored.gold_output.query_task == fx.gold_output.query_task
    assert restored.protected_fact_index == fx.protected_fact_index


def test_existing_2_0_0_synthetic_fixtures_still_load():
    fixtures = load_fixtures(FIXTURE_DIR)
    fixtures_2_0_0 = [fx for fx in fixtures if fx.schema_version == "2.0.0"]
    assert len(fixtures_2_0_0) >= 1
    for fx in fixtures_2_0_0:
        assert fx.gold_output.expected_compression_eligibility is None
        assert fx.gold_output.expected_allowed_bypass_reasons == []
        assert fx.protected_fact_index == {}
