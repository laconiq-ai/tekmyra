import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tekmyra.baselines import (
    BASELINES,
    DeterministicExtractiveBaseline,
    LLMLingua2Baseline,
    MissingOptionalDependency,
    TekMyraLearnedBaseline,
    RerankerChunkBaseline,
    run_baseline,
)
from tekmyra.pii_vocabulary import DETECTION_TYPES
from tekmyra.fixtures import load_fixtures
from tekmyra.learned_compressor.compressor import ProseCompressor

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "fixtures" / "synthetic"
LONG_CONTEXT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "fixtures" / "long_context_v1"


def test_baseline_registry():
    assert set(BASELINES.keys()) == {
        "no_compression",
        "passthrough",
        "deterministic",
        "deterministic_extractive",
        "reranker_chunk",
        "llmlingua2",
        "tekmyra_learned",
        "nova_learned",
        "learned",
    }


def test_no_compression_baseline():
    fixtures = load_fixtures(FIXTURE_DIR)
    fx = fixtures[0]
    resp = run_baseline("no_compression", fx)
    assert resp.global_metrics.compression_ratio == 1.0
    assert resp.compressed_segments[0].compressed_content == fx.content
    assert resp.observed_result is not None
    assert resp.observed_result.learned_compressor_invoked is False
    assert resp.observed_result.protected_disposition_valid is True


def test_deterministic_extractive_baseline():
    fixtures = load_fixtures(LONG_CONTEXT_DIR)
    fx = next(f for f in fixtures if f.fixture_id.startswith("lc-"))
    baseline = DeterministicExtractiveBaseline()
    resp = baseline.run(fx, target_ratio=0.82)
    assert resp.global_metrics.compression_ratio <= 1.0
    assert resp.global_metrics.compression_ratio > 0.0
    assert resp.observed_result is not None
    assert resp.observed_result.learned_compressor_invoked is False
    assert resp.observed_result.required_answer_fact_recall is not None


def test_deterministic_extractive_preserves_protected_spans():
    fixtures = load_fixtures(LONG_CONTEXT_DIR)
    fx = next(f for f in fixtures if f.protected_spans)
    baseline = DeterministicExtractiveBaseline()
    resp = baseline.run(fx, target_ratio=0.82)
    compressed = resp.compressed_segments[0].compressed_content
    for span in fx.protected_spans:
        assert span.text in compressed


def test_reranker_chunk_baseline():
    fixtures = load_fixtures(LONG_CONTEXT_DIR)
    fx = next(f for f in fixtures if f.fixture_id.startswith("lc-"))
    baseline = RerankerChunkBaseline()
    resp = baseline.run(fx, target_ratio=0.82)
    assert 0.0 < resp.global_metrics.compression_ratio <= 1.0
    assert resp.observed_result is not None
    assert resp.observed_result.learned_compressor_invoked is False


def test_reranker_embedding_model_is_loaded_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = 0
    revisions: list[str | None] = []

    class FakeSentenceTransformer:
        def __init__(self, _model_name: str, *, revision: str | None = None) -> None:
            nonlocal loads
            loads += 1
            revisions.append(revision)

    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    baseline = RerankerChunkBaseline()

    first = baseline._load_once()
    second = baseline._load_once()

    assert first is second
    assert loads == 1
    assert revisions == ["1110a243fdf4706b3f48f1d95db1a4f5529b4d41"]


def test_reranker_held_out_ratios_stay_in_checkpoint_tolerance() -> None:
    split = json.loads(
        (LONG_CONTEXT_DIR / "held_out_split.json").read_text(encoding="utf-8")
    )
    held_out = set(split["held_out"])
    fixtures = [
        fixture
        for fixture in load_fixtures(LONG_CONTEXT_DIR)
        if fixture.fixture_id in held_out
    ]
    baseline = RerankerChunkBaseline()
    baseline._embedding_model_loaded = True
    baseline._embedding_model = None

    for fixture in fixtures:
        response = baseline.run(fixture, target_ratio=0.82)
        assert 0.77 <= response.global_metrics.compression_ratio <= 0.87
        assert response.observed_result is not None
        assert response.observed_result.protected_disposition_valid is True
        assert response.observed_result.introduced_duplicate_count == 0
        assert response.observed_result.required_answer_fact_recall == 1.0


def test_llmlingua2_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """When llmlingua is not importable the baseline raises MissingOptionalDependency."""
    import sys

    monkeypatch.setitem(sys.modules, "llmlingua", None)
    fixtures = load_fixtures(FIXTURE_DIR)
    fx = fixtures[0]
    baseline = LLMLingua2Baseline()
    with pytest.raises(MissingOptionalDependency):
        baseline.run(fx, target_ratio=0.82)


def test_llmlingua2_model_is_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = 0

    class FakePromptCompressor:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal loads
            loads += 1

    monkeypatch.setitem(
        __import__("sys").modules,
        "llmlingua",
        SimpleNamespace(PromptCompressor=FakePromptCompressor),
    )
    baseline = LLMLingua2Baseline()

    first = baseline._load_once()
    second = baseline._load_once()

    assert first is second
    assert loads == 1


def test_tekmyra_learned_fail_closed_without_env():
    fixtures = load_fixtures(LONG_CONTEXT_DIR)
    fx = next(f for f in fixtures if f.fixture_id.startswith("lc-"))
    baseline = TekMyraLearnedBaseline()
    resp = baseline.run(fx, target_ratio=0.82)
    assert resp.observed_result is not None
    assert resp.observed_result.learned_compressor_invoked is False


def test_tekmyra_learned_fail_closed_with_short_fixture():
    fixtures = load_fixtures(FIXTURE_DIR)
    fx = fixtures[0]
    baseline = TekMyraLearnedBaseline()
    # Short fixture should not invoke learned model even if flag were set.
    resp = baseline.run(fx, target_ratio=0.82)
    assert resp.observed_result is not None
    assert resp.observed_result.learned_compressor_invoked is False


def test_tekmyra_learned_does_not_claim_invocation_without_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures = load_fixtures(LONG_CONTEXT_DIR)
    fx = next(
        fixture
        for fixture in fixtures
        if fixture.risk_tier.value in {"low", "medium"}
        and len(fixture.content.split()) >= 2000
        # Mirrors TekMyraLearnedBaseline's own PII gate. It REIMPLEMENTS that
        # gate rather than calling it, so the two can silently disagree — at
        # minimum it must not also carry its own copy of the vocabulary, nor
        # the retired pii_ hatch that production no longer honours.
        and not any(
            span.span_type in DETECTION_TYPES for span in fixture.protected_spans
        )
    )
    monkeypatch.setenv("TEKMYRA_LEARNED_ENABLED", "true")
    monkeypatch.setattr(
        "tekmyra.learned_compressor.compressor.load_compressor",
        lambda: ProseCompressor(None),
    )

    response = TekMyraLearnedBaseline().run(fx, target_ratio=0.82)

    assert response.observed_result is not None
    assert response.observed_result.learned_compressor_invoked is False
    assert response.global_metrics.compression_ratio == 1.0


def test_deprecated_nova_flag_and_baseline_id_remain_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures = load_fixtures(FIXTURE_DIR)
    monkeypatch.setenv("NOVA_LEARNED_ENABLED", "true")
    with pytest.warns(DeprecationWarning, match="TEKMYRA_LEARNED_ENABLED"):
        response = run_baseline("nova_learned", fixtures[0], target_ratio=0.82)
    assert BASELINES["nova_learned"] is TekMyraLearnedBaseline
    assert response.observed_result is not None
    assert response.observed_result.learned_compressor_invoked is False
