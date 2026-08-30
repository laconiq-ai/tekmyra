"""Tests for tekmyra.learned_compressor.transformer_model."""

from __future__ import annotations

import math

import pytest

# THE SUITE MUST PASS WITH ONLY THE `dev` EXTRA INSTALLED, because `dev` is
# what the payload offers as the way to run it. `transformer_model` imports
# torch and transformers at module level, and torch lives in the OPTIONAL
# `transformer` extra by design — so importing this module unguarded made a
# clean install abort COLLECTION, which is not this file skipping, it is the
# entire suite running zero tests and `verify_open_core.py` refusing a healthy
# payload.
#
# Module level rather than inside each test, which is where the other eight
# guarded files put it, because the failing import is itself at module level:
# a guard inside the test bodies runs far too late to stop collection.
pytest.importorskip("torch")

from tekmyra.learned_compressor.transformer_model import (  # noqa: E402
    TransformerCompressorModel,
    TransformerConfig,
)


def test_predict_without_load_returns_keep() -> None:
    """Before load_pretrained/fit is called predict should always-keep."""
    model = TransformerCompressorModel(TransformerConfig())
    scores = model.predict(["hello world", "foo bar"])
    assert len(scores) == 2
    assert all(s.keep_prob == 1.0 for s in scores)
    assert all(s.drop_prob == 0.0 for s in scores)


def test_predict_empty_sentences() -> None:
    """Predicting an empty list should return an empty list."""
    model = TransformerCompressorModel(TransformerConfig())
    assert model.predict([]) == []


def test_fit_single_class_labels_fallback() -> None:
    """Fitting with a single class clears the head and predicts always-keep."""
    model = TransformerCompressorModel(TransformerConfig())
    model.fit(["a sentence", "another sentence"], [1, 1])
    assert model.head is None
    scores = model.predict(["something else"])
    assert len(scores) == 1
    assert scores[0].keep_prob == 1.0


def test_fit_requires_nonempty_inputs() -> None:
    """fit must reject empty sentences or labels."""
    model = TransformerCompressorModel(TransformerConfig())
    with pytest.raises(ValueError):
        model.fit([], [])
    with pytest.raises(ValueError):
        model.fit(["only sentence"], [])
    with pytest.raises(ValueError):
        model.fit([], [1])


def test_fit_requires_matching_lengths() -> None:
    """fit must reject mismatched sentence and label counts."""
    model = TransformerCompressorModel(TransformerConfig())
    with pytest.raises(ValueError):
        model.fit(["one", "two"], [1])


@pytest.mark.slow
def test_fit_predict_save_load_roundtrip(tmp_path) -> None:
    """End-to-end training, prediction, save and load with ModernBERT.

    This test exercises the full pipeline.  It does not assert perfect
    classification because the tiny dataset is not enough to guarantee
    separation of the two synthetic classes.
    """
    model = TransformerCompressorModel(
        TransformerConfig(model_name="answerdotai/ModernBERT-base")
    )

    keep_sentences = [
        "Quarterly revenue increased by twelve percent year over year.",
        "The company expanded into three new international markets.",
        "Customer retention reached a record high this quarter.",
        "Product launch exceeded sales expectations by a wide margin.",
    ]
    drop_sentences = [
        "um well like you know maybe sort of",
        "anyway so yeah I was just thinking whatever",
        "like literally no way I mean whatever",
        "so um yeah basically just stuff and things",
    ]
    sentences = keep_sentences + drop_sentences
    labels = [1] * len(keep_sentences) + [0] * len(drop_sentences)

    model.fit(sentences, labels)

    scores = model.predict(sentences)
    assert len(scores) == len(sentences)
    for s in scores:
        assert 0.0 <= s.keep_prob <= 1.0
        assert 0.0 <= s.drop_prob <= 1.0
        assert math.isclose(s.keep_prob + s.drop_prob, 1.0, rel_tol=1e-5)

    save_dir = tmp_path / "transformer_compressor"
    model.save(save_dir)

    assert (save_dir / "encoder").exists()
    assert (save_dir / "config.json").exists()
    assert (save_dir / "metadata.json").exists()
    assert (save_dir / "head.pt").exists()

    loaded_model = TransformerCompressorModel.load(save_dir)
    loaded_scores = loaded_model.predict(sentences)
    assert len(loaded_scores) == len(sentences)
    for original, loaded_score in zip(scores, loaded_scores):
        assert math.isclose(original.keep_prob, loaded_score.keep_prob, rel_tol=1e-5)
        assert math.isclose(original.drop_prob, loaded_score.drop_prob, rel_tol=1e-5)
