"""Shared helper for training a tiny transformer compressor in tests."""

from __future__ import annotations

from pathlib import Path

from tekmyra.learned_compressor.transformer_model import (
    TransformerCompressorModel,
    TransformerConfig,
)

_KEEP_SENTENCES = [
    "Critical update: revenue increased by twelve percent year over year.",
    "Important: customer churn remained low across all market segments.",
    "Key result: the product launch exceeded sales expectations.",
    "Major milestone: operating margins improved significantly.",
    "Alert: the board will meet next Tuesday to review the plan.",
]

_DROP_SENTENCES = [
    "The weather was nice and the sky was blue today.",
    "Birds sang loudly from the branches of the old oak tree.",
    "A gentle breeze moved through the tall grass by the river.",
    "The cat slept peacefully on the windowsill all afternoon.",
    "Visitors paused to rest beside the cool mountain stream.",
]


def train_tiny_transformer_artifact(tmp_path: Path) -> Path:
    """Train and save a tiny transformer compressor, returning the artifact dir."""
    train_data = [(s, 1) for s in _KEEP_SENTENCES] + [(s, 0) for s in _DROP_SENTENCES]
    # Repeat the dataset so the tiny head sees enough examples in three epochs.
    sentences = [s for s, _ in train_data * 3]
    labels = [label for _, label in train_data * 3]

    config = TransformerConfig(
        model_name="prajjwal1/bert-tiny",
        max_length=128,
        hidden_size=128,
        device="cpu",
        version="2.0.0",
    )
    model = TransformerCompressorModel(config)
    model.fit(sentences, labels)

    artifact_dir = tmp_path / "transformer_compressor"
    model.save(artifact_dir)
    return artifact_dir
