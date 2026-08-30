"""ONNX inference wrapper for transformer sentence compressor models.

This module keeps the runtime dependency on ``torch`` optional: only
``transformers`` (for the tokenizer) and ``onnxruntime`` are required when
a transformer compressor is deployed as an ONNX artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tekmyra.learned_compressor.model import KeepDropScores
from tekmyra.onnx_runtime import OnnxSessionManager


class OnnxTransformerCompressorModel:
    """ONNX-backed keep/drop sentence compressor using a transformer encoder."""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self._session = OnnxSessionManager(model_path)
        self._tokenizer = tokenizer
        self._max_length = max_length

    def predict(self, sentences: list[str]) -> list[KeepDropScores]:
        """Return keep/drop probabilities for each input sentence."""
        if not sentences:
            return []

        if not self._session.loaded or self._tokenizer is None:
            return [
                KeepDropScores(keep_prob=1.0, drop_prob=0.0)
                for _ in sentences
            ]

        encoded = self._tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="np",
        )
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)

        outputs = self._session.run({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })
        logits = outputs["logits"]
        probs = self._softmax(logits)

        return [
            KeepDropScores(keep_prob=float(p[1]), drop_prob=float(p[0]))
            for p in probs
        ]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)
