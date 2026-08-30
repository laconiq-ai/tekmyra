"""Tiny scikit-learn keep/drop model for sentence compression."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _assert_classifier_converged(classifier: Any) -> None:
    """Fail closed if the learned compressor exhausts its iteration budget."""
    iterations: Any = np.asarray(getattr(classifier, "n_iter_", ()), dtype=int)
    max_iter = getattr(classifier, "max_iter", None)
    if iterations.size == 0 or not isinstance(max_iter, int) or max_iter <= 0:
        raise RuntimeError("compressor classifier convergence state is unavailable")
    if np.any(iterations >= max_iter):
        raise RuntimeError(
            "compressor classifier did not converge below "
            f"max_iter={max_iter}: n_iter={iterations.tolist()}"
        )


@dataclass(frozen=True)
class KeepDropScores:
    keep_prob: float
    drop_prob: float


class SentenceCompressorModel:
    """Logistic-regression sentence keep/drop classifier."""

    def __init__(self) -> None:
        self._vectorizer: Any = None
        self._clf: Any = None
        self._version = "1.0.0"

    def fit(self, sentences: list[str], labels: list[int]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

        if not sentences or not labels:
            raise ValueError("sentences and labels must not be empty")
        if len(set(labels)) < 2:
            # Fall back to a trivial always-keep model when all labels agree.
            self._clf = None
            self._vectorizer = None
            return

        self._vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X = self._vectorizer.fit_transform(sentences)
        self._clf = LogisticRegression(max_iter=1000, random_state=0)
        self._clf.fit(X, labels)
        _assert_classifier_converged(self._clf)

    def predict(self, sentences: list[str]) -> list[KeepDropScores]:
        if not sentences:
            return []
        if self._clf is None or self._vectorizer is None:
            return [KeepDropScores(keep_prob=1.0, drop_prob=0.0) for _ in sentences]

        X = self._vectorizer.transform(sentences)
        probs = self._clf.predict_proba(X)
        # Determine which column is keep (label 1).
        keep_idx = list(self._clf.classes_).index(1) if 1 in self._clf.classes_ else 1
        drop_idx = 1 - keep_idx
        return [
            KeepDropScores(keep_prob=float(p[keep_idx]), drop_prob=float(p[drop_idx]))
            for p in probs
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self._vectorizer is not None and hasattr(
            self._vectorizer, "_stop_words_id"
        ):
            self._vectorizer._stop_words_id = None
        artifact = {
            "vectorizer": self._vectorizer,
            "clf": self._clf,
            "version": self._version,
        }
        with open(directory / "compressor.joblib", "wb") as fh:
            pickle.dump(artifact, fh)
        with open(directory / "metadata.json", "w") as fh:
            json.dump({"version": self._version, "kind": "sentence_compressor"}, fh)

    @classmethod
    def load(cls, directory: Path) -> "SentenceCompressorModel":
        return cls.load_bytes((directory / "compressor.joblib").read_bytes())

    @classmethod
    def load_bytes(cls, payload: bytes) -> "SentenceCompressorModel":
        """Load exactly the caller-verified serialized bytes.

        Deployed callers use this path so signature/hash verification and
        deserialization cannot be separated by a filesystem path TOCTOU.
        """

        artifact = pickle.loads(payload)
        if not isinstance(artifact, dict):
            raise ValueError("compressor artifact payload must be a mapping")
        model = cls()
        model._vectorizer = artifact["vectorizer"]
        model._clf = artifact["clf"]
        model._version = artifact.get("version", "1.0.0")
        return model


class OnnxCompressorModel:
    """ONNX-backed keep/drop sentence compressor."""

    def __init__(self, session, vectorizer, version: str = "1.0.0") -> None:
        from tekmyra.onnx_runtime import OnnxSessionManager

        self._session = (
            session
            if isinstance(session, OnnxSessionManager)
            else OnnxSessionManager(session)
        )
        self._vectorizer = vectorizer
        self._version = version

    def predict(self, sentences: list[str]) -> list[KeepDropScores]:
        if not sentences:
            return []
        if not self._session.loaded:
            return [KeepDropScores(keep_prob=1.0, drop_prob=0.0) for _ in sentences]

        import scipy.sparse as sp  # type: ignore[import-untyped]

        X = self._vectorizer.transform(sentences)
        if sp.issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)
        outputs = self._session.run({"float_input": X})
        probs = list(outputs.values())[1]

        if probs.ndim == 1:
            return [KeepDropScores(keep_prob=float(probs[i]), drop_prob=1.0 - float(probs[i])) for i in range(len(sentences))]

        # Map columns: label 1 (keep) should have higher probability for kept samples.
        keep_idx = 1 if probs.shape[1] > 1 else 0
        drop_idx = 1 - keep_idx
        return [
            KeepDropScores(keep_prob=float(p[keep_idx]), drop_prob=float(p[drop_idx]))
            for p in probs
        ]
