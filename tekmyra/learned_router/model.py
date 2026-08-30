from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tekmyra.learned_router.config import LearnedRouterConfig
from tekmyra.onnx_runtime import OnnxSessionManager

logger = logging.getLogger("tekmyra.learned_router.model")

AnyEstimator = Any
AnyEncoder = Any
AnyVectorizer = Any

# Fitted properties of a trained model, not runtime policy knobs. These are read
# FROM the artifact rather than asserted against it: requiring the shipped config
# to restate a value produced by training is an unenforceable manual obligation,
# and getting it wrong degraded every run to rules-only while still reporting a
# ratio (2026-08-07).
_ARTIFACT_DERIVED_FIELDS = ("temperature", "calibration_method")


class RouterArtifactLoadError(RuntimeError):
    """An artifact is present at the configured path but could not be loaded.

    Deliberately distinct from "no artifact at all". A missing artifact is a
    legitimate deployment state and abstaining is the documented fail-closed
    behaviour. An artifact that is PRESENT and unloadable means someone put a
    model there and it silently did not take effect — that must be loud. An
    equality check whose failure is swallowed is worse than no check: it yields
    the confident wrong answer rather than the missing one.
    """


def _stabilize_vectorizer_pickle(vectorizer: AnyVectorizer) -> None:
    """Remove process-local cache identities from nested text vectorizers."""
    if vectorizer is None:
        return
    if hasattr(vectorizer, "_stop_words_id"):
        vectorizer._stop_words_id = None
    for _name, transformer in getattr(vectorizer, "transformer_list", ()):  # type: ignore[union-attr]
        _stabilize_vectorizer_pickle(transformer)


def _assert_classifier_converged(classifier: AnyEstimator, *, head: str) -> None:
    """Fail closed when sklearn exhausts the configured iteration budget."""
    iterations: Any = np.asarray(getattr(classifier, "n_iter_", ()), dtype=int)
    max_iter = getattr(classifier, "max_iter", None)
    if iterations.size == 0 or not isinstance(max_iter, int) or max_iter <= 0:
        raise RuntimeError(f"{head} classifier convergence state is unavailable")
    if np.any(iterations >= max_iter):
        raise RuntimeError(
            f"{head} classifier did not converge below max_iter={max_iter}: "
            f"n_iter={iterations.tolist()}"
        )


@dataclass(frozen=True)
class RawModelOutput:
    route_probs: dict[str, float]
    risk_probs: dict[str, float]
    modality_scores: dict[str, float]
    model_version: str


class SklearnRouterModel:
    def __init__(self, config: LearnedRouterConfig) -> None:
        self.config = config
        self._route_clf: AnyEstimator = None
        self._route_encoder: AnyEncoder = None
        self._risk_clf: AnyEstimator = None
        self._risk_encoder: AnyEncoder = None
        self._modality_clf: AnyEstimator = None
        self._vectorizer: AnyVectorizer = None
        self._numeric_scaler: AnyEstimator = None

    def fit(
        self,
        texts: list[str],
        numeric: list[list[float]],
        routes: list[str],
        risks: list[str] | None = None,
        modalities: list[list[str]] | None = None,
    ) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
        from sklearn.multiclass import OneVsRestClassifier  # type: ignore[import-untyped]
        from sklearn.pipeline import FeatureUnion  # type: ignore[import-untyped]
        from sklearn.preprocessing import (  # type: ignore[import-untyped]
            LabelEncoder,
            MultiLabelBinarizer,
            StandardScaler,
        )

        # Word and character views are complementary here: word n-grams carry
        # semantic routing cues while character n-grams remain stable across
        # punctuation and envelope changes in previously unseen renderers.
        self._vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        max_features=10_000,
                        ngram_range=(1, 3),
                        strip_accents="unicode",
                        sublinear_tf=True,
                    ),
                ),
                (
                    "character",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        max_features=30_000,
                        ngram_range=(2, 6),
                        strip_accents="unicode",
                        sublinear_tf=True,
                    ),
                ),
            ],
            transformer_weights={"word": 2.0, "character": 1.0},
        )
        X_text = self._vectorizer.fit_transform(texts)
        # Keep sklearn training in float64 to match TF-IDF's native precision.
        # Premature float32 quantization can make the multiclass route head hit
        # its iteration cap even though the same problem converges cleanly at
        # full precision. ONNX input conversion remains explicitly float32 in
        # ``OnnxRouterModel`` below.
        self._numeric_scaler = StandardScaler()
        X_num = self._numeric_scaler.fit_transform(
            np.asarray(numeric, dtype=np.float64)
        )
        import scipy.sparse as sp  # type: ignore[import-untyped]
        X = sp.hstack([X_text, X_num])

        self._route_encoder = LabelEncoder()
        self._route_encoder.fit(self.config.route_classes)
        self._route_clf = LogisticRegression(max_iter=5000, random_state=0)
        self._route_clf.fit(X, self._route_encoder.transform(routes))
        _assert_classifier_converged(self._route_clf, head="route")

        if risks:
            self._risk_encoder = LabelEncoder()
            self._risk_encoder.fit(self.config.risk_classes)
            self._risk_clf = LogisticRegression(max_iter=5000, random_state=0)
            self._risk_clf.fit(X, self._risk_encoder.transform(risks))
            _assert_classifier_converged(self._risk_clf, head="risk")

        if modalities:
            mlb = MultiLabelBinarizer(classes=self.config.modalities)
            Y_mod = mlb.fit_transform(modalities)
            self._modality_clf = OneVsRestClassifier(
                LogisticRegression(max_iter=5000, random_state=0)
            )
            self._modality_clf.fit(X, Y_mod)
            for index, estimator in enumerate(self._modality_clf.estimators_):
                # OneVsRestClassifier deliberately substitutes sklearn's
                # constant predictor when a modality column has only one
                # observed value. No iterative optimization occurs there.
                if (
                    type(estimator).__module__ == "sklearn.multiclass"
                    and type(estimator).__name__ == "_ConstantPredictor"
                ):
                    continue
                _assert_classifier_converged(estimator, head=f"modality[{index}]")

    def predict(self, texts: list[str], numeric: list[list[float]]) -> RawModelOutput:
        if self._vectorizer is None or self._route_clf is None:
            raise RuntimeError("Model not fitted")
        import scipy.sparse as sp  # type: ignore[import-untyped]
        X_text = self._vectorizer.transform(texts)
        if self._numeric_scaler is None:
            raise RuntimeError("Numeric feature scaler not fitted")
        X_num = self._numeric_scaler.transform(
            np.asarray(numeric, dtype=np.float64)
        )
        X = sp.hstack([X_text, X_num])

        route_labels = self._route_encoder.inverse_transform(
            self._route_clf.classes_.astype(int)
        )
        route_probs = {label: 0.0 for label in self.config.route_classes}
        route_probs.update(
            {
                str(label): float(probability)
                for label, probability in zip(
                    route_labels, self._route_clf.predict_proba(X)[0]
                )
            }
        )
        risk_probs = {
            r: (1.0 if r == "low" else 0.0) for r in self.config.risk_classes
        }
        if self._risk_clf:
            risk_labels = self._risk_encoder.inverse_transform(
                self._risk_clf.classes_.astype(int)
            )
            risk_probs = {label: 0.0 for label in self.config.risk_classes}
            risk_probs.update(
                {
                    str(label): float(probability)
                    for label, probability in zip(
                        risk_labels, self._risk_clf.predict_proba(X)[0]
                    )
                }
            )
        modality_scores = {m: 0.0 for m in self.config.modalities}
        if self._modality_clf:
            mod_proba = self._modality_clf.predict_proba(X)
            # OneVsRestClassifier returns either a list of per-label binary
            # probability arrays or a single 2D array when there is one label.
            if isinstance(mod_proba, list):
                for idx, arr in enumerate(mod_proba):
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        modality_scores[self.config.modalities[idx]] = float(arr[0, 1])
                    else:
                        modality_scores[self.config.modalities[idx]] = float(arr[0])
            else:
                arr = mod_proba
                if arr.ndim == 2:
                    if arr.shape[1] == len(self.config.modalities):
                        for idx in range(arr.shape[1]):
                            modality_scores[self.config.modalities[idx]] = float(arr[0, idx])
                    elif arr.shape[1] == 2 and len(self.config.modalities) == 1:
                        modality_scores[self.config.modalities[0]] = float(arr[0, 1])
                    else:
                        for idx in range(min(arr.shape[1], len(self.config.modalities))):
                            modality_scores[self.config.modalities[idx]] = float(arr[0, idx])

        return RawModelOutput(
            route_probs=route_probs,
            risk_probs=risk_probs,
            modality_scores=modality_scores,
            model_version=self.config.version,
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        # scikit-learn caches ``id(stop_words)`` in this private field. The
        # value is a process-local memory address, is not inference state, and
        # otherwise makes byte-for-byte identical retrains serialize to
        # different artifacts. Force the cache marker to a stable empty value;
        # sklearn safely recomputes it on the first transform after loading.
        _stabilize_vectorizer_pickle(self._vectorizer)
        artifact = {
            "config": self.config.model_dump(mode="json"),
            "vectorizer": self._vectorizer,
            "numeric_scaler": self._numeric_scaler,
            "route_clf": self._route_clf,
            "route_encoder": self._route_encoder,
            "risk_clf": self._risk_clf,
            "risk_encoder": self._risk_encoder,
            "modality_clf": self._modality_clf,
        }
        with open(directory / self.config.sklearn_file_name, "wb") as fh:
            pickle.dump(artifact, fh)
        with open(directory / "metadata.json", "w") as fh:
            json.dump(
                {
                    "version": self.config.version,
                    "kind": "sklearn_router",
                    "config": self.config.model_dump(mode="json"),
                },
                fh,
                indent=2,
                sort_keys=True,
            )

    @classmethod
    def load(cls, directory: Path, config: LearnedRouterConfig) -> "SklearnRouterModel":
        with open(directory / config.sklearn_file_name, "rb") as fh:
            artifact = pickle.load(fh)
        artifact_config = LearnedRouterConfig.model_validate(artifact["config"])
        artifact_config = artifact_config.model_copy(
            update={"model_artifact_dir": config.model_artifact_dir}
        )
        # Fitted parameters come from the artifact; the rest is a genuine
        # compatibility contract (label spaces, thresholds, tokenisation) and is
        # still compared. Any residual mismatch names the offending fields
        # instead of raising a sentence nobody can act on.
        config = config.model_copy(
            update={
                field: getattr(artifact_config, field)
                for field in _ARTIFACT_DERIVED_FIELDS
            }
        )
        if artifact_config != config:
            artifact_fields = artifact_config.model_dump(mode="json")
            runtime_fields = config.model_dump(mode="json")
            differing = sorted(
                name
                for name in set(artifact_fields) | set(runtime_fields)
                if artifact_fields.get(name) != runtime_fields.get(name)
            )
            detail = "; ".join(
                f"{name}: artifact={artifact_fields.get(name)!r} "
                f"runtime={runtime_fields.get(name)!r}"
                for name in differing
            )
            raise RouterArtifactLoadError(
                "serialized router config is incompatible with the runtime config "
                f"({len(differing)} field(s) differ) — {detail}"
            )
        model = cls(config)
        model._vectorizer = artifact["vectorizer"]
        model._numeric_scaler = artifact["numeric_scaler"]
        model._route_clf = artifact["route_clf"]
        model._route_encoder = artifact["route_encoder"]
        model._risk_clf = artifact["risk_clf"]
        model._risk_encoder = artifact["risk_encoder"]
        model._modality_clf = artifact["modality_clf"]
        return model


class OnnxRouterModel:
    def __init__(
        self,
        config: LearnedRouterConfig,
        sessions: dict[str, Any],
        vectorizer: Any,
        metadata: dict[str, Any],
        numeric_scaler: AnyEstimator = None,
    ) -> None:
        self.config = config
        self._sessions = sessions
        self._vectorizer = vectorizer
        self._numeric_scaler = numeric_scaler
        self.metadata = metadata

    @staticmethod
    def _prob_dict(
        raw: np.ndarray,
        onnx_labels: list[str],
        all_labels: list[str],
    ) -> dict[str, float]:
        """Map ONNX classifier output to a probability dict.

        Binary classifiers may emit a single positive-class probability; in
        that case the complementary probability is assigned to the first label.
        Missing classes (present in ``all_labels`` but not ``onnx_labels``)
        receive probability 0.0.
        """
        result: dict[str, float] = {label: 0.0 for label in all_labels}
        arr = np.asarray(raw).reshape(-1)
        if arr.shape[0] == len(onnx_labels):
            for label, p in zip(onnx_labels, arr):
                result[label] = float(p)
        elif arr.shape[0] == 1 and len(onnx_labels) == 2:
            p = float(np.clip(arr[0], 0.0, 1.0))
            result[onnx_labels[0]] = 1.0 - p
            result[onnx_labels[1]] = p
        return result

    def predict(self, texts: list[str], numeric: list[list[float]]) -> RawModelOutput:
        if not texts:
            raise ValueError("texts must not be empty")
        import scipy.sparse as sp

        X_text = self._vectorizer.transform([texts[0]])
        X_num: Any = np.asarray([numeric[0]], dtype=np.float64)
        if self._numeric_scaler is not None:
            X_num = self._numeric_scaler.transform(X_num)
        X_num = X_num.astype(np.float32)
        X = sp.hstack([X_text, X_num]).toarray().astype(np.float32)

        route_out = list(self._sessions["route"].run({"float_input": X}).values())[1]
        risk_out = list(self._sessions["risk"].run({"float_input": X}).values())[1]
        modality_out = list(self._sessions["modality"].run({"float_input": X}).values())[1]

        route_probs = self._prob_dict(
            route_out,
            self.metadata["route_onnx_classes"],
            self.metadata["route_classes"],
        )
        risk_probs = self._prob_dict(
            risk_out,
            self.metadata["risk_onnx_classes"],
            self.metadata["risk_classes"],
        )
        modality_scores = self._prob_dict(
            modality_out,
            self.metadata["modality_onnx_classes"],
            self.metadata["modalities"],
        )
        return RawModelOutput(
            route_probs=route_probs,
            risk_probs=risk_probs,
            modality_scores=modality_scores,
            model_version=self.metadata.get("version", self.config.version),
        )


class ModelManager:
    def __init__(self, config: LearnedRouterConfig) -> None:
        self.config = config
        # ONE RESOLUTION, SHARED — see LearnedRouterConfig.resolved_artifact_dir.
        # Resolving the relative path here, against this process's working
        # directory, is the defect: run from anywhere but the tree root and
        # the artifact silently is not found.
        _resolution = config.resolved_artifact_dir()
        self._artifact_dir = _resolution.path or _resolution.fallback()

    def artifact_dir(self) -> Path:
        return self._artifact_dir

    @staticmethod
    def resolve_config(config: LearnedRouterConfig) -> LearnedRouterConfig:
        """Load signed-artifact configuration while preserving its mount root.

        The artifact is the authority on its own fitted parameters. Requiring the
        shipped YAML to restate a value produced by training is an unenforceable
        manual obligation — and when the placeholder ``temperature: 1.0`` failed
        the equality check in :meth:`SklearnRouterModel.load`, every run degraded
        to rules-only and still reported a headline ratio.

        Two sources, in order: ``metadata.json`` when it carries a ``config``
        block, otherwise the sklearn artifact's own embedded config. The second
        path matters because ONNX-exported artifacts write a metadata file
        WITHOUT a ``config`` key, which is exactly the shape that regressed.
        """
        _resolution = config.resolved_artifact_dir()
        artifact_dir = _resolution.path or _resolution.fallback()

        metadata_path = artifact_dir / "metadata.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                artifact_config = LearnedRouterConfig.model_validate(metadata["config"])
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                artifact_config = None
            if artifact_config is not None:
                if artifact_config.version != config.version:
                    return config
                return artifact_config.model_copy(
                    update={"model_artifact_dir": config.model_artifact_dir}
                )

        # Fall back to the fitted parameters carried inside the sklearn artifact.
        # Only the derived fields are adopted here: the structural contract stays
        # a real comparison rather than being silently overwritten by the file it
        # is supposed to be checking.
        sklearn_path = artifact_dir / config.sklearn_file_name
        if not sklearn_path.is_file():
            return config
        try:
            with open(sklearn_path, "rb") as fh:
                embedded = LearnedRouterConfig.model_validate(pickle.load(fh)["config"])
        except (OSError, KeyError, ValueError, pickle.UnpicklingError):
            return config
        if embedded.version != config.version:
            return config
        return config.model_copy(
            update={
                field: getattr(embedded, field) for field in _ARTIFACT_DERIVED_FIELDS
            }
        )

    def load(self) -> SklearnRouterModel | OnnxRouterModel | None:
        sklearn_path = self._artifact_dir / self.config.sklearn_file_name
        metadata_path = self._artifact_dir / "metadata.json"

        route_onnx = self._artifact_dir / "router_route.onnx"
        risk_onnx = self._artifact_dir / "router_risk.onnx"
        modality_onnx = self._artifact_dir / "router_modality.onnx"
        onnx_files = {
            "route": route_onnx,
            "risk": risk_onnx,
            "modality": modality_onnx,
        }

        if (
            self.config.use_onnx_if_available
            and all(p.exists() for p in onnx_files.values())
            and metadata_path.exists()
            and sklearn_path.exists()
        ):
            try:
                sessions = {
                    name: OnnxSessionManager(path)
                    for name, path in onnx_files.items()
                }
                if not all(s.loaded for s in sessions.values()):
                    raise RuntimeError("One or more ONNX sessions failed to load")
                with open(metadata_path) as fh:
                    metadata = json.load(fh)
                with open(sklearn_path, "rb") as fh:
                    artifact = pickle.load(fh)
                return OnnxRouterModel(
                    self.config,
                    sessions,
                    artifact["vectorizer"],
                    metadata,
                    artifact.get("numeric_scaler"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "onnx router load failed, falling back to sklearn",
                    extra={"error": str(exc)},
                )

        if sklearn_path.exists():
            try:
                return SklearnRouterModel.load(self._artifact_dir, self.config)
            except Exception as exc:
                # An artifact IS present here and did not load. Swallowing this
                # into a warning is what let a one-field config mismatch degrade
                # every run to rules-only while the benchmark still printed a
                # headline ratio. Absence is a valid state and returns None
                # below; a present-but-unloadable artifact is a defect and says
                # so. Chained so the original cause survives.
                raise RouterArtifactLoadError(
                    f"router artifact at {self._artifact_dir} is present but "
                    f"failed to load: {type(exc).__name__}: {exc}"
                ) from exc

        return None
