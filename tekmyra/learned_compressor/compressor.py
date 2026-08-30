"""Prose/RAG/agent-trace compressor orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

from tekmyra import artifact_paths
from tekmyra.codecs.base import CodecResult
from tekmyra.compression_budget import DEFAULT_BUDGET_RATIO
from tekmyra.contracts import CompressorVariant, FeatureVector
from tekmyra.entity_fidelity import top_up_retention
from tekmyra.learned_compressor.features import (
    compute_entity_mention_ranks,
    compute_entity_sentence_members,
    extract_sentence_features,
)
from tekmyra.learned_compressor.labeler import label_keep
from tekmyra.learned_compressor.model import (
    KeepDropScores,
    OnnxCompressorModel,
    SentenceCompressorModel,
)
from tekmyra.learned_compressor.segmenter import segment_sentences
from tekmyra.learned_compressor.selector import select_sentences
from tekmyra.numeric_fidelity import (
    extract_numerics,
    extract_numerics_in_spans,
)


class _SentenceScorer(Protocol):
    """Protocol for any model that can score sentences for keep/drop."""

    def predict(self, sentences: list[str]) -> list[KeepDropScores]: ...

logger = logging.getLogger("tekmyra.learned_compressor.compressor")


#: WHERE the compressor artifact lives, RELATIVE to a root that
#: :mod:`tekmyra.artifact_paths` resolves. This used to be the bare
#: relative path handed straight to the loader, resolved against the working
#: directory — so running from anywhere but the tree root found nothing, the
#: loader returned its no-op, and the benchmark printed a compression ratio for
#: a compressor that compresses nothing. Measured: 82.9711% from the tree root,
#: 3083 bytes in and 3083 out from ``/tmp``, no error either way.
ARTIFACT_RELATIVE_DIR = Path(".cache/compressor/1.0.0")

#: Retained under its old name because callers import it. It is the RELATIVE
#: path and must not be used to decide whether an artifact is present — ask
#: :func:`resolve_artifact_dir`, which reports every location it searched.
_ARTIFACT_DIR = ARTIFACT_RELATIVE_DIR

_LAST_RESOLUTION: artifact_paths.Resolution | None = None


def resolve_artifact_dir(
    artifact_dir: Path | None = None,
) -> artifact_paths.Resolution:
    """Locate the compressor artifact, recording every candidate tried."""

    return artifact_paths.resolve(ARTIFACT_RELATIVE_DIR, explicit=artifact_dir)


def last_artifact_resolution() -> artifact_paths.Resolution | None:
    """The search :func:`load_compressor` last performed, or ``None``.

    THIS IS THE ACCESSOR A DIAGNOSTIC SHOULD READ. ``run_three_method_comparison``
    reported the cache directories PRESENT inside the exception explaining the
    artifact was missing, because the loader resolved against the working
    directory and the diagnostic re-derived its own answer against the
    repository root. Neither was wrong about its own path; there was no way to
    ask the loader where it had actually looked. Now there is, and a diagnostic
    that re-derives instead of reading this is choosing to guess.
    """

    return _LAST_RESOLUTION

_DEFAULT_BUDGET = DEFAULT_BUDGET_RATIO

# Populated by ``tekmyra.service.configure`` so the service's compressor
# version/variant gating is respected by the runtime compressor loader.
_DEFAULT_VERSION: str | None = None
_DEFAULT_VARIANT: str | None = None
_VERIFIED_DEFAULT: ProseCompressor | None = None


def clear_verified_default() -> None:
    """Discard any previously installed deployed artifact snapshot."""

    global _VERIFIED_DEFAULT
    _VERIFIED_DEFAULT = None


def install_verified_default(compressor: ProseCompressor) -> None:
    """Install the exact signature-bound in-memory deployed compressor."""

    global _VERIFIED_DEFAULT
    _VERIFIED_DEFAULT = compressor


def load_compressor_bytes(payload: bytes) -> ProseCompressor:
    """Deserialize one already-verified sklearn compressor byte snapshot."""

    return ProseCompressor(SentenceCompressorModel.load_bytes(payload))


def _read_metadata_version(directory: Path, metadata_name: str = "metadata.json") -> str | None:
    """Return the ``version`` field from an ONNX metadata file if it exists."""
    path = directory / "onnx" / metadata_name
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data.get("version")
    except Exception:  # noqa: BLE001
        return None


class ProseCompressor:
    """Tiny learned extractive compressor for prose-like content."""

    def __init__(self, model: _SentenceScorer | None = None) -> None:
        self._model = model

    def compress(
        self,
        text: str,
        features: FeatureVector,
        locked_spans: list[dict[str, object]],
        budget_ratio: float = _DEFAULT_BUDGET,
    ) -> CodecResult:
        """Compress prose by dropping low-importance sentences.

        Locked spans are always preserved. If no model is loaded, fall back to
        passthrough (fail-closed). Invalid budget ratios are rejected and also
        fall back to passthrough.
        """
        if not 0.0 <= budget_ratio <= 1.0:
            return CodecResult(
                success=True,
                output=text,
                ratio=1.0,
                error="invalid_budget_ratio",
            )

        if not text:
            return CodecResult(success=True, output=text, ratio=1.0)

        if self._model is None:
            return CodecResult(success=True, output=text, ratio=1.0, error="no_compressor_model")

        sentences = segment_sentences(text)
        if not sentences:
            return CodecResult(success=True, output=text, ratio=1.0)

        locked_texts = {str(s.get("text", "")) for s in locked_spans if s.get("text")}
        # Document-level, computed once: a sentence cannot see that it carries
        # the 87th of 110 mentions by looking at itself. Same helper the trainer
        # uses, so the feature at inference is the feature the model learned.
        _entity_ranks = compute_entity_mention_ranks(sentences)
        _entity_members = compute_entity_sentence_members(sentences)
        sentence_features = [
            extract_sentence_features(
                s, features, locked_texts, len(sentences), _entity_ranks
            )
            for s in sentences
        ]
        labels = [label_keep(f) for f in sentence_features]

        # Train on the fly if needed; only the sklearn model supports fitting.
        if isinstance(self._model, SentenceCompressorModel) and self._model._clf is None:
            try:
                self._model.fit([s.text for s in sentences], labels)
            except Exception:
                return CodecResult(success=True, output=text, ratio=1.0, error="compressor_fit_failed")

        scores = self._model.predict([s.text for s in sentences])
        # The selector may back the budget off when locked spans saturate it
        # (Vega ruling 47); the achieved figure comes back through diagnostics
        # so it can travel with the result rather than being inferred from the
        # ratio, which is a different quantity measured in different units.
        selection_diagnostics: dict[str, object] = {}
        selected = select_sentences(
            sentences,
            sentence_features,
            scores,
            budget_ratio,
            _entity_members,
            diagnostics=selection_diagnostics,
        )

        if not selected:
            # Never return empty; fall back to a single sentence.
            selected = sentences[:1]

        # --- numeric fidelity ------------------------------------------------
        # verify() requires every numeric outside a locked span to survive
        # compression (its numeric_preserved check), but sentence selection is
        # score-based and knows nothing of that contract, so the sentence
        # carrying the only occurrence of a figure could be dropped outright.
        #
        # Measured before this guard: prose-000 compressed to ratio 0.656 by
        # deleting "The quarterly review showed a 12% increase in recurring
        # revenue." — losing the figure entirely while locked-span preservation
        # still reported a clean 68/68. Every safety indicator read green while
        # the output silently lost a business fact, and the run then died at the
        # verifier instead of degrading.
        #
        # Restore the SENTENCE carrying a dropped numeric, never the bare token:
        # prepending "12" would satisfy the check while destroying what the
        # figure measures. The numeric definition comes from the shared
        # numeric_fidelity contract rather than being re-expressed here — a
        # producer guessing at "what counts as a numeric" would drift from the
        # check it has to satisfy.
        selected_ids = {id(s) for s in selected}
        locked_numerics = extract_numerics_in_spans(text, locked_spans)
        required_numerics = extract_numerics(text) - locked_numerics
        missing_numerics = required_numerics - extract_numerics(
            " ".join(s.text for s in sentences if id(s) in selected_ids)
        )
        for sentence in sentences:
            if not missing_numerics:
                break
            if id(sentence) in selected_ids:
                continue
            recovered = extract_numerics(sentence.text) & missing_numerics
            if recovered:
                selected_ids.add(id(sentence))
                missing_numerics -= recovered

        # Rebuild in original document order so restored sentences read in place
        # rather than being appended out of sequence.
        output = " ".join(s.text for s in sentences if id(s) in selected_ids)

        # --- entity retention top-up ----------------------------------------
        # THE SAME SHAPE AS THE NUMERIC GUARD ABOVE, for the other contract
        # verify() enforces. The selector allocates its entity floor against the
        # budget it aimed at; the check grades against the ratio actually
        # achieved. Where the codec undershoots its budget — which is the normal
        # case — the graded floor sits ABOVE the allocated one and a correct
        # producer fails by a mention or two. Measured: one fixture short by a
        # single mention (retention 0.200 against a floor of 0.2093) fell all
        # the way back to passthrough, trading a 58% saving for zero.
        #
        # Adding a sentence lengthens the output, which raises the ratio and so
        # raises the floor — hence the loop. It converges quickly because a
        # sentence contributes far more to one entity's retention than its own
        # characters contribute to the document ratio, but it is BOUNDED rather
        # than trusted to converge: an unbounded producer-side repair loop is a
        # hang, and a hang is worse than the refusal it was avoiding.
        # Indices rather than ids, because the shared helper serves a codec that
        # slices a document by offset as well as this one that joins sentence
        # objects. Selection is still by position, so identical sentence texts
        # stay distinguishable.
        selected_indices = {
            i for i, sentence in enumerate(sentences) if id(sentence) in selected_ids
        }
        selected_indices, output = top_up_retention(
            text,
            [s.text for s in sentences],
            selected_indices,
            lambda chosen: " ".join(
                s.text for i, s in enumerate(sentences) if i in chosen
            ),
        )

        # Ensure any locked-span text that didn't already fall inside a kept
        # sentence is explicitly prepended so must-preserve spans survive the
        # budget reduction (e.g. dates, identifiers, citations that aren't
        # aligned to sentence boundaries).
        missing_locks = [
            span_text
            for span_text in locked_texts
            if span_text and span_text not in output
        ]
        if missing_locks:
            prefix = " ".join(missing_locks)
            output = (prefix + " " + output).strip() if output else prefix

        ratio = len(output) / max(1, len(text))
        achieved = selection_diagnostics.get("budget_achieved")
        return CodecResult(
            success=True,
            output=output,
            ratio=ratio,
            learned_compressor_invoked=True,
            budget_requested=budget_ratio,
            budget_achieved=float(achieved) if isinstance(achieved, (int, float)) else None,
            selection_diagnostics=selection_diagnostics,
        )


def _is_transformer_artifact(directory: Path) -> bool:
    """Return True if ``directory`` contains transformer compressor artifacts."""
    return (
        (directory / "encoder").is_dir()
        or (directory / "onnx" / "transformer_compressor.onnx").exists()
    )


def _load_transformer_compressor(
    directory: Path,
    version: str | None = None,
    variant: str | None = None,
) -> ProseCompressor | None:
    """Load a transformer compressor, gated by version and variant.

    Variant selection order when ``variant`` is ``None``:

    1. Dynamo-exported ONNX v2 (``transformer_compressor_v2.onnx``).
    2. Legacy ONNX v1 (``transformer_compressor.onnx``).
    3. PyTorch checkpoint.

    Explicit variants skip the auto-selection and load only the requested path.
    If ``version`` is supplied, ONNX artifacts whose metadata version does not
    match are skipped.

    Returns ``None`` when no transformer artifact is present or loading fails,
    allowing the caller to fall back to the sklearn compressor ladder.
    """
    if not _is_transformer_artifact(directory):
        return None

    onnx_candidates: list[tuple[Path, str]] = []
    if variant is None or variant == CompressorVariant.ONNX_V2.value:
        onnx_candidates.append(
            (directory / "onnx" / "transformer_compressor_v2.onnx", "metadata_v2.json")
        )
    if variant is None or variant == CompressorVariant.ONNX_V1.value:
        onnx_candidates.append(
            (directory / "onnx" / "transformer_compressor.onnx", "metadata.json")
        )
    if variant == CompressorVariant.ONNX_STATIC_INT8.value:
        onnx_candidates.extend(
            [
                (
                    directory / "onnx" / "transformer_compressor_v2.static.int8.onnx",
                    "metadata_v2.json",
                ),
                (
                    directory / "onnx" / "transformer_compressor.static.int8.onnx",
                    "metadata.json",
                ),
            ]
        )

    if onnx_candidates:
        try:
            from transformers import AutoTokenizer

            from tekmyra.learned_compressor.transformer_onnx_model import (
                OnnxTransformerCompressorModel,
            )

            tokenizer = AutoTokenizer.from_pretrained(directory / "encoder")
            config_path = directory / "config.json"
            default_max_length = 512  # matches TransformerConfig.max_length
            if config_path.exists():
                with open(config_path) as fh:
                    config_data = json.load(fh)
                default_max_length = config_data.get("max_length", default_max_length)

            for onnx_path, metadata_name in onnx_candidates:
                if not onnx_path.exists():
                    continue
                if version is not None:
                    metadata_version = _read_metadata_version(directory, metadata_name)
                    if metadata_version != version:
                        logger.info(
                            "skipping_onnx_version_mismatch",
                            extra={
                                "artifact_dir": str(directory),
                                "path": str(onnx_path),
                                "expected": version,
                                "found": metadata_version,
                            },
                        )
                        continue
                metadata_path = directory / "onnx" / metadata_name
                max_length = default_max_length
                if metadata_path.exists():
                    with open(metadata_path) as fh:
                        metadata = json.load(fh)
                    max_length = metadata.get("max_length", max_length)
                onnx_model = OnnxTransformerCompressorModel(
                    onnx_path, tokenizer, max_length=max_length
                )
                if onnx_model._session.loaded:
                    return ProseCompressor(onnx_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "transformer_onnx_load_failed",
                extra={"artifact_dir": str(directory), "error": str(exc)},
            )

    if variant is not None and variant != CompressorVariant.PYTORCH.value:
        # An explicit non-PyTorch variant was requested but could not be loaded.
        return None

    try:
        from tekmyra.learned_compressor.transformer_model import (
            TransformerCompressorModel,
        )

        model = TransformerCompressorModel.load(directory)
        return ProseCompressor(model)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "transformer_pytorch_load_failed",
            extra={"artifact_dir": str(directory), "error": str(exc)},
        )

    return None


def _load_sklearn_compressor(directory: Path) -> ProseCompressor | None:
    """Load the legacy sklearn/ONNX compressor ladder."""
    onnx_dir = directory / "onnx"
    onnx_path = onnx_dir / "compressor.onnx"
    metadata_path = onnx_dir / "metadata.json"

    if onnx_path.exists() and metadata_path.exists():
        try:
            with open(metadata_path) as fh:
                metadata = json.load(fh)
            sklearn_model = SentenceCompressorModel.load(directory)
            onnx_model = OnnxCompressorModel(
                onnx_path, sklearn_model._vectorizer, metadata.get("version", "1.0.0")
            )
            if onnx_model._session.loaded:
                return ProseCompressor(onnx_model)
        except Exception:  # noqa: BLE001
            pass

    try:
        model = SentenceCompressorModel.load(directory)
        return ProseCompressor(model)
    except Exception:  # noqa: BLE001
        return None


def load_compressor(
    artifact_dir: Path | None = None,
    version: str | None = None,
    variant: str | None = None,
) -> ProseCompressor:
    """Load a trained compressor artifact, or return a no-op compressor.

    Args:
        artifact_dir: Directory containing the saved compressor artifact. When
            omitted, :mod:`tekmyra.artifact_paths` searches an ordered
            list of locations and records every one of them.
        version: Optional version gate; falls back to the service default.
        variant: Optional variant gate; falls back to the service default.

    Loading order:

    1. Transformer compressor (ONNX v2, ONNX v1, then PyTorch).
    2. Legacy sklearn/ONNX compressor.
    3. A NO-OP model.

    **Step 3 was previously described as "fail-closed", and that word was doing
    damage.** The no-op passes text through unchanged, so it is fail-closed for
    FIDELITY — no protected span is ever dropped by it. It is fail-OPEN for any
    FIGURE: a ratio of 1.000 comes out and is indistinguishable from a
    measurement. A reader auditing precisely that question found the reassuring
    word at precisely this site and moved on.

    So the no-op remains — the runtime must degrade rather than crash on a
    machine with no model, and ``codecs/selector`` depends on that — but the
    resolution that led to it is now recorded in
    :func:`last_artifact_resolution`, and any caller that turns this into a
    published number is expected to consult it and refuse. ``benchmark.py``'s
    provenance block already does, and marks such a run not-quotable.
    """
    global _LAST_RESOLUTION

    if artifact_dir is None and _VERIFIED_DEFAULT is not None:
        return _VERIFIED_DEFAULT

    resolution = resolve_artifact_dir(artifact_dir)
    _LAST_RESOLUTION = resolution
    directory = resolution.path or resolution.fallback()
    effective_version = version or _DEFAULT_VERSION
    effective_variant = variant or _DEFAULT_VARIANT

    transformer_compressor = _load_transformer_compressor(
        directory,
        version=effective_version,
        variant=effective_variant,
    )
    if transformer_compressor is not None:
        return transformer_compressor

    sklearn_compressor = _load_sklearn_compressor(directory)
    if sklearn_compressor is not None:
        return sklearn_compressor

    return ProseCompressor(None)
