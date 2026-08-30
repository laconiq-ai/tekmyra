"""Codec selector and fallback ladder.

Maps route decisions to deterministic codecs and applies verifier-based
fallback when a codec result is unsafe.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from tekmyra.codecs.base import CodecResult
from tekmyra.codecs.safe_mode import compress_safe_mode
from tekmyra.codecs.code_codec import compress_code
from tekmyra.codecs.diff_codec import compress_diff
from tekmyra.codecs.json_codec import compress_json
from tekmyra.codecs.log_codec import compress_logs
from tekmyra.codecs.stacktrace_codec import compress_stacktrace
from tekmyra.compression_budget import (
    DEFAULT_BUDGET_RATIO as _APPLY_CODEC_DEFAULT_BUDGET,
)
from tekmyra.contracts import FeatureVector, RouteClass


RouteCodec = Callable[..., CodecResult]




def _passthrough(text: str, _features: FeatureVector, **_kwargs: object) -> CodecResult:
    return CodecResult(success=True, output=text, ratio=1.0)


def _prose_route(text: str, _features: FeatureVector, **kwargs: object) -> CodecResult:
    # Lazy import breaks the circular dependency between codecs and the compressor.
    # The budget default does NOT come from here — it comes from the
    # compression_budget leaf, imported at module scope, because the operating
    # point had EIGHT definitions on 2026-08-09 and every one was found by a
    # guard failing rather than by reading the code.
    from tekmyra.learned_compressor import load_compressor

    # Force passthrough for HIGH/CRITICAL safety tiers so must-preserve spans
    # and protected content are never routed through the lossy prose codec.
    safety_tier = kwargs.get("safety_tier")
    if safety_tier is not None:
        tier_value = safety_tier.value if hasattr(safety_tier, "value") else str(safety_tier)
        if str(tier_value).lower() in {"high", "critical"}:
            return CodecResult(success=True, output=text, ratio=1.0)

    raw_locked_spans = kwargs.get("locked_spans")
    locked_spans = (
        cast(list[dict[str, object]], raw_locked_spans)
        if isinstance(raw_locked_spans, list)
        else []
    )
    raw_budget_ratio = kwargs.get("budget_ratio", _APPLY_CODEC_DEFAULT_BUDGET)
    budget_ratio = (
        float(raw_budget_ratio)
        if isinstance(raw_budget_ratio, (int, float, str))
        else _APPLY_CODEC_DEFAULT_BUDGET
    )
    compressor = load_compressor()
    return compressor.compress(  # type: ignore[arg-type]
        text,
        _features,
        locked_spans=locked_spans,
        budget_ratio=budget_ratio,
    )


def _safe_mode_route(text: str, features: FeatureVector, **kwargs: object) -> CodecResult:
    """Safe-mode compression with a structural preservation guarantee."""
    return compress_safe_mode(text, features, **kwargs)


def _json_route(text: str, _features: FeatureVector, **_kwargs: object) -> CodecResult:
    return compress_json(text)


def _log_route(text: str, features: FeatureVector, **_kwargs: object) -> CodecResult:
    # Stack traces are routed through the log compressor route class to avoid
    # schema churn, but they are compressed by the deterministic stack-trace
    # codec which preserves exception messages and application frames.
    if features.stacktrace_hits > 0:
        return compress_stacktrace(text)
    return compress_logs(text)


def _stacktrace_route(text: str, _features: FeatureVector, **_kwargs: object) -> CodecResult:
    return compress_stacktrace(text)


def _diff_route(text: str, _features: FeatureVector, **_kwargs: object) -> CodecResult:
    return compress_diff(text)


def _code_route(text: str, features: FeatureVector, **_kwargs: object) -> CodecResult:
    # Comment stripping is this codec's only compression lever, and it was
    # disabled here — silently, in a parameter default — because stripping
    # comments loses the numerics they carry and `verify()` correctly rejected
    # the result. With it off, this route returned its input unchanged on every
    # code fixture while reporting success, which no guard could see: an
    # uncompressed output passes every fidelity check by construction.
    #
    # Vega ruling 72 replaced the on/off choice with a per-comment test — a
    # comment goes only if every numeric in it survives in the residue — so the
    # lever is safe to enable and the contract is untouched. The reason lives
    # at the call site deliberately: the previous decision to disable a whole
    # capability was recorded nowhere but an argument default, which is why it
    # went undiscovered.
    return compress_code(text, language=features.code_language, strip_comments=True)


def _unimplemented_result(text: str, route: RouteClass) -> CodecResult:
    """Fail-closed result for a route declared in the taxonomy but not built.

    ``success=False`` is the contract's own fail-closed signal (see
    ``CodecResult``), and ``verify()`` checks it first, so the runtime refuses
    rather than handing back the uncompressed input as a successful
    compression. ``ratio`` stays 1.0 because nothing was compressed — the
    honest number — and reporting excludes these rows from ratio aggregates
    instead of letting a no-op read as a real 1.0.
    """
    return CodecResult(
        success=False,
        output=text,
        ratio=1.0,
        error=f"no_codec_defined_for_{route.value}",
    )


# Routes below are declared in the frozen v1.0.0 taxonomy and are deliberately
# retained (pruning them would trip the WP-3 revision gate + version bump).
# Each has its own function so the per-route reason stays legible.


def _summarizer_route(text: str, _features: FeatureVector, **_kwargs: object) -> CodecResult:
    # Abstractive summarisation is not built. Roadmap item, not a bug.
    return _unimplemented_result(text, RouteClass.SUMMARIZER)


def _extractive_keypoint_route(
    text: str, _features: FeatureVector, **_kwargs: object
) -> CodecResult:
    # Extractive key-point selection is not built.
    return _unimplemented_result(text, RouteClass.EXTRACTIVE_KEYPOINT_COMPRESSOR)


def _ocr_first_route(text: str, _features: FeatureVector, **_kwargs: object) -> CodecResult:
    # OCR ingestion is planned (backlog E06-007 / E04-010), not yet built.
    return _unimplemented_result(text, RouteClass.OCR_FIRST)


def _image_optimizer_route(
    text: str, _features: FeatureVector, **_kwargs: object
) -> CodecResult:
    # Image optimisation has no text-domain codec; declared for future binaries.
    return _unimplemented_result(text, RouteClass.IMAGE_OPTIMIZER)


def _chart_table_route(text: str, _features: FeatureVector, **_kwargs: object) -> CodecResult:
    # Table-skeleton codec is planned (backlog E07-008), not yet built.
    return _unimplemented_result(text, RouteClass.CHART_TABLE_EXTRACTION)


# Declared-but-unbuilt routes, kept in one place so the map and the
# UNIMPLEMENTED_CODEC_ROUTES set below are derived from a single source and
# cannot drift apart.
_UNIMPLEMENTED_CODECS: dict[RouteClass, RouteCodec] = {
    RouteClass.SUMMARIZER: _summarizer_route,
    RouteClass.EXTRACTIVE_KEYPOINT_COMPRESSOR: _extractive_keypoint_route,
    RouteClass.OCR_FIRST: _ocr_first_route,
    RouteClass.IMAGE_OPTIMIZER: _image_optimizer_route,
    RouteClass.CHART_TABLE_EXTRACTION: _chart_table_route,
}

UNIMPLEMENTED_CODEC_ROUTES: frozenset[RouteClass] = frozenset(_UNIMPLEMENTED_CODECS)


def _safe_terminal_routes(codec_map: dict[RouteClass, RouteCodec]) -> frozenset[RouteClass]:
    """Routes whose codec returns the input untouched.

    These never transform content, so they always verify and there is nothing
    safer to fall back to: a fallback equal to one of them is a legitimate
    terminal, not the futile self-reference the runtime guards against.

    DERIVED from the codec map rather than listed, so that building a real
    codec for one of these routes automatically removes it from the exempt
    set. An earlier revision hardcoded only two of the six and thereby let the
    guard rewrite the recorded route for benign REVERSIBLE_RETRIEVAL_MODE
    traffic — the exact harm the exemption exists to prevent.

    Distinct from hierarchical_compressor._SAFE_ROUTES, which answers a
    different question (which routes must never be compressed lossily).
    """
    return frozenset(
        route for route, codec in codec_map.items() if codec is _passthrough
    )


_CODEC_MAP: dict[RouteClass, RouteCodec] = {
    RouteClass.PASSTHROUGH: _passthrough,
    RouteClass.DETERMINISTIC_JSON_COMPRESSOR: _json_route,
    RouteClass.LOG_COMPRESSOR: _log_route,
    RouteClass.DIFF_COMPRESSOR: _diff_route,
    RouteClass.CODE_COMPRESSOR: _code_route,
    # LEGAL_SECURITY_SAFE_MODE is now BUILT. The gate had already decided this
    # content belongs in a safe compression mode; that mode did not exist, so
    # the route mapped to a no-op and 11.5% of corpus tokens sat on a
    # passthrough while looking like a policy decision.
    #
    # GOVERNANCE_SAFE_COMPRESSION deliberately stays passthrough. Measured: once
    # its protected spans and their predicating clauses are preserved, its
    # genuine sub-sentence headroom is ~691 tokens across 8 fixtures — and 77%
    # of what looked like headroom was one repeated generator phrase. A codec
    # there would compress the corpus builder's habit, not the document.
    RouteClass.PII_REDACTION_BEFORE_COMPRESSION: _passthrough,
    RouteClass.GOVERNANCE_SAFE_COMPRESSION: _passthrough,
    RouteClass.LEGAL_SECURITY_SAFE_MODE: _safe_mode_route,
    # Reversible / review routes are intentionally not compressed.
    RouteClass.REVERSIBLE_RETRIEVAL_MODE: _passthrough,
    RouteClass.REJECT_OR_HUMAN_REVIEW: _passthrough,
    RouteClass.PROSE_TOKEN_COMPRESSOR: _prose_route,
    # Declared-but-unbuilt routes: fail closed, never silent passthrough.
    **_UNIMPLEMENTED_CODECS,
}


SAFE_TERMINAL_ROUTES: frozenset[RouteClass] = _safe_terminal_routes(_CODEC_MAP)


def select_codec(route: RouteClass) -> RouteCodec | None:
    """Return the codec callable for a route, or None if no codec is defined."""
    return _CODEC_MAP.get(route)


def apply_codec(
    text: str,
    route: RouteClass,
    features: FeatureVector,
    locked_spans: list[dict[str, object]] | None = None,
    safety_tier: str | None = None,
    protected_spans: list[dict[str, object]] | None = None,
    budget_ratio: float = _APPLY_CODEC_DEFAULT_BUDGET,
) -> CodecResult:
    """Apply the deterministic codec for ``route`` to ``text``.

    Every member of the canonical ``RouteClass`` taxonomy has a map entry, so
    this branch fires only for a genuinely unrecognised future value. It
    returns ``success=False`` — not passthrough-as-success — so ``verify()``
    fails closed instead of the caller receiving uncompressed input reported
    as a successful compression.
    """
    codec = select_codec(route)
    if codec is None:
        return _unimplemented_result(text, route)

    # Merge any additional protected spans passed by the caller so the
    # prose compressor can guarantee their texts are preserved.
    merged_locks: list[dict[str, object]] = list(locked_spans or [])
    if protected_spans:
        existing = {str(s.get("text", "")) for s in merged_locks if s.get("text")}
        for span in protected_spans:
            text_value = str(span.get("text", ""))
            if text_value and text_value not in existing:
                merged_locks.append(span)
                existing.add(text_value)

    return codec(
        text,
        features,
        locked_spans=merged_locks,
        safety_tier=safety_tier,
        budget_ratio=budget_ratio,
    )
