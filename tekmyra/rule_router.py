"""Deterministic rule router for the TekMyra runtime.

The rule router consumes deterministic features and the safety-gate decision,
then applies a short-circuit rule set.  Safety overrides always win.  Every
emitted decision carries the rules that fired, explanation tags, a fallback
route, and deterministic confidence (1.0).
"""

from __future__ import annotations

import json
import re
import uuid

from tekmyra.contracts import (
    FeatureVector,
    RiskClass,
    RouteClass,
)
from tekmyra.safety_gate import SafetyDecision


class RuleRouterError(Exception):
    """Raised when the rule router cannot produce a safe decision."""


class RuleRouter:
    """High-precision deterministic rule router.

    Rules are ordered by precision.  The first matching rule wins unless the
    safety gate has already forced a more restrictive route.  Abstention is
    represented as route_after_override == REVERSIBLE_RETRIEVAL_MODE so that
    unresolved ambiguous residue is never lossy-compressed.
    """

    def __init__(self, passthrough_token_budget: int = 50) -> None:
        self.passthrough_token_budget = passthrough_token_budget

    def route(
        self,
        artifact_id: str,
        features: FeatureVector,
        safety: SafetyDecision,
    ) -> "RouteDecisionRecord":
        try:
            return self._route(
                artifact_id,
                features,
                safety,
                allow_short_passthrough=True,
            )
        except RuleRouterError:
            raise
        except Exception as exc:
            raise RuleRouterError(f"Rule router internal error: {exc}") from exc

    def route_intent(
        self,
        artifact_id: str,
        features: FeatureVector,
        safety: SafetyDecision,
    ) -> "RouteDecisionRecord":
        """Classify codec intent independently of execution eligibility.

        Safety overrides still apply to ``route_after_override``.  Only the
        operational short-input bypass is omitted from
        ``route_before_override`` so routing quality and eligibility can be
        measured as separate contracts.
        """
        try:
            return self._route(
                artifact_id,
                features,
                safety,
                allow_short_passthrough=False,
            )
        except RuleRouterError:
            raise
        except Exception as exc:
            raise RuleRouterError(f"Rule router internal error: {exc}") from exc

    def _route(
        self,
        artifact_id: str,
        features: FeatureVector,
        safety: SafetyDecision,
        *,
        allow_short_passthrough: bool,
    ) -> "RouteDecisionRecord":
        rules_fired: list[str] = []
        explanation_tags: list[str] = list(safety.explanation_tags)

        # Critical safety situations bypass rule selection entirely.
        if safety.risk_class == RiskClass.CRITICAL:
            return RouteDecisionRecord(
                artifact_id=artifact_id,
                route_before_override=RouteClass.REJECT_OR_HUMAN_REVIEW,
                route_after_override=RouteClass.REJECT_OR_HUMAN_REVIEW,
                risk_class=safety.risk_class,
                risk_score=safety.risk_score,
                confidence_calibrated=1.0,
                abstained=False,
                fallback_route=RouteClass.REJECT_OR_HUMAN_REVIEW,
                rules_fired=["SAFETY_OVERRIDE"],
                explanation_tags=explanation_tags,
                model_top_k=[],
                verifier_status="skipped",
                versions={},
            )

        # Primary content-based dispatch (uses text and content_type signals).
        route, rule_name, fallback_route = self._classify_by_content(
            features.text or "",
            features.content_type,
            allow_short_passthrough=allow_short_passthrough,
        )
        rules_fired.append(rule_name)
        explanation_tags.append(rule_name.lower())

        # Apply safety override after rule selection, preserving the rule trail.
        abstained = route == RouteClass.REVERSIBLE_RETRIEVAL_MODE
        route_after_override = route
        if safety.forced_route is not None and safety.risk_class != RiskClass.CRITICAL:
            route_after_override = safety.forced_route
            rules_fired = rules_fired + ["SAFETY_OVERRIDE"]
            if route_after_override in (
                RouteClass.REVERSIBLE_RETRIEVAL_MODE,
                RouteClass.REJECT_OR_HUMAN_REVIEW,
            ):
                abstained = True

        return RouteDecisionRecord(
            artifact_id=artifact_id,
            route_before_override=route,
            route_after_override=route_after_override,
            risk_class=safety.risk_class,
            risk_score=safety.risk_score,
            confidence_calibrated=1.0,
            abstained=abstained,
            fallback_route=fallback_route,
            rules_fired=rules_fired,
            explanation_tags=explanation_tags,
            model_top_k=[],
            verifier_status="skipped",
            versions={},
        )

    # --- Rule predicates -----------------------------------------------------

    def _is_json(self, features: FeatureVector) -> bool:
        return features.json_parse_success

    def _is_diff(self, features: FeatureVector) -> bool:
        return features.diff_markers >= 2

    def _is_stacktrace(self, features: FeatureVector) -> bool:
        return features.stacktrace_hits >= 2

    def _is_log(self, features: FeatureVector) -> bool:
        return (
            features.log_timestamp_density >= 0.3
            or (features.log_timestamp_density >= 0.1 and features.log_repetition_ratio >= 0.3)
        )

    def _is_code(self, features: FeatureVector) -> bool:
        return features.ast_parse_success and features.code_language is not None

    def _is_short_safe(self, features: FeatureVector, safety: SafetyDecision) -> bool:
        return (
            features.token_len <= self.passthrough_token_budget
            and safety.risk_class == RiskClass.LOW
        )

    def _classify_by_content(
        self,
        text: str,
        content_type: str,
        *,
        allow_short_passthrough: bool = True,
    ) -> tuple[RouteClass, str, RouteClass]:
        """Return (route, rule_name, fallback_route) from content heuristics.

        The order below is chosen so that distinctive structural signals are
        tested before generic prose/passthrough fallbacks.
        """
        lowered = text.lower()
        content_type = content_type.lower()
        tokens = text.split()
        token_len = len(tokens)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        line_count = len(lines)

        # Mixed-modality and agent-state payloads must remain intact.  A
        # single embedded log, JSON, or code fragment must not collapse the
        # whole document into that fragment's codec.
        if content_type == "mixed":
            return RouteClass.PASSTHROUGH, "R_MIXED_SAFE", RouteClass.PASSTHROUGH
        if content_type == "agent_trace":
            return RouteClass.PASSTHROUGH, "R_AGENT_TRACE_SAFE", RouteClass.PASSTHROUGH

        # JSON-like structured content.
        try:
            json.loads(text)
            if text.strip().startswith(("{", "[")):
                return RouteClass.DETERMINISTIC_JSON_COMPRESSOR, "R_JSON_VALID", RouteClass.REVERSIBLE_RETRIEVAL_MODE
        except Exception:
            pass

        # Diff / patch signatures.
        if re.search(r"^(---|\+\+\+|@@\s+-\d+|@@\s+\+\d+|diff\s+--git)", text, re.M):
            return RouteClass.DIFF_COMPRESSOR, "R_DIFF_SIGNATURE", RouteClass.PASSTHROUGH

        # Stack traces (exception names + file/line frames).
        if (
            len(re.findall(r'File\s+"[^"]+",\s+line\s+\d+', text)) >= 2
            or len(re.findall(r'\s+at\s+\S+\s+\([^)]+:\d+:\d+\)', text)) >= 2
            or len(re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:Error|Exception|Failure|Fault)\b', text)) >= 1
        ):
            return RouteClass.LOG_COMPRESSOR, "R_STACKTRACE", RouteClass.REVERSIBLE_RETRIEVAL_MODE

        # Log-like dense timestamps + severity.
        ts_lines = sum(
            1
            for ln in lines
            if re.search(
                r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", ln
            )
        )
        severity_hits = len(
            re.findall(r"\b(?:INFO|WARN|WARNING|ERROR|FATAL|DEBUG|TRACE|CRITICAL)\b", text, re.I)
        )
        if line_count and (ts_lines / line_count >= 0.3 or (ts_lines / line_count >= 0.1 and severity_hits >= 2)):
            return RouteClass.LOG_COMPRESSOR, "R_LOG_DENSE", RouteClass.REVERSIBLE_RETRIEVAL_MODE

        # Code blocks / function definitions.
        if (
            re.search(r"\bdef\s+\w+\s*\(|\bclass\s+\w+\s*[:\(]|\bfunc\s+\w+\s*\(|\bfn\s+\w+\s*\(|\bimport\s+\w+|\bfrom\s+\S+\s+import\b", text)
            or re.search(r"\b(?:SELECT|INSERT|UPDATE|DELETE)\s+.+\bFROM\b|\bINSERT\s+INTO\b|\bUPDATE\s+\w+\s+SET\b", text, re.I)
            or content_type == "code"
        ):
            return RouteClass.CODE_COMPRESSOR, "R_CODE_AST_VALID", RouteClass.REVERSIBLE_RETRIEVAL_MODE

        # Markdown table or pipe-separated rows.
        pipe_rows = sum(1 for ln in lines if ln.strip().startswith("|") and "|" in ln[1:])
        if pipe_rows >= 2 or content_type in ("table", "chart"):
            return RouteClass.CHART_TABLE_EXTRACTION, "R_TABLE_MARKDOWN", RouteClass.REVERSIBLE_RETRIEVAL_MODE

        # OCR / scanned document hints.
        ocr_score = sum(1 for kw in ("ocr", "scanned", "scan", "skew", "memorandum", "document_scan", "handwriting") if kw in lowered)
        if ocr_score >= 2 or content_type in ("ocr_text", "ocr"):
            return RouteClass.OCR_FIRST, "R_OCR_HINT", RouteClass.REVERSIBLE_RETRIEVAL_MODE

        # Image / figure / photo metadata.
        image_score = sum(
            1
            for kw in (
                "image", "photo", "picture", "figure", "diagram", "png", "jpg", "jpeg",
                "gif", "webp", "svg", "bitmap", "resolution", "dimensions", "width=",
                "height=", "base64", "codec=", "color_space", "pixel", "rgba",
            )
            if kw in lowered
        )
        if image_score >= 3 or content_type in ("image_metadata", "image"):
            return RouteClass.IMAGE_OPTIMIZER, "R_IMAGE_METADATA", RouteClass.PASSTHROUGH

        # Bullet / keypoint heavy notes.
        bullet_lines = sum(
            1
            for ln in lines
            if re.match(r"^\s*(?:[-*•]|\d+[.\)])\s+\S", ln)
        )
        if line_count and bullet_lines / line_count >= 0.4:
            return RouteClass.EXTRACTIVE_KEYPOINT_COMPRESSOR, "R_BULLET_KEYPOINTS", RouteClass.REVERSIBLE_RETRIEVAL_MODE

        # Summarizer cues: memo/executive summary + prose length.
        summary_score = sum(
            1
            for kw in (
                "memo", "summary", "summarize", "executive summary", "key points",
                "blocker", "blockers", "identified", "requested", "decisions",
                "quarterly planning", "weekly planning", "status update",
            )
            if kw in lowered
        )
        if summary_score >= 2 and token_len > 50:
            return RouteClass.SUMMARIZER, "R_SUMMARY_CUES", RouteClass.PROSE_TOKEN_COMPRESSOR

        # Long prose. Fallback is a built, non-compressing route: SUMMARIZER is
        # declared-but-unbuilt (fails closed by design), so using it here made the
        # fallback incapable of succeeding and turned any prose verification
        # failure into a terminal error instead of a safe no-compression result.
        if token_len > 200:
            return (
                RouteClass.PROSE_TOKEN_COMPRESSOR,
                "R_LONG_PROSE",
                RouteClass.REVERSIBLE_RETRIEVAL_MODE,
            )

        # Retrieval / RAG context markers.
        if re.search(r"\[CTX_REF:|\[ANCHOR:|\[DOC_REF:|\[SOURCE:|\[RETRIEVAL", text, re.I):
            return RouteClass.REVERSIBLE_RETRIEVAL_MODE, "R_RETRIEVAL_CONTEXT", RouteClass.REJECT_OR_HUMAN_REVIEW

        # Short safe payload.
        if allow_short_passthrough and token_len <= self.passthrough_token_budget:
            return RouteClass.PASSTHROUGH, "R_SHORT_SAFE", RouteClass.PASSTHROUGH

        # Declared semantic content type is the final deterministic intent
        # signal after higher-precision structure markers and the operational
        # short-input bypass have been evaluated.
        semantic_routes: dict[str, tuple[RouteClass, str, RouteClass]] = {
            # Fallback mirrors "rag" below: a built, non-compressing route. The
            # previous SUMMARIZER fallback is declared-but-unbuilt and fails
            # closed, so prose had no recoverable fallback at all.
            "prose": (
                RouteClass.PROSE_TOKEN_COMPRESSOR,
                "R_CONTENT_TYPE_PROSE",
                RouteClass.REVERSIBLE_RETRIEVAL_MODE,
            ),
            "rag": (
                RouteClass.PROSE_TOKEN_COMPRESSOR,
                "R_CONTENT_TYPE_RAG",
                RouteClass.REVERSIBLE_RETRIEVAL_MODE,
            ),
            "bullet_notes": (
                RouteClass.EXTRACTIVE_KEYPOINT_COMPRESSOR,
                "R_CONTENT_TYPE_BULLETS",
                RouteClass.REVERSIBLE_RETRIEVAL_MODE,
            ),
            "governance": (
                RouteClass.GOVERNANCE_SAFE_COMPRESSION,
                "R_CONTENT_TYPE_GOVERNANCE",
                RouteClass.REJECT_OR_HUMAN_REVIEW,
            ),
            "pii_message": (
                RouteClass.PII_REDACTION_BEFORE_COMPRESSION,
                "R_CONTENT_TYPE_PII",
                RouteClass.REJECT_OR_HUMAN_REVIEW,
            ),
            "retrieval_context": (
                RouteClass.REVERSIBLE_RETRIEVAL_MODE,
                "R_CONTENT_TYPE_RETRIEVAL",
                RouteClass.REJECT_OR_HUMAN_REVIEW,
            ),
            "legal": (
                RouteClass.LEGAL_SECURITY_SAFE_MODE,
                "R_CONTENT_TYPE_LEGAL",
                RouteClass.REJECT_OR_HUMAN_REVIEW,
            ),
            "security": (
                RouteClass.LEGAL_SECURITY_SAFE_MODE,
                "R_CONTENT_TYPE_SECURITY",
                RouteClass.REJECT_OR_HUMAN_REVIEW,
            ),
            "summary": (
                RouteClass.SUMMARIZER,
                "R_CONTENT_TYPE_SUMMARY",
                RouteClass.PROSE_TOKEN_COMPRESSOR,
            ),
        }
        if content_type in semantic_routes:
            return semantic_routes[content_type]

        # Residue: reversible retrieval.
        return RouteClass.REVERSIBLE_RETRIEVAL_MODE, "R_UNRESOLVED_AMBIGUOUS", RouteClass.REJECT_OR_HUMAN_REVIEW


class RouteDecisionRecord:
    """Internal lightweight route decision (matches the contract shape)."""

    def __init__(
        self,
        artifact_id: str,
        route_before_override: RouteClass,
        route_after_override: RouteClass,
        risk_class: RiskClass,
        risk_score: float,
        confidence_calibrated: float,
        abstained: bool,
        fallback_route: RouteClass | None,
        rules_fired: list[str],
        explanation_tags: list[str],
        model_top_k: list[tuple[RouteClass, float]],
        verifier_status: str,
        versions: dict[str, str],
    ) -> None:
        self.decision_id = str(uuid.uuid4())
        self.artifact_id = artifact_id
        self.route_before_override = route_before_override
        self.route_after_override = route_after_override
        self.risk_class = risk_class
        self.risk_score = risk_score
        self.confidence_calibrated = confidence_calibrated
        self.abstained = abstained
        self.fallback_route = fallback_route
        self.rules_fired = rules_fired
        self.explanation_tags = explanation_tags
        self.model_top_k = model_top_k
        self.verifier_status = verifier_status
        self.versions = versions


def route_deterministic(
    artifact_id: str,
    features: FeatureVector,
    safety: SafetyDecision | None = None,
) -> RouteDecisionRecord:
    """Convenience function using the default rule router."""
    if safety is None:
        from tekmyra.safety_gate import SafetyGate

        safety = SafetyGate().classify(features)
    return RuleRouter().route(artifact_id, features, safety)
