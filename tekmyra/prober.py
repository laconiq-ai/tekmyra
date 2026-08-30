"""Deterministic parser/prober for the TekMyra runtime.

Computes a FeatureVector from artifact text and metadata without any
learned models.  All signals are heuristic, auditable, and fail-closed:
if a probe cannot decide, it returns the neutral/default value.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

from tekmyra.contracts import (
    ArtifactEnvelope,
    FeatureVector,
    PolicyTier,
    SourceChannel,
)


# --- Keyword inventories ----------------------------------------------------

LEGAL_KEYWORDS = {
    "contract", "agreement", "liable", "liability", "warranty", "indemnify",
    "litigation", "attorney", "counsel", "subpoena", "regulatory", "compliance",
    "violation", "fine", "audit", "retention", "gdpr", "hipaa", "sox",
    "pci-dss", "dora", "ccpa", "legal", "binding", "disclosure", "confidential",
    "privileged", "non-disclosure", "nda", "terms of service", "privacy policy",
}

SECURITY_KEYWORDS = {
    "vulnerability", "cve", "exploit", "malware", "ransomware", "breach",
    "intrusion", "attack", "threat", "phishing", "credential", "password",
    "secret", "token", "api key", "private key", "encryption", "firewall",
    "siem", "incident response", "forensics", "penetration", "zero-day",
    "sqli", "xss", "csrf", "backdoor", "rootkit", " trojan ", "logic bomb",
}

GOVERNANCE_KEYWORDS = {
    "policy", "control", "procedure", "standard", "guideline", "framework",
    "governance", "committee", "approval", "authorization", "finding",
    "exception", "waive", "waiver", "risk appetite", "sla", "slo", "kpi",
    "okr", "annual audit", "stakeholder", "fiduciary",
}

FINANCIAL_PATTERNS = [
    re.compile(r"\$\s*[\d,]+(?:\.\d{2})?"),
    re.compile(r"\b\d{1,3}(?:,\d{3})+\.\d{2}\b"),
    re.compile(r"\b(?:USD|EUR|GBP|JPY|CAD|AUD)\s*[\d,]+(?:\.\d{2})?"),
]

DEADLINE_PATTERNS = [
    re.compile(r"\b(?:deadline|due date|due by|expires?|expiration|valid until|must complete by)\b", re.I),
]

PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

CODE_LANGUAGE_HINTS = {
    "python": re.compile(r"\bdef\s+|\bclass\s+|\bimport\s+|\bfrom\s+\S+\s+import\b|:\s*$", re.M),
    "javascript": re.compile(r"\bconst\s+|\blet\s+|\bvar\s+|\bfunction\s+|\b=>\s*\{|\bconsole\.log\("),
    "typescript": re.compile(r"\binterface\s+|\btype\s+\w+\s*=\s*|:\s*(string|number|boolean|any)\b"),
    "java": re.compile(r"\bpublic\s+(?:class|static|void)\b|\bprivate\s+\w+\s+\w+\s*\("),
    "go": re.compile(r"\bfunc\s+\w+\s*\(|\bpackage\s+\w+\b|\btype\s+\w+\s+struct\b"),
    "rust": re.compile(r"\bfn\s+\w+\s*\(|\blet\s+mut\s+|\bimpl\s+|\buse\s+\w+::"),
    "sql": re.compile(r"\bSELECT\s+.+\bFROM\b|\bINSERT\s+INTO\b|\bUPDATE\s+\w+\s+SET\b", re.I),
}

DIFF_MARKERS = re.compile(r"^(---|\+\+\+|@@\s+-\d+|@@\s+\+\d+|diff\s+--git)", re.M)
LOG_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
LOG_SEVERITY_RE = re.compile(r"\b(?:INFO|WARN|WARNING|ERROR|FATAL|DEBUG|TRACE|CRITICAL)\b", re.I)
STACK_FRAME_RE = re.compile(r"\bFile\s+\"[^\"]+\",\s+line\s+\d+|\s+at\s+\S+\s+\([^)]+:\d+:\d+\)|^[\w\.]+\.[\w\.]+\([^)]+\)", re.M)
EXCEPTION_RE = re.compile(r"\b[A-Z][a-zA-Z0-9_]*(?:Error|Exception|Failure|Fault)\b")
URL_RE = re.compile(r"https?://\S+")
TABLE_RE = re.compile(r"\|\s*[-:]+\s*\||^\|.*\|$", re.M)
MARKDOWN_RE = re.compile(r"[#*_\[\]`~!]|\[.*?\]\(.*?\)|^>{1,}\s+", re.M)
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.\)])\s+\S", re.M)
IMAGE_KEYWORDS = {
    "image", "photo", "picture", "figure", "diagram", "png", "jpg", "jpeg",
    "gif", "webp", "svg", "bitmap", "resolution", "dimensions", "width=",
    "height=", "base64", "codec=", "color_space", "pixel", "rgba",
}
OCR_KEYWORDS = {
    "ocr", "scanned", "scan", "skew", "memorandum", "document_scan",
    "image of text", "handwriting", "typed document", "scan_reference",
}
SUMMARIZER_KEYWORDS = {
    "memo", "summary", "summarize", "executive summary", "key points",
    "blocker", "blockers", "identified", "requested", "decisions",
    "quarterly planning", "weekly planning", "status update",
}


def _token_count(text: str) -> int:
    """Cheap whitespace token estimate."""
    return max(1, len(text.split()))


def _char_entropy(text: str) -> float:
    """Shannon entropy over characters (bits per char)."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def _json_probe(text: str) -> tuple[bool, int]:
    """Return (parse_success, max_depth)."""
    try:
        data = json.loads(text)
    except Exception:
        return False, 0

    def _depth(obj: Any) -> int:
        if isinstance(obj, dict):
            return 1 + max((_depth(v) for v in obj.values()), default=0)
        if isinstance(obj, list):
            return 1 + max((_depth(v) for v in obj), default=0)
        return 0

    return True, _depth(data)


def _code_language(text: str) -> str | None:
    scores = {lang: len(pat.findall(text)) for lang, pat in CODE_LANGUAGE_HINTS.items()}
    if not any(scores.values()):
        return None
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] >= 2 else None


def _ast_parse_success(text: str, language: str | None) -> bool:
    """Lightweight AST validity check for Python and JSON only."""
    if language == "python":
        try:
            compile(text, "<string>", "exec")
            return True
        except Exception:
            return False
    if language is None and _json_probe(text)[0]:
        return True
    return False


def _markdown_density(text: str) -> float:
    if not text:
        return 0.0
    marker_chars = sum(1 for ch in text if ch in "#*_`~[]()>|")
    return min(1.0, marker_chars / max(1, len(text)))


def _diff_markers(text: str) -> int:
    return len(DIFF_MARKERS.findall(text))


def _log_timestamp_density(text: str) -> float:
    lines = text.splitlines()
    if not lines:
        return 0.0
    hits = sum(1 for line in lines if LOG_TIMESTAMP_RE.search(line))
    return hits / len(lines)


def _log_repetition_ratio(text: str) -> float:
    lines = text.splitlines()
    if not lines:
        return 0.0
    stripped = [re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*", "<TS>", line) for line in lines]
    stripped = [re.sub(r"\b(?:request|trace|span|correlation)-\S+", "<ID>", line) for line in stripped]
    unique = len(set(stripped))
    return 1.0 - (unique / max(1, len(lines)))


def _stacktrace_hits(text: str) -> int:
    return len(STACK_FRAME_RE.findall(text)) + len(EXCEPTION_RE.findall(text))


def _table_density(text: str) -> float:
    lines = text.splitlines()
    if not lines:
        return 0.0
    hits = sum(1 for line in lines if TABLE_RE.search(line))
    return hits / len(lines)


def _bullet_line_ratio(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    hits = sum(1 for line in lines if BULLET_RE.search(line))
    return hits / len(lines)


def _image_keyword_count(text: str) -> int:
    lowered = text.lower()
    return sum(1 for kw in IMAGE_KEYWORDS if kw in lowered)


def _ocr_keyword_count(text: str) -> int:
    lowered = text.lower()
    return sum(1 for kw in OCR_KEYWORDS if kw in lowered)


def _summarizer_keyword_count(text: str) -> int:
    lowered = text.lower()
    return sum(1 for kw in SUMMARIZER_KEYWORDS if kw in lowered)


def _prose_sentence_density(text: str) -> float:
    """Approximate sentences per line for generic prose detection."""
    if not text:
        return 0.0
    sentences = max(1, len(re.split(r"[.!?]+", text)) - 1)
    lines = max(1, text.count("\n") + 1)
    return sentences / lines


def _url_density(text: str) -> float:
    if not text:
        return 0.0
    return len(URL_RE.findall(text)) / max(1, _token_count(text))


def _pii_scan(text: str) -> tuple[int, list[str], float]:
    types_found: list[str] = []
    max_confidence = 0.0
    total = 0
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            types_found.append(pii_type)
            total += len(matches)
            max_confidence = max(max_confidence, 0.9 if pii_type != "email" else 0.95)
    return total, types_found, max_confidence


def _keyword_count(text: str, keywords: set[str]) -> int:
    """Count keyword phrases without matching inside larger words."""
    lowered = text.lower()
    return sum(
        1
        for keyword in keywords
        if re.search(
            rf"(?<!\w){re.escape(keyword.strip())}(?!\w)",
            lowered,
        )
    )


def _financial_amount_count(text: str) -> int:
    return sum(len(p.findall(text)) for p in FINANCIAL_PATTERNS)


def _deadline_date_count(text: str) -> int:
    return sum(len(p.findall(text)) for p in DEADLINE_PATTERNS)


def _image_hint(text: str) -> str:
    """Default image hint for text-only artifacts."""
    return "none"


def probe_text(
    text: str,
    source_channel: SourceChannel,
    policy_tier: PolicyTier,
    content_type: str = "unknown",
) -> FeatureVector:
    """Compute deterministic features for raw text."""
    if not text:
        # Fail-closed: empty text returns zeroed features but still valid.
        return FeatureVector(
            token_len=0,
            char_len=0,
            line_count=0,
            byte_len=0,
            char_entropy=0.0,
            json_parse_success=False,
            json_depth=0,
            markdown_density=0.0,
            code_language=None,
            ast_parse_success=False,
            diff_markers=0,
            log_timestamp_density=0.0,
            log_repetition_ratio=0.0,
            stacktrace_hits=0,
            table_density=0.0,
            bullet_line_ratio=0.0,
            image_keyword_count=0,
            ocr_keyword_count=0,
            summarizer_keyword_count=0,
            prose_sentence_density=0.0,
            url_density=0.0,
            pii_entity_count=0,
            pii_entity_types=[],
            pii_max_confidence=0.0,
            legal_keyword_count=0,
            security_keyword_count=0,
            governance_keyword_count=0,
            financial_amount_count=0,
            deadline_date_count=0,
            embedded_text_ratio=0.0,
            ocr_confidence=0.0,
            image_hint="none",
            source_channel=source_channel,
            policy_tier=policy_tier,
            content_type=content_type,
            text=text,
        )

    token_len = _token_count(text)
    char_len = len(text)
    line_count = text.count("\n") + 1
    byte_len = len(text.encode("utf-8"))
    entropy = _char_entropy(text)
    json_success, json_depth = _json_probe(text)
    md_density = _markdown_density(text)
    code_language = _code_language(text)
    ast_success = _ast_parse_success(text, code_language)
    diff_markers = _diff_markers(text)
    log_ts_density = _log_timestamp_density(text)
    log_repetition = _log_repetition_ratio(text)
    stack_hits = _stacktrace_hits(text)
    table_density = _table_density(text)
    bullet_ratio = _bullet_line_ratio(text)
    image_kw_count = _image_keyword_count(text)
    ocr_kw_count = _ocr_keyword_count(text)
    summarizer_kw_count = _summarizer_keyword_count(text)
    prose_sentence_density = _prose_sentence_density(text)
    url_density = _url_density(text)
    pii_count, pii_types, pii_max_conf = _pii_scan(text)
    legal_count = _keyword_count(text, LEGAL_KEYWORDS)
    security_count = _keyword_count(text, SECURITY_KEYWORDS)
    governance_count = _keyword_count(text, GOVERNANCE_KEYWORDS)
    financial_count = _financial_amount_count(text)
    deadline_count = _deadline_date_count(text)

    return FeatureVector(
        token_len=token_len,
        char_len=char_len,
        line_count=line_count,
        byte_len=byte_len,
        char_entropy=entropy,
        json_parse_success=json_success,
        json_depth=json_depth,
        markdown_density=md_density,
        code_language=code_language,
        ast_parse_success=ast_success,
        diff_markers=diff_markers,
        log_timestamp_density=log_ts_density,
        log_repetition_ratio=log_repetition,
        stacktrace_hits=stack_hits,
        table_density=table_density,
        bullet_line_ratio=bullet_ratio,
        image_keyword_count=image_kw_count,
        ocr_keyword_count=ocr_kw_count,
        summarizer_keyword_count=summarizer_kw_count,
        prose_sentence_density=prose_sentence_density,
        url_density=url_density,
        pii_entity_count=pii_count,
        pii_entity_types=pii_types,
        pii_max_confidence=pii_max_conf,
        legal_keyword_count=legal_count,
        security_keyword_count=security_count,
        governance_keyword_count=governance_count,
        financial_amount_count=financial_count,
        deadline_date_count=deadline_count,
        embedded_text_ratio=1.0,  # text artifact
        ocr_confidence=0.0,
        image_hint="none",
        source_channel=source_channel,
        policy_tier=policy_tier,
        content_type=content_type,
        text=text,
    )


def probe(envelope: ArtifactEnvelope) -> FeatureVector:
    """Compute deterministic features from an artifact envelope."""
    text = envelope.text_content or ""
    content_type = envelope.mime_type.split("/")[-1] if envelope.mime_type else "unknown"
    return probe_text(
        text,
        source_channel=envelope.source_channel,
        policy_tier=envelope.metadata.policy_tier,
        content_type=content_type,
    )
