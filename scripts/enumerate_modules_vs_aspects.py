#!/usr/bin/env python3
"""File-by-file enumeration: the 54 published modules against the aspect set.

PRODUCED BY SCRIPT, NOT ASSEMBLED BY HAND, for the same reason the staging tree
is: a hand-built table cannot be re-derived by the person who has to rely on it,
and this one is going to counsel.

WHAT A ROW MEANS, AND WHAT IT DOES NOT. A hit is a NAMED SYMBOL located in a
published module — evidence that the module is where a limb would be found, and
a pointer for a reader who needs to go look. **It is not a verdict that the
module practises the aspect.** Every verdict in the map itself rests on a
behavioural measurement under proven isolation; this table exists so that a
reader can check the map against the artifact rather than take its word.

The symbols below were chosen AFTER the measurements, from the code paths those
measurements actually exercised, so the table points at the mechanism rather
than at whatever shares a word with the aspect's wording. A module with no hits
is listed too — silence is part of the enumeration, and a table that omitted the
misses would read as though 54 modules were all implicated.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

def staged_tree() -> Path:
    """The tree to enumerate. REQUIRED — there is deliberately no default.

    A default would silently enumerate whatever tree happened to be lying at a
    hardcoded path, and the resulting table would carry a commit's authority
    while describing some other build. The caller states which artifact this
    describes, or gets nothing.
    """
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    raise SystemExit(
        "usage: enumerate_modules_vs_aspects.py <staged-tree>\n"
        "  produce one with: scripts/stage_open_core.py --dest <path>"
    )

# aspect -> (short label, [regexes of NAMED symbols, not topic words])
ASPECTS: dict[str, tuple[str, list[str]]] = {
    "A1.L1": ("detect protected spans", [r"\bbuild_fact_index\b", r"\bProtectedSpan\b", r"\bdetect_protected_spans\b"]),
    "A1.L2/A2": ("disposition + monotone escalation", [r"\bpolicy_for_risk\b", r"\banchorize\b", r"\bhuman_review\b"]),
    "A1.L3": ("encrypted tenant-partitioned store", [r"\bAES\b", r"\bGCM\b", r"\bFernet\b", r"\bcryptography\b", r"\b_init_schema\b"]),
    "A1.L4": ("mint in-band structured token", [r"\bAnchorRecord\b", r"CTX_REF", r"\bcontent_hash\b", r"\banchor_id\b"]),
    "A1.L5": ("compress the artifact", [r"\bCodecResult\b", r"\bcompress\w*\(", r"\bselect_codec\b"]),
    "A1.L6": ("verify exactly-once", [r"_count_exact_span_dispositions", r"_check_protected_disposition", r"\b_span_key\b"]),
    "A1.L7": ("emit only on success", [r"\bTerminalVerificationError\b"]),
    "A3": ("store unavailability -> refusal", [r"\banchors_resolvable\b", r"NonResolvingAnchorStore"]),
    "A4": ("fallback then terminal", [r"\bfallback_route\b", r"verification_failed_after_fallback"]),
    "A5": ("one id one hash, refuse collisions", [r"\bcollision\b", r"IntegrityError", r"UNIQUE"]),
    "A6": ("AEAD, anchor id as AAD", [r"\baad\b", r"associated_data", r"\bencrypt\("]),
    "A7": ("keyed digests only", [r"\bhmac\b", r"hash_salt", r"\bkeyed\b"]),
    "A8": ("cross-tenant refusal + audit", [r"requesting_tenant_id", r"tenant_context_mismatch", r"\baudit_log\b"]),
    "A9": ("ex-ante economics gate", [r"\bpredicted_savings\b", r"\bsafety_margin\b", r"\bbudget_policy\b", r"\beligibility\b"]),
    "A10": ("ratio pinned to signed benchmark", [r"\bsigned_benchmark\b", r"calibrated_target_ratio"]),
    "A11": ("tenant erasure + tombstone", [r"\btombstone\b", r"\berasure\b", r"\berase_tenant\b"]),
    "A12": ("learned risk raises only; lossy->reversible", [r"_highest_risk", r"SAFETY_OVERRIDE_LEARNED_LOSSY", r"_LOSSY_ROUTES", r"REVERSIBLE_RETRIEVAL_MODE"]),
}


def main() -> int:
    STAGED = staged_tree()
    modules = sorted(
        p for p in (STAGED / "tekmyra").rglob("*.py") if "__pycache__" not in p.parts
    )
    rows = []
    per_aspect: dict[str, list[str]] = {k: [] for k in ASPECTS}
    for path in modules:
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(STAGED).as_posix()
        hits: dict[str, list[str]] = {}
        for aspect, (_label, patterns) in ASPECTS.items():
            found = sorted({
                m.group(0)
                for pattern in patterns
                for m in re.finditer(pattern, text)
            })
            if found:
                hits[aspect] = found[:4]
                per_aspect[aspect].append(rel)
        rows.append({"module": rel, "lines": text.count("\n") + 1, "aspects": hits})

    silent = [r["module"] for r in rows if not r["aspects"]]
    print(json.dumps({
        "staged_modules": len(modules),
        "modules_with_no_aspect_symbol": len(silent),
        "aspects_with_no_module_in_the_published_surface": [
            a for a, mods in per_aspect.items() if not mods
        ],
        "modules_per_aspect": {a: len(m) for a, m in per_aspect.items()},
        "per_aspect_modules": per_aspect,
        "rows": rows,
        "silent_modules": silent,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
