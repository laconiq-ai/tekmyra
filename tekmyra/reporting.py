"""Report builders for benchmark results."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from tekmyra import __version__
from tekmyra.codecs.selector import UNIMPLEMENTED_CODEC_ROUTES
from tekmyra.contracts import RouteClass
from tekmyra.pii_vocabulary import LEAKAGE_TYPES, REPORTING_TYPES


STRUCTURAL_TYPES = {"json", "code", "diff", "log", "stack"}
# Route names whose codec cannot compress. Derived from the selector so a
# newly-built codec drops out of this set automatically; a no-op must never
# be averaged into the compression-ratio aggregates as a real 1.0.
UNIMPLEMENTED_ROUTE_NAMES = frozenset(r.value for r in UNIMPLEMENTED_CODEC_ROUTES)
LOSSY_ROUTES = {
    "prose_token_compressor",
    "summarizer",
    "extractive_keypoint_compressor",
}


def _compute_baseline_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics for a single baseline's result rows."""
    if not rows:
        return {
            "overall_route_accuracy": 0.0,
            "structural_route_precision": 0.0,
            "risk_accuracy": 0.0,
            "high_risk_count": 0,
            "high_risk_lossy_count": 0,
            "unimplemented_codec_count": 0,
            "mean_compression_ratio": 0.0,
            "median_compression_ratio": 0.0,
            "min_compression_ratio": 0.0,
            "max_compression_ratio": 0.0,
            "accuracy_by_route_class": {rc.value: 0.0 for rc in RouteClass},
            "compression_ratio_by_block_type": {},
        }

    total = len(rows)
    route_correct = sum(1 for r in rows if r.get("predicted_route") == r.get("route"))
    structural_rows = [r for r in rows if r.get("content_type") in STRUCTURAL_TYPES]
    structural_correct = sum(
        1 for r in structural_rows if r.get("predicted_route") == r.get("route")
    )
    risk_correct = sum(1 for r in rows if r.get("predicted_risk") == r.get("risk"))
    high_risk_rows = [r for r in rows if r.get("risk") in {"high", "critical"}]
    high_risk_lossy = sum(
        1 for r in high_risk_rows if r.get("predicted_route") in LOSSY_ROUTES
    )

    detected_rows = [r for r in rows if r.get("locked_spans_detected", 0) > 0]
    total_detected = sum(r.get("locked_spans_detected", 0) for r in detected_rows)
    total_preserved = sum(r.get("locked_spans_preserved", 0) for r in detected_rows)
    locked_span_recall = total_preserved / total_detected if total_detected > 0 else 1.0

    pii_leakage_count = 0
    for r in rows:
        if r.get("predicted_route") not in LOSSY_ROUTES:
            continue
        protected_spans = r.get("protected_spans", [])
        pii_spans = [
            span
            for span in protected_spans
            if span.get("span_type") in LEAKAGE_TYPES
        ]
        if not pii_spans:
            continue
        output_text = r.get("output_text", "")
        if any(span.get("text", "") in output_text for span in pii_spans):
            pii_leakage_count += 1

    # Rows routed to a declared-but-unbuilt codec produce no compression at
    # all. They are counted as their own bucket and kept OUT of the ratio
    # aggregates, so a no-op can never read as a genuine 1.0 result.
    unimplemented_codec_count = sum(
        1 for r in rows if r.get("predicted_route") in UNIMPLEMENTED_ROUTE_NAMES
    )

    ratios = [
        float(value)
        for r in rows
        if r.get("predicted_route") not in UNIMPLEMENTED_ROUTE_NAMES
        and isinstance((value := r.get("compression_ratio")), (int, float))
        and not isinstance(value, bool)
    ]
    if ratios:
        mean_compression_ratio = mean(ratios)
        median_compression_ratio = median(ratios)
        min_compression_ratio = min(ratios)
        max_compression_ratio = max(ratios)
    elif unimplemented_codec_count == len(rows):
        # Every row hit an unbuilt codec, so nothing was compressed. On this
        # scale 1.0 IS that truth; 0.0 would read as PERFECT compression, the
        # exact opposite. Matches the `if ... else 1.0` idiom in
        # benchmarks/learned_compressor/evaluate_hierarchical.py. The 0.0
        # default below is preserved for its own distinct case: rows whose
        # compression_ratio field is absent or non-numeric.
        mean_compression_ratio = median_compression_ratio = min_compression_ratio = (
            max_compression_ratio
        ) = 1.0
    else:
        mean_compression_ratio = median_compression_ratio = min_compression_ratio = (
            max_compression_ratio
        ) = 0.0

    accuracy_by_route_class: dict[str, float] = {}
    for rc in RouteClass:
        class_total = sum(1 for r in rows if r.get("route") == rc.value)
        class_correct = sum(
            1
            for r in rows
            if r.get("route") == rc.value and r.get("predicted_route") == rc.value
        )
        accuracy_by_route_class[rc.value] = (
            class_correct / class_total if class_total > 0 else 0.0
        )

    block_ratio_sums: dict[str, float] = defaultdict(float)
    block_ratio_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        blocks = r.get("blocks")
        if not blocks:
            continue
        ratio = r.get("compression_ratio")
        if ratio is None:
            continue
        for block in blocks:
            block_type = (
                block.get("type")
                if isinstance(block, dict)
                else getattr(block, "type", None)
            )
            if block_type:
                block_ratio_sums[block_type] += ratio
                block_ratio_counts[block_type] += 1
    compression_ratio_by_block_type = {
        block_type: block_ratio_sums[block_type] / block_ratio_counts[block_type]
        for block_type in block_ratio_counts
    }

    def _ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator > 0 else 0.0

    return {
        "overall_route_accuracy": _ratio(route_correct, total),
        "structural_route_precision": _ratio(structural_correct, len(structural_rows)),
        "risk_accuracy": _ratio(risk_correct, total),
        "high_risk_count": len(high_risk_rows),
        "high_risk_lossy_count": high_risk_lossy,
        "locked_span_recall": locked_span_recall,
        "pii_leakage_count": pii_leakage_count,
        "unimplemented_codec_count": unimplemented_codec_count,
        "mean_compression_ratio": mean_compression_ratio,
        "median_compression_ratio": median_compression_ratio,
        "min_compression_ratio": min_compression_ratio,
        "max_compression_ratio": max_compression_ratio,
        "accuracy_by_route_class": accuracy_by_route_class,
        "compression_ratio_by_block_type": compression_ratio_by_block_type,
    }


def build_json_report(
    results: list[dict[str, Any]],
    fixture_count: int | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the benchmark report.

    ``provenance`` records WHICH SYSTEM produced these numbers and is required
    before any figure here may be quoted (Vega ruling section 0, 2026-08-07).
    Without it a run against a loaded learned compressor and a run against a
    NO-OP fallback render identically — and they are not the same claim.

    That is not hypothetical on this path. ``benchmark.py`` exercises the
    BASELINES, and the ``learned``/``tekmyra_learned`` baselines call
    ``load_compressor()``, which returns a no-op ``ProseCompressor(None)`` when
    ``.cache/compressor`` is absent. ``.cache/`` is gitignored, so a clean
    checkout measures the no-op and reports a compression ratio for it.

    Defaults to ``None`` rather than to an empty dict so that an omitted
    provenance is recorded as UNKNOWN and marked not-quotable, instead of
    rendering as "we checked and there was nothing to say".
    """
    by_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_baseline[r.get("baseline", "unknown")].append(r)

    summary = {
        "by_baseline": {
            baseline: _compute_baseline_summary(rows)
            for baseline, rows in by_baseline.items()
        },
    }

    if provenance is None:
        provenance = {
            "compressor_artifact_loaded": "UNKNOWN",
            "quotable": False,
            "note": (
                "NO PROVENANCE RECORDED. This report cannot say whether the learned "
                "compressor artifact was loaded or whether a no-op fallback produced "
                "these ratios. Per Vega ruling section 0 it must not be published or "
                "cited internally."
            ),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_version": __version__,
        "total_fixtures": fixture_count if fixture_count is not None else len(results),
        "total_results": len(results),
        "provenance": provenance,
        "summary": summary,
        "results": results,
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# TekMyra Benchmark Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Fixtures: {report['total_fixtures']}",
        "",
    ]

    # The human artifact must not be able to omit what the JSON records. A
    # markdown report is what actually gets pasted into a message, so a
    # provenance block present in the JSON and absent here would be provenance
    # that exists and never reaches a reader.
    prov = report.get("provenance") or {}
    loaded = prov.get("compressor_artifact_loaded", "UNKNOWN")
    if prov.get("quotable") is True:
        lines += [f"Compressor artifact: LOADED ({prov.get('artifact_digest', 'no digest')[:16]})", ""]
    else:
        lines += [
            "> **NOT QUOTABLE — DO NOT PUBLISH OR CITE.**",
            f"> Compressor artifact loaded: `{loaded}`.",
            f"> {prov.get('note', 'No provenance recorded.')}",
            "",
        ]

    lines += [
        "## Summary",
        "",
    ]

    summary = report.get("summary", {})
    by_baseline = summary.get("by_baseline", {})
    if by_baseline:
        for baseline, metrics in by_baseline.items():
            lines.append(
                f"**{baseline}**: route_accuracy={metrics['overall_route_accuracy']:.2f}, "
                f"structural_precision={metrics['structural_route_precision']:.2f}, "
                f"risk_accuracy={metrics['risk_accuracy']:.2f}, "
                f"high_risk_count={metrics['high_risk_count']}, "
                f"high_risk_lossy_count={metrics['high_risk_lossy_count']}, "
                f"locked_span_recall={metrics['locked_span_recall']:.2f}, "
                f"pii_leakage_count={metrics['pii_leakage_count']}."
            )
    else:
        lines.append("No aggregate metrics available.")

    lines.extend(
        [
            "",
            "| fixture | baseline | expected_route | predicted_route | expected_risk | predicted_risk | rules_fired | ratio | verifier |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for r in report["results"]:
        rules = ",".join(r.get("rules_fired", [])) or "-"
        lines.append(
            f"| {r['fixture_id']} | {r['baseline']} | {r['route']} | "
            f"{r.get('predicted_route', '-')} | {r['risk']} | "
            f"{r.get('predicted_risk', '-')} | {rules} | "
            f"{r['compression_ratio']:.2f} | {r['verifier_pass']} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    with open(output_dir / "benchmark_report.md", "w") as f:
        f.write(build_markdown_report(report))
