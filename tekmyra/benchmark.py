"""Click-driven benchmark harness for TekMyra runtime baselines."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import click

from tekmyra.baselines import BASELINES, run_baseline
from tekmyra.benchmark_compressor import (
    print_compressor_report,
    run_compressor_benchmark,
)
from tekmyra.contracts import CompressionResponse
from tekmyra.fixtures import Fixture, load_fixtures
from tekmyra.onnx_runtime import engine_availability
from tekmyra.pii_vocabulary import REPORTING_TYPES
from tekmyra.reporting import build_json_report, write_reports

DEFAULT_FIXTURE_DIR = Path("benchmarks/fixtures/synthetic")
DEFAULT_OUTPUT_DIR = Path("benchmarks/results")


def _extract_decision(response: CompressionResponse, fixture: Fixture) -> dict[str, Any]:
    """Pull deterministic decision fields out of a CompressionResponse."""
    explanation = response.explanations[0] if response.explanations else {}
    return {
        "predicted_route": explanation.get("route_after_override") or fixture.gold_output.expected_route.value,
        "predicted_risk": explanation.get("risk_class") or fixture.gold_output.expected_risk.value,
        "rules_fired": explanation.get("rules_fired", []),
        "explanation_tags": explanation.get("explanation_tags", []),
        "fallback_route": explanation.get("fallback_route"),
        "abstained": explanation.get("abstained", False),
        "fail_safe_triggered": response.global_metrics.fail_safe_triggered,
    }


def _extract_output_text(response: CompressionResponse, fixture: Fixture) -> str:
    """Return the compressed output text, falling back to original content."""
    if response.compressed_segments:
        return response.compressed_segments[0].compressed_content
    return fixture.content


def _extract_preservation(response: CompressionResponse, fixture: Fixture) -> dict[str, int]:
    """Return locked-span preservation counts from the response or a safe fallback."""
    report = response.preservation_report
    if report is None:
        span_count = len(fixture.protected_spans)
        return {
            "locked_spans_detected": span_count,
            "locked_spans_preserved": span_count,
        }
    return {
        "locked_spans_detected": report.locked_spans_detected,
        "locked_spans_preserved": report.locked_spans_preserved,
    }


LEARNED_BASELINES = frozenset({"learned", "tekmyra_learned", "nova_learned"})


def capture_compressor_provenance(baselines: list[str]) -> dict[str, Any]:
    """Record whether the learned compressor artifact actually loaded.

    Vega ruling section 0 (2026-08-07) requires a report to say which system
    produced its numbers. On THIS path the relevant system is not a router --
    ``run_benchmark`` exercises the baselines and none of them construct a
    LearnedRouter -- it is the prose compressor, which the ``learned`` family
    of baselines obtains from ``load_compressor()``.

    That matters because ``load_compressor`` returns a no-op
    ``ProseCompressor(None)`` when its artifact is missing, and the artifact
    lives in ``.cache/compressor``, which is gitignored. So a clean checkout
    runs the "learned" baseline against a compressor that compresses nothing
    and reports a ratio for it, indistinguishable from a real result.
    """
    from tekmyra.learned_compressor.compressor import (
        last_artifact_resolution,
        load_compressor,
    )

    uses_learned = bool(LEARNED_BASELINES.intersection(baselines))
    try:
        loaded = load_compressor()._model is not None
    except Exception as exc:  # noqa: BLE001 — provenance must not break a run
        return {
            "compressor_artifact_loaded": "UNKNOWN",
            "quotable": False,
            "note": f"could not determine artifact state: {exc}",
        }

    # ASK THE LOADER WHERE IT LOOKED; DO NOT RE-DERIVE IT.
    #
    # This previously took the module's relative constant and resolved it here,
    # against this process's working directory. When the loader resolved to a
    # different place — which was the whole defect — the two disagreed, and the
    # provenance block reported on a directory the loader had never opened.
    # That is how `run_three_method_comparison` came to print the cache
    # directories PRESENT inside the exception saying they were missing.
    resolution = last_artifact_resolution()
    artifact_dir = resolution.path or resolution.fallback() if resolution else Path(".")
    files = sorted(p for p in artifact_dir.rglob("*") if p.is_file()) if artifact_dir.is_dir() else []
    digest = "UNKNOWN"
    if files:
        h = hashlib.sha256()
        for p in files:
            h.update(str(p.relative_to(artifact_dir)).encode())
            h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode())
        digest = h.hexdigest()

    # A run that does not exercise a learned baseline is quotable without the
    # artifact: it never claimed to use one. A run that DOES is not.
    quotable = loaded or not uses_learned
    return {
        "compressor_artifact_loaded": loaded,
        "artifact_dir": str(artifact_dir),
        # EVERY LOCATION THE LOADER TRIED, in order, with what was there. A
        # missing artifact is now answerable — "it looked here, here and here"
        # — instead of leaving the reader to guess which root was meant.
        "artifact_search": resolution.describe() if resolution else "not searched",
        "artifact_digest": digest,
        "artifact_tracked_in_git": False,  # .cache/ is gitignored by construction
        "artifact_files": len(files),
        "learned_baselines_requested": sorted(LEARNED_BASELINES.intersection(baselines)),
        "quotable": quotable,
        "note": (
            "artifact present; it is gitignored and machine-local, so this digest must "
            "travel with any figure quoted from this run"
            if loaded else
            "LEARNED BASELINE RUN WITHOUT ITS ARTIFACT — load_compressor returned a "
            "no-op, so every 'learned' ratio here describes a compressor that "
            "compresses nothing. Not a worse result: a different system."
            if uses_learned else
            "no learned baseline requested; the missing artifact does not affect these "
            "baselines"
        ),
    }


def run_benchmark(
    fixture_dir: Path,
    baselines: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    fixtures = load_fixtures(fixture_dir)
    provenance = capture_compressor_provenance(baselines)
    results: list[dict[str, Any]] = []

    for fixture in fixtures:
        for baseline_name in baselines:
            start = time.perf_counter()
            response = run_baseline(baseline_name, fixture)
            elapsed_ms = (time.perf_counter() - start) * 1000

            decision = _extract_decision(response, fixture)
            preservation = _extract_preservation(response, fixture)
            output_text = _extract_output_text(response, fixture)
            pii_entity_types = sorted({
                span.span_type for span in fixture.protected_spans
                if span.span_type in REPORTING_TYPES
            })

            result = {
                "fixture_id": fixture.fixture_id,
                "content_type": fixture.content_type,
                "baseline": baseline_name,
                "route": fixture.gold_output.expected_route.value,
                "risk": fixture.gold_output.expected_risk.value,
                "compression_ratio": response.global_metrics.compression_ratio,
                "input_tokens": response.global_metrics.input_tokens,
                "output_tokens": response.global_metrics.output_tokens,
                "verifier_pass": response.global_metrics.verifier_pass,
                "latency_ms": elapsed_ms,
                "locked_spans_detected": preservation["locked_spans_detected"],
                "locked_spans_preserved": preservation["locked_spans_preserved"],
                "pii_entity_types": pii_entity_types,
                "output_text": output_text,
                "protected_spans": [span.model_dump() for span in fixture.protected_spans],
                "blocks": [block.model_dump() for block in fixture.blocks],
                **decision,
            }
            results.append(result)

    report = build_json_report(results, len(fixtures), provenance=provenance)
    report["engine_availability"] = engine_availability()
    write_reports(output_dir, report)
    return report


@click.group()
def cli() -> None:
    """TekMyra Phase 0 benchmark harness."""


@cli.command()
@click.option("--fixture-dir", type=click.Path(path_type=Path), default=DEFAULT_FIXTURE_DIR)
@click.option("--baselines", default="no_compression,passthrough,deterministic,learned", help="Comma-separated baseline names")
@click.option("--output-dir", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_DIR)
def run(fixture_dir: Path, baselines: str, output_dir: Path) -> None:
    """Run selected baselines against fixtures and write reports."""
    baseline_list = [b.strip() for b in baselines.split(",") if b.strip()]
    for b in baseline_list:
        if b not in BASELINES:
            raise click.BadParameter(f"Unknown baseline: {b}")
    # A SHIPPED SUBCOMMAND MUST NOT DIE WITH A RAW TRACEBACK ON THE BUILD THAT
    # SHIPS IT. The open core cannot resolve anchors, so the deterministic and
    # learned baselines legitimately REFUSE on fixtures with protected spans —
    # correct behaviour, reported here as a stack trace ending in
    # TerminalVerificationError. Same seam class as the five tests that asserted
    # commercial outcomes: a caller written when the commercial build was the
    # only build.
    #
    # This is NOT documented in the payload's README or CONTRIBUTING — the
    # documented path is `benchmark compressor`, which works. Fixed anyway,
    # because "nobody is directed to it" is the reasoning that left those five
    # tests broken, and a stranger reading `--help` finds this command in one
    # line. It refuses with a reason instead of crashing.
    from tekmyra.runtime import TerminalVerificationError

    try:
        report = run_benchmark(fixture_dir, baseline_list, output_dir)
    except TerminalVerificationError as exc:
        raise click.ClickException(
            f"the verifier refused this run: {exc}. This build cannot resolve "
            "anchors, so baselines that must restore protected spans decline "
            "rather than emit a number — that refusal is the correct outcome, "
            "not a crash. Use `benchmark compressor --allow-rules-only` for a "
            "figure this build can produce."
        ) from None
    engines = report.get("engine_availability", {})
    if engines.get("complete", True):
        click.echo(f"Benchmark complete. {report['total_fixtures']} fixtures evaluated.")
    else:
        # Not a refusal: these figures do not depend on the ONNX path, and a
        # core install legitimately has no onnxruntime. But the completion line
        # must not read the same as a run where every engine came up.
        click.echo(
            f"Benchmark complete WITH DEGRADED ENGINES. "
            f"{report['total_fixtures']} fixtures evaluated; "
            f"{engines['failed']} of {engines['requested']} ONNX engine loads FAILED "
            f"across {engines['distinct_models_failed']} distinct model(s).",
            err=True,
        )
        for failure in engines["failures"]:
            click.echo(
                f"  engine unavailable: {Path(failure['model']).name} "
                f"({failure['attempts']} attempts) — {failure['error']}",
                err=True,
            )
        click.echo(
            "  These figures do not depend on the ONNX path. Install the "
            "'transformer' extra to exercise it. Recorded under "
            "'engine_availability' in the JSON report.",
            err=True,
        )
    click.echo(f"Reports written to {output_dir}")


@cli.command()
@click.option("--fixture-dir", type=click.Path(path_type=Path), default=DEFAULT_FIXTURE_DIR)
def list_fixtures(fixture_dir: Path) -> None:
    """List loaded fixtures and their labels."""
    fixtures = load_fixtures(fixture_dir)
    for fx in fixtures:
        click.echo(f"{fx.fixture_id}\t{fx.content_type}\t{fx.risk_tier.value}\t{fx.gold_output.expected_route.value}")


@cli.command()
@click.option(
    "--fixture-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_FIXTURE_DIR,
    show_default=True,
    help="Corpus to measure. The benchmark corpora ship in the repository, "
         "not inside the installed wheel — run from a clone's root, or point "
         "this at your own fixture directory.",
)
@click.option("--passthrough-token-budget", type=int, default=0)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
# THE ERROR MESSAGE PROMISED THIS FLAG AND THE COMMAND DID NOT HAVE IT. Without
# a trained artifact this command raises RouterArtifactUnavailable, whose text
# says "Re-run with --allow-rules-only" — and the only definition of that flag
# lived in benchmarks/evaluate_compressor.py, which does not ship. So the first
# command the published CONTRIBUTING.md tells a stranger to run failed, and its
# own stated remedy could not be obtained from anything in the published tree.
# The keyword already existed on run_compressor_benchmark and was already
# exercised by tests; only the CLI wiring was missing.
@click.option(
    "--allow-rules-only",
    is_flag=True,
    default=False,
    help="Record an explicitly rules-only run when no learned artifact is "
         "present. Such a run CANNOT emit a headline compression ratio.",
)
def compressor(fixture_dir: Path, passthrough_token_budget: int,
               output_dir: Path | None, allow_rules_only: bool) -> None:
    """Run a dedicated Phase 5 compressor benchmark on ambiguous fixtures."""
    # THE DEFAULT USED TO POINT AT A CORPUS THAT DOES NOT SHIP.
    # ``benchmarks/fixtures/learned_train`` is a training corpus and is not
    # part of the published package, so the default invocation — the first
    # command a new cloner types — failed on a missing path. A default must
    # point at something the artifact contains, or refuse with a reason;
    # failing on absence is the footgun class this release is removing, not an
    # instance of it worth keeping. (This history lived in the docstring, which
    # click renders as ``--help`` — an internal postmortem is not user help.)
    #
    # AND THE REFUSAL BELOW MUST NOT LIE ABOUT WHERE CORPORA LIVE. The wheel
    # contains the library and its resources but NO benchmarks/ — measured by
    # listing a built wheel — so a pip-install user hitting this refusal was
    # told the corpora live "under benchmarks/fixtures/" of a tree they do not
    # have. The corpora ship in the REPOSITORY; the message says which artifact
    # carries what.
    if not fixture_dir.is_dir():
        raise click.BadParameter(
            f"fixture directory not found: {fixture_dir}\n"
            "The benchmark corpora ship in the repository, not inside the "
            "installed wheel. Clone the repository and run this command from "
            "the clone's root, or pass --fixture-dir pointing at a fixture "
            "directory you have.",
            param_hint="--fixture-dir",
        )
    output_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "compressor_report.json"
    # THE SAME RULE AS `run` ABOVE, AND THIS IS THE COMMAND THE README ACTUALLY
    # PUBLISHES. Measured in a clean payload with no model present: this died
    # with a 25-line stack trace ending in require_loaded. The refusal itself is
    # correct and its text is already good — it names every path searched and
    # both remedies — but a traceback tells a first-time reader they hit a bug,
    # not that they skipped a documented download step. `run` was given this
    # treatment in ef2078a and this one was missed in the same file, which is
    # why the fix is applied by class here rather than to a single symptom.
    #
    # One thing the underlying message cannot know: it says "if your artifacts
    # live elsewhere", which presumes the reader HAS artifacts. On a fresh clone
    # they have none, and that is the overwhelmingly likely case for anyone
    # seeing this. The CLI adds that, because it is the layer that knows the
    # command was typed by hand.
    from tekmyra.learned_router.model import RouterArtifactLoadError
    from tekmyra.router_provenance import RouterArtifactUnavailable

    try:
        report = run_compressor_benchmark(
            fixture_dir=fixture_dir,
            passthrough_token_budget=passthrough_token_budget,
            output_path=output_path,
            allow_rules_only=allow_rules_only,
        )
    except RouterArtifactUnavailable as exc:
        raise click.ClickException(
            f"{exc}\n\n"
            "If you have not downloaded the model yet, that is almost certainly "
            "the cause, and it is the expected state of a fresh clone: the "
            "trained artifact is a separate release asset, not part of this "
            "repository. See 'You need a model' in the README for the fetch "
            "procedure. To measure what this build can produce without one, "
            "re-run with --allow-rules-only — that records an explicitly "
            "rules-only run and cannot emit a headline compression ratio."
        ) from None
    except RouterArtifactLoadError as exc:
        raise click.ClickException(
            f"the learned router artifact is present but could not be loaded: "
            f"{exc}\n\n"
            "This is a different problem from a missing model: something is "
            "there and it is not usable. Check that the download completed and "
            "that its digest matches the release notes before re-fetching. "
            "`--allow-rules-only` will still produce a rules-only measurement."
        ) from None
    print_compressor_report(report)
    if output_path:
        click.echo(f"Report written to {output_path}")


if __name__ == "__main__":
    cli()
