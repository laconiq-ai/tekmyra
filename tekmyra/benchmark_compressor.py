"""Dedicated Phase 5 compressor benchmark.

Bypasses the short-safe passthrough rule so the learned compressor is actually
invoked on ambiguous prose/RAG/agent-trace fixtures.

Honesty contract (Vega ruling 2026-08-07 §0, blocking):

* the report records which router actually ran — version, backend, artifact path
  and per-file digests, and whether that artifact ships in the repo;
* a missing or unloadable artifact **refuses to run** rather than abstaining
  silently and reporting a headline ratio;
* a run in which the learned router abstained on every row cannot emit a
  headline ratio — the field is withheld and the reason recorded;
* fixtures the pipeline refused (terminal verification failures) are counted and
  reported in the same table, not dropped so the survivors look clean;
* every aggregate is reported BESIDE its per-route distribution, because a
  corpus mean hides which routes earned it. If two routes compress hard and four
  are flat, the mean describes neither, and abstention (routing to
  REVERSIBLE_RETRIEVAL_MODE rather than compressing) is invisible inside it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tekmyra.codecs.selector import (
    SAFE_TERMINAL_ROUTES,
    UNIMPLEMENTED_CODEC_ROUTES,
)
from tekmyra.contracts import RouteClass
from tekmyra.corpus_vocabulary import MEASURABLE_CONTENT_TYPES
from tekmyra.fixtures import Fixture, load_fixtures
from tekmyra.hybrid_router import HybridRouter
from tekmyra.anchor_interface import default_anchor_store
from tekmyra.router_provenance import (
    capture,
    capture_anchor_store,
    require_loaded,
)
from tekmyra.runtime import (
    _DEFAULT_BUDGET_RATIO,
    TerminalVerificationError,
    run_hybrid_pipeline,
)
from tekmyra.token_measure import (
    ENCODING_NAME,
    TokenizerUnavailable,
    count_tokens,
    tokenizer_available,
)

class CorpusEligibilityError(RuntimeError):
    """The corpus contains fixtures, but the eligibility filter matched none.

    Distinct from an empty corpus. An empty result here means the harness and
    the corpus disagree about content-type vocabulary, which is a configuration
    defect, not a measurement of zero.
    """


# THE ELIGIBLE VOCABULARY, AND WHY IT IS LONGER THAN IT LOOKS.
#
# Three corpora were built at different times and named the same kinds of text
# differently. This set originally read {"prose", "rag", "agent_trace"} — which
# matches `prose` in the synthetic corpus, `agent_trace` in the training corpus,
# and NOTHING AT ALL in long_context_v1, whose six types are all natural-language
# text a prose compressor is squarely meant to handle. The corpus was not
# excluded on the merits; it was excluded because nobody reconciled two
# vocabularies. That is how the only long-context corpus we own went unmeasured
# while every published figure came from 250-character snippets.
#
# `rag` is kept although it currently matches no fixture in any corpus: removing
# it would be a second silent vocabulary change, and the drift guard below names
# it out loud instead.
#
# DELIBERATELY NOT ADDED: the synthetic corpus's `retrieval_context`. It is the
# RAG analogue and probably belongs here, but adding it would move the published
# synthetic headline (0.7409 over 28 prose fixtures) in the same commit that
# unblocks a different corpus. A number that moves for two reasons at once
# cannot be attributed to either. That is its own reviewable change.
#
# DERIVED, NOT RESTATED — see tekmyra.corpus_vocabulary. This set and the
# trainer's had drifted apart by one type with nobody deciding to, and the guard
# written after the first drift watched only this copy.
_ELIGIBLE_TYPES = MEASURABLE_CONTENT_TYPES
_COMPRESSOR_ROUTE = RouteClass.PROSE_TOKEN_COMPRESSOR

# Routes that actually try to compress: the taxonomy minus the ones sharing the
# passthrough no-op and minus the ones declared but unbuilt.
#
# DERIVED, NOT LISTED. A hardcoded pair here would have been correct for
# exactly three days — LEGAL_SECURITY_SAFE_MODE moved from passthrough to a
# real codec at c2b9e45, and a literal list would have quietly kept reporting
# it as non-compressing. Both source sets are themselves derived from the codec
# map, so building a codec moves a route into this set automatically.
_COMPRESSING_ROUTES: frozenset[str] = frozenset(
    route.value
    for route in RouteClass
    if route not in SAFE_TERMINAL_ROUTES and route not in UNIMPLEMENTED_CODEC_ROUTES
)

# Explanation tags/rules emitted when the learned router declined to decide.
_ABSTENTION_MARKERS = ("LEARNED_ABSTAINED", "LEARNED_NO_TEXT")


def _is_eligible(fixture: Fixture) -> bool:
    return fixture.content_type in _ELIGIBLE_TYPES


def _abstention_reason(explanation: dict[str, Any]) -> str | None:
    """Return the learned-router abstention reason for a row, if it abstained."""
    rules = [str(r) for r in (explanation.get("rules_fired") or [])]
    if not any(marker in rules for marker in _ABSTENTION_MARKERS):
        return None
    tags = [str(t) for t in (explanation.get("explanation_tags") or [])]
    for tag in tags:
        if tag.startswith(("no_model_artifact", "inference_error", "low_", "margin_")):
            return tag
    return tags[-1] if tags else "learned_abstained"


def _budget_achieved_summary(
    rows: list[dict[str, Any]], requested: float
) -> dict[str, Any]:
    """Distribution of the budget actually honoured, against the one requested.

    WITHHELD, NOT ZEROED, when no row carries an achieved figure. Only the prose
    codec backs its budget off; a corpus that never reaches it has nothing to
    report here, and reporting 0.0 would read as "achieved a budget of zero" —
    the most flattering possible misreading of an absent measurement.
    """
    achieved = [
        float(r["budget_achieved"])
        for r in rows
        if isinstance(r.get("budget_achieved"), (int, float))
    ]
    if not achieved:
        return {
            "requested": requested,
            "rows_reporting": 0,
            "withheld_reason": (
                "no row carried an achieved budget — no codec on this corpus "
                "backs its budget off"
            ),
        }
    ordered = sorted(achieved)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "requested": requested,
        "rows_reporting": len(ordered),
        "min": ordered[0],
        "median": median,
        "max": ordered[-1],
        # The figure a reader most needs and would never think to ask for.
        "rows_reaching_request": sum(1 for a in ordered if a <= requested + 1e-9),
    }


def _output_ratio_summary(
    rows: list[dict[str, Any]], requested: float
) -> dict[str, Any]:
    """What share of the input actually came out, per fixture that was compressed.

    A DIFFERENT QUANTITY FROM ``budget_achieved`` AND THE DISTINCTION IS THE
    WHOLE POINT. The budget is a SENTENCE-COUNT target the selector honours;
    this is the TOKEN share of the document that survived. They diverge because
    sentences are not equal in length and because locked spans, numerics and the
    entity top-up are kept whole irrespective of the target.

    Measured on long_context_v1: requesting 0.30 achieves a median token ratio
    of 0.343 and NOT ONE of the 32 compressed fixtures reaches the request;
    requesting 0.05 achieves a median of 0.236. When a reader hears "budget
    0.05" they mean this number, not the other one, so a report that offers only
    the request invites a conclusion four times better than the truth.

    Scoped to fixtures whose route ATTEMPTED compression: including the
    passthrough routes would mix in ratios of 1.0 that describe a decision not
    to compress, which is a different fact and belongs in the coverage figures.
    """
    compressing = [
        r
        for r in rows
        if r.get("attempted_route") in _COMPRESSING_ROUTES
        and isinstance(r.get("input_tokens"), int)
        and isinstance(r.get("output_tokens"), int)
        and r["input_tokens"]
    ]
    if not compressing:
        return {
            "requested": requested,
            "rows_reporting": 0,
            "withheld_reason": "no fixture attempted a compressing route",
        }
    ordered = sorted(r["output_tokens"] / r["input_tokens"] for r in compressing)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "requested": requested,
        "rows_reporting": len(ordered),
        "min": ordered[0],
        "median": median,
        "max": ordered[-1],
        "rows_reaching_request": sum(1 for a in ordered if a <= requested + 1e-9),
    }


def run_compressor_benchmark(
    fixture_dir: Path,
    passthrough_token_budget: int = 0,
    output_path: Path | None = None,
    allow_rules_only: bool = False,
    budget_ratio: float = _DEFAULT_BUDGET_RATIO,
    use_expected_route: bool = False,
) -> dict[str, Any]:
    """Run the compressor benchmark and return a provenance-bound report.

    Raises :class:`~tekmyra.router_provenance.RouterArtifactUnavailable`
    when the learned artifact did not load and ``allow_rules_only`` is False.
    """
    # NOTHING MATCHED IS NOT NOTHING TO DO.
    #
    # This filter silently discarded an entire corpus: long_context_v1 carries
    # content types long_prose / rag_multi_doc / agent_handoffs / issue_threads /
    # incident_reports / distractor_dense, none of which are in _ELIGIBLE_TYPES,
    # so the harness reported zero eligible and produced a clean empty run. The
    # only long-context corpus we own went unmeasured for months while every
    # published figure came from 250-character snippets.
    #
    # A filter that reports on its survivors without reporting its exclusions is
    # the same defect as a ratio without a denominator. Both are recorded here,
    # and a filter matching nothing REFUSES rather than returning an empty result
    # that reads as a completed measurement.
    all_fixtures = list(load_fixtures(fixture_dir))
    fixtures = [fx for fx in all_fixtures if _is_eligible(fx)]
    excluded: dict[str, int] = {}
    for fx in all_fixtures:
        if not _is_eligible(fx):
            key = str(fx.content_type)
            excluded[key] = excluded.get(key, 0) + 1

    if all_fixtures and not fixtures:
        raise CorpusEligibilityError(
            f"no fixture in {fixture_dir} matches the compressor's eligible "
            f"content types {sorted(_ELIGIBLE_TYPES)}. Found "
            f"{len(all_fixtures)} fixture(s) of types {sorted(excluded)}. "
            "Refusing to report an empty run as a completed measurement — this "
            "is what hid the long-context corpus."
        )
    router = HybridRouter(passthrough_token_budget=passthrough_token_budget)

    anchor_store = default_anchor_store()
    provenance = capture(router.learned_router)
    build_provenance = capture_anchor_store(anchor_store)
    if not allow_rules_only:
        require_loaded(provenance)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    # Resolved once, up front. A token figure that silently vanishes mid-run
    # leaves a table with holes that read as zeros; the reason is recorded and
    # travels with the report instead.
    tokens_measured, tokenizer_unavailable_reason = tokenizer_available()

    for fixture in fixtures:
        start = time.perf_counter()
        try:
            response = run_hybrid_pipeline(
                fixture,
                router=router,
                anchor_store=anchor_store,
                budget_ratio=budget_ratio,
                use_expected_route=use_expected_route,
            )
        except TerminalVerificationError as exc:
            # Refused output is a result, not an absence of one. Record it so the
            # denominator stays honest instead of dropping the row.
            #
            # THIS COMMENT WAS TRUE AND THE CODE DID THE OPPOSITE (Vega ruling 7).
            # The row was recorded, but corpus totals summed over `rows` alone, so
            # a refused fixture's tokens left the denominator entirely and
            # REFUSING PAID: the open build scored +6.09% against commercial's
            # +5.15% while saving fewer tokens, purely by refusing twice as much.
            # A comment asserting an invariant nobody tested is load-bearing in
            # the wrong direction — a reviewer auditing this exact question finds
            # an assurance at the exact site and moves on. Enforced now by
            # tests/test_refusal_denominator.py, cited here so the claim carries
            # its evidence.
            #
            # input_tokens is captured HERE because it cannot be recovered later:
            # the failure record had no token count, so the summation had nothing
            # to add even if it wanted to.
            #
            # The route and failed checks make the refusal ATTRIBUTABLE: without
            # them the per-route table would carry compression figures next to a
            # refusal count belonging to no row in it, and a route that refuses
            # everything it touches would look identical to one that never ran.
            failures.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "content_type": fixture.content_type,
                    "risk_tier": fixture.risk_tier.value,
                    "failure": type(exc).__name__,
                    "detail": str(exc),
                    "route": getattr(exc, "route", None)
                    and exc.route.value,
                    "failed_checks": list(getattr(exc, "failed_checks", ())),
                    # A refusal costs the caller the FULL original: nothing was
                    # emitted, so they send the untouched text. Input and output
                    # are therefore equal and the saving is exactly zero — not
                    # absent, zero. Recorded as one number used on both sides.
                    "input_tokens": (
                        count_tokens(fixture.content) if tokens_measured else None
                    ),
                    "latency_ms": (time.perf_counter() - start) * 1000,
                }
            )
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000

        explanation = response.explanations[0] if response.explanations else {}
        preservation = response.preservation_report

        # TOKENS ARE COUNTED ON THE EMITTED ARTIFACT, both sides from the same
        # segment. Measuring the codec's own view instead is how a pipeline that
        # INCREASED tokens by 7.06% reported compression_ratio 1.0000.
        segment = response.compressed_segments[0] if response.compressed_segments else None
        input_tokens = output_tokens = None
        if segment is not None and tokens_measured:
            input_tokens = count_tokens(segment.original_content)
            output_tokens = count_tokens(segment.compressed_content)

        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "content_type": fixture.content_type,
                "risk_tier": fixture.risk_tier.value,
                "route": explanation.get("route_after_override"),
                "compression_ratio": response.global_metrics.compression_ratio,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "codec_transformed": explanation.get("codec_transformed"),
                "attempted_route": explanation.get("attempted_route"),
                "fell_back": explanation.get("fell_back"),
                "fallback_failed_checks": explanation.get(
                    "fallback_failed_checks"
                ),
                # THE REQUEST AND THE OUTCOME, SIDE BY SIDE ON EVERY ROW.
                # `budget_ratio` in the report header is what was ASKED FOR; the
                # producer backs it off per fixture when locked spans saturate
                # it, and on this corpus no fixture reaches the request. The
                # floor fields say whether the entity guarantee is binding
                # preventively — the only thing that distinguishes a check that
                # never fails from a check that is not there.
                "budget_requested": explanation.get("budget_requested"),
                "budget_achieved": explanation.get("budget_achieved"),
                "budget_backoff_steps": explanation.get("budget_backoff_steps"),
                "entity_floor_demand": explanation.get("entity_floor_demand"),
                "entity_floor_granted": explanation.get("entity_floor_granted"),
                "entity_floor_starved": explanation.get("entity_floor_starved"),
                "entity_floor_starved_gross_legacy": explanation.get(
                    "entity_floor_starved_gross_legacy"
                ),
                "entity_floor_unmet": explanation.get("entity_floor_unmet"),
                "floor_cap": explanation.get("floor_cap"),
                "locked_spans_detected": preservation.locked_spans_detected,
                "locked_spans_preserved": preservation.locked_spans_preserved,
                "rules_fired": explanation.get("rules_fired", []),
                "learned_abstention_reason": _abstention_reason(explanation),
                "latency_ms": elapsed_ms,
            }
        )

    reached_rows = [r for r in rows if r["route"] == _COMPRESSOR_ROUTE.value]
    prose_rows = [r for r in rows if r["content_type"] == "prose"]
    prose_reached = [r for r in prose_rows if r["route"] == _COMPRESSOR_ROUTE.value]

    abstained_rows = [r for r in rows if r["learned_abstention_reason"] is not None]
    abstention_reasons: dict[str, int] = {}
    for r in abstained_rows:
        key = str(r["learned_abstention_reason"])
        abstention_reasons[key] = abstention_reasons.get(key, 0) + 1

    locked_detected = sum(r["locked_spans_detected"] for r in rows)
    locked_preserved = sum(r["locked_spans_preserved"] for r in rows)
    locked_preservation = (
        locked_preserved / locked_detected if locked_detected > 0 else None
    )

    def _avg(ratios: list[float]) -> float | None:
        return sum(ratios) / len(ratios) if ratios else None

    # --- Per-route distribution ---------------------------------------------
    #
    # THE MEAN IS NOT THE RESULT. A corpus aggregate answers "what does this
    # cost on average" and nothing else; it cannot say whether the saving comes
    # from every route shaving a little or two routes doing all the work while
    # the rest are flat. Those are different products, and only one of them is
    # worth selling.
    #
    # ABSTENTION IS A FIRST-CLASS ROW, not a footnote. Routing to
    # REVERSIBLE_RETRIEVAL_MODE is the pipeline declining to compress — a
    # legitimate, safe outcome, and the honest name for it is abstention. Folded
    # into a mean it reads as a route that merely compresses poorly, which
    # flatters the routes that did compress and hides how often the product
    # chose not to act.
    #
    # Token totals are summed, not averaged over per-fixture percentages: the
    # mean of ratios weights a 200-token fixture the same as a 6,000-token one,
    # and the customer pays for tokens, not for fixtures.
    route_names = sorted(
        {str(r["route"]) for r in rows if r["route"] is not None}
        | {str(f["route"]) for f in failures if f.get("route")}
    )
    by_route: dict[str, dict[str, Any]] = {}
    for name in route_names:
        route_rows = [r for r in rows if str(r["route"]) == name]
        route_failures = [f for f in failures if str(f.get("route")) == name]
        # Refusals carry their full weight here too, at zero saving. A route
        # that refuses everything it touches must not report a clean figure
        # over the handful it happened to complete.
        route_refused_tokens = (
            sum(f["input_tokens"] or 0 for f in route_failures)
            if tokens_measured
            else 0
        )
        tok_in = (
            sum(r["input_tokens"] or 0 for r in route_rows) + route_refused_tokens
            if tokens_measured
            else None
        )
        tok_out = (
            sum(r["output_tokens"] or 0 for r in route_rows) + route_refused_tokens
            if tokens_measured
            else None
        )
        # A ROUTE NAME IS NOT EVIDENCE THAT COMPRESSION HAPPENED.
        #
        # The routes in `SAFE_TERMINAL_ROUTES` share one `_passthrough` codec,
        # and the prose codec also
        # returns its input untouched on HIGH/CRITICAL tiers. So the table used
        # to invite the reading "four routes, varying compression performance"
        # when the truth on long_context_v1 was one compressing route and three
        # sharing a no-op — with two of those three showing NON-ZERO token
        # reductions that were the safety merge shortening text, credited to
        # compression because the delta landed on whatever route the fixture
        # took. Same defect as a corpus mean hiding its distribution, one level
        # down, and introduced by this very table.
        transformed = sum(1 for r in route_rows if r.get("codec_transformed"))
        by_route[name] = {
            "completed_fixtures": len(route_rows),
            "refused_fixtures": len(route_failures),
            "codec_transformed_fixtures": transformed,
            # True when the codec changed nothing on ANY fixture it saw: the
            # route did run, and it emitted its input.
            "codec_was_noop": bool(route_rows) and transformed == 0,
            # Named so it cannot be read as anything else.
            "is_abstention_route": name == RouteClass.REVERSIBLE_RETRIEVAL_MODE.value,
            "input_tokens": tok_in,
            "output_tokens": tok_out,
            "token_delta": (tok_in - tok_out) if tokens_measured else None,
            "token_reduction_pct": (
                (tok_in - tok_out) / tok_in * 100.0
                if tokens_measured and tok_in
                else None
            ),
            "avg_compression_ratio": _avg([r["compression_ratio"] for r in route_rows]),
            "learned_abstained_rows": sum(
                1 for r in route_rows if r["learned_abstention_reason"] is not None
            ),
            "locked_spans_detected": sum(r["locked_spans_detected"] for r in route_rows),
            "locked_spans_preserved": sum(r["locked_spans_preserved"] for r in route_rows),
        }

    # A DENOMINATOR THE SYSTEM CANNOT CHOOSE (Vega ruling 7).
    #
    # These summed over `rows` alone, so every refusal removed its own tokens
    # from the denominator and refusing RAISED the reported percentage. No
    # intent needed — the arithmetic did it unprompted, and it did it to both
    # builds, which is why the open core read as better at compression while
    # doing strictly less work.
    #
    # Refusals now enter with output EQUAL TO input: they contribute their full
    # weight to the denominator and exactly zero saving to the numerator. Both
    # builds then report against the same full-corpus denominator, which is the
    # only thing that makes two builds comparable at all.
    refused_tokens = (
        sum(f["input_tokens"] or 0 for f in failures) if tokens_measured else 0
    )
    corpus_tokens_in = (
        sum(r["input_tokens"] or 0 for r in rows) + refused_tokens
        if tokens_measured
        else None
    )
    corpus_tokens_out = (
        sum(r["output_tokens"] or 0 for r in rows) + refused_tokens
        if tokens_measured
        else None
    )
    abstention_rows = [
        r for r in rows if str(r["route"]) == RouteClass.REVERSIBLE_RETRIEVAL_MODE.value
    ]

    # A headline ratio is withheld — not zeroed — when the learned router
    # contributed to no row, or when nothing reached the compressor. A withheld
    # number is a refusal; 0.0 would read as a measurement.
    every_row_abstained = bool(rows) and len(abstained_rows) == len(rows)
    headline_withheld_reason: str | None = None
    if not provenance["model_loaded"]:
        headline_withheld_reason = (
            "learned router artifact did not load — run is rules-only"
        )
    elif every_row_abstained:
        headline_withheld_reason = (
            "learned router abstained on every row — run is rules-only"
        )
    elif not reached_rows:
        headline_withheld_reason = "no fixture reached the compressor route"

    headline = (
        None
        if headline_withheld_reason
        else _avg([r["compression_ratio"] for r in reached_rows])
    )

    # DERIVED FROM THE CORPUS IN HAND, NEVER RESTATED. The first version of the
    # routing caveat below hardcoded "HALF THIS CORPUS (20 of 40 fixtures)" —
    # true of long_context_v1, emitted verbatim for every corpus, and therefore
    # a false statement in every synthetic report (301 fixtures, none of them
    # labelled for an unbuilt codec). A disclosure that misdescribes the corpus
    # it travels with is worse than none: it teaches the reader to skip the
    # honest ones.
    unbuilt_labelled = sorted(
        f.fixture_id
        for f in fixtures
        if f.gold_output.route_intent in UNIMPLEMENTED_CODEC_ROUTES
    )

    report = {
        "fixture_dir": str(fixture_dir),
        "passthrough_token_budget": passthrough_token_budget,
        # A COMPRESSION FIGURE WITHOUT ITS BUDGET IS A RATE WITHOUT ITS
        # DENOMINATOR. The target the number was measured at travels with it.
        "budget_ratio": budget_ratio,
        # AND THE BUDGET REQUESTED IS NOT THE BUDGET ACHIEVED (Vega ruling 49).
        #
        # `budget_ratio` above is a REQUEST. The producer backs it off per
        # fixture when locked spans saturate it, and on long_context_v1 not one
        # fixture reaches the request at either 0.30 or 0.05 — asking for 0.05
        # achieves a median near 0.24. A reader given the request alone would
        # conclude we compressed to five percent, which is the provenance
        # defect this file has already fixed for eligibility, for refusals and
        # for gold-intent routing. It does not get to reappear under the best
        # number we have produced.
        "budget_achieved": _budget_achieved_summary(rows, budget_ratio),
        # And what a reader actually means by "compressed to X" — see
        # _output_ratio_summary for why these are two figures and not one.
        "output_ratio_achieved": _output_ratio_summary(rows, budget_ratio),
        # ROUTED BLIND, AND WHAT THAT DOES AND DOES NOT MEAN (Vega ruling 34).
        #
        # The pipeline used to hand the router fixture.gold_output.route_intent
        # — a label production never has — so every routing figure this harness
        # produced was a measurement of a router being told the answer.
        "routed_with_gold_intent": use_expected_route,
        "routing_caveat": (
            None
            if use_expected_route or not unbuilt_labelled
            else f"Routed blind. {len(unbuilt_labelled)} of {len(fixtures)} "
            "eligible fixtures in this corpus are labelled with a route_intent "
            "that is DECLARED BUT NEVER BUILT ("
            + ", ".join(sorted({
                f.gold_output.route_intent.value
                for f in fixtures
                if f.gold_output.route_intent in UNIMPLEMENTED_CODEC_ROUTES
            }))
            + "). Routing blind sends those fixtures to prose_token_compressor "
            "instead. Any improvement therefore means 'we compressed it with a "
            "different codec than the labels specify', NOT 'we compressed what "
            "we intended'. Both are true of this number and only one is what a "
            "reader will assume."
        ),
        "router_provenance": provenance,
        "build_provenance": build_provenance,
        "rules_only_run": not provenance["model_loaded"],
        "corpus_fixtures": len(all_fixtures),
        "eligible_fixtures": len(fixtures),
        "excluded_by_content_type": excluded,
        "completed_fixtures": len(rows),
        "refused_fixtures": len(failures),
        "compressor_reached": len(reached_rows),
        "headline_avg_compression_ratio_reached": headline,
        "headline_withheld_reason": headline_withheld_reason,
        "avg_compression_ratio_reached": _avg(
            [r["compression_ratio"] for r in reached_rows]
        ),
        "avg_compression_ratio_all_completed": _avg(
            [r["compression_ratio"] for r in rows]
        ),
        "by_route": by_route,
        "token_encoding": ENCODING_NAME if tokens_measured else None,
        "tokens_measured": tokens_measured,
        "tokens_withheld_reason": tokenizer_unavailable_reason,
        "corpus_input_tokens": corpus_tokens_in,
        "corpus_output_tokens": corpus_tokens_out,
        "corpus_token_reduction_pct": (
            (corpus_tokens_in - corpus_tokens_out) / corpus_tokens_in * 100.0
            if tokens_measured and corpus_tokens_in
            else None
        ),
        "abstention_rows": len(abstention_rows),
        "locked_span_preservation": locked_preservation,
        "locked_spans_detected": locked_detected,
        "locked_spans_preserved": locked_preserved,
        "learned_abstained_rows": len(abstained_rows),
        "learned_abstention_reasons": abstention_reasons,
        "prose_eligible": len(prose_rows),
        "prose_reached": len(prose_reached),
        "avg_prose_compression_ratio": _avg(
            [r["compression_ratio"] for r in prose_reached]
        ),
        "rows": rows,
        "failures": failures,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")

    return report


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "WITHHELD" if value is None else format(value, spec)


def _print_per_route(report: dict[str, Any]) -> None:
    """Print the distribution the corpus mean is an average OF.

    Deliberately printed BEFORE the headline. A reader who sees the aggregate
    first has already formed the impression the breakdown exists to correct.
    """
    by_route = report.get("by_route") or {}
    print("## Per route")
    if not report.get("tokens_measured"):
        print(f"Token columns WITHHELD — {report.get('tokens_withheld_reason')}")
    else:
        print(f"Tokens counted on the emitted artifact under {report['token_encoding']}.")
    if not by_route:
        print("No route produced a row.")
        print()
        return
    header = (
        f"| {'route':52s} | {'done':>4s} | {'refused':>7s} | {'codec':>5s} | {'tok in':>8s} "
        f"| {'tok out':>8s} | {'delta':>8s} | {'reduc%':>7s} | {'ratio':>6s} |"
    )
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for name, stats in by_route.items():
        # THE LABEL NAMES THE ROUTE'S ROLE; THE COLUMNS SAY WHAT HAPPENED, AND
        # THE TWO WERE READ AS ONE. This printed the bare word "ABSTENTION"
        # whenever the route was the DESIGNATED abstention route, with no
        # reference to whether anything abstained there. In the open build the
        # row reads done=0, refused=14 — and the word ABSTENTION sat beside the
        # 14. Our own published README then said the product "abstains on 14",
        # which is false twice over: abstentions are ZERO in both builds
        # (measured 2026-08-12 at this commit, open and commercial, one
        # variable), and those fixtures LAND here after their first route fails
        # verification rather than being routed here. A careful reader wrote
        # that sentence from this row. Correcting the sentence alone would have
        # left the instrument free to mint it again — so the label now names a
        # ROUTE rather than an outcome, and the legend below says which is
        # which.
        label = name
        if stats["is_abstention_route"]:
            label += "  (ABSTENTION ROUTE)"
        if stats["codec_was_noop"]:
            label = name + "  (NO-OP CODEC)"
        print(
            f"| {label:52s} | {stats['completed_fixtures']:4d} "
            f"| {stats['refused_fixtures']:7d} "
            f"| {stats['codec_transformed_fixtures']:5d} "
            f"| {_fmt(stats['input_tokens'], 'd') if stats['input_tokens'] is not None else 'WITHHELD':>8s} "
            f"| {_fmt(stats['output_tokens'], 'd') if stats['output_tokens'] is not None else 'WITHHELD':>8s} "
            f"| {_fmt(stats['token_delta'], '+d') if stats['token_delta'] is not None else 'WITHHELD':>8s} "
            f"| {_fmt(stats['token_reduction_pct'], '+.2f') if stats['token_reduction_pct'] is not None else 'WITHHELD':>7s} "
            f"| {_fmt(stats['avg_compression_ratio'], '.4f'):>6s} |"
        )
    if any(s["is_abstention_route"] for s in by_route.values()):
        # The legend, not the label, carries the precision — a label long
        # enough to state the distinction did not fit the column, and a table
        # that misaligns to explain itself is its own defect. (The column was
        # 38 while the worst-case label was already 46, so long route names
        # were silently ragged before this; it is 52 now, which fits.)
        print("  ABSTENTION ROUTE = the route designated for declining to "
              "compress. It says what the route is FOR,")
        print("  not what happened on it: read `done` for fixtures that "
              "abstained there and `refused` for ones that")
        print("  arrived after their first route failed verification. The two "
              "are different outcomes and different fixtures.")
    print()
    # The `codec` column is the number of fixtures whose codec actually changed
    # the text. Any non-zero token movement on a NO-OP CODEC row is the safety
    # merge, not compression, and saying so in words costs one line.
    noop_with_movement = [
        (name, stats)
        for name, stats in by_route.items()
        if stats["codec_was_noop"] and (stats["token_delta"] or 0) != 0
    ]
    for name, stats in noop_with_movement:
        print(
            f"NOTE: {name} shows {stats['token_delta']:+d} tokens "
            f"({stats['token_reduction_pct']:+.2f}%) with a codec that changed "
            f"nothing on any of its {stats['completed_fixtures']} fixture(s). "
            "That movement is the safety merge, not compression."
        )
    if noop_with_movement:
        print()
    # Restated in words. A reader scanning the table can miss that one of its
    # rows is the pipeline declining to compress rather than compressing badly.
    print(
        f"Abstentions (routed to {RouteClass.REVERSIBLE_RETRIEVAL_MODE.value}): "
        f"{report.get('abstention_rows', 0)}/{report['completed_fixtures']} completed. "
        f"Refusals (verification failed after fallback): {report['refused_fixtures']}."
    )
    if report.get("tokens_measured"):
        print(
            f"Corpus totals: {report['corpus_input_tokens']} -> "
            f"{report['corpus_output_tokens']} tokens "
            f"({_fmt(report['corpus_token_reduction_pct'], '+.2f')}% reduction), "
            f"summed over tokens rather than averaged over fixtures."
        )
    print()


def print_compressor_report(report: dict[str, Any]) -> None:
    """Print a human-readable summary of the compressor benchmark."""
    prov = report["router_provenance"]

    build = report.get("build_provenance", {})

    print("# Phase 5 Compressor Benchmark")
    print(f"Fixture dir: {report['fixture_dir']}")
    print(f"Passthrough token budget: {report['passthrough_token_budget']}")
    if build:
        # Printed beside the headline, not in an appendix: both builds produce
        # the same ratio on a corpus without anchorized drops, so a number
        # without this label cannot be attributed to either.
        print()
        print(f"## Build: {build['build'].upper()}")
        print(f"Anchor store: {build['anchor_store']} ({build['anchor_store_module']})")
        print(f"Anchor resolution available: {build['anchor_resolution_available']}")
        print(f"  {build['note']}")
    print()
    print("## Router provenance")
    print(f"Model loaded: {prov['model_loaded']}  (backend: {prov['model_backend']})")
    print(f"Model version: {prov['model_version']}")
    print(f"Artifact dir: {prov['artifact_dir']}  (exists: {prov['artifact_dir_exists']})")
    ships = prov["artifact_ships_in_repo"]
    print(f"Artifact ships in repo: {'UNKNOWN' if ships is None else ships}")
    print(f"Calibration temperature: {prov['calibration_temperature']}")
    for f in prov["artifact_files"]:
        tracked = f["tracked_in_git"]
        mark = "tracked" if tracked else ("UNKNOWN" if tracked is None else "UNTRACKED")
        print(f"  - {f['name']}  sha256={f['sha256'][:16]}…  {f['bytes']}B  [{mark}]")
    if report["rules_only_run"]:
        print("  *** RULES-ONLY RUN — the learned router did not participate ***")
    print()

    print("## Coverage")
    print(
        f"Eligible fixtures: {report['eligible_fixtures']}  "
        f"(completed: {report['completed_fixtures']}, "
        f"refused: {report['refused_fixtures']}, "
        f"compressor reached: {report['compressor_reached']})"
    )
    print(
        f"Learned-router abstentions: {report['learned_abstained_rows']}"
        f"/{report['completed_fixtures']}"
    )
    if report["learned_abstention_reasons"]:
        print(f"  reasons: {report['learned_abstention_reasons']}")
    print()

    _print_per_route(report)

    print("## Result")
    headline = report["headline_avg_compression_ratio_reached"]
    if headline is None:
        print(f"Headline compression ratio: WITHHELD — {report['headline_withheld_reason']}")
    else:
        print(
            f"Headline compression ratio: {headline:.4f} "
            f"over {report['compressor_reached']} fixtures actually reached"
        )
    print(
        f"Avg compression ratio (all completed): "
        f"{_fmt(report['avg_compression_ratio_all_completed'])}"
    )
    lsp = report["locked_span_preservation"]
    print(
        f"Locked-span preservation: {_fmt(lsp)} "
        f"({report['locked_spans_preserved']}/{report['locked_spans_detected']} spans)"
    )
    print(
        f"Prose: {report['prose_reached']}/{report['prose_eligible']} reached, "
        f"avg ratio {_fmt(report['avg_prose_compression_ratio'])}"
    )
    # THE CAVEAT PRINTS WHERE THE NUMBER PRINTS. Its own text calls the
    # distinction load-bearing; a disclosure that lives only in the JSON while
    # the number lives in the terminal was withheld from exactly the reader it
    # was written for.
    if report.get("routing_caveat"):
        print()
        print(f"Routing caveat: {report['routing_caveat']}")
    print()

    print("| fixture_id | content_type | risk_tier | route | ratio | locked_preserved |")
    print("|---|---|---|---|---|---|")
    for r in report["rows"]:
        print(
            f"| {r['fixture_id']} | {r['content_type']} | {r['risk_tier']} | "
            f"{r['route']} | {r['compression_ratio']:.3f} | "
            f"{r['locked_spans_preserved']}/{r['locked_spans_detected']} |"
        )
    for f in report["failures"]:
        print(
            f"| {f['fixture_id']} | {f['content_type']} | {f['risk_tier']} | "
            f"REFUSED | — | {f['detail']} |"
        )
