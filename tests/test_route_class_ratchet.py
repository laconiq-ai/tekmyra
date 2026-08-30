"""Every declared route must be built, or explicitly baselined as unbuilt.

Vega's ruling, 2026-08-07: the defect behind the prose terminal failure was not
the routing table entry. It was that **a route can be declared without being
built and nothing fails** — so the next declared-but-unbuilt route repeats it.

Two guards here:

1. The set of unbuilt routes matches the baseline exactly and may only SHRINK.
2. No rule falls back to an unbuilt route. That is the concrete bug that shipped:
   prose fell back to SUMMARIZER, which fails closed by design, so a prose
   verification failure had no recoverable path and went terminal.

The unbuilt set is derived BEHAVIOURALLY (routes are probed through apply_codec)
rather than read from the codec map, so a route that stops failing closed is
detected however the map happens to be spelled. The routing table is read by
AST rather than regex, matching the seal-sheet guard's precedent in this repo —
a regex over multi-line tuples is how that count was first got wrong.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tekmyra.codecs.selector import apply_codec
from tekmyra.contracts import FeatureVector, RouteClass
from tekmyra.prober import probe_text
from tekmyra.contracts import PolicyTier, SourceChannel

BASELINE = Path(__file__).parent / "baselines" / "unbuilt_routes.json"
RULE_ROUTER = Path(__file__).parent.parent / "tekmyra" / "rule_router.py"


def _baseline() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def _features() -> FeatureVector:
    return probe_text(
        "A short paragraph of ordinary prose used to probe every codec.",
        SourceChannel.API,
        PolicyTier.STANDARD,
    )


def _observed_unbuilt() -> set[str]:
    """Routes whose codec reports no implementation, probed through apply_codec."""
    features = _features()
    unbuilt: set[str] = set()
    for route in RouteClass:
        result = apply_codec("Probe text for codec availability.", route, features)
        if not result.success and (result.error or "").startswith("no_codec_defined_for_"):
            unbuilt.add(route.value)
    return unbuilt


def _rule_table() -> list[tuple[str, str, str]]:
    """Extract (primary_route, rule_name, fallback_route) triples by AST.

    Every rule in ``_classify_by_content`` is expressed as a 3-tuple of
    (RouteClass, "R_NAME", RouteClass), whether returned directly or held in the
    semantic_routes mapping. Both shapes are collected here.
    """
    tree = ast.parse(RULE_ROUTER.read_text(encoding="utf-8"))
    triples: list[tuple[str, str, str]] = []

    def route_name(node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "RouteClass"
        ):
            return node.attr
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Tuple) or len(node.elts) != 3:
            continue
        primary, name, fallback = node.elts
        if not isinstance(name, ast.Constant) or not isinstance(name.value, str):
            continue
        if not name.value.startswith("R_"):
            continue
        p, f = route_name(primary), route_name(fallback)
        if p and f:
            triples.append((p, name.value, f))
    return triples


def _to_value(member_name: str) -> str:
    return getattr(RouteClass, member_name).value


def test_every_declared_route_is_built_or_baselined() -> None:
    observed = _observed_unbuilt()
    baseline = set(_baseline()["unbuilt_routes"])

    newly_unbuilt = observed - baseline
    assert not newly_unbuilt, (
        f"route(s) declared but not built and not baselined: {sorted(newly_unbuilt)}. "
        "Build a codec, or add them to tests/baselines/unbuilt_routes.json and raise "
        "max_unbuilt_routes deliberately."
    )

    now_built = baseline - observed
    assert not now_built, (
        f"route(s) baselined as unbuilt now have a real codec: {sorted(now_built)}. "
        "Remove them from tests/baselines/unbuilt_routes.json — the list may only shrink."
    )

    # Nothing may be unaccounted for in either direction.
    all_routes = {r.value for r in RouteClass}
    assert observed <= all_routes


def test_the_unbuilt_list_may_only_shrink() -> None:
    data = _baseline()
    recorded = data["unbuilt_routes"]
    assert len(recorded) <= data["max_unbuilt_routes"], (
        "more unbuilt routes than the recorded ceiling — this number may only "
        "be decreased, never raised to absorb a new one silently"
    )
    assert len(recorded) == len(set(recorded)), "duplicate entries in the baseline"
    assert recorded == sorted(recorded), "keep the baseline sorted for reviewable diffs"


def test_no_rule_falls_back_to_an_unbuilt_route() -> None:
    """The concrete bug: a fallback that structurally cannot succeed.

    A fallback exists to rescue a failed primary. Pointing it at a route that
    fails closed by design means there is no recoverable path at all, and the
    pipeline dies instead of degrading.
    """
    unbuilt = _observed_unbuilt()
    triples = _rule_table()
    assert triples, "AST extraction found no rules — the guard would be vacuous"

    offenders = [
        (rule, _to_value(fallback))
        for _primary, rule, fallback in triples
        if _to_value(fallback) in unbuilt
    ]
    assert not offenders, (
        "rule(s) fall back to a declared-but-unbuilt route, so the fallback can "
        f"never succeed: {offenders}"
    )


def test_rules_with_an_unbuilt_primary_match_the_recorded_set() -> None:
    """Safe today (their fallbacks are built) but recorded so a NEW one fails."""
    unbuilt = _observed_unbuilt()
    recorded = set(_baseline()["rules_whose_primary_route_is_unbuilt"]) - {"_comment"}

    observed = {
        rule
        for primary, rule, _fallback in _rule_table()
        if _to_value(primary) in unbuilt
    }

    assert observed == recorded, (
        "rules routing PRIMARY traffic to an unbuilt codec changed.\n"
        f"  new (treat as a defect): {sorted(observed - recorded)}\n"
        f"  resolved (remove from baseline): {sorted(recorded - observed)}"
    )


@pytest.mark.parametrize("route_name", sorted(_baseline()["unbuilt_routes"]))
def test_baselined_routes_fail_closed_rather_than_passthrough(route_name: str) -> None:
    """An unbuilt route must refuse, never return input as a successful compression.

    Mapping these to passthrough would convert a fail-closed control into a
    silent success — the defect class this whole effort exists to remove.
    """
    route = RouteClass(route_name)
    result = apply_codec("Some content.", route, _features())
    assert result.success is False
    assert (result.error or "").startswith("no_codec_defined_for_")
