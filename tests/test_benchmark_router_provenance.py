"""The benchmark must never report a number without saying which router ran.

These lock in Vega's 2026-08-07 blocking requirement. The failure they prevent is
not a crash: it is a run that completes, prints a headline ratio produced by the
rule router alone, and is indistinguishable in the artifact from a trained-router
run. That is how a GO gets recorded against a measurement nobody can defend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tekmyra.benchmark_compressor import run_compressor_benchmark
from tekmyra.learned_router.config import LearnedRouterConfig
from tekmyra.learned_router.router import LearnedRouter
from tekmyra.resources import resource_path
from tekmyra.router_provenance import (
    RouterArtifactUnavailable,
    capture,
    require_loaded,
)

FIXTURES = Path("benchmarks/fixtures/synthetic")


def _config_without_artifact(tmp_path: Path) -> LearnedRouterConfig:
    """A config whose artifact directory is guaranteed empty — a fresh clone."""
    base = LearnedRouterConfig.from_yaml(
        resource_path("configs", "model_router_small.yaml")
    )
    return base.model_copy(update={"model_artifact_dir": tmp_path / "absent"})


def test_shipped_config_temperature_matches_the_artifact_it_describes() -> None:
    """The shipped config must load the shipped artifact.

    A placeholder temperature silently failed the config==artifact equality
    guard in SklearnRouterModel.load, so every run degraded to rules-only while
    still reporting a ratio.
    """
    router = LearnedRouter()
    provenance = capture(router)
    if not provenance["artifact_dir_exists"]:
        pytest.skip("no local router artifact to check the shipped config against")
    assert router.model_loaded, (
        "shipped config did not load the artifact present at "
        f"{provenance['artifact_dir']} — backend={provenance['model_backend']}"
    )
    assert provenance["model_version"] != "none"


def test_require_loaded_refuses_when_the_artifact_is_absent(tmp_path: Path) -> None:
    router = LearnedRouter(_config_without_artifact(tmp_path))
    provenance = capture(router)

    assert provenance["model_loaded"] is False
    assert provenance["model_version"] == "none"
    assert provenance["artifact_ships_in_repo"] is False

    with pytest.raises(RouterArtifactUnavailable):
        require_loaded(provenance)


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="synthetic fixtures not present")
def test_benchmark_refuses_to_run_without_an_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing artifact must refuse, not abstain quietly and report a ratio."""
    cfg = _config_without_artifact(tmp_path)
    monkeypatch.setattr(
        "tekmyra.hybrid_router.LearnedRouter",
        lambda _cfg=None: LearnedRouter(cfg),
    )
    with pytest.raises(RouterArtifactUnavailable):
        run_compressor_benchmark(fixture_dir=FIXTURES)


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="synthetic fixtures not present")
def test_rules_only_run_cannot_emit_a_headline_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicitly rules-only run is allowed, but may not headline a number."""
    cfg = _config_without_artifact(tmp_path)
    monkeypatch.setattr(
        "tekmyra.hybrid_router.LearnedRouter",
        lambda _cfg=None: LearnedRouter(cfg),
    )
    report = run_compressor_benchmark(fixture_dir=FIXTURES, allow_rules_only=True)

    assert report["rules_only_run"] is True
    assert report["headline_avg_compression_ratio_reached"] is None
    assert report["headline_withheld_reason"]
    assert report["router_provenance"]["model_loaded"] is False


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="synthetic fixtures not present")
def test_report_says_which_BUILD_produced_the_number() -> None:
    """Open and commercial builds produce the same ratio. Say which one ran.

    On a corpus without anchorized drops the two builds are numerically
    identical, so a report omitting this cannot be attributed to either half.
    That ambiguity is exactly what made the first number unpublishable and left
    the 2026-07-25 GO marked NOT-DEFENDED.
    """
    report = run_compressor_benchmark(fixture_dir=FIXTURES)
    build = report["build_provenance"]

    assert build["build"] in {"open-core", "commercial"}
    assert build["anchor_store"]
    assert build["anchor_store_module"]
    assert isinstance(build["anchor_resolution_available"], bool)

    # The label must follow the store that actually ran, not a guess.
    is_open = build["anchor_store"] == "NonResolvingAnchorStore"
    assert build["build"] == ("open-core" if is_open else "commercial")
    assert build["anchor_resolution_available"] is not is_open


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="synthetic fixtures not present")
def test_report_carries_router_provenance_and_honest_denominators() -> None:
    """Every number must arrive with the provenance needed to defend it."""
    report = run_compressor_benchmark(fixture_dir=FIXTURES)
    provenance = report["router_provenance"]

    for key in (
        "model_loaded",
        "model_version",
        "model_backend",
        "artifact_dir",
        "artifact_ships_in_repo",
        "artifact_files",
    ):
        assert key in provenance, f"provenance missing {key}"

    for entry in provenance["artifact_files"]:
        assert len(entry["sha256"]) == 64

    # The denominator must be stated, and every eligible fixture accounted for.
    assert (
        report["completed_fixtures"] + report["refused_fixtures"]
        == report["eligible_fixtures"]
    )
    assert report["compressor_reached"] <= report["completed_fixtures"]
    if report["headline_avg_compression_ratio_reached"] is not None:
        assert report["compressor_reached"] > 0
