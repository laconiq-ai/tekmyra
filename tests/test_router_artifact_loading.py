"""Loading a router artifact must be self-describing and loud when it fails.

The regression these lock in (2026-08-07): the shipped config carried a
placeholder ``temperature: 1.0`` while the artifact carried its fitted
0.19764397711114967. ``SklearnRouterModel.load`` asserted config == artifact,
raised, and ``ModelManager.load`` swallowed that into a warning and returned
None — so the router abstained, the run went rules-only, and the benchmark still
printed a headline ratio. One field, silently, for every run.

Fixing the VALUE would have left the mechanism intact and the next retrain would
reproduce it. So: fitted parameters are derived FROM the artifact, and an
artifact that is present but unloadable fails loudly.
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import pytest

from tekmyra.learned_router.config import LearnedRouterConfig
from tekmyra.learned_router.model import (
    ModelManager,
    RouterArtifactLoadError,
)
from tekmyra.learned_router.router import LearnedRouter
from tekmyra.resources import resource_path

LIVE_ARTIFACT = Path(".cache/router/1.0.0")


def _base_config() -> LearnedRouterConfig:
    return LearnedRouterConfig.from_yaml(
        resource_path("configs", "model_router_small.yaml")
    )


def _staged_artifact(tmp_path: Path) -> Path:
    """Copy the live sklearn artifact under a temp mount root."""
    dst = tmp_path / "1.0.0"
    dst.mkdir(parents=True)
    shutil.copy(LIVE_ARTIFACT / "router.joblib", dst / "router.joblib")
    return dst


def test_shipped_config_does_not_restate_a_fitted_value() -> None:
    """The YAML must not carry a hand-copied calibration value.

    A comment saying "re-derive this on retrain" is an unenforceable manual
    obligation; this asserts the mechanism replaced it rather than the value
    being corrected in place.
    """
    config = _base_config()
    assert config.temperature == 1.0, (
        "the shipped config should carry the neutral default; calibration is "
        "derived from the artifact at load time"
    )


@pytest.mark.skipif(not LIVE_ARTIFACT.is_dir(), reason="no local router artifact")
def test_calibration_is_derived_from_the_artifact_not_the_config() -> None:
    """The exact regression: a neutral config must still load the model."""
    base = _base_config()
    resolved = ModelManager.resolve_config(base)

    assert resolved.temperature != base.temperature, (
        "resolve_config did not derive calibration from the artifact"
    )

    router = LearnedRouter()
    assert router.model_loaded
    assert router.model_version == "1.0.0"
    assert router.config.temperature == resolved.temperature


@pytest.mark.skipif(not LIVE_ARTIFACT.is_dir(), reason="no local router artifact")
def test_derivation_works_without_a_config_block_in_metadata(tmp_path: Path) -> None:
    """ONNX-exported artifacts write metadata.json WITHOUT a config key.

    That is precisely the artifact shape that regressed, so the fallback to the
    sklearn artifact's embedded config is the path that matters.
    """
    staged = _staged_artifact(tmp_path)
    (staged / "metadata.json").write_text('{"version": "1.0.0", "kind": "onnx_router"}')

    config = _base_config().model_copy(update={"model_artifact_dir": tmp_path})
    resolved = ModelManager.resolve_config(config)

    with open(staged / "router.joblib", "rb") as fh:
        embedded = LearnedRouterConfig.model_validate(pickle.load(fh)["config"])

    assert resolved.temperature == embedded.temperature
    assert ModelManager(resolved).load() is not None


@pytest.mark.skipif(not LIVE_ARTIFACT.is_dir(), reason="no local router artifact")
def test_present_but_incompatible_artifact_fails_loudly(tmp_path: Path) -> None:
    """A structural mismatch must raise and name the fields, not warn."""
    _staged_artifact(tmp_path)
    bad = _base_config().model_copy(
        update={
            "model_artifact_dir": tmp_path,
            "risk_classes": ["low", "medium", "high"],
        }
    )
    with pytest.raises(RouterArtifactLoadError) as excinfo:
        ModelManager(ModelManager.resolve_config(bad)).load()

    message = str(excinfo.value)
    assert "risk_classes" in message, "the error must name the offending field"
    assert "artifact=" in message and "runtime=" in message, (
        "the error must show both values so it is actionable"
    )


def test_absent_artifact_still_abstains_rather_than_raising(tmp_path: Path) -> None:
    """Absence is a legitimate deployment state; only presence-with-failure is loud."""
    config = _base_config().model_copy(update={"model_artifact_dir": tmp_path / "none"})
    assert ModelManager(ModelManager.resolve_config(config)).load() is None

    router = LearnedRouter(config)
    assert router.model_loaded is False
    decision = router.route("probe", "some prose text", [0.0])
    assert decision.abstained
    assert decision.abstention_reason == "no_model_artifact"
