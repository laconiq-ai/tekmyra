"""THE WRONG-DIRECTORY FOOTGUN, AND THE STRUCTURE THAT KEEPS IT DEAD.

The defect: both artifact directories were bare relative paths resolved against
the WORKING DIRECTORY. Run the package from anywhere but the tree root and the
compressor loader found nothing, returned its no-op, and the benchmark printed a
compression ratio for a compressor that compresses nothing. Measured before the
fix: 82.9711% reduction from the tree root, 3083 bytes in and 3083 out from
``/tmp``, no error either way.

**NO TEST HERE NEEDS A TRAINED MODEL.** These run on a fresh clone, which is the
whole point — a test suite that goes red on first clone because an artifact is
absent is the same footgun class wearing a different hat.

TWO KINDS OF TEST, DELIBERATELY. The behavioural ones drive the resolver over
constructed directories, including the case where it must FAIL — a resolver that
always finds something would pass every success assertion and be worthless. The
structural ones assert that the CONSUMERS route through the one shared
resolution, because the behavioural tests cannot see a fourth call site
re-deriving the path itself, and that is exactly how this defect got in.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from tekmyra import artifact_paths

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- BEHAVIOUR -------------------------------------------------------------


def test_the_working_directory_is_still_tried_first(tmp_path, monkeypatch):
    """Nothing that works today changes. Cwd stays ahead of the package root."""
    monkeypatch.delenv(artifact_paths.ARTIFACT_ROOT_ENV, raising=False)
    (tmp_path / ".cache" / "compressor" / "1.0.0").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    result = artifact_paths.resolve(".cache/compressor/1.0.0")

    assert result.found
    assert result.path == tmp_path / ".cache" / "compressor" / "1.0.0"
    assert result.candidates[0][0] == result.path


def test_the_package_root_is_found_from_the_WRONG_directory(tmp_path, monkeypatch):
    """THE FIX. Cwd has no artifact; the one beside the package is found."""
    monkeypatch.delenv(artifact_paths.ARTIFACT_ROOT_ENV, raising=False)
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    relative = "constructed-artifacts/v1"
    (artifact_paths.PACKAGE_ROOT / relative).mkdir(parents=True, exist_ok=True)
    try:
        result = artifact_paths.resolve(relative)
        assert result.found, result.describe()
        assert result.path == artifact_paths.PACKAGE_ROOT / relative
        # And the cwd miss is RECORDED, not skipped — the search has to be
        # reportable or the diagnostic downstream has nothing to say.
        assert result.candidates[0] == (elsewhere / relative, False)
    finally:
        (artifact_paths.PACKAGE_ROOT / relative).rmdir()
        (artifact_paths.PACKAGE_ROOT / "constructed-artifacts").rmdir()


def test_it_still_FAILS_when_nothing_is_there(tmp_path, monkeypatch):
    """NON-VACUITY. A resolver that always finds something proves nothing.

    Every success assertion above would pass against a resolver that returned
    the first candidate unconditionally. This is the test that separates them.
    """
    monkeypatch.delenv(artifact_paths.ARTIFACT_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    result = artifact_paths.resolve("no/such/artifact/anywhere")

    assert not result.found
    assert result.path is None
    assert all(exists is False for _, exists in result.candidates)
    assert len(result.candidates) >= 2, "the package-root candidate was not tried"


def test_the_env_override_is_tried_before_the_defaults(tmp_path, monkeypatch):
    """The documented escape hatch for a clone that keeps models elsewhere."""
    root = tmp_path / "models"
    (root / "a/b").mkdir(parents=True)
    (tmp_path / "a/b").mkdir(parents=True)  # a cwd candidate that also exists
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(artifact_paths.ARTIFACT_ROOT_ENV, str(root))

    result = artifact_paths.resolve("a/b")

    assert result.path == root / "a/b", "the override did not win"


def test_an_explicit_directory_wins_over_everything(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    monkeypatch.setenv(artifact_paths.ARTIFACT_ROOT_ENV, str(tmp_path))
    monkeypatch.chdir(tmp_path)

    result = artifact_paths.resolve("anything", explicit=explicit)

    assert result.path == explicit
    assert result.candidates[0][0] == explicit


def test_an_explicit_MISS_is_never_rescued_by_a_fallback(tmp_path, monkeypatch):
    """THE DEFECT THE FIX ITSELF INTRODUCED, AND THE TEST THAT KILLED IT.

    An earlier draft appended the default candidates after ``explicit``. Asking
    for a directory that is empty then loaded a DIFFERENT model from the package
    root and answered to the name of the one requested. Four existing tests went
    red — one asking for ``/nonexistent`` and expecting the no-op, one exporting
    an ONNX router to a tmp dir and receiving ``SklearnRouterModel`` from the
    repository's own artifacts. Both looked exactly like stale tests.

    Losing a model is a bug. Substituting one is worse, and it is quieter.
    """
    monkeypatch.delenv(artifact_paths.ARTIFACT_ROOT_ENV, raising=False)
    # A real artifact exists at the package root; the explicit ask must not see it.
    (tmp_path / ".cache" / "compressor" / "1.0.0").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    result = artifact_paths.resolve(
        ".cache/compressor/1.0.0", explicit=tmp_path / "definitely-not-here"
    )

    assert not result.found
    assert result.path is None
    assert len(result.candidates) == 1, (
        "an explicit ask searched somewhere else too: " + result.describe()
    )


def test_an_absolute_path_is_not_listed_three_times(tmp_path, monkeypatch):
    """An already-absolute relative collapses every candidate onto one path.

    Without deduplication the search prints the same directory three times,
    which reads as three checks having been made.
    """
    monkeypatch.delenv(artifact_paths.ARTIFACT_ROOT_ENV, raising=False)
    target = tmp_path / "abs"
    target.mkdir()

    result = artifact_paths.resolve(target)

    assert result.found
    assert len(result.candidates) == 1, result.describe()


def test_describe_names_every_location_and_says_which_existed(tmp_path, monkeypatch):
    """The refusal downstream is only as good as this string."""
    monkeypatch.delenv(artifact_paths.ARTIFACT_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    text = artifact_paths.resolve("missing/dir").describe()

    assert "could not resolve" in text
    assert "absent" in text
    assert str(tmp_path) in text
    assert str(artifact_paths.PACKAGE_ROOT) in text


def test_the_fallback_path_is_one_that_was_actually_tried(tmp_path, monkeypatch):
    """An error naming a directory the code never opened is the class of lie
    this whole module exists to remove."""
    monkeypatch.delenv(artifact_paths.ARTIFACT_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    result = artifact_paths.resolve("missing/dir")

    assert result.fallback() in [path for path, _ in result.candidates]


# --- STRUCTURE -------------------------------------------------------------
#
# These exist because the behavioural tests above cannot see a NEW call site
# that re-derives the path for itself, and that is precisely how the defect
# arrived: three sites each computed `config.model_artifact_dir / config.version`
# independently. Three copies of a path rule diverge silently, and they diverge
# on the path deciding whether a published number describes a model or a no-op.


def _sources() -> dict[str, str]:
    package = REPO_ROOT / "tekmyra"
    return {
        name: (package / name).read_text(encoding="utf-8")
        for name in (
            "router_provenance.py",
            "learned_router/model.py",
            "learned_router/config.py",
            "learned_compressor/compressor.py",
            "benchmark.py",
        )
    }


def test_no_consumer_re_derives_the_router_artifact_path() -> None:
    """Exactly one place composes the router's versioned directory."""
    offenders = [
        name
        for name, text in _sources().items()
        if "model_artifact_dir / config.version" in text
        or "model_artifact_dir) / config.version" in text
    ]
    # config.py names the pattern in prose, explaining why it is centralised.
    offenders = [n for n in offenders if n != "learned_router/config.py"]
    assert not offenders, (
        f"{offenders} compose the artifact path themselves instead of calling "
        "LearnedRouterConfig.resolved_artifact_dir() — the divergence this fix removed"
    )


def test_the_artifact_constants_are_relative_and_resolved_elsewhere() -> None:
    """A constant that is already absolute at import time bakes in whatever
    directory the interpreter happened to start in."""
    from tekmyra.learned_compressor import compressor

    assert not compressor.ARTIFACT_RELATIVE_DIR.is_absolute()
    assert compressor._ARTIFACT_DIR == compressor.ARTIFACT_RELATIVE_DIR


def test_the_provenance_block_reports_the_search_it_did_not_invent() -> None:
    """`benchmark.py` used to resolve the relative constant against its OWN cwd,
    so it could report on a directory the loader never opened."""
    text = _sources()["benchmark.py"]
    assert "last_artifact_resolution" in text
    assert "artifact_search" in text
    tree = ast.parse(text)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "_ARTIFACT_DIR" not in imported, (
        "benchmark.py imports the relative constant again; it will resolve it "
        "against its own working directory and disagree with the loader"
    )


def test_the_router_refusal_carries_the_search() -> None:
    """A refusal naming one directory sends the reader to check a path that may
    not be the one that was searched."""
    text = _sources()["router_provenance.py"]
    assert "artifact_search" in text
    assert artifact_paths.ARTIFACT_ROOT_ENV in text or "ARTIFACT_ROOT_ENV" in text


def test_the_compressor_default_fixture_dir_ships_with_the_package() -> None:
    """The `compressor` command's default pointed at a training corpus that is
    not published, so the first command a new cloner types failed on a missing
    path."""
    text = _sources()["benchmark.py"]
    assert "learned_train" not in text.split("def compressor")[0].split("@cli.command()")[-1], (
        "the compressor command still defaults to an unshipped corpus"
    )


@pytest.mark.parametrize(
    "corpus", ["benchmarks/fixtures/synthetic", "benchmarks/fixtures/long_context_v1"]
)
def test_the_default_corpus_actually_exists_in_this_tree(corpus: str) -> None:
    """Pins the default against reality rather than against a string."""
    assert (REPO_ROOT / corpus).is_dir(), f"{corpus} is not in this tree"
