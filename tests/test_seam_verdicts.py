"""The seam guard's two trees, and the verdicts a reader is given.

Written because adversarial QA found the branch UNTESTED after it shipped: an
inversion of a safety guard that nothing exercised, so a future refactor could
have silently reopened it and nothing would have failed red.

Two things are pinned here:

* `validate_exclusions` answers the SAME measurement oppositely in the source
  repository and in a published payload, and both directions must hold. In the
  source repo a named exemption matching no file protects nothing and must
  raise. In a published tree the exclusions have already been applied, so every
  entry SHOULD match nothing — and a module that IS present is a leak.
* `RouterArtifactUnavailable` carries "missing" and "failed to load" apart as
  data. The verdict that reassures a reader ("expected on a fresh clone") may
  only be reached on positive evidence that the artifact is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import open_surface  # noqa: E402
from tekmyra.router_provenance import (  # noqa: E402
    RouterArtifactUnavailable, require_loaded)


class TestTheGuardAnswersEachTreeInItsOwnDirection:
    def test_a_published_tree_refuses_a_withheld_module_that_shipped(
            self, tmp_path, monkeypatch):
        """The inversion. A payload has no `open_core_files/`, so every excluded
        path SHOULD be absent; one that is present is a leak, and the crashing
        version this replaced could never have detected it — it aborted on its
        own success before reaching any such check."""
        leaked = next(iter(open_surface.CATEGORICAL_EXCLUSIONS))
        (tmp_path / leaked).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / leaked).write_text("# withheld module that shipped anyway\n")
        monkeypatch.setattr(open_surface, "REPO_ROOT", tmp_path)
        with pytest.raises(open_surface.ExclusionListError, match="PRESENT"):
            open_surface.validate_exclusions()

    def test_a_clean_published_tree_passes_where_it_used_to_crash(
            self, tmp_path, monkeypatch):
        """The bug this fixed: a reader following the README's one recommended
        command got a traceback naming all 25 withheld modules, before anything
        was proven."""
        monkeypatch.setattr(open_surface, "REPO_ROOT", tmp_path)
        open_surface.validate_exclusions()

    def test_the_discriminator_is_the_staging_input_directory(self, tmp_path,
                                                              monkeypatch):
        """Recreating `open_core_files/` flips the tree back to source-repo
        rules. Pinned so the discriminator is a decision someone has to change
        deliberately, not an accident of what happens to be on disk."""
        (tmp_path / open_surface.SOURCE_ONLY_MARKER).mkdir()
        monkeypatch.setattr(open_surface, "REPO_ROOT", tmp_path)
        with pytest.raises(open_surface.ExclusionListError, match="do not exist"):
            open_surface.validate_exclusions()


class TestMissingAndUnusableAreDifferentAnswers:
    @staticmethod
    def _raise(**provenance):
        with pytest.raises(RouterArtifactUnavailable) as caught:
            require_loaded({"model_loaded": False, **provenance})
        return caught.value

    def test_an_absent_artifact_says_so_as_data_not_only_in_prose(self):
        exc = self._raise(artifact_dir="/x/y", artifact_dir_exists=False)
        assert exc.artifact_dir_exists is False
        assert exc.artifact_dir == "/x/y"

    def test_a_present_but_unloadable_artifact_is_distinguishable(self):
        """Same exception, opposite situation, different remedy — download it,
        versus work out why the one you have will not open."""
        exc = self._raise(artifact_dir="/x/y", artifact_dir_exists=True)
        assert exc.artifact_dir_exists is True

    def test_an_undetermined_state_is_neither(self):
        """None must never be read as False: the reassuring verdict requires
        positive evidence, and an unknown falls through to the generic failure
        rather than inheriting the comfortable reading."""
        exc = self._raise()
        assert exc.artifact_dir_exists is None

    def test_a_class_that_merely_shares_the_name_is_not_this_one(self):
        """The defect the type check replaced: matching on `type(exc).__name__`
        accepted any impostor, and a corrupt-fixture error was reported to a
        reader as a benign missing model."""
        impostor = type("RouterArtifactUnavailable", (Exception,), {})()
        assert type(impostor).__name__ == RouterArtifactUnavailable.__name__
        assert not isinstance(impostor, RouterArtifactUnavailable)

    def test_the_message_still_names_every_path_it_tried(self):
        """The prose contract is unchanged by carrying the same facts as data."""
        exc = self._raise(artifact_dir="/x/y", artifact_dir_exists=False,
                          artifact_search="  absent  /x/y")
        assert "/x/y" in str(exc)
        assert "--allow-rules-only" in str(exc)


class TestTheChildCatchesByTypeNotByName:
    """Adversarial QA reverted the child's except-clause to its pre-fix
    string-matching form — `type(exc).__name__ == "RouterArtifactUnavailable"`,
    the defect the whole exit-code-6 path exists to prevent — and the ENTIRE
    seam suite stayed green. Nothing exercised the vulnerable call site; the
    tests above cover the exception class's semantics one level removed from it.

    This reads the child SOURCE rather than executing it, and says so. The
    child runs as a subprocess inside a freshly built isolation tree, so an
    execution test costs minutes and a model artifact this repository
    deliberately does not carry. A source assertion is the weaker instrument,
    but it catches the exact mutation QA performed, which the strong-looking
    tests did not.
    """

    @staticmethod
    def _child_source() -> str:
        script = Path(__file__).resolve().parent.parent / "scripts" / "verify_open_core.py"
        return script.read_text(encoding="utf-8")

    def test_the_child_does_not_compare_an_exception_class_name(self):
        source = self._child_source()
        assert 'type(exc).__name__ == "RouterArtifactUnavailable"' not in source
        assert "__name__ !=" not in source

    def test_the_child_catches_the_imported_class(self):
        source = self._child_source()
        assert "except _ArtifactUnavailable as exc:" in source
        assert "RouterArtifactUnavailable as _ArtifactUnavailable" in source

    def test_the_reassuring_verdict_requires_positive_evidence(self):
        """Exit 6 says "expected on a fresh clone". It may be reached only where
        the artifact directory is KNOWN ABSENT — `is False`, never a truthiness
        test, because None means nobody determined it."""
        source = self._child_source()
        assert "if exists is False:" in source
        assert "if exists is True:" in source
