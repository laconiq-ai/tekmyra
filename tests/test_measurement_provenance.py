"""Tests for measurement provenance.

The tests that matter are the REFUSALS. A provenance module that records
everything and blocks nothing would let an unreproducible figure reach
publication with a manifest attached, which is worse than no manifest -- it
looks like rigour.
"""

from __future__ import annotations

import json

import pytest

from tests import measurement_provenance as mp


def _manifest(**overrides) -> mp.MeasurementManifest:
    base = {
        "commit": "a" * 40,
        "tree_dirty": False,
        "artifacts": {"benchmarks/x.pkl": "b" * 64},
        "corpora": {"benchmarks/fixtures/synthetic": {"digest": "c" * 64, "files": 10}},
        "libraries": {"python": "3.12.13", "numpy": "2.5.1", "sklearn": "1.9.0"},
        "invocation": ["pytest"],
        "measured_corpus": "benchmarks/fixtures/long_context_v1",
        "budget": 0.30,
        "basis": "attempted",
        "runtime_artifacts": {
            ".cache/compressor/1.0.0": {
                "present": True, "digest": "e" * 64, "files": 8, "tracked_in_git": False,
            },
        },
    }
    base.update(overrides)
    return mp.MeasurementManifest(**base)


# --- the three fields whose absence caused the dispute -------------------

@pytest.mark.parametrize("missing", ["measured_corpus", "budget", "basis"])
def test_a_figure_without_corpus_budget_or_basis_is_not_quotable(missing: str) -> None:
    """These are not decoration.

    63.15% is long_context_v1; 55.24% is the Congressional Record corpus. The
    operating point is a budget: 0.30/51.38, 0.15/58.99, 0.05/61.15. And on the
    Congressional Record corpus, attempted basis reads 55.24% while survivor
    basis reads 60.39% — basis alone moves the number 5.15 points, MORE than the
    entire disputed gap. A figure that does not name all three cannot be
    distinguished from a figure that disagrees.
    """
    with pytest.raises(mp.NotQuotable, match=missing):
        mp.require_quotable(_manifest(**{missing: mp.UNKNOWN}))


def test_collect_defaults_those_three_to_unknown_not_to_a_plausible_value() -> None:
    """A default of 0.30 would let a caller that never considered the budget
    emit a manifest asserting one — provenance the figure has not earned."""
    m = mp.collect()
    assert m.measured_corpus == mp.UNKNOWN
    assert m.budget == mp.UNKNOWN
    assert m.basis == mp.UNKNOWN
    assert m.is_quotable() is False


# --- the untracked runtime artifacts: the false-assurance fix -------------

def test_an_absent_runtime_artifact_blocks_quotation() -> None:
    """The correction that matters most.

    An earlier revision digested only the two TRACKED .pkl files, found them
    byte-identical to HEAD, and stamped figures QUOTABLE — while the artifact
    serving every disputed prose figure lives in gitignored .cache/ and was
    never looked at. Absence is not a neutral fact about the machine: with the
    prose artifact missing, load_compressor returns a no-op and the codec reads
    as INERT rather than broken.
    """
    m = _manifest(runtime_artifacts={
        ".cache/compressor/1.0.0": {
            "present": False, "digest": mp.UNKNOWN, "files": 0, "tracked_in_git": False,
        },
    })
    assert m.is_quotable() is False
    with pytest.raises(mp.NotQuotable, match="ABSENT"):
        mp.require_quotable(m)


def test_the_absence_message_forbids_baselining_not_just_quoting() -> None:
    """Baselining an absent artifact records it as a property of the codec —
    which is the specific mistake codec_effectiveness.py's docstring warns of."""
    m = _manifest(runtime_artifacts={
        ".cache/router": {"present": False, "digest": mp.UNKNOWN, "files": 0,
                          "tracked_in_git": False},
    })
    with pytest.raises(mp.NotQuotable, match="baseline"):
        mp.require_quotable(m)


def test_collect_reports_every_runtime_artifact_dir_consistently() -> None:
    """Machine-INDEPENDENT by design, and the reason is the finding itself.

    An earlier version of this test asserted the artifacts are PRESENT. It
    passed in the main checkout and failed in a git worktree of the SAME
    COMMIT — because ``.cache/`` is gitignored, so a fresh checkout does not
    receive it. That is not a flaky test; it is the defect reproducing on
    demand, and asserting presence would have hard-coded one machine's state
    into the suite.

    What must hold everywhere is CONSISTENCY: presence and digest agree, and
    the manifest never claims a digest for something it does not have.
    """
    m = mp.collect()
    assert set(m.runtime_artifacts) == set(mp.RUNTIME_ARTIFACT_DIRS)
    for rel, info in m.runtime_artifacts.items():
        if info["present"]:
            assert info["digest"] != mp.UNKNOWN, f"{rel} present but undigested"
            assert int(info["files"]) > 0, f"{rel} present but digested zero files"
        else:
            assert info["digest"] == mp.UNKNOWN, f"{rel} absent but claims a digest"
            assert int(info["files"]) == 0


def test_a_checkout_without_the_artifacts_is_refused_rather_than_measured() -> None:
    """The clean-checkout case, asserted wherever it happens to apply.

    In a checkout that lacks the gitignored artifacts, no figure may be
    quoted — that run measures a no-op codec, not a worse one.
    """
    m = mp.collect()
    missing = [r for r, i in m.runtime_artifacts.items() if not i["present"]]
    if not missing:
        pytest.skip("this checkout has the artifacts; the refusal path is covered "
                    "by test_an_absent_runtime_artifact_blocks_quotation")
    assert m.is_quotable() is False
    # Assert the CONDITION, not which refusal fires first. A dirty tree is a
    # more fundamental disqualification and legitimately preempts this one, so
    # pinning the message would make the test depend on whether the checkout
    # happened to be clean when it ran.
    assert any(u.startswith("runtime_artifact_ABSENT:") for u in m.unknowns())
    with pytest.raises(mp.NotQuotable):
        mp.require_quotable(m)


def test_runtime_artifacts_are_recorded_as_untracked_when_present() -> None:
    """Recorded explicitly, and expected False where the directory exists.

    The point is not that they SHOULD be tracked — it is that a figure measured
    against a directory git cannot see is reproducible only by someone holding
    that exact directory, so its digest must travel with the number.
    """
    m = mp.collect()
    for rel, info in m.runtime_artifacts.items():
        if info["present"]:
            assert info["tracked_in_git"] is False, f"{rel} is tracked — update the comment"


def test_a_present_but_untracked_artifact_is_still_quotable_when_digested() -> None:
    """Untracked is not disqualifying; UNRECORDED is. The digest is what lets a
    second machine prove it measured the same thing."""
    assert _manifest().is_quotable() is True


def test_collect_records_them_when_the_caller_states_them() -> None:
    m = mp.collect(
        measured_corpus="benchmarks/fixtures/public_domain_v1",
        budget=0.15,
        basis="survivor",
    )
    assert m.measured_corpus.endswith("public_domain_v1")
    assert m.budget == 0.15
    assert m.basis == "survivor"


# --- the real repository -------------------------------------------------

# THE LEARNED BENCHMARK ARTIFACTS ARE DELIBERATELY UNPUBLISHED (README: "What
# you cannot do, stated rather than left to be discovered") and the reference
# bundle does not carry them either — they exist only in the measuring tree.
# On a clone this test must SKIP with that stated, never fail (which blames
# the cloner for an absence we chose) and never silently pass (which would
# claim a pin over files that were not there). Every other test in this module
# still runs: absence handling is their subject, not their precondition.
_missing_learned_artifacts = tuple(
    rel for rel in mp.ARTIFACTS if not (mp.REPO_SUBDIR / rel).is_file()
)


@pytest.mark.skipif(
    bool(_missing_learned_artifacts),
    reason=(
        "learned benchmark artifacts absent "
        f"({', '.join(_missing_learned_artifacts)}) — these are trained "
        "artifacts, deliberately unpublished and not part of "
        "tekmyra-reference-artifacts.tar.gz; content pinning is only "
        "checkable in the measuring tree. See README \"What you cannot do\". "
        "Absence handling itself is verified by the other tests here, e.g. "
        "test_an_absent_runtime_artifact_blocks_quotation."
    ),
)
def test_collect_pins_the_two_learned_artifacts_by_content() -> None:
    m = mp.collect()
    assert set(m.artifacts) == set(mp.ARTIFACTS)
    for rel, digest in m.artifacts.items():
        assert digest != mp.UNKNOWN, f"{rel} could not be digested"
        assert len(digest) == 64


def test_collect_finds_a_real_commit_not_the_unknown_sentinel() -> None:
    """Guards the repo-root trap.

    This repository's git root is one level above the package directory. A git
    invocation written relative to the package returns empty output, which
    silently compared a real artifact against an empty blob during this
    module's own development and read as "the artifact differs from HEAD".
    """
    m = mp.collect()
    assert m.commit != mp.UNKNOWN
    assert len(m.commit) == 40


def test_every_declared_corpus_is_present_and_non_empty() -> None:
    """A digest over zero files is still a valid-looking hex string.

    Same shape as a suite exiting 0 having collected nothing: the artefact of
    the measurement looks identical whether or not any measuring happened.
    """
    m = mp.collect()
    for rel, info in m.corpora.items():
        assert info["digest"] != mp.UNKNOWN, f"{rel} missing or empty"
        assert info["files"] > 0, f"{rel} digested zero files"


def test_the_four_public_corpora_are_covered() -> None:
    """138 public fixtures exist and no committed baseline measures them."""
    m = mp.collect()
    public = {k: v for k, v in m.corpora.items() if "public_" in k}
    assert len(public) == 4
    assert sum(int(v["files"]) for v in public.values()) == 138


def test_corpus_digest_is_stable_across_calls() -> None:
    assert mp.collect().corpora == mp.collect().corpora


# --- the refusals --------------------------------------------------------

def test_dirty_tree_refuses_publication() -> None:
    with pytest.raises(mp.NotQuotable, match="DIRTY"):
        mp.require_quotable(_manifest(tree_dirty=True))


def test_unknown_artifact_refuses_publication() -> None:
    with pytest.raises(mp.NotQuotable, match="UNKNOWN"):
        mp.require_quotable(_manifest(artifacts={"x.pkl": mp.UNKNOWN}))


def test_unknown_commit_refuses_publication() -> None:
    with pytest.raises(mp.NotQuotable, match="commit"):
        mp.require_quotable(_manifest(commit=mp.UNKNOWN))


def test_unknown_corpus_refuses_publication() -> None:
    with pytest.raises(mp.NotQuotable, match="corpus"):
        mp.require_quotable(
            _manifest(corpora={"c": {"digest": mp.UNKNOWN, "files": 0}})
        )


def test_unknown_tree_state_refuses_rather_than_assuming_clean() -> None:
    """UNKNOWN is not clean. Defaulting an unknown to the permissive value is
    how a check ends up passing on exactly the state it exists to catch."""
    with pytest.raises(mp.NotQuotable):
        mp.require_quotable(_manifest(tree_dirty=mp.UNKNOWN))


def test_a_fully_pinned_clean_manifest_is_quotable() -> None:
    m = _manifest()
    assert m.unknowns() == []
    assert m.is_quotable() is True
    mp.require_quotable(m)  # must not raise


def test_unknowns_enumerates_every_missing_field_not_just_the_first() -> None:
    """Reporting one cause at a time turns one fix into N round trips."""
    m = _manifest(
        commit=mp.UNKNOWN,
        artifacts={"a.pkl": mp.UNKNOWN, "b.pkl": "d" * 64},
        libraries={"numpy": mp.UNKNOWN},
    )
    found = m.unknowns()
    assert "commit" in found
    assert "artifact:a.pkl" in found
    assert "artifact:b.pkl" not in found
    assert "library:numpy" in found


def test_manifest_serialises_to_json() -> None:
    parsed = json.loads(_manifest().to_json())
    assert parsed["commit"] == "a" * 40
    assert parsed["libraries"]["sklearn"] == "1.9.0"


def test_invocation_is_captured_so_two_runs_of_the_same_benchmark_differ_visibly() -> None:
    m = mp.collect(invocation=["bench", "--budget", "0.30"])
    assert m.invocation == ["bench", "--budget", "0.30"]
