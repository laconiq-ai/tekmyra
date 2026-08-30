"""The ratchet: keep the public-corpus figure honest on the NEXT run.

Vega §6.11 — everything else describes how to produce an honest number ONCE.
This is what keeps it honest afterwards, and until now it was attached to the
DIAGNOSTIC path (``codec_effectiveness``) rather than to the path that produces
a quotable claim. Satisfying that in form while missing it in substance is the
specific failure the clause was written around.

FAILS ON DRIFT IN EITHER DIRECTION, and the improving direction is the one that
matters. A number that silently degrades gets investigated; a number that
silently improves gets quoted. Both mean the measured system changed without
anyone deciding to change it, and only one of them is self-correcting.

ASSERTS THE NON-ZERO CONTROL ITSELF rather than trusting the module to have
done so. Otherwise a future change that breaks the compressor produces a table
of honest-looking zeros, every cell matches nothing in particular, and nothing
trips.

Regenerate deliberately, never to make a red test green:

    python -c "..." > tests/baselines/public_corpus_baseline.json

and re-read the diff before committing it. A ratchet regenerated on failure is
a ratchet that has been disabled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tekmyra import artifact_paths

from tests import measurement_provenance as mp
from tests import public_corpus_baseline as b

BASELINE_PATH = Path(__file__).resolve().parent / "baselines" / "public_corpus_baseline.json"

# ON A CLEAN CLONE THE RATE TESTS MUST SKIP, VISIBLY — NEVER PASS. Without the
# reference artifacts the learned compressor is a silent no-op, so this run
# measures a DIFFERENT system than the baseline's and its rates are honestly
# incomparable. A pass here would certify a number that was not reproduced; a
# fail blames the cloner for following the README. The skip states which it is.
# Resolution is the same search the code itself uses (TEKMYRA_ARTIFACT_ROOT
# included), so the guard and the codec cannot disagree about what "installed"
# means. The corpus-integrity tests below run regardless: bytes and attribution
# verify without a model, and test_public_corpus_notice.py covers the rest.
REFERENCE_ARTIFACT_DIRS = (".cache/compressor/1.0.0", ".cache/router/1.0.0")
_missing_reference_artifacts = tuple(
    rel for rel in REFERENCE_ARTIFACT_DIRS if not artifact_paths.resolve(rel).found
)
requires_reference_artifacts = pytest.mark.skipif(
    bool(_missing_reference_artifacts),
    reason=(
        "reference artifacts not installed "
        f"({', '.join(_missing_reference_artifacts)}) — the learned compressor "
        "is a no-op here, so measured rates describe a different system than "
        "the baseline's. Fetch tekmyra-reference-artifacts.tar.gz per the "
        'README section "You need a model, and here is how to get one" and '
        "unpack it at the clone root (or point TEKMYRA_ARTIFACT_ROOT at it). "
        "Corpus bytes and attribution still verify without a model via "
        "test_public_corpus_notice.py."
    ),
)

# Byte-exact arithmetic on identical inputs should be identical. The tolerance
# exists only to absorb float formatting at the fourth decimal, not to make room
# for real movement -- anything larger is drift and must be seen.
TOLERANCE_PCT = 0.0001


@pytest.fixture(scope="module")
def baseline() -> dict:
    assert BASELINE_PATH.exists(), f"missing committed baseline at {BASELINE_PATH}"
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="module")
def measured() -> tuple[dict, dict]:
    return b.measure()


def test_the_baseline_carries_its_own_provenance(baseline: dict) -> None:
    """The figure and what produced it must live in ONE artifact.

    A committed number whose provenance lives somewhere else is the same defect
    as a number quoted out of its table.
    """
    m = baseline["manifest"]
    assert m["commit"] and m["commit"] != mp.UNKNOWN
    assert m["tree_dirty"] is False, "baseline was generated from a dirty tree"
    assert m["measured_corpus"] != mp.UNKNOWN
    assert m["basis"] != mp.UNKNOWN
    assert m["budget"] != mp.UNKNOWN
    for rel, info in m["runtime_artifacts"].items():
        assert info["present"], f"baseline generated without {rel}"
        assert info["digest"] != mp.UNKNOWN


def test_the_measurement_still_registers_a_saving(measured) -> None:
    """The control, asserted HERE and not only in the module.

    Without this, a broken compressor yields a table of zeros that matches
    nothing suspicious and trips no other assertion.
    """
    cells, _ = measured
    assert b.assert_non_zero_control(cells)


def test_corpus_size_has_not_changed(baseline: dict, measured) -> None:
    """A fixture added or lost changes the number for a legitimate reason —
    which is exactly why it must not happen silently."""
    _, counts = measured
    assert counts == baseline["measurement"]["corpus_file_counts"]


@requires_reference_artifacts
@pytest.mark.parametrize(
    "metric",
    [
        "HEADLINE_effective_reduction_pct",
        "DIAGNOSTIC_on_accepted_reduction_pct",
        "refusal_rate_bytes_pct",
        "refusal_rate_fixtures_pct",
    ],
)
def test_headline_metrics_have_not_drifted(baseline: dict, measured, metric: str) -> None:
    cells, _ = measured
    now = b.corpus_totals(cells)[metric]
    then = baseline["measurement"]["totals"][metric]
    delta = float(now) - float(then)
    assert abs(delta) <= TOLERANCE_PCT, (
        f"{metric} DRIFTED {delta:+.4f} points: baseline {then}, now {now}. "
        "Drift in EITHER direction is a failure — an unexplained improvement is "
        "not good news, it means the measured system changed and nobody decided "
        "to. Investigate, then regenerate the baseline deliberately."
    )


@requires_reference_artifacts
def test_byte_totals_are_identical_not_merely_close(baseline: dict, measured) -> None:
    """Percentages can agree while the underlying bytes differ.

    Two different corpora can round to the same reduction, so the percentage
    check alone would pass on a substituted corpus.
    """
    cells, _ = measured
    now = b.corpus_totals(cells)
    then = baseline["measurement"]["totals"]
    assert now["input_bytes"] == then["input_bytes"]
    assert now["output_bytes"] == then["output_bytes"]
    assert now["fixtures"] == then["fixtures"]
    assert now["refused_fixtures"] == then["refused_fixtures"]


def test_no_cell_appeared_or_disappeared(baseline: dict, measured) -> None:
    """The three-versus-four case, caught by machine instead of by a second seat.

    A cell arriving or vanishing changes what the table means even when every
    surviving number is unchanged.
    """
    cells, _ = measured
    assert set(cells) == set(baseline["measurement"]["by_route_and_content_type"])


def test_no_cell_is_emitted_with_zero_fixtures(measured) -> None:
    """A phantom row is the shape that survives review and embarrasses later."""
    cells, _ = measured
    assert [k for k, c in cells.items() if c.fixtures == 0] == []


@requires_reference_artifacts
def test_every_cell_matches_the_baseline(baseline: dict, measured) -> None:
    cells, _ = measured
    for key, expected in baseline["measurement"]["by_route_and_content_type"].items():
        cell = cells[key]
        assert cell.fixtures == expected["fixtures"], key
        assert cell.refused == expected["refused"], key
        assert cell.input_bytes_all == expected["input_bytes"], key
        assert cell.output_bytes_all == expected["output_bytes"], key
        assert abs(cell.effective_reduction_pct
                   - expected["HEADLINE_effective_reduction_pct"]) <= TOLERANCE_PCT, key


def test_cell_fixture_counts_sum_to_the_corpus(baseline: dict) -> None:
    """Reconciles the table against the corpus rather than against itself."""
    per_cell = baseline["measurement"]["by_route_and_content_type"]
    assert sum(c["fixtures"] for c in per_cell.values()) == sum(
        baseline["measurement"]["corpus_file_counts"].values()
    )


# THE PIN THE README POINTS AT, SO THE SENTENCE POINTING AT IT IS TRUE.
# The baseline's own long_context_v1 digest was taken in the private tree over
# 44 files including two unpublished generator scripts, so a cloner recomputing
# it gets a mismatch by construction — and the README paragraph explaining that
# briefly claimed "the ratchet suite proves the 42 shipped files are the 42
# that were measured" while nothing did: a reviewer appended a canary byte to a
# fixture and the whole suite stayed green. This is that proof. It binds the
# FIXTURE BYTES ONLY (.json and .md, the extensions staging publishes and
# load_fixtures reads), so it computes identically in the private tree and in
# the payload. Regenerate the constants only for a deliberate corpus change,
# and re-read the diff before committing — the same rule as every ratchet in
# this file.
LONG_CONTEXT_FIXTURE_FILES = 42
LONG_CONTEXT_FIXTURE_DIGEST = "985ad775820de4b91f3dfbf9bdf77ae9c4523f1ae521a6147b844a4b9935da8e"


def test_long_context_corpus_bytes_are_pinned() -> None:
    corpus = mp.REPO_SUBDIR / "benchmarks" / "fixtures" / "long_context_v1"
    files = sorted(
        p for p in corpus.rglob("*")
        if p.is_file() and p.suffix in {".json", ".md"}
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(corpus).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    assert len(files) == LONG_CONTEXT_FIXTURE_FILES, (
        f"long_context_v1 carries {len(files)} fixture files, expected "
        f"{LONG_CONTEXT_FIXTURE_FILES} — a file appeared or vanished without "
        "a deliberate corpus change"
    )
    assert digest.hexdigest() == LONG_CONTEXT_FIXTURE_DIGEST, (
        "long_context_v1 fixture bytes changed. The 0.2929 headline and the "
        "refusal accounting were measured on the pinned bytes; regenerate the "
        "pin only for a corpus change somebody decided on."
    )
