"""A codec that reports success and returns its input is not a built codec.

The RouteClass ratchet (``test_route_class_ratchet.py``) closed the defect
"a route can be declared without being built and nothing fails". It closed it
at the only granularity it can see: whether ``apply_codec`` refuses. A codec
that RUNS and returns the input unchanged reports ``success=True``, which is
byte-for-byte indistinguishable from a real compression that saved nothing.

Measured at 9122dab, two routes were in exactly that state and neither was
visible to any existing guard or benchmark — ``MEASURABLE_CONTENT_TYPES``
excludes code, log, diff and json, so no aggregate ever included them:

  code_compressor  returned its input on 12 of 12 code fixtures
  log_compressor   returned its input on 15 of 15 log fixtures

A published table listing six built codecs would have been accurate and
misleading in the same sentence. That is what this file exists to prevent.

WHY THE UNIT IS (ROUTE, CONTENT TYPE). At route granularity ``log_compressor``
reads as effective — mean ratio 0.999 — purely because eight ``stacktrace`` and
``mixed`` fixtures reach it and compress slightly. Aggregating over the route
rescues the codec with content it was not being judged on.

WHY "refused" IS RECORDED BUT NEVER RULED ON HERE. Refusal is the RouteClass
ratchet's jurisdiction and its baseline is the authority. This guard records
the bucket only so the three are exhaustive; two baselines that could both rule
on one route would eventually disagree, and the disagreement would be silent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.codec_effectiveness import (
    CORPORA,
    as_baseline,
    compressor_artifact_loaded,
    measure,
)

BASELINE = Path(__file__).parent / "baselines" / "codec_effectiveness.json"


def _baseline() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def _cells(bucket: dict) -> dict[str, dict]:
    return {k: v for k, v in bucket.items() if not k.startswith("_")}


@pytest.fixture(scope="module")
def measured() -> dict:
    """Measure once; refuse to report at all if the instrument is degraded.

    ``load_compressor`` returns a silent no-op when ``.cache/compressor`` is
    absent, and that directory is gitignored. Without it the prose codec
    compresses nothing and every prose cell reads INERT — so an unguarded run
    would not merely give a wrong answer, it would offer to BASELINE the
    missing artifact as a property of the codec. This fails loudly instead of
    skipping, because a skipped guard and a passing guard are the same colour.
    """
    if not any(c.is_dir() for c in CORPORA):
        pytest.skip("no fixture corpus present")
    assert compressor_artifact_loaded(), (
        "no trained prose compressor artifact is loaded, so every prose route "
        "would measure as INERT and this guard would record the absence of an "
        "artifact as a defect in the codec. Restore .cache/compressor (it is "
        "gitignored) before trusting any result from this file."
    )
    return as_baseline(measure())


def test_no_codec_is_silently_inert(measured: dict) -> None:
    """The guard itself: a codec that does nothing must be recorded as such."""
    observed = set(_cells(measured.get("inert", {})))
    recorded = set(_cells(_baseline()["inert"]))

    unrecorded = observed - recorded
    assert not unrecorded, (
        "codec(s) ran, reported success, and returned the input unchanged "
        f"without being recorded: {sorted(unrecorded)}. Either the codec is "
        "broken, or it is inert by design and belongs in "
        "tests/baselines/codec_effectiveness.json WITH A REASON — 'inert by "
        "design' and 'inert by defect' look identical from outside."
    )


def test_a_codec_that_started_working_is_removed_from_the_inert_list(
    measured: dict,
) -> None:
    """The ratchet direction: the inert list may only SHRINK."""
    observed = set(_cells(measured.get("inert", {})))
    recorded = set(_cells(_baseline()["inert"]))

    now_working = recorded - observed
    assert not now_working, (
        f"codec(s) recorded as inert now compress: {sorted(now_working)}. "
        "Remove them from the baseline — the list may only shrink, and a stale "
        "entry would keep excusing a codec that no longer needs excusing."
    )


def test_no_effective_codec_regressed_to_doing_nothing(measured: dict) -> None:
    """The regression this guard is worth having for.

    A codec quietly losing its compression — a disabled flag, a changed
    default, a refactor that returns early — currently shows up nowhere. It
    does not fail verification, because returning the input intact passes every
    fidelity check there is. It is the perfectly safe answer to the wrong
    question.
    """
    effective_now = set(_cells(measured.get("effective", {})))
    inert_now = set(_cells(measured.get("inert", {})))
    was_effective = set(_cells(_baseline()["effective"]))

    regressed = sorted(was_effective & inert_now)
    assert not regressed, (
        f"codec(s) that used to compress now return the input unchanged: "
        f"{regressed}. Nothing else in the suite catches this: an uncompressed "
        "output passes every fidelity check by construction."
    )

    vanished = sorted(was_effective - effective_now - inert_now)
    assert not vanished, (
        f"cell(s) in the baseline no longer receive any traffic: {vanished}. "
        "Routing changed, so this guard silently stopped covering them."
    )


def test_no_codec_mutates_content_without_compressing_it(measured: dict) -> None:
    """A ratio of 1.0 is not the same fact as an unchanged output.

    This guard classified by LENGTH ALONE until 2026-08-09, so a codec that
    changed content to the same length read as "inert" — the guard written to
    catch codecs that do nothing was blind to a codec doing something wrong.

    An independent QA seat found exactly that: a Go file misdetected as Python
    returned ``success=True`` and ``ratio=1.0`` while NOT being byte-identical
    to its input, because the Python tokenizer round-trip turned a tab inside a
    ``//`` comment into a space. Silent mutation on a route whose contract is
    that it does not alter content, reported as a clean pass.
    """
    offenders = []
    for bucket in ("effective", "inert"):
        for key, row in _cells(measured.get(bucket, {})).items():
            if row.get("mutated_without_compressing"):
                offenders.append((key, row["mutated_without_compressing"]))
    assert not offenders, (
        f"codec(s) changed content without compressing it: {offenders}. "
        "Same length, different bytes — the output is not the input and the "
        "codec saved nothing, so the change is pure loss."
    )


def test_the_mutation_check_can_actually_fire() -> None:
    """Zero mutations is the reassuring answer, and needs a control.

    Constructed rather than measured, because no codec currently produces the
    condition on these corpora — which is precisely why a passing result above
    proves nothing without this.
    """
    from tests.codec_effectiveness import Cell

    cell = Cell(route="r", content_type="c")
    text, same_length_but_different = "abc\tdef", "abc def"
    assert len(text) == len(same_length_but_different)

    ratio = len(same_length_but_different) / len(text)
    assert ratio == 1.0, "the control must sit exactly where the old guard was blind"
    if ratio < 1.0:
        cell.compressed += 1
    elif same_length_but_different == text:
        cell.returned_input_unchanged += 1
    else:
        cell.mutated_without_compressing += 1

    assert cell.mutated_without_compressing == 1, (
        "the classification does not separate 'returned its input' from "
        "'same length, different bytes', so the guard above cannot fire"
    )
    assert cell.returned_input_unchanged == 0


def test_every_inert_entry_carries_a_reason() -> None:
    """A named exemption owes an assertion — a bare entry is a silent excuse."""
    for key, entry in _cells(_baseline()["inert"]).items():
        assert entry.get("reason"), f"{key} is exempted without a reason"
        assert entry.get("classification"), (
            f"{key} does not say whether it is inert BY DESIGN or BY DEFECT, "
            "which is the only thing a reader needs from this file"
        )


def test_new_routing_coverage_is_recorded(measured: dict) -> None:
    """A new (route, content_type) pair is new coverage and must be seen."""
    recorded = set()
    for bucket in ("effective", "inert", "refused"):
        recorded |= set(_cells(_baseline()[bucket]))
    observed = set()
    for bucket in ("effective", "inert", "refused"):
        observed |= set(_cells(measured.get(bucket, {})))

    assert observed == recorded, (
        "the set of (route, content_type) pairs the corpus exercises changed.\n"
        f"  new, unrecorded: {sorted(observed - recorded)}\n"
        f"  recorded but no longer exercised: {sorted(recorded - observed)}"
    )


def test_the_guard_is_not_vacuous(measured: dict) -> None:
    """A guard that measured nothing passes every assertion above."""
    effective = _cells(measured.get("effective", {}))
    assert effective, (
        "no codec compressed anything at all — this guard would pass while "
        "measuring nothing, which is the failure mode it was written against"
    )
    assert sum(v["fixtures"] for v in effective.values()) > 0
    assert any(
        k.startswith("prose_token_compressor/") for k in effective
    ), "the prose compressor is the workhorse; if it is not effective, the instrument is broken, not the codec"
