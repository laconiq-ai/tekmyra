"""Skip the tests that need a trained artifact, and SAY WHERE WE LOOKED.

**Trained model artifacts are not published.** They are the commercial half, and
that is a decision rather than an oversight — so a fresh clone has no
``.cache/compressor/1.0.0`` and no ``.cache/router/1.0.0``, and every test that
measures the learned path has nothing to measure.

TWO WRONG ANSWERS, AND WHY EACH IS WRONG.

*Let them fail.* A test suite that goes red the moment a stranger clones it
teaches them the project is broken, and it is the same footgun class as the
compressor that silently returned a no-op from the wrong directory: the observed
signal describes the environment, not the code.

*Delete them.* That guts the suite's real coverage. The tests that exercise the
learned compressor are the ones a contributor most needs when they change it, and
a repository whose only green tests are the ones that could not fail is not
evidence of anything.

So they SKIP, and the skip states its reason and **every location that was
searched** — the same legibility principle as the artifact resolver itself. A
skip reading "artifact missing" sends the reader to guess a path; one that lists
the three places we looked lets them fix it in a single step, usually by setting
``TEKMYRA_ARTIFACT_ROOT``.

THE LIST IS CURATED AND THEREFORE DECLARED. Which tests need an artifact is not
computable from imports — they reach it through the runtime, several layers down
— so each entry carries its reason and the register is visible rather than
implicit. It sits beside the computed rule in ``scripts/open_surface.py`` that
handles everything computable.

**This file is inert wherever the artifacts are present**, which includes the
development repository: nothing is skipped there and the suite behaves exactly
as it did before.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

#: module basename -> why it cannot run without a trained artifact
NEEDS_TRAINED_ARTIFACT: dict[str, str] = {
    "test_benchmark_per_route.py": "the per-route table is produced by the learned router, which refuses to emit a number without its artifact",
    "test_benchmark_router_provenance.py": "asserts on the provenance of a loaded router artifact",
    "test_refusal_denominator.py": "counts refusals in a benchmark run, which cannot start without the router artifact",
    "test_codec_effectiveness_ratchet.py": "guards prose-route effectiveness; with no trained compressor every prose route measures as INERT and the guard would record the absence as a regression",
    "test_wp7_real_runtime.py": "drives the real runtime end to end over the corpus",
    "test_compressor_entity_fidelity.py": "measures entity retention through the learned compressor",
}

#: module basename -> the optional dependency it needs, and why it is optional
NEEDS_OPTIONAL_DEPENDENCY: dict[str, tuple[str, str]] = {
    "test_onnx_runtime.py": (
        "onnxruntime",
        "the ONNX execution path is an optional extra; the deterministic and "
        "sklearn paths work without it",
    ),
}


def anchor_resolution_available() -> bool:
    """Whether THIS build can resolve anchors, asked of the store itself.

    Deliberately not a module-presence check. ``import tekmyra.anchor_store``
    answers "is the file reachable", and an editable-install finder or a loose
    directory on ``sys.path`` can make that true in a tree that does not contain
    it — which is the exact trap that made the verification script report a
    passing suite on a tree whose suite does not run. The runtime asks the STORE
    what it is; so does this.
    """
    from tekmyra.anchor_interface import default_anchor_store
    from tekmyra.router_provenance import capture_anchor_store

    return capture_anchor_store(default_anchor_store())["anchor_resolution_available"]


def run_or_assert_open_refusal(run):
    """Run ``run()``, or prove the open build refused for the one reason it may.

    THE OPEN CORE REFUSES STRICTLY MORE THAN THE COMMERCIAL BUILD, BY DESIGN. It
    cannot resolve anchors, so it declines cases the commercial build accepts as
    recoverable. Five shipped tests asserted completion regardless and therefore
    could not pass in the build that ships them; they only appeared to pass
    because the gate running them had the withheld module on its path.

    A skip would have been the easy repair and the wrong one. The refusal is
    CORRECT behaviour and it is checkable, so asserting it makes the seam a
    documented property of the product rather than a gap nobody mentions.

    Returns the response, or ``None`` when the open core correctly refused —
    callers skip their completion assertions in that case and keep them for
    every fixture the open build does handle.

    WHAT IS ASSERTED, AND WHAT DELIBERATELY IS NOT. The exception type, its
    message, and the presence of ``anchors_resolvable`` among the failed checks
    are invariant across both shipped corpora and both entry points. ``route``
    and the exact ``failed_checks`` tuple are NOT: they are ``passthrough`` with
    two checks on the synthetic corpus and ``reversible_retrieval_mode`` with
    three on long_context_v1. Pinning either would go red for a reason unrelated
    to the seam the moment a fixture with protected facts appears — and a test
    that fails for the wrong reason is the one somebody deletes at an
    inconvenient moment. ``anchors_resolvable`` is also the only member that is
    there BECAUSE the store cannot resolve; ``locked_spans_preserved`` is
    downstream of it, a consequence rather than the cause.
    """
    from tekmyra.runtime import TerminalVerificationError

    try:
        return run()
    except TerminalVerificationError as exc:
        if anchor_resolution_available():
            raise
        assert str(exc) == "verification_failed_after_fallback", str(exc)
        assert "anchors_resolvable" in exc.failed_checks, exc.failed_checks
        return None


@pytest.fixture
def run_or_refusal():
    """``run_or_assert_open_refusal``, injected rather than imported.

    ``from conftest import ...`` relies on pytest inserting the test directory
    into ``sys.path``, which does not happen under ``--import-mode=importlib``.
    A fixture works under every import mode.
    """
    return run_or_assert_open_refusal


def _artifact_search() -> tuple[bool, str]:
    """Ask the package where it looks, rather than re-deriving an answer here.

    Re-deriving is the defect this project spent a day removing: a diagnostic
    that computes its own path can report an artifact PRESENT inside the message
    explaining it is missing.
    """
    try:
        from tekmyra import artifact_paths
    except Exception as exc:  # pragma: no cover - the package must import
        return False, f"tekmyra.artifact_paths could not be imported: {exc}"

    compressor = artifact_paths.resolve(Path(".cache/compressor/1.0.0"))
    router = artifact_paths.resolve(Path(".cache/router/1.0.0"))
    found = compressor.found and router.found
    detail = (
        "No trained artifact was found. Trained artifacts are not published — "
        "they are the commercial half — so this is the expected state of a fresh "
        "clone.\n\n"
        f"{compressor.describe()}\n\n{router.describe()}\n\n"
        f"Set ${artifact_paths.ARTIFACT_ROOT_ENV} to the directory holding "
        ".cache/, or download the demo artifacts named in the README."
    )
    return found, detail


def pytest_collection_modifyitems(config, items) -> None:  # noqa: ARG001
    artifact_found, artifact_detail = _artifact_search()

    for item in items:
        name = Path(str(item.fspath)).name

        reason = NEEDS_TRAINED_ARTIFACT.get(name)
        if reason is not None and not artifact_found:
            item.add_marker(
                pytest.mark.skip(reason=f"{reason}.\n\n{artifact_detail}")
            )
            continue

        optional = NEEDS_OPTIONAL_DEPENDENCY.get(name)
        if optional is not None:
            module, why = optional
            try:
                importlib.import_module(module)
            except ImportError:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"{module} is not installed. {why}. Install it with "
                            f"the matching extra to run these."
                        )
                    )
                )
