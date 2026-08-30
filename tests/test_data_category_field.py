"""The category dimension exists on the contract, and stays unpopulated.

Both span models are ``extra="forbid"``, so the field has to be declared before
anything can carry it — that is the whole of what is authorised here. The value
set, and populating it at the five real construction sites, are the taxonomy
ruling and are held.

THE RISK THIS FILE POLICES is not that the field is wrong. It is that a nullable
category quietly becomes a THIRD way for the PII leakage check to look alive
while asserting nothing. The check has read False forever because it compares a
handling label against a category list; add a field that is None everywhere, let
one consumer read None as "nothing to report", and the estate acquires a fresh
zero that looks like a measurement. ``None`` MEANS NOT CATEGORISED.

So the guard is deliberately awkward: no shipped module may READ this field
yet. Wiring the first consumer requires deleting a test, which is the reviewable
act — the same shape as the closed-set ratchet and the dead-name guard.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tekmyra.contracts import BlockSpan, ProtectedSpan

REPO_ROOT = Path(__file__).resolve().parent.parent
FIELD = "data_category"

# REPOSITORY SCOPE, NOT A DIRECTORY LIST — Vega's ruling, applied here because
# this guard had the same defect as the PII one it was written alongside. That
# one scanned four directories, reported clean, and missed sites in tests/ and
# benchmarks/; widening it three times found more every time. A guard with a
# directory scope inherits the defect it was built to fix: it reports clean
# about what it does not scan.
EXCLUDED_DIRS = frozenset({"__pycache__", ".git", "build", "dist", ".cache"})
EXCLUDED_FILES = frozenset({"contracts.py", "test_data_category_field.py"})


def _environment_roots() -> frozenset[Path]:
    """Virtualenvs in the tree, found by their marker file rather than by name.

    ``.venv`` used to sit in EXCLUDED_DIRS, which meant this guard worked for
    people who picked that one name and broke for everyone else. Measured on the
    payload: a virtualenv at ``venv/`` or ``env/`` makes both scanning tests
    fail with a UnicodeDecodeError raised inside ``<frozen codecs>``, naming
    neither the file nor the cause -- ``rglob`` reaches an installed ``.py``
    that is not UTF-8. ``python -m venv venv`` is the invocation in Python's own
    documentation, so the common case was the broken one, and the README tells
    readers this suite is green.

    A name list cannot be finished; ``pyvenv.cfg`` is written by every stdlib
    ``venv`` and by ``uv``, so the marker can. This keeps the repository-wide
    scope the guard is built on -- it prunes environments, never project code.
    """
    return frozenset(marker.parent for marker in REPO_ROOT.rglob("pyvenv.cfg"))


def _repository_python_files() -> list[Path]:
    environments = _environment_roots()
    return [
        path
        for path in sorted(REPO_ROOT.rglob("*.py"))
        if not (EXCLUDED_DIRS & set(path.parts))
        and path.name not in EXCLUDED_FILES
        and not any(env in path.parents for env in environments)
    ]


def _source_of(path: Path) -> str:
    """Read a scanned file, failing with the path instead of a codec traceback."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise AssertionError(
            f"{path.relative_to(REPO_ROOT)} is not valid UTF-8 "
            f"({exc.reason}, byte {exc.start}). This guard scans the whole "
            "repository deliberately. If this file belongs to a tool or an "
            "environment rather than to the project, exclude it structurally "
            "-- see _environment_roots -- rather than by adding a name."
        ) from None


def _scope_note() -> str:
    """A guard must state its own scope; silence is not a clean report."""
    environments = _environment_roots()
    pruned = sorted(env.relative_to(REPO_ROOT).as_posix() for env in environments)
    return (
        f"[scope: {len(_repository_python_files())} .py files under "
        f"{REPO_ROOT.name}, excluding {sorted(EXCLUDED_DIRS)}, "
        f"{sorted(EXCLUDED_FILES)} and virtualenvs {pruned}]"
    )


def test_both_span_models_declare_the_field() -> None:
    """extra='forbid' means it cannot be added at a construction site."""
    for model in (ProtectedSpan, BlockSpan):
        assert FIELD in model.model_fields, f"{model.__name__} lacks {FIELD}"
        assert model.model_config.get("extra") == "forbid", (
            f"{model.__name__} no longer forbids extras, so the field could be "
            "smuggled in ad hoc and the declaration stops being load-bearing"
        )


def test_the_default_is_none_and_none_means_uncategorised() -> None:
    """A span with no category must not claim to have none."""
    span = ProtectedSpan(
        span_type="locked",
        start=0,
        end=3,
        text="abc",
        policy="preserve",
        detector="test",
        confidence=1.0,
    )
    assert span.data_category is None

    block_span = BlockSpan(
        start=0, end=3, label="email", span_type="locked", detector="test", confidence=1.0
    )
    assert block_span.data_category is None


def test_the_field_accepts_a_value_without_constraining_it() -> None:
    """Deliberately unconstrained: the value set is the taxonomy's to rule."""
    span = BlockSpan(
        start=0,
        end=3,
        label="email",
        span_type="locked",
        detector="test",
        confidence=1.0,
        data_category="email",
    )
    assert span.data_category == "email"


def test_nothing_reads_the_field_yet() -> None:
    """Held: populating and consuming are the taxonomy ruling, not this.

    Reading it before anything writes it means reading None everywhere, which
    is exactly the failure this field is meant to end rather than repeat. When
    the first real consumer lands, delete this test on purpose.
    """
    readers: list[str] = []
    for path in _repository_python_files():
        tree = ast.parse(_source_of(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == FIELD:
                readers.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif (
                isinstance(node, ast.Constant)
                and node.value == FIELD
                and isinstance(node.value, str)
            ):
                readers.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} (key)")
    assert not readers, (
        f"{FIELD} is read at {readers} {_scope_note()} while nothing "
        "populates it, so every "
        "read returns None. If None is being treated as 'no PII', that is the "
        "dead check reborn. Delete this test deliberately when the first "
        "writer lands."
    )


def test_no_construction_site_populates_it_yet() -> None:
    """The five real sites are held pending the taxonomy."""
    writers: list[str] = []
    for path in _repository_python_files():
        tree = ast.parse(_source_of(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == FIELD:
                writers.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not writers, (
        f"{FIELD} is populated at {writers} {_scope_note()}. Sourcing its "
        "value is the hard "
        "part and is held: a value assigned before the vocabulary exists is "
        "how the next vocabulary gets born."
    )
