"""The compression operating point must have exactly one definition.

Hari ruled the operating point to 0.15 on 2026-08-09, choosing from 0.30 /
0.15 / 0.05. Landing that ruling turned up why the previous value had never
been safely changeable: **it existed at eight sites**, all reading 0.8.

    tekmyra/runtime.py            _DEFAULT_BUDGET_RATIO = 0.8
    tekmyra/runtime.py            budget_ratio: float = 0.8   (x2 signatures)
    learned_compressor/compressor.py     _DEFAULT_BUDGET = 0.8
    codecs/selector.py                   budget_ratio: float = 0.8   (apply_codec)
    codecs/selector.py                   kwargs.get("budget_ratio", 0.8)
    codecs/selector.py                   ... else 0.8   (malformed-value path)
    codecs/safe_mode.py                  the same two, again

The first carried a comment saying it was "named here so the layers cannot
drift". Naming it in one layer did not stop the other seven keeping their own
copy, and changing any single one would have left the rest silently enforcing
the old point — the change would have looked landed and not been.

The count reached eight one failure at a time. Every site after the first two
was found by a guard in this file failing, never by reading the code, which is
the only reason to believe there is not a ninth.

Worse, the shipped 0.8 was never the number anyone reported. Every published
figure was measured at 0.30, which the benchmark harness passes explicitly. At
the shipped default the corpus delivers **16.30%** character reduction against
a reported 51.38%. The gap between "what we measure at" and "what it runs at"
was invisible precisely because no single place held the answer.

So this file asserts the property rather than the value: one definition, and
every other site derives from it. A test pinning 0.15 would go stale the next
time the founder rules; a test pinning the *shape* stays true.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tekmyra import runtime
from tekmyra.learned_compressor import compressor

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTOR = REPO_ROOT / "tekmyra" / "codecs" / "selector.py"


def test_runtime_derives_the_operating_point_rather_than_restating_it() -> None:
    """Identity, not equality — two constants that merely agree today drift."""
    assert runtime._DEFAULT_BUDGET_RATIO is compressor._DEFAULT_BUDGET, (
        "runtime restates the operating point instead of importing it. Two "
        "constants that happen to be equal are exactly what produced the "
        "four-site split; equality is not the property, single definition is."
    )


def test_the_operating_point_is_a_usable_ratio() -> None:
    """A guard against landing a founder ruling as a typo."""
    value = compressor._DEFAULT_BUDGET
    assert isinstance(value, float)
    assert 0.0 < value <= 1.0, f"not a usable budget ratio: {value}"


def _budget_get_defaults(source: str) -> list[object]:
    """Every literal default handed to a ``kwargs.get("budget_ratio", X)`` call.

    Narrowed to that call deliberately. An earlier version of this guard
    rejected ANY float literal in the function and failed on ``ratio=1.0`` in
    the safety-tier passthrough — a correct literal that has nothing to do with
    the operating point. A guard that fires on legitimate code gets relaxed by
    the next person to hit it, and then it is not a guard.
    """
    found: list[object] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "get"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "budget_ratio":
            continue
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            found.append(node.args[1].value)
    return found


def test_no_codec_defaults_the_budget_to_a_literal() -> None:
    """The codec sites, checked structurally rather than by grep.

    Both codec selectors read the budget from kwargs and previously defaulted
    to a bare 0.8. A literal here silently overrides the ruled operating point
    for any caller that does not pass one — which is the shape the whole defect
    had, eight times over.
    """
    for path in (SELECTOR, SELECTOR.parent / "safe_mode.py"):
        literals = _budget_get_defaults(path.read_text(encoding="utf-8"))
        assert not literals, (
            f"{path.name} defaults budget_ratio to literal(s) {literals}. It "
            "must come from the single definition in "
            "learned_compressor.compressor, or changing the operating point "
            "silently misses this path."
        )


def test_the_guard_would_notice_a_reintroduced_literal() -> None:
    """Negative control — the walk must see a literal when one is there.

    Matched by PATTERN, not by the identifier currently in the source. An
    earlier version pinned the exact name ``_DEFAULT_BUDGET``; renaming it to
    ``_APPLY_CODEC_DEFAULT_BUDGET`` during the leaf extraction made the patch a
    no-op, so the control silently stopped injecting anything and failed. A
    negative control that depends on a name is a control that expires.
    """
    source = re.sub(
        r'kwargs\.get\("budget_ratio",\s*[A-Za-z_][A-Za-z0-9_]*\)',
        'kwargs.get("budget_ratio", 0.8)',
        SELECTOR.read_text(encoding="utf-8"),
    )
    assert 0.8 in _budget_get_defaults(source), (
        "the AST walk does not see a literal budget default, so the guard "
        "above passes vacuously"
    )


def test_the_guard_ignores_literals_that_are_not_budgets() -> None:
    """It must not fire on `ratio=1.0`, or it will be relaxed rather than obeyed."""
    source = 'def f(**kwargs):\n    return CodecResult(success=True, ratio=1.0)\n'
    assert _budget_get_defaults(source) == []


@pytest.mark.parametrize(
    "module_path",
    ["tekmyra/runtime.py", "tekmyra/learned_compressor/compressor.py"],
)
def test_no_module_hardcodes_the_old_operating_point(module_path: str) -> None:
    """0.8 was the stale value at every one of the four sites.

    Not a ban on the number — it is a legitimate ratio elsewhere. This asserts
    it is not sitting next to the word ``budget`` again, which is where every
    copy of it was found.
    """
    source = (REPO_ROOT / module_path).read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in source.splitlines()
        if "budget" in line.lower()
        and "0.8" in line
        and not line.strip().startswith("#")
    ]
    assert not offenders, f"{module_path} reintroduces a budget literal: {offenders}"
