"""The README's headline strings, bound to the evidence they quote.

THE PROSE HAD NO RATCHET. The red-team pass falsified three README numbers
(0.7409 to 0.7500, 62.1458 to 63.0000, 704/704 to 700/700) and the full suite
plus ``verify_open_core.py`` stayed green: every ratchet bound the baseline
JSON and the live run to each other, and nothing bound the prose to either.
Doc drift is exactly how the paper briefly carried a scikit-learn claim the
README had already corrected. This file closes the class: the public-corpus
sentences must match ``public_corpus_baseline.json``, and the benchmark-table
strings are pinned so they can only change when someone decides they should.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_SUBDIR = Path(__file__).resolve().parent.parent
README = REPO_SUBDIR / "open_core_files" / "README.md"
BASELINE = Path(__file__).resolve().parent / "baselines" / "public_corpus_baseline.json"


def _readme_text() -> str:
    if README.is_file():
        return README.read_text(encoding="utf-8")
    # In the staged payload the decided files sit at the repository root.
    return (REPO_SUBDIR / "README.md").read_text(encoding="utf-8")


def test_public_corpus_prose_matches_the_baseline() -> None:
    text = _readme_text()
    totals = json.loads(BASELINE.read_text())["measurement"]["totals"]
    expected = [
        f"{totals['HEADLINE_effective_reduction_pct']:.4f}%",
        f"{totals['DIAGNOSTIC_on_accepted_reduction_pct']:.4f}%",
        # The count is bold-wrapped in prose ("over **138"), so match the
        # number with its markdown emphasis rather than a spacing guess.
        f"**{totals['fixtures']}",
        f"{totals['input_bytes']:,}",
        f"{totals['output_bytes']:,}",
        f"{totals['refused_fixtures']} refusals",
    ]
    missing = [s for s in expected if s not in text]
    assert not missing, (
        f"README no longer quotes the baseline it cites; missing {missing}. "
        "Either the prose drifted (fix the prose) or the baseline was "
        "deliberately regenerated (update the prose in the same change)."
    )


def test_benchmark_table_strings_are_pinned() -> None:
    """The ratio/token table, pinned as strings.

    These values are bound to the reference bundle by the live benchmark and
    the ratchet suite; this pin binds the PROSE to them, so a stray edit to
    the README cannot ship a table the benchmark contradicts.
    """
    text = _readme_text()
    pinned = ["0.7409", "+25.86%", "68/68", "0.2929", "+48.44%", "+70.40%",
              "704/704", "14 refused", "40 eligible of 40 shipped (26 reached"]
    missing = [s for s in pinned if s not in text]
    assert not missing, (
        f"README benchmark table lost {missing}. Change it only alongside a "
        "deliberate re-measurement, never alone."
    )
