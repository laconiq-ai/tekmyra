"""What each codec actually DOES, measured per (route, content type).

The RouteClass ratchet asks whether a codec EXISTS. It cannot see a codec that
exists and does nothing, because ``apply_codec`` returns ``success=True`` with
the input unchanged and that is indistinguishable from a real compression that
happened to save nothing.

This module is the shared definition of "effective", used by both the guard in
``tests/test_codec_effectiveness_ratchet.py`` and the script that regenerates
its baseline — one definition, so the baseline cannot drift from the check that
reads it. That is the same shared-leaf discipline as ``entity_fidelity.py``.

THE UNIT IS (ROUTE, CONTENT TYPE), NOT ROUTE. Measured at 9122dab,
``log_compressor`` looks effective at route granularity (mean ratio 0.999)
purely because eight ``stacktrace`` and ``mixed`` fixtures reach it and
compress slightly. On its own ``log`` content it returns the input unchanged on
15 of 15. A route-level aggregate would have reported that codec as working.

THREE OUTCOMES, and the difference between the last two is the whole point:

``refused``    ``success=False``. The route declines. Already owned by the
               RouteClass ratchet, recorded here only so the buckets are
               exhaustive — this module never rules on it.
``inert``      ``success=True`` on every fixture and never a byte shorter. The
               codec ran and did nothing while reporting that it worked.
``effective``  ``success=True`` and at least one fixture came back shorter.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tekmyra.codecs.selector import apply_codec
from tekmyra.contracts import PolicyTier, SourceChannel
from tekmyra.fixtures import load_fixtures
from tekmyra.learned_compressor.compressor import load_compressor
from tekmyra.prober import probe_text
from tekmyra.rule_router import route_deterministic

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPORA = (
    REPO_ROOT / "benchmarks" / "fixtures" / "synthetic",
    REPO_ROOT / "benchmarks" / "fixtures" / "long_context_v1",
)


@dataclass
class Cell:
    """One (route, content_type) pair, as measured."""

    route: str
    content_type: str
    fixtures: int = 0
    refused: int = 0
    compressed: int = 0
    ratios: list[float] = field(default_factory=list)
    # RATIO 1.0 IS NOT "RETURNED ITS INPUT", AND THIS GUARD TREATED THEM AS THE
    # SAME THING UNTIL 2026-08-09.
    #
    # An independent QA seat found a go file, misdetected as python, returning
    # success=True and ratio=1.0 while NOT being byte-identical to its input —
    # the python tokenizer round-trip had silently turned a tab inside a `//`
    # comment into a space. A fidelity claim failing quietly on a route whose
    # contract is that it does not mutate content.
    #
    # This guard could not have seen it. It classified by length alone, so a
    # codec that mutates content to the SAME LENGTH reads as "inert" — the
    # guard written to catch codecs that do nothing is blind to a codec that
    # does something wrong. Counted separately now: identical output is a
    # different fact from an unchanged length, and only one of them is safe.
    returned_input_unchanged: int = 0
    mutated_without_compressing: int = 0

    @property
    def status(self) -> str:
        if self.ratios == []:
            return "refused"
        return "effective" if self.compressed else "inert"

    @property
    def best_ratio(self) -> float:
        return round(min(self.ratios), 4) if self.ratios else 1.0

    @property
    def mean_ratio(self) -> float:
        return round(sum(self.ratios) / len(self.ratios), 4) if self.ratios else 1.0

    @property
    def key(self) -> str:
        return f"{self.route}/{self.content_type}"


def compressor_artifact_loaded() -> bool:
    """Is a trained prose compressor actually present?

    ``load_compressor`` returns a no-op ``ProseCompressor(None)`` when the
    artifact is missing, and ``.cache/compressor`` is gitignored — so in a
    clean checkout the prose codec compresses nothing and reads as INERT. A
    guard that baselined that would record the absence of an artifact as a
    property of the codec. Callers must refuse to classify without it.
    """
    return load_compressor()._model is not None


def measure() -> dict[str, Cell]:
    """Route every fixture the way production does, then apply its codec."""
    cells: dict[str, Cell] = {}
    grouped: dict[tuple[str, str], Cell] = {}

    for corpus in CORPORA:
        if not corpus.is_dir():
            continue
        for fixture in load_fixtures(corpus):
            text = fixture.text if hasattr(fixture, "text") else fixture.content
            if not text:
                continue
            content_type = getattr(fixture, "content_type", None)
            content_type = str(getattr(content_type, "value", content_type))

            features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
            record = route_deterministic(fixture.fixture_id, features)
            route = record.route_after_override or record.fallback_route
            result = apply_codec(text, route, features)

            cell = grouped.setdefault(
                (route.value, content_type), Cell(route.value, content_type)
            )
            cell.fixtures += 1
            if not result.success:
                cell.refused += 1
                continue
            output = result.output or ""
            ratio = len(output) / len(text)
            cell.ratios.append(ratio)
            if ratio < 1.0:
                cell.compressed += 1
            elif output == text:
                cell.returned_input_unchanged += 1
            else:
                # Same length, different bytes: the codec CHANGED the
                # content without compressing it. Silent mutation.
                cell.mutated_without_compressing += 1

    for cell in grouped.values():
        cells[cell.key] = cell
    return cells


def as_baseline(cells: dict[str, Cell]) -> dict:
    """Render the measurement in the shape the baseline file stores."""
    out: dict[str, dict] = defaultdict(dict)
    for key, cell in sorted(cells.items()):
        out[cell.status][key] = {
            "fixtures": cell.fixtures,
            "best_ratio": cell.best_ratio,
            "mean_ratio": cell.mean_ratio,
            "reduction_pct": round(100 * (1 - cell.mean_ratio), 2),
            "returned_input_unchanged": cell.returned_input_unchanged,
            "mutated_without_compressing": cell.mutated_without_compressing,
        }
    return dict(out)
