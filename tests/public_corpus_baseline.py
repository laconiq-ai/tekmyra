"""The public-corpus measurement: do the figures hold on text we did not write?

Four external corpora hold 138 fixtures and, until this module, NO committed
baseline measured any of them. Standard benchmark datasets would have been
layered on top of that gap.

Shares ``codec_effectiveness``'s discipline but answers a different question and
so keeps its own definition. That module measures whether a codec DOES ANYTHING,
per (route, content type), in CHARACTER ratios at the default budget, and its
baseline says in terms that its numbers must never be quoted as the published
figures. This one measures the RATE, on public text, in the denominators Vega
ruled — and is therefore the path that could produce a quotable number, which is
why every rule below is load-bearing rather than stylistic.

DENOMINATORS, RULED BEFORE MEASURING so the result cannot be chosen after seeing
it:

* **BYTES, never fixture counts.** 58% of items and 58% of bytes are different
  numbers, and a count-weighted acceptance rate printed beside a byte ratio is
  two denominators in one sentence.
* **The headline is the EFFECTIVE rate over the WHOLE corpus, refusals
  contributing zero saving.** A refusal is part of the result, not an exclusion
  from it — a codec that refuses 90% of its input and compresses the rest
  brilliantly has not achieved the on-accepted figure.
* **The on-accepted ratio is a DIAGNOSTIC published beside the headline, never
  the headline.** It is the more flattering number and it answers a narrower
  question.

AND THE CONTROL THAT MAKES A ZERO MEANINGFUL. Before any ratio here is quotable
the harness must show a NON-ZERO result on an independently known case.
Otherwise "0.00%, and that is discipline" is indistinguishable from a broken
codec — and the log codec, which legitimately reports 0.00%, is exactly where
that ambiguity would land.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tekmyra.codecs.selector import apply_codec
from tekmyra.contracts import PolicyTier, SourceChannel
from tekmyra.fixtures import load_fixtures
from tekmyra.prober import probe_text
from tekmyra.rule_router import route_deterministic

REPO_SUBDIR = Path(__file__).resolve().parent.parent

PUBLIC_CORPORA = (
    "benchmarks/fixtures/public_domain_v1",
    "benchmarks/fixtures/public_code_v1",
    "benchmarks/fixtures/public_diff_v1",
    "benchmarks/fixtures/public_json_v1",
)


@dataclass
class Cell:
    """One (route, content_type) pair, measured in bytes."""

    route: str
    content_type: str
    fixtures: int = 0
    refused: int = 0
    # Byte totals over the WHOLE cell. Refused fixtures contribute their input
    # bytes to both totals, i.e. zero saving — that is what makes the headline
    # an effective rate rather than an on-accepted one.
    input_bytes_all: int = 0
    output_bytes_all: int = 0
    # The diagnostic: accepted fixtures only.
    input_bytes_accepted: int = 0
    output_bytes_accepted: int = 0
    per_fixture_ratios: list[float] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.route}/{self.content_type}"

    @property
    def effective_reduction_pct(self) -> float:
        """THE HEADLINE. Bytes saved over every byte offered to this cell."""
        if not self.input_bytes_all:
            return 0.0
        return round(100 * (1 - self.output_bytes_all / self.input_bytes_all), 4)

    @property
    def on_accepted_reduction_pct(self) -> float:
        """DIAGNOSTIC ONLY. Never the headline; it excludes refusals."""
        if not self.input_bytes_accepted:
            return 0.0
        return round(100 * (1 - self.output_bytes_accepted / self.input_bytes_accepted), 4)

    @property
    def refusal_rate_bytes_pct(self) -> float:
        """Refusals weighted by BYTES, matching the headline's denominator."""
        if not self.input_bytes_all:
            return 0.0
        refused_bytes = self.input_bytes_all - self.input_bytes_accepted
        return round(100 * refused_bytes / self.input_bytes_all, 4)


def measure(corpora: tuple[str, ...] = PUBLIC_CORPORA) -> tuple[dict[str, Cell], dict[str, int]]:
    """Route every fixture the way production does, then apply its codec.

    Returns (cells, corpus_file_counts). The counts travel with the result
    because a measurement over zero fixtures produces a perfectly well-formed
    table of zeros — the empty-collection trap, and the reason a file count
    belongs beside every digest rather than only inside the code.
    """
    grouped: dict[tuple[str, str], Cell] = {}
    counts: dict[str, int] = {}

    for corpus_rel in corpora:
        corpus = REPO_SUBDIR / corpus_rel
        seen = 0
        if not corpus.is_dir():
            counts[corpus_rel] = 0
            continue
        for fixture in load_fixtures(corpus):
            text = fixture.text if hasattr(fixture, "text") else fixture.content
            if not text:
                continue
            seen += 1
            content_type = getattr(fixture, "content_type", None)
            content_type = str(getattr(content_type, "value", content_type))

            features = probe_text(text, SourceChannel.API, PolicyTier.STANDARD)
            record = route_deterministic(fixture.fixture_id, features)
            route = record.route_after_override or record.fallback_route
            result = apply_codec(text, route, features)

            in_bytes = len(text.encode("utf-8"))
            cell = grouped.setdefault(
                (route.value, content_type), Cell(route.value, content_type)
            )
            cell.fixtures += 1
            cell.input_bytes_all += in_bytes

            if not result.success:
                cell.refused += 1
                # Refused: zero saving, counted in the headline denominator.
                cell.output_bytes_all += in_bytes
                continue

            out_bytes = len((result.output or "").encode("utf-8"))
            cell.output_bytes_all += out_bytes
            cell.input_bytes_accepted += in_bytes
            cell.output_bytes_accepted += out_bytes
            cell.per_fixture_ratios.append(out_bytes / in_bytes if in_bytes else 1.0)

        counts[corpus_rel] = seen

    return ({c.key: c for c in grouped.values()}, counts)


def corpus_totals(cells: dict[str, Cell]) -> dict[str, float | int]:
    """Whole-corpus rollup, in the ruled denominators."""
    in_all = sum(c.input_bytes_all for c in cells.values())
    out_all = sum(c.output_bytes_all for c in cells.values())
    in_acc = sum(c.input_bytes_accepted for c in cells.values())
    out_acc = sum(c.output_bytes_accepted for c in cells.values())
    fixtures = sum(c.fixtures for c in cells.values())
    refused = sum(c.refused for c in cells.values())
    return {
        "fixtures": fixtures,
        "refused_fixtures": refused,
        "input_bytes": in_all,
        "output_bytes": out_all,
        "HEADLINE_effective_reduction_pct": (
            round(100 * (1 - out_all / in_all), 4) if in_all else 0.0
        ),
        "DIAGNOSTIC_on_accepted_reduction_pct": (
            round(100 * (1 - out_acc / in_acc), 4) if in_acc else 0.0
        ),
        "refusal_rate_bytes_pct": (
            round(100 * (in_all - in_acc) / in_all, 4) if in_all else 0.0
        ),
        "refusal_rate_fixtures_pct": (
            round(100 * refused / fixtures, 4) if fixtures else 0.0
        ),
    }


class NoNonZeroControl(RuntimeError):
    """The harness never demonstrated it can measure a saving at all."""


def assert_non_zero_control(cells: dict[str, Cell]) -> str:
    """Prove the instrument can register compression before any zero is believed.

    Without this, a table of 0.00% is indistinguishable from a codec that is
    broken, a corpus that failed to load, or a measurement wired to the wrong
    field. The control is deliberately taken from the SAME run rather than a
    fixture written to pass: at least one cell must show a real byte saving.
    """
    winners = [
        (k, c.effective_reduction_pct) for k, c in cells.items()
        if c.effective_reduction_pct > 0
    ]
    if not winners:
        raise NoNonZeroControl(
            "no cell showed ANY byte saving across the whole public corpus. Before "
            "reporting 0.00% as discipline, rule out a broken codec, an empty corpus "
            "and a mis-wired measurement — a harness that has never registered a "
            "saving cannot distinguish those from a true zero."
        )
    best = max(winners, key=lambda kv: kv[1])
    return f"{best[0]} at {best[1]}% effective byte reduction"


def as_baseline(cells: dict[str, Cell], counts: dict[str, int]) -> dict:
    """Render for the committed baseline file."""
    per_cell: dict[str, dict] = {}
    for key, c in sorted(cells.items()):
        per_cell[key] = {
            "fixtures": c.fixtures,
            "refused": c.refused,
            "input_bytes": c.input_bytes_all,
            "output_bytes": c.output_bytes_all,
            "HEADLINE_effective_reduction_pct": c.effective_reduction_pct,
            "DIAGNOSTIC_on_accepted_reduction_pct": c.on_accepted_reduction_pct,
            "refusal_rate_bytes_pct": c.refusal_rate_bytes_pct,
        }
    return {
        "corpus_file_counts": counts,
        "totals": corpus_totals(cells),
        "by_route_and_content_type": per_cell,
    }
