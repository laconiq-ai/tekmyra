"""The repeated-entity retention contract, shared by producers and the verifier.

``verify()`` fails an artifact when a frequently repeated entity is
near-eliminated: retention must not fall below a proportion of the output's own
length ratio. THE COMPRESSOR WAS NEVER TAUGHT THAT CONTRACT. ``SentenceFeatures``
carried length, position, numeric/citation/agent-state flags and locked-span
membership — no repetition signal and no entity signal at all — so the model had
no representation of "this sentence carries a frequently repeated entity" and
dropped those sentences unless they happened to contain a numeric.

Measured on long_context_v1 at budget 0.8, where the document keeps four fifths
of its length: "Platform Reliability Team" 110 mentions -> 2, "SEV-2" 109 -> 2,
"TICKET-48291" 49 -> 2. That is 1.8% retention against a 40% floor. The check is
right and the compressor is wrong.

THIS MODULE EXISTS FOR EXACTLY THE REASON ``numeric_fidelity`` DOES: a producer
that re-expressed the definition with its own regex would drift from the check it
must satisfy. The numeric contract got its shared leaf; the entity contract never
did, and the gap between them is the whole defect.

PROPORTIONAL, NOT CATEGORICAL — the design constraint that decides whether this
works. A keep signal that preserved EVERY sentence mentioning a repeated entity
would let the compressor satisfy the check by not compressing: a remedy that
defeats the thing it protects. ``RETENTION_FACTOR`` is the one constant both
sides use, so the share the featuriser preserves and the share the verifier
demands cannot drift apart.

A leaf: it imports only ``sentence_boundaries``, itself a zero-dependency module
already inside the trainer's provenance surface. Importing these helpers from
``verifier`` instead would drag the verifier and ``anchor_store`` into the
compressor's training surface — the mistake this package already made once.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

from tekmyra.sentence_boundaries import ABBREVIATIONS

# Capitalized runs of 1-3 tokens ("Dr. Aris Thorne", "Platform Reliability
# Team") — the crude but deterministic proxy for repeated named entities.
ENTITY_RE = re.compile(r"\b[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,2}\b")

# Only entities repeated at least this often are guarded; below it, the
# presence-based locked-span machinery is the right tool.
MIN_OCCURRENCES = 8

# An entity's retention may fall to this fraction of the output's own length
# ratio before the guard fires. ONE CONSTANT, BOTH SIDES: the verifier's floor
# and the share a producer must preserve are the same number by construction.
RETENTION_FACTOR = 0.5

# Bound on the retention top-up loop. See `top_up_retention`.
MAX_TOPUP_PASSES = 8

# Capitalization alone is not a proper-noun signal. Candidates are dropped when
# any token is fully uppercase (log levels, HTTP verbs, keywords: INFO/ERROR/
# GET/TODO — the normal vocabulary of log and code content, which a summarizing
# compressor legitimately collapses), or when every token is a common
# capitalized function word. Single-token candidates additionally count only
# their occurrences NOT at a sentence start, since sentence position already
# explains that capitalization.
STOPWORDS = frozenset(
    """
    a an the this that these those it its he she they we you i his her our
    your their my me him them us who whom whose which what why how when
    where if then else so as to of by at in on for from with and or but
    nor not no yes is are was were be been am do does did has have had can
    could will would shall should may might must there here now also after
    before during since until because however meanwhile finally first
    next last one two three
    """.split()
)

_ALLCAPS_TOKEN_RE = re.compile(r"[A-Z0-9]{2,}$")
_RUN_TOKEN_RE = re.compile(r"\S+")


def truncate_at_sentence_end(run: str) -> str:
    """Cut a capitalized run at the first sentence-terminal token.

    The entity regex joins tokens across any whitespace, so "Thank You. Issue"
    matches as one three-token run spanning a sentence boundary. A token ending
    in .!? closes the run — unless its trailing period belongs to a shared-set
    abbreviation ("Dr. Aris Thorne" stays whole).
    """
    for match in _RUN_TOKEN_RE.finditer(run):
        token = match.group(0)
        if token.endswith(("!", "?")):
            return run[: match.end()]
        if token.endswith(".") and token[:-1].lower() not in ABBREVIATIONS:
            return run[: match.end()]
    return run


def content_tokens(tokens: list[str]) -> int:
    """Count tokens that carry proper-noun evidence (0 = not a candidate).

    Fully uppercase tokens disqualify the whole run (structured vocabulary,
    never a name). Capitalized function words contribute nothing on their own:
    "Thank You" has ONE content token and must be treated with the same
    sentence-position skepticism as a lone capitalized word, or every courtesy
    sign-off in a transcript becomes a guarded "entity". Two or more content
    tokens ("Aris Thorne") are strong evidence at any position. Abbreviation
    stems are titles, not names: a bare vocative "Dr." used eight times is
    address, not an entity.
    """
    stems = [token.rstrip("'.-").lower() for token in tokens]
    if any(_ALLCAPS_TOKEN_RE.fullmatch(token.rstrip("'.-")) for token in tokens):
        return 0
    return sum(1 for stem in stems if stem not in STOPWORDS and stem not in ABBREVIATIONS)


def sentence_initial(text: str, start: int) -> bool:
    """True when position ``start`` begins the text or follows .!?/newline."""
    cursor = start - 1
    while cursor >= 0 and text[cursor] in " \t":
        cursor -= 1
    return cursor < 0 or text[cursor] in ".!?\n"


def iter_entity_mentions(text: str):
    """Yield ``(entity, start, end)`` for every qualifying entity mention.

    One definition of "entity mention", used by the verifier that enforces
    retention and by the featuriser that has to earn it.
    """
    for match in ENTITY_RE.finditer(text):
        entity = truncate_at_sentence_end(match.group(0))
        content = content_tokens(entity.split())
        if content == 0:
            continue
        if content == 1 and sentence_initial(text, match.start()):
            # Capitalization explained by sentence position proves nothing.
            continue
        yield entity, match.start(), match.start() + len(entity)


def count_entities(text: str) -> Counter[str]:
    """Return mention counts per entity."""
    counts: Counter[str] = Counter()
    for entity, _, _ in iter_entity_mentions(text):
        counts[entity] += 1
    return counts


def repeated_entities(text: str, min_occurrences: int = MIN_OCCURRENCES) -> set[str]:
    """Entities mentioned often enough to carry a retention obligation."""
    return {
        entity
        for entity, count in count_entities(text).items()
        if count >= min_occurrences
    }


def retention_shortfalls(
    original: str, output: str, min_occurrences: int = MIN_OCCURRENCES
) -> dict[str, int]:
    """Entities under-retained in ``output``, and how many mentions each still needs.

    THE PRODUCER CANNOT KNOW ITS FLOOR BEFORE IT SELECTS — that is circular, so
    it must correct AFTER. The selector allocates against the budget it aimed
    at; ``verify()`` grades against the ratio it actually achieved, and a codec
    that falls short of its own budget is therefore graded to a HIGHER floor
    than it allocated for. Measured on lc-issue_threads-calibration-001: the
    producer delivered "Platform Reliability Team" at retention 0.200 against a
    graded floor of 0.2093 — short by a single mention, and the whole fixture
    fell back to passthrough at zero saving.

    The comment that used to sit over the selector's floor said the two "cannot
    drift" because the arithmetic is one definition. The arithmetic was one
    definition; the INPUTS were two, and asserting the invariant is where
    everyone stopped checking it. This function closes it by measurement rather
    than by assertion (Vega rulings 46 and 47).

    DELIBERATELY CONSERVATIVE ON TWO AXES, because a producer that
    over-estimates its obligation ships a slightly larger artifact while one
    that under-estimates ships a REFUSAL:

    * ``count`` by substring, matching what the check counts in the output,
      rather than the mention iterator used to find candidates.
    * no credit for secrecy-masked occurrences. ``verify()`` discounts mentions
      destroyed by redaction or anchorization, which only ever LOWERS the
      denominator it grades against; ignoring that here asks for more retention
      than strictly required, never less.
    """
    ratio = len(output) / max(1, len(original))
    floor = retention_floor(ratio)
    shortfalls: dict[str, int] = {}
    for entity, total in count_entities(original).items():
        if total < min_occurrences:
            continue
        needed = math.ceil(total * floor)
        kept = output.count(entity)
        if kept < needed:
            shortfalls[entity] = needed - kept
    return shortfalls


def top_up_retention(
    original: str,
    segments: list[str],
    selected: set[int],
    build: Callable[[set[int]], str],
    max_passes: int = MAX_TOPUP_PASSES,
) -> tuple[set[int], str]:
    """Restore segments until no repeated entity is under-retained.

    ONE COPY, TWO CODECS. Both built codecs face the identical problem and would
    otherwise each grow their own loop — and two implementations of a contract
    are exactly what ``entity_fidelity`` and ``numeric_fidelity`` exist to
    prevent. They differ only in how an output is assembled from its parts, so
    that is the callback and everything else is shared.

    Adding a segment lengthens the output, which raises the ratio and so raises
    the floor being chased. It converges quickly — a segment contributes far
    more to one entity's retention than its own characters contribute to the
    document ratio — but it is BOUNDED rather than trusted, because a
    producer-side repair loop that fails to converge hangs the pipeline, and a
    hang is strictly worse than the refusal it was trying to avoid.

    Returns the updated selection and the rebuilt output. Stops early when no
    unselected segment carries the missing mentions: that obligation cannot be
    met from this document, and spinning would not discover otherwise.
    """
    output = build(selected)
    for _ in range(max_passes):
        shortfalls = retention_shortfalls(original, output)
        if not shortfalls:
            break
        added = False
        for index, segment in enumerate(segments):
            if not shortfalls:
                break
            if index in selected:
                continue
            helps = {
                entity: segment.count(entity)
                for entity in shortfalls
                if entity in segment
            }
            if not helps:
                continue
            selected.add(index)
            added = True
            for entity, gained in helps.items():
                remaining = shortfalls[entity] - gained
                if remaining <= 0:
                    shortfalls.pop(entity)
                else:
                    shortfalls[entity] = remaining
        if not added:
            break
        output = build(selected)
    return selected, output


def retention_floor(length_ratio: float) -> float:
    """The share of mentions that must survive, given the output's length ratio.

    Capped at full retention: a ratio above 1.0 means the surviving material
    grew, and demanding MORE than every occurrence back is a bar no correct
    output can clear.

    THE VERIFIER'S FLOOR AND THE PRODUCER'S TARGET ARE THIS FUNCTION. If a
    producer preserves this share, the check it is graded against passes by
    construction rather than by luck.
    """
    return min(1.0, length_ratio * RETENTION_FACTOR)
