"""Where trained artifacts live, and what to say when they are not there.

THE DEFECT THIS EXISTS TO KILL. Both artifact directories were bare relative
paths — ``Path(".cache/compressor/1.0.0")`` and ``Path(".cache/router")`` —
resolved against the WORKING DIRECTORY. Run the package from anywhere other
than the tree root and the compressor loader found nothing and returned its
no-op, silently, and the benchmark reported a compression ratio for a
compressor that compresses nothing. Measured: 82.9711% reduction from the tree
root, 3083 bytes in and 3083 bytes out from ``/tmp``, no error either way.

For an open-source release that is the first outsider's experience: clone, run
from anywhere, conclude the thing does not work.

**PASSTHROUGH IS FAIL-CLOSED FOR THE SPAN AND FAIL-OPEN FOR THE FIGURE.** The
old docstrings called the no-op fallback "fail-closed", and for fidelity it is
— nothing is dropped. But a number still comes out, at ratio 1.000, and it is
indistinguishable from a measurement. A reader auditing exactly this question
found the words "fail-closed" at exactly the site and moved on. So the fallback
is no longer described that way, and it now records why it happened.

TWO HALVES, AND THE SECOND ONE IS THE POINT

1. RESOLUTION — a documented, ordered search rather than one bare relative path.
2. LEGIBILITY — every candidate location is RECORDED, so a caller can report
   what was actually tried instead of re-deriving a guess of its own.

The second half fixes a defect one layer out: ``run_three_method_comparison.py``
reported the cache directories PRESENT inside the exception explaining the
artifact was missing, because the loader resolved against the working directory
and the diagnostic resolved against the repository root. Neither was wrong about
its own path; there was simply no way to ask the loader where it had looked.
A diagnostic that contradicts the failure it exists to explain is worse than no
diagnostic, because it sends the reader to look for a different bug.

WHY A CWD CANDIDATE SURVIVES AT ALL. Everything that works today works from the
tree root, and freeze week is the wrong week to move a path everyone's muscle
memory depends on. The working directory stays FIRST; the package root is added
BEHIND it. So no existing invocation changes behaviour, and the invocation that
used to fail silently now succeeds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# The directory containing the ``tekmyra`` package — the repository root
# in a checkout or an editable install. Deliberately the LAST candidate: in a
# non-editable install this is site-packages, where nobody keeps a model.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

#: Override the root that artifact-relative paths are joined to. This is the
#: documented escape hatch for a clone that keeps its models elsewhere, and it
#: is what the README tells a first-time cloner to set.
ARTIFACT_ROOT_ENV = "TEKMYRA_ARTIFACT_ROOT"


@dataclass(frozen=True)
class Resolution:
    """Where an artifact directory was looked for, and what was found.

    ``candidates`` is the whole search in order, each with whether it existed,
    so a caller reports the search rather than reconstructing one. ``path`` is
    the first candidate that existed, or ``None``.
    """

    relative: str
    candidates: tuple[tuple[Path, bool], ...] = field(default_factory=tuple)
    path: Path | None = None

    @property
    def found(self) -> bool:
        return self.path is not None

    def fallback(self) -> Path:
        """The path to report when nothing was found.

        The FIRST candidate, not an invented one — an error naming a directory
        the code never actually tried is the same class of lie as the diagnostic
        this module exists to fix.
        """
        return self.candidates[0][0] if self.candidates else Path(self.relative)

    def describe(self) -> str:
        """Every location searched, in order, with what was there."""
        if not self.candidates:
            return f"no candidate locations were searched for {self.relative!r}"
        lines = [
            f"  {'FOUND  ' if exists else 'absent '} {path}"
            for path, exists in self.candidates
        ]
        head = (
            f"resolved {self.relative!r} to {self.path}"
            if self.found
            else f"could not resolve {self.relative!r}; searched, in order:"
        )
        return head + "\n" + "\n".join(lines)


def resolve(
    relative: str | Path,
    explicit: str | Path | None = None,
    *,
    env_var: str = ARTIFACT_ROOT_ENV,
    cwd: Path | None = None,
) -> Resolution:
    """Search for an artifact directory and record every place tried.

    **AN EXPLICIT DIRECTORY IS THE ONLY CANDIDATE.** A caller that names a
    directory means it, and searching on past it is a silent substitution: the
    caller asks for the model in *this* directory, that directory is empty, and
    a DIFFERENT model loads from somewhere else and answers to its name.

    That is not hypothetical — an earlier draft of this function appended the
    defaults after ``explicit`` and four tests turned red saying so. One asked
    for ``/nonexistent`` and expected the no-op; it got a real compressor from
    the package root. Another exported an ONNX router into a temporary
    directory and asserted the loaded backend; it got ``SklearnRouterModel``
    from the repository's own artifacts. **Both would have read as stale tests
    to be updated.** They were the fix reporting a defect worse than the one it
    was written to remove: the old bug lost a model, this one substitutes one.

    Order for the DEFAULT case, and each step's reason:

    1. ``$TEKMYRA_ARTIFACT_ROOT/<relative>`` — the documented override.
    2. ``<cwd>/<relative>`` — what every invocation does today. First of the
       implicit defaults, so nothing that works now changes.
    3. ``<package root>/<relative>`` — the fix. A clone run from anywhere finds
       the models sitting beside the package.

    This never raises and never creates anything. It answers where, and the
    caller decides what a miss means — the loader may degrade, a measurement
    path must refuse, and only the caller knows which it is.
    """
    relative = Path(relative)
    base = Path(cwd) if cwd is not None else Path.cwd()

    if explicit is not None:
        candidate = Path(explicit)
        exists = candidate.is_dir()
        return Resolution(
            relative=relative.as_posix(),
            candidates=((candidate, exists),),
            path=candidate if exists else None,
        )

    ordered: list[Path] = []
    root_override = os.environ.get(env_var)
    if root_override:
        ordered.append(Path(root_override) / relative)
    ordered.append(base / relative)
    ordered.append(PACKAGE_ROOT / relative)

    # DEDUPE, because an ALREADY-ABSOLUTE `relative` collapses every candidate
    # onto the same path (pathlib's `/` discards the left operand), and a
    # search that lists one directory three times reads as three checks.
    seen: list[tuple[Path, bool]] = []
    chosen: Path | None = None
    recorded: set[Path] = set()
    for candidate in ordered:
        if candidate in recorded:
            continue
        recorded.add(candidate)
        exists = candidate.is_dir()
        seen.append((candidate, exists))
        if exists and chosen is None:
            chosen = candidate

    return Resolution(
        relative=relative.as_posix(),
        candidates=tuple(seen),
        path=chosen,
    )
