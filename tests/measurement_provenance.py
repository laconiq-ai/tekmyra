"""Provenance for any figure we might quote.

WHY THIS EXISTS. Two seats measured what should have been identical inputs and
got 53.89% against 55.24%, with refusal counts matching exactly. Nobody can say
why, and that is the problem this module addresses -- not by preventing the
discrepancy, which we do not yet understand, but by making the NEXT one
diagnosable in one diff instead of arguable.

WHAT IS ALREADY PINNED, measured at 644c90c rather than assumed:

  * both learned artifacts ARE git-tracked and byte-identical to HEAD
    (``hierarchical_model.pkl`` 48,439 B, ``model.pkl`` 258,047 B, committed at
    caeb6cb on 2026-07-25)
  * 1,611 fixture files are tracked
  * ``uv.lock`` is tracked and pins scikit-learn 1.9.0 / numpy 2.5.1, and the
    venv matches it

So "the artifacts are untracked" is NOT the explanation for the discrepancy,
and pinning them is not the fix -- they were already pinned. What was missing
is any record, attached to a figure, of the state it came from. A number in
prose cannot be checked against anything.

WHAT THIS RECORDS, and each field earns its place by being something that has
already differed or could silently differ between seats:

  commit + dirty     which tree. A dirty tree makes the commit a lie.
  artifact digests   the two learned models, by content not by path
  corpus digests     per corpus, over sorted (relpath, sha256) pairs, so a
                     fixture edited or added changes the digest
  interpreter/libs   the loader, not just the data. A pickle read under a
                     different library version can decode differently.
  invocation         the exact argv, because "the same benchmark" run two ways
                     is two measurements

POLARITY, and it is the rule the rest of the estate now runs on: an unknown is
never recorded as a zero or a blank. If a digest cannot be computed the
manifest says UNKNOWN for that field, and ``require_quotable`` refuses. A
figure carrying a manifest full of UNKNOWNs must not read as a figure carrying
provenance.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_SUBDIR = Path(__file__).resolve().parent.parent
UNKNOWN = "UNKNOWN"

# Learned artifacts, relative to the package subdirectory. Content-addressed in
# the manifest, so a rename shows up as a missing path rather than passing
# silently under a new name.
ARTIFACTS = (
    "benchmarks/learned_compressor/hierarchical_model.pkl",
    "benchmarks/learned_router/model.pkl",
)

CORPORA = (
    "benchmarks/fixtures/synthetic",
    "benchmarks/fixtures/long_context_v1",
    "benchmarks/fixtures/public_domain_v1",
    "benchmarks/fixtures/public_code_v1",
    "benchmarks/fixtures/public_diff_v1",
    "benchmarks/fixtures/public_json_v1",
)

# GITIGNORED, MACHINE-LOCAL, AND LOAD-BEARING. This is the correction that
# matters most in this module's history.
#
# An earlier revision digested only the two tracked ``.pkl`` files, found them
# byte-identical to HEAD, and stamped figures QUOTABLE. That was FALSE
# ASSURANCE: the artifacts serving every disputed figure are these, and
# ``.cache/`` is gitignored at line 6 of .gitignore. Both directories exist on
# the machine this was written on and would be ABSENT in a clean checkout.
#
# What absence does, per ``codec_effectiveness.py``'s own docstring: with the
# prose artifact missing, ``load_compressor`` returns a no-op
# ``ProseCompressor(None)``, the prose codec compresses nothing, and it reads as
# INERT rather than broken. So the same commit measures two materially different
# systems depending on a directory git has never heard of, and the difference is
# invisible in every artifact except the number.
#
# A tool that certifies the exact measurement it exists to catch is worse than
# no tool, because it converts an open question into a settled one.
RUNTIME_ARTIFACT_DIRS = (
    ".cache/compressor/1.0.0",
    ".cache/router",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return UNKNOWN
    return h.hexdigest()


def _git(*args: str) -> str:
    """Run git inside the repo. Returns UNKNOWN rather than raising or guessing.

    NOTE the ``cwd``. This repository's git root is one level ABOVE the package
    directory, so a pathspec or ``git show`` reference written relative to the
    package resolves to nothing and returns empty output. That silently
    compared a real file against an empty blob during this module's own
    investigation and read as "the artifact differs from HEAD". Everything here
    is therefore run with an explicit cwd and repo-relative paths.
    """
    try:
        out = subprocess.run(
            ("git", *args), cwd=REPO_SUBDIR, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN
    return out.stdout.strip()


def _corpus_digest(corpus_rel: str) -> tuple[str, int]:
    """Digest a corpus over sorted (relpath, filehash) pairs.

    Includes the path, so moving a fixture changes the digest; sorted, so
    filesystem ordering cannot change it. Returns (digest, file_count) and the
    count is reported alongside because a digest over ZERO files is a perfectly
    valid-looking hex string -- the empty-collection trap, which is the same
    shape as a test suite exiting 0 having collected nothing.
    """
    root = REPO_SUBDIR / corpus_rel
    if not root.is_dir():
        return (UNKNOWN, 0)
    entries: list[tuple[str, str]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        entries.append((str(p.relative_to(root)), _sha256_file(p)))
    if not entries:
        return (UNKNOWN, 0)
    h = hashlib.sha256()
    for rel, digest in entries:
        h.update(rel.encode())
        h.update(digest.encode())
    return (h.hexdigest(), len(entries))


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": ".".join(str(x) for x in sys.version_info[:3]),
    }
    for name in ("numpy", "sklearn"):
        try:
            mod = __import__(name)
        except ImportError:
            versions[name] = "not-installed"
            continue
        versions[name] = getattr(mod, "__version__", UNKNOWN)
    return versions


@dataclass(frozen=True)
class MeasurementManifest:
    """Everything needed to reproduce, or to explain why two runs differ.

    ``measured_corpus`` and ``budget`` are not decoration — on the evidence they
    are the two fields whose absence caused the dispute this module exists for.
    The private measurement record carries 63.15% on ``long_context_v1`` and
    55.24% on the Congressional Record corpus, and separately records the
    operating point as a budget setting: 0.30/51.38%, 0.15/58.99%,
    0.05/61.15%. So a "disagreement"
    between 63.15% and 58.99% is very likely two different measurements being
    compared as though they were one, and a figure that does not name its corpus
    and budget cannot be told apart from a figure that does not agree.

    ``basis`` is here for the same reason and is the sharpest of the three: on
    the Congressional Record corpus the attempted basis reads 55.24% while the
    survivor basis reads 60.39%. **Basis choice alone moves that number by 5.15
    points — more than the entire disputed gap.** An unnamed basis is therefore
    not a missing detail, it is a five-point degree of freedom.
    """

    commit: str
    tree_dirty: bool | str
    artifacts: dict[str, str]
    corpora: dict[str, dict[str, object]]
    libraries: dict[str, str]
    invocation: list[str] = field(default_factory=list)
    measured_corpus: str = UNKNOWN
    budget: float | str = UNKNOWN
    basis: str = UNKNOWN
    runtime_artifacts: dict[str, dict[str, object]] = field(default_factory=dict)

    def unknowns(self) -> list[str]:
        """Every field that could not be determined. Empty means fully pinned."""
        bad: list[str] = []
        if self.commit == UNKNOWN:
            bad.append("commit")
        if self.tree_dirty == UNKNOWN:
            bad.append("tree_dirty")
        if self.measured_corpus == UNKNOWN:
            bad.append("measured_corpus")
        if self.budget == UNKNOWN:
            bad.append("budget")
        if self.basis == UNKNOWN:
            bad.append("basis")
        # An ABSENT runtime artifact is not a neutral fact about the machine —
        # it changes what was measured. Absence must block quotation, not be
        # recorded as a property of the codec.
        for rel, info in self.runtime_artifacts.items():
            if not info.get("present"):
                bad.append(f"runtime_artifact_ABSENT:{rel}")
            elif info.get("digest") == UNKNOWN:
                bad.append(f"runtime_artifact:{rel}")
        bad += [f"artifact:{k}" for k, v in self.artifacts.items() if v == UNKNOWN]
        bad += [
            f"corpus:{k}" for k, v in self.corpora.items()
            if v.get("digest") == UNKNOWN
        ]
        bad += [f"library:{k}" for k, v in self.libraries.items() if v == UNKNOWN]
        return bad

    def is_quotable(self) -> bool:
        """A figure may be quoted only from a clean tree with no unknowns."""
        return not self.unknowns() and self.tree_dirty is False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def collect(
    invocation: list[str] | None = None,
    *,
    measured_corpus: str = UNKNOWN,
    budget: float | str = UNKNOWN,
    basis: str = UNKNOWN,
) -> MeasurementManifest:
    """Capture the current measurement state.

    The three keyword fields default to UNKNOWN rather than to a plausible
    value. A default of "long_context_v1" or 0.30 would let a caller that never
    considered the question emit a manifest asserting one — which is how a
    figure acquires provenance it has not earned.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    dirty: bool | str = UNKNOWN if status == UNKNOWN else bool(status)

    corpora: dict[str, dict[str, object]] = {}
    for rel in CORPORA:
        digest, count = _corpus_digest(rel)
        corpora[rel] = {"digest": digest, "files": count}

    # Gitignored artifacts. ``tracked`` is recorded explicitly and expected to
    # be False: the point is not that it should be True, but that a figure
    # measured against a directory git cannot see is only reproducible by
    # someone who has that exact directory — so the digest must travel WITH the
    # number, or the number cannot be checked by anyone else.
    runtime: dict[str, dict[str, object]] = {}
    for rel in RUNTIME_ARTIFACT_DIRS:
        digest, count = _corpus_digest(rel)
        present = digest != UNKNOWN
        runtime[rel] = {
            "present": present,
            "digest": digest,
            "files": count,
            "tracked_in_git": _git("ls-files", "--", rel) not in ("", UNKNOWN),
            "note": (
                "gitignored and machine-local; reproduction requires this exact "
                "directory" if present else
                "ABSENT — codecs backed by this artifact are no-ops here, so any "
                "figure from this run describes a different system"
            ),
        }

    return MeasurementManifest(
        commit=commit,
        tree_dirty=dirty,
        artifacts={rel: _sha256_file(REPO_SUBDIR / rel) for rel in ARTIFACTS},
        corpora=corpora,
        libraries=_library_versions(),
        invocation=list(invocation if invocation is not None else sys.argv),
        measured_corpus=measured_corpus,
        budget=budget,
        basis=basis,
        runtime_artifacts=runtime,
    )


class NotQuotable(RuntimeError):
    """Raised when a figure is about to be published without full provenance."""


def require_quotable(manifest: MeasurementManifest) -> None:
    """Gate for anything that will be published, quoted, or shown to counsel.

    Refuses rather than warns. A warning attached to a number is read by whoever
    copies the number, which is nobody.
    """
    if manifest.tree_dirty is True:
        raise NotQuotable(
            f"working tree is DIRTY at {manifest.commit[:12]} — this figure cannot be "
            "reproduced from any commit, so it must not be quoted, published, or "
            "shown to counsel. Commit or stash, then re-measure."
        )
    absent = [m for m in manifest.unknowns() if m.startswith("runtime_artifact_ABSENT:")]
    if absent:
        raise NotQuotable(
            "a load-bearing runtime artifact is ABSENT: "
            f"{', '.join(a.split(':', 1)[1] for a in absent)}. These directories are "
            "gitignored, so a clean checkout does not have them — and without them the "
            "codecs they back become no-ops that read as INERT rather than broken. A "
            "figure from this run describes a different system, not a worse one. Do not "
            "quote it, and do not baseline it: that would record the absence of an "
            "artifact as a property of the codec."
        )
    missing = manifest.unknowns()
    if missing:
        raise NotQuotable(
            "measurement provenance is incomplete, so this figure is not quotable. "
            f"UNKNOWN: {', '.join(missing)}. An UNKNOWN is not a zero and must not be "
            "recorded as one — resolve each before publishing."
        )
