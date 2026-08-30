"""Router provenance capture for benchmark reports.

A benchmark number is only a claim about the product if the report says which
router produced it. Without that, a rules-only run and a learned-router run are
indistinguishable in the artifact, and any GO recorded from one cannot be
defended as the other.

This module answers, for a given :class:`LearnedRouter`:

* did the configured learned artifact actually load, and as which backend;
* where the artifact was resolved from, and whether that path ships in the repo;
* the sha256 of every file in the artifact directory.

It is read-only and makes no network calls.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tekmyra import artifact_paths

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tekmyra.learned_router.router import LearnedRouter

_DIGEST_CHUNK = 1 << 20


class RouterArtifactUnavailable(RuntimeError):
    """The configured learned artifact is missing or failed to load.

    Raised where a benchmark would otherwise proceed rules-only and report a
    headline number that reads as a learned-router result.

    IT CARRIES THE TWO CASES APART. "Missing" and "failed to load" are
    different situations with different remedies — download the artifact, or
    work out why the one you have will not open — and a caller that has to
    regex the message to tell them apart will eventually get it wrong. The
    raise site already formats both into its text; these attributes make the
    same facts readable without parsing prose. `artifact_dir_exists` is None
    when nothing determined it, and None must not be read as False.
    """

    def __init__(self, message: str, *, artifact_dir: Any = None,
                 artifact_dir_exists: bool | None = None,
                 backend: Any = None) -> None:
        super().__init__(message)
        self.artifact_dir = artifact_dir
        self.artifact_dir_exists = artifact_dir_exists
        self.backend = backend


def file_digest(path: Path) -> str:
    """Return the sha256 of ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_DIGEST_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_is_tracked(path: Path) -> bool | None:
    """Whether ``path`` is tracked by git.

    Returns ``None`` when git cannot answer (not a repo, git absent) rather than
    guessing — an unknown answer must not read as "ships".
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.returncode == 0


def capture(router: "LearnedRouter") -> dict[str, Any]:
    """Capture the provenance of the artifact ``router`` is running on."""
    config = router.config
    # ONE RESOLUTION, SHARED. This used to resolve the relative artifact path
    # against this process's working directory, independently of every other
    # site that did the same — so a run from the wrong directory reported a
    # missing artifact while some other site reported it present.
    resolution = config.resolved_artifact_dir()
    artifact_dir = resolution.path or resolution.fallback()
    model = getattr(router, "_model", None)

    files: list[dict[str, Any]] = []
    if artifact_dir.is_dir():
        for entry in sorted(artifact_dir.iterdir()):
            if not entry.is_file():
                continue
            files.append(
                {
                    "name": entry.name,
                    "bytes": entry.stat().st_size,
                    "sha256": file_digest(entry),
                    "tracked_in_git": path_is_tracked(entry),
                }
            )

    tracked_flags = [f["tracked_in_git"] for f in files]
    if not files:
        ships = False
    elif any(flag is None for flag in tracked_flags):
        ships = None
    else:
        ships = all(bool(flag) for flag in tracked_flags)

    return {
        "model_loaded": router.model_loaded,
        "model_version": router.model_version,
        "model_backend": type(model).__name__ if model is not None else "none",
        "artifact_dir": str(artifact_dir),
        "artifact_dir_exists": artifact_dir.is_dir(),
        # THE WHOLE SEARCH, so `require_loaded`'s refusal can name every place
        # that was tried. A refusal saying only "artifact_dir=X exists=False"
        # sends the reader to check X, which may not be the directory they
        # meant — and if some other component resolved a different root, the
        # two messages contradict each other about the same missing model.
        "artifact_search": resolution.describe(),
        "artifact_ships_in_repo": ships,
        "calibration_temperature": config.temperature,
        "use_onnx_if_available": config.use_onnx_if_available,
        "artifact_files": files,
    }


def capture_anchor_store(store: object) -> dict[str, Any]:
    """Record WHICH BUILD produced a number — open core or commercial.

    Both builds run the same compression path and, on a corpus without
    anchorized drops, produce the identical headline ratio. So a report that
    omits this cannot distinguish them, and a reader has no way to tell whether
    the number came from the half we publish or the half we sell.

    That is not hypothetical: the first number this project produced was a
    compressor figure reported in a way that let it be read as a router figure,
    and the resulting ambiguity is why the 2026-07-25 GO is marked NOT-DEFENDED.
    An unlabelled number will be misread.
    """
    from tekmyra.anchor_interface import NonResolvingAnchorStore

    is_open = isinstance(store, NonResolvingAnchorStore)
    return {
        "anchor_store": type(store).__name__,
        "anchor_store_module": type(store).__module__,
        "build": "open-core" if is_open else "commercial",
        "anchor_resolution_available": not is_open,
        "note": (
            "the open core cannot resolve anchors, so it REFUSES dropped "
            "protected spans the commercial build may accept as recoverable"
            if is_open
            else "commercial build: anchor resolution available"
        ),
    }


def require_loaded(provenance: dict[str, Any]) -> None:
    """Refuse to proceed when the learned artifact did not load.

    Vega's blocking requirement (2026-08-07): a missing or unloadable artifact
    must refuse to run, not abstain silently and report a headline ratio.
    """
    if provenance.get("model_loaded"):
        return
    # THE SEARCH GOES IN THE REFUSAL. Naming one directory left a reader
    # checking a path the loader may never have opened, and it is how a
    # diagnostic elsewhere came to report the cache directories PRESENT inside
    # the exception explaining they were missing. The message now states every
    # location tried, in order, with what was there.
    search = provenance.get("artifact_search")
    raise RouterArtifactUnavailable(
        "Learned router artifact did not load — refusing to produce a benchmark "
        f"number. artifact_dir={provenance.get('artifact_dir')!r} "
        f"exists={provenance.get('artifact_dir_exists')} "
        f"backend={provenance.get('model_backend')!r}. "
        + (f"\n{search}\n" if search else "")
        + f"Set ${artifact_paths.ARTIFACT_ROOT_ENV} to the directory holding "
        ".cache/ if your artifacts live elsewhere. "
        "Re-run with --allow-rules-only to record an explicitly rules-only run; "
        "such a run cannot emit a headline compression ratio.",
        artifact_dir=provenance.get("artifact_dir"),
        artifact_dir_exists=provenance.get("artifact_dir_exists"),
        backend=provenance.get("model_backend"),
    )
