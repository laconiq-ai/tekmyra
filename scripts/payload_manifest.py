#!/usr/bin/env python3
"""One manifest governing the publication payload, and the check that proves it.

BUILD AND VERIFY ARE ONE FILE AND SHARE ONE DERIVATION, for the reason
``stage_open_core`` and ``verify_open_core`` share ``TreeCategory.matches``: two
mechanisms that must agree are made one mechanism, because agreement maintained
by discipline is a scheduled divergence.

WHY "VERIFIED AGAINST GIT ARCHIVE OF A STATED COMMIT" IS NOT THE WHOLE STORY.
The staged tree is **not** a subset of any git tree. Measured on the tree at
c645049: 425 files, of which

    420   byte-identical to a blob at the stated commit
      1   pyproject.toml — EXISTS at that commit and DIFFERS from it, because
          the staging script generates it from the allowlisted modules' own
          imports rather than copying the closed project's manifest
      4   OUTSTANDING/*.OUTSTANDING.md — written by the staging script; no
          blob exists at any commit

A manifest that verified path-existence at the commit would pass pyproject.toml
while its bytes differ. That is the failure this file is built to make
impossible: **git provenance is asserted per file only after a byte comparison
against the blob**, never from the path.

THE GENERATED FILES ARE NOT "TRUST ME" EITHER. They are deterministic given the
script and the commit, so ``verify --rederive`` re-runs ``stage_open_core.py``
into a temporary directory and compares. Every one of the 425 files is then
re-derivable by someone who has only this repository: 420 from git, 5 from a
script at a stated commit. Nothing rests on this machine's private state, which
is the same criterion the staging script states for the numbers the repo claims.

VERIFY COMPARES IN BOTH DIRECTIONS. A file present in the tree and absent from
the manifest is as much a defect as the reverse, and a one-directional check
lets a payload that quietly grew pass — the absence looks like agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The staged tree lives under the package dir; git paths are repo-root relative.
GIT_ROOT = REPO_ROOT.parent
GIT_PREFIX = REPO_ROOT.name

SCHEMA = "tekmyra-publication-payload/1"
GENERATOR = "scripts/stage_open_core.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=GIT_ROOT, capture_output=True, check=False
    )


def _source_candidates(relative: str) -> tuple[str, ...]:
    """Repo paths a payload file may have come from, payload-relative first.

    **A PAYLOAD PATH IS NOT ALWAYS ITS REPOSITORY PATH.** The licence files are
    copied from ``open_core_files/`` to the tree root, because this repository's
    own root ``LICENSE`` is the LaconIQ Proprietary one and must not be
    overwritten by the payload's Apache-2.0 text.

    Without this mapping the manifest compared the payload's Apache LICENSE
    against the proprietary blob at the same path, found them different, and
    recorded the most important file in the artifact as ``generated`` — sound,
    because ``--rederive`` still reproduced it, and WRONG, because the file is
    fully git-backed and a reader checking its provenance would be told to
    trust a script instead of a blob. A record that reads as one thing while
    being another is the defect class this file exists to close.
    """
    candidates = [f"{GIT_PREFIX}/{relative}"]
    if "/" not in relative:
        candidates.append(f"{GIT_PREFIX}/open_core_files/{relative}")
    return tuple(candidates)


def _blob_matches(commit: str, relative: str, path: Path) -> tuple[str, str] | None:
    """Return (blob sha, repo path) when byte-identical to one, else None.

    The byte comparison is the point. Asserting git provenance from the path
    alone would certify pyproject.toml, which exists at every commit and is
    generated fresh by the staging script.
    """
    contents = path.read_bytes()
    for candidate in _source_candidates(relative):
        rev = _git("rev-parse", f"{commit}:{candidate}")
        if rev.returncode != 0:
            continue
        blob = rev.stdout.decode().strip()
        show = _git("cat-file", "blob", blob)
        if show.returncode == 0 and show.stdout == contents:
            return blob, candidate
    return None


#: Local tooling and build detritus that ordinarily coexists with a checkout
#: and is never payload content.
#:
#: Only ``.git`` was filtered. So a reader who followed the README's OWN setup —
#: ``uv venv`` then ``pip install -e ".[dev]"`` — and then ran ``verify`` against
#: a legitimate manifest got thousands of spurious "IN THE PAYLOAD, ABSENT FROM
#: THE MANIFEST" errors and a REFUSED verdict on a completely healthy payload:
#: 5,049 files where the real answer is 484, every extra one misclassified
#: "generated".
#:
#: The mirror image of the suite runner that measured its own environment and
#: went quiet. Same root — a tool scanning "the tree" without accounting for
#: what normally sits in one — opposite symptom: false alarm rather than false
#: confidence. A gate that cries wolf on the documented workflow gets ignored,
#: and an ignored gate and a broken one differ only in how long it takes.
#:
#: ``.cache`` is DELIBERATELY ABSENT from this set. It holds model artifacts,
#: it is not published, and its presence in a staged payload would be a real
#: finding — filtering it here would convert a leak into silence, which is the
#: one direction this tool must never fail in.
IGNORED_DIR_NAMES = frozenset({
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".tox", "node_modules", "build", "dist",
})


def _is_payload_content(parts: tuple[str, ...]) -> bool:
    if any(part in IGNORED_DIR_NAMES for part in parts):
        return False
    if any(part.endswith((".egg-info", ".dist-info")) for part in parts):
        return False
    return not parts[-1].endswith((".pyc", ".pyo"))


def derive(tree: Path, commit: str) -> dict:
    """The single derivation both build and verify use."""
    files = sorted(
        p for p in tree.rglob("*")
        if p.is_file() and _is_payload_content(p.relative_to(tree).parts)
    )
    entries = []
    for path in files:
        relative = path.relative_to(tree).as_posix()
        match = _blob_matches(commit, relative, path)
        entry = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if match:
            blob, source = match
            entry["provenance"] = "git"
            entry["blob"] = blob
            # The repo path the bytes came from, which is not always the
            # payload path — the licence files are copied from open_core_files/.
            entry["source"] = source
        else:
            entry["provenance"] = "generated"
            entry["generated_by"] = GENERATOR
        entries.append(entry)

    workflows = [e["path"] for e in entries if ".github/workflows/" in e["path"]]
    remotes = subprocess.run(
        ["git", "remote"], cwd=tree, capture_output=True, text=True, check=False
    ).stdout.split()

    return {
        "schema": SCHEMA,
        "commit": commit,
        "counts": {
            "files": len(entries),
            "from_git": sum(1 for e in entries if e["provenance"] == "git"),
            "generated": sum(1 for e in entries if e["provenance"] == "generated"),
        },
        # THE PAYLOAD ASSERTIONS, in the manifest rather than in a README, so a
        # reader checks the artifact instead of a sentence about it.
        "assertions": {
            "ci_workflows": workflows,
            "git_remotes": remotes,
        },
        "files": entries,
    }


def rederive_generated(tree: Path, manifest: dict) -> list[str]:
    """Re-run the staging script and confirm the generated files reproduce.

    This is what turns "generated, take our word for it" into a claim a stranger
    can check. It stages into a temporary directory; the payload is untouched.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from stage_open_core import stage  # noqa: PLC0415

    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="payload-rederive-") as tmp:
        fresh = Path(tmp) / "tree"
        stage(fresh, force=True)
        for entry in manifest["files"]:
            if entry["provenance"] != "generated":
                continue
            candidate = fresh / entry["path"]
            if not candidate.is_file():
                problems.append(f"{entry['path']}: re-stage did not produce it")
            elif sha256(candidate) != entry["sha256"]:
                problems.append(f"{entry['path']}: re-stage produced different bytes")
    return problems


def verify(tree: Path, manifest: dict, *, rederive: bool) -> list[str]:
    problems: list[str] = []
    if manifest.get("schema") != SCHEMA:
        problems.append(f"schema is {manifest.get('schema')!r}, expected {SCHEMA!r}")
        return problems

    live = derive(tree, manifest["commit"])
    recorded = {e["path"]: e for e in manifest["files"]}
    observed = {e["path"]: e for e in live["files"]}

    # BOTH DIRECTIONS. A payload that grew is the failure a one-way check misses.
    for path in sorted(set(observed) - set(recorded)):
        problems.append(f"{path}: IN THE PAYLOAD, ABSENT FROM THE MANIFEST")
    for path in sorted(set(recorded) - set(observed)):
        problems.append(f"{path}: IN THE MANIFEST, ABSENT FROM THE PAYLOAD")

    for path in sorted(set(recorded) & set(observed)):
        want, got = recorded[path], observed[path]
        for field in ("sha256", "bytes", "provenance"):
            if want.get(field) != got.get(field):
                problems.append(
                    f"{path}: {field} {want.get(field)!r} -> {got.get(field)!r}"
                )
        if want.get("provenance") == "git":
            if want.get("blob") != got.get("blob"):
                problems.append(f"{path}: blob {want.get('blob')} -> {got.get('blob')}")
            if want.get("source") != got.get("source"):
                problems.append(
                    f"{path}: source {want.get('source')!r} -> {got.get('source')!r}"
                )

    # KEY MATERIAL IS REFUSED WITHOUT CONSULTING THE MANIFEST.
    #
    # Every other check above asks whether the payload matches the record. That
    # is the wrong question for a credential: a key that someone added to the
    # manifest would pass every comparison in this function, because the
    # manifest would agree it belongs. Measured on the negative controls, a
    # planted key was caught only as an unmanifested file — one manifest
    # regeneration away from being blessed.
    #
    # This is the same defence-in-depth `open_surface.NEVER_IN_TREE` applies at
    # staging time, restated here because a payload check that can be satisfied
    # by updating the record is not a check on the payload.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from open_surface import is_key_material  # noqa: PLC0415

    for entry in live["files"]:
        if is_key_material(entry["path"]):
            problems.append(
                f"{entry['path']}: KEY MATERIAL IN THE PAYLOAD — refused "
                "unconditionally, whatever the manifest says"
            )

    if live["assertions"]["ci_workflows"]:
        problems.append(
            f"CI workflows present in the payload: {live['assertions']['ci_workflows']}"
        )
    if live["assertions"]["git_remotes"]:
        problems.append(
            "THE PAYLOAD HAS A GIT REMOTE — the publication gate is expressed as "
            f"the absence of one: {live['assertions']['git_remotes']}"
        )

    if rederive:
        problems.extend(rederive_generated(tree, manifest))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    build = sub.add_parser("build")
    build.add_argument("--tree", required=True, type=Path)
    build.add_argument("--commit", required=True)
    build.add_argument("--out", required=True, type=Path)

    check = sub.add_parser("verify")
    check.add_argument("--tree", required=True, type=Path)
    check.add_argument("--manifest", required=True, type=Path)
    check.add_argument(
        "--rederive",
        action="store_true",
        help="also re-run the staging script and confirm the generated files reproduce",
    )

    args = parser.parse_args()

    if args.mode == "build":
        manifest = derive(args.tree, args.commit)
        args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: manifest[k] for k in ("schema", "commit", "counts", "assertions")}, indent=2))
        return 0

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    problems = verify(args.tree, manifest, rederive=args.rederive)
    if problems:
        print(f"PAYLOAD MANIFEST: REFUSED — {len(problems)} problem(s)")
        for problem in problems[:40]:
            print(f"  {problem}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 2
    print(
        "PAYLOAD MANIFEST: OK — "
        f"{manifest['counts']['files']} files "
        f"({manifest['counts']['from_git']} byte-identical to blobs at "
        f"{manifest['commit'][:7]}, {manifest['counts']['generated']} generated"
        + (" and re-derived" if args.rederive else "")
        + "), no CI workflows, no git remote"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
