#!/usr/bin/env python3
"""Stage the open core as a complete, reviewable repository. Publishes nothing.

"Staged" means: the public repo exists LOCALLY, complete enough to review and
to run, sitting behind Hari's exact-phrase publication gate. No remote is
created, no repository is made public, and no CI workflow is written into the
tree.

THE TREE IS PRODUCED BY THIS SCRIPT, NOT ASSEMBLED BY HAND, and that is a
readiness requirement rather than convenience. The gate criterion is: A
FIRST-TIME CLONER MUST BE ABLE TO REPRODUCE EVERY NUMBER THE REPO CLAIMS, OR
THE REPO SHOULD NOT CLAIM IT. A hand-copied tree cannot satisfy that, because
nobody can re-derive it — the same class of defect as a lock that disagrees
with its manifest, or verification digests that cannot be recomputed from a
clean clone. Three costumes, one flaw: reproducibility that depends on a
machine's private state.

IT SHARES ONE PREDICATE WITH THE VERIFIER. The set of files staged here is
decided by exactly the function ``verify_open_core`` uses to decide what may
exist in a built tree — ``TreeCategory.matches`` and the import closure. Two
mechanisms that must agree are made one mechanism; agreement maintained by
discipline is a scheduled divergence.

THE FOUR LICENCE-DEPENDENT FILES ARE NOW DECIDED AND WRITTEN. Apache-2.0 was
approved 2026-08-11 with the patent practice map's open items disclosed; LICENSE,
NOTICE, CONTRIBUTING.md and SECURITY.md live under ``open_core_files/`` and are
copied to the payload root.

They have their own directory because THIS repository is licensed LaconIQ
Proprietary, and a payload LICENSE written to the repository root would
mislicense the closed repo in its own working tree.

THE OUTSTANDING MECHANISM STAYS THOUGH THE DICT IS EMPTY. Until each was
decided, every one shipped as a marker naming what was missing and WHY — the
last of them recording that the obvious security address would bounce, because
the domain published no MX record. Writing any of them early would have embedded
an answer nobody gave. A placeholder that says "outstanding" is honest; a
plausible default is not, and the next undecided file should get the same
treatment rather than a guess.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from open_surface import (  # noqa: E402
    OPEN_TREE_CATEGORIES,
    build_allowlist,
    is_key_material,
    validate_exclusions,
)
from public_corpus_notice import assert_notice_current  # noqa: E402

# Derived from the allowlisted modules' own imports, not copied from the
# closed project's manifest. tiktoken is CORE here and not an extra: it is
# imported by token_measure.py, which is in the allowlist, so the headline
# token figure is unobtainable without it — the "WITHHELD on a bare install"
# problem stated as a dependency rather than a symptom.
# A DECLARED FLOOR IS A COMPATIBILITY CLAIM, AND OURS HAS BEEN WRONG TWICE —
# BOTH TIMES BECAUSE THE CLAIM OUTLIVED THE ARTIFACT IT WAS MEASURED ON.
# Magnus swept scikit-learn on 2026-08-12: 1.4.2 failed outright
# (`NotFittedError`) while 1.5.2 through 1.9.0 all produced 0.2929, so the
# floor became 1.5.2. The published reference bundle was then re-exported
# under scikit-learn >= 1.8, whose LogisticRegression no longer carries the
# removed `multi_class` attribute — and loading that pickle into 1.5.2, 1.6.1
# or 1.7.2 now crashes at model load (`AttributeError: 'LogisticRegression'
# object has no attribute 'multi_class'`), measured against the published
# bundle on 2026-08-30, with numpy/scipy confounds controlled. 1.8.0 and
# 1.9.0 both reproduce 0.2929 exactly.
#
# So 1.8 is the MEASURED floor for the artifact we actually ship. Declaring
# the measured value is the safe direction: widening later on evidence costs
# nothing, narrowing later breaks whoever believed the wider claim — and a
# floor swept against one export must be re-swept when the artifact moves,
# which is the rule this comment exists to keep.
OPEN_CORE_DEPENDENCIES = [
    "click>=8.1",
    "numpy>=1.24",
    "pydantic>=2.5",
    "scikit-learn>=1.8",
    "scipy>=1.10",
    "tiktoken>=0.7",
    "pyyaml>=6.0",
]

OPTIONAL_DEPENDENCIES = {
    # The transformer compressor path. Heavy, and the deterministic and
    # sklearn paths work without it.
    # onnx IS NOT onnxruntime, and only one of them was declared. Found by the
    # deferred-import guard on its first run: tests/test_onnx_runtime.py:12
    # imports onnx inside a helper its tests actually call, unguarded, and onnx
    # appears in neither this list nor the SOURCE pyproject — it resolves in our
    # development environment only because something else drags it in
    # transitively. A user installing [transformer] to exercise the ONNX path
    # would have hit ModuleNotFoundError at CALL time, inside a helper, with
    # nothing connecting it back to packaging. Declared here rather than in
    # 'dev' because the test is only reachable when the transformer path is
    # installed, and putting an ONNX dependency in 'dev' would charge it to
    # everyone running the ordinary suite.
    "transformer": ["torch>=2.0", "transformers>=4.40", "onnxruntime>=1.16",
                    "onnx>=1.15"],
    # Comparison baselines only. Not needed to run or reproduce the headline.
    "baselines": ["llmlingua>=0.2", "sentence-transformers>=2.2"],
    # THE SUITE SHIPS, SO ITS RUNNER MUST BE DECLARED. Adding the test suite to
    # the payload without declaring pytest would ship "you can verify our seam
    # yourself" alongside no way to run the verification — the same shape as a
    # README quoting a figure the repository cannot reproduce, one layer down.
    # THE RUNNER IS NOT THE WHOLE DEPENDENCY. The reasoning above is right and
    # it stopped one step short: the suite needs pytest AND everything the
    # shipped tests import. `tests/test_contracts.py` imports jsonschema to
    # validate the shipped `contracts/*.json`, jsonschema is correctly absent
    # from the runtime dependencies because no shipped `tekmyra` module imports
    # it, and nothing connected those two facts. On a clean install pytest
    # aborted COLLECTION — not two files skipping, the whole suite running zero
    # tests — and `verify_open_core.py` then refused a healthy payload.
    #
    # A comment that names the defect class and fixes one instance of it is how
    # the second instance survives. `tests/test_payload_dependency_closure.py`
    # now checks this mechanically so the next one cannot.
    "dev": ["pytest>=8.0", "jsonschema>=4.0"],
}

# DECIDED, AND THEREFORE REAL FILES RATHER THAN MARKERS. Hari approved
# Apache-2.0 on 2026-08-11 with the patent practice map's open items disclosed
# and read, and named the security reporting address, so ALL FOUR are written
# and live under ``open_core_files/`` — a directory of their own, because the
# repository this is extracted from is licensed LaconIQ Proprietary and its own
# LICENSE must not be overwritten by the payload's.
#
# ARCHITECTURE.md joins them at the ROOT, not under ``docs/``. The payload has
# no ``docs/`` directory and this tuple is the only mechanism that puts a
# written document into it, so a file authored at ``docs/ARCHITECTURE.md``
# would have shipped nowhere at all while appearing, in the source tree, to
# have been published.
OPEN_CORE_FILES_DIR = "open_core_files"
DECIDED_ROOT_FILES = ("LICENSE", "NOTICE", "CONTRIBUTING.md", "SECURITY.md",
                      "README.md", "ARCHITECTURE.md",
                      "constraints-reproduce.txt", ".gitignore", "CITATION.cff")

# EMPTY, AND THE MECHANISM STAYS. All four licence-dependent files are decided
# and written, so no OUTSTANDING marker is produced. The machinery is kept
# rather than deleted because the next undecided file should get the same
# treatment — a placeholder that states what is missing and why, instead of a
# plausible default that embeds an answer nobody gave.
OUTSTANDING: dict[str, str] = {}


def stage(destination: Path, *, force: bool = False) -> dict:
    # Public corpora carry third-party source. A stale NOTICE is a legally
    # incomplete payload, so freshness is a staging precondition.
    assert_notice_current(REPO_ROOT)
    validate_exclusions()
    if destination.exists():
        if not force:
            raise SystemExit(f"{destination} exists; pass --force to replace")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    allowlist = build_allowlist()
    staged: list[str] = []

    for rel in sorted(allowlist["allowed_paths"]):
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, target)
        staged.append(rel)

    for category in OPEN_TREE_CATEGORIES:
        # The demo artifacts are a VERIFICATION input, not part of the
        # published repository — Vega Q2 keeps trained artifacts closed and
        # ships a demo model as a release asset with a recorded sha256, never
        # committed. So the staged repo omits them, and the README will have to
        # say how a cloner obtains one; that is a readiness item, not a copy.
        if category.name == "demo artifacts":
            continue
        for root in category.roots:
            source_root = REPO_ROOT / root
            if not source_root.is_dir():
                continue
            for source in sorted(source_root.rglob("*")):
                if not source.is_file() or "__pycache__" in source.parts:
                    continue
                rel = source.relative_to(REPO_ROOT).as_posix()
                if not category.matches(rel) or is_key_material(rel):
                    continue
                target = destination / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                staged.append(rel)

    # THE DECIDED LICENCE FILES, COPIED RATHER THAN GENERATED. Apache-2.0's text
    # may not be modified, so it travels as bytes and its digest is checkable
    # against apache.org. Missing ones RAISE: staging a payload without its
    # licence would produce a tree that looks complete and grants nothing.
    for name in DECIDED_ROOT_FILES:
        source = REPO_ROOT / OPEN_CORE_FILES_DIR / name
        if not source.is_file():
            raise SystemExit(
                f"{OPEN_CORE_FILES_DIR}/{name} is missing. It is a DECIDED file, "
                "not an outstanding one — refusing to stage a payload whose "
                "licence terms are silently absent."
            )
        text = source.read_text(encoding="utf-8", errors="replace")
        unresolved = re.findall(r"<<<([A-Z0-9_]+)>>>", text)
        if unresolved:
            # A DOCUMENTED DOWNLOAD THAT 404s IS THE UNREPRODUCIBLE-FIGURE
            # DEFECT ONE LAYER DOWN. The README describes fetching a demo model
            # and checking its digest; shipping it with the URL still a
            # placeholder would tell a stranger to run a command that cannot
            # work. Refuse, and name what is unfilled.
            raise SystemExit(
                f"{OPEN_CORE_FILES_DIR}/{name} still contains unresolved "
                f"placeholder(s): {sorted(set(unresolved))}. Fill them or remove "
                "the section — a published instruction that cannot be followed is "
                "worse than an absent one."
            )
        shutil.copy2(source, destination / name)
        staged.append(name)

    # THE PAYLOAD'S ``__version__`` IS THE DISTRIBUTION'S, NOT THE SOURCE
    # TREE'S. The module line is copied from a tree whose version is an
    # internal release-candidate number; leaving it would make
    # ``tekmyra.__version__`` and ``pip show tekmyra-core`` disagree inside
    # the same install. Rewritten here — the single place the payload is
    # derived — from the same constant the generated pyproject uses. Exactly
    # one line must match; zero or many means the source layout changed and
    # this rewrite is editing blind, so it refuses.
    staged_init = destination / "tekmyra" / "__init__.py"
    init_text = staged_init.read_text(encoding="utf-8")
    init_text, replaced = re.subn(
        r'^__version__ = "[^"]*"$',
        f'__version__ = "{OPEN_CORE_VERSION}"',
        init_text,
        flags=re.MULTILINE,
    )
    if replaced != 1:
        raise SystemExit(
            f"expected exactly one __version__ line in {staged_init}, found "
            f"{replaced} — refusing to stage a payload whose version cannot "
            "be aligned with its distribution metadata."
        )
    staged_init.write_text(init_text, encoding="utf-8")

    (destination / "pyproject.toml").write_text(_pyproject(), encoding="utf-8")
    staged.append("pyproject.toml")

    # THE SDIST IS THE WHEEL'S SOURCE, NOT THE REPOSITORY. Without this,
    # setuptools swept a PARTIAL tests/ tree into the sdist (47 of 56 files,
    # no baselines, no benchmarks) and the README's own pytest command
    # produced seven collection errors inside it - a broken suite shipped to
    # exactly the person who downloads sources to check them. Tests and
    # corpora live in the repository; the sdist carries what builds the wheel.
    (destination / "MANIFEST.in").write_text(
        "prune tests\nprune benchmarks\n", encoding="utf-8"
    )
    staged.append("MANIFEST.in")

    # ONLY WHEN SOMETHING IS ACTUALLY OUTSTANDING. An empty OUTSTANDING/
    # directory in a published tree asks a question it does not answer — the
    # reader looks for what is missing and finds nothing, which reads as a
    # truncated artifact rather than a complete one.
    outstanding_dir = destination / "OUTSTANDING"
    if OUTSTANDING:
        outstanding_dir.mkdir()
    for name, reason in OUTSTANDING.items():
        (outstanding_dir / f"{name}.OUTSTANDING.md").write_text(
            f"# {name} — NOT WRITTEN, AND THAT IS DELIBERATE\n\n{reason}",
            encoding="utf-8",
        )

    # A repository, so it is reviewable and diffable — with NO REMOTE. The
    # absence of a remote is the publication gate expressed as a fact about the
    # tree rather than an intention about behaviour.
    subprocess.run(["git", "init", "-q"], cwd=destination, check=True)

    workflows = list(destination.rglob(".github/workflows/*"))
    keys = [p for p in destination.rglob("*") if p.is_file() and is_key_material(str(p))]

    return {
        "destination": str(destination),
        "files_staged": len(staged),
        "python_modules": len(allowlist["allowed_paths"]),
        "ci_workflows_present": len(workflows),
        "key_material_present": len(keys),
        "outstanding_files": sorted(OUTSTANDING),
        "git_remotes": subprocess.run(
            ["git", "remote"], cwd=destination, capture_output=True, text=True
        ).stdout.split(),
    }


# 0.1.0, RULED BY HARI 2026-08-13, and deliberately NOT the source version.
# The source tree is at an internal release-candidate number; publishing that
# would leak our release cadence and mean nothing to a reader. The previous
# value here was "0.0.0", which to any engineer reads as "placeholder, not a
# real release" — it is what pip reports to every user, and it undersold work
# that is certified.
#
# NOT 1.0.0, and the reason is a fact rather than modesty: 1.0.0 is a promise
# about INTERFACE STABILITY, not quality, and we already intend to break it —
# moving contracts/ and configs/ inside the package to be read via
# importlib.resources is registered post-launch work that changes how callers
# touch this package. A promise you already know you will break is the worst
# kind. 0.1.0 says real, works, early, and things may still move.
#
# ONE CONSTANT, TWO CONSUMERS: the generated pyproject below and the
# ``__version__`` rewrite in ``stage()``. Before this constant existed the
# staged package's ``tekmyra.__version__`` reported the SOURCE tree's
# release-candidate number while ``pip show`` reported 0.1.0 — two answers to
# one question, and the wrong one was the one code reads at runtime.
OPEN_CORE_VERSION = "0.1.0"


def _pyproject() -> str:
    extras = "\n".join(
        f'{name} = {json.dumps(deps)}' for name, deps in OPTIONAL_DEPENDENCIES.items()
    )
    return f"""[project]
name = "tekmyra-core"
version = "{OPEN_CORE_VERSION}"
authors = [{{ name = "LaconIQ", email = "hello@tekmyra.ai" }}]
keywords = ["llm", "context", "compression", "prompt", "tokens"]
# A PyPI PAGE WITH NO LINKS IS AN ORPHAN. Measured on the red-team pass:
# the built wheel's METADATA carried zero Project-URL headers, so the
# package page would have linked to nothing - not the repo, not the paper,
# not the site that explains the refusal semantics.
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
# Without this the PyPI page is BARE: twine warns long_description missing and
# the project renders as a name with no story. The staged README is the page.
readme = "README.md"
description = "Deterministic and learned context compression, with fidelity verification."
requires-python = ">=3.11"
# THE PAYLOAD SHIPPED AN APACHE-2.0 LICENSE FILE AND DECLARED NO LICENCE AT
# ALL IN ITS METADATA. Tools read the metadata, not the file: PyPI, SBOM
# scanners and corporate policy gates would have seen an unlicensed package,
# which blocks exactly the enterprise consumer this release is aimed at and
# muddies a grant we did make. This states the decision of 2026-08-11 on the
# tree that decision was FOR; it creates no new grant.
license = "Apache-2.0"
# DERIVED from the staged modules' own imports, not copied from the closed
# project's manifest. tiktoken is a CORE dependency and not an extra: it is
# imported by token_measure.py, so the token-reduction figure is unobtainable
# without it. In the closed repo it sat in an optional extra that was missing
# from the lock entirely, which is why a bare install printed WITHHELD instead
# of a number.
dependencies = {json.dumps(OPEN_CORE_DEPENDENCIES, indent=4)}

[project.optional-dependencies]
{extras}

[project.urls]
Homepage = "https://tekmyra.ai"
Repository = "https://github.com/laconiq-ai/tekmyra"
Documentation = "https://tekmyra.ai/docs.html"
Paper = "https://tekmyra.ai/tekmyra-paper.html"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["tekmyra*"]

# WITHOUT THIS THE WHEEL DOES NOT IMPORT. `packages.find` collects PYTHON
# MODULES and nothing else, so a payload declaring only that section built a
# wheel of 61 files, every one a .py, with `contracts/` and `configs/` silently
# absent. `tekmyra/contracts.py:15` resolves CONTRACTS_DIR at MODULE level, and
# `tekmyra.contracts` is imported by anchor_interface, runtime, baselines and
# benchmark — so a normal `pip install` of the published package raised
# FileNotFoundError on the first import of essentially the whole library.
#
# `tekmyra/resources.py` looks in exactly two places: beside the source tree,
# and `share/tekmyra` under the install's data path. The first only ever
# resolves for an EDITABLE install, where `contracts/` is still a sibling of
# `tekmyra/`. That is why this survived every measurement we took — the
# README's own `pip install -e .`, the verification script, and the container
# rehearsal all installed editable or ran from a cwd that happened to have
# `contracts/` sitting next to `tekmyra/`, and `sys.path[0]` did the rest.
#
# The source repo declares the same section; the generated one omitted it, so
# the two builds diverged in the one direction nobody tests. Mirrored here
# rather than reinvented, because the source mechanism is the one `resources.py`
# was written against. The durable fix is to move this data INSIDE the package
# and read it through importlib.resources, which removes the install-layout
# dependency entirely — structural, and post-launch.
[tool.setuptools.data-files]
"share/tekmyra/contracts" = ["contracts/*.json"]
"share/tekmyra/configs" = ["configs/*.yaml"]
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(stage(args.dest, force=args.force), indent=2))
