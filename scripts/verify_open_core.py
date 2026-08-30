#!/usr/bin/env python3
"""Can the open core reproduce the published number with the closed half DELETED?

This is Vega's hard test — "independently useful, buildable and testable with the
closed half absent" — made executable, so it is checked by a machine on every run
rather than asserted once by whoever last looked.

It is expected to FAIL until the seam extractions land. That is the point: it is
a progress meter that goes green exactly when the seam becomes real, and the
blockers it prints are the remaining work.

WHY IT PROVES ISOLATION BEFORE MEASURING ANYTHING. The first version of this test
reported "IMPORT OK" and reproduced the headline number exactly, with the closed
half supposedly deleted. That was entirely false: the venv's ``__editable__``
meta-path finder resolved every deleted submodule back to the real checkout, so
the run measured the closed half while claiming it was absent. Asserting module
identity on the top-level package passed and proved nothing.

A false pass here is far worse than a false failure — it would report the seam
already clean, cancel the extractions, and ship a package that silently depends
on the closed half. So this refuses to print a number until it has demonstrated
that every closed module is unimportable.

Two adjacent traps, both of which MIMIC a seam failure and neither of which is
one: filtering sys.path by a bare substring of the repo path also strips its
.venv/site-packages (the run then dies on a missing third-party dependency), and
copying the package without its resource directories fails inside contracts.py.
Both are handled below; recorded so the next reader does not misdiagnose them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: EXIT CODES ARE THE MACHINE-READABLE CHANNEL AND THEY MUST NOT COLLAPSE.
#:
#:   0  verified, with numbers measured
#:   2  failed — the seam is not proven, read the prose
#:   3  ISOLATION_FAILED — a withheld module was importable
#:   4  NOT YET — the open core does not build standalone
#:   5  WRONG BUILD — the closed half was deleted but the benchmark did not
#:      select the open store, so the number does not describe the open core
#:   6  SEAM PROVEN, NUMBERS NOT MEASURED — no model artifact. EXPECTED ON A
#:      FRESH CLONE. Non-zero deliberately, so an unmeasured run can never be
#:      quoted as success; distinct, so a caller can allow it on purpose.
#:   7  SEAM PROVEN, MODEL PRESENT BUT UNUSABLE — not the fresh-clone state
#:
#: The child already distinguished all of these and the PARENT COLLAPSED THEM
#: INTO 1 AND 2. Worse, 1 meant both "the open core does not build standalone"
#: and "seam proven, healthy fresh clone" — opposite conclusions sharing a code.
#: The earlier fix repaired the PROSE, so a human read an accurate verdict while
#: the machine-readable channel stayed exactly as ambiguous as before. That is
#: the same defect as the ABSTENTION label: the rendering was corrected and the
#: underlying signal was left conflated, which is harder to spot afterwards
#: precisely because the visible half now looks right.
#:
#: Anyone wiring this into CI on a fresh clone should allow 6:
#:     python scripts/verify_open_core.py || [ $? -eq 6 ]
REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_surface import (  # noqa: E402
    CATEGORICAL_EXCLUSIONS,
    OPEN_TREE_CATEGORIES,
    build_allowlist,
    tree_strays,
    validate_exclusions,
)

# THE CLOSED HALF, retained ONLY to prove isolation. It is no longer what
# decides the open surface — see build_open_tree, which copies an ALLOWLIST.
#
# The distinction is the whole point of this file's 2026-08-09 rewrite. A
# denylist publishes by default: measured, deleting these eight names and
# copying the rest shipped 88 modules / 22,266 lines, of which 36 modules /
# 11,312 lines were reached by no documented entry point at all — including six
# paid-provider adapters, a spend ledger, a credit-balance module and the spend
# caps. Nobody decided to publish any of them; nobody had to.
def _closed_submodules() -> list[str]:
    """DERIVED from the exclusion list, never curated.

    This was a hardcoded list of EIGHT while ``CATEGORICAL_EXCLUSIONS`` carried
    twenty-five. So ``ISOLATION_OK`` proved eight modules unimportable and said
    nothing whatever about the other seventeen — which include every
    paid-provider adapter and the entire money/quota group: spend_ledger,
    credit_balance, budget_policy, global_rate_limit. Adversarial review
    extended the same leak test to three of the untested ones and all three
    leaked identically, so the silence was not benign.

    A PROOF WHOSE SCOPE IS NARROWER THAN THE CLAIM IT APPEARS TO SUPPORT is the
    defect this whole file exists to catch, and it was sitting in this file. A
    curated list drifts the moment an exclusion is added, and drifts silently,
    because nothing fails — the verdict just quietly covers less.

    Returns dotted submodule paths relative to ``tekmyra``, so a nested
    exclusion such as ``tekmyra/providers/openai_provider.py`` is proven at
    ``tekmyra.providers.openai_provider`` rather than at a bare stem that
    imports nothing.
    """
    names = set()
    for path in CATEGORICAL_EXCLUSIONS:
        parts = Path(path).with_suffix("").parts
        if parts and parts[0] == "tekmyra" and len(parts) > 1:
            names.add(".".join(parts[1:]))
    return sorted(names)


CLOSED_MODULES = _closed_submodules()

# SUPPORT_DIRS IS GONE, DELIBERATELY. It named three directories that were
# copied WHOLESALE — configs, contracts and benchmarks — alongside an
# unconditional copy of .cache. That produced a 2,033-file tree of which 1,979
# files nobody had decided to publish, and it is how four paid-provider
# runners, two model pickles, internal sellability tooling and a live signing
# key reached a publication tool's output.
#
# Their replacement is OPEN_TREE_RULES in open_surface.py: the same material
# selected by pattern, each rule carrying its reason. A directory name is not a
# decision about its contents.

#: STRIPS THE AMBIENT DEV ENVIRONMENT OUT OF A CHILD INTERPRETER, and is shared
#: by every child this script starts.
#:
#: It was written once, for the isolation child, and the suite runner a few
#: hundred lines below never used it — it was a bare `subprocess.run` inheriting
#: this interpreter, whose site-packages carries the editable install's finder
#: pointing straight back at the real checkout. So the gate that answers "does
#: the payload's suite pass" reported SUITE_OK 297 passed / 80 skipped on a tree
#: where a genuinely clean interpreter collects two errors and runs ZERO tests.
#:
#: The docstring on that runner reasons carefully about copying to a
#: payload-SHAPED tree so it does not measure a configuration nobody receives —
#: and then measured it through a contaminated interpreter, which undoes the
#: care entirely. A gate that repairs the very thing it is checking will always
#: pass, and this one did, through every commit on this branch.
#:
#: FACTORED so the two children cannot drift apart again. One of them being
#: correct while the other was not is the whole incident.
_DECONTAMINATE = '''
import site
import sys
import sysconfig
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
ORIGINAL = sys.argv[2]

removed = []
for finder in list(sys.meta_path):
    label = f"{getattr(finder, '__module__', '')}.{type(finder).__name__}".lower()
    if "editable" in label:
        sys.meta_path.remove(finder)
        removed.append(label)


# KEEP THE INTERPRETER'S OWN DEPENDENCY DIRECTORIES, IDENTIFIED STRUCTURALLY.
# This used to keep exactly one path spelled f"{ORIGINAL}/.venv", so it worked
# for people who named their virtualenv .venv and stripped the site-packages of
# everyone else. Measured: with the env at venv/, pytest vanishes inside the
# child and this script reports "SUITE NOT RUN" and then "NOT YET -- the open
# core does not build standalone." That is the flagship command in the README
# telling a reader the project is broken because of what they called a folder.
# `python -m venv venv` is the invocation in Python's own documentation.
#
# sysconfig/site report the running interpreter's actual dependency roots
# whatever the environment is called, so the keep-set is derived rather than
# spelled. Everything else under ORIGINAL is still stripped -- that stripping is
# the whole isolation proof and is deliberately unchanged.
_KEEP = set()
for _p in (sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")):
    if _p:
        _KEEP.add(str(Path(_p).resolve()))
try:
    _KEEP.update(str(Path(p).resolve()) for p in site.getsitepackages())
except Exception:
    pass
try:
    _KEEP.add(str(Path(site.getusersitepackages()).resolve()))
except Exception:
    pass


def _is_original_source(entry):
    if not entry:
        return False
    resolved = str(Path(entry).resolve())
    if resolved in _KEEP:
        return False
    return resolved.startswith(ORIGINAL)


sys.path = [p for p in sys.path if not _is_original_source(p)]
sys.path.insert(0, str(ROOT))
'''


#: Runs the payload's own suite with the ambient environment stripped. Invoking
#: pytest.main in-process, after decontamination, rather than `-m pytest`, which
#: would re-enter a fresh interpreter and re-inherit everything just removed.
_SUITE_CHILD = _DECONTAMINATE + '''
import pytest

raise SystemExit(pytest.main(["-q", "tests", "--no-header"]))
'''


#: Additionally proves the withheld modules are unimportable once the ambient
#: environment is gone, then measures.
_CHILD = _DECONTAMINATE + '''
CLOSED = sys.argv[3].split(",")
leaked = []
for name in CLOSED:
    try:
        __import__("tekmyra." + name)
        leaked.append((name, sys.modules["tekmyra." + name].__file__))
    except ImportError:
        pass
    except Exception:
        pass

if leaked:
    print("ISOLATION_FAILED")
    for name, path in leaked:
        print(f"  {name} -> {path}")
    raise SystemExit(3)

# SAY WHAT WAS MEASURED, AND SAY WHEN IT WAS VACUOUS.
#
# This check answers exactly one question: can the closed half be reached FROM
# WITHIN THE BUILT TREE. It cannot answer the broader one, and not because of an
# oversight — once `tekmyra` resolves to the tree, `tekmyra.__path__` is
# tree-only, so a second `tekmyra` elsewhere on sys.path contributes no
# submodules at all. Measured: with the editable finder stripped, a planted
# module in another site-packages `tekmyra` is invisible from here. Running this
# child from an unrelated cwd would NOT widen the question; it would silently
# change which artifact is under test, because there `tekmyra` resolves to
# whatever is installed rather than to the tree. That is the wheel guard's
# question and the wheel guard is where it is answered.
#
# AND WHEN finders_removed IS 0, THIS PROVED NOTHING THE EXCLUSION CHECK HAD NOT
# ALREADY PROVED. There was no contamination to strip, so "not importable" and
# "not present as a file" are the same observation. A verdict that reads
# identically whether it did real work or none is the shape this whole file
# exists to refuse.
_scope = "from within the built tree"
if removed:
    print(f"ISOLATION_OK {_scope} finders_removed={len(removed)}")
else:
    print(f"ISOLATION_OK {_scope} finders_removed=0 — no ambient contamination "
          "was present to strip, so this adds nothing to the exclusion check")

try:
    from tekmyra.benchmark_compressor import run_compressor_benchmark
    # Imported for `except` by TYPE. Matching on the class NAME was defeated in
    # one move by an unrelated class that happened to share it, and reported a
    # corrupt fixture as a benign missing model. Inside this try because a
    # failure to import it is the same BLOCKED condition as the line above.
    from tekmyra.router_provenance import (
        RouterArtifactUnavailable as _ArtifactUnavailable)
    # A CORRUPT ARTIFACT RAISES A DIFFERENT CLASS, AND NOTHING CAUGHT IT.
    # learned_router/model.py raises RouterArtifactLoadError when the directory
    # and router.joblib both exist but the file will not unpickle — which is
    # the realistic corruption case exit 7 is NAMED for, "a corrupt, partial or
    # version-mismatched artifact looks exactly like a present one until
    # something opens it". Because it is not a RouterArtifactUnavailable, the
    # careful artifact_dir_exists disambiguation below never ran, and the whole
    # thing surfaced as the generic "child exited N". Exit 7 was reachable only
    # for "directory present, expected file absent" — a narrower and far less
    # likely case than the one its own message describes.
    from tekmyra.learned_router.model import (
        RouterArtifactLoadError as _ArtifactLoadError)
except Exception as exc:
    import traceback
    print("BLOCKED")
    tb = traceback.extract_tb(sys.exc_info()[2])
    for frame in tb:
        if "tekmyra" in frame.filename:
            module = Path(frame.filename).name
            print(f"  {module}:{frame.lineno}  {frame.line}")
    print(f"  {type(exc).__name__}: {exc}")
    raise SystemExit(4)

# EVERY CORPUS THIS RUNS, NAMED WITH ITS MEAN FIXTURE SIZE.
#
# This used to run `synthetic` alone and print a bare headline. Both halves of
# that were a problem. `synthetic` is ~250 characters a fixture; long_context_v1
# is ~23,600 — a 95x gap — and the sentence "the open core reproduces the
# headline" was a claim about the corpus we had already established does not
# describe realistic content. A number with no corpus beside it cannot be
# misread, because there is nothing to read it against.
#
# So: run both, and print the corpus name, fixture count and mean size next to
# every figure. The isolation proof above is unchanged and is what makes any of
# this a seam verdict; this only fixes what the number is a number OF.
CORPORA = ["synthetic", "long_context_v1"]

payloads = []
for corpus_name in CORPORA:
    corpus_dir = ROOT / "benchmarks" / "fixtures" / corpus_name
    if not corpus_dir.is_dir():
        # Absent is reported, never skipped silently — a corpus that vanishes
        # from the copied tree would otherwise shrink the evidence unremarked.
        payloads.append({"corpus": corpus_name, "status": "ABSENT"})
        continue

    try:
        report = run_compressor_benchmark(fixture_dir=corpus_dir)
    except _ArtifactLoadError as exc:
        # THE CASE EXIT 7 IS NAMED FOR, and until now the one case it could not
        # report. An artifact that is present and will not unpickle is
        # unambiguously "present but unusable" — no disambiguation needed,
        # because this class is only ever raised when the file was found and
        # opened. Listed BEFORE the _ArtifactUnavailable clause so the ordering
        # states the intent even though the two are unrelated types today; if
        # they are ever unified, the specific one must still win.
        print("MODEL_PRESENT_BUT_UNUSABLE")
        print(f"  corpus {corpus_name}: {exc}")
        raise SystemExit(7) from None
    except _ArtifactUnavailable as exc:
        # NO MODEL IS NOT A BROKEN SEAM, AND THE TWO MUST NOT SHARE A VERDICT.
        # The artifact is a separate download, so a fresh clone has none, and
        # the benchmark refuses rather than reporting 1.000 — correct, and the
        # README documents it. But it surfaced as the generic "child exited 1"
        # / RESULT: ERROR, identical to a real failure, on the one command the
        # README tells a reader to run first.
        #
        # MATCHED BY TYPE, NOT BY NAME. The first version of this compared
        # `type(exc).__name__` to a string, and QA defeated it in one move: any
        # unrelated exception class that happens to share the name — a
        # corrupt-fixture error was the demonstration — was reported as a
        # benign missing model, which is the "real failure reported as success"
        # case this whole exit code exists to prevent. The real class is
        # importable here; the string check was never necessary.
        #
        # The class is also OVERLOADED upstream: require_loaded() raises it
        # both when the artifact is ABSENT and when it is present but failed to
        # load. Only the first is "expected on a fresh clone", so the two are
        # separated here rather than both wearing the reassuring verdict.
        #
        # THE REASSURING VERDICT REQUIRES POSITIVE EVIDENCE. Exit 6 is claimed
        # only where the artifact directory is KNOWN ABSENT. `exists` is None
        # when nothing determined it, and an unknown must not inherit the
        # comfortable reading — it falls through to the generic failure, which
        # is the safe direction to be wrong in.
        exists = getattr(exc, "artifact_dir_exists", None)
        if exists is False:
            print("NO_MODEL")
            print(f"  corpus {corpus_name}: {exc}")
            raise SystemExit(6) from None
        if exists is True:
            print("MODEL_PRESENT_BUT_UNUSABLE")
            print(f"  corpus {corpus_name}: {exc}")
            raise SystemExit(7) from None
        raise
    build = report.get("build_provenance", {})
    if build.get("build") != "open-core":
        # The tree had the closed half deleted, so the benchmark MUST have
        # selected the open store. If it did not, something resolved anchors
        # anyway and the number does not describe the open build.
        print("WRONG_BUILD")
        print(f"  corpus {corpus_name}: expected open-core, got "
              f"{build.get('build')!r} via {build.get('anchor_store')!r}")
        raise SystemExit(5)

    sizes = [len(f.content) for f in __import__(
        "tekmyra.fixtures", fromlist=["load_fixtures"]
    ).load_fixtures(corpus_dir)]
    payloads.append({
        "corpus": corpus_name,
        "status": "RAN",
        "fixtures": len(sizes),
        "mean_chars": round(sum(sizes) / len(sizes)) if sizes else 0,
        "build": build.get("build"),
        "anchor_store": build.get("anchor_store"),
        "anchor_resolution_available": build.get("anchor_resolution_available"),
        "headline": report["headline_avg_compression_ratio_reached"],
        "reached": report["compressor_reached"],
        "eligible": report["eligible_fixtures"],
        "refused": report["refused_fixtures"],
        "locked_preserved": report["locked_spans_preserved"],
        "locked_detected": report["locked_spans_detected"],
        "token_reduction_pct": report.get("corpus_token_reduction_pct"),
        "abstention_rows": report.get("abstention_rows"),
        "by_route": {
            name: {
                "done": s["completed_fixtures"],
                "refused": s["refused_fixtures"],
                "reduction_pct": s["token_reduction_pct"],
                "ratio": s["avg_compression_ratio"],
                "abstention": s["is_abstention_route"],
            }
            for name, s in (report.get("by_route") or {}).items()
        },
    })

print("RAN")
print(json.dumps(payloads))
'''


def build_open_tree(destination: Path) -> dict:
    """Copy ONLY the allowlisted paths. Inclusion is proved by the build itself.

    Inverted 2026-08-09. This used to ``copytree`` the whole package and delete
    eight files, which published everything nobody had thought about.

    Copying from the allowlist makes the run its own INCLUSION test: if the
    allowlist is missing a module the runtime needs, the child process fails to
    import and the seam is reported broken. There is no separate assertion to
    keep in step, and no way for the list to be quietly incomplete while the
    verdict stays green.
    """
    validate_exclusions()
    allowlist = build_allowlist()
    allowed = sorted(allowlist["allowed_paths"])

    for rel in allowed:
        source = REPO_ROOT / rel
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    # EVERYTHING ELSE IS COPIED BY RULE, NOT BY DIRECTORY.
    #
    # This used to copytree the whole of configs/, contracts/, benchmarks/ and
    # .cache — 1,979 files nobody had decided to publish, including four
    # paid-provider runners, two trained model pickles, internal sellability
    # tooling and a live Ed25519 signing key. Copying a directory wholesale is
    # the denylist, one level up from the module list we had just replaced: a
    # single missing check away from shipping whatever happens to be adjacent.
    #
    # benchmarks/ alone held 1,812 files; the verification needs the corpora.
    # .cache held 24 directories; it needs two.
    for category in OPEN_TREE_CATEGORIES:
        for root in category.roots:
            source_root = REPO_ROOT / root
            if not source_root.is_dir():
                continue
            for source in sorted(source_root.rglob('*')):
                if not source.is_file() or '__pycache__' in source.parts:
                    continue
                rel = source.relative_to(REPO_ROOT).as_posix()
                # The category decides, and the SAME predicate the check
                # uses decides it — build and verify cannot disagree about
                # what belongs, which is how a tree passes its own check
                # while containing something nobody allowed.
                if not category.matches(rel):
                    continue
                target = destination / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    for pycache in destination.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)

    return {"allowed": allowed, "excluded": allowlist["excluded_from_floor"]}


def assert_exclusion(tree: Path) -> list[tuple[str, str]]:
    """Nothing may exist in the built tree that nobody decided to publish.

    REWRITTEN 2026-08-09 after an independent QA seat returned NO-GO. The
    previous implementation quoted the rule correctly in its own docstring and
    then scanned ``(tree / "tekmyra").rglob("*.py")`` — not the tree but
    one subdirectory of it, and not every file but only Python. It verified 54
    files of a 2,033-file artifact and printed EXCLUSION_OK. Model pickles,
    caches, keys, data and tooling were structurally invisible to the function
    written to find exactly them.

    That is a sharper failure than an oversight: the rule was right, quoted,
    and implemented at a scope that could not see what it existed to catch.

    Now enumerates the WHOLE tree — every file, every extension, no subtree —
    per Vega's correction that he had "specified a property of the computation
    when I needed a property of the artifact". Key material is refused
    unconditionally, so the class that was invisible cannot be readmitted by
    someone writing a permissive rule.
    """
    allowed = set(build_allowlist()["allowed_paths"])
    return tree_strays(tree, allowed)


def _run_payload_suite(tree: Path) -> dict:
    """Run the shipped suite in a tree shaped like the PAYLOAD, not like this one.

    **THE DEMO ARTIFACTS ARE REMOVED FIRST, AND THAT IS THE WHOLE POINT.** This
    verification tree carries ``.cache/`` because the measurement below cannot
    produce a number without it. The published payload does not carry it — Vega
    Q2 keeps trained artifacts closed. Running the suite in THIS tree would test
    a configuration nobody ever receives, and the first run of that gate proved
    it: with the artifacts present the suite reported 3 failed and 9 errors,
    because those tests need more than a model, while the payload-shaped tree
    passes 283 and skips 83 with a stated reason.

    A gate that runs against a tree the audience never sees is measuring the
    wrong artifact, however green it goes.

    Returns rather than raises, so the caller decides the verdict — and reports
    counts either way, because "the suite ran and 283 passed" and "pytest is not
    installed" must never render identically. An absent runner is NOT RUN, never
    a pass.
    """
    payload_shape = tree.parent / "payload-shape"
    shutil.copytree(tree, payload_shape, ignore=shutil.ignore_patterns(".cache"))

    # THE PAYLOAD'S SHAPE INCLUDES THE DECIDED ROOT FILES AND A COMMIT.
    # Measured 2026-08-30 in the published payload itself: without these this
    # gate REFUSED a healthy tree, twice over — NOTICE was reported "stale"
    # when it was ABSENT (the notice test cannot tell the difference), and the
    # provenance test found no commit because the shape had no repository. The
    # audience's tree is `stage_open_core.py`'s output committed and cloned;
    # a shape missing its LICENCE, NOTICE, README and pyproject is not the
    # payload's shape, it is a third artifact nobody ships. stage_open_core is
    # a published sibling script, so this works identically inside the payload
    # (where the decided files sit at the ROOT, not under open_core_files/).
    import stage_open_core

    for name in stage_open_core.DECIDED_ROOT_FILES:
        for candidate in (
            REPO_ROOT / stage_open_core.OPEN_CORE_FILES_DIR / name,
            REPO_ROOT / name,
        ):
            if candidate.is_file():
                shutil.copy2(candidate, payload_shape / name)
                break
        else:
            return {
                "returncode": 1,
                "line": (
                    f"SUITE NOT RUN  decided root file {name!r} exists in "
                    "neither open_core_files/ nor the tree root; a payload "
                    "shape cannot be assembled without it"
                ),
                "tail": "",
            }
    (payload_shape / "pyproject.toml").write_text(
        stage_open_core._pyproject(), encoding="utf-8"
    )
    for args in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.name=verify-open-core",
         "-c", "user.email=verify@localhost",
         "commit", "-qm", "payload shape assembled for suite verification"],
    ):
        subprocess.run(args, cwd=payload_shape, check=True, capture_output=True)

    # DECONTAMINATED, like the isolation child. Previously a bare `-m pytest`
    # that inherited this interpreter — and its editable-install finder, which
    # resolves every module the payload does not contain straight back to the
    # real checkout. That is why this gate reported a passing suite on a tree
    # whose suite does not run at all.
    result = subprocess.run(
        [sys.executable, "-c", _SUITE_CHILD, str(payload_shape), str(REPO_ROOT)],
        cwd=payload_shape,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    # Both spellings: `-m pytest` says `No module named pytest`, an import
    # inside the child says `No module named 'pytest'`. Matching one of them
    # would turn an absent runner back into a silent pass, which is the exact
    # thing the docstring above promises never to do.
    if "No module named pytest" in output or "No module named 'pytest'" in output:
        return {
            "returncode": 0,
            "line": "SUITE NOT RUN  pytest is not installed; the payload's own "
                    "suite was NOT checked by this run",
            "tail": "",
        }
    summary = ""
    for line in reversed(output.strip().splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            summary = line.strip()
            break
    verdict = "SUITE_OK  " if result.returncode == 0 else "SUITE_FAILED  "
    return {
        "returncode": result.returncode,
        "line": f"{verdict}{summary or 'no summary line'}",
        "tail": "\n".join(output.strip().splitlines()[-25:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true", help="keep the built tree")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="tekmyra-open-core-"))
    tree = workdir / "open-core"
    tree.mkdir(parents=True)

    print("# Open-core verification — INCLUSION, EXCLUSION and ISOLATION")
    built = build_open_tree(tree)
    print(f"allowlist: copied {len(built['allowed'])} paths")
    if built["excluded"]:
        print(
            "rule 3 removed from the runtime floor "
            f"({len(built['excluded'])}): {', '.join(built['excluded'])}"
        )

    # EXCLUSION, checked before anything is measured. Vega's rule: an allowlist
    # that is never checked for what it left out is a denylist wearing a new
    # name — it can only prove the parts you remembered are present.
    strays = assert_exclusion(tree)
    if strays:
        print()
        print(f"EXCLUSION_FAILED — {len(strays)} path(s) nobody decided to publish:")
        for path, why in strays[:40]:
            extra = CATEGORICAL_EXCLUSIONS.get(path)
            print(f"  {path}\n      {why}" + (f"   [{extra}]" if extra else ""))
        if len(strays) > 40:
            print(f"  ... and {len(strays) - 40} more")
        print()
        print("RESULT: REFUSED — the open tree contains code nobody allowed. "
              "This is the denylist failure mode and it publishes by default.")
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        return 2
    print(f"EXCLUSION_OK   nothing outside the allowlist reached the tree")
    print(f"tree: {tree}")
    print()

    # THE PAYLOAD'S OWN SUITE, RUN IN THE PAYLOAD.
    #
    # Promoted from discovery method to standing guard. Every correction to the
    # publishability criterion — three of them — was found by staging the tree
    # and RUNNING it, never by re-reading the rule. A static criterion is a
    # proxy; the run is the fact, so the run is now the gate.
    #
    # SCOPE, STATED SO THE PASS IS NOT READ AS MORE THAN IT IS: this tree
    # carries the demo artifacts, because the measurement below needs them. So
    # the artifact-dependent tests RUN here rather than skipping, and the
    # fresh-clone skip path is NOT exercised by this gate. That path is
    # exercised by staging with `stage_open_core.py`, which omits them.
    suite_result = _run_payload_suite(tree)
    print(suite_result["line"])
    if suite_result["returncode"] != 0:
        print()
        print("RESULT: REFUSED — the published tree ships a suite that does not "
              "pass in it. A repository whose claim is that it checks its own "
              "work cannot ship a red check.")
        print(suite_result["tail"])
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        return 2
    print()

    child = workdir / "child.py"
    child.write_text("import json\n" + _CHILD, encoding="utf-8")
    # Already dotted submodule paths — do NOT re-stem. Taking `.stem` of
    # "providers.openai_provider" yields "openai_provider", which imports
    # nothing under tekmyra and would pass the leak test by failing to look.
    names = ",".join(CLOSED_MODULES)

    result = subprocess.run(
        [sys.executable, str(child), str(tree), str(REPO_ROOT), names],
        capture_output=True,
        text=True,
        cwd=tree,
    )
    output = result.stdout.strip()
    print(output)
    if result.stderr.strip() and result.returncode not in (3, 4):
        print(result.stderr.strip()[-800:], file=sys.stderr)

    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    if result.returncode == 3:
        print("RESULT: REFUSED — the closed half was still reachable. A number "
              "measured here would describe the closed build. Not a seam verdict.")
        return 3
    if result.returncode == 4:
        print("RESULT: NOT YET — the open core does not build standalone.")
        print("The frames above are the remaining seam work. This is the expected")
        print("state until the extractions land; it goes green when they do.")
        return 4
    if result.returncode == 5:
        print("RESULT: WRONG BUILD — the tree had the closed half deleted but the "
              "benchmark did not select the open store. The number above does not "
              "describe the open core.")
        return 5
    if result.returncode == 6:
        print("RESULT: SEAM PROVEN, NUMBERS NOT MEASURED — no model artifact.")
        print("Everything this command exists to check has passed: the surface was")
        print("built from the allowlist, nothing outside it reached the tree, the")
        print("closed half was proven unimportable FROM WITHIN THE BUILT TREE, and")
        print("the payload's own suite ran")
        print("green inside the built tree. Only the benchmark did not run, because")
        print("the trained model is a separate download and this clone has none.")
        print("That is the expected state on a fresh clone — see 'You need a model'")
        print("in the README. No compression figure is printed, deliberately: the")
        print("seam is proven, the numbers are not, and the two are not the same")
        print("claim.")
        return 6
    if result.returncode == 7:
        print("RESULT: SEAM PROVEN, MODEL PRESENT BUT UNUSABLE.")
        print("The seam checks all passed. An artifact directory IS here and the")
        print("router still did not load from it — so this is NOT the expected")
        print("fresh-clone state, and it is not a missing download. Read the")
        print("frames above: a corrupt, partial or version-mismatched artifact")
        print("looks exactly like a present one until something opens it.")
        return 7
    if result.returncode != 0:
        print(f"RESULT: ERROR — child exited {result.returncode}.")
        return 2

    payloads = json.loads(output.splitlines()[-1])
    ran = [p for p in payloads if p.get("status") == "RAN"]
    absent = [p["corpus"] for p in payloads if p.get("status") == "ABSENT"]
    if not ran:
        print("RESULT: REFUSED — no corpus was available to measure. The seam "
              "may be sound, but nothing here evidences it.")
        return 2

    first = ran[0]
    print(f"RESULT: OPEN CORE BUILDS AND REPRODUCES — "
          f"build={first['build']} via {first['anchor_store']} "
          f"(anchor resolution: {first['anchor_resolution_available']})")
    for payload in ran:
        headline = payload["headline"]
        headline_text = "WITHHELD" if headline is None else f"{headline:.4f}"
        # The corpus and its mean fixture size sit BESIDE the number, always.
        # A headline with no corpus attached is what let a 250-character result
        # stand in for realistic content for months.
        print(f"\n  corpus {payload['corpus']} — {payload['fixtures']} fixtures, "
              f"mean {payload['mean_chars']:,} chars")
        print(f"    headline {headline_text} over {payload['reached']} reached "
              f"of {payload['eligible']} eligible, {payload['refused']} refused")
        print(f"    locked spans {payload['locked_preserved']}/"
              f"{payload['locked_detected']}")
        reduction = payload.get("token_reduction_pct")
        if reduction is not None:
            print(f"    tokens {reduction:+.2f}%  "
                  f"(abstentions: {payload.get('abstention_rows')})")
        for name, stats in (payload.get("by_route") or {}).items():
            label = name + ("  (ABSTENTION)" if stats["abstention"] else "")
            pct = stats["reduction_pct"]
            pct_text = "  WITHHELD" if pct is None else f"{pct:+7.2f}%"
            print(f"      {label:42s} done={stats['done']:3d} "
                  f"refused={stats['refused']:3d} {pct_text}")
    if absent:
        print(f"\n  NOT MEASURED (absent from the built tree): {', '.join(absent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
