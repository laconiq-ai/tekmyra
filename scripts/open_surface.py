"""Compute the open surface as an ALLOWLIST OVER PATHS, from measurement.

Vega's rule, and the reason this module exists rather than a list someone
maintains by hand: the open surface is an allowlist over PATHS — a module graph
alone cannot decide ``.cache/`` or ``benchmarks/results/`` — built by three
tests applied in order.

1. RUNTIME FLOOR — the import closure of the documented public entry points,
   computed with DEFERRED (in-function) imports INCLUDED. A top-level-only
   closure ships a core that imports cleanly and then crashes the moment it
   selects the learned compressor, because that import happens inside a
   function. Ten of the modules in this closure are reachable ONLY that way.

2. CONTRIBUTOR PROMISE — what a stranger needs to build, test and extend.
   Note "test-reached" is NOT usable as a criterion: the suite imports 88 of
   96 modules, so it would readmit almost everything the exclusion removes.

3. CATEGORICAL EXCLUSION — money/quota/metering, tenancy/audit/recovery,
   internal ops tooling, trained artifacts and internal corpora, calibration
   thresholds, the PII rule corpus, paid-provider adapters, historical
   results. **This OVERRIDES 1 and 2.** If an open module imports an excluded
   one, that is a seam defect to CUT, not a licence to publish the excluded
   module.

WHY AN ALLOWLIST AND NOT THE DENYLIST WE HAD. ``verify_open_core.py`` decided
the surface with eight closed filenames and copied the rest. Measured at
d30ea62 that publishes 88 modules / 22,266 lines against a runtime floor of
far fewer — every module nobody thought about shipped, including a live
OpenAI adapter that reads ``OPENAI_API_KEY`` and a module of spend caps. **A
denylist publishes by default.** The failure is silent and it is one-way.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

from public_corpus_notice import PUBLIC_CORPUS_ROOTS

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "tekmyra"

# The documented public entry points. README.md documents the benchmark CLI;
# the pipeline API is what an embedder calls. Everything else must be REACHED
# from these, never added because it looked useful.
ENTRY_POINTS = (
    "tekmyra.benchmark",
    "tekmyra.benchmark_compressor",
    "tekmyra.runtime",
)

# Rule 3. Each entry states the CATEGORY it is excluded under, so the list can
# be argued with per module instead of trusted wholesale.
#: Present in the source repository, absent from every staged payload — the
#: staging step reads its root documents (LICENSE, README, …) from here and
#: does not copy the directory itself. Used to tell "the tree these exclusions
#: are applied FROM" apart from "the tree they have already been applied TO",
#: which are the two trees `validate_exclusions` has to answer differently.
SOURCE_ONLY_MARKER = "open_core_files"

CATEGORICAL_EXCLUSIONS: dict[str, str] = {
    "tekmyra/anchor_store.py": "tenancy — the anchorization implementation",
    "tekmyra/shared_anchor_store.py": "tenancy — multi-tenant enforcement",
    "tekmyra/service.py": "internal ops — orchestration, never on the compressor path",
    "tekmyra/benchmark_service.py": "internal ops",
    "tekmyra/recovery_server.py": "tenancy/recovery",
    "tekmyra/recovery_api.py": "tenancy/recovery",
    "tekmyra/shadow_harness.py": "internal ops",
    "tekmyra/shadow_cli.py": "internal ops",
    "tekmyra/budget_policy.py": "money/quota — route classification and spend caps",
    "tekmyra/eligibility.py": "money/quota — imports budget_policy",
    "tekmyra/credit_balance.py": "money/quota",
    "tekmyra/spend_ledger.py": "money/quota — the spend ledger",
    "tekmyra/global_rate_limit.py": "money/quota — rate limiting",
    "tekmyra/governed_provider_eval.py": "paid-provider evaluation",
    "tekmyra/providers/openai_provider.py": "paid-provider adapter — posts to api.openai.com, reads OPENAI_API_KEY",
    "tekmyra/providers/anthropic.py": "paid-provider adapter",
    "tekmyra/providers/deepseek.py": "paid-provider adapter",
    "tekmyra/providers/gemini.py": "paid-provider adapter",
    "tekmyra/providers/kimchi.py": "paid-provider adapter",
    "tekmyra/providers/openrouter.py": "paid-provider adapter",
    "tekmyra/dashboard.py": "internal ops tooling",
    "tekmyra/dashboard_app.py": "internal ops tooling",
    "tekmyra/review_app.py": "internal ops tooling",
    "tekmyra/review_store.py": "internal ops tooling",
    "tekmyra/telemetry.py": "internal ops tooling",
}


# ---------------------------------------------------------------------------
# THE REST OF THE ARTIFACT — everything that is not a package module.
#
# ADDED 2026-08-09 after an independent QA seat returned NO-GO on the first
# inversion. The allowlist governed ``tekmyra/**/*.py`` and the check
# scanned exactly that, so it verified 54 files of a 2,033-file tree — 97% of
# the artifact was structurally invisible — and printed EXCLUSION_OK. Present
# and unseen: four paid-provider runners, two trained model pickles, internal
# corporate-sellability tooling, and a real Ed25519 PRIVATE KEY.
#
# THE ROOT CAUSE WAS NOT THE SCAN. ``build_open_tree`` copied whole directories
# — every support dir and the entire ``.cache`` — which is denylist behaviour
# one level up from the one we had just removed. Narrowing the scan alone would
# have failed every legitimate config and README, and the reflex under that
# pressure is to allowlist each failure one at a time, which rebuilds the
# denylist with a green verdict. So both halves change together: the tree is
# built from these rules, and the check enumerates the tree against them.
#
# Each rule is a POSITIVE statement of what the artifact is made of, with its
# reason. Seven of them describe 2,033 files' worth of former guesswork.


@dataclass(frozen=True)
class TreeCategory:
    """One KIND of thing the artifact is made of, not one path that failed.

    The distinction is the whole difference between this and a relapse. A list
    of globs accumulated by adding whatever the check rejected is a denylist
    with extra steps: each entry answers "did this fail?" rather than "what is
    this artifact made of?", and nothing tells the two apart from outside.

    So a category states a ROOT, an EXTENSION SET, and a PRINCIPLE. Anything
    matching the root but not the principle is refused, which is what makes the
    principle load-bearing rather than decorative — see CORPORA, where the
    extension set is the only thing keeping the corpus generators out of a
    published artifact.
    """

    name: str
    roots: tuple[str, ...]
    extensions: tuple[str, ...]  # empty = any extension permitted under root
    reason: str
    #: An optional COMPUTED membership test, for categories whose rule cannot be
    #: written as a path shape. Added for the test suite, where the rule is "its
    #: import closure lies inside the allowlist" — a fact about the file's
    #: contents, not its name. A curated list of filenames would be a denylist
    #: wearing a new name: the next test written against a closed module ships
    #: and fails on a stranger's machine, and nothing catches it.
    predicate: Callable[[str], bool] | None = None

    def matches(self, relative_path: str) -> bool:
        if not any(relative_path.startswith(root) for root in self.roots):
            return False
        if self.extensions and not any(
            relative_path.endswith(ext) for ext in self.extensions
        ):
            return False
        if self.predicate is not None and not self.predicate(relative_path):
            return False
        return True


# ---------------------------------------------------------------------------
# THE VERIFICATION SURFACE — what lets a stranger check our work rather than
# take it on trust.
#
# A library-plus-corpora cannot say "you can verify our seam yourself", and for
# a project whose entire claim is that it checks its own output, that silence
# argues against us on our own front page.
#
# SCRIPTS ARE NAMED INDIVIDUALLY AND THAT IS DELIBERATE. `scripts/` holds 50
# files and 28 of them must never leave the building — release signing,
# production probes, rights signing, secrets preparation. Publishing the
# DIRECTORY is the denylist failure one level up, and this project has already
# been bitten by that shape twice: a module denylist that shipped 88 modules
# including a live OpenAI adapter, and a copytree that put 2,033 files
# including a real Ed25519 private key into a publication tool's output. A
# third instance would not be bad luck.
#
# Each entry is reasoned, and `validate_publishable_scripts` raises if one names
# a file that does not exist — an exemption matching nothing reads as a decision
# in force while doing nothing at all.
PUBLISHABLE_SCRIPTS: dict[str, str] = {
    "verify_open_core.py": "proves the closed half is unimportable, then measures; refuses a number otherwise",
    "open_surface.py": "the rule deciding what is published, so the decision is auditable rather than asserted",
    "stage_open_core.py": "builds the tree, so a cloner can re-derive the artifact rather than trust it",
    "public_corpus_notice.py": "derives public-corpus attribution and refuses an uncovered source",
    "payload_manifest.py": "every published file re-derived from git or from a script at a stated commit",
    "enumerate_modules_vs_aspects.py": "the published surface against the provisional's aspects, by script",
    # verify_artifacts.py IS DELIBERATELY ABSENT. It shipped in an earlier
    # draft as "the reproduce path". Measured in the payload, BOTH its modes
    # exit 2: `reproduce` needs a retrainer, a training corpus and a committed
    # manifest that are all correctly withheld, and `check` needs that same
    # manifest — which, read in full before admitting it, turns out to be STALE
    # against the artifacts we publish (it predates the 2026-08-08 compressor
    # retrain and was never regenerated).
    #
    # A shipped check that FAILS against the bundle we point readers at is worse
    # than no check: it manufactures a false alarm on the one artifact we are
    # asking them to trust. Verification is carried by payload_manifest.py and
    # verify_open_core.py, which run green in the payload today.
}


#: Baseline data the shipped tests read at import time. Named individually, not
#: taken by directory, because one of them is a decision rather than a file a
#: test happens to open.
#:
#: ``public_corpus_baseline.json`` carries the 62.1458% headline measured on the
#: four shipped ``public_*`` corpora. Its per-corpus sha256 digests bind the
#: visible fixture bytes to the baseline, so a reader can verify that the corpus
#: they received is byte-for-byte the one the published figure measured.
PUBLISHABLE_TEST_BASELINES: tuple[str, ...] = (
    "unbuilt_routes.json",
    "codec_effectiveness.json",
    "public_corpus_baseline.json",
)

#: THE FOURTH AXIS: SUBJECT MATTER, WHICH NO IMPORT COMPUTATION CAN SEE.
#:
#: A test can have every import inside the allowlist, need no unstaged corpus,
#: open no unpublished script — and still be out of scope, because what it
#: ASSERTS ABOUT is the internal repository rather than the published package.
#: Several of these fail in the payload precisely BECAUSE the payload is
#: correct: ``test_open_surface_allowlist`` requires the closed modules to
#: exist, and their absence is the whole point of the artifact.
#:
#: Subject matter cannot be computed, so this list is CURATED — and therefore
#: DECLARED, with a stated reason per entry, so the next reader meets a reasoned
#: register rather than a silent hole. That is the honest form of a denylist:
#: small, named, and sitting beside the computed rule that handles everything
#: computable.
EXCLUDED_META_TESTS: dict[str, str] = {
    "test_open_surface_allowlist.py": "asserts on THIS repository's exclusion list; the closed modules it requires are absent from the payload by design",
    "test_open_core_import_graph.py": "asserts the closed half exists in the tree; in the payload its absence is the correct state",
    "test_nova_rehearsal_deploy.py": "deployment rehearsal files, which are internal ops and ship nowhere",
    "test_global_rate_limit_deploy.py": "deployment configuration for a module excluded under money/quota",
    "test_seal_sheet_gate_count.py": "counts gates in a release-evidence document that is not part of the package",
    "test_package_install_smoke.py": "asserts the internal distribution's shape, not the published one",
    "test_resources.py": "asserts the internal resource layout, including docs/ADR for adrs.py — a module the import closure does not reach and the payload does not contain",
    "test_monitoring_contract.py": "static contract checks over deploy/monitoring/tekmyra-alerts.yml; deployment configuration is internal ops and ships nowhere",
    "test_seam_source_posture.py": "asserts THIS repository's licence posture and that the withheld modules are still inside the packaging glob; every one of those is correctly FALSE in the payload, so shipping it made verify_open_core.py refuse a healthy tree",
    "test_payload_artifact_integrity.py": "stages a payload out of THIS repository and reads stage_open_core's dependency declarations; it also builds a wheel and installs it into a throwaway venv, which is a release-gate cost the payload's own consumers must not inherit",
    # test_wp7_real_runtime.py was briefly listed here after it reported
    # 6 failed / 9 errors against a healthy payload — but pulling the file
    # threw away the only end-to-end pipeline runs a cloner had. It ships
    # again, with the broken pieces gated on their actual dependencies inside
    # the file: the metrics classes skip when the unpublished evaluator's
    # artifact is absent, the three commercial-pinned sample fixtures skip on
    # the open build, and the sample deterministically includes a PII fixture
    # that runs on both builds — the seeded draw alone put both of its PII
    # fixtures in the commercial-only set, which made the leakage and recall
    # assertions dormant exactly where the skip message claimed them.
}


@lru_cache(maxsize=1)
def _open_module_names() -> frozenset[str]:
    """Module names in the allowlist, for deciding what a test may import."""
    names: set[str] = set()
    for path in build_allowlist()["allowed_paths"]:
        module = path[: -len(".py")].replace("/", ".")
        names.add(module)
        if module.endswith(".__init__"):
            names.add(module[: -len(".__init__")])
    return frozenset(names)


@lru_cache(maxsize=1)
def _unstaged_corpora() -> tuple[str, ...]:
    """Corpora present in the repository that the published tree does NOT carry."""
    staged = {
        root.rstrip("/").rsplit("/", 1)[-1]
        for category in OPEN_TREE_CATEGORIES
        if category.name == "corpora"
        for root in category.roots
    }
    fixtures = REPO_ROOT / "benchmarks" / "fixtures"
    if not fixtures.is_dir():
        return ()
    return tuple(
        sorted(p.name for p in fixtures.iterdir() if p.is_dir() and p.name not in staged)
    )


#: Top-level first-party packages a test may name. Everything under one of
#: these is resolved against what the payload ACTUALLY contains.
FIRST_PARTY_ROOTS = ("tekmyra", "scripts", "benchmarks", "training", "tests")

#: Third-party modules that are NOT core dependencies of the published package.
#: A test importing one of these goes red on a fresh clone that installed only
#: what pyproject declares — which is the footgun class, not an instance of a
#: test being unlucky.
NON_CORE_THIRD_PARTY = (
    "onnxruntime",
    "torch",
    "transformers",
    "llmlingua",
    "sentence_transformers",
    "optimum",
)


def _imported_modules(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names |= {f"{node.module}.{alias.name}" for alias in node.names}
    return names


def _import_is_satisfied(name: str) -> bool:
    """Will this import resolve inside the PUBLISHED tree?"""
    root = name.split(".", 1)[0]

    if root in NON_CORE_THIRD_PARTY:
        return False
    if root not in FIRST_PARTY_ROOTS:
        return True  # stdlib or a declared core dependency

    if root == "tekmyra":
        modules = _open_module_names()
        return name in modules or name.rsplit(".", 1)[0] in modules
    if root == "scripts":
        parts = name.split(".")
        return len(parts) >= 2 and f"{parts[1]}.py" in PUBLISHABLE_SCRIPTS
    if root == "tests":
        parts = name.split(".")
        if len(parts) < 2:
            return True
        # A helper is published only if it passes this same test — the rule is
        # recursive because a clean test importing a dirty helper is a broken
        # clone, and the helper is where that breaks silently.
        return test_is_publishable(f"tests/{parts[1]}.py")
    # benchmarks/ ships fixture DATA only; training/ ships nothing.
    return False


def test_is_publishable(relative_path: str) -> bool:
    """Can this file be collected and run inside the PUBLISHED tree?

    Three questions, all about the file's CONTENTS rather than its name.

    1. Does every import resolve in the payload — first-party against what is
       actually published, third-party against the declared core dependencies?
    2. Does it need a corpus the published tree does not carry?
    3. (Enforced by the run, not here.) Does it collect?

    **THIS CRITERION HAS BEEN WRONG TWICE, BOTH TIMES BY BEING NARROWER THAN
    THE CLAIM IT STANDS FOR, AND BOTH TIMES IT LOOKED RIGHT.**

    First draft matched a bare corpus name anywhere in the file. That excluded
    seven tests that merely MENTION one — including the regression test whose
    whole job is asserting a default no longer points at ``learned_train``. A
    criterion that drops the guard because it names the thing it forbids is
    worse than no criterion. Fixed by matching the corpus as a PATH.

    Second draft checked only imports beginning ``tekmyra``. Staging on
    that basis produced a payload whose suite raised **44 collection errors** —
    tests importing ``scripts.*``, ``benchmarks.*``, ``training.*``, unpublished
    ``tests.*`` helpers, and ``onnxruntime``. The claim was "its import closure
    lies inside the allowlist"; the instrument read one prefix of it.

    So the static answer is a PRE-FILTER and never the verdict. The verdict is
    the suite running green inside the staged tree, which ``verify_open_core``
    now performs. A static criterion is a proxy; the run is the fact.
    """
    path = REPO_ROOT / relative_path
    if not path.is_file():
        return False
    if path.name in EXCLUDED_META_TESTS:
        return False
    source = path.read_text(encoding="utf-8", errors="replace")

    for name in _imported_modules(source):
        if not _import_is_satisfied(name):
            return False

    for corpus in _unstaged_corpora():
        escaped = re.escape(corpus)
        if re.search(rf"fixtures[/\\]{escaped}\b", source) or re.search(
            rf'["\']{escaped}["\']\s*\)', source
        ):
            return False

    # A FILE PATH IS NOT AN IMPORT, and the import check cannot see one. Several
    # tests read a script's SOURCE — to assert on its AST, or to run it — rather
    # than importing it. Staging on the import check alone shipped a test that
    # opens ``scripts/wp7_fast_track_decision_record.py``, a file that must never
    # be published, and it failed at collection in the payload.
    for match in re.finditer(r'["\']?scripts["\']?\s*[/,]\s*["\']([\w.]+\.py)["\']', source):
        if match.group(1) not in PUBLISHABLE_SCRIPTS:
            return False
    return True


class PublishableScriptError(RuntimeError):
    """A named publishable script does not exist."""


def validate_publishable_scripts() -> None:
    """A named entry that matches no file enforces nothing and reads as a decision.

    Covers all three curated lists. The failure this prevents is not
    hypothetical: an earlier categorical-exclusion list named
    ``providers/deepseek_provider.py`` when the file is ``providers/deepseek.py``,
    so the entry silently protected nothing while reading as protection.
    """
    problems: list[str] = []
    problems += [
        f"scripts/{name}"
        for name in PUBLISHABLE_SCRIPTS
        if not (REPO_ROOT / "scripts" / name).is_file()
    ]
    problems += [
        f"tests/baselines/{name}"
        for name in PUBLISHABLE_TEST_BASELINES
        if not (REPO_ROOT / "tests" / "baselines" / name).is_file()
    ]
    problems += [
        f"tests/{name}"
        for name in EXCLUDED_META_TESTS
        if not (REPO_ROOT / "tests" / name).is_file()
    ]
    if problems:
        raise PublishableScriptError(
            f"list entr(ies) named but absent: {sorted(problems)}. Fix the name "
            "or delete the entry — an entry that matches no file reads as a "
            "decision that was taken and is enforced, while doing nothing."
        )


OPEN_TREE_CATEGORIES: tuple[TreeCategory, ...] = (
    TreeCategory(
        name="verification scripts",
        roots=("scripts/",),
        extensions=(".py",),
        predicate=lambda rel: rel.rsplit("/", 1)[-1] in PUBLISHABLE_SCRIPTS,
        reason=(
            "the tooling a stranger needs to check our claims instead of "
            "believing them. NAMED INDIVIDUALLY: 28 of the 50 files under "
            "scripts/ must never ship, and publishing the directory is the "
            "denylist failure this allowlist exists to prevent."
        ),
    ),
    TreeCategory(
        name="test suite",
        roots=("tests/",),
        extensions=(".py",),
        predicate=test_is_publishable,
        reason=(
            "COMPUTED, never curated: a test ships iff its import closure lies "
            "inside the allowlist and it needs no unstaged corpus. A hand-picked "
            "list of filenames is a denylist wearing a new name — the next test "
            "written against a closed module would ship and fail on a stranger's "
            "machine with nothing to catch it."
        ),
    ),
    TreeCategory(
        name="test baselines",
        roots=("tests/baselines/",),
        extensions=(".json",),
        predicate=lambda rel: rel.rsplit("/", 1)[-1] in PUBLISHABLE_TEST_BASELINES,
        reason=(
            "the ratchet baselines the shipped tests read at import time. Named "
            "individually rather than by directory: public_corpus_baseline.json "
            "is the evidence artifact for a figure measured on corpora this tree "
            "does not carry, and whether it ships is a decision about what we "
            "claim, not a consequence of what a test happens to open."
        ),
    ),
    TreeCategory(
        name="runtime configuration",
        roots=("configs/",),
        extensions=(".yaml",),
        reason=(
            "the package reads these at import time. Restricted to .yaml so a "
            "script or key dropped into configs/ is not admitted by proximity."
        ),
    ),
    TreeCategory(
        name="schemas",
        roots=("contracts/",),
        extensions=(),
        reason=(
            "the contract definitions the package validates against. Any "
            "extension, because a schema directory is data by construction and "
            "the categorical-exclusion list governs what may live there."
        ),
    ),
    TreeCategory(
        name="corpora",
        roots=(
            "benchmarks/fixtures/synthetic/",
            "benchmarks/fixtures/long_context_v1/",
            *PUBLIC_CORPUS_ROOTS,
        ),
        extensions=(".json", ".md"),
        reason=(
            "the measurement substrate, including the four attributed public "
            "corpora. A CORPUS CONTRIBUTES DATA, NEVER CODE — the extension set "
            "is what keeps fixture_templates.py and "
            "generate_long_context_fixtures.py out, and both live inside these "
            "roots. Dropping the restriction would publish the generators."
        ),
    ),
    TreeCategory(
        name="demo artifacts",
        roots=(".cache/compressor/1.0.0/", ".cache/router/1.0.0/"),
        extensions=(),
        reason=(
            "the trained models the verification run measures. NOT part of the "
            "published open core — Vega Q2 keeps trained artifacts closed and "
            "ships a demo model as a release asset with a recorded sha256 — but "
            "the verification tree needs one to produce a number at all. The "
            "version-pinned roots are deliberate: .cache holds 24 directories "
            "including two signing dirs, and naming '.cache/' would readmit "
            "every one of them."
        ),
    ),
)


# REFUSED UNCONDITIONALLY, even if a rule above would match. Defence in depth,
# and deliberately not merely "absent from the rules": the class that was
# invisible must become the class that is provably caught, so it gets its own
# check that cannot be satisfied by someone adding a permissive rule.
#
# .cache held THREE .pem files including a live signing private key, copied
# wholesale into a publication tool's output. Nothing was published and there
# was no exposure — but the guard that exists to prevent one could not see it.
NEVER_IN_TREE: tuple[str, ...] = (
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "*.jks",
    "id_rsa*",
    "id_ed25519*",
    "*.env",
    "*credentials*",
    "*secret*",
)


def is_key_material(relative_path: str) -> bool:
    """Does this path look like a credential, whatever else permits it?"""
    from fnmatch import fnmatch

    name = relative_path.rsplit("/", 1)[-1].lower()
    return any(fnmatch(name, pattern) for pattern in NEVER_IN_TREE)


def tree_strays(tree: Path, allowed_package_paths: set[str]) -> list[tuple[str, str]]:
    """Every file in the WHOLE tree that nobody decided to publish.

    Enumerates the ARTIFACT — every file, every extension, no subtree filter.
    Vega's correction after the NO-GO: "I specified a property of the
    computation when I needed a property of the artifact." Asserting on the
    import closure proves something about how the list was computed; asserting
    on the shipped tree proves what is in it, and copying is a step that writes
    into the tree.
    """
    strays: list[tuple[str, str]] = []
    for path in sorted(tree.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(tree).as_posix()
        if is_key_material(rel):
            strays.append((rel, "KEY MATERIAL — refused unconditionally"))
            continue
        if rel in allowed_package_paths:
            continue
        if any(cat.matches(rel) for cat in OPEN_TREE_CATEGORIES):
            continue
        strays.append((rel, "matched by no rule — nobody decided to publish it"))
    return strays


class ExclusionListError(RuntimeError):
    """A categorical exclusion names a path that does not exist."""


def validate_exclusions() -> None:
    """A named exemption that matches no file is worse than no exemption.

    It reads as a decision that was taken and is enforced, while doing
    nothing at all. This is not hypothetical: the first draft of the list
    above named ``providers/deepseek_provider.py`` and
    ``providers/openrouter_provider.py``. The real files are
    ``providers/deepseek.py`` and ``providers/openrouter.py``, so both
    entries silently protected nothing, and three further paid adapters
    (anthropic, gemini, kimchi) were missing outright — caught by measuring
    the gap, not by rereading the list.

    So this RAISES rather than skipping. A filter that cannot represent a
    case must raise.

    IN A PUBLISHED TREE THE SAME MEASUREMENT MEANS THE OPPOSITE. The staged
    payload is the tree these exclusions have already been applied to, so every
    entry there SHOULD match no file — and this guard, run from inside it,
    fired on its own success. That is not hypothetical either: a reader who
    followed the README to its one recommended command, ``verify_open_core.py``,
    got this traceback listing all 25 withheld modules before a single thing
    had been proven. One check could not tell "the list went stale" from "the
    list did its job", and reported the second as the first.

    The two trees are told apart by ``open_core_files/``, the directory the
    staging step reads its root documents from. It exists only in the source
    repository, and a tree without it cannot be a staging source. In a
    published tree the check INVERTS, to the assertion actually worth making
    to a reader holding that tree: every withheld module is absent, and one
    that is present is a leak. That is stricter than skipping, and it is the
    question the reader came to ask.
    """
    if not (REPO_ROOT / SOURCE_ONLY_MARKER).exists():
        leaked = sorted(p for p in CATEGORICAL_EXCLUSIONS if (REPO_ROOT / p).exists())
        if leaked:
            raise ExclusionListError(
                "withheld module(s) are PRESENT in this published tree: "
                f"{leaked}. A categorical exclusion that shipped anyway is a "
                "leak, not a stale entry."
            )
        return
    missing = sorted(p for p in CATEGORICAL_EXCLUSIONS if not (REPO_ROOT / p).exists())
    if missing:
        raise ExclusionListError(
            "categorical exclusion(s) name paths that do not exist, so they "
            f"enforce nothing: {missing}. Fix the path or delete the entry — "
            "an exemption that matches no file reads as a decision in force."
        )


@dataclass
class Closure:
    """The measured runtime floor, with each module's reason for inclusion."""

    modules: set[str] = field(default_factory=set)
    deferred_only: set[str] = field(default_factory=set)
    reached_from: dict[str, str] = field(default_factory=dict)


def _module_path(module: str) -> Path | None:
    rel = Path(*module.split("."))
    for candidate in (REPO_ROOT / rel.with_suffix(".py"), REPO_ROOT / rel / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _imports(tree: ast.AST, module: str) -> tuple[set[str], set[str]]:
    """Return (all imported modules, those imported only inside a function).

    The deferred set is tracked separately because it is the trap: a
    top-level-only walk produces a closure that imports cleanly and then
    fails at runtime on the first in-function import.
    """
    top_level: set[str] = set()
    deferred: set[str] = set()

    function_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                function_nodes.add(id(inner))

    package = module.rsplit(".", 1)[0] if "." in module else module
    for node in ast.walk(tree):
        names: set[str] = set()
        if isinstance(node, ast.Import):
            names = {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                base = package
                for _ in range(node.level - 1):
                    base = base.rsplit(".", 1)[0]
                names = {f"{base}.{node.module}" if node.module else base}
            elif node.module:
                names = {node.module}
                # `from pkg.mod import name` may name a submodule, not a symbol.
                names |= {f"{node.module}.{a.name}" for a in node.names}
        else:
            continue
        names = {n for n in names if n.startswith(PACKAGE)}
        if id(node) in function_nodes:
            deferred |= names
        else:
            top_level |= names
    return top_level | deferred, deferred


def compute_closure(entry_points: tuple[str, ...] = ENTRY_POINTS) -> Closure:
    """Walk the import graph from the entry points, deferred imports included."""
    closure = Closure()
    queue = list(entry_points)
    seen_deferred: set[str] = set()
    for entry in entry_points:
        closure.reached_from[entry] = "entry point"

    while queue:
        module = queue.pop()
        if module in closure.modules:
            continue
        path = _module_path(module)
        if path is None:
            continue
        closure.modules.add(module)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found, deferred = _imports(tree, module)
        seen_deferred |= deferred
        for name in found:
            if _module_path(name) is None:
                continue
            closure.reached_from.setdefault(name, module)
            if name not in closure.modules:
                queue.append(name)

    # A module is deferred-only if nothing imports it at module level.
    for module in closure.modules:
        if module in seen_deferred:
            closure.deferred_only.add(module)
    return closure


def closure_paths(closure: Closure) -> set[str]:
    """Repo-relative paths for every module in the closure."""
    paths: set[str] = set()
    for module in closure.modules:
        path = _module_path(module)
        if path is not None:
            paths.add(str(path.relative_to(REPO_ROOT)))
    return paths


def build_allowlist() -> dict:
    """Apply the three rules in order and report what each one decided."""
    validate_exclusions()
    closure = compute_closure()
    floor = closure_paths(closure)
    excluded = {p for p in floor if p in CATEGORICAL_EXCLUSIONS}
    allowed = sorted(floor - excluded)
    return {
        "entry_points": list(ENTRY_POINTS),
        "runtime_floor_modules": len(floor),
        "excluded_from_floor": sorted(excluded),
        "allowed_paths": allowed,
        "deferred_only_modules": sorted(closure.deferred_only),
    }


if __name__ == "__main__":
    print(json.dumps(build_allowlist(), indent=2))
