# TekMyra Architecture

TekMyra shortens what an application sends to a language model, and refuses to
touch what it cannot vouch for. This document explains how the pieces fit, in
the order a request meets them.

**Every claim here names the file that implements it, and describes the code in
this repository rather than the system as designed.** Where the shipped build
does less than the commercial one, that is stated in the section where you would
otherwise assume otherwise. Where a number is easy to misread, the way to
misread it is spelled out.

---

## 1. The one rule: decide before you remove

Most compression tools shorten first and repair after. TekMyra inverts that. It
works out what must survive *before* it removes anything, and when it cannot
guarantee survival it declines: the content passes through unchanged, and the
run record says which mechanism declined and why.

Declining is a recorded outcome, not an error. It counts against every headline
number we publish: a build that refuses more would otherwise score better for
refusing, so refusals sit in the denominator; see the README's `Numbers`
section, which reports the 14 refusals on `long_context_v1` measured against
the commercial build. "Refusal" and "abstention" are separate outcomes counted
from separate code paths; §2.5 below is where the mechanisms are kept apart.

---

## 2. The request path

### 2.1 Routing: "what kind of content is this?"

- `tekmyra/rule_router.py`: deterministic classification by content shape.
- `tekmyra/learned_router/`: the learned classifier, made up of `features.py`,
  `heads.py`, `calibration.py`, `model.py`, `tokenizer_window.py`, `router.py`,
  `config.py`.
- `tekmyra/hybrid_router.py`: combines the two. The learned router may
  **abstain** from a decision; when it does, the rule router's decision stands
  and the record carries the tag `LEARNED_ABSTAINED` with a reason
  (`hybrid_router.py:164-165`). Disagreements resolve conservatively.

`tekmyra/router_provenance.py` records which router decided and on what basis.

### 2.2 Protection: "what must not change?"

`tekmyra/protected_spans.py` detects spans whose exact wording carries the
meaning: identifiers, amounts, dates, quoted terms. Detected spans are pinned
before any rewriting is considered; later stages receive them as constraints.

**The vocabulary modules are not part of this.** `pii_vocabulary.py`,
`rights_vocabulary.py` and `corpus_vocabulary.py` sit next to
`protected_spans.py` in the package and look like its inputs. They are not: the
only first-party modules `protected_spans.py` imports are `contracts`,
`content_hash` and `fixtures`.
The vocabularies are consumed by `fixtures.py`, `benchmark.py`, `baselines.py`,
`reporting.py` and `benchmark_compressor.py`; they serve fixture generation and
measurement, not detection. Detection is regex- and rule-driven inside
`protected_spans.py` itself. Verify with:

```bash
python -c "import ast; print(sorted({n.module.split('.')[-1] for n in ast.walk(ast.parse(open('tekmyra/protected_spans.py').read())) if isinstance(n, ast.ImportFrom) and n.module and n.module.split('.')[0] == 'tekmyra'}))"
```

That prints exactly the three names above. It filters to first-party imports on
purpose: the unfiltered form also lists `__future__`, `dataclasses` and
`typing`, so its output does not visibly confirm the sentence it is placed
under; a reader has to do the filtering in their head and take the claim on
trust, which is the one thing this document is trying not to ask of anyone.

### 2.3 Compression: "shorten only what is safe"

Two engine families, selected by route:

- **Deterministic codecs**, in `tekmyra/codecs/`: `json_codec.py`, `log_codec.py`,
  `code_codec.py`, `diff_codec.py`, `stacktrace_codec.py`, with `base.py`,
  `safe_mode.py` and `selector.py`. Same input, same output, every time. No
  model involved.
- **Learned compressor**, in `tekmyra/learned_compressor/`: `segmenter.py` →
  `features.py` → `model.py` (or `transformer_model.py` /
  `transformer_onnx_model.py`, executed through `tekmyra/onnx_runtime.py`) →
  `selector.py`, under a budget from `tekmyra/compression_budget.py`.

The model is a **versioned release artifact with a fingerprint**: `version`
fields and the sha256 capture in `router_provenance.py` are what make that
checkable. It is **not in this repository**; see §6.

**Not every route in the taxonomy has a codec.** The routes in
`UNIMPLEMENTED_CODEC_ROUTES` return `_unimplemented_result`, which fails closed:
they are roadmap entries, not working compressors. At the time of writing the
sixteen routes split six working codecs, five that pass through untouched
(`SAFE_TERMINAL_ROUTES`, §2.5), and five unbuilt. Both sets are derived from the
codec map, so count them rather than trusting the sentence:

```bash
python -c "from tekmyra.codecs.selector import _CODEC_MAP, SAFE_TERMINAL_ROUTES as s, UNIMPLEMENTED_CODEC_ROUTES as u; print(f'{len(_CODEC_MAP)} routes: {len(_CODEC_MAP)-len(s)-len(u)} working, {len(s)} passthrough, {len(u)} unbuilt')"
```

### 2.4 Verification: "prove the spans survived"

`tekmyra/verifier.py` re-checks every pinned span against the compressed output.
`entity_fidelity.py` and `numeric_fidelity.py` hold the span-class checks;
`sentence_boundaries.py` guards structural cuts.

`tekmyra/safety_gate.py` classifies risk (`classify_safety(features) ->
SafetyDecision`); `tekmyra/safety_merge.py` applies the merge that locks spans
(`apply_safety_merge`). Neither of these makes the final ship/decline call; see
the next section, which is where an earlier draft of this document was wrong.

### 2.5 The three ways TekMyra declines

"It abstains" is one phrase covering three distinct mechanisms with three
different owners. They fail differently, so they are worth separating.

**1. The router abstains.** The learned router declines to classify; the rule
router's decision stands. Owner: `hybrid_router.py:106-165`. Visible in the
record as `LEARNED_ABSTAINED` plus a reason.

**2. The codec declines to transform.** The routes in `SAFE_TERMINAL_ROUTES`
share one `_passthrough` codec that returns its input verbatim, and *the prose
codec also returns its input untouched on HIGH and CRITICAL risk tiers*. Owner:
the codec layer, driven by the risk tier from `safety_gate.py`.

That set is **derived from the codec map, not declared**, so ask the code rather
than trusting this sentence (an earlier version of it said "six routes" and the
answer is five):

```bash
python -c "from tekmyra.codecs.selector import SAFE_TERMINAL_ROUTES as s; print(len(s), sorted(r.value for r in s))"
```

**3. Verification fails.** `runtime.py` verifies (line 230). On failure it
retries through a fallback route if one exists (lines 258-283) and re-verifies.
**If the fallback also fails, it raises `TerminalVerificationError`**
(`runtime.py:285-295`). It does not quietly return the original: it refuses to
construct a response at all, because the alternative is returning output the
verifier has rejected. Callers must handle that exception.

That third case is the one to know about. A verifier failure is not a soft
degradation; it is terminal, by design, and it is the reason "a verifier failure
can never ship a degraded prompt" is true. `benchmark_compressor.py` catches it
(line 287), records the fixture as a refusal with its failed checks, and counts
its tokens at zero saving: a decline recorded, not swallowed.

**"Abstention" and "refusal" are two different counts, and the benchmark keeps
them apart.** They are computed from disjoint populations, which is easy to miss
because both mean "we did not compress this":

- a **refusal** is a fixture that raised `TerminalVerificationError`. It is
  appended to `failures` and the loop moves on, so it never reaches the results
  table at all (`benchmark_compressor.py:287-310`, then `continue`).
- an **abstention** is a fixture that COMPLETED, having been routed to
  `reversible_retrieval_mode`, filtered out of the results rows afterwards
  (`benchmark_compressor.py`, `abstention_rows`).

The report prints them on separate lines with separate labels, and this document
does not attribute either count to a single mechanism from the list above,
because the code does not.

### 2.6 Records

`tekmyra/contracts.py` holds the typed record contracts, and `runtime.py`
assembles the per-item record: the `CompressionSegment` and the
`CompressionResponse` returned by `_run_pipeline`.

**The per-run telemetry emitter and the HTML console are not in this
repository.** They are internal operations tooling and are excluded by category
in `scripts/open_surface.py`. If you want run records, `contracts.py` gives you
the shapes and `runtime.py` shows the assembly; the writer and the renderer are
yours to supply. The README's `What is deliberately absent` section states the
same exclusion.

---

## 3. Reading the numbers honestly

**A route name does not tell you whether compression happened**, and a per-route
token delta is not evidence that it did. Two things produce a token reduction:
the codec, and the safety merge's anchorization; the merge's saving lands
on whatever route the fixture happened to take.

On `long_context_v1` we found **two routes reporting non-zero token reductions
while their codec was a no-op on every fixture**: anchorization credited to
compression.

**That finding is an engineering note, not a shipped artifact.** It is recorded
in a source comment above the `codec_transformed` field in `runtime.py`, and no
result file in this repository carries it: reproducing it needs the model
artifact, which is fetched separately. Treat it as the reason the field exists,
and check your own runs rather than taking the number from us.

The fix is in the record, not in the prose: each item carries
`codec_transformed`, computed by comparing the codec's own input against its
output. Both `apply_codec` call sites (initial and fallback) are fed
`merge_result.transformed_text`, so that comparison is exact on either path.
When you read our numbers, or your own, `codec_transformed` is the field that
separates "the compressor worked" from "something upstream shortened the text".

---

## 4. Recovery, and what this build cannot do

`tekmyra/anchor_interface.py` defines the anchor contract: content removed on a
reversible route is given an addressable identifier rather than simply being
gone. `AnchorStoreProtocol` (line 49) is that interface. Five modules use it:
`runtime.py`, `verifier.py`, `safety_merge.py`, `router_provenance.py`,
`benchmark_compressor.py`.

**The store this repository ships resolves nothing.** `NonResolvingAnchorStore`
(line 72) mints identifiers and then cannot resolve them:
`get()` is `return None` for every identifier, including ones it minted itself
(line 117-119). The encrypted, tenant-partitioned, retention-bounded,
retrieval-audited store is the commercial half.

The consequence is not a gap, it is a behaviour: **this build refuses more than
the commercial one.** It cannot ask "was that dropped span recoverable?", so it
treats the drop as unrecoverable and declines. That mechanism is checkable here:
`verifier.py`'s anchor-resolvability check calls `anchor_store.get()`, and this
store's answer is always `None`.

The size of the difference is not checkable here. On our live PII route the
commercial build completes 30 of 30 rows and this one refuses 4: a measurement
on production traffic, which no shipped fixture contains. It is stated because
withholding it would be worse, and flagged as what it is: our claim about a
build you do not have, not a number this repository can reproduce.

It is not an empty store. It is a store that *cannot* resolve, which is the
stronger position: a build that returned `None` because the store happened to
be empty would behave differently once someone populated it.

---

## 5. Artifact resolution

`tekmyra/artifact_paths.py` resolves **the directory holding the model
artifact**, not every artifact read and write in the system. `resolve()` (line
101) returns a `Resolution` that reports whether it `found` anything, its
`fallback()`, and a `describe()` string naming every path tried, so a wrong
working directory produces a findable failure rather than a silent one.

Two consumers: `learned_compressor/compressor.py:64` (the resolving call) and
`learned_router/config.py:74`. `router_provenance.py:185` references only the
environment variable name, in an error message.

Set `TEKMYRA_ARTIFACT_ROOT` to override. Resolution order and troubleshooting
are in the README.

---

## 6. The open/closed seam

**In this repository (Apache-2.0):** the router stack, both engine families, the
protected-span detector, the verifier and fidelity checks, the safety gate and
merge, the anchor *interface* with its non-resolving store, the benchmark
harness, the fixture corpora, and the contracts.

**Not in this repository:** the trained model artifact and the training stack;
the resolving anchor store; the economics gate (its data contract is present
because the fixture schema references it, but nothing computes it); paid-provider
adapters; tenancy, audit and authentication; the run-telemetry emitter and
console.

The model artifact is a **release asset, fetched separately**; the README's
`You need a model` section is the procedure. It is not in the repository and
never was; a repository that carried it could not be cloned at a sensible size,
and pinning it by fingerprint is what makes the receipt meaningful.

**How the seam is decided:** by an allowlist over paths, computed rather than
maintained. The allowlist is the import closure of the documented entry points,
plus what a contributor needs to build and test, minus categorical exclusions
that override both. `scripts/open_surface.py` is that rule, and it ships,
including the exclusion list naming every withheld module and why.

Prove the seam rather than trusting this document:

```bash
python scripts/verify_open_core.py
```

It builds the published surface from the allowlist, proves every closed module
is **unimportable** before measuring anything, runs the suite inside the built
tree, and refuses to print a number if any of that fails.

---

## 7. Where numbers are allowed to live

| Surface | What it may show |
|---|---|
| Run records | the facts, as emitted |
| The benchmark harness | aggregations derived from records at run time |
| The paper | measured results with method, traceable to records |
| The website | derives nothing; it may quote committed, evidence-backed figures verbatim, each with its denominators |

A figure that cannot name its corpus, its denominator, its basis and the
execution path that produced it does not go in a table.
