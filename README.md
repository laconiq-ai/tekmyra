# TekMyra, by LaconIQ

TekMyra is a compressor that refuses rather than guesses.

TekMyra compresses context for language models and then *checks its own work*.
Before anything is emitted, a verifier confirms that every protected span (an
account number, a citation, a file path, a monetary amount, a policy
identifier) is represented **exactly once** in the output: as verbatim text,
as an approved typed redaction marker, or as a token that resolves to the
original. If that check fails, the compressor retries on a safer route. If the
retry fails too, it raises and emits nothing.

"Exactly once" is counted per occurrence, and the verifier
(`tekmyra/verifier.py`, `_check_protected_disposition`) applies it like this:
every locked span must be accounted for, and a protected value may not appear
more often than it did in the input. A value that occurs three times in the
input can therefore occur three times in the output. Diff routes skip the
duplicate half of the check, because a diff legitimately prints the same path
on both sides (`a/src/config.yaml`, `b/src/config.yaml`); a missing span
still fails there.

That is the whole idea. A compressor that silently drops a dollar figure is
worse than no compressor, because the output still looks fine.

The rest follows from it. A deterministic safety gate assigns a risk tier; a
learned router proposes a route and **may raise the risk tier but never lower
it**; a lossy route proposed at elevated risk is overridden to a reversible one.
Protected spans get a disposition (preserve, redact, anchorize, or
human-review) that tightens monotonically with risk and never loosens.

## What you get, honestly

This is an **open core**, not the whole product. It is the compression and
verification path: routing, the codecs, the learned selector, protected-span
detection, the safety merge, and the verifier. **55 modules**, six benchmark
corpora, the configuration and contracts they need, the test suite, and the
scripts that let you check our claims instead of believing them.

What is deliberately *not* here, and why, is at the bottom of this file. Each
absence is a design decision and is documented as one.

## Install

From PyPI (Python 3.11+; the distribution is `tekmyra-core`, the import name is
`tekmyra`):

```bash
pip install tekmyra-core
```

Or from source, which is what the rest of this page assumes:

```bash
git clone https://github.com/laconiq-ai/tekmyra.git
cd tekmyra
pip install -e '.[dev]'   # the library plus pytest, to run the suite below
```

If you do not want the test dependencies, `pip install -e .` installs the
library alone.

Core dependencies are `click`, `numpy`, `pydantic`, `scikit-learn`, `scipy`,
`tiktoken` and `pyyaml`. `tiktoken` is a **core** dependency, not an extra: the
token-reduction figures are unobtainable without it, and a bare install that
printed `WITHHELD` instead of a number is the defect that forced this decision.

Optional extras: `[transformer]` for the transformer compressor path,
`[baselines]` for comparison baselines. Neither is needed to reproduce anything
below.

## You need a model, and here is how to get one

**Trained model artifacts are not in this repository and never will be.** They
are the commercial half.

The **reference artifact bundle** is published as a release asset on this repository's Releases
page, `tekmyra-reference-artifacts.tar.gz`, with its SHA-256 printed in the release
notes. Download it, **check the digest before extracting**, and unpack it at the
root of your clone:

```bash
curl -L -O https://github.com/laconiq-ai/tekmyra/releases/latest/download/tekmyra-reference-artifacts.tar.gz
shasum -a 256 tekmyra-reference-artifacts.tar.gz   # compare against the release notes
tar xzf tekmyra-reference-artifacts.tar.gz         # unpacks .cache/compressor/1.0.0 and .cache/router/1.0.0
```

The digest is published beside the asset rather than pinned in this file, so
that a reader checks the artifact they actually downloaded against the record
for that release rather than against a constant that a later release would
silently falsify.

**What you cannot do, stated rather than left to be discovered:** you cannot
retrain these artifacts from this repository. The training corpus and the
trainer are deliberately not published, under the same rule that keeps trained
artifacts closed. The bundle is verifiable as a receipt (compare its digest)
and is not reproducible from source. No tool in this payload claims otherwise.

The artifact directories are resolved in this order, and a failure prints every
location it tried:

1. `$TEKMYRA_ARTIFACT_ROOT/.cache/...` if that variable is set
2. `.cache/...` under your working directory
3. `.cache/...` beside the installed package

**Without a model, the benchmark refuses to give you a number.** That is not a
bug, and it is worth seeing; it is the same instinct as the verifier:

```
Error: Learned router artifact did not load — refusing to produce a benchmark
number. artifact_dir='<your clone>/.cache/router/1.0.0' exists=False
backend='none'.
could not resolve '.cache/router/1.0.0'; searched, in order:
  absent  <your clone>/.cache/router/1.0.0
Set $TEKMYRA_ARTIFACT_ROOT to the directory holding .cache/ if your artifacts
live elsewhere. Re-run with --allow-rules-only to record an explicitly
rules-only run; such a run cannot emit a headline compression ratio.

If you have not downloaded the model yet, that is almost certainly the cause,
and it is the expected state of a fresh clone: the trained artifact is a
separate release asset, not part of this repository. See 'You need a model' in
the README for the fetch procedure. To measure what this build can produce
without one, re-run with --allow-rules-only - that records an explicitly
rules-only run and cannot emit a headline compression ratio.
```

(That transcript is pasted from a real run in a fresh clone, where the working
directory and the editable install are the same place, so the search has one
candidate; run from anywhere else and the list grows to every location tried.)

Take the escape hatch it offers and you get a run that is honest about what it
is: **the headline is withheld** (not zero, not one, withheld), token reduction
is 0.00%, and **68 of 68 locked spans are still preserved** across 28 reached
fixtures. The safety machinery works without a model. The compression does not.

## Numbers

Every figure names its corpus, its denominator, its basis, and **which execution
path produced it**. Figures on different bases are not comparable and are not
put in one table.

**With the reference artifact bundle, on the corpora in this repository**, via the
joblib/scikit-learn path (no `onnxruntime` installed):

| corpus | fixtures | mean size | ratio on reached fixtures | tokens | locked spans |
|---|---|---|---|---|---|
| `synthetic` | 28 eligible of 301 shipped (28 reached, 0 refused) | 250 chars (over 301 shipped) | **0.7409** | **+25.86%** | **68/68** |
| `long_context_v1` | 40 eligible of 40 shipped (26 reached, **14 refused**) | 23,644 chars | **0.2929** | **+48.44%** corpus / +70.40% on the prose route | **704/704** |

*Ratio on reached fixtures* is the share of content kept, averaged over the
fixtures the compressor reached; it is a reached-set diagnostic, not a
headline. *Tokens* is token reduction over the eligible set **with refusals in
the denominator at zero saving**: a refusal saves nothing and is counted as
saving nothing, because a build that refuses more would otherwise score better
for refusing. A fixture is eligible when its content type has a codec on its
route in this build; eligibility is fixed before any run (`synthetic` ships
301 fixtures, 28 prose-eligible; `long_context_v1` ships 40, all eligible).

One more thing both rows share, which the benchmark prints beside every number
and this table must therefore say too: **routing is blind, and some labels name
a codec that does not exist.** 20 of synthetic's 28 eligible fixtures are
labelled `route_intent=summarizer` and 20 of `long_context_v1`'s 40 are
labelled `extractive_keypoint_compressor` (both declared but never built), so
routing blind sends them to `prose_token_compressor` instead. An improvement on
those fixtures means "compressed with a different codec than the labels
specify", not "compressed what the labels intended"; both are true of these
numbers and only one is what a reader assumes. The report's `routing_caveat`
field carries this per-corpus, derived from the fixtures at run time.

**The 14 refusals on `long_context_v1` are the headline of that row, not a
footnote to it**, and they are refusals, not abstentions. Both words appear in
this project and they are different outcomes, so here is what was measured, on
both builds at the same commit with the anchor store as the only variable:

- **Abstentions are zero in both builds.** An abstention is a fixture that
  completes untouched. None did.
- **The 14 are refusals, and they exist only in this build.** The commercial
  build completes all 40. The refused fixtures are the same 14 either way, and
  the 26 this build completes are exactly a subset of the commercial build's 40.
- **They fail three checks at once** (`locked_spans_preserved`,
  `facts_preserved`, `anchors_resolvable`), not the anchor check alone. This
  build cannot resolve anchors (see *What is deliberately absent*), so spans the
  commercial build recovers are unrecoverable here, and verification refuses
  rather than shipping the loss.
- **They are not "routed to" `reversible_retrieval_mode`.** That is where they
  LAND after their first route fails verification; the router sent them to
  `legal_security_safe_mode` (6) and `governance_safe_compression` (8). Routing
  is identical in both builds (same routes, same ratios on every shared
  fixture), so the anchor store changes what survives verification, never where
  anything was sent.

**This build is not worse at compressing.** It reaches the compressor on exactly
the same 26 fixtures and produces the identical **0.2929**. It refuses 14 that
the commercial build completes, for a stated reason, and counts them as saving
nothing. A system that compresses 26 of 40 and refuses 14 rather than ship an
unverifiable result is making a different promise from one that compresses all
40; that difference is what the open/closed seam costs, stated rather than
smoothed.

**Locked spans: name the build before quoting the ratio.** Both builds report
**1.000**, over different denominators: **704/704** here, **1785/1785**
commercial. The gap is the 1,081 spans belonging to the 14 refused fixtures,
which leave this build's count with the fixtures. Span *detection* is identical
across both builds on every shared fixture.

### The public-corpus figure and what this repository can reproduce

Our public-corpus baseline is **62.1458% effective byte reduction** over **138
fixtures** (2,708,761 → 1,025,379 bytes) with **6 refusals** (4.35% of fixtures,
3.55% of bytes), refusals contributing zero saving; the on-accepted diagnostic is
64.4356%. Measured at commit `14c63bc` on
`benchmarks/fixtures/public_{domain,code,diff,json}_v1`.

Those four corpora ship in this repository. A clean clone can verify that their
fixture bytes match the committed baseline and that every third-party source is
covered by `NOTICE`:

```bash
python -m pytest tests/test_public_corpus_notice.py
```

That does **not** reproduce the 62.1458% rate. Reproducing it also requires the
trained artifacts, which the reference bundle supplies as the versioned
`.cache/compressor/1.0.0` and `.cache/router/1.0.0` directories the code loads;
a clean checkout has neither and measures a no-op prose compressor. With the
bundle unpacked, run:

```bash
python -m pytest tests/test_public_corpus_ratchet.py
```

The rate is on a **byte basis** and is not comparable to the ratio and token
figures above. The evidence artifact,
`tests/baselines/public_corpus_baseline.json`, carries the figure, its
denominators, the corpus file counts and the sha256 digest of each corpus.

Which of those digests you can check here is worth stating precisely, because
the baseline is the record of a measurement taken in the private tree. The
digests for the four `public_*` corpora and `synthetic` match this
repository's bytes and you can recompute them (the algorithm is
`tests/measurement_provenance.py::_corpus_digest`). Three entries describe the
private measuring tree and will NOT match what you have:
`benchmarks/fixtures/long_context_v1` was digested with its two generator
scripts, which are deliberately unpublished (the 42 fixture files you have
are byte-identical to the 42 that were measured:
`test_long_context_corpus_bytes_are_pinned` in the ratchet suite binds their
content to a committed digest on every run, and it exists because a reviewer
appended a canary byte to a fixture and caught an earlier draft of this
sentence claiming a proof nothing performed); and `.cache/router` was
digested over the full private
artifact directory, of which the reference bundle carries the versioned
`1.0.0` subset the code actually loads. The third,
`.cache/compressor/1.0.0`, has the file count you have
(8) but a different rolled digest. Because that record is a single digest over
the whole directory rather than a hash per file, the difference cannot be
attributed to a particular file, by us or by you; what can be checked is that
these artifacts still produce the recorded measurement exactly, which
`tests/test_public_corpus_ratchet.py` re-runs and pins cell by cell. Editing
any record to match the payload would fabricate provenance, so all three ship
as measured, with this paragraph instead.

## Reproducing what is here

**Install against the constraints file if you want the numbers, not just the
library.** Dependencies here are declared as ranges so TekMyra can live inside
your application; ranges float, so the figures above were measured against the
exact versions recorded in `constraints-reproduce.txt`:

```bash
pip install -c constraints-reproduce.txt -e '.[dev]'
```

That command needs **Python 3.12 or newer**. The library itself supports 3.11,
but two of the measured versions do not: `numpy==2.5.1` and `scipy==1.18.0`
both declare `requires_python >= 3.12`, so on 3.11 pip stops with
`ResolutionImpossible`. On 3.11, `pip install -e '.[dev]'` installs the
declared ranges instead, which is fine for using TekMyra but is not the set
the figures were measured against.

How much that matters is measured rather than asserted, against the published
bundle: on scikit-learn 1.8.0 and 1.9.0 the `long_context_v1` headline is
**0.2929 both times**, identical to four decimals. The constraints file is a
durability guarantee, not the only thing holding these figures up. Below the
declared floor is different, and worth stating precisely: the bundle's router
was exported under scikit-learn ≥ 1.8, and loading it into 1.5.2–1.7.2
crashes at model load (`AttributeError` on the removed `multi_class`
attribute); it does not degrade, it refuses to start. That is why
`pyproject.toml` declares `scikit-learn>=1.8`.

```bash
python -m tekmyra.benchmark compressor --fixture-dir benchmarks/fixtures/synthetic
python -m tekmyra.benchmark compressor --fixture-dir benchmarks/fixtures/long_context_v1
python -m pytest tests/test_public_corpus_notice.py   # public bytes + attribution
python -m pytest tests                    # the suite, green without a model
python scripts/verify_open_core.py        # the seam, proved rather than asserted
```

One exit-code note before you wire that last command into anything with
`set -e`: on a fresh clone with no model, `verify_open_core.py` exits **6**
("seam proven, numbers not measured"), deliberately non-zero so an unmeasured
run can never be quoted as success. That is the healthy fresh-clone state; in
CI, allow it explicitly: `python scripts/verify_open_core.py || [ $? -eq 6 ]`.
With the reference bundle installed it measures and exits 0. The full exit-code
contract is documented at the top of the script.

`scripts/payload_manifest.py` also ships, and it is worth saying plainly what it
is: **an internal check, which you cannot run against this clone.** It proves
every published file was derived from our private source at a stated commit:
each one compared byte-for-byte against the git blob it came from, or re-derived
by re-running the staging script. That needs the private repository's history,
which you do not have and will not get. It is here because the staging rule and
its checker ship together or neither is auditable, not because it is a check we
are offering you. Nothing in this README rests on it.

`verify_open_core.py` is the one worth running. It builds the published surface
from the allowlist, proves every closed module is **unimportable** before
measuring anything, runs this suite inside the built tree, and refuses to print a
number if any of that fails. Its output includes the by-route table, so the
refusals and abstentions above are something you can watch rather than take on
trust: the 14 refusals appear as the `refused` column of the
`reversible_retrieval_mode (ABSTENTION)` row, and that row's `done` column, the
abstentions, is 0.

The suite passes without a model: the tests that need a trained artifact skip,
each skip says what it was waiting for, and the ones that exercise artifact
resolution list the directories they searched.

### Troubleshooting

**"It compresses nothing" / every ratio is 1.000.** Almost always a missing
model. The benchmark refuses rather than reporting 1.000, but a direct API call
degrades to passthrough, which is safe for your protected spans and useless as a
measurement. Check the resolution order above, or set `TEKMYRA_ARTIFACT_ROOT`.

**Running from another directory.** Artifact paths resolve against your working
directory *first* and then beside the installed package, so a model placed next
to the package is found from anywhere. If in doubt, run the benchmark once: any
failure prints every path it tried.

## What is deliberately absent

These are not omissions. Each is a decision, and the code is built so that its
absence is *safe* rather than merely quiet.

**The anchor store resolves nothing.** When a protected span is anchorized, this
build mints an anchor identifier and then cannot resolve it:
`NonResolvingAnchorStore.get()` returns `None` for every identifier, including
ones it minted itself. The encrypted, tenant-partitioned, retention-bounded,
retrieval-audited store is the commercial half.

The consequence is that **this build refuses more than the commercial one**, and
that is measured rather than claimed: on the live PII route the commercial build
completes 30 of 30 rows and this one refuses 4 (a measurement on production
traffic, which no shipped fixture contains), so it is reported here as context
rather than as a number you can reproduce from this repository. The open half cannot ask "was
that dropped span recoverable?" (that question needs the store), so it treats
the drop as unrecoverable and refuses. It is not an empty store. It is a store
that *cannot* resolve, which is the stronger position.

**There is no economics gate.** The ex-ante cost model that decides whether
compressing is worth it (spend caps, credit balance, budget policy) is not
here. Its data contract is (`EligibilityOutcome`, `CompressionEconomics`,
`EligibilityReason`), because the fixture schema references it, but nothing
computes it.

**There are no paid-provider adapters**, no tenancy or audit layer, no recovery
server, no internal operations tooling, and **no trained artifacts or training
corpora**. One nuance so you don't have to discover it: two schema files under
`contracts/` (`production-admission-v1.json`, `provider-evaluation-policy-v1.json`)
name specific providers, because they are the validation contracts for the
commercial evaluation process: the contracts ship (the fixture schema
references them); the adapters and the process they govern do not.

**Nothing here authenticates anyone**, because there is nothing here to
authenticate to. This is a library and a benchmark harness.

## About the fixtures

Every fixture in `synthetic` and `long_context_v1` is **machine-generated** by
the scripts named in its own provenance block.
They contain no third-party text and no real personal data.
Identifiers that look like personal data (social-security numbers, card numbers,
addresses at `example.com`) are synthetic values written to exercise the
protected-span detector.

Some fixtures carry legacy provenance labels (`source: public_sample`,
`license: MIT`, `license: proprietary_laconiq_mock`, and
`adjudicated_by: nextgen_router.adjudicator`) that predate this
repository and describe the *shape* the generator was imitating rather than an
upstream source. **There is no upstream and no third-party attribution is owed.**
The labels are kept as they are because rewriting them would change the corpus
digests our published baselines pin.

The four `public_*_v1` corpora are different by design: they contain text from
the Congressional Record and six permissively licensed open-source projects.
Every fixture names its exact source and licence in its provenance block; the
full project attribution and licence text is in `NOTICE`.

`nextgen_router` in that last label is this project's own former name, before it
was renamed to TekMyra. It names no component you will find in this repository,
and it is recorded here because you will find it in every fixture and the
explanation should not be one paragraph away from the thing it explains.

## How the published surface is decided

By an allowlist over paths, computed rather than maintained: the import closure
of the documented entry points (deferred, in-function imports included), plus
what a contributor needs to build and test, minus categorical exclusions that
override both. `scripts/open_surface.py` is that rule, and it ships, including
the exclusion list, which names every withheld module and why.

A denylist publishes by default. Measured, the denylist version of this
repository would have shipped 88 modules including a live OpenAI adapter and a
module of spend caps, none of which anyone had decided to publish.

## Links

The website is [tekmyra.ai](https://tekmyra.ai). The technical paper is at
[tekmyra.ai/tekmyra-paper.html](https://tekmyra.ai/tekmyra-paper.html)
([PDF](https://tekmyra.ai/tekmyra-paper.pdf)). The paper is archived with a
permanent DOI at 10.5281/zenodo.22243913
(https://doi.org/10.5281/zenodo.22243913). The package is
[tekmyra-core on PyPI](https://pypi.org/project/tekmyra-core/).

## On AI assistance

Substantial portions of this codebase were written with AI assistance, directed
and gated by a human maintainer. The private development history carries
`Co-Authored-By` trailers naming the assistants involved. Nothing published here
rests on that: every figure in this README names its corpus, its basis, and
whether you can re-run it: the standard we would want applied to any codebase
regardless of what typed it.

You will meet names in the comments: citations like `Vega ruling 34`, or notes
attributed to `Atlas`, `Magnus` or `Sentinel`. Those are the project's internal
AI review seats (architect, lead, platform, QA), and a "ruling" is a recorded
review decision from that process: each one was adversarially reviewed and then
gated by the human maintainer before merging, the same as any other change. The
citations are kept in the code because they are the provenance of the decisions
they sit next to; stripping them would make the comments look self-evident when
they were in fact argued.

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`.

Contributions are under the DCO; see `CONTRIBUTING.md`. Security reports do not
go in the issue tracker; see `SECURITY.md`.
