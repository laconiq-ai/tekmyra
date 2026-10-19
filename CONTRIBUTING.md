# Contributing to TekMyra Core

Thanks for considering it. This document says what you are agreeing to, what the
project will and will not merge, and how to make a change that can be verified.

## What you grant

**We use the Developer Certificate of Origin (DCO), not a CLA.** Sign off each
commit:

```
git commit -s -m "your message"
```

That adds a `Signed-off-by:` line, which is your statement that you have the
right to submit the work under this project's licence. The full text is at
<https://developercertificate.org/>.

Your contribution is licensed to everyone under **Apache-2.0**, the same licence
as the rest of the repository, including its patent grant in §3. We do not ask
you to assign copyright and we do not ask for rights beyond the licence.

**What that means in both directions, stated plainly:** because there is no CLA,
this project cannot relicense your contribution. If the project is ever offered
under different terms, contributed code stays under Apache-2.0 unless its author
agrees otherwise. That constraint is deliberate and it binds the maintainers too.

## What this project is

The open core is the compression and verification path: routing, codecs, the
learned selector, protected-span detection, the safety merge, and the verifier.
It is deliberately not the whole product; see the README for what is absent and
why. Pull requests that reintroduce an excluded component will be declined on
that ground rather than on quality.

## The bar for a change

This codebase is unusually strict about evidence, and the strictness is the
point rather than ceremony.

- **A number needs its corpus, its denominator and its basis.** "12% better" is
  not reviewable; "62.1458% effective byte reduction over 138 fixtures with 6
  refusals, refusals contributing zero saving" is. Percentages of different
  bases do not go in one table.
- **A test that cannot fail is worse than no test.** If a check would pass when
  the thing it guards is absent, say so and fix the check. Several of the
  comments in this tree exist because exactly that happened here.
- **Behaviour claims come from a run, not from reading the source.** An empty
  counter and a measured zero look identical in the output; only the instrument
  tells them apart.
- **Don't widen the published surface by accident.** `scripts/open_surface.py`
  decides what is in this repository, as an allowlist over paths with a stated
  reason per rule. Adding a module that imports an excluded one is a seam defect
  to cut, not a reason to publish the excluded module.

## Which branch

**Open pull requests against `dev`.** Changes land on `dev`, are promoted to
`qa`, and reach `main` only after `qa` passes CI, so `main` is always the
released state. GitHub offers `main` by default because it is the default
branch; if your pull request targets `main`, a maintainer will retarget it to
`dev`, and you do not need to do anything.

## Before you open a pull request

Run the suite and the seam check. Both ship, and both pass without a trained
model artifact:

```bash
python -m pytest
python scripts/verify_open_core.py
```

Then check all six shipped corpora and confirm your change did not move a number
you did not intend to move:

```bash
python -m tekmyra.benchmark compressor --fixture-dir benchmarks/fixtures/synthetic --allow-rules-only
python -m tekmyra.benchmark compressor --fixture-dir benchmarks/fixtures/long_context_v1 --allow-rules-only
python -m pytest tests/test_public_corpus_notice.py
```

`--allow-rules-only` is what you need when there is no trained artifact, and
this repository does not publish one. It records an explicitly rules-only run,
which **cannot emit a headline compression ratio**, so it will show you that a
routing decision moved, and it will not show you that a compression figure did.
Drop the flag if you have an artifact of your own.

The public-corpus rate has a stronger artifact requirement: run
`python -m pytest tests/test_public_corpus_ratchet.py` only with the exact
reference runtime-artifact directories recorded in its baseline. A clean clone
correctly fails that ratchet because its prose compressor is a no-op.

Please include, in the pull request body:

1. what changed and why, in prose a reviewer can check against the diff;
2. the command you ran and its output, including the exit code;
3. what you did **not** cover, stated rather than left to be discovered.

Point 3 is not a formality. A change described as complete and found to be
partial costs more review than one that named its own gap.

## Reporting a bug

Open an issue with the input, the command, the observed output and the expected
output. If the bug is a wrong *number* rather than a crash, include the corpus
and the basis; those are usually where the disagreement actually is.

**Security issues do not go in the issue tracker.** See SECURITY.md.

## Code of conduct

Be straightforward and assume competence. Disagreement about a measurement is
resolved by measuring, not by seniority.

The full text, and how to report a problem, is in CODE_OF_CONDUCT.md.
