# Changelog

Notable changes to TekMyra Core, newest first. Versions follow the package
version in `pyproject.toml`. Each entry says what changed and, where a number
or a claim is involved, how it was checked.

## Unreleased

### Added
- **CI** (`.github/workflows/ci.yml`). On every pull request and on pushes to
  `dev`, `qa` and `main`: the test suite on Python 3.11 and 3.12, and
  `scripts/verify_open_core.py` with exit 6 (seam proven, numbers not measured)
  allowed. A second job downloads the v0.1.0 reference bundle, checks it
  against the sha256 in the v0.1.0 release notes, runs the suite with the
  artifact-gated tests included, runs both README benchmark commands, and
  requires `verify_open_core.py` to measure, exit 0, and print every figure in
  the README table. A weekly scheduled run
  catches dependency drift inside the declared ranges.
- A CI badge in the README.
- This changelog, issue and pull-request templates, and `CODE_OF_CONDUCT.md`.

### Changed
- **Branches.** Changes now land on `dev`, are promoted to `qa`, and reach
  `main` only after `qa` passes CI. `main` stays the default branch, so what you
  see first is the released state. See CONTRIBUTING.md.

### Fixed (documentation)
- The README now says what "exactly once" means in the verifier: counted per
  occurrence, a protected value may not appear more often than it did in the
  input, and diff routes skip the duplicate half of the check.
- The README's description of the seam-check table now distinguishes refusals
  (14 on `long_context_v1`) from abstentions (0).
- The README now says that the reproduce install
  (`pip install -c constraints-reproduce.txt -e '.[dev]'`) needs Python 3.12
  or newer: `numpy==2.5.1` and `scipy==1.18.0` require it. The library itself
  supports 3.11. Found by the first CI run.

### Unchanged, and checked
- Every figure in the README table reproduces exactly from the v0.1.0 bundle:
  `synthetic` 0.7409, +25.86%, 68/68; `long_context_v1` 0.2929, +48.44%
  (+70.40% on the prose route), 704/704, 14 refused. Measured locally at
  `b9b7806` on Python 3.12.14, and checked on every CI run by the
  `reference-bundle` job.

## 0.1.0 — 2026-08-31

First open-source release: the compression and verification core, the
benchmark harness, six corpora, the test suite, and the scripts that check the
published claims. The reference artifact bundle is a release asset with its
sha256 in the release notes. Published on PyPI as `tekmyra-core`.

Documentation commits after the tag, before this changelog existed:
- 2026-08-31: README precision pass; reached-set ratio labelling and
  eligible-denominator scoping; `numpy` pinned in `constraints-reproduce.txt`
  to the measured 2.5.1.
- 2026-09-02: the paper cited by its Zenodo DOI (10.5281/zenodo.22243913) in
  `CITATION.cff` and the README; release date corrected to 2026-08-31.
