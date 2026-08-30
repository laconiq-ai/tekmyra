"""Fixture schema, loader, and validator for synthetic evaluation data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from tekmyra.contracts import Block, EligibilityReason, RiskClass, RouteClass
from tekmyra.pii_vocabulary import DETECTION_TYPES


LOSSY_ROUTE_VALUES = {
    RouteClass.PROSE_TOKEN_COMPRESSOR.value,
    RouteClass.SUMMARIZER.value,
    RouteClass.EXTRACTIVE_KEYPOINT_COMPRESSOR.value,
}

SAFE_PII_ROUTE_VALUES = {
    RouteClass.PII_REDACTION_BEFORE_COMPRESSION.value,
    RouteClass.REJECT_OR_HUMAN_REVIEW.value,
}


def is_lossy(route: RouteClass | str) -> bool:
    """Return True if the route is a lossy compressor."""
    value = route.value if isinstance(route, RouteClass) else route
    return value in LOSSY_ROUTE_VALUES


def is_safe_for_pii(route: RouteClass | str) -> bool:
    """Return True if the route is an allowed PII-safe outcome."""
    value = route.value if isinstance(route, RouteClass) else route
    return value in SAFE_PII_ROUTE_VALUES


def classify_route_outcome(
    fixture: Fixture,
    predicted_route: RouteClass | str,
) -> dict[str, Any]:
    """Classify a predicted route against the fixture's route-intent contract."""
    predicted = predicted_route.value if isinstance(predicted_route, RouteClass) else predicted_route
    intent = fixture.gold_output.route_intent.value
    allowed = {r.value for r in fixture.gold_output.allowed_outcomes}
    return {
        "route_intent": intent,
        "allowed_outcomes": sorted(allowed),
        "route_intent_match": predicted == intent,
        "allowed_outcome_match": predicted in allowed,
        "lossy_final_route": is_lossy(predicted),
    }


class ProtectedSpanGold(BaseModel):
    span_type: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    text: str
    policy: Literal["preserve", "redact", "anchorize", "human_review"] = "preserve"
    detector: str = "gold"
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    span_id: str | None = None
    fact_id: str | None = None

    model_config = ConfigDict(extra="forbid")


class FixtureGoldOutput(BaseModel):
    route_intent: RouteClass
    allowed_outcomes: list[RouteClass] = Field(..., min_length=1)
    compressible: bool = True
    passthrough: bool = False
    redaction_only: bool = False
    human_review: bool = False
    expected_content: str | None = None
    expected_route: RouteClass
    expected_risk: RiskClass
    expected_fallback: str | None = None
    must_preserve_spans: list[ProtectedSpanGold] = Field(default_factory=list)
    expected_compression_eligibility: bool | None = None
    expected_allowed_bypass_reasons: list[EligibilityReason] = Field(default_factory=list)
    query_task: str | None = None
    answer_required_fact_ids: list[str] = Field(default_factory=list)
    allowed_safety_overrides: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _pii_default_non_compressible(cls, data: Any) -> Any:
        """PII fixtures default to non-compressible with redaction-only mode.

        THIS VALIDATOR HAS NEVER FIRED. Its only condition was the abandoned
        ``pii_`` prefix convention, which zero of 8,092 spans across all three
        corpora use, so ``has_pii`` has been False for every fixture ever
        loaded. Retiring the hatch does not change behaviour; it makes a dead
        branch visible instead of leaving it looking operational.

        REPOINTED AT ``DETECTION_TYPES``, and this is a scope extension beyond
        "retire the hatch" — flagged rather than buried. Leaving the condition
        as a constant False would knowingly ship a validator that cannot fire.

        The obvious objection is that gold ``must_preserve_spans`` DO carry data
        categories (email 57, ssn 35, phone 29 across the corpora), so switching
        the condition could start defaulting real fixtures to non-compressible
        and redaction-only — a corpus change wearing the clothes of a refactor.
        MEASURED ACROSS ALL 1,461 GOLD BLOCKS: 981 declare ``compressible``
        explicitly and this validator never touches them; 480 omit it and would
        take the default; and of those 480, ZERO carry a DETECTION_TYPES span.
        Nothing flips. The change is neutral on every fixture we own and only
        affects fixtures written after it.

        It does NOT resolve the taxonomy: ``span_type`` here is still being read
        as a data category, which works only because gold spans happen to carry
        categories where ``protected_spans`` carry handling labels. That
        inconsistency is the taxonomy's to settle.
        """
        if not isinstance(data, dict):
            return data
        spans = data.get("must_preserve_spans", []) or []
        has_pii = any(
            isinstance(s, dict) and s.get("span_type") in DETECTION_TYPES
            for s in spans
        )
        if has_pii:
            if "compressible" not in data:
                data["compressible"] = False
            if "redaction_only" not in data and "passthrough" not in data and "human_review" not in data:
                data["redaction_only"] = True
        return data

    @model_validator(mode="after")
    def allowed_outcomes_contains_intent(self) -> "FixtureGoldOutput":
        if self.route_intent not in self.allowed_outcomes:
            raise ValueError("allowed_outcomes must contain route_intent")
        return self

    @model_validator(mode="after")
    def mutually_exclusive_non_compressible_flags(self) -> "FixtureGoldOutput":
        flags = [self.passthrough, self.redaction_only, self.human_review]
        if sum(flags) > 1:
            raise ValueError(
                "at most one of passthrough, redaction_only, human_review may be True"
            )
        return self

    @model_validator(mode="after")
    def compressible_false_requires_non_compressible_flag(self) -> "FixtureGoldOutput":
        if not self.compressible and not any(
            [self.passthrough, self.redaction_only, self.human_review]
        ):
            raise ValueError(
                "when compressible is False, one of passthrough, redaction_only, "
                "human_review must be True"
            )
        return self


class ExternalSourceProvenance(BaseModel):
    """Where content we did NOT write came from, and on whose authority.

    ADDED 2026-08-09. The standard could not express an external dataset at
    all: ``generated_by`` was required, an external source has no generator, so
    the record could never be completed and the check correctly refused. That
    refusal was right and it was also a wall — every figure LaconIQ owns was
    measured on text LaconIQ wrote, and nothing else could be admitted.

    THE PUBLISHER, NOT THE AGGREGATOR. ``publisher`` is who actually issued the
    material, and it is separate from wherever we fetched it because the
    restriction usually lives in the packaging rather than the content: court
    opinions are public domain while one compilation of them is NoDerivatives;
    council meetings are public proceedings while one bundle of them is
    NonCommercial-ShareAlike. Recording only the fetch location would let a
    borrowed restriction ride in unnoticed, or a real one be lost.

    THE EVIDENCE IS AN ARTIFACT, NEVER A CLAIM. ``licence_evidence_sha256``
    digests the publisher's own captured statement. A licence recorded as a
    sentence in a document is a recollection, and recollections have already
    been wrong here — a dedication was recorded as living in per-file headers
    when the publisher documents signed affidavits held in a safe. Neither is a
    small error at publication time.
    """

    publisher: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    retrieved_at: datetime
    # "17 U.S.C. §105", an SPDX identifier, or a publisher's own dedication.
    licence_basis: str = Field(min_length=1)
    licence_evidence_url: str = Field(min_length=1)
    licence_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # What the licence actually grants, from rights_vocabulary. Never inferred:
    # an external source grants what it says and nothing by convenience.
    permitted_scopes: list[str] = Field(min_length=1)
    # Set when the material was produced by running something ourselves — a
    # test suite's log, a diff from checked-out history. The repository licence
    # covers what it authored, not what its CI happened to print, so generating
    # the artifact is how we avoid inheriting terms we never read.
    generated_locally_by: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _scopes_are_in_the_vocabulary(self) -> "ExternalSourceProvenance":
        from tekmyra.rights_vocabulary import validate_scopes

        validate_scopes(self.permitted_scopes)
        return self


class FixtureProvenance(BaseModel):
    source: str
    license: str
    # OPTIONAL SINCE 2026-08-09, and exactly one of this or ``external`` must be
    # present. Content we generated has a generator; content someone else
    # published has a publisher. A record carrying neither cannot be audited,
    # and one carrying both is claiming two incompatible origins.
    generated_by: str | None = None
    external: ExternalSourceProvenance | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    adjudicated_by: str | None = None
    # Optional content-addressed generator lineage.  Legacy fixtures remain
    # valid; governed corpora can require these fields in their own manifest.
    generator_id: str | None = None
    generator_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    template_id: str | None = None
    scenario_id: str | None = None
    augmentation_id: str | None = None
    source_lineage_id: str | None = None
    group_id: str | None = None
    joint_stratum: str | None = None
    family_index: int | None = Field(None, ge=0)
    variant_index: int | None = Field(None, ge=0)
    content_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    normalized_content_sha256: str | None = Field(
        None, pattern=r"^[0-9a-f]{64}$"
    )
    rng_seed_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    source_date_epoch: int | None = Field(None, ge=0)
    counterexample_pair_id: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def _omit_external_when_absent(self, handler):  # type: ignore[no-untyped-def]
        """Never emit ``external: null`` — an absence is not a record.

        THIS IS LOAD-BEARING, AND NOT FOR TIDINESS. The v1 long-context corpus
        is byte-frozen, and ``governed_provider_eval`` binds it with
        ``APPROVED_FIXTURE_SUITE_SHA256`` — a hash over EVERY file in the
        corpus directory, which includes the corpus's own generator source.

        So the obvious fix, adding "external" to the generator's exclusion set,
        edits a file inside the hashed directory and invalidates an approved
        checkpoint that gates PAID provider evaluation. It was measured: doing
        that broke 18 governed-eval tests, none of which are about provenance.
        Updating the approved hash to match would be silently re-approving a
        governance artifact to accommodate a schema change.

        Omitting the field here keeps every existing fixture byte-identical, so
        the frozen corpus and its approval both stand untouched.
        """
        data = handler(self)
        if isinstance(data, dict) and data.get("external") is None:
            data.pop("external", None)
        return data

    @model_validator(mode="after")
    def _origin_is_recorded_exactly_once(self) -> "FixtureProvenance":
        """Every fixture states where it came from, and states it one way.

        Making ``generated_by`` optional to admit external sources would
        otherwise admit fixtures with NO origin at all — the field would become
        skippable for everyone, not just for external content. That is the
        shape of the defect this whole extension exists to avoid: a rule
        loosened for a real case, in a way that quietly covers the unreal ones.
        """
        if self.generated_by is None and self.external is None:
            raise ValueError(
                "provenance records no origin: set generated_by for content we "
                "produced, or external for content someone else published"
            )
        if self.generated_by is not None and self.external is not None:
            raise ValueError(
                "provenance claims both a local generator and an external "
                "publisher; a fixture has one origin. If a script transformed "
                "external material, record the publisher in external and name "
                "the script in external.generated_locally_by"
            )
        return self


class Fixture(BaseModel):
    schema_version: Literal["2.0.0", "2.1.0"] = "2.1.0"
    fixture_id: str
    tenant_id: str = Field(default="tenant-benchmark", min_length=1, max_length=256)
    content_type: str
    risk_tier: RiskClass
    content: str
    route_labels: list[RouteClass]
    protected_spans: list[ProtectedSpanGold] = Field(default_factory=list)
    gold_output: FixtureGoldOutput
    provenance: FixtureProvenance
    tags: list[str] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)
    adjudicated: bool = False
    adjudicated_at: str | None = None
    protected_fact_index: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v


def is_pii_fixture(fixture: Fixture) -> bool:
    """Identify PII fixtures by id, tags, span types, or redaction-only mode."""
    if fixture.fixture_id.startswith("pii-"):
        return True
    if any(tag.lower().startswith("pii") for tag in fixture.tags):
        return True
    spans: list[ProtectedSpanGold] = []
    spans.extend(fixture.gold_output.must_preserve_spans or [])
    spans.extend(fixture.protected_spans or [])
    for span in spans:
        if span.span_type in DETECTION_TYPES:
            return True
    if fixture.gold_output.redaction_only:
        return True
    return False


def load_fixtures(path: Path | str) -> list[Fixture]:
    fixtures: list[Fixture] = []
    path = Path(path)
    if path.is_dir():
        # Recurse into category subdirectories; skip suite metadata files.
        files = sorted(
            p for p in path.rglob("*.json") if p.name != "held_out_split.json"
        )
    else:
        files = [path]
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            fixtures.extend(Fixture.model_validate(item) for item in data)
        else:
            fixtures.append(Fixture.model_validate(data))
    return fixtures
