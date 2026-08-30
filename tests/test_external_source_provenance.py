"""Fixture provenance must be able to express content we did not write.

Before 2026-08-09 it could not, and the refusal was correct rather than buggy:
``generated_by`` was required, an external source has no generator, so the
record could never be completed. The consequence was that **every figure
LaconIQ has ever published was measured on text LaconIQ wrote**, and the
standard had no way to admit anything else.

Two gaps, both closed here.

ORIGIN. A fixture now records either a local generator or an external
publisher, and exactly one. Making ``generated_by`` merely optional would have
admitted fixtures with no origin at all — a rule loosened for a real case in a
way that quietly covers the unreal ones, which is the defect shape this whole
extension exists to avoid.

PERMITTED USE. The seven original scopes covered using, modifying,
distributing, training on, commercial use, redistributing models, and
open-source distribution. **None covered publishing a figure CALCULATED FROM
the content** — a different act, and the one we exercise first and most
publicly, since that is what a report or a README does. A licence can permit
the measurement and forbid the redistribution, or the reverse; collapsing them
loses the distinction in the direction that publishes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tekmyra.fixtures import ExternalSourceProvenance, FixtureProvenance
from tekmyra.rights_vocabulary import (
    PERMITTED_SCOPES,
    PUBLISH_DERIVED_MEASUREMENTS,
    UnknownScopeError,
    validate_scopes,
)

SHA = "a" * 64


def _external(**overrides) -> dict:
    base = {
        "publisher": "U.S. Government Publishing Office",
        "source_url": "https://www.govinfo.gov/app/collection/CREC",
        "retrieved_at": datetime(2026, 8, 9, tzinfo=timezone.utc),
        "licence_basis": "17 U.S.C. §105",
        "licence_evidence_url": "https://www.govinfo.gov/about/policies",
        "licence_evidence_sha256": SHA,
        "permitted_scopes": [PUBLISH_DERIVED_MEASUREMENTS, "use_in_laconiq_product"],
    }
    base.update(overrides)
    return base


def test_an_external_source_can_now_be_recorded_at_all() -> None:
    """The wall this extension removes."""
    provenance = FixtureProvenance(
        source="congressional-record",
        license="public-domain-usgov",
        external=ExternalSourceProvenance(**_external()),
    )
    assert provenance.generated_by is None
    assert provenance.external is not None
    assert provenance.external.publisher.startswith("U.S. Government")


def test_a_generated_fixture_still_validates_unchanged() -> None:
    """The extension must not cost the 341 fixtures already in the corpus."""
    provenance = FixtureProvenance(
        source="synthetic",
        license="CC0-1.0",
        generated_by="benchmarks/fixtures/generate_fixtures.py",
    )
    assert provenance.external is None


def test_a_fixture_with_no_origin_at_all_is_refused() -> None:
    """Loosening a required field must not make it skippable for everyone."""
    with pytest.raises(ValidationError, match="records no origin"):
        FixtureProvenance(source="mystery", license="CC0-1.0")


def test_a_fixture_cannot_claim_both_origins() -> None:
    """One fixture, one origin — otherwise the record cannot be audited."""
    with pytest.raises(ValidationError, match="one origin"):
        FixtureProvenance(
            source="confused",
            license="CC0-1.0",
            generated_by="generate_fixtures.py",
            external=ExternalSourceProvenance(**_external()),
        )


def test_the_publisher_is_recorded_separately_from_where_we_fetched_it() -> None:
    """The restriction usually lives in the packaging, not the content.

    Court opinions are public domain while one compilation of them is
    NoDerivatives; council meetings are public proceedings while one bundle is
    NonCommercial-ShareAlike. A record naming only the fetch location lets a
    borrowed restriction ride in unnoticed.
    """
    fields = set(ExternalSourceProvenance.model_fields)
    assert "publisher" in fields
    assert "source_url" in fields
    assert "licence_evidence_url" in fields, (
        "the grant must point at where the PUBLISHER states it, which is not "
        "necessarily where the content was fetched"
    )


def test_licence_evidence_is_an_artifact_not_a_sentence() -> None:
    """A licence recorded as prose is a recollection, and one has been wrong."""
    with pytest.raises(ValidationError):
        ExternalSourceProvenance(**_external(licence_evidence_sha256="not-a-digest"))


def test_publishing_a_measurement_is_now_expressible() -> None:
    """The gap that mattered most, because it is the use exercised first."""
    assert PUBLISH_DERIVED_MEASUREMENTS in PERMITTED_SCOPES
    assert PUBLISH_DERIVED_MEASUREMENTS not in {
        "use_in_laconiq_product",
        "modify_fixture_content",
        "distribute_fixture_content",
        "train_models",
        "commercial_product_use",
        "redistribute_trained_models_and_derived_artifacts",
        "open_source_product_distribution",
    }, "the new scope must be distinct from the seven it was missing from"


def test_measuring_and_redistributing_are_separable() -> None:
    """A source may permit the number and forbid the corpus, or the reverse."""
    measure_only = ExternalSourceProvenance(
        **_external(permitted_scopes=[PUBLISH_DERIVED_MEASUREMENTS])
    )
    assert "distribute_fixture_content" not in measure_only.permitted_scopes


def test_an_unrecognised_scope_raises_rather_than_reading_as_ungranted() -> None:
    """A typo must not present as a permission that was never given.

    Silently treating an unknown string as 'not granted' reports a missing
    permission that was in fact written down — and the reader cannot tell that
    from a permission genuinely absent.
    """
    with pytest.raises(UnknownScopeError, match="not in the permitted-use vocabulary"):
        validate_scopes(["open_sorce_product_distribution"])


def test_an_unrecognised_scope_is_refused_at_the_model_boundary_too() -> None:
    """Not just in the helper — the record itself must not hold one."""
    with pytest.raises(ValidationError):
        ExternalSourceProvenance(**_external(permitted_scopes=["invented_scope"]))


def test_locally_generated_external_material_names_what_produced_it() -> None:
    """Logs and diffs are GENERATED by running the repo, never harvested.

    Published CI output can embed third-party tool output the repository
    licence does not reach. Running the suite ourselves makes the log our own
    output, and the field records which side of that line the artifact is on.
    """
    provenance = ExternalSourceProvenance(
        **_external(generated_locally_by="scripts/generate_repo_logs.py")
    )
    assert provenance.generated_locally_by == "scripts/generate_repo_logs.py"


def test_the_vocabulary_is_enumerated_so_completeness_is_answerable() -> None:
    """A container is not the vocabulary; the values are."""
    assert len(PERMITTED_SCOPES) == 8
    assert all(isinstance(s, str) and s for s in PERMITTED_SCOPES)
