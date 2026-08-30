"""The permitted-use vocabulary for fixture content, enumerated in one place.

WHY THIS IS A MODULE AND NOT A LIST IN A MANIFEST. The scopes were free strings
validated only by a subset check: a manifest could grant
``open_sorce_distribution`` and the check would report it as an unmet
requirement rather than a typo, and — worse — nothing could ask "is this
vocabulary complete?", because a container is not the vocabulary, the values
are. Enumerating them makes both questions answerable.

THE GAP THIS MODULE EXISTS TO CLOSE. The seven original scopes covered using
the content, modifying it, distributing it, training on it, commercial use,
redistributing models, and open-source distribution. **None covered publishing
a figure CALCULATED FROM the content**, which is a different act: we would be
distributing a measurement, not the corpus.

That omission mattered because it is the use we exercise first and most
publicly. Every compression figure LaconIQ has ever quoted is a measurement
derived from fixture content, and for content we wrote ourselves the question
never arose — we hold everything. The moment a benchmark includes material
published by someone else, "may we state a number computed from your text"
becomes a real question with a real answer, and our own standard had no field
in which to record it.

A licence can permit the measurement and forbid the redistribution, or the
reverse. Collapsing them loses that, and it loses it in the direction that
publishes.
"""

from __future__ import annotations

# Using the content inside the product, and shaping it.
USE_IN_PRODUCT = "use_in_laconiq_product"
MODIFY_CONTENT = "modify_fixture_content"
DISTRIBUTE_CONTENT = "distribute_fixture_content"

# Training, and what may be done with what training produces.
TRAIN_MODELS = "train_models"
REDISTRIBUTE_TRAINED_MODELS = "redistribute_trained_models_and_derived_artifacts"

# Commercial and open-source distribution of the product itself.
COMMERCIAL_USE = "commercial_product_use"
OPEN_SOURCE_DISTRIBUTION = "open_source_product_distribution"

# ADDED 2026-08-09. Publishing a number computed FROM the content — a benchmark
# ratio, a retention rate, an aggregate — without distributing the content
# itself. Distinct from DISTRIBUTE_CONTENT and from MODIFY_CONTENT, and the one
# we exercise first: it is what a report, a deck or a README does.
PUBLISH_DERIVED_MEASUREMENTS = "publish_derived_measurements"

PERMITTED_SCOPES: frozenset[str] = frozenset(
    {
        USE_IN_PRODUCT,
        MODIFY_CONTENT,
        DISTRIBUTE_CONTENT,
        TRAIN_MODELS,
        REDISTRIBUTE_TRAINED_MODELS,
        COMMERCIAL_USE,
        OPEN_SOURCE_DISTRIBUTION,
        PUBLISH_DERIVED_MEASUREMENTS,
    }
)

# Scopes an EXTERNAL source is never assumed to grant. Content we wrote
# ourselves grants everything because we hold everything; content someone else
# published grants exactly what its licence says and nothing by convenience.
#
# Kept as a named set rather than a comment because "we would never assume
# that" is the kind of thing that is true until someone is in a hurry.
NEVER_ASSUMED_FOR_EXTERNAL: frozenset[str] = frozenset(
    {
        MODIFY_CONTENT,
        DISTRIBUTE_CONTENT,
        REDISTRIBUTE_TRAINED_MODELS,
        OPEN_SOURCE_DISTRIBUTION,
    }
)


class UnknownScopeError(ValueError):
    """A scope string that is not in the vocabulary.

    Raised rather than ignored. An unrecognised scope in a rights manifest is
    either a typo — in which case treating it as "not granted" reports a
    missing permission that was actually written down — or a new permission
    nobody added here, in which case it is being relied on without definition.
    Both need a human; neither is served by silence.
    """


def validate_scopes(scopes: object) -> frozenset[str]:
    """Return the scopes as a set, refusing anything outside the vocabulary."""
    if not isinstance(scopes, (list, tuple, set, frozenset)):
        raise UnknownScopeError(f"scopes must be a collection, got {type(scopes).__name__}")
    values = {str(s) for s in scopes}
    unknown = sorted(values - PERMITTED_SCOPES)
    if unknown:
        raise UnknownScopeError(
            f"scope(s) not in the permitted-use vocabulary: {unknown}. "
            "Add them to rights_vocabulary with a definition, or fix the "
            "spelling — an unrecognised scope silently reads as 'not granted'."
        )
    return frozenset(values)
