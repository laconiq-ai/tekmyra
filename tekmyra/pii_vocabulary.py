"""The PII span-type vocabularies, in one place, still distinct.

Six definition sites held THREE DIFFERENT SETS, and the same identifier meant
two different things: ``PII_SPAN_TYPES`` was 9 values in ``reporting`` and 5 in
``fixtures``, ``eligibility`` and the trainer. A reader confirming that two
files agreed could open the wrong pair and come away satisfied.

CONSOLIDATED, NOT MERGED. Vega's ruling is explicit that these must not be
unioned: a union of denylists is not an allowlist, and unioning these would
adopt ``dob`` and ``ip_address`` into the leakage definition purely because
somebody once typed them somewhere. So each set keeps its own name and its own
reason, and the divergence is now visible side by side instead of spread across
six files. Deciding which of them is *right* is the taxonomy ruling, which is
not this module's job and must not be smuggled in by defining a fourth set.

NONE OF THESE CAN CURRENTLY MATCH ANYTHING. Every value here is a DATA
CATEGORY, while ``span_type`` on every span in the estate carries a HANDLING
LABEL — must-preserve, locked, fact, anchor. ``span_type in PII_LEAKAGE_TYPES``
compares an instruction against a category list: both sides are strings, the
comparison typechecks, and it returns False forever. Aligning these vocabularies
does not fix that and must not be allowed to close it — the fix is a separate
dimension on the span contract. Recorded here because this module is where
someone will come looking when they ask why the leakage counter reads zero.

A leaf: it imports nothing, so every consumer can share it without inheriting
anything.
"""

from __future__ import annotations

# What counts as a PII LEAK in an emitted artifact. Used by the preservation
# oracle and by reporting, which held byte-identical copies of it.
LEAKAGE_TYPES = frozenset(
    {
        "phone",
        "ssn",
        "email",
        "name",
        "address",
        "tax_id",
        "account_number",
    }
)

# What the fixture loader, the eligibility gate and the compressor trainer treat
# as a PII-bearing span. Narrower than LEAKAGE_TYPES and NOT a subset of it:
# it adds credit_card and ip_address while omitting name, address, tax_id and
# account_number. Whether that asymmetry is deliberate is a taxonomy question.
DETECTION_TYPES = frozenset(
    {
        "ssn",
        "phone",
        "email",
        "credit_card",
        "ip_address",
    }
)

# The widest set, used only by reporting's span accounting. LEAKAGE_TYPES plus
# credit_card and dob. `dob` appears NOWHERE ELSE in the codebase — not in a
# fixture, not in a detector, not in another list.
REPORTING_TYPES = frozenset(LEAKAGE_TYPES | {"credit_card", "dob"})

# The abandoned convention. Five shipped call sites tested
# ``span_type.startswith("pii_")`` as an escape hatch beside the sets above.
# Measured across all three corpora, over 8,092 spans covering both
# ``protected_spans`` and gold ``must_preserve_spans``: ZERO use it, and none of
# the 30 distinct span_type values in the estate adopt the convention. The
# hatches are retired; this constant exists so the guard test has one definition
# to police rather than a string repeated in six places.
RETIRED_PREFIX = "pii_"
