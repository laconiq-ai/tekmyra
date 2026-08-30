"""The compression operating point. One definition, and nothing else here.

THE SHARE OF CONTENT THE COMPRESSOR AIMS TO KEEP. Every layer that needs a
default budget imports this; none of them restates it.

WHY THIS MODULE EXISTS RATHER THAN A CONSTANT IN ONE OF THOSE LAYERS. On
2026-08-09, landing Hari's ruling of 0.15 found the same value written at
EIGHT sites — ``runtime`` (a named constant plus two signature defaults),
``codecs/selector`` (a signature default, an absent-key default and a
malformed-value fallback), ``codecs/safe_mode`` (two more) and the compressor
itself. The constant in ``runtime`` carried a comment claiming it was "named
here so the layers cannot drift". It had drifted anyway, because naming a
value in one layer does not make it the definition — only the other layers
importing it does.

Every one of those sites was found by a guard failing, not by reading the code.

Worse than duplication: the shipped 0.8 was never the number anyone reported.
Every published figure was measured at 0.30, which the benchmark harness passes
explicitly, so the default delivered **16.30%** corpus character reduction
against a reported 51.38%. No single place held the answer to "what does it
actually run at", so nobody could notice the gap.

THIS MODULE MUST IMPORT NOTHING FROM THE PACKAGE. The obvious home was the
compressor, and it produced a real circular import — ``compressor`` imports
``codecs.base``, which loads ``codecs/__init__``, which loads the selectors,
which would import back into the partially initialised compressor. A leaf
cannot do that, which is the whole reason ``numeric_fidelity``,
``entity_fidelity`` and ``content_hash`` are leaves too.
"""

from __future__ import annotations

# Hari's ruling, 2026-08-09, chosen from 0.30 / 0.15 / 0.05 with the full suite
# green at it. Changing this changes the shipped behaviour of every codec; it is
# a founder decision, not a tuning knob.
DEFAULT_BUDGET_RATIO = 0.15
