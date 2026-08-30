"""Shared sentence-boundary constants.

Single source of truth for the abbreviation set used by both the baseline
splitter (`tekmyra.baselines`) and the learned-compressor segmenter
(`tekmyra.learned_compressor.segmenter`). Lives in its own
zero-dependency module because the segmenter is part of the trainer's
provenance-bound import surface and must not drag the baseline harness
(runtime, prober, downstream_eval) into it, while the baselines module must
not import the learned_compressor package at module level (its learned path
is deliberately lazy and fail-closed).
"""

from __future__ import annotations

# Abbreviations whose trailing period should not end a sentence.
ABBREVIATIONS = frozenset(
    {
        "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st",
        "inc", "corp", "ltd", "llc", "plc", "llp",
        "eg", "ie", "etc", "et al", "vs", "vol", "vols",
        "fig", "figs", "no", "nos", "pp", "pg",
    }
)
