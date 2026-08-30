"""One vocabulary for corpus content types, and one place to state divergence.

The set of content types the compressor MEASURES and the set it TRAINS ON were
written twice, in ``benchmark_compressor`` and ``training/train_compressor``,
and they had drifted apart by exactly one type without anyone deciding to. The
guard added after the first vocabulary drift covered only the measuring copy, so
it passed green while the two disagreed — a guard against divergence that could
see one of the two things that can diverge.

They are allowed to differ. Measuring a content type and training on it are
different questions, and there are good reasons to measure something you do not
train on. What is not allowed is for the difference to be invisible: a
divergence nobody chose is how the whole class of defect starts.

So the vocabulary is defined ONCE, and every departure from it is a named
constant with a reason attached. Both consumers derive their set from here, and
the drift guard checks both.

Deliberately a leaf: it imports nothing, so the trainer, the benchmark and the
tests can all share it without any of them inheriting the others' dependencies.
"""

from __future__ import annotations

# Every content type the compressor benchmark will measure. Three corpora were
# built at different times with three vocabularies for the same kinds of text;
# these are all of them that name natural-language content a prose compressor is
# meant to handle.
MEASURABLE_CONTENT_TYPES = frozenset(
    {
        # Synthetic corpus.
        "prose",
        # Training corpus (learned_train), used by the router.
        "agent_trace",
        # long_context_v1 — all six types.
        "long_prose",
        "rag_multi_doc",
        "agent_handoffs",
        "incident_reports",
        "issue_threads",
        "distractor_dense",
        # Matches no fixture in any corpus today; see KNOWN_DEAD_CONTENT_TYPES.
        "rag",
    }
)

# MEASURED BUT NOT TRAINED ON, and this is the ONLY such type.
#
# `incident_reports` is absent from the compressor's training set while being
# present in the measurement set. That asymmetry is recorded here rather than
# silently encoded in a second literal, because it is load-bearing for an open
# question: all six incident_reports fixtures in long_context_v1 refuse
# verification, and content_type is a live training feature, so out-of-
# distribution abstention is a plausible mechanism.
#
# IT IS NOT ESTABLISHED AS THE CAUSE. Two other training-excluded fixtures — one
# distractor_dense and one rag_multi_doc, both excluded on RISK TIER rather than
# content type — do not refuse, so "training-excluded implies refuses" is false.
# This constant exists to make the divergence reviewable, NOT to assert why it
# matters. Adding incident_reports here would retrain the model and change the
# published numbers, so it is a deliberate, separate decision.
TRAINING_EXCLUDED_CONTENT_TYPES = frozenset({"incident_reports"})

TRAINABLE_CONTENT_TYPES = MEASURABLE_CONTENT_TYPES - TRAINING_EXCLUDED_CONTENT_TYPES

# Declared names that match no fixture anywhere, acknowledged on purpose. A name
# here is a promise that it is dead deliberately; the drift guard fails if a name
# is dead WITHOUT being listed, and also if a listed name comes back to life.
#
# `rag` matches nothing in any corpus we own. `agent_trace` is NOT dead in
# general — learned_train carries 160 of them and the router trains on it — but
# it matches nothing in the compressor's own corpora. The distinction matters:
# deleting `agent_trace` because the compressor cannot see it would silently
# narrow a vocabulary the router depends on.
KNOWN_DEAD_CONTENT_TYPES = frozenset({"rag"})
