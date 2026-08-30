from tekmyra.reporting import _compute_baseline_summary


def _make_row(
    predicted_route="passthrough",
    detected=0,
    preserved=0,
    protected_spans=None,
    output_text="",
    content_type="prose",
    risk="low",
    route="passthrough",
    compression_ratio=1.0,
    blocks=None,
):
    return {
        "predicted_route": predicted_route,
        "route": route,
        "predicted_risk": risk,
        "risk": risk,
        "content_type": content_type,
        "locked_spans_detected": detected,
        "locked_spans_preserved": preserved,
        "protected_spans": protected_spans or [],
        "output_text": output_text,
        "compression_ratio": compression_ratio,
        "blocks": blocks or [],
    }


def test_locked_span_recall_defaults_to_one_when_no_detected_spans():
    rows = [_make_row(detected=0, preserved=0)]
    summary = _compute_baseline_summary(rows)
    assert summary["locked_span_recall"] == 1.0


def test_locked_span_recall_computed_across_detected_rows():
    rows = [
        _make_row(detected=3, preserved=3),
        _make_row(detected=2, preserved=1),
        _make_row(detected=0, preserved=0),
    ]
    summary = _compute_baseline_summary(rows)
    assert summary["locked_span_recall"] == 4 / 5


def test_pii_leakage_zero_when_route_is_not_lossy():
    rows = [
        _make_row(
            predicted_route="passthrough",
            protected_spans=[{"span_type": "email", "text": "user@example.com"}],
            output_text="contact user@example.com",
        )
    ]
    summary = _compute_baseline_summary(rows)
    assert summary["pii_leakage_count"] == 0


def test_pii_leakage_counts_lossy_route_with_pii_text_present():
    rows = [
        _make_row(
            predicted_route="prose_token_compressor",
            protected_spans=[{"span_type": "email", "text": "user@example.com"}],
            output_text="contact user@example.com please",
        )
    ]
    summary = _compute_baseline_summary(rows)
    assert summary["pii_leakage_count"] == 1


def test_pii_leakage_ignores_non_pii_span_types():
    rows = [
        _make_row(
            predicted_route="prose_token_compressor",
            protected_spans=[{"span_type": "request_id", "text": "REQ-123"}],
            output_text="REQ-123",
        )
    ]
    summary = _compute_baseline_summary(rows)
    assert summary["pii_leakage_count"] == 0


def test_pii_leakage_zero_when_pii_text_removed():
    rows = [
        _make_row(
            predicted_route="summarizer",
            protected_spans=[{"span_type": "ssn", "text": "123-45-6789"}],
            output_text="redacted",
        )
    ]
    summary = _compute_baseline_summary(rows)
    assert summary["pii_leakage_count"] == 0


def test_compression_ratio_aggregates():
    rows = [
        _make_row(compression_ratio=0.5),
        _make_row(compression_ratio=1.0),
        _make_row(compression_ratio=1.5),
    ]
    summary = _compute_baseline_summary(rows)
    assert summary["mean_compression_ratio"] == 1.0
    assert summary["median_compression_ratio"] == 1.0
    assert summary["min_compression_ratio"] == 0.5
    assert summary["max_compression_ratio"] == 1.5


def test_compression_ratio_defaults_to_zero_when_missing():
    rows = [{"predicted_route": "passthrough", "route": "passthrough", "content_type": "prose", "risk": "low", "predicted_risk": "low"}]
    summary = _compute_baseline_summary(rows)
    assert summary["mean_compression_ratio"] == 0.0
    assert summary["median_compression_ratio"] == 0.0
    assert summary["min_compression_ratio"] == 0.0
    assert summary["max_compression_ratio"] == 0.0


def test_accuracy_by_route_class():
    rows = [
        _make_row(route="passthrough", predicted_route="passthrough"),
        _make_row(route="passthrough", predicted_route="prose_token_compressor"),
        _make_row(route="deterministic_json_compressor", predicted_route="deterministic_json_compressor"),
    ]
    summary = _compute_baseline_summary(rows)
    assert summary["accuracy_by_route_class"]["passthrough"] == 0.5
    assert summary["accuracy_by_route_class"]["deterministic_json_compressor"] == 1.0
    assert summary["accuracy_by_route_class"]["prose_token_compressor"] == 0.0


def test_compression_ratio_by_block_type():
    rows = [
        _make_row(compression_ratio=0.8, blocks=[{"type": "prose"}, {"type": "json"}]),
        _make_row(compression_ratio=1.2, blocks=[{"type": "prose"}]),
    ]
    summary = _compute_baseline_summary(rows)
    assert summary["compression_ratio_by_block_type"]["prose"] == (0.8 + 1.2) / 2
    assert summary["compression_ratio_by_block_type"]["json"] == 0.8


def test_compression_ratio_by_block_type_empty_when_no_blocks():
    rows = [_make_row(compression_ratio=1.0)]
    summary = _compute_baseline_summary(rows)
    assert summary["compression_ratio_by_block_type"] == {}
