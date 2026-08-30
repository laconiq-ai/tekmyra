from pathlib import Path

from tekmyra.benchmark import run_benchmark

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "fixtures" / "synthetic"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results" / "test"


def test_run_benchmark():
    report = run_benchmark(FIXTURE_DIR, ["no_compression", "passthrough"], OUTPUT_DIR)
    fixture_count = report["total_fixtures"]
    assert fixture_count >= 50
    assert len(report["results"]) == fixture_count * 2
    assert OUTPUT_DIR.exists()
    assert (OUTPUT_DIR / "benchmark_report.json").exists()
    assert (OUTPUT_DIR / "benchmark_report.md").exists()

    for result in report["results"]:
        assert "locked_spans_detected" in result
        assert "locked_spans_preserved" in result
        assert "pii_entity_types" in result
        assert "output_text" in result
        assert "protected_spans" in result

    for baseline in ["no_compression", "passthrough"]:
        metrics = report["summary"]["by_baseline"][baseline]
        assert metrics["locked_span_recall"] == 1.0
        assert "pii_leakage_count" in metrics
