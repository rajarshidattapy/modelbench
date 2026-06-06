from __future__ import annotations

from modelbench.config import BenchmarkConfig, BenchmarkReport, MetricResult, ModelBenchmarkResult
from modelbench.reporters import get_reporter


def _report() -> BenchmarkReport:
    config = BenchmarkConfig(
        model_paths=["a.pt", "b.pt"],
        model_format="pytorch",
        input_shape=[4],
        batch_sizes=[1],
    )
    results = [
        ModelBenchmarkResult(
            model_path="a.pt",
            device="cpu",
            batch_size=1,
            timestamp="2024-01-01T00:00:00Z",
            duration_seconds=1.0,
            metrics=[
                MetricResult(name="p50_ms", value=5.0, unit="ms"),
                MetricResult(name="p99_ms", value=8.0, unit="ms"),
                MetricResult(name="samples_per_second", value=100.0, unit="samples/sec"),
                MetricResult(name="peak_memory_mb", value=25.0, unit="MB"),
                MetricResult(name="param_count", value=1_000_000, unit="count"),
                MetricResult(name="gflops_per_inference", value=2.0, unit="GFLOPs"),
            ],
        )
    ]
    return BenchmarkReport(config=config, results=results, comparison={"winner": "a.pt"})


def test_json_reporter_renders_valid_json():
    rendered = get_reporter("json").render(_report())

    assert '"winner": "a.pt"' in rendered
    assert '"model_path": "a.pt"' in rendered


def test_markdown_reporter_renders_table():
    rendered = get_reporter("markdown").render(_report())

    assert "| Model | Batch |" in rendered
    assert "Winner: **a.pt**" in rendered


def test_terminal_reporter_renders_summary():
    rendered = get_reporter("terminal").render(_report())

    assert "ModelBench Report" in rendered
    assert "Winner: a.pt" in rendered

