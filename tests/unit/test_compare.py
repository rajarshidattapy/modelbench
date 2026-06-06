from __future__ import annotations

from pathlib import Path

from modelbench.compare import build_comparison, load_reports_from_path, merge_reports
from modelbench.config import BenchmarkConfig, BenchmarkReport, MetricResult, ModelBenchmarkResult


def _result(model_path: str, batch_size: int, throughput: float, p99: float) -> ModelBenchmarkResult:
    return ModelBenchmarkResult(
        model_path=model_path,
        device="cpu",
        batch_size=batch_size,
        timestamp="2024-01-01T00:00:00Z",
        duration_seconds=1.0,
        metrics=[
            MetricResult(name="samples_per_second", value=throughput, unit="samples/sec"),
            MetricResult(name="p99_ms", value=p99, unit="ms"),
            MetricResult(name="peak_memory_mb", value=100.0 + batch_size, unit="MB"),
            MetricResult(name="model_size_mb", value=10.0, unit="MB"),
            MetricResult(name="gflops_per_inference", value=2.0, unit="GFLOPs"),
        ],
    )


def test_build_comparison_ranks_models_by_weighted_score():
    results = [
        _result("a.pt", 1, throughput=100.0, p99=10.0),
        _result("b.pt", 1, throughput=200.0, p99=8.0),
        _result("a.pt", 4, throughput=300.0, p99=20.0),
        _result("b.pt", 4, throughput=500.0, p99=18.0),
    ]

    comparison = build_comparison(results)

    assert comparison["winner"] == "b.pt"
    assert comparison["by_batch_size"]["1"]["winner"] == "b.pt"
    assert comparison["by_batch_size"]["4"]["winner"] == "b.pt"
    assert comparison["by_batch_size"]["1"]["metric_winners"]["samples_per_second"] == "b.pt"


def test_load_and_merge_reports_from_directory(tmp_path):
    config = BenchmarkConfig(
        model_paths=["a.pt", "b.pt"],
        model_format="pytorch",
        input_shape=[4],
        batch_sizes=[1],
    )
    report_one = BenchmarkReport(config=config, results=[_result("a.pt", 1, 100.0, 10.0)])
    report_two = BenchmarkReport(config=config, results=[_result("b.pt", 1, 200.0, 8.0)])

    path_one = tmp_path / "report_one.json"
    path_two = tmp_path / "report_two.json"
    path_one.write_text(report_one.model_dump_json(indent=2), encoding="utf-8")
    path_two.write_text(report_two.model_dump_json(indent=2), encoding="utf-8")

    reports = load_reports_from_path(str(tmp_path))
    merged = merge_reports(reports)

    assert len(reports) == 2
    assert len(merged.results) == 2
    assert merged.comparison["winner"] == "b.pt"


def test_load_reports_from_single_file(tmp_path):
    config = BenchmarkConfig(
        model_paths=["a.pt"],
        model_format="pytorch",
        input_shape=[4],
        batch_sizes=[1],
    )
    report = BenchmarkReport(config=config, results=[_result("a.pt", 1, 100.0, 10.0)])
    report_path = tmp_path / "report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    loaded = load_reports_from_path(str(report_path))

    assert len(loaded) == 1
    assert loaded[0].results[0].model_path == "a.pt"

