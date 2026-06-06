from __future__ import annotations

from click.testing import CliRunner

from modelbench.cli import cli
from modelbench.config import BenchmarkConfig, BenchmarkReport, MetricResult, ModelBenchmarkResult


def test_cli_lists_metrics_and_version():
    runner = CliRunner()

    metrics_result = runner.invoke(cli, ["list-metrics"])
    version_result = runner.invoke(cli, ["version"])

    assert metrics_result.exit_code == 0
    assert "latency" in metrics_result.output
    assert version_result.exit_code == 0
    assert "0.1.0" in version_result.output


def test_cli_compare_renders_sorted_report(tmp_path):
    config = BenchmarkConfig(
        model_paths=["a.pt", "b.pt"],
        model_format="pytorch",
        input_shape=[4],
        batch_sizes=[1],
    )
    report = BenchmarkReport(
        config=config,
        results=[
            ModelBenchmarkResult(
                model_path="a.pt",
                device="cpu",
                batch_size=1,
                timestamp="2024-01-01T00:00:00Z",
                duration_seconds=1.0,
                metrics=[
                    MetricResult(name="samples_per_second", value=100.0, unit="samples/sec"),
                    MetricResult(name="p99_ms", value=12.0, unit="ms"),
                ],
            ),
            ModelBenchmarkResult(
                model_path="b.pt",
                device="cpu",
                batch_size=1,
                timestamp="2024-01-01T00:00:00Z",
                duration_seconds=1.0,
                metrics=[
                    MetricResult(name="samples_per_second", value=200.0, unit="samples/sec"),
                    MetricResult(name="p99_ms", value=10.0, unit="ms"),
                ],
            ),
        ],
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["compare", "--results", str(report_path), "--sort", "throughput"])

    assert result.exit_code == 0
    assert "ModelBench Report" in result.output
