"""Click CLI for ModelBench."""

from __future__ import annotations

import json
from pathlib import Path

import click

from modelbench import __version__
from modelbench.compare import load_reports_from_path, merge_reports
from modelbench.config import BenchmarkConfig
from modelbench.metrics import list_metric_names
from modelbench.reporters import get_reporter
from modelbench.runner import ModelBench


@click.group()
def cli() -> None:
    """CLI entrypoint for ModelBench."""


@cli.command()
@click.option("--model", "model_paths", multiple=True, required=True, type=click.Path(exists=True))
@click.option("--format", "model_format", required=True, type=click.Choice(["pytorch", "onnx", "torchscript"]))
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda", "mps"]))
@click.option("--batch-sizes", multiple=True, type=int, default=(1, 8, 32))
@click.option("--input-shape", multiple=True, type=int, required=True)
@click.option("--warmup", "warmup_runs", default=10, type=int)
@click.option("--runs", "benchmark_runs", default=100, type=int)
@click.option("--throughput-duration", default=30.0, type=float, show_default=True)
@click.option("--dtype", default="float32", show_default=True)
@click.option("--output", "output_dir", default="./modelbench_results", type=click.Path())
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    default=("latency", "throughput", "memory", "model_size", "flops"),
)
@click.option("--export", "export_formats", multiple=True, default=("json", "markdown", "terminal"))
@click.option("--weights", default=None, help="JSON object overriding comparison weights.")
def run(
    model_paths,
    model_format,
    device,
    batch_sizes,
    input_shape,
    warmup_runs,
    benchmark_runs,
    throughput_duration,
    dtype,
    output_dir,
    metrics,
    export_formats,
    weights,
) -> None:
    """Run a benchmark job."""

    comparison_weights = json.loads(weights) if weights else None
    config = BenchmarkConfig(
        model_paths=list(model_paths),
        model_format=model_format,
        device=device,
        batch_sizes=list(batch_sizes),
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        throughput_duration_seconds=throughput_duration,
        input_shape=list(input_shape),
        dtype=dtype,
        output_dir=output_dir,
        metrics=list(metrics),
        export_formats=list(export_formats),
        comparison_weights=comparison_weights or None,
    )
    bench = ModelBench(config)
    report = bench.run()
    exported = bench.export(report)

    if "terminal" in export_formats:
        click.echo(get_reporter("terminal").render(report))
    for name, path in exported.items():
        if name != "terminal":
            click.echo(f"{name}: {path}")


@cli.command()
@click.option("--results", "results_path", required=True, type=click.Path(exists=True))
@click.option("--sort", "sort_metric", default="samples_per_second")
def compare(results_path: str, sort_metric: str) -> None:
    """Load JSON report files and render a merged comparison."""

    reports = load_reports_from_path(results_path)
    report = merge_reports(reports)
    if sort_metric:
        report.results.sort(
            key=lambda result: result.metric_map().get(sort_metric).value
            if result.metric_map().get(sort_metric)
            else float("-inf"),
            reverse=True,
        )

    click.echo(get_reporter("terminal").render(report))


@cli.command("list-metrics")
def list_metrics() -> None:
    """Print available metric collectors."""

    for metric_name in list_metric_names():
        click.echo(metric_name)


@cli.command()
def version() -> None:
    """Print package version."""

    click.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    cli()
