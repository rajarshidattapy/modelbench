"""Markdown report rendering."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List

from modelbench.config import BenchmarkReport, ModelBenchmarkResult
from modelbench.reporters.base import Reporter


class MarkdownReporter(Reporter):
    """Renders GitHub-friendly comparison tables."""

    file_extension = "md"

    def render(self, report: BenchmarkReport) -> str:
        lines = [f"## ModelBench Report - {datetime.now(timezone.utc).isoformat()} UTC", ""]
        lines.extend(self._build_table(report.results))

        if report.comparison:
            lines.extend(["", "### Comparison", "", f"Winner: **{report.comparison.get('winner', 'n/a')}**"])

        return "\n".join(lines)

    def _build_table(self, results: List[ModelBenchmarkResult]) -> List[str]:
        header = [
            "| Model | Batch | P50 (ms) | P99 (ms) | Throughput (s/s) | Peak Mem (MB) | Params (M) | GFLOPs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        rows = []
        for result in results:
            metrics = result.metric_map()
            rows.append(
                "| {model} | {batch} | {p50:.3f} | {p99:.3f} | {throughput:.3f} | {memory:.3f} | {params:.3f} | {gflops:.3f} |".format(
                    model=Path(result.model_path).name,
                    batch=result.batch_size,
                    p50=_metric_value(metrics, "p50_ms"),
                    p99=_metric_value(metrics, "p99_ms"),
                    throughput=_metric_value(metrics, "samples_per_second"),
                    memory=_metric_value(metrics, "peak_memory_mb"),
                    params=_metric_value(metrics, "param_count") / 1e6,
                    gflops=_metric_value(metrics, "gflops_per_inference"),
                )
            )
        return header + rows


def _metric_value(metrics: dict, name: str) -> float:
    metric = metrics.get(name)
    return float(metric.value) if metric else 0.0

