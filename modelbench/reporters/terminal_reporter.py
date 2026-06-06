"""Rich terminal report rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table

from modelbench.compare import LOWER_IS_BETTER
from modelbench.config import BenchmarkReport, ModelBenchmarkResult
from modelbench.reporters.base import Reporter


class TerminalReporter(Reporter):
    """Renders a terminal-friendly summary using Rich."""

    file_extension = "txt"

    def render(self, report: BenchmarkReport) -> str:
        console = Console(record=True, width=120)
        table = Table(title="ModelBench Report")
        for column in ["Model", "Batch", "P50 (ms)", "P99 (ms)", "Throughput", "Peak Mem", "Params (M)", "GFLOPs"]:
            table.add_column(column, justify="right" if column != "Model" else "left")

        highlights = self._highlight_map(report.results)
        for result in report.results:
            model_name = Path(result.model_path).name
            metrics = result.metric_map()
            table.add_row(
                model_name,
                str(result.batch_size),
                self._styled_number("p50_ms", metrics, highlights),
                self._styled_number("p99_ms", metrics, highlights),
                self._styled_number("samples_per_second", metrics, highlights),
                self._styled_number("peak_memory_mb", metrics, highlights),
                self._styled_number("param_count", metrics, highlights, scale=1e6),
                self._styled_number("gflops_per_inference", metrics, highlights),
            )

        console.print(table)
        if report.comparison and report.comparison.get("winner"):
            console.print(f"Winner: [bold green]{report.comparison['winner']}[/bold green]")
        return console.export_text(styles=True)

    def _highlight_map(self, results: List[ModelBenchmarkResult]) -> Dict[str, Dict[str, float]]:
        metric_names = ["p50_ms", "p99_ms", "samples_per_second", "peak_memory_mb", "param_count", "gflops_per_inference"]
        highlight_map: Dict[str, Dict[str, float]] = {}
        for metric_name in metric_names:
            values = []
            for result in results:
                metric = result.metric_map().get(metric_name)
                if metric is not None:
                    values.append(metric.value)
            if values:
                highlight_map[metric_name] = {"best": min(values), "worst": max(values)}
                if metric_name not in LOWER_IS_BETTER:
                    highlight_map[metric_name] = {"best": max(values), "worst": min(values)}
        return highlight_map

    def _styled_number(
        self,
        metric_name: str,
        metrics: dict,
        highlights: Dict[str, Dict[str, float]],
        scale: float = 1.0,
    ) -> str:
        metric = metrics.get(metric_name)
        if not metric:
            return "-"
        value = metric.value / scale
        style = "white"
        metric_highlights = highlights.get(metric_name, {})
        if metric_highlights:
            if metric.value == metric_highlights["best"]:
                style = "green"
            elif metric.value == metric_highlights["worst"]:
                style = "red"
        return f"[{style}]{value:.3f}[/{style}]"
