"""Benchmark orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Dict, List

from modelbench.compare import build_comparison
from modelbench.config import BenchmarkConfig, BenchmarkReport, ModelBenchmarkResult
from modelbench.loaders import ONNXModelLoader, PyTorchModelLoader
from modelbench.metrics import get_metric_collectors


LOADER_MAP = {
    "pytorch": PyTorchModelLoader,
    "torchscript": PyTorchModelLoader,
    "onnx": ONNXModelLoader,
}


class ModelBench:
    """Runs benchmark jobs from a validated config."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def run(self) -> BenchmarkReport:
        loader = self._build_loader()
        collectors = get_metric_collectors(self.config.metrics)
        results: List[ModelBenchmarkResult] = []

        for model_path in self.config.model_paths:
            loaded_model = loader.load(model_path, self.config.device)
            for batch_size in self.config.batch_sizes:
                input_data = loader.make_input(
                    batch_size=batch_size,
                    input_shape=self.config.input_shape,
                    dtype=self.config.dtype,
                    device=self.config.device,
                )
                started_at = perf_counter()
                metrics = []
                for collector in collectors:
                    metrics.extend(collector.collect(loaded_model, input_data, self.config))
                results.append(
                    ModelBenchmarkResult(
                        model_path=model_path,
                        device=self.config.device,
                        batch_size=batch_size,
                        metrics=metrics,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        duration_seconds=perf_counter() - started_at,
                    )
                )

        comparison = None
        if len(self.config.model_paths) > 1:
            comparison = build_comparison(results, self.config.comparison_weights)

        return BenchmarkReport(config=self.config, results=results, comparison=comparison)

    def export(self, report: BenchmarkReport) -> Dict[str, str]:
        exported = {}
        for export_format in self.config.export_formats:
            exported[export_format] = str(report.export(export_format, self.config.output_dir))
        return exported

    def _build_loader(self):
        loader_cls = LOADER_MAP[self.config.model_format]
        return loader_cls()

