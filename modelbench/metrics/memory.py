"""Memory metric collection."""

from __future__ import annotations

import tracemalloc
from typing import Any, List

import psutil

from modelbench.config import BenchmarkConfig, MetricResult
from modelbench.loaders.base import LoadedModel
from modelbench.metrics.base import MetricCollector


class MemoryCollector(MetricCollector):
    """Measures peak memory usage during inference."""

    name = "memory"

    def collect(
        self,
        model: LoadedModel,
        input_data: Any,
        config: BenchmarkConfig,
    ) -> List[MetricResult]:
        del config
        if model.device == "cuda":
            model.reset_peak_memory()
            model.infer(input_data)
            model.synchronize()
            peak_bytes = model.peak_memory_bytes() or 0
            return [MetricResult(name="peak_memory_mb", value=peak_bytes / 1e6, unit="MB")]

        if model.device == "mps":
            model.infer(input_data)
            peak_bytes = model.peak_memory_bytes() or 0
            return [MetricResult(name="peak_memory_mb", value=peak_bytes / 1e6, unit="MB")]

        process = psutil.Process()
        rss_before = process.memory_info().rss
        tracemalloc.start()
        model.infer(input_data)
        current_bytes, peak_tracemalloc_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del current_bytes
        rss_after = process.memory_info().rss
        peak_bytes = max(max(rss_after - rss_before, 0), peak_tracemalloc_bytes)
        return [
            MetricResult(
                name="peak_memory_mb",
                value=float(peak_bytes / 1e6),
                unit="MB",
                metadata={"rss_delta_bytes": max(rss_after - rss_before, 0)},
            )
        ]

