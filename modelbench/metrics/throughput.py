"""Throughput metric collection."""

from __future__ import annotations

import time
from typing import Any, List

from modelbench.config import BenchmarkConfig, MetricResult
from modelbench.loaders.base import LoadedModel
from modelbench.metrics.base import MetricCollector


class ThroughputCollector(MetricCollector):
    """Measures sustained samples per second."""

    name = "throughput"

    def collect(
        self,
        model: LoadedModel,
        input_data: Any,
        config: BenchmarkConfig,
    ) -> List[MetricResult]:
        batch_size = len(input_data) if hasattr(input_data, "__len__") else 1

        for _ in range(max(1, min(config.warmup_runs, 5))):
            model.infer(input_data)
            model.synchronize()

        total_samples = 0
        started_at = time.perf_counter()
        deadline = started_at + config.throughput_duration_seconds
        while time.perf_counter() < deadline:
            model.infer(input_data)
            model.synchronize()
            total_samples += batch_size

        duration = max(time.perf_counter() - started_at, 1e-9)
        return [
            MetricResult(
                name="samples_per_second",
                value=float(total_samples / duration),
                unit="samples/sec",
                metadata={"duration_seconds": duration},
            )
        ]

