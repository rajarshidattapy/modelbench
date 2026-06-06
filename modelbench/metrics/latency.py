"""Latency metric collection."""

from __future__ import annotations

import statistics
import time
from typing import Any, List

import numpy as np

from modelbench.config import BenchmarkConfig, MetricResult
from modelbench.loaders.base import LoadedModel
from modelbench.metrics.base import MetricCollector


class LatencyCollector(MetricCollector):
    """Measures per-inference latency after warmup."""

    name = "latency"

    def collect(
        self,
        model: LoadedModel,
        input_data: Any,
        config: BenchmarkConfig,
    ) -> List[MetricResult]:
        for _ in range(config.warmup_runs):
            model.infer(input_data)
            model.synchronize()

        latencies_ms = []
        for _ in range(config.benchmark_runs):
            model.synchronize()
            started_at = time.perf_counter()
            model.infer(input_data)
            model.synchronize()
            latencies_ms.append((time.perf_counter() - started_at) * 1000.0)

        return [
            MetricResult(name="p50_ms", value=float(np.percentile(latencies_ms, 50)), unit="ms"),
            MetricResult(name="p95_ms", value=float(np.percentile(latencies_ms, 95)), unit="ms"),
            MetricResult(name="p99_ms", value=float(np.percentile(latencies_ms, 99)), unit="ms"),
            MetricResult(name="mean_ms", value=float(statistics.fmean(latencies_ms)), unit="ms"),
            MetricResult(
                name="std_ms",
                value=float(statistics.pstdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0,
                unit="ms",
            ),
        ]

