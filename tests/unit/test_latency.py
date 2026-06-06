from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelbench.config import BenchmarkConfig
from modelbench.metrics.latency import LatencyCollector
from tests.helpers import build_loaded_torch_model


def test_latency_collector_reports_expected_metrics(tmp_path):
    loaded_model, input_tensor = build_loaded_torch_model(tmp_path)
    config = BenchmarkConfig(
        model_paths=[loaded_model.model_path],
        model_format="pytorch",
        input_shape=[4],
        batch_sizes=[1],
        warmup_runs=1,
        benchmark_runs=3,
    )

    metrics = LatencyCollector().collect(loaded_model, input_tensor, config)
    metric_names = {metric.name for metric in metrics}

    assert {"p50_ms", "p95_ms", "p99_ms", "mean_ms", "std_ms"} <= metric_names
    assert all(metric.value >= 0 for metric in metrics)

