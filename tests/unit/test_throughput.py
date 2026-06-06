from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelbench.config import BenchmarkConfig
from modelbench.metrics.throughput import ThroughputCollector
from tests.helpers import build_loaded_torch_model


def test_throughput_collector_reports_samples_per_second(tmp_path):
    loaded_model, input_tensor = build_loaded_torch_model(tmp_path)
    config = BenchmarkConfig(
        model_paths=[loaded_model.model_path],
        model_format="pytorch",
        input_shape=[4],
        batch_sizes=[1],
        warmup_runs=1,
        benchmark_runs=2,
        throughput_duration_seconds=0.01,
    )

    metrics = ThroughputCollector().collect(loaded_model, input_tensor, config)

    assert len(metrics) == 1
    assert metrics[0].name == "samples_per_second"
    assert metrics[0].value > 0

