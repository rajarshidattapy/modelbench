from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelbench.config import BenchmarkConfig
from modelbench.metrics.memory import MemoryCollector
from tests.helpers import build_loaded_torch_model


def test_memory_collector_reports_peak_memory(tmp_path):
    loaded_model, input_tensor = build_loaded_torch_model(tmp_path)
    config = BenchmarkConfig(
        model_paths=[loaded_model.model_path],
        model_format="pytorch",
        input_shape=[4],
        batch_sizes=[1],
    )

    metrics = MemoryCollector().collect(loaded_model, input_tensor, config)

    assert len(metrics) == 1
    assert metrics[0].name == "peak_memory_mb"
    assert metrics[0].value >= 0

