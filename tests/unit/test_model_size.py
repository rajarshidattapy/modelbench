from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelbench.config import BenchmarkConfig
from modelbench.metrics.model_size import ModelSizeCollector
from tests.helpers import build_loaded_torch_model


def test_model_size_collector_reports_size_and_parameters(tmp_path):
    loaded_model, input_tensor = build_loaded_torch_model(tmp_path)
    config = BenchmarkConfig(
        model_paths=[loaded_model.model_path],
        model_format="pytorch",
        input_shape=[4],
        batch_sizes=[1],
    )

    metrics = ModelSizeCollector().collect(loaded_model, input_tensor, config)
    metric_names = {metric.name for metric in metrics}

    assert "model_size_mb" in metric_names
    assert "param_count" in metric_names
    assert any(metric.value > 0 for metric in metrics)

