from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")

from modelbench.config import BenchmarkConfig
from modelbench.metrics.flops import FLOPsCollector
from tests.helpers import build_loaded_torch_model


def test_flops_collector_skips_onnx_with_warning(tmp_path):
    loaded_model, input_tensor = build_loaded_torch_model(tmp_path)
    loaded_model.format_name = "onnx"
    config = BenchmarkConfig(
        model_paths=[loaded_model.model_path],
        model_format="onnx",
        input_shape=[4],
        batch_sizes=[1],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = FLOPsCollector().collect(loaded_model, input_tensor, config)

    assert metrics == []
    assert any("Skipping FLOPs" in str(item.message) for item in caught)

