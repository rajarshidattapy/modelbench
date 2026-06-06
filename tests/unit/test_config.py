from __future__ import annotations

import pytest
from pydantic import ValidationError

from modelbench.config import BenchmarkConfig


def test_config_validates_empty_lists():
    with pytest.raises(ValidationError):
        BenchmarkConfig(model_paths=[], model_format="pytorch", input_shape=[4])


def test_config_validates_positive_values():
    with pytest.raises(ValidationError):
        BenchmarkConfig(model_paths=["a.pt"], model_format="pytorch", input_shape=[4], batch_sizes=[0])

    with pytest.raises(ValidationError):
        BenchmarkConfig(model_paths=["a.pt"], model_format="pytorch", input_shape=[4], warmup_runs=0)

    with pytest.raises(ValidationError):
        BenchmarkConfig(
            model_paths=["a.pt"],
            model_format="pytorch",
            input_shape=[4],
            throughput_duration_seconds=0,
        )


def test_output_path_returns_path_instance():
    config = BenchmarkConfig(model_paths=["a.pt"], model_format="pytorch", input_shape=[4])

    assert str(config.output_path()).endswith("modelbench_results")

