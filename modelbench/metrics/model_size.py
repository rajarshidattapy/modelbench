"""Model size and parameter count metrics."""

from __future__ import annotations

import os
from typing import Any, List

from modelbench.config import BenchmarkConfig, MetricResult
from modelbench.loaders.base import LoadedModel
from modelbench.metrics.base import MetricCollector


class ModelSizeCollector(MetricCollector):
    """Reports model size and parameter counts."""

    name = "model_size"

    def collect(
        self,
        model: LoadedModel,
        input_data: Any,
        config: BenchmarkConfig,
    ) -> List[MetricResult]:
        del input_data
        del config
        metrics = [
            MetricResult(
                name="model_size_mb",
                value=float(os.path.getsize(model.model_path) / 1e6),
                unit="MB",
            )
        ]

        parameter_count = model.parameter_count()
        if parameter_count is not None:
            metrics.append(MetricResult(name="param_count", value=float(parameter_count), unit="count"))

        non_trainable = model.non_trainable_parameter_count()
        if non_trainable is not None:
            metrics.append(
                MetricResult(name="non_trainable_param_count", value=float(non_trainable), unit="count")
            )

        return metrics

