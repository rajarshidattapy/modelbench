"""FLOPs metric collection with graceful fallbacks."""

from __future__ import annotations

import warnings
from typing import Any, List

from modelbench.config import BenchmarkConfig, MetricResult
from modelbench.loaders.base import LoadedModel
from modelbench.metrics.base import MetricCollector


class FLOPsCollector(MetricCollector):
    """Reports FLOPs per inference when supported."""

    name = "flops"

    def collect(
        self,
        model: LoadedModel,
        input_data: Any,
        config: BenchmarkConfig,
    ) -> List[MetricResult]:
        del config
        if model.format_name == "onnx":
            warnings.warn("Skipping FLOPs for ONNX models: computation is not implemented.", stacklevel=2)
            return []

        try:
            from fvcore.nn import FlopCountAnalysis  # type: ignore

            total_flops = float(FlopCountAnalysis(model.backend, input_data).total())
            return self._build_results(total_flops)
        except Exception:
            pass

        try:
            from thop import profile  # type: ignore

            total_flops, _ = profile(model.backend, inputs=(input_data,), verbose=False)
            return self._build_results(float(total_flops))
        except Exception as exc:
            warnings.warn(
                f"Skipping FLOPs for {model.model_path}: {exc}",
                stacklevel=2,
            )
            return []

    def _build_results(self, total_flops: float) -> List[MetricResult]:
        return [
            MetricResult(name="flops_per_inference", value=total_flops, unit="FLOPs"),
            MetricResult(name="gflops_per_inference", value=total_flops / 1e9, unit="GFLOPs"),
        ]

