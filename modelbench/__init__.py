"""Public package interface for ModelBench."""

from modelbench.config import BenchmarkConfig, BenchmarkReport, MetricResult, ModelBenchmarkResult
from modelbench.runner import ModelBench

__all__ = [
    "BenchmarkConfig",
    "BenchmarkReport",
    "MetricResult",
    "ModelBench",
    "ModelBenchmarkResult",
]

__version__ = "0.1.0"

