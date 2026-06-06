"""Configuration and report models for ModelBench."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


DEFAULT_METRICS = ["latency", "throughput", "memory", "model_size", "flops"]
DEFAULT_EXPORT_FORMATS = ["json", "markdown"]
DEFAULT_COMPARISON_WEIGHTS = {
    "throughput": 0.30,
    "p99_ms": 0.25,
    "peak_memory_mb": 0.20,
    "model_size_mb": 0.15,
    "gflops_per_inference": 0.10,
}


class BenchmarkConfig(BaseModel):
    """User-provided benchmark configuration."""

    model_paths: List[str]
    model_format: Literal["pytorch", "onnx", "torchscript"]
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    batch_sizes: List[int] = Field(default_factory=lambda: [1, 8, 32])
    warmup_runs: int = 10
    benchmark_runs: int = 100
    throughput_duration_seconds: float = 30.0
    input_shape: List[int]
    dtype: str = "float32"
    output_dir: str = "./modelbench_results"
    metrics: List[str] = Field(default_factory=lambda: list(DEFAULT_METRICS))
    export_formats: List[str] = Field(default_factory=lambda: list(DEFAULT_EXPORT_FORMATS))
    comparison_weights: Dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_COMPARISON_WEIGHTS)
    )

    @field_validator("model_paths", "batch_sizes", "input_shape", mode="after")
    @classmethod
    def _must_not_be_empty(cls, value: List[Any]) -> List[Any]:
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("batch_sizes", mode="after")
    @classmethod
    def _validate_batch_sizes(cls, value: List[int]) -> List[int]:
        if any(size <= 0 for size in value):
            raise ValueError("batch_sizes must contain only positive integers")
        return value

    @field_validator("warmup_runs", "benchmark_runs")
    @classmethod
    def _validate_runs(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("run counts must be positive")
        return value

    @field_validator("throughput_duration_seconds")
    @classmethod
    def _validate_duration(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("throughput_duration_seconds must be positive")
        return value

    def output_path(self) -> Path:
        return Path(self.output_dir)


class MetricResult(BaseModel):
    """A single measured metric value."""

    name: str
    value: float
    unit: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelBenchmarkResult(BaseModel):
    """The collected metrics for a model and batch-size combination."""

    model_path: str
    device: str
    batch_size: int
    metrics: List[MetricResult]
    timestamp: str
    duration_seconds: float

    def metric_map(self) -> Dict[str, MetricResult]:
        return {metric.name: metric for metric in self.metrics}


class BenchmarkReport(BaseModel):
    """Top-level report returned by the runner and exporters."""

    config: BenchmarkConfig
    results: List[ModelBenchmarkResult]
    comparison: Optional[Dict[str, Any]] = None

    def export(self, export_format: str, output_dir: str) -> Path:
        from modelbench.reporters import get_reporter

        reporter = get_reporter(export_format)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = reporter.file_extension
        destination = output_path / f"modelbench_report_{timestamp}.{suffix}"
        destination.write_text(reporter.render(self), encoding="utf-8")
        return destination

