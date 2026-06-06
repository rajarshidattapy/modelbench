"""Metric collector registry."""

from modelbench.metrics.base import MetricCollector
from modelbench.metrics.flops import FLOPsCollector
from modelbench.metrics.latency import LatencyCollector
from modelbench.metrics.memory import MemoryCollector
from modelbench.metrics.model_size import ModelSizeCollector
from modelbench.metrics.throughput import ThroughputCollector


BUILTIN_COLLECTORS = {
    "latency": LatencyCollector,
    "throughput": ThroughputCollector,
    "memory": MemoryCollector,
    "model_size": ModelSizeCollector,
    "flops": FLOPsCollector,
}


def get_metric_collectors(metric_names: list[str]) -> list[MetricCollector]:
    collectors = []
    for name in metric_names:
        if name not in BUILTIN_COLLECTORS:
            raise KeyError(f"Unknown metric: {name}")
        collectors.append(BUILTIN_COLLECTORS[name]())
    return collectors


def list_metric_names() -> list[str]:
    return sorted(BUILTIN_COLLECTORS)

