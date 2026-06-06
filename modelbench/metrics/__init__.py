"""Metric collector registry."""

from importlib.metadata import entry_points

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
    registry = _metric_registry()
    collectors = []
    for name in metric_names:
        if name not in registry:
            raise KeyError(f"Unknown metric: {name}")
        collectors.append(registry[name]())
    return collectors


def list_metric_names() -> list[str]:
    return sorted(_metric_registry())


def _metric_registry():
    registry = dict(BUILTIN_COLLECTORS)
    for entry_point in entry_points(group="modelbench.metrics"):
        collector_cls = entry_point.load()
        registry[entry_point.name] = collector_cls
    return registry
