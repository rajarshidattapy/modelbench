"""Base metric collector API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from modelbench.config import BenchmarkConfig, MetricResult
from modelbench.loaders.base import LoadedModel


class MetricCollector(ABC):
    """Collects one or more metric values for a model/input pair."""

    name: str

    @abstractmethod
    def collect(
        self,
        model: LoadedModel,
        input_data: Any,
        config: BenchmarkConfig,
    ) -> List[MetricResult]:
        raise NotImplementedError
