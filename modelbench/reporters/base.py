"""Reporter abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from modelbench.config import BenchmarkReport


class Reporter(ABC):
    """Renders a benchmark report into a target format."""

    file_extension: str

    @abstractmethod
    def render(self, report: BenchmarkReport) -> str:
        raise NotImplementedError

