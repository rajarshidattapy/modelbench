"""JSON report rendering."""

from __future__ import annotations

from modelbench.config import BenchmarkReport
from modelbench.reporters.base import Reporter


class JSONReporter(Reporter):
    """Serializes benchmark reports as JSON."""

    file_extension = "json"

    def render(self, report: BenchmarkReport) -> str:
        return report.model_dump_json(indent=2)

