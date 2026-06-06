"""Reporter registry."""

from importlib.metadata import entry_points

from modelbench.reporters.base import Reporter
from modelbench.reporters.json_reporter import JSONReporter
from modelbench.reporters.markdown_reporter import MarkdownReporter
from modelbench.reporters.terminal_reporter import TerminalReporter


REPORTERS = {
    "json": JSONReporter(),
    "markdown": MarkdownReporter(),
    "terminal": TerminalReporter(),
}


def get_reporter(name: str) -> Reporter:
    registry = _reporter_registry()
    if name not in registry:
        raise KeyError(f"Unknown reporter: {name}")
    return registry[name]


def _reporter_registry():
    registry = dict(REPORTERS)
    for entry_point in entry_points(group="modelbench.reporters"):
        registry[entry_point.name] = entry_point.load()()
    return registry
