"""Reporter registry."""

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
    if name not in REPORTERS:
        raise KeyError(f"Unknown reporter: {name}")
    return REPORTERS[name]

