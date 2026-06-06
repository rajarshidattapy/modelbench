from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

torch = pytest.importorskip("torch")

from modelbench.cli import cli
from modelbench.compare import load_reports_from_path
from modelbench.config import BenchmarkReport
from tests.helpers import build_torchscript_model_file


def test_cli_run_writes_valid_json_and_markdown(tmp_path):
    model_path = build_torchscript_model_file(tmp_path)
    output_dir = tmp_path / "results"
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "run",
            "--model",
            str(model_path),
            "--format",
            "pytorch",
            "--device",
            "cpu",
            "--batch-sizes",
            "1",
            "--batch-sizes",
            "4",
            "--input-shape",
            "4",
            "--warmup",
            "1",
            "--runs",
            "2",
            "--throughput-duration",
            "0.01",
            "--output",
            str(output_dir),
            "--export",
            "json",
            "--export",
            "markdown",
        ],
    )

    assert result.exit_code == 0, result.output

    reports = load_reports_from_path(str(output_dir))
    assert len(reports) == 1

    json_report_paths = list(output_dir.glob("*.json"))
    markdown_report_paths = list(output_dir.glob("*.md"))
    assert json_report_paths
    assert markdown_report_paths

    validated = BenchmarkReport.model_validate_json(json_report_paths[0].read_text(encoding="utf-8"))
    assert len(validated.results) == 2

