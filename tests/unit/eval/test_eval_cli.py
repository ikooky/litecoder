from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from litecoder.eval import cli as eval_cli
from litecoder.eval.domain import RunReport


runner = CliRunner()


def test_evaluation_entrypoints_are_not_in_product_cli_package() -> None:
    assert importlib.util.find_spec("litecoder.cli.eval") is None
    assert importlib.util.find_spec("litecoder.cli.eval_suite") is None
    assert importlib.util.find_spec("litecoder.eval.cli") is not None
    assert importlib.util.find_spec("litecoder.eval.suite") is not None


def test_make_run_location_uses_structured_layout(tmp_path: Path) -> None:
    run_id, path = eval_cli.make_run_location(
        "agent-benchmark",
        tmp_path,
        now=datetime(2026, 7, 27, 20, 0, tzinfo=UTC),
        suffix="a1b2c3d4",
    )

    assert run_id == "20260727T200000Z-a1b2c3d4-agent-benchmark"
    assert path == tmp_path / "2026-07-27" / "200000-a1b2c3d4" / "agent-benchmark"


def test_write_eval_reports_writes_run_json_and_report_md(tmp_path: Path) -> None:
    report = RunReport(
        "run-1",
        "agent-benchmark",
        ("humaneval",),
        tmp_path,
        (),
    )

    path = eval_cli.write_eval_reports(report)

    assert path.name == "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-1"
    assert "schema_version" not in payload
    assert payload["metadata"] == {}
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Schema:" not in markdown


def test_report_command_renders_run_json(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "mode": "agent-benchmark",
                "datasets": ["humaneval"],
                "status": "completed",
                "primary_metrics": {},
                "supporting_metrics": {},
                "guardrails": {},
                "cases": [],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(eval_cli.app, ["report", str(path)])

    assert result.exit_code == 0
    assert "# agent-benchmark" in result.output
    assert "Run: run-1" in result.output
    assert "| Failure |" in result.output
