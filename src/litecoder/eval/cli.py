"""Command-line interface for evaluation workflows."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from litecoder.cli.app import build_runtime
from litecoder.eval.domain import (
    DatasetSelection,
    RunReport,
    RunSpec,
    validate_dataset,
    validate_mode,
)
from litecoder.eval.evalplus import (
    EvalPlusExecutionError,
    EvalPlusTask,
    EvalPlusUnavailable,
    load_evalplus_tasks,
)
from litecoder.eval.execution import RuntimeCaseExecutor
from litecoder.eval.orchestrator import EvalOrchestrator
from litecoder.eval.provenance import collect_provenance
from litecoder.eval.reporting import (
    render_json,
    render_markdown,
    render_markdown_payload,
)
from litecoder.eval.validation import EvalPlusValidator
from litecoder.ui.renderers.terminal import TerminalRenderer, TerminalUISink


app = typer.Typer(no_args_is_help=True)


def make_run_location(
    mode: str,
    output_root: Path,
    *,
    now: datetime | None = None,
    suffix: str | None = None,
) -> tuple[str, Path]:
    """Build the run location."""
    selected = validate_mode(mode)
    local_now = now if now is not None else datetime.now().astimezone()
    if local_now.tzinfo is None or local_now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    run_suffix = suffix if suffix is not None else uuid.uuid4().hex[:8]
    if len(run_suffix) != 8 or any(
        character not in "0123456789abcdefABCDEF" for character in run_suffix
    ):
        raise ValueError("suffix must be 8 hexadecimal characters")
    timestamp = local_now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{run_suffix}-{selected}"
    run_dir = (
        output_root
        / local_now.strftime("%Y-%m-%d")
        / f"{local_now.strftime('%H%M%S')}-{run_suffix}"
        / selected
    )
    return run_id, run_dir


class TerminalEvalProgress:
    """Component responsible for the terminal eval progress."""
    def case_started(
        self, *, index: int, total: int, task_id: str, workspace: object
    ) -> None:
        """Handle the case started operation."""
        del workspace
        typer.echo(f"[{index}/{total}] {task_id} started")

    def case_finished(
        self,
        *,
        index: int,
        total: int,
        task_id: str,
        status: str,
        failure_reason: str,
    ) -> None:
        """Handle the case finished operation."""
        suffix = f": {failure_reason}" if failure_reason else ""
        typer.echo(f"[{index}/{total}] {task_id} {status}{suffix}")


@app.command("run")
def run_eval(
    mode: Annotated[str, typer.Argument(help="Evaluation mode.")],
    dataset: Annotated[
        str, typer.Option(help="EvalPlus dataset: humaneval or mbpp.")
    ] = "humaneval",
    limit: Annotated[int, typer.Option(help="Number of EvalPlus tasks to run.")] = 15,
    seed: Annotated[int | None, typer.Option(help="Random task selection seed.")] = None,
    output_dir: Annotated[
        Path, typer.Option(help="Directory for evaluation runs.")
    ] = Path("eval-runs"),
) -> None:
    """Run the eval."""
    try:
        selected_mode = validate_mode(mode)
        selected_dataset = validate_dataset(dataset)
        options: dict[str, int] = {"limit": limit}
        if seed is not None:
            options["seed"] = seed
        tasks = load_evalplus_tasks(selected_dataset, **options)
    except (ValueError, EvalPlusUnavailable) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    run_id, run_dir = make_run_location(selected_mode, output_dir)
    typer.echo(f"Run: {run_dir} ({len(tasks)} cases)")
    try:
        report = execute_eval_run(
            mode=selected_mode,
            datasets=selected_dataset,
            output_dir=run_dir,
            tasks=tasks,
            run_id=run_id,
        )
    except EvalPlusExecutionError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(3) from error
    path = write_eval_reports(report)
    typer.echo(str(path))
    if report.status == "interrupted":
        raise typer.Exit(130)


def execute_eval_run(
    *,
    mode: str,
    datasets: DatasetSelection,
    output_dir: Path,
    tasks: tuple[EvalPlusTask, ...],
    run_id: str,
) -> RunReport:
    """Execute the eval run."""
    orchestrator = EvalOrchestrator(
        RuntimeCaseExecutor(
            build_runtime,
            live_ui_sink_factory=lambda workspace: TerminalUISink(
                TerminalRenderer(workspace_root=workspace)
            ),
        ),
        EvalPlusValidator(),
        progress=TerminalEvalProgress(),
    )
    report = asyncio.run(
        orchestrator.run(RunSpec(run_id, mode, datasets, output_dir), tasks)
    )
    return replace(
        report,
        metadata={
            **report.metadata,
            "provenance": collect_provenance(),
        },
    )


def write_eval_reports(report: RunReport) -> Path:
    """Write the eval reports."""
    report.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = report.output_dir / "run.json"
    run_path.write_text(render_json(report), encoding="utf-8")
    (report.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return run_path


@app.command("report")
def report(run_path: Path) -> None:
    """Handle the report operation."""
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter("run report must contain a JSON object")
    typer.echo(render_markdown_payload(payload), nl=False)


def run() -> None:
    """Run the requested operation."""
    app()


if __name__ == "__main__":
    run()
