"""Task-management tool adapters."""

from __future__ import annotations

import typer

from litecoder.cli.commands import current_paths
from litecoder.tasks.models import TaskRecord, TaskStatus
from litecoder.tasks.store import TaskStore


app = typer.Typer(help="Inspect LiteCoder tasks.")


@app.command("list")
def list_tasks() -> None:
    """Handle the list tasks operation."""
    try:
        records = _load_tasks()
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    typer.echo(render_task_list(records))


@app.command("show")
def show_task(task_id: str) -> None:
    """Handle the show task operation."""
    try:
        records = _load_tasks()
        rendered = render_task_detail(records, task_id)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    except KeyError as error:
        raise typer.BadParameter(error.args[0]) from None
    typer.echo(rendered)


def _load_tasks() -> list[TaskRecord]:
    paths = current_paths()
    return TaskStore(paths.project_dir / "tasks").read_all()


def render_task_list(records: list[TaskRecord]) -> str:
    """Render the task list."""
    if not records:
        return "No tasks."
    by_id = {record.id: record for record in records}
    return "\n".join(
        f"{record.id}\t{_derived_state(record, by_id)}\t{record.subject}"
        for record in records
    )


def render_task_detail(records: list[TaskRecord], task_id: str) -> str:
    """Render the task detail."""
    by_id = {record.id: record for record in records}
    record = by_id.get(task_id)
    if record is None:
        raise KeyError(f"Unknown task {task_id!r}")
    return "\n".join(
        [
            f"id: {record.id}",
            f"subject: {record.subject}",
            f"description: {record.description}",
            f"status: {_derived_state(record, by_id)}",
            f"owner: {record.owner_agent_id or ''}",
            f"dependencies: {', '.join(record.dependencies)}",
        ]
    )


def _derived_state(
    record: TaskRecord,
    by_id: dict[str, TaskRecord],
) -> str:
    if record.status is not TaskStatus.PENDING:
        return record.status.value
    for dependency_id in record.dependencies:
        dependency = by_id[dependency_id]
        if dependency.status is not TaskStatus.COMPLETED:
            return "blocked"
    return record.status.value