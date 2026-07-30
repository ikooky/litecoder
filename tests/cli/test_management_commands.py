from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from typer.testing import CliRunner

from litecoder.cli.app import app
from litecoder.cli.commands import remove_session_files, trace_path
from litecoder.cli.tasks import render_task_detail, render_task_list
from litecoder.context.session.models import MessageRecord, SessionRecord, SessionStatus
from litecoder.context.session.store import (
    DeleteSessionTreeResult,
    SQLiteSessionStore,
)
from litecoder.paths import AppPaths
from litecoder.tasks.models import TaskRecord, TaskStatus
from litecoder.tasks.store import TaskStore


runner = CliRunner()


def make_paths(tmp_path: Path) -> AppPaths:
    return AppPaths(
        user_dir=tmp_path / ".litecoder",
        sessions_db=tmp_path / ".litecoder" / "sessions.db",
        project_id="project-1",
        project_dir=tmp_path / ".litecoder" / "projects" / "project-1",
        workspace_id="workspace-1",
        workspace_root=tmp_path,
    )


def use_paths(monkeypatch: pytest.MonkeyPatch, paths: AppPaths) -> None:
    monkeypatch.setattr(
        AppPaths,
        "discover",
        classmethod(lambda cls, cwd, home=None: paths),
    )


async def seed_session_tree(paths: AppPaths) -> tuple[Path, Path]:
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    trace_path = paths.project_dir / "traces" / "root-session.jsonl"
    artifact_path = paths.project_dir / "outputs" / "artifact-root.txt"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text('{"trace_id":"trace-1","event":"root"}\n', encoding="utf-8")
    artifact_path.write_text("large output", encoding="utf-8")
    try:
        await store.create_session(
            SessionRecord.new(
                "root-session",
                paths.project_id,
                paths.workspace_id,
                "fake",
                "model-a",
                title="Root Session",
                status=SessionStatus.IDLE,
                workspace_path=str(paths.workspace_root),
            )
        )
        await store.create_session(
            SessionRecord.new(
                "child-session",
                paths.project_id,
                paths.workspace_id,
                "fake",
                "model-b",
                session_type="derived",
                parent_session_id="root-session",
                status=SessionStatus.INCOMPLETE,
                workspace_path=str(paths.workspace_root),
            )
        )
        await store.append_message(
            MessageRecord(
                "root-session",
                "user",
                [{"type": "text", "text": "hello"}],
            )
        )
        await store.append_message(
            MessageRecord(
                "child-session",
                "user",
                [
                    {
                        "type": "tool_result",
                        "metadata": {
                            "artifact": {"path": str(artifact_path)}
                        },
                    }
                ],
            )
        )
    finally:
        await store.close()
    return trace_path, artifact_path

async def seed_root_session(paths: AppPaths, session_id: str) -> None:
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    try:
        await store.create_session(
            SessionRecord.new(
                session_id,
                paths.project_id,
                paths.workspace_id,
                "fake",
                "model-a",
                workspace_path=str(paths.workspace_root),
                status=SessionStatus.IDLE,
            )
        )
    finally:
        await store.close()



async def session_exists(paths: AppPaths, session_id: str) -> bool:
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    try:
        try:
            await store.load_context(session_id)
            return True
        except KeyError:
            return False
    finally:
        await store.close()


def seed_tasks(paths: AppPaths) -> None:
    store = TaskStore(paths.project_dir / "tasks")
    store.replace_many(
        [
            TaskRecord(
                "dependency",
                "Dependency",
                "Must finish first",
                status=TaskStatus.PENDING,
            ),
            TaskRecord(
                "blocked-task",
                "Blocked task",
                "Waits on dependency",
                dependencies=["dependency"],
                status=TaskStatus.PENDING,
            ),
        ]
    )


def blocked_task_records() -> list[TaskRecord]:
    return [
        TaskRecord(
            "dependency",
            "Dependency",
            "Must finish first",
            status=TaskStatus.PENDING,
        ),
        TaskRecord(
            "blocked-task",
            "Blocked task",
            "Waits on dependency",
            dependencies=["dependency"],
            status=TaskStatus.PENDING,
        ),
    ]


def test_render_task_list_marks_derived_blocked_state() -> None:
    assert render_task_list(blocked_task_records()) == (
        "dependency\tpending\tDependency\n"
        "blocked-task\tblocked\tBlocked task"
    )


def test_render_task_list_reports_no_tasks() -> None:
    assert render_task_list([]) == "No tasks."


def test_render_task_detail_marks_blocked_and_renders_fields() -> None:
    assert render_task_detail(blocked_task_records(), "blocked-task") == (
        "id: blocked-task\n"
        "subject: Blocked task\n"
        "description: Waits on dependency\n"
        "status: blocked\n"
        "owner: \n"
        "dependencies: dependency"
    )


def test_render_task_detail_rejects_unknown_task() -> None:
    with pytest.raises(KeyError) as caught:
        render_task_detail(blocked_task_records(), "missing")

    assert caught.value.args == ("Unknown task 'missing'",)


def test_tasks_list_marks_derived_blocked_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    use_paths(monkeypatch, paths)
    seed_tasks(paths)

    result = runner.invoke(app, ["tasks", "list"])

    assert result.exit_code == 0, result.output
    assert "blocked-task" in result.output
    assert "blocked" in result.output


def test_tasks_show_renders_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    use_paths(monkeypatch, paths)
    seed_tasks(paths)

    result = runner.invoke(app, ["tasks", "show", "blocked-task"])

    assert result.exit_code == 0, result.output
    assert "blocked-task" in result.output
    assert "dependency" in result.output
    assert "blocked" in result.output


def test_sessions_list_and_show_render_seeded_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    use_paths(monkeypatch, paths)
    asyncio.run(seed_session_tree(paths))

    listed = runner.invoke(app, ["sessions", "list"])
    shown = runner.invoke(app, ["sessions", "show", "root-session"])

    assert listed.exit_code == 0, listed.output
    assert "root-session" in listed.output
    assert "child-session" in listed.output
    assert shown.exit_code == 0, shown.output
    assert "Root Session" in shown.output
    assert "messages: 1" in shown.output


def test_sessions_delete_requires_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    use_paths(monkeypatch, paths)
    trace_path, artifact_path = asyncio.run(seed_session_tree(paths))

    result = runner.invoke(
        app, ["sessions", "delete", "root-session"], input="n\n"
    )

    assert result.exit_code == 1
    assert asyncio.run(session_exists(paths, "root-session")) is True
    assert asyncio.run(session_exists(paths, "child-session")) is True
    assert trace_path.exists()
    assert artifact_path.exists()


def test_sessions_delete_cascades_children_trace_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    use_paths(monkeypatch, paths)
    trace_path, artifact_path = asyncio.run(seed_session_tree(paths))

    result = runner.invoke(
        app, ["sessions", "delete", "root-session"], input="y\n"
    )

    assert result.exit_code == 0, result.output
    assert asyncio.run(session_exists(paths, "root-session")) is False
    assert asyncio.run(session_exists(paths, "child-session")) is False
    assert not trace_path.exists()
    assert not artifact_path.exists()


def test_trace_path_preserves_current_project_location_for_valid_root_id(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)

    assert trace_path(paths, "root-session") == (
        paths.project_dir / "traces" / "root-session.jsonl"
    ).resolve()


def test_trace_path_rejects_root_id_traversal_without_reporting_outside_path(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    outside_trace = (
        paths.project_dir / "traces" / "../../outside.jsonl"
    ).resolve()
    outside_trace.parent.mkdir(parents=True, exist_ok=True)
    outside_trace.write_text("outside", encoding="utf-8")

    with pytest.raises(ValueError, match="Trace is unavailable"):
        trace_path(paths, "../../outside")

    assert outside_trace.read_text(encoding="utf-8") == "outside"


def test_remove_session_files_rejects_root_id_traversal_without_unlinking(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    outside_trace = (
        paths.project_dir / "traces" / "../../outside.jsonl"
    ).resolve()
    outside_trace.parent.mkdir(parents=True, exist_ok=True)
    outside_trace.write_text("outside", encoding="utf-8")
    deletion = DeleteSessionTreeResult(
        root_session_id="../../outside",
        deleted_session_ids=("../../outside",),
        artifact_paths=(),
    )

    with pytest.raises(ValueError, match="Trace is unavailable"):
        remove_session_files(paths, deletion)

    assert outside_trace.exists()


def test_trace_command_resolves_child_session_to_root_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    use_paths(monkeypatch, paths)
    asyncio.run(seed_session_tree(paths))

    result = runner.invoke(app, ["trace", "child-session"])

    assert result.exit_code == 0, result.output
    assert '"trace_id":"trace-1"' in result.output
    assert '"event":"root"' in result.output

def test_trace_command_rejects_traversal_without_reading_outside(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    use_paths(monkeypatch, paths)
    asyncio.run(seed_root_session(paths, "../../outside"))
    outside_trace = (
        paths.project_dir / "traces" / "../../outside.jsonl"
    ).resolve()
    outside_trace.parent.mkdir(parents=True, exist_ok=True)
    outside_trace.write_text("outside trace contents", encoding="utf-8")

    result = runner.invoke(app, ["trace", "../../outside"])

    assert result.exit_code == 2
    assert "Trace is unavailable" in result.output
    assert "outside trace contents" not in result.output
    assert outside_trace.exists()


def test_sessions_delete_rejects_traversal_without_unlinking_outside(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    use_paths(monkeypatch, paths)
    asyncio.run(seed_root_session(paths, "../../outside"))
    outside_trace = (
        paths.project_dir / "traces" / "../../outside.jsonl"
    ).resolve()
    outside_trace.parent.mkdir(parents=True, exist_ok=True)
    outside_trace.write_text("outside trace contents", encoding="utf-8")

    result = runner.invoke(
        app, ["sessions", "delete", "../../outside"], input="y\n"
    )

    assert result.exit_code == 2
    assert "Trace is unavailable" in result.output
    assert "outside trace contents" not in result.output
    assert outside_trace.exists()
    assert asyncio.run(session_exists(paths, "../../outside")) is True
