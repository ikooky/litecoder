from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from litecoder.hooks import HookManager
from litecoder.tools.background import (
    BackgroundManager,
    BackgroundStatus,
    register_background_tools,
)
from litecoder.tools.duplicate_guard import DuplicateGuard
from litecoder.tools.models import (
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolFailure,
    ToolResult,
    ToolSpec,
)
from litecoder.tools.executor import ToolExecutor
from litecoder.tools.permission import PermissionService
from litecoder.tools.registry import ToolRegistry
from litecoder.tools.workspace_version import WorkspaceStateRegistry
from tests.unit.tools.test_executor_pipeline import RecordingTrace


@pytest.mark.asyncio
async def test_background_completion_becomes_runtime_notification() -> None:
    manager = BackgroundManager(id_factory=lambda: "bg-1")

    handle = await manager.start(
        asyncio.sleep(0, result="done"),
        {"tool": "run_shell"},
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    notifications = await manager.drain_notifications()

    assert handle.id == "bg-1"
    assert manager.status(handle.id).status is BackgroundStatus.COMPLETED
    assert notifications[0].background_id == handle.id
    assert notifications[0].status is BackgroundStatus.COMPLETED
    assert "done" in notifications[0].content
    await manager.close()


@pytest.mark.asyncio
async def test_shutdown_cancels_running_background_tasks() -> None:
    manager = BackgroundManager(id_factory=lambda: "bg-1")
    handle = await manager.start(asyncio.Event().wait(), {})

    await manager.close()

    assert manager.status(handle.id).status is BackgroundStatus.CANCELLED


@pytest.mark.asyncio
async def test_background_cancel_emits_cancelled_notification() -> None:
    manager = BackgroundManager(id_factory=lambda: "bg-1")
    handle = await manager.start(asyncio.Event().wait(), {})

    state = await manager.cancel(handle.id)
    notifications = await manager.drain_notifications()

    assert state.status is BackgroundStatus.CANCELLED
    assert notifications[0].status is BackgroundStatus.CANCELLED
    await manager.close()


@pytest.mark.asyncio
async def test_background_tools_schedule_registered_tool_pipeline(
    tmp_path: Path,
) -> None:
    manager = BackgroundManager(id_factory=lambda: "bg-1")
    calls: list[tuple[str, dict[str, object]]] = []

    async def runner(
        tool_name: str,
        arguments: dict[str, object],
        context: ToolContext,
    ) -> ToolResult:
        calls.append((tool_name, arguments))
        return ToolResult("nested-call", "success", "finished")

    registry = ToolRegistry()
    register_background_tools(registry, manager, runner)
    context = ToolContext("session", "workspace", tmp_path)
    execution = await registry.require("background_start").execute(
        ToolCall(
            "call-1",
            "background_start",
            {"tool_name": "read_file", "input": {"path": "README.md"}},
        ),
        context,
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    notifications = await manager.drain_notifications()

    assert json.loads(execution.content)["background_id"] == "bg-1"
    assert calls == [("read_file", {"path": "README.md"})]
    assert notifications[0].content == "finished"
    assert {
        tool.spec.name for tool in registry.list()
    } == {"background_cancel", "background_start", "background_status"}
    assert all(
        tool.spec.workspace_lock is False
        for tool in registry.list()
    )
    await manager.close()


@pytest.mark.asyncio
async def test_background_cancel_can_cancel_task_holding_workspace_lock(
    tmp_path: Path,
) -> None:
    manager = BackgroundManager(id_factory=lambda: "bg-1")
    writer_started = asyncio.Event()
    registry = ToolRegistry()

    class BlockingWriter:
        spec = ToolSpec("write", "write", {}, True)

        async def execute(
            self, call: ToolCall, context: ToolContext
        ) -> ToolExecution:
            del call, context
            writer_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    registry.register(BlockingWriter())
    executor: ToolExecutor

    async def runner(
        tool_name: str,
        arguments: dict[str, object],
        context: ToolContext,
    ) -> ToolResult:
        return await executor.execute(
            ToolCall(f"background-{tool_name}", tool_name, arguments), context
        )

    register_background_tools(registry, manager, runner)
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=RecordingTrace()),
        DuplicateGuard(),
        PermissionService(prompt=lambda _: "Allow once"),
        WorkspaceStateRegistry(),
    )
    context = ToolContext("session", "workspace", tmp_path)
    started = await executor.execute(
        ToolCall(
            "start",
            "background_start",
            {"tool_name": "write", "input": {}},
        ),
        context,
    )
    background_id = json.loads(started.content)["background_id"]
    await asyncio.wait_for(writer_started.wait(), timeout=1.0)

    cancelled = await asyncio.wait_for(
        executor.execute(
            ToolCall(
                "cancel", "background_cancel", {"background_id": background_id}
            ),
            context,
        ),
        timeout=1.0,
    )

    assert cancelled.status == "success"
    assert manager.status(background_id).status is BackgroundStatus.CANCELLED
    await manager.close()

@pytest.mark.asyncio
async def test_background_status_and_cancel_are_session_scoped(
    tmp_path: Path,
) -> None:
    manager = BackgroundManager(id_factory=lambda: "bg-1")

    async def runner(
        tool_name: str,
        arguments: dict[str, object],
        context: ToolContext,
    ) -> ToolResult:
        del tool_name, arguments, context
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    registry = ToolRegistry()
    register_background_tools(registry, manager, runner)
    owner = ToolContext("session-a", "workspace", tmp_path)
    other = ToolContext("session-b", "workspace", tmp_path)
    execution = await registry.require("background_start").execute(
        ToolCall(
            "call-1",
            "background_start",
            {"tool_name": "read_file", "input": {"path": "README.md"}},
        ),
        owner,
    )
    background_id = json.loads(execution.content)["background_id"]

    with pytest.raises(ToolFailure, match="not owned"):
        await registry.require("background_status").execute(
            ToolCall(
                "call-2",
                "background_status",
                {"background_id": background_id},
            ),
            other,
        )
    with pytest.raises(ToolFailure, match="not owned"):
        await registry.require("background_cancel").execute(
            ToolCall(
                "call-3",
                "background_cancel",
                {"background_id": background_id},
            ),
            other,
        )

    owner_result = await registry.require("background_cancel").execute(
        ToolCall(
            "call-4",
            "background_cancel",
            {"background_id": background_id},
        ),
        owner,
    )

    assert json.loads(owner_result.content)["status"] == "cancelled"
    await manager.close()


@pytest.mark.asyncio
async def test_real_runtime_registers_background_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from litecoder.cli.app import build_runtime
    from litecoder.paths import AppPaths

    user_dir = tmp_path / ".litecoder"
    user_dir.mkdir()
    (user_dir / "config.toml").write_text(
        'default_provider = "fake"\n'
        'default_model = "model"\n'
        '[providers.fake]\n'
        'type = "openai-chat-completions"\n'
        'model = "model"\n'
        'api_key = "key"\n',
        encoding="utf-8",
    )
    paths = AppPaths(
        user_dir=user_dir,
        sessions_db=user_dir / "sessions.db",
        project_id="project",
        project_dir=user_dir / "projects" / "project",
        workspace_id="workspace",
        workspace_root=tmp_path,
    )
    monkeypatch.setattr(
        "litecoder.cli.app.AppPaths.discover",
        staticmethod(lambda cwd: paths),
    )
    names: list[str] = []
    original = ToolRegistry.register

    def recording_register(self: ToolRegistry, tool: object) -> None:
        names.append(tool.spec.name)  # type: ignore[attr-defined]
        original(self, tool)  # type: ignore[arg-type]

    monkeypatch.setattr(ToolRegistry, "register", recording_register)
    runtime = await build_runtime(tmp_path)
    try:
        assert {
            "background_cancel",
            "background_start",
            "background_status",
            "todo_write",
            "task_create",
            "spawn_subagent",
            "team_create",
            "team_send",
            "worktree_create",
        }.issubset(names)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_close_cancels_background_manager(tmp_path: Path) -> None:
    from litecoder.agent.runtime import AgentRuntime
    from litecoder.context.session.store import SQLiteSessionStore
    from litecoder.paths import AppPaths

    paths = AppPaths(
        user_dir=tmp_path / ".litecoder",
        sessions_db=tmp_path / ".litecoder" / "sessions.db",
        project_id="project",
        project_dir=tmp_path / ".litecoder" / "projects" / "project",
        workspace_id="workspace",
        workspace_root=tmp_path,
    )
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    manager = BackgroundManager(id_factory=lambda: "bg-1")
    handle = await manager.start(
        asyncio.Event().wait(),
        {"agent_session_id": "session-a"},
    )
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda provider, model, turn: pytest.fail("not used"),
        background_manager=manager,
    )

    await runtime.close()

    assert manager.status(handle.id).status is BackgroundStatus.CANCELLED
@pytest.mark.asyncio
async def test_tool_result_metadata_cannot_reassign_background_owner() -> None:
    manager = BackgroundManager(id_factory=lambda: "bg-1")
    handle = await manager.start(
        asyncio.sleep(
            0,
            result=ToolResult(
                "nested-call",
                "success",
                "finished",
                {"agent_session_id": "session-b", "detail": "ok"},
            ),
        ),
        {"agent_session_id": "session-a", "tool_name": "read_file"},
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    session_a = await manager.drain_notifications("session-a")
    session_b = await manager.drain_notifications("session-b")
    state = manager.status(handle.id)

    assert [item.background_id for item in session_a] == ["bg-1"]
    assert session_b == []
    assert state.metadata["agent_session_id"] == "session-a"
    assert state.metadata["detail"] == "ok"
    await manager.close()


@pytest.mark.asyncio
async def test_shutdown_is_bounded_for_cancellation_suppressing_tasks() -> None:
    manager = BackgroundManager(id_factory=lambda: "bg-1", close_timeout=0.02)
    release = asyncio.Event()
    stubborn_tasks: list[asyncio.Task[object]] = []

    async def stubborn() -> None:
        current = asyncio.current_task()
        assert current is not None
        stubborn_tasks.append(current)
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue

    handle = await manager.start(stubborn(), {"agent_session_id": "session-a"})
    close_task = asyncio.create_task(manager.close())
    done, _ = await asyncio.wait({close_task}, timeout=0.2)
    finished_within_deadline = close_task in done
    release.set()
    if not close_task.done():
        await asyncio.wait_for(close_task, timeout=0.2)
    if stubborn_tasks:
        await asyncio.wait_for(stubborn_tasks[0], timeout=0.2)

    assert finished_within_deadline
    assert manager.status(handle.id).status is BackgroundStatus.CANCELLED


@pytest.mark.asyncio
async def test_unknown_background_ids_are_safe_tool_failures(
    tmp_path: Path,
) -> None:
    manager = BackgroundManager(id_factory=lambda: "bg-1")
    registry = ToolRegistry()

    async def runner(
        tool_name: str,
        arguments: dict[str, object],
        context: ToolContext,
    ) -> ToolResult:
        del tool_name, arguments, context
        return ToolResult("nested-call", "success", "unused")

    register_background_tools(registry, manager, runner)
    context = ToolContext("session-a", "workspace", tmp_path)

    with pytest.raises(ToolFailure, match="not found"):
        await registry.require("background_status").execute(
            ToolCall(
                "call-1",
                "background_status",
                {"background_id": "missing"},
            ),
            context,
        )
    with pytest.raises(ToolFailure, match="not found"):
        await registry.require("background_cancel").execute(
            ToolCall(
                "call-2",
                "background_cancel",
                {"background_id": "missing"},
            ),
            context,
        )
