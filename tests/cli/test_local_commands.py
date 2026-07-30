from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.agent.result import AgentResult

from litecoder.common.locks import NamedFileLock, ResourceLockUnavailable
from litecoder.cli.local_commands import LOCAL_COMMANDS, LocalCommandRouter
from litecoder.context.manual_compaction import ManualCompactionReport
from litecoder.context.session.models import MessageRecord, SessionRecord, SessionStatus
from litecoder.context.session.store import SQLiteSessionStore
from litecoder.memory.models import MemoryEntry
from litecoder.memory.store import MemoryStore
from litecoder.paths import AppPaths
from litecoder.providers.models import Usage
from litecoder.tasks.models import TaskRecord
from litecoder.tasks.store import TaskStore


class RuntimeDouble:
    def __init__(self, paths: AppPaths, store: object) -> None:
        self.paths = paths
        self.store = store
        self.provider_name = "fake"
        self.model = "model-a"
        self.provider_models = {"fake": "model-a", "other": "model-b"}


class ModelRuntime(RuntimeDouble):
    def __init__(self, paths: AppPaths) -> None:
        super().__init__(paths, object())
        self.switches: list[tuple[str, str, str]] = []

    async def switch_provider(
        self,
        session_id: str,
        provider: str,
        model: str | None = None,
    ) -> AgentResult:
        assert model is not None
        self.switches.append((session_id, provider, model))
        self.provider_name = provider
        self.model = model
        return AgentResult("derived", "ready", "provider switched", Usage(0, 0))


def make_paths(tmp_path: Path) -> AppPaths:
    return AppPaths(
        user_dir=tmp_path / ".litecoder",
        sessions_db=tmp_path / ".litecoder" / "sessions.db",
        project_id="project-1",
        project_dir=tmp_path / ".litecoder" / "projects" / "project-1",
        workspace_id="workspace-1",
        workspace_root=tmp_path,
    )


async def open_store(paths: AppPaths) -> SQLiteSessionStore:
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    return store


async def test_local_command_surface_contains_only_real_commands(tmp_path: Path) -> None:
    runtime = RuntimeDouble(make_paths(tmp_path), object())
    router = LocalCommandRouter(runtime)  # type: ignore[arg-type]

    assert router.names() == [
        "/clear", "/compact", "/context", "/exit", "/help",
        "/memory", "/model", "/tasks", "/trace",
    ]
    assert set(LOCAL_COMMANDS) == set(router.names())
    assert "/cost" not in LOCAL_COMMANDS
    assert "/team" not in LOCAL_COMMANDS
    assert not any(name.startswith("_") for name in LOCAL_COMMANDS)


async def test_help_lists_exact_usage_without_placeholder_language(tmp_path: Path) -> None:
    runtime = RuntimeDouble(make_paths(tmp_path), object())
    result = await LocalCommandRouter(runtime).dispatch("/help", session_id=None)  # type: ignore[arg-type]

    assert result.message == (
        "Local commands:\n"
        "  /clear\n"
        "  /compact\n"
        "  /context\n"
        "  /exit\n"
        "  /help\n"
        "  /memory [name]\n"
        "  /model [provider] [model]\n"
        "  /tasks [task-id]\n"
        "  /trace"
    )
    assert "milestone" not in result.message.lower()
    assert "not available" not in result.message.lower()


async def test_model_without_arguments_lists_current_and_configured(
    tmp_path: Path,
) -> None:
    runtime = RuntimeDouble(make_paths(tmp_path), object())
    runtime.model = ""
    runtime.provider_models = {
        "zeta": None,
        "fake": "model-a",
        "other": "",
    }

    result = await LocalCommandRouter(runtime).dispatch(
        "/model", session_id=None
    )  # type: ignore[arg-type]

    assert result.message == (
        "Current: fake (no model configured)\n"
        "Configured:\n"
        "  fake model-a\n"
        "  other (no model configured)\n"
        "  zeta (no model configured)"
    )


async def test_model_switch_uses_configured_model_and_returns_child_session(
    tmp_path: Path,
) -> None:
    runtime = ModelRuntime(make_paths(tmp_path))

    result = await LocalCommandRouter(runtime).dispatch(
        "/model other", session_id="root"
    )  # type: ignore[arg-type]

    assert runtime.switches == [("root", "other", "model-b")]
    assert result.replacement_session_id == "derived"
    assert result.message == "Switched to other model-b; session=derived"


async def test_model_switch_requires_active_session(tmp_path: Path) -> None:
    runtime = ModelRuntime(make_paths(tmp_path))

    result = await LocalCommandRouter(runtime).dispatch(
        "/model other", session_id=None
    )  # type: ignore[arg-type]

    assert result.message == "No active session to switch."
    assert result.replacement_session_id is None
    assert runtime.switches == []


async def test_model_switch_rejects_unknown_provider(tmp_path: Path) -> None:
    runtime = ModelRuntime(make_paths(tmp_path))

    result = await LocalCommandRouter(runtime).dispatch(
        "/model missing", session_id="root"
    )  # type: ignore[arg-type]

    assert result.message == "Unknown provider 'missing'."
    assert result.replacement_session_id is None
    assert runtime.switches == []


async def test_model_switch_rejects_extra_arguments(tmp_path: Path) -> None:
    runtime = ModelRuntime(make_paths(tmp_path))

    result = await LocalCommandRouter(runtime).dispatch(
        "/model fake model-a extra", session_id="root"
    )  # type: ignore[arg-type]

    assert result.message == "Usage: /model [provider] [model]"
    assert result.replacement_session_id is None
    assert runtime.switches == []


@pytest.mark.parametrize("configured_model", [None, ""])
async def test_model_switch_rejects_provider_without_configured_model(
    tmp_path: Path,
    configured_model: str | None,
) -> None:
    runtime = ModelRuntime(make_paths(tmp_path))
    runtime.provider_models["other"] = configured_model

    result = await LocalCommandRouter(runtime).dispatch(
        "/model other", session_id="root"
    )  # type: ignore[arg-type]

    assert result.message == "Provider 'other' has no configured model."
    assert result.replacement_session_id is None
    assert runtime.switches == []


async def test_model_switch_allows_explicit_model_override(tmp_path: Path) -> None:
    runtime = ModelRuntime(make_paths(tmp_path))
    runtime.provider_models["other"] = None

    result = await LocalCommandRouter(runtime).dispatch(
        "/model other custom-model", session_id="root"
    )  # type: ignore[arg-type]

    assert runtime.switches == [("root", "other", "custom-model")]
    assert result.replacement_session_id == "derived"
    assert result.message == "Switched to other custom-model; session=derived"


async def test_memory_lists_index_and_shows_named_entry(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    store = await open_store(paths)
    try:
        memory = MemoryStore(paths.workspace_root / ".memory")
        memory.replace_all([
            MemoryEntry(
                "project-facts",
                "Stable project facts",
                "project",
                "Uses pytest.",
            ),
        ])
        router = LocalCommandRouter(RuntimeDouble(paths, store))  # type: ignore[arg-type]

        listed = await router.dispatch("/memory", session_id=None)
        shown = await router.dispatch("/memory project-facts", session_id=None)

        assert listed.message == (
            "- [project-facts](project-facts.md) - Stable project facts"
        )
        assert shown.message == (
            "---\n"
            "name: project-facts\n"
            "description: Stable project facts\n"
            "type: project\n"
            "---\n\n"
            "Uses pytest."
        )
    finally:
        await store.close()


async def test_memory_treats_an_absent_store_as_empty(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    store = await open_store(paths)
    try:
        router = LocalCommandRouter(RuntimeDouble(paths, store))  # type: ignore[arg-type]

        listed = await router.dispatch("/memory", session_id=None)
        shown = await router.dispatch("/memory missing", session_id=None)

        assert listed.message == "No memory entries."
        assert listed.audit_status == "success"
        assert listed.audit_outcome == "empty"
        assert shown.message == "Unknown memory 'missing'"
        assert shown.audit_status == "rejected"
        assert shown.audit_code == "not_found"
        assert not (paths.workspace_root / ".memory").exists()
    finally:
        await store.close()


async def test_memory_unknown_name_is_a_clear_diagnostic(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    store = await open_store(paths)
    try:
        memory = MemoryStore(paths.workspace_root / ".memory")
        memory.replace_all([MemoryEntry("known", "Known memory", "project", "body")])
        result = await LocalCommandRouter(RuntimeDouble(paths, store)).dispatch(
            "/memory missing", session_id=None
        )  # type: ignore[arg-type]
        assert result.message == "Unknown memory 'missing'"
    finally:
        await store.close()


async def test_memory_lock_contention_is_a_bounded_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_paths(tmp_path)
    MemoryStore(paths.workspace_root / ".memory").replace_all((
        MemoryEntry("known", "Known memory", "project", "body"),
    ))

    class FailingLock:
        def __enter__(self) -> object:
            raise ResourceLockUnavailable("memory", paths.user_dir / "memory.lock")

        def __exit__(self, *exc_info: object) -> None:
            return None

    def failing_memory(cls: type[NamedFileLock], project_id: str, lock_dir: Path):
        return FailingLock()

    monkeypatch.setattr(NamedFileLock, "memory", classmethod(failing_memory))
    runtime = RuntimeDouble(paths, object())
    result = await LocalCommandRouter(runtime).dispatch(
        "/memory", session_id=None
    )  # type: ignore[arg-type]

    assert result.message == "Memory is unavailable"


async def test_tasks_lists_and_shows_real_project_records(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    store = await open_store(paths)
    try:
        TaskStore(paths.project_dir / "tasks").replace_many([
            TaskRecord("dependency", "Dependency", "First"),
            TaskRecord("blocked", "Blocked", "Waits", dependencies=["dependency"]),
        ])
        router = LocalCommandRouter(RuntimeDouble(paths, store))  # type: ignore[arg-type]

        listed = await router.dispatch("/tasks", session_id=None)
        shown = await router.dispatch("/tasks blocked", session_id=None)

        assert listed.message == "blocked\tblocked\tBlocked\ndependency\tpending\tDependency"
        assert shown.message == (
            "id: blocked\n"
            "subject: Blocked\n"
            "description: Waits\n"
            "status: blocked\n"
            "owner: \n"
            "dependencies: dependency"
        )
    finally:
        await store.close()


async def test_tasks_unknown_id_is_a_clear_diagnostic(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    store = await open_store(paths)
    try:
        TaskStore(paths.project_dir / "tasks").replace_many([
            TaskRecord("known", "Known", "Task"),
        ])
        result = await LocalCommandRouter(RuntimeDouble(paths, store)).dispatch(
            "/tasks missing", session_id=None
        )  # type: ignore[arg-type]
        assert result.message == "Unknown task 'missing'"
    finally:
        await store.close()


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/memory one two", "Usage: /memory [name]"),
        ("/tasks one two", "Usage: /tasks [task-id]"),
    ],
)
async def test_memory_and_tasks_reject_surplus_arguments(
    tmp_path: Path,
    command: str,
    expected: str,
) -> None:
    paths = make_paths(tmp_path)
    store = await open_store(paths)
    try:
        result = await LocalCommandRouter(RuntimeDouble(paths, store)).dispatch(
            command, session_id=None
        )  # type: ignore[arg-type]
        assert result.message == expected
    finally:
        await store.close()


async def test_context_and_trace_report_real_session_state(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    store = await open_store(paths)
    try:
        await store.create_session(
            SessionRecord.new(
                "root",
                paths.project_id,
                paths.workspace_id,
                "fake",
                "root-model",
                workspace_path=str(paths.workspace_root),
                status=SessionStatus.IDLE,
            )
        )
        await store.create_session(
            SessionRecord.new(
                "child",
                paths.project_id,
                paths.workspace_id,
                "fake",
                "model-a",
                workspace_path=str(paths.workspace_root),
                parent_session_id="root",
                status=SessionStatus.IDLE,
            )
        )
        await store.append_message(
            MessageRecord("child", "user", [{"type": "text", "text": "hello"}])
        )
        trace = paths.project_dir / "traces" / "root.jsonl"
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_text("{}\n{}\n", encoding="utf-8")
        router = LocalCommandRouter(RuntimeDouble(paths, store))  # type: ignore[arg-type]

        context = await router.dispatch("/context", session_id="child")
        traced = await router.dispatch("/trace", session_id="child")

        assert "session=child" in context.message
        assert "provider=fake" in context.message
        assert "model=model-a" in context.message
        assert "messages=1" in context.message
        assert "context_tokens=" in context.message
        assert str(trace.resolve()) in traced.message
        assert "status=present" in traced.message
        assert "events=2" in traced.message
    finally:
        await store.close()


async def test_trace_missing_reports_real_path_and_missing_status(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    store = await open_store(paths)
    try:
        await store.create_session(
            SessionRecord.new(
                "root",
                paths.project_id,
                paths.workspace_id,
                "fake",
                "model-a",
                workspace_path=str(paths.workspace_root),
                status=SessionStatus.IDLE,
            )
        )
        result = await LocalCommandRouter(RuntimeDouble(paths, store)).dispatch(
            "/trace", session_id="root"
        )  # type: ignore[arg-type]
        trace = paths.project_dir / "traces" / "root.jsonl"
        assert result.message.splitlines() == [
            f"Trace: path={trace.resolve()} status=missing",
            f"Command audit: path={paths.command_audit_path} "
            "status=present events=2",
        ]
    finally:
        await store.close()


async def test_trace_rejects_stored_root_id_outside_trace_directory(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    store = await open_store(paths)
    try:
        root_id = "../../outside"
        await store.create_session(
            SessionRecord.new(
                root_id,
                paths.project_id,
                paths.workspace_id,
                "fake",
                "model-a",
                workspace_path=str(paths.workspace_root),
                status=SessionStatus.IDLE,
            )
        )
        external_trace = (
            paths.project_dir / "traces" / f"{root_id}.jsonl"
        ).resolve()
        external_trace.parent.mkdir(parents=True, exist_ok=True)
        external_trace.write_text("{}\n{}\n", encoding="utf-8")

        result = await LocalCommandRouter(RuntimeDouble(paths, store)).dispatch(
            "/trace", session_id=root_id
        )  # type: ignore[arg-type]

        assert result.message.splitlines() == [
            "Trace is unavailable",
            f"Command audit: path={paths.command_audit_path} "
            "status=present events=2",
        ]
        assert str(external_trace) not in result.message
    finally:
        await store.close()


class CompactRuntime(RuntimeDouble):
    def __init__(self, paths: AppPaths, store: object) -> None:
        super().__init__(paths, store)
        self.compact_calls: list[str] = []

    async def compact_session(self, session_id: str) -> ManualCompactionReport:
        self.compact_calls.append(session_id)
        return ManualCompactionReport(900, 300, True)


async def test_compact_reports_no_session_and_real_report(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    store = await open_store(paths)
    try:
        runtime = CompactRuntime(paths, store)
        router = LocalCommandRouter(runtime)  # type: ignore[arg-type]

        missing = await router.dispatch("/compact", session_id=None)
        compacted = await router.dispatch("/compact", session_id="root")

        assert missing.message == "No active session to compact."
        assert compacted.message == (
            "Context compacted: before=900 after=300 saved=600 summary=yes"
        )
        assert runtime.compact_calls == ["root"]
    finally:
        await store.close()


class NoReductionRuntime(RuntimeDouble):
    async def compact_session(self, session_id: str) -> ManualCompactionReport:
        return ManualCompactionReport(100, 100, False)


async def test_compact_reports_no_reduction_without_success_claim(
    tmp_path: Path,
) -> None:
    runtime = NoReductionRuntime(make_paths(tmp_path), object())
    result = await LocalCommandRouter(runtime).dispatch(
        "/compact", session_id="root"
    )  # type: ignore[arg-type]

    assert result.message == (
        "Context not compacted: before=100 after=100 "
        "saved=0 summary=no reason=no_reduction"
    )
    assert "Context compacted" not in result.message


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/clear extra", "Usage: /clear"),
        ("/exit extra", "Usage: /exit"),
        ("/compact extra", "Usage: /compact"),
        ("/context extra", "Usage: /context"),
        ("/trace extra", "Usage: /trace"),
    ],
)
async def test_commands_reject_arguments(
    tmp_path: Path,
    command: str,
    expected: str,
) -> None:
    runtime = RuntimeDouble(make_paths(tmp_path), object())
    result = await LocalCommandRouter(runtime).dispatch(command, session_id=None)  # type: ignore[arg-type]
    assert result.message == expected
    assert result.clear_requested is False
    assert result.exit_requested is False


async def test_clear_and_exit_return_their_flags(tmp_path: Path) -> None:
    runtime = RuntimeDouble(make_paths(tmp_path), object())
    router = LocalCommandRouter(runtime)  # type: ignore[arg-type]

    clear = await router.dispatch("/clear", session_id="session")
    exit_result = await router.dispatch("/exit", session_id="session")

    assert clear.handled is True
    assert clear.clear_requested is True
    assert clear.exit_requested is False
    assert exit_result.handled is True
    assert exit_result.exit_requested is True
    assert exit_result.clear_requested is False


async def test_non_slash_input_is_forwarded(tmp_path: Path) -> None:
    runtime = RuntimeDouble(make_paths(tmp_path), object())
    result = await LocalCommandRouter(runtime).dispatch("hello model", session_id=None)  # type: ignore[arg-type]

    assert result.handled is False
    assert result.forward_to_model is True


async def test_unknown_slash_input_is_diagnostic_and_not_forwarded(tmp_path: Path) -> None:
    runtime = RuntimeDouble(make_paths(tmp_path), object())
    result = await LocalCommandRouter(runtime).dispatch("/unknown arg", session_id=None)  # type: ignore[arg-type]

    assert result.handled is True
    assert result.forward_to_model is False
    assert result.message == "Unknown local command: /unknown"


async def test_data_commands_require_an_active_session_when_needed(tmp_path: Path) -> None:
    runtime = RuntimeDouble(make_paths(tmp_path), object())
    router = LocalCommandRouter(runtime)  # type: ignore[arg-type]

    context = await router.dispatch("/context", session_id=None)
    trace = await router.dispatch("/trace", session_id=None)

    assert context.message == "No active session to inspect."
    assert trace.message.splitlines() == [
        "No active session to trace.",
        f"Command audit: path={runtime.paths.command_audit_path} "
        "status=present events=4",
    ]
