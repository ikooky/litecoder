from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from litecoder.agent.runtime import AgentRuntime, InvalidTaskGraphMode
from litecoder.common.locks import NamedFileLock, ResourceLockUnavailable
from litecoder.common.trace import SecretRedactor
from litecoder.context.session.models import SessionRecord, SessionStatus
from litecoder.context.session.store import SQLiteSessionStore
from litecoder.paths import AppPaths
from litecoder.tasks.manager import TaskManager
from litecoder.tasks.models import TaskRecord, TaskStatus
from litecoder.tasks.planning import TaskCycleError
from litecoder.tasks.store import TaskStore


def paths_for(tmp_path: Path) -> AppPaths:
    return AppPaths(
        user_dir=tmp_path / ".litecoder",
        sessions_db=tmp_path / ".litecoder" / "sessions.db",
        project_id="project-1",
        project_dir=tmp_path / ".litecoder" / "projects" / "project-1",
        workspace_id="workspace-1",
        workspace_root=tmp_path,
    )


def task(
    task_id: str,
    status: TaskStatus,
    dependencies: list[str] | None = None,
) -> TaskRecord:
    return TaskRecord(
        id=task_id,
        subject=task_id,
        description=task_id,
        status=status,
        dependencies=[] if dependencies is None else dependencies,
        owner_agent_id="agent-a" if status is TaskStatus.IN_PROGRESS else None,
    )


@pytest.mark.asyncio
async def test_runtime_start_marks_stale_sessions_and_tasks_interrupted(
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
    ))
    task_store = TaskStore(paths.project_dir / "tasks")
    task_store.replace_many([
        task("running", TaskStatus.IN_PROGRESS),
        task("pending", TaskStatus.PENDING),
    ])
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
        trace_redactor=SecretRedactor.with_values(()),
        startup_lock=NamedFileLock.startup(paths.project_id, paths.user_dir),
        session_lock_factory=lambda root_id: NamedFileLock.session_tree(
            root_id, paths.user_dir
        ),
        task_manager=TaskManager(task_store),
    )

    await runtime.start()

    assert (
        await store.load_context("session-1")
    ).session.status is SessionStatus.INCOMPLETE
    assert task_store.read("running").status is TaskStatus.INTERRUPTED
    assert task_store.read("pending").status is TaskStatus.PENDING
    await runtime.close()


@pytest.mark.asyncio
async def test_concurrent_runtime_start_calls_serialize_startup_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    recovery_started = asyncio.Event()
    release_recovery = asyncio.Event()
    recovery_calls = 0

    async def blocked_recovery(
        project_id: str,
        exclude_session_ids: tuple[str, ...] = (),
        target_session_ids: tuple[str, ...] | None = None,
    ) -> list[str]:
        nonlocal recovery_calls
        recovery_calls += 1
        if recovery_calls == 1:
            recovery_started.set()
            await release_recovery.wait()
        return []

    monkeypatch.setattr(store, "recover_active_sessions", blocked_recovery)
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
        startup_lock=NamedFileLock.startup(paths.project_id, paths.user_dir),
    )
    first = asyncio.create_task(runtime.start())
    await recovery_started.wait()
    second = asyncio.create_task(runtime.start())

    try:
        await asyncio.sleep(0.05)

        assert not second.done()
        assert recovery_calls == 1
        with pytest.raises(ResourceLockUnavailable):
            with NamedFileLock(
                f"startup-{paths.project_id}",
                paths.user_dir,
                fail_when_locked=True,
            ):
                pass
    finally:
        release_recovery.set()
        await asyncio.gather(first, second)
        await runtime.close()


@pytest.mark.asyncio
async def test_resume_waits_for_in_progress_startup_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from litecoder.agent.result import AgentResult
    from litecoder.providers.models import Usage

    paths = paths_for(tmp_path)
    turn_store = SQLiteSessionStore(paths.sessions_db)
    await turn_store.open()
    await turn_store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
        status=SessionStatus.IDLE,
    ))
    task_store = TaskStore(paths.project_dir / "tasks")
    task_store.replace_many([task("work", TaskStatus.PENDING)])
    turn_task_manager = TaskManager(task_store)
    entered = asyncio.Event()

    class ClaimingLoop:
        async def run_turn(self, session_id: str, prompt: str) -> AgentResult:
            claimed = await turn_task_manager.claim("work", "agent-a")
            assert claimed.status is TaskStatus.IN_PROGRESS
            entered.set()
            await turn_store.mark_status(session_id, SessionStatus.IDLE)
            return AgentResult(session_id, "completed", "done", Usage(0, 0))

    turn_runtime = AgentRuntime(
        store=turn_store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: ClaimingLoop(),
        startup_lock=NamedFileLock.startup(paths.project_id, paths.user_dir),
        session_lock_factory=lambda root_id: NamedFileLock.session_tree(
            root_id, paths.user_dir
        ),
    )
    await turn_runtime.start()

    recovery_store = SQLiteSessionStore(paths.sessions_db)
    await recovery_store.open()
    recovery_started = asyncio.Event()
    release_recovery = asyncio.Event()

    async def blocked_recovery(
        project_id: str,
        exclude_session_ids: tuple[str, ...] = (),
        target_session_ids: tuple[str, ...] | None = None,
    ) -> list[str]:
        recovery_started.set()
        await release_recovery.wait()
        return []

    monkeypatch.setattr(
        recovery_store, "recover_active_sessions", blocked_recovery
    )
    recovery_runtime = AgentRuntime(
        store=recovery_store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
        startup_lock=NamedFileLock.startup(paths.project_id, paths.user_dir),
        session_lock_factory=lambda root_id: NamedFileLock.session_tree(
            root_id, paths.user_dir
        ),
        task_manager=TaskManager(task_store),
    )
    starting = asyncio.create_task(recovery_runtime.start())
    await recovery_started.wait()
    resumed = asyncio.create_task(turn_runtime.resume("session-1", "continue"))

    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(entered.wait(), timeout=0.1)
        assert task_store.read("work").status is TaskStatus.PENDING

        release_recovery.set()
        await starting
        await resumed

        assert entered.is_set()
        assert task_store.read("work").status is TaskStatus.IN_PROGRESS
    finally:
        release_recovery.set()
        await asyncio.gather(starting, resumed, return_exceptions=True)
        await recovery_runtime.close()
        await turn_runtime.close()


@pytest.mark.asyncio
async def test_runtime_start_does_not_recover_locked_session_tree(
    tmp_path: Path,
) -> None:
    from litecoder.agent.result import AgentResult
    from litecoder.providers.models import Usage

    paths = paths_for(tmp_path)
    first_store = SQLiteSessionStore(paths.sessions_db)
    await first_store.open()
    for session_id in ("session-1", "session-2"):
        await first_store.create_session(SessionRecord.new(
            session_id,
            paths.project_id,
            paths.workspace_id,
            "fake",
            "model",
            workspace_path=str(tmp_path),
        ))
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingLoop:
        async def run_turn(self, session_id: str, prompt: str) -> AgentResult:
            entered.set()
            await release.wait()
            return AgentResult(session_id, "completed", "done", Usage(0, 0))

    first_runtime = AgentRuntime(
        store=first_store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: BlockingLoop(),
        session_lock_factory=lambda root_id: NamedFileLock.session_tree(
            root_id, paths.user_dir
        ),
    )
    running = asyncio.create_task(first_runtime.resume("session-1", "first"))
    await entered.wait()

    second_store = SQLiteSessionStore(paths.sessions_db)
    await second_store.open()
    second_runtime = AgentRuntime(
        store=second_store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
        startup_lock=NamedFileLock.startup(paths.project_id, paths.user_dir),
        session_lock_factory=lambda root_id: NamedFileLock.session_tree(
            root_id, paths.user_dir
        ),
    )

    try:
        await second_runtime.start()

        assert (
            await second_store.load_context("session-1")
        ).session.status is SessionStatus.ACTIVE
        assert (
            await second_store.load_context("session-2")
        ).session.status is SessionStatus.INCOMPLETE
    finally:
        release.set()
        await running
        await second_runtime.close()
        await first_runtime.close()


@pytest.mark.asyncio
async def test_runtime_start_preserves_live_tasks_when_session_tree_is_locked(
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
    ))
    task_store = TaskStore(paths.project_dir / "tasks")
    task_store.replace_many([
        task("running", TaskStatus.IN_PROGRESS),
        task("pending", TaskStatus.PENDING),
    ])
    live_lock = NamedFileLock.session_tree("session-1", paths.user_dir)
    live_lock.acquire()
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
        startup_lock=NamedFileLock.startup(paths.project_id, paths.user_dir),
        session_lock_factory=lambda root_id: NamedFileLock.session_tree(
            root_id, paths.user_dir
        ),
        task_manager=TaskManager(task_store),
    )

    try:
        await runtime.start()

        assert (
            await store.load_context("session-1")
        ).session.status is SessionStatus.ACTIVE
        assert task_store.read("running").status is TaskStatus.IN_PROGRESS
        assert task_store.read("pending").status is TaskStatus.PENDING
    finally:
        live_lock.release()
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_start_preserves_live_tasks_when_idle_root_is_locked(
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
        status=SessionStatus.IDLE,
    ))
    task_store = TaskStore(paths.project_dir / "tasks")
    task_store.replace_many([
        task("running", TaskStatus.IN_PROGRESS),
        task("pending", TaskStatus.PENDING),
    ])
    live_lock = NamedFileLock.session_tree("session-1", paths.user_dir)
    live_lock.acquire()
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
        startup_lock=NamedFileLock.startup(paths.project_id, paths.user_dir),
        session_lock_factory=lambda root_id: NamedFileLock.session_tree(
            root_id, paths.user_dir
        ),
        task_manager=TaskManager(task_store),
    )

    try:
        await runtime.start()

        assert (
            await store.load_context("session-1")
        ).session.status is SessionStatus.IDLE
        assert task_store.read("running").status is TaskStatus.IN_PROGRESS
        assert task_store.read("pending").status is TaskStatus.PENDING
    finally:
        live_lock.release()
        await runtime.close()

@pytest.mark.asyncio
async def test_runtime_start_validates_tasks_when_session_tree_is_locked(
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
    ))
    task_store = TaskStore(paths.project_dir / "tasks")
    task_store.replace_many([
        task("A", TaskStatus.IN_PROGRESS),
        task("B", TaskStatus.PENDING),
    ])
    a = task_store.read("A")
    b = task_store.read("B")
    a.dependencies = ["B"]
    b.dependencies = ["A"]
    task_store.write(a)
    task_store.write(b)
    live_lock = NamedFileLock.session_tree("session-1", paths.user_dir)
    live_lock.acquire()
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
        startup_lock=NamedFileLock.startup(paths.project_id, paths.user_dir),
        session_lock_factory=lambda root_id: NamedFileLock.session_tree(
            root_id, paths.user_dir
        ),
        task_manager=TaskManager(task_store),
    )

    try:
        await runtime.start()

        assert isinstance(runtime.invalid_task_graph, TaskCycleError)
        assert task_store.read("A").status is TaskStatus.IN_PROGRESS
        assert task_store.read("B").status is TaskStatus.PENDING
    finally:
        live_lock.release()
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_start_retains_probed_root_lock_during_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
    ))
    held_roots: set[str] = set()
    original_recover = store.recover_active_sessions

    class ObservedLock:
        def __init__(self, root_id: str) -> None:
            self.root_id = root_id

        @asynccontextmanager
        async def acquired_async(self):
            held_roots.add(self.root_id)
            try:
                yield self
            finally:
                held_roots.remove(self.root_id)

    async def recover_while_observing_lock(
        project_id: str,
        exclude_session_ids: tuple[str, ...] = (),
        target_session_ids: tuple[str, ...] | None = None,
    ) -> list[str]:
        assert held_roots == {"session-1"}
        await store.create_session(SessionRecord.new(
            "session-2",
            paths.project_id,
            paths.workspace_id,
            "fake",
            "model",
            workspace_path=str(tmp_path),
        ))
        live_lock = NamedFileLock.session_tree("session-2", paths.user_dir)
        async with live_lock.acquired_async():
            if target_session_ids is None:
                return await original_recover(project_id, exclude_session_ids)
            return await original_recover(
                project_id,
                exclude_session_ids=exclude_session_ids,
                target_session_ids=target_session_ids,
            )

    monkeypatch.setattr(
        store, "recover_active_sessions", recover_while_observing_lock
    )
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
        startup_lock=NamedFileLock.startup(paths.project_id, paths.user_dir),
        session_lock_factory=ObservedLock,
    )

    try:
        await runtime.start()

        assert held_roots == set()
        assert (
            await store.load_context("session-1")
        ).session.status is SessionStatus.INCOMPLETE
        assert (
            await store.load_context("session-2")
        ).session.status is SessionStatus.ACTIVE
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_invalid_task_graph_blocks_execution_without_mutating_tasks(
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
    ))
    task_store = TaskStore(paths.project_dir / "tasks")
    task_store.replace_many([
        task("A", TaskStatus.IN_PROGRESS),
        task("B", TaskStatus.PENDING),
    ])
    a = task_store.read("A")
    b = task_store.read("B")
    a.dependencies = ["B"]
    b.dependencies = ["A"]
    task_store.write(a)
    task_store.write(b)
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
        trace_redactor=SecretRedactor.with_values(()),
        startup_lock=NamedFileLock.startup(paths.project_id, paths.user_dir),
        task_manager=TaskManager(task_store),
    )

    await runtime.start()

    assert isinstance(runtime.invalid_task_graph, TaskCycleError)
    assert task_store.read("A").status is TaskStatus.IN_PROGRESS
    with pytest.raises(InvalidTaskGraphMode):
        await runtime.run("must be blocked")
    await runtime.close()


@pytest.mark.asyncio
async def test_run_persists_new_session_idle_before_tree_lock_acquisition(
    tmp_path: Path,
) -> None:
    from litecoder.agent.result import AgentResult
    from litecoder.providers.models import Usage

    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    acquiring = asyncio.Event()
    release = asyncio.Event()

    class BlockingLock:
        @asynccontextmanager
        async def acquired_async(self):
            acquiring.set()
            await release.wait()
            yield self

    class CompletingLoop:
        async def run_turn(self, session_id: str, prompt: str) -> AgentResult:
            return AgentResult(session_id, "completed", "done", Usage(0, 0))

    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: CompletingLoop(),
        id_factory=lambda: "session-1",
        session_lock_factory=lambda _root_id: BlockingLock(),
    )
    running = asyncio.create_task(runtime.run("start"))

    try:
        await acquiring.wait()

        assert (
            await store.load_context("session-1")
        ).session.status is SessionStatus.IDLE
    finally:
        release.set()
        await running
        await runtime.close()


@pytest.mark.asyncio
async def test_same_root_session_tree_fails_fast_when_already_running(
    tmp_path: Path,
) -> None:
    from litecoder.agent.result import AgentResult
    from litecoder.common.locks import NamedFileLock, SessionAlreadyActive
    from litecoder.providers.models import Usage

    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(
        SessionRecord.new(
            "session-1",
            paths.project_id,
            paths.workspace_id,
            "fake",
            "model",
            workspace_path=str(tmp_path),
        )
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingLoop:
        async def run_turn(self, session_id: str, prompt: str) -> AgentResult:
            entered.set()
            await release.wait()
            return AgentResult(session_id, "completed", "done", Usage(0, 0))

    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: BlockingLoop(),
        startup_lock=NamedFileLock.startup(paths.project_id, paths.user_dir),
        session_lock_factory=lambda root_id: NamedFileLock.session_tree(
            root_id, paths.user_dir
        ),
    )
    await runtime.start()
    first = asyncio.create_task(runtime.resume("session-1", "first"))
    await entered.wait()

    with pytest.raises(SessionAlreadyActive):
        await runtime.resume("session-1", "second")

    release.set()
    await first
    await runtime.close()


@pytest.mark.asyncio
async def test_independent_root_sessions_can_run_concurrently(
    tmp_path: Path,
) -> None:
    from litecoder.agent.result import AgentResult
    from litecoder.common.locks import NamedFileLock
    from litecoder.providers.models import Usage

    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    for session_id in ("session-1", "session-2"):
        await store.create_session(
            SessionRecord.new(
                session_id,
                paths.project_id,
                paths.workspace_id,
                "fake",
                "model",
                workspace_path=str(tmp_path),
            )
        )
    entered: asyncio.Queue[str] = asyncio.Queue()
    release = asyncio.Event()

    class BlockingLoop:
        async def run_turn(self, session_id: str, prompt: str) -> AgentResult:
            entered.put_nowait(session_id)
            await release.wait()
            return AgentResult(session_id, "completed", "done", Usage(0, 0))

    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: BlockingLoop(),
        startup_lock=NamedFileLock.startup(paths.project_id, paths.user_dir),
        session_lock_factory=lambda root_id: NamedFileLock.session_tree(
            root_id, paths.user_dir
        ),
    )
    await runtime.start()
    first = asyncio.create_task(runtime.resume("session-1", "first"))
    second = asyncio.create_task(runtime.resume("session-2", "second"))
    seen = {
        await asyncio.wait_for(entered.get(), timeout=1),
        await asyncio.wait_for(entered.get(), timeout=1),
    }

    assert seen == {"session-1", "session-2"}

    release.set()
    await asyncio.gather(first, second)
    await runtime.close()


@pytest.mark.asyncio
async def test_root_resolution_failure_marks_resumed_session_failed(
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    for session_id, parent_session_id in (
        ("session-1", None),
        ("session-2", "session-1"),
    ):
        await store.create_session(SessionRecord.new(
            session_id,
            paths.project_id,
            paths.workspace_id,
            "fake",
            "model",
            workspace_path=str(tmp_path),
            parent_session_id=parent_session_id,
        ))
    connection = store.connection
    assert connection is not None
    await connection.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
        ("session-2", "session-1"),
    )
    await connection.commit()
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
    )

    try:
        with pytest.raises(RuntimeError, match="session parent cycle detected"):
            await runtime.resume("session-1", "continue")

        assert (
            await store.load_context("session-1")
        ).session.status is SessionStatus.FAILED
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_cancellation_during_root_resolution_leaves_status_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
        status=SessionStatus.IDLE,
    ))
    resolving = asyncio.Event()
    release = asyncio.Event()
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
    )

    async def blocked_root_resolution(
        _session: SessionRecord,
    ) -> str:
        resolving.set()
        await release.wait()
        raise AssertionError("root resolution must be cancelled")

    monkeypatch.setattr(runtime, "_root_session_id", blocked_root_resolution)
    resumed = asyncio.create_task(runtime.resume("session-1", "continue"))
    await resolving.wait()
    resumed.cancel()

    try:
        with pytest.raises(asyncio.CancelledError):
            await resumed

        assert (
            await store.load_context("session-1")
        ).session.status is SessionStatus.IDLE
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_cancellation_during_session_lock_acquisition_leaves_status_unchanged(
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
    ))
    acquiring = asyncio.Event()
    release = asyncio.Event()

    class BlockingLock:
        @asynccontextmanager
        async def acquired_async(self):
            acquiring.set()
            await release.wait()
            yield self

    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
        session_lock_factory=lambda _root_id: BlockingLock(),
    )
    resumed = asyncio.create_task(runtime.resume("session-1", "continue"))
    await acquiring.wait()
    resumed.cancel()

    try:
        with pytest.raises(asyncio.CancelledError):
            await resumed

        assert (
            await store.load_context("session-1")
        ).session.status is SessionStatus.ACTIVE
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_cancellation_during_owned_turn_marks_session_cancelled(
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
    ))
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingLoop:
        async def run_turn(self, session_id: str, prompt: str) -> None:
            entered.set()
            await release.wait()

    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: BlockingLoop(),
        session_lock_factory=lambda root_id: NamedFileLock.session_tree(
            root_id, paths.user_dir
        ),
    )
    resumed = asyncio.create_task(runtime.resume("session-1", "continue"))
    await entered.wait()
    resumed.cancel()

    try:
        with pytest.raises(asyncio.CancelledError):
            await resumed

        assert (
            await store.load_context("session-1")
        ).session.status is SessionStatus.CANCELLED
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_cancellation_during_tree_lock_release_preserves_idle_status(
    tmp_path: Path,
) -> None:
    from litecoder.agent.result import AgentResult
    from litecoder.providers.models import Usage

    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
    ))
    releasing = asyncio.Event()
    release = asyncio.Event()

    class BlockingReleaseLock:
        @asynccontextmanager
        async def acquired_async(self):
            try:
                yield self
            finally:
                releasing.set()
                await release.wait()

    class CompletingLoop:
        async def run_turn(self, session_id: str, prompt: str) -> AgentResult:
            await store.mark_status(session_id, SessionStatus.IDLE)
            return AgentResult(session_id, "completed", "done", Usage(0, 0))

    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: CompletingLoop(),
        session_lock_factory=lambda _root_id: BlockingReleaseLock(),
    )
    resumed = asyncio.create_task(runtime.resume("session-1", "continue"))
    await releasing.wait()
    resumed.cancel()

    try:
        with pytest.raises(asyncio.CancelledError):
            await resumed

        assert (
            await store.load_context("session-1")
        ).session.status is SessionStatus.IDLE
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_cancelled_busy_session_lock_leaves_live_session_active(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
    ))
    holder = NamedFileLock.session_tree("session-1", paths.user_dir)
    contender = NamedFileLock.session_tree("session-1", paths.user_dir)
    acquisition_started = threading.Event()
    continue_acquisition = threading.Event()
    original_acquire = contender.acquire

    def acquire_after_signal() -> NamedFileLock:
        acquisition_started.set()
        continue_acquisition.wait()
        return original_acquire()

    monkeypatch.setattr(contender, "acquire", acquire_after_signal)
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
        session_lock_factory=lambda _root_id: contender,
    )
    holder.acquire()
    resumed = asyncio.create_task(runtime.resume("session-1", "continue"))

    try:
        await asyncio.wait_for(
            asyncio.to_thread(acquisition_started.wait), timeout=1
        )
        resumed.cancel()
        continue_acquisition.set()

        with pytest.raises(asyncio.CancelledError):
            await resumed

        assert (
            await store.load_context("session-1")
        ).session.status is SessionStatus.ACTIVE
    finally:
        continue_acquisition.set()
        holder.release()
        contender.release()
        await runtime.close()

@pytest.mark.asyncio
async def test_runtime_persists_cancelled_status_before_tree_lock_release(
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
    ))

    class InspectingLock:
        status_at_release: SessionStatus | None = None

        @asynccontextmanager
        async def acquired_async(self):
            try:
                yield self
            finally:
                self.status_at_release = (
                    await store.load_context("session-1")
                ).session.status

    class BlockingLoop:
        def __init__(self) -> None:
            self.entered = asyncio.Event()

        async def run_turn(self, session_id: str, prompt: str) -> None:
            self.entered.set()
            await asyncio.Event().wait()

    lock = InspectingLock()
    loop = BlockingLoop()
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: loop,
        session_lock_factory=lambda _root_id: lock,  # type: ignore[arg-type]
    )
    resumed = asyncio.create_task(runtime.resume("session-1", "continue"))
    await loop.entered.wait()
    resumed.cancel()

    try:
        with pytest.raises(asyncio.CancelledError):
            await resumed

        assert lock.status_at_release is SessionStatus.CANCELLED
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_persists_failed_status_before_tree_lock_release(
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
    ))

    class InspectingLock:
        status_at_release: SessionStatus | None = None

        @asynccontextmanager
        async def acquired_async(self):
            try:
                yield self
            finally:
                self.status_at_release = (
                    await store.load_context("session-1")
                ).session.status

    class FailingLoop:
        async def run_turn(self, session_id: str, prompt: str) -> None:
            raise RuntimeError("turn failed")

    lock = InspectingLock()
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: FailingLoop(),
        session_lock_factory=lambda _root_id: lock,  # type: ignore[arg-type]
    )

    try:
        with pytest.raises(RuntimeError, match="turn failed"):
            await runtime.resume("session-1", "continue")

        assert lock.status_at_release is SessionStatus.FAILED
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_activation_cancellation_persists_status_before_tree_lock_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
        status=SessionStatus.IDLE,
    ))
    activation_started = asyncio.Event()
    original_mark_status = store.mark_status

    async def block_activation(
        session_id: str, status: SessionStatus
    ) -> None:
        if status is SessionStatus.ACTIVE:
            activation_started.set()
            await asyncio.Event().wait()
        await original_mark_status(session_id, status)

    monkeypatch.setattr(store, "mark_status", block_activation)

    class InspectingLock:
        status_at_release: SessionStatus | None = None

        @asynccontextmanager
        async def acquired_async(self):
            try:
                yield self
            finally:
                self.status_at_release = (
                    await store.load_context("session-1")
                ).session.status

    lock = InspectingLock()
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
        session_lock_factory=lambda _root_id: lock,  # type: ignore[arg-type]
    )
    resumed = asyncio.create_task(runtime.resume("session-1", "continue"))
    await activation_started.wait()
    resumed.cancel()

    try:
        with pytest.raises(asyncio.CancelledError):
            await resumed

        assert lock.status_at_release is SessionStatus.CANCELLED
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_activation_failure_persists_status_before_tree_lock_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
        status=SessionStatus.IDLE,
    ))
    original_mark_status = store.mark_status

    async def fail_activation(session_id: str, status: SessionStatus) -> None:
        if status is SessionStatus.ACTIVE:
            raise RuntimeError("activation failed")
        await original_mark_status(session_id, status)

    monkeypatch.setattr(store, "mark_status", fail_activation)

    class InspectingLock:
        status_at_release: SessionStatus | None = None

        @asynccontextmanager
        async def acquired_async(self):
            try:
                yield self
            finally:
                self.status_at_release = (
                    await store.load_context("session-1")
                ).session.status

    lock = InspectingLock()
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: pytest.fail("loop must not start"),
        session_lock_factory=lambda _root_id: lock,  # type: ignore[arg-type]
    )

    try:
        with pytest.raises(RuntimeError, match="activation failed"):
            await runtime.resume("session-1", "continue")

        assert lock.status_at_release is SessionStatus.FAILED
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_cancellation_waits_for_final_status_before_tree_lock_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = paths_for(tmp_path)
    store = SQLiteSessionStore(paths.sessions_db)
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        paths.project_id,
        paths.workspace_id,
        "fake",
        "model",
        workspace_path=str(tmp_path),
    ))
    finalization_started = asyncio.Event()
    release_finalization = asyncio.Event()
    original_mark_status = store.mark_status

    async def block_finalization(
        session_id: str, status: SessionStatus
    ) -> None:
        if status is SessionStatus.CANCELLED:
            finalization_started.set()
            await release_finalization.wait()
        await original_mark_status(session_id, status)

    monkeypatch.setattr(store, "mark_status", block_finalization)

    class InspectingLock:
        released = False
        status_at_release: SessionStatus | None = None

        @asynccontextmanager
        async def acquired_async(self):
            try:
                yield self
            finally:
                self.released = True
                self.status_at_release = (
                    await store.load_context("session-1")
                ).session.status

    class BlockingLoop:
        def __init__(self) -> None:
            self.entered = asyncio.Event()

        async def run_turn(self, session_id: str, prompt: str) -> None:
            self.entered.set()
            await asyncio.Event().wait()

    lock = InspectingLock()
    loop = BlockingLoop()
    runtime = AgentRuntime(
        store=store,
        paths=paths,
        provider_name="fake",
        model="model",
        loop_factory=lambda *_args: loop,
        cleanup_timeout=0.01,
        session_lock_factory=lambda _root_id: lock,  # type: ignore[arg-type]
    )
    resumed = asyncio.create_task(runtime.resume("session-1", "continue"))
    await loop.entered.wait()
    resumed.cancel()
    await finalization_started.wait()

    try:
        await asyncio.sleep(0.03)

        assert not resumed.done()
        assert not lock.released

        release_finalization.set()
        with pytest.raises(asyncio.CancelledError):
            await resumed

        assert lock.status_at_release is SessionStatus.CANCELLED
    finally:
        release_finalization.set()
        if not resumed.done():
            resumed.cancel()
            with pytest.raises(asyncio.CancelledError):
                await resumed
        await runtime.close()
