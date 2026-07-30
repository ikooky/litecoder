"""Runtime leases and lifecycle management for agents."""

from __future__ import annotations

import asyncio
import inspect
import contextvars
import json
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from litecoder.agent.loop import AgentLoop
from litecoder.agent.result import AgentResult
from litecoder.common.locks import NamedFileLock, ResourceLockUnavailable
from litecoder.common.trace import SecretRedactor, TraceRecorder
from litecoder.context.prompt import provider_neutral_summary
from litecoder.context.session.models import (
    MessageRecord, SessionRecord, SessionStatus,
)
from litecoder.context.session.store import SQLiteSessionStore
from litecoder.paths import AppPaths
from litecoder.providers.models import Usage
from litecoder.tasks.manager import TaskManager
from litecoder.tools.background import BackgroundManager
from litecoder.tools.permission import PermissionMode


class InvalidTaskGraphMode(RuntimeError):
    """Component responsible for the invalid task graph mode."""
    def __init__(self, error: ValueError) -> None:
        super().__init__(f"task graph is invalid: {error}")
        self.error = error


_ROOT_TURN_LEASE_ISSUER = object()


class RootTurnLease:
    """Private-capability token for child work performed under a root turn."""

    __slots__ = (
        "root_session_id",
        "trace_id",
        "trace_recorder",
        "_issuer",
        "_active",
    )

    def __init__(
        self,
        root_session_id: str,
        trace_id: str | None,
        trace_recorder: TraceRecorder | None,
        *,
        _issuer: object,
    ) -> None:
        if _issuer is not _ROOT_TURN_LEASE_ISSUER:
            raise TypeError("RootTurnLease is issued by AgentRuntime only")
        self.root_session_id = root_session_id
        self.trace_id = trace_id
        self.trace_recorder = trace_recorder
        self._issuer = _issuer
        self._active = True

    def _matches(self, root_session_id: str) -> bool:
        return (
            self._issuer is _ROOT_TURN_LEASE_ISSUER
            and self._active
            and self.root_session_id == root_session_id
            and self.trace_id is not None
            and self.trace_recorder is not None
        )

    def _revoke(self) -> None:
        self._active = False


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Data model representing the runtime context."""
    root_session_id: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    agent_id: str
    trace_recorder: TraceRecorder
    redactor: SecretRedactor
    secret_environment_names: tuple[str, ...]
    secret_values: tuple[str, ...]
    parent_permission_broker: object | None = None
    permission_mode: str = PermissionMode.ASK.value
    permission_mode_resolver: Callable[[], str] | None = None
    memory_eligible: bool = True


LoopFactory = Callable[[str, str, RuntimeContext], AgentLoop]
SessionLockFactory = Callable[[str], NamedFileLock]


def _root_session_ids(sessions: list[SessionRecord]) -> dict[str, str]:
    sessions_by_id = {session.id: session for session in sessions}
    roots_by_session: dict[str, str] = {}
    for session_id in sessions_by_id:
        path: list[str] = []
        current_session_id = session_id
        while current_session_id not in roots_by_session:
            if current_session_id in path:
                raise RuntimeError("session parent cycle detected")
            path.append(current_session_id)
            current = sessions_by_id.get(current_session_id)
            if current is None:
                raise KeyError(session_id)
            if current.parent_session_id is None:
                root_id = current.id
                break
            current_session_id = current.parent_session_id
        else:
            root_id = roots_by_session[current_session_id]
        roots_by_session.update(dict.fromkeys(path, root_id))
    return roots_by_session


class AgentRuntime:
    """Component responsible for the agent runtime."""
    def __init__(
        self,
        *,
        store: SQLiteSessionStore,
        paths: AppPaths,
        provider_name: str,
        model: str,
        loop_factory: LoopFactory,
        id_factory: Callable[[], str] | None = None,
        trace_redactor: SecretRedactor | None = None,
        secret_environment_names: tuple[str, ...] = (),
        secret_values: tuple[str, ...] = (),
        startup_lock: NamedFileLock | None = None,
        session_lock_factory: SessionLockFactory | None = None,
        task_manager: TaskManager | None = None,
        background_manager: BackgroundManager | None = None,
        cleanup_timeout: float = 1.0,
        session_type: str = "root",
        parent_session_id: str | None = None,
        agent_id: str = "lead",
        span_id: str | None = None,
        parent_span_id: str | None = None,
        parent_permission_broker: object | None = None,
        permission_mode: PermissionMode | str = PermissionMode.ASK,
        closeables: tuple[object, ...] = (),
        manual_compactor: object | None = None,
        provider_models: Mapping[str, str | None] | None = None,
        root_turn_lease: RootTurnLease | None = None,
        owns_store: bool = True,
        declared_session_id: str | None = None,
        turn_end_resources: tuple[object, ...] = (),
    ) -> None:
        self.store = store
        self.paths = paths
        self.provider_name = provider_name
        self.model = model
        self.loop_factory = loop_factory
        self.id_factory = id_factory or (lambda: uuid.uuid4().hex)
        configured_redactor = trace_redactor or SecretRedactor.with_values(())
        self.secret_environment_names = tuple(dict.fromkeys(secret_environment_names))
        self.secret_values = tuple(
            dict.fromkeys(
                value
                for value in (*configured_redactor.values, *secret_values)
                if value
            )
        )
        self.trace_redactor = SecretRedactor.with_values(self.secret_values)
        self.startup_lock = startup_lock
        self.session_lock_factory = session_lock_factory
        self.task_manager = task_manager
        self.background_manager = background_manager
        self.session_type = session_type
        self.parent_session_id = parent_session_id
        self.agent_id = agent_id or "lead"
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.parent_permission_broker = parent_permission_broker
        self.permission_mode = _permission_mode_value(permission_mode)
        self.closeables = tuple(closeables)
        self.manual_compactor = manual_compactor
        self.provider_models = MappingProxyType(dict(provider_models or {}))
        if root_turn_lease is not None and not isinstance(
            root_turn_lease, RootTurnLease
        ):
            raise TypeError("root_turn_lease must be a RootTurnLease")
        if not isinstance(owns_store, bool):
            raise TypeError("owns_store must be a bool")
        if declared_session_id is not None and not declared_session_id.strip():
            raise ValueError("declared_session_id must not be empty")
        self.root_turn_lease = root_turn_lease
        self.owns_store = owns_store
        self.declared_session_id = declared_session_id
        self.session_id = declared_session_id
        self.turn_end_resources = list(turn_end_resources)
        self._root_turn_leases: dict[str, RootTurnLease] = {}
        self._turn_lease_context: contextvars.ContextVar[
            RootTurnLease | None
        ] = contextvars.ContextVar(
            f"litecoder-root-turn-lease-{id(self)}",
            default=None,
        )
        self.invalid_task_graph: ValueError | None = None
        self._started = startup_lock is None and task_manager is None
        self._startup_guard = asyncio.Lock()
        self._recorders: dict[str, TraceRecorder] = {}
        self._trace_ids: dict[str, str] = {}
        self._condition = asyncio.Condition()
        self._active_turns = 0
        self._closing = False
        self._closed = False
        self._active_session_id: str | None = None
        self._active_session_context: contextvars.ContextVar[
            str | None
        ] = contextvars.ContextVar(
            f"litecoder-active-session-{id(self)}",
            default=None,
        )
        if cleanup_timeout <= 0:
            raise ValueError("cleanup_timeout must be positive")
        self.cleanup_timeout = cleanup_timeout

    async def start(self) -> None:
        """Start the managed runtime."""
        if self._started:
            return
        async with self._startup_guard:
            if self._started:
                return
            if self._closing or self._closed:
                raise RuntimeError("Agent runtime is closing")
            if self.startup_lock is None:
                await self._recover_startup_state()
            else:
                async with self.startup_lock.acquired_async():
                    await self._recover_startup_state()
            self._started = True

    async def _recover_startup_state(self) -> None:
        live_root_detected = False
        if self.session_lock_factory is None:
            await self.store.recover_active_sessions(self.paths.project_id)
        else:
            project_sessions = await self.store.list_sessions(
                project_id=self.paths.project_id
            )
            active_session_ids = [
                session
                for session in project_sessions
                if session.status is SessionStatus.ACTIVE
            ]
            roots_by_session = _root_session_ids(project_sessions)
            locked_roots: set[str] = set()
            async with AsyncExitStack() as retained_locks:
                for root_id in dict.fromkeys(roots_by_session.values()):
                    lock = self.session_lock_factory(root_id)
                    try:
                        await retained_locks.enter_async_context(
                            lock.acquired_async()
                        )
                    except ResourceLockUnavailable:
                        locked_roots.add(root_id)
                live_root_detected = bool(locked_roots)
                await self.store.recover_active_sessions(
                    self.paths.project_id,
                    target_session_ids=tuple(
                        session.id
                        for session in active_session_ids
                        if roots_by_session[session.id] not in locked_roots
                    ),
                )
        if self.task_manager is None:
            return
        try:
            if live_root_detected:
                await self.task_manager.validate_graph()
            else:
                await self.task_manager.recover_interrupted()
        except ValueError as error:
            self.invalid_task_graph = error

    async def _ensure_operational(self) -> None:
        if not self._started:
            await self.start()
        if self.invalid_task_graph is not None:
            raise InvalidTaskGraphMode(self.invalid_task_graph)

    async def run(self, prompt: str) -> AgentResult:
        """Run the requested operation."""
        await self._ensure_operational()
        session_id = self.declared_session_id or self.id_factory()
        self.session_id = session_id
        await self.store.create_session(
            SessionRecord.new(
                session_id,
                self.paths.project_id,
                self.paths.workspace_id,
                self.provider_name,
                self.model,
                workspace_path=str(self.paths.workspace_root),
                status=SessionStatus.IDLE,
                session_type=self.session_type,
                parent_session_id=self.parent_session_id,
            )
        )
        return await self.resume(session_id, prompt)

    async def compact_session(self, session_id: str) -> object:
        """Handle the compact session operation."""
        if self.manual_compactor is None:
            raise RuntimeError("context compaction is not configured")
        compact = getattr(self.manual_compactor, "compact", None)
        if not callable(compact):
            raise RuntimeError("context compaction is not configured")
        return await compact(session_id)

    async def resume(
        self, session_id: str, prompt: str | None = None
    ) -> AgentResult:
        """Resume a paused task or session."""
        await self._ensure_operational()
        session = (await self.store.load_context(session_id)).session
        if prompt is None:
            return AgentResult(session_id, "ready", "session resumed", Usage(0, 0))
        tree_owned = False
        turn_started = False
        turn_completed = False
        turn_root_id: str | None = None
        lease_token: contextvars.Token[RootTurnLease | None] | None = None
        active_session_token = self._active_session_context.set(session_id)
        previous_active_session = self._active_session_id
        self._active_session_id = session_id
        try:
            root_id = await self._root_session_id(session)
            async with AsyncExitStack() as turn_locks:
                if self.startup_lock is None:
                    await turn_locks.enter_async_context(
                        self._session_tree_lock(root_id)
                    )
                    tree_owned = True
                else:
                    async with self._startup_guard:
                        async with self.startup_lock.acquired_async():
                            await turn_locks.enter_async_context(
                                self._session_tree_lock(root_id)
                            )
                            tree_owned = True
                try:
                    await self.store.mark_status(session_id, SessionStatus.ACTIVE)
                    turn = await self._begin_turn(session, root_id)
                    turn_started = True
                    turn_root_id = root_id
                    if self.root_turn_lease is None:
                        lease_token = self._turn_lease_context.set(
                            self._root_turn_leases[root_id]
                        )
                    loop = self.loop_factory(session.provider, session.model, turn)
                    result = await loop.run_turn(session_id, prompt)
                    turn_completed = True
                    return result
                except asyncio.CancelledError:
                    if not turn_completed:
                        await self._mark_status_while_tree_owned(
                            session_id, SessionStatus.CANCELLED
                        )
                    raise
                except ResourceLockUnavailable:
                    raise
                except Exception:
                    await self._mark_status_while_tree_owned(
                        session_id, SessionStatus.FAILED
                    )
                    raise
        except asyncio.CancelledError:
            raise
        except ResourceLockUnavailable:
            raise
        except Exception:
            if not tree_owned:
                await self._mark_status_bounded(session_id, SessionStatus.FAILED)
            raise
        finally:
            try:
                if turn_started and turn_root_id is not None:
                    await self._end_turn(turn_root_id)
            finally:
                if lease_token is not None:
                    self._turn_lease_context.reset(lease_token)
                self._active_session_context.reset(active_session_token)
                self._active_session_id = previous_active_session
    async def switch_provider(
        self, session_id: str, provider_name: str, model: str | None = None
    ) -> AgentResult:
        """Handle the switch provider operation."""
        parent = await self.store.load_context(session_id)
        selected_model = model or self.model
        if (
            parent.session.provider == provider_name
            and parent.session.model == selected_model
        ):
            self.provider_name, self.model = provider_name, selected_model
            return AgentResult(
                session_id, "ready", "provider unchanged", Usage(0, 0)
            )
        child_id = self.id_factory()
        await self.store.create_session(
            SessionRecord.new(
                child_id,
                parent.session.project_id,
                parent.session.workspace_id,
                provider_name,
                selected_model,
                session_type="derived",
                status=SessionStatus.IDLE,
                title=parent.session.title,
                workspace_path=parent.session.workspace_path,
                parent_session_id=session_id,
                metadata={"derived_for_provider_switch": True},
            )
        )
        messages = [
            {"role": item.role, "content": item.content}
            for item in parent.messages
        ]
        summary = self.trace_redactor.redact_data(
            provider_neutral_summary(messages)
        )
        if not isinstance(summary, dict):
            raise RuntimeError("provider summary redaction failed")
        await self.store.append_message(
            MessageRecord(
                session_id=child_id,
                role="system",
                content=[summary],
            )
        )
        self.provider_name, self.model = provider_name, selected_model
        return AgentResult(child_id, "ready", "provider switched", Usage(0, 0))

    async def close(self) -> None:
        """Close the managed resource and release any lock."""
        async with self._condition:
            if self._closed:
                return
            self._closing = True
            await self._condition.wait_for(lambda: self._active_turns == 0)
            recorders = tuple(self._recorders.values())
            self._recorders.clear()
            self._trace_ids.clear()
            self._closed = True
        failures: list[BaseException] = []
        if self.background_manager is not None:
            try:
                await self.background_manager.close()
            except BaseException as error:
                failures.append(error)
        for closeable in self.closeables:
            try:
                await _close_runtime_resource(closeable)
            except BaseException as error:
                failures.append(error)
        for recorder in recorders:
            try:
                await recorder.close()
            except BaseException as error:
                failures.append(error)
        if self.owns_store:
            try:
                await self.store.close()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise failures[0]

    @asynccontextmanager
    async def _session_tree_lock(
        self, root_session_id: str
    ) -> AsyncIterator[None]:
        if self.root_turn_lease is not None:
            if not self.root_turn_lease._matches(root_session_id):
                raise RuntimeError("root turn lease is inactive or does not match")
            yield
            return
        if self.session_lock_factory is None:
            yield
            return
        lock = self.session_lock_factory(root_session_id)
        async with lock.acquired_async():
            yield

    async def _root_session_id(self, session: SessionRecord) -> str:
        return await self.store.root_session_id(session.id)

    async def _begin_turn(
        self, session: SessionRecord, root_id: str
    ) -> RuntimeContext:
        async with self._condition:
            if self._closing or self._closed:
                raise RuntimeError("Agent runtime is closing")
            if self.root_turn_lease is not None:
                if not self.root_turn_lease._matches(root_id):
                    raise RuntimeError("root turn lease is inactive or does not match")
                recorder = self.root_turn_lease.trace_recorder
                trace_id = self.root_turn_lease.trace_id
                assert recorder is not None and trace_id is not None
            else:
                recorder = self._recorders.get(root_id)
                if recorder is None:
                    path = self._trace_path(root_id)
                    trace_id = _recover_trace_id(path) or uuid.uuid4().hex
                    recorder = TraceRecorder(path, self.trace_redactor)
                    await recorder.start()
                    self._recorders[root_id] = recorder
                    self._trace_ids[root_id] = trace_id
                else:
                    trace_id = self._trace_ids[root_id]
                if root_id in self._root_turn_leases:
                    raise RuntimeError("a root turn lease is already active")
                self._root_turn_leases[root_id] = RootTurnLease(
                    root_id,
                    trace_id,
                    recorder,
                    _issuer=_ROOT_TURN_LEASE_ISSUER,
                )
            self._active_turns += 1
            span_id = self._turn_span_id(session, root_id)
            permission_mode = self.current_permission_mode()
            return RuntimeContext(
                root_session_id=root_id,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=self._turn_parent_span_id(session, root_id, span_id),
                agent_id=self._turn_agent_id(session, root_id),
                trace_recorder=recorder,
                redactor=self.trace_redactor,
                secret_environment_names=self.secret_environment_names,
                secret_values=self.secret_values,
                parent_permission_broker=self.parent_permission_broker,
                permission_mode=permission_mode,
                permission_mode_resolver=self.current_permission_mode,
                memory_eligible=(
                    self.agent_id == "lead" and session.id == root_id
                ),
            )

    @property
    def active_session_id(self) -> str | None:
        """Handle the active session id operation."""
        return self._active_session_context.get() or self._active_session_id

    @property
    def active_root_turn_lease(self) -> RootTurnLease | None:
        """Handle the active root turn lease operation."""
        lease = self._turn_lease_context.get()
        return lease if lease is not None and lease._active else None

    def register_turn_end_resource(self, resource: object) -> None:
        """Register the turn end resource."""
        if resource not in self.turn_end_resources:
            self.turn_end_resources.append(resource)

    def current_permission_mode(self) -> str:
        """Handle the current permission mode operation."""
        return _permission_mode_value(self.permission_mode)

    def _turn_span_id(self, session: SessionRecord, root_id: str) -> str:
        if self.span_id is not None and self.span_id.strip():
            return self.span_id
        if session.id == root_id:
            return "root"
        return f"agent:{session.id}"

    def _turn_parent_span_id(
        self, session: SessionRecord, root_id: str, span_id: str
    ) -> str | None:
        if self.parent_span_id is not None:
            return self.parent_span_id
        if session.id == root_id or span_id == "root":
            return None
        return "root"

    def _turn_agent_id(self, session: SessionRecord, root_id: str) -> str:
        if session.id == root_id or self.agent_id != "lead":
            return self.agent_id
        return session.id

    async def _mark_status_bounded(
        self, session_id: str, status: SessionStatus
    ) -> None:
        task = asyncio.create_task(self.store.mark_status(session_id, status))
        done, pending = await asyncio.wait(
            {task}, timeout=self.cleanup_timeout
        )
        if done:
            _consume_runtime_task(task)
            return
        task.cancel()
        _, pending = await asyncio.wait(
            pending, timeout=min(0.01, self.cleanup_timeout)
        )
        for item in pending:
            item.cancel()
            item.add_done_callback(_consume_runtime_task)

    async def _mark_status_while_tree_owned(
        self, session_id: str, status: SessionStatus
    ) -> None:
        write = asyncio.create_task(self.store.mark_status(session_id, status))
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(write)
                break
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
                _clear_current_task_cancellation()
                if write.done():
                    write.result()
                    break
        if cancellation is not None:
            raise cancellation
    async def _end_turn(self, root_session_id: str) -> None:
        failure: BaseException | None = None
        if self.root_turn_lease is None:
            for resource in tuple(self.turn_end_resources):
                try:
                    await _end_runtime_turn_resource(resource)
                except BaseException as error:
                    failure = failure or error
            lease = self._root_turn_leases.pop(root_session_id, None)
            if lease is not None:
                lease._revoke()
        async with self._condition:
            self._active_turns -= 1
            self._condition.notify_all()
        if failure is not None:
            raise failure

    def _trace_path(self, root_session_id: str) -> Path:
        return self.paths.trace_path(root_session_id)




def _permission_mode_value(mode: PermissionMode | str) -> str:
    return PermissionMode(str(mode)).value


async def _end_runtime_turn_resource(resource: object) -> None:
    end_turn = getattr(resource, "end_turn", None)
    if not callable(end_turn):
        return
    result = end_turn()
    if inspect.isawaitable(result):
        await result

async def _close_runtime_resource(resource: object) -> None:
    close = (
        getattr(resource, "aclose", None)
        or getattr(resource, "close_all", None)
        or getattr(resource, "close", None)
    )
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result

def _consume_runtime_task(task: asyncio.Task[object]) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


def _clear_current_task_cancellation() -> None:
    task = asyncio.current_task()
    if task is None:
        return
    while task.cancelling():
        task.uncancel()

def _recover_trace_id(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                trace_id = value.get("trace_id") if isinstance(value, dict) else None
                if isinstance(trace_id, str) and trace_id:
                    return trace_id
    except OSError:
        return None
    return None
