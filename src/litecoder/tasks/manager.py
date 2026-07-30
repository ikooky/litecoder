"""Context assembly, persistence, and compaction coordination."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from litecoder.common.locks import NamedFileLock
from litecoder.tasks.graph import TaskGraph
from litecoder.tasks.models import (
    TaskCreate,
    TaskRecord,
    TaskStatus,
    allows_task_transition,
)
from litecoder.tasks.planning import PlanningView
from litecoder.tasks.store import TaskStore


class TaskManagerError(ValueError):
    """Raised when the task manager error conditions occur."""
    pass


class TaskAlreadyExists(TaskManagerError):
    """Component responsible for the task already exists."""
    def __init__(self, task_id: str) -> None:
        super().__init__(f"task {task_id!r} already exists")
        self.task_id = task_id


class TaskNotFound(TaskManagerError):
    """Component responsible for the task not found."""
    def __init__(self, task_id: str) -> None:
        super().__init__(f"task {task_id!r} was not found")
        self.task_id = task_id


class InvalidTaskTransition(TaskManagerError):
    """Component responsible for the invalid task transition."""
    def __init__(
        self,
        task_id: str,
        current: TaskStatus,
        target: TaskStatus,
    ) -> None:
        super().__init__(
            f"task {task_id!r} cannot transition from {current.value} "
            f"to {target.value}"
        )
        self.task_id = task_id
        self.current = current
        self.target = target


class TaskNotClaimable(TaskManagerError):
    """Component responsible for the task not claimable."""
    def __init__(self, task_id: str) -> None:
        super().__init__(f"task {task_id!r} is not claimable")
        self.task_id = task_id


class TaskBlocked(TaskManagerError):
    """Component responsible for the task blocked."""
    def __init__(self, task_id: str, blocking_ids: list[str]) -> None:
        super().__init__(
            f"task {task_id!r} is blocked by {', '.join(blocking_ids)}"
        )
        self.task_id = task_id
        self.blocking_ids = blocking_ids


class TaskOwnershipError(TaskManagerError):
    """Raised when the task ownership error conditions occur."""
    def __init__(self, task_id: str, agent_id: str) -> None:
        super().__init__(
            f"agent {agent_id!r} does not own task {task_id!r}"
        )
        self.task_id = task_id
        self.agent_id = agent_id


class TaskManager:
    """Manager coordinating the task manager."""
    def __init__(
        self,
        store: TaskStore,
        *,
        lock: asyncio.Lock | None = None,
        file_lock: NamedFileLock | None = None,
    ) -> None:
        self.store = store
        self.lock = lock or asyncio.Lock()
        self.file_lock = file_lock

    @asynccontextmanager
    async def _locked(self) -> AsyncIterator[None]:
        async with self.lock:
            if self.file_lock is None:
                yield
                return
            async with self.file_lock.acquired_async():
                yield

    async def create(self, request: TaskCreate) -> TaskRecord:
        """Create the requested object."""
        if not isinstance(request, TaskCreate):
            raise ValueError("task create request is invalid")
        async with self._locked():
            records = self.store.read_all()
            key = request.id.casefold()
            if any(record.id.casefold() == key for record in records):
                raise TaskAlreadyExists(request.id)
            record = request.to_record()
            PlanningView.ordered_tasks([*records, record])
            self.store.write(record)
            return record

    async def list(self) -> list[TaskRecord]:
        """Return a stable dependency-aware view of all durable tasks."""
        async with self._locked():
            return list(PlanningView.ordered_tasks(self.store.read_all()))

    async def get(self, task_id: str) -> TaskRecord:
        """Read one durable task through the same lock as state transitions."""
        async with self._locked():
            _, task = self._load(task_id)
            return task

    async def bind_worktree(
        self, task_id: str, worktree_id: str
    ) -> TaskRecord:
        """Persist an idempotent one-to-one task/worktree association."""
        worktree_id = _validate_worktree_id(worktree_id)
        async with self._locked():
            _, task = self._load(task_id)
            if task.worktree_id not in {None, worktree_id}:
                raise TaskManagerError(
                    f"task {task_id!r} is already bound to another worktree"
                )
            task.worktree_id = worktree_id
            self.store.write(task)
            return task

    async def unbind_worktree(
        self, task_id: str, worktree_id: str
    ) -> TaskRecord:
        """Clear the association only when it names the expected worktree."""
        worktree_id = _validate_worktree_id(worktree_id)
        async with self._locked():
            _, task = self._load(task_id)
            if task.worktree_id is None:
                return task
            if task.worktree_id != worktree_id:
                raise TaskManagerError(
                    f"task {task_id!r} is bound to another worktree"
                )
            task.worktree_id = None
            self.store.write(task)
            return task

    async def claim(self, task_id: str, agent_id: str) -> TaskRecord:
        """Claim a pending task for an agent."""
        agent_id = _validate_agent_id(agent_id)
        async with self._locked():
            records, task = self._load(task_id)
            if (
                task.status is not TaskStatus.PENDING
                or task.owner_agent_id is not None
            ):
                raise TaskNotClaimable(task_id)
            self._ensure_unblocked(task, records)
            task.owner_agent_id = agent_id
            task.status = TaskStatus.IN_PROGRESS
            self.store.write(task)
            return task

    async def start(self, task_id: str, agent_id: str) -> TaskRecord:
        """Start the managed runtime."""
        agent_id = _validate_agent_id(agent_id)
        async with self._locked():
            records, task = self._load(task_id)
            self._require_owner(task, agent_id)
            self._ensure_unblocked(task, records)
            self._transition(task, TaskStatus.IN_PROGRESS)
            self.store.write(task)
            return task

    async def complete(self, task_id: str, agent_id: str) -> TaskRecord:
        """Complete the requested operation."""
        return await self._owned_transition(
            task_id, agent_id, TaskStatus.COMPLETED
        )

    async def fail(self, task_id: str, agent_id: str) -> TaskRecord:
        """Mark the task as failed."""
        return await self._owned_transition(
            task_id, agent_id, TaskStatus.FAILED
        )

    async def cancel(
        self,
        task_id: str,
        agent_id: str | None = None,
    ) -> TaskRecord:
        """Cancel the pending operation."""
        if agent_id is not None:
            agent_id = _validate_agent_id(agent_id)
        async with self._locked():
            _, task = self._load(task_id)
            if task.owner_agent_id is not None:
                if agent_id is None:
                    raise TaskOwnershipError(task_id, "")
                self._require_owner(task, agent_id)
            self._transition(task, TaskStatus.CANCELLED)
            self.store.write(task)
            return task

    async def resume(self, task_id: str, agent_id: str) -> TaskRecord:
        """Resume a paused task or session."""
        agent_id = _validate_agent_id(agent_id)
        async with self._locked():
            records, task = self._load(task_id)
            self._require_owner(task, agent_id)
            self._ensure_unblocked(task, records)
            self._transition(task, TaskStatus.IN_PROGRESS)
            self.store.write(task)
            return task

    async def reset(self, task_id: str) -> TaskRecord:
        """Reset the task state."""
        async with self._locked():
            _, task = self._load(task_id)
            self._transition(task, TaskStatus.PENDING)
            task.owner_agent_id = None
            task.worktree_id = None
            self.store.write(task)
            return task

    async def reassign(self, task_id: str, agent_id: str) -> TaskRecord:
        """Assign the task to a different agent."""
        agent_id = _validate_agent_id(agent_id)
        async with self._locked():
            _, task = self._load(task_id)
            if task.status in {
                TaskStatus.IN_PROGRESS,
                TaskStatus.COMPLETED,
                TaskStatus.CANCELLED,
            }:
                raise TaskNotClaimable(task_id)
            task.owner_agent_id = agent_id
            self.store.write(task)
            return task

    async def validate_graph(self) -> None:
        """Validate the graph."""
        async with self._locked():
            records = self.store.read_all(validate_graph=False)
            TaskGraph.from_records(records).validate_all()

    async def recover_interrupted(self) -> list[TaskRecord]:
        """Handle the recover interrupted operation."""
        async with self._locked():
            records = self.store.read_all(validate_graph=False)
            TaskGraph.from_records(records).validate_all()
            changed: list[TaskRecord] = []
            for record in records:
                if record.status is TaskStatus.IN_PROGRESS:
                    record.status = TaskStatus.INTERRUPTED
                    changed.append(record)
            if changed:
                self.store.replace_many(records)
            return changed

    async def _owned_transition(
        self,
        task_id: str,
        agent_id: str,
        target: TaskStatus,
    ) -> TaskRecord:
        agent_id = _validate_agent_id(agent_id)
        async with self._locked():
            _, task = self._load(task_id)
            self._require_owner(task, agent_id)
            self._transition(task, target)
            self.store.write(task)
            return task

    def _load(
        self, task_id: str
    ) -> tuple[dict[str, TaskRecord], TaskRecord]:
        records = {record.id: record for record in self.store.read_all()}
        try:
            task = records[task_id]
        except KeyError:
            raise TaskNotFound(task_id) from None
        return records, task

    def _ensure_unblocked(
        self,
        task: TaskRecord,
        records: dict[str, TaskRecord],
    ) -> None:
        blocking = [
            dependency_id
            for dependency_id in task.dependencies
            if records[dependency_id].status is not TaskStatus.COMPLETED
        ]
        if blocking:
            raise TaskBlocked(task.id, blocking)

    @staticmethod
    def _require_owner(task: TaskRecord, agent_id: str) -> None:
        if task.owner_agent_id != agent_id:
            raise TaskOwnershipError(task.id, agent_id)

    @staticmethod
    def _transition(task: TaskRecord, target: TaskStatus) -> None:
        if not allows_task_transition(task.status, target):
            raise InvalidTaskTransition(task.id, task.status, target)
        task.status = target


def _validate_agent_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
    ):
        raise ValueError("agent id is invalid")
    return value

def _validate_worktree_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
    ):
        raise ValueError("worktree id is invalid")
    return value
