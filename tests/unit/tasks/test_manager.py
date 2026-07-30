from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from litecoder.tasks.manager import (
    InvalidTaskTransition,
    TaskAlreadyExists,
    TaskBlocked,
    TaskManager,
    TaskNotClaimable,
    TaskOwnershipError,
)
from litecoder.tasks.models import TaskCreate, TaskStatus
from litecoder.tasks.store import TaskStore


@pytest.fixture
def manager(tmp_path: Path) -> TaskManager:
    return TaskManager(TaskStore(tmp_path / "tasks"))


async def completed_task(
    manager: TaskManager,
    task_id: str,
    agent_id: str = "agent-a",
) -> None:
    await manager.create(TaskCreate(task_id, task_id, f"Description {task_id}"))
    await manager.claim(task_id, agent_id)
    await manager.complete(task_id, agent_id)


@pytest.mark.asyncio
async def test_completed_task_cannot_return_to_in_progress(
    manager: TaskManager,
) -> None:
    await completed_task(manager, "t1")

    with pytest.raises(InvalidTaskTransition):
        await manager.start("t1", "agent-a")


@pytest.mark.asyncio
async def test_preassigned_pending_task_can_be_started_by_owner(
    manager: TaskManager,
) -> None:
    await manager.create(TaskCreate(
        "t1",
        "subject",
        "description",
        owner_agent_id="agent-a",
    ))

    task = await manager.start("t1", "agent-a")

    assert task.status is TaskStatus.IN_PROGRESS
    assert task.owner_agent_id == "agent-a"


@pytest.mark.asyncio
async def test_claim_rejects_task_until_dependencies_complete(
    manager: TaskManager,
) -> None:
    await manager.create(TaskCreate("dep", "dependency", "dependency"))
    await manager.create(TaskCreate(
        "work",
        "work",
        "work",
        dependencies=("dep",),
    ))

    with pytest.raises(TaskBlocked) as error:
        await manager.claim("work", "agent-a")

    assert error.value.blocking_ids == ["dep"]
    await manager.claim("dep", "agent-b")
    await manager.complete("dep", "agent-b")
    claimed = await manager.claim("work", "agent-a")
    assert claimed.status is TaskStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_wrong_owner_cannot_complete_task(manager: TaskManager) -> None:
    await manager.create(TaskCreate("t1", "subject", "description"))
    await manager.claim("t1", "agent-a")

    with pytest.raises(TaskOwnershipError):
        await manager.complete("t1", "agent-b")

    assert manager.store.read("t1").status is TaskStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_failed_task_reset_returns_to_unowned_pending(
    manager: TaskManager,
) -> None:
    await manager.create(TaskCreate("t1", "subject", "description"))
    await manager.claim("t1", "agent-a")
    await manager.fail("t1", "agent-a")

    task = await manager.reset("t1")

    assert task.status is TaskStatus.PENDING
    assert task.owner_agent_id is None


@pytest.mark.asyncio
async def test_reassign_rejects_running_task(manager: TaskManager) -> None:
    await manager.create(TaskCreate("t1", "subject", "description"))
    await manager.claim("t1", "agent-a")

    with pytest.raises(TaskNotClaimable):
        await manager.reassign("t1", "agent-b")


@pytest.mark.asyncio
async def test_worktree_binding_is_idempotent_and_conflict_safe(
    manager: TaskManager,
) -> None:
    await manager.create(TaskCreate("t1", "subject", "description"))

    first = await manager.bind_worktree("t1", "worktree-a")
    repeated = await manager.bind_worktree("t1", "worktree-a")

    assert first.worktree_id == "worktree-a"
    assert repeated.worktree_id == "worktree-a"
    with pytest.raises(ValueError, match="another worktree"):
        await manager.bind_worktree("t1", "worktree-b")

    cleared = await manager.unbind_worktree("t1", "worktree-a")
    repeated_clear = await manager.unbind_worktree("t1", "worktree-a")
    assert cleared.worktree_id is None
    assert repeated_clear.worktree_id is None


@pytest.mark.asyncio
async def test_duplicate_task_id_is_rejected(manager: TaskManager) -> None:
    request = TaskCreate("t1", "subject", "description")
    await manager.create(request)

    with pytest.raises(TaskAlreadyExists):
        await manager.create(request)


@pytest.mark.asyncio
async def test_task_ids_reject_case_insensitive_file_collisions(
    manager: TaskManager,
) -> None:
    await manager.create(TaskCreate("Task", "subject", "description"))

    with pytest.raises(TaskAlreadyExists):
        await manager.create(TaskCreate("task", "other", "other"))


@pytest.mark.asyncio
async def test_interrupted_task_can_resume_by_owner(
    manager: TaskManager,
) -> None:
    await manager.create(TaskCreate("t1", "subject", "description"))
    await manager.claim("t1", "agent-a")
    interrupted = manager.store.read("t1")
    interrupted.status = TaskStatus.INTERRUPTED
    manager.store.write(interrupted)

    task = await manager.resume("t1", "agent-a")

    assert task.status is TaskStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_cancelled_task_can_reset_to_pending(
    manager: TaskManager,
) -> None:
    await manager.create(TaskCreate("t1", "subject", "description"))
    cancelled = await manager.cancel("t1")
    reset = await manager.reset("t1")

    assert cancelled.status is TaskStatus.CANCELLED
    assert reset.status is TaskStatus.PENDING
    assert reset.owner_agent_id is None


@pytest.mark.asyncio
async def test_pending_task_can_be_reassigned_and_started(
    manager: TaskManager,
) -> None:
    await manager.create(TaskCreate(
        "t1",
        "subject",
        "description",
        owner_agent_id="agent-a",
    ))

    reassigned = await manager.reassign("t1", "agent-b")
    started = await manager.start("t1", "agent-b")

    assert reassigned.owner_agent_id == "agent-b"
    assert started.status is TaskStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_separate_task_managers_serialize_claims_with_file_lock(
    tmp_path: Path,
) -> None:
    from litecoder.common.locks import NamedFileLock

    store = TaskStore(tmp_path / "tasks")
    setup = TaskManager(
        store,
        file_lock=NamedFileLock.tasks("project-1", tmp_path),
    )
    await setup.create(TaskCreate("t1", "subject", "description"))

    first = TaskManager(
        TaskStore(tmp_path / "tasks"),
        file_lock=NamedFileLock.tasks("project-1", tmp_path),
    )
    second = TaskManager(
        TaskStore(tmp_path / "tasks"),
        file_lock=NamedFileLock.tasks("project-1", tmp_path),
    )

    results = await asyncio.gather(
        _claim_status(first, "t1", "agent-a"),
        _claim_status(second, "t1", "agent-b"),
    )

    assert sorted(results) == ["claimed", "not-claimable"]
    final = store.read("t1")
    assert final.status is TaskStatus.IN_PROGRESS
    assert final.owner_agent_id in {"agent-a", "agent-b"}


async def _claim_status(
    manager: TaskManager, task_id: str, agent_id: str
) -> str:
    try:
        await manager.claim(task_id, agent_id)
    except TaskNotClaimable:
        return "not-claimable"
    return "claimed"
