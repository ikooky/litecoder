from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.tasks.manager import TaskManager
from litecoder.tasks.models import TaskRecord, TaskStatus
from litecoder.tasks.planning import TaskCycleError
from litecoder.tasks.store import TaskStore


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
async def test_recover_interrupted_marks_running_tasks_without_starting_them(
    tmp_path: Path,
) -> None:
    store = TaskStore(tmp_path / "tasks")
    store.replace_many([
        task("running", TaskStatus.IN_PROGRESS),
        task("pending", TaskStatus.PENDING),
    ])
    manager = TaskManager(store)

    changed = await manager.recover_interrupted()

    assert [record.id for record in changed] == ["running"]
    assert store.read("running").status is TaskStatus.INTERRUPTED
    assert store.read("pending").status is TaskStatus.PENDING


@pytest.mark.asyncio
async def test_corrupt_graph_is_not_mutated_during_recovery(
    tmp_path: Path,
) -> None:
    store = TaskStore(tmp_path / "tasks")
    store.replace_many([
        task("A", TaskStatus.IN_PROGRESS),
        task("B", TaskStatus.PENDING),
    ])
    a = store.read("A")
    b = store.read("B")
    a.dependencies = ["B"]
    b.dependencies = ["A"]
    store.write(a)
    store.write(b)
    manager = TaskManager(store)

    with pytest.raises(TaskCycleError):
        await manager.recover_interrupted()

    assert store.read("A").status is TaskStatus.IN_PROGRESS
    assert store.read("B").status is TaskStatus.PENDING


@pytest.mark.asyncio
async def test_validate_graph_rejects_corruption_without_mutating_tasks(
    tmp_path: Path,
) -> None:
    store = TaskStore(tmp_path / "tasks")
    store.replace_many([
        task("A", TaskStatus.IN_PROGRESS),
        task("B", TaskStatus.PENDING),
    ])
    a = store.read("A")
    b = store.read("B")
    a.dependencies = ["B"]
    b.dependencies = ["A"]
    store.write(a)
    store.write(b)
    manager = TaskManager(store)

    with pytest.raises(TaskCycleError):
        await manager.validate_graph()

    assert store.read("A").status is TaskStatus.IN_PROGRESS
    assert store.read("B").status is TaskStatus.PENDING
