from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from litecoder.tasks.manager import TaskManager, TaskNotClaimable
from litecoder.tasks.models import TaskCreate, TaskRecord, TaskStatus
from litecoder.tasks.store import TaskStore


@pytest.mark.asyncio
async def test_claim_is_atomic_and_only_one_owner_wins(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks")
    manager = TaskManager(store)
    await manager.create(TaskCreate("t1", "subject", "description"))

    results = await asyncio.gather(
        manager.claim("t1", "agent-a"),
        manager.claim("t1", "agent-b"),
        return_exceptions=True,
    )

    successes = [item for item in results if isinstance(item, TaskRecord)]
    failures = [item for item in results if isinstance(item, Exception)]
    persisted = store.read("t1")

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], TaskNotClaimable)
    assert persisted.status is TaskStatus.IN_PROGRESS
    assert persisted.owner_agent_id == successes[0].owner_agent_id
    assert not list(store.root.glob("*.tmp"))