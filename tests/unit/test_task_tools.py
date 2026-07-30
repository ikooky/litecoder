from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.tasks.manager import TaskManager
from litecoder.tasks.models import TaskCreate, TaskStatus
from litecoder.tasks.store import TaskStore
from litecoder.tools.models import ToolCall, ToolContext, ToolDenied, ToolFailure
from litecoder.tools.tasks import (
    TaskCancelTool,
    TaskClaimTool,
    TaskCompleteTool,
    TaskCreateTool,
    TaskFailTool,
    TaskGetTool,
    TaskListTool,
)


@pytest.fixture
def manager(tmp_path: Path) -> TaskManager:
    return TaskManager(TaskStore(tmp_path / "tasks"))


def _context(
    tmp_path: Path, agent_id: str, *, task_ids: list[str] | None = None
) -> ToolContext:
    metadata: dict[str, object] = {"agent_id": agent_id}
    if task_ids is not None:
        metadata["task_ids"] = task_ids
    return ToolContext("session-1", "workspace-1", tmp_path, metadata=metadata)


def _create_call(call_id: str, task_id: str) -> ToolCall:
    return ToolCall(
        call_id,
        "task_create",
        {"id": task_id, "subject": task_id, "description": f"Description for {task_id}"},
    )


@pytest.mark.asyncio
async def test_task_manager_lists_dependency_order_and_get_reads_persisted_task(
    manager: TaskManager,
) -> None:
    await manager.create(TaskCreate("dependency", "Dependency", "first"))
    await manager.create(TaskCreate(
        "dependent", "Dependent", "second", dependencies=("dependency",)
    ))

    listed = await manager.list()
    loaded = await manager.get("dependent")
    loaded.subject = "mutated local copy"

    assert [task.id for task in listed] == ["dependency", "dependent"]
    assert (await manager.get("dependent")).subject == "Dependent"


@pytest.mark.asyncio
async def test_task_tools_use_context_agent_id_for_claim_and_owned_transitions(
    tmp_path: Path, manager: TaskManager
) -> None:
    lead = _context(tmp_path, "lead")
    delegated = ["complete-me", "fail-me", "cancel-me"]
    worker = _context(tmp_path, "worker-1", task_ids=delegated)
    intruder = _context(tmp_path, "worker-2", task_ids=delegated)
    create = TaskCreateTool(manager)
    listing = TaskListTool(manager)
    get = TaskGetTool(manager)
    claim = TaskClaimTool(manager)
    complete = TaskCompleteTool(manager)
    fail = TaskFailTool(manager)
    cancel = TaskCancelTool(manager)

    created = await create.execute(_create_call("create-1", "complete-me"), lead)
    assert created.metadata["task"]["owner_agent_id"] is None
    assert [task["id"] for task in (await listing.execute(
        ToolCall("list", "task_list", {}), worker
    )).metadata["tasks"]] == ["complete-me"]
    assert (await get.execute(
        ToolCall("get", "task_get", {"id": "complete-me"}), worker
    )).metadata["task"]["id"] == "complete-me"

    claimed = await claim.execute(
        ToolCall("claim", "task_claim", {"id": "complete-me"}), worker
    )
    assert claimed.metadata["task"]["owner_agent_id"] == "worker-1"
    with pytest.raises(ToolFailure, match="not owned"):
        await complete.execute(
            ToolCall("complete-wrong", "task_complete", {"id": "complete-me"}),
            intruder,
        )
    assert (await complete.execute(
        ToolCall("complete", "task_complete", {"id": "complete-me"}), worker
    )).metadata["task"]["status"] == TaskStatus.COMPLETED.value

    await create.execute(_create_call("create-2", "fail-me"), lead)
    await claim.execute(ToolCall("claim-2", "task_claim", {"id": "fail-me"}), worker)
    assert (await fail.execute(
        ToolCall("fail", "task_fail", {"id": "fail-me"}), worker
    )).metadata["task"]["status"] == TaskStatus.FAILED.value

    await create.execute(_create_call("create-3", "cancel-me"), lead)
    await claim.execute(ToolCall("claim-3", "task_claim", {"id": "cancel-me"}), worker)
    with pytest.raises(ToolFailure, match="not owned"):
        await cancel.execute(
            ToolCall("cancel-wrong", "task_cancel", {"id": "cancel-me"}), intruder
        )
    assert (await cancel.execute(
        ToolCall("cancel", "task_cancel", {"id": "cancel-me"}), worker
    )).metadata["task"]["status"] == TaskStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_delegated_agent_cannot_create_project_tasks(
    tmp_path: Path, manager: TaskManager
) -> None:
    context = _context(tmp_path, "worker-1", task_ids=["delegated"])

    with pytest.raises(ToolDenied, match="delegated"):
        await TaskCreateTool(manager).execute(_create_call("create", "new-task"), context)


@pytest.mark.asyncio
async def test_task_list_is_limited_to_the_delegated_task_set(
    tmp_path: Path, manager: TaskManager
) -> None:
    await manager.create(TaskCreate("delegated", "Delegated", "allowed"))
    await manager.create(TaskCreate("outside", "Outside", "not delegated"))

    result = await TaskListTool(manager).execute(
        ToolCall("list", "task_list", {}),
        _context(tmp_path, "worker-1", task_ids=["delegated"]),
    )

    assert [task["id"] for task in result.metadata["tasks"]] == ["delegated"]


@pytest.mark.asyncio
async def test_task_get_rejects_a_task_outside_the_delegated_set(
    tmp_path: Path, manager: TaskManager
) -> None:
    await manager.create(TaskCreate("outside", "Outside", "not delegated"))

    with pytest.raises(ToolDenied, match="delegated"):
        await TaskGetTool(manager).execute(
            ToolCall("get", "task_get", {"id": "outside"}),
            _context(tmp_path, "worker-1", task_ids=["delegated"]),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_type", [
    TaskClaimTool, TaskCompleteTool, TaskFailTool, TaskCancelTool,
])
async def test_task_mutations_reject_tasks_outside_delegated_set(
    tmp_path: Path, manager: TaskManager, tool_type: type[object]
) -> None:
    await manager.create(TaskCreate("outside", "Outside", "not delegated"))
    context = _context(tmp_path, "worker-1", task_ids=["delegated"])
    if tool_type is not TaskClaimTool:
        await manager.claim("outside", "worker-1")

    with pytest.raises(ToolDenied, match="delegated"):
        await tool_type(manager).execute(  # type: ignore[operator]
            ToolCall("mutation", "task_mutation", {"id": "outside"}), context
        )
