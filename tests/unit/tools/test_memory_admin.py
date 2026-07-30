from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.memory.store import MemoryStore
from litecoder.tools.memory import (
    MemoryDeleteTool,
    MemoryListTool,
    MemoryReadTool,
    MemoryUpdateTool,
)
from litecoder.tools.models import ToolCall, ToolContext, ToolFailure
from litecoder.tools.permission import PermissionMode, PermissionService, PromptChoice


def _context(root: Path, **metadata: object) -> ToolContext:
    values = {
        "root_session_id": "root",
        "agent_id": "lead",
    }
    values.update(metadata)
    return ToolContext("root", "workspace", root, metadata=values)


def test_memory_tools_require_root_turn_without_intent_authorization(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / ".memory")
    tool = MemoryListTool(store)
    assert tool.hard_guard(
        ToolCall("one", "memory_list", {}),
        ToolContext("root", "workspace", tmp_path),
    )
    assert tool.hard_guard(
        ToolCall("two", "memory_list", {}),
        _context(tmp_path, agent_id="child"),
    )
    assert (
        tool.hard_guard(
            ToolCall("three", "memory_list", {}),
            _context(tmp_path),
        )
        is None
    )


@pytest.mark.asyncio
async def test_list_and_read_do_not_create_memory_store(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / ".memory")
    context = _context(tmp_path)

    listed = await MemoryListTool(store).execute(
        ToolCall("list", "memory_list", {}), context
    )

    assert listed.status == "success"
    assert listed.metadata["count"] == 0
    assert not store.root.exists()
    with pytest.raises(ToolFailure, match="Memory entry is unavailable"):
        await MemoryReadTool(store).execute(
            ToolCall("read", "memory_read", {"name": "missing"}), context
        )
    assert not store.root.exists()


@pytest.mark.asyncio
async def test_update_and_delete_keep_index_consistent(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / ".memory")
    context = _context(tmp_path)
    update = MemoryUpdateTool(store)

    result = await update.execute(
        ToolCall(
            "update",
            "memory_update",
            {
                "name": "user-style",
                "type": "user",
                "description": "Preferred response style",
                "body": "Begin replies with 喵~.",
            },
        ),
        context,
    )

    assert result.changed_workspace
    assert store.read("user-style").body == "Begin replies with 喵~."
    assert "[user-style](user-style.md)" in store.read_index()

    deleted = await MemoryDeleteTool(store).execute(
        ToolCall("delete", "memory_delete", {"name": "user-style"}),
        context,
    )

    assert deleted.changed_workspace
    assert store.scan() == ()
    assert "user-style" not in store.read_index()


@pytest.mark.asyncio
async def test_update_rejects_configured_secrets_without_writing(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / ".memory")
    context = ToolContext(
        "root",
        "workspace",
        tmp_path,
        metadata={
            "root_session_id": "root",
            "agent_id": "lead",
        },
        secret_values=("sk-private",),
    )

    with pytest.raises(ToolFailure, match="Memory content was rejected"):
        await MemoryUpdateTool(store).execute(
            ToolCall(
                "update",
                "memory_update",
                {
                    "name": "credentials",
                    "type": "project",
                    "description": "Credentials",
                    "body": "token=sk-private",
                },
            ),
            context,
        )

    assert not store.root.exists()


@pytest.mark.asyncio
async def test_delete_always_prompts_even_after_root_session_approval(
    tmp_path: Path,
) -> None:
    prompts = []

    async def approve(request):
        prompts.append(request)
        return PromptChoice.ALLOW_FOR_ROOT_SESSION

    service = PermissionService(prompt=approve)
    spec = MemoryDeleteTool(MemoryStore(tmp_path / ".memory")).spec
    context = _context(
        tmp_path,
        permission_mode="bypass",
        bypass_authorized=True,
    )

    assert service.classify(PermissionMode.READ_ONLY, spec, context).action == "deny"
    assert service.classify(PermissionMode.BYPASS, spec, context).action == "prompt"
    assert (
        await service.decide(
            spec, ToolCall("one", "memory_delete", {"name": "one"}), context
        )
    ).allowed
    assert (
        await service.decide(
            spec, ToolCall("two", "memory_delete", {"name": "one"}), context
        )
    ).allowed
    assert len(prompts) == 2
