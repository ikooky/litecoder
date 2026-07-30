from __future__ import annotations

from types import SimpleNamespace

import pytest

from litecoder.hooks import HookManager
from litecoder.tools import (
    DuplicateGuard,
    PermissionService,
    PromptChoice,
    ToolCall,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    WorkspaceStateRegistry,
)
from litecoder.tools.mcp import MCPToolAdapter


class _Trace:
    async def record(self, _payload: object) -> None:
        return None


@pytest.mark.asyncio
async def test_mcp_tool_uses_standard_executor_pipeline(tmp_path) -> None:
    prompts = []

    async def prompt(request):
        prompts.append(request)
        return PromptChoice.ALLOW_ONCE

    class Session:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        async def call_tool(self, name: str, *, arguments: dict[str, object]):
            self.calls.append((name, arguments))
            return SimpleNamespace(content=[{"type": "text", "text": "done"}])

    remote = SimpleNamespace(
        name="write_note",
        description="Write a note",
        inputSchema={"type": "object"},
        annotations=None,
    )
    session = Session()
    adapter = await MCPToolAdapter.from_remote("notes", remote, session)
    registry = ToolRegistry()
    registry.register(adapter)
    workspaces = WorkspaceStateRegistry()
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=_Trace()),
        DuplicateGuard(annotation=lambda **_: None),
        PermissionService(prompt=prompt),
        workspaces,
    )

    result = await executor.execute(
        ToolCall("call-1", "mcp__notes__write_note", {"text": "hello"}),
        ToolContext(
            "agent",
            "workspace",
            tmp_path,
            metadata={"round_number": 1, "permission_mode": "ask"},
        ),
    )

    assert result.status == "success"
    assert result.content == "done"
    assert session.calls == [("write_note", {"text": "hello"})]
    assert [item.tool_name for item in prompts] == ["mcp__notes__write_note"]
    assert workspaces.get("workspace").version == 1