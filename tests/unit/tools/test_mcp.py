from __future__ import annotations

from types import SimpleNamespace

import pytest

from litecoder.settings import MCPServerSettings
from litecoder.tools.mcp import MCPConnectionManager, MCPToolAdapter
from litecoder.tools.models import ToolCall, ToolContext
from litecoder.tools.permission import PermissionService
from litecoder.tools.registry import ToolRegistry


def remote_tool(
    name: str = "lookup",
    *,
    annotations: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description="Lookup docs",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
        annotations=annotations,
    )


@pytest.mark.asyncio
async def test_unknown_mcp_metadata_fails_closed() -> None:
    adapter = await MCPToolAdapter.from_remote(
        "docs", remote_tool(annotations=None), SimpleNamespace()
    )

    assert adapter.spec.name == "mcp__docs__lookup"
    assert adapter.spec.mutates_workspace is True
    assert adapter.spec.permission_risk == "external"
    assert adapter.spec.concurrency == "exclusive"
    assert adapter.spec.requires_confirmation is True


@pytest.mark.asyncio
async def test_read_only_closed_world_metadata_is_safe_shared() -> None:
    adapter = await MCPToolAdapter.from_remote(
        "docs",
        remote_tool(
            annotations={
                "readOnlyHint": True,
                "destructiveHint": False,
                "openWorldHint": False,
            }
        ),
        SimpleNamespace(),
    )

    assert adapter.spec.mutates_workspace is False
    assert adapter.spec.permission_risk == "safe"
    assert adapter.spec.concurrency == "shared"


@pytest.mark.asyncio
async def test_adapter_executes_remote_tool_and_normalizes_text(tmp_path) -> None:
    class Session:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        async def call_tool(
            self, name: str, *, arguments: dict[str, object]
        ) -> SimpleNamespace:
            self.calls.append((name, arguments))
            return SimpleNamespace(
                content=[
                    {"type": "text", "text": "first"},
                    SimpleNamespace(type="text", text="second"),
                ]
            )

    session = Session()
    adapter = await MCPToolAdapter.from_remote(
        "docs",
        remote_tool(annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}),
        session,
    )

    execution = await adapter.execute(
        ToolCall("call-1", "mcp__docs__lookup", {"query": "litecoder"}),
        ToolContext("agent", "workspace", tmp_path),
    )

    assert session.calls == [("lookup", {"query": "litecoder"})]
    assert execution.content == "first\nsecond"
    assert execution.changed_workspace is False


@pytest.mark.asyncio
async def test_connection_manager_registers_namespaced_tools() -> None:
    class Session:
        async def initialize(self) -> None:
            return None

        async def list_tools(self) -> SimpleNamespace:
            return SimpleNamespace(
                tools=[
                    remote_tool(
                        "lookup",
                        annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False},
                    )
                ]
            )

        async def aclose(self) -> None:
            return None

    async def session_factory(
        _name: str, _settings: MCPServerSettings
    ) -> Session:
        return Session()

    registry = ToolRegistry()
    manager = MCPConnectionManager(registry, session_factory=session_factory)

    await manager.connect_all(
        {
            "docs": MCPServerSettings(
                transport="stdio", command="fake-mcp-server"
            )
        }
    )
    try:
        assert registry.require("mcp__docs__lookup").spec.description == "Lookup docs"
    finally:
        await manager.close_all()

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "annotations",
    [
        {"readOnlyHint": True, "openWorldHint": False},
        {
            "readOnlyHint": True,
            "destructiveHint": False,
            "openWorldHint": "false",
        },
        {
            "readOnlyHint": True,
            "destructiveHint": True,
            "openWorldHint": False,
        },
    ],
)
async def test_incomplete_or_invalid_mcp_metadata_forces_confirmation(
    annotations: object,
) -> None:
    adapter = await MCPToolAdapter.from_remote(
        "docs", remote_tool(annotations=annotations), SimpleNamespace()
    )

    assert adapter.spec.mutates_workspace is True
    assert adapter.spec.permission_risk == "external"
    assert adapter.spec.requires_confirmation is True


@pytest.mark.asyncio
async def test_unknown_mcp_metadata_cannot_bypass_confirmation() -> None:
    adapter = await MCPToolAdapter.from_remote(
        "docs", remote_tool(annotations=None), SimpleNamespace()
    )

    decision = PermissionService().classify("bypass", adapter.spec)

    assert decision.action == "prompt"
    assert decision.allowed is False