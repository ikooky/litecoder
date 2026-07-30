"""Model Context Protocol tool integration."""

from __future__ import annotations

import inspect
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from typing import Any

from litecoder.settings import MCPServerSettings
from litecoder.tools.models import (
    PermissionRisk,
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolFailure,
    ToolSpec,
)
from litecoder.tools.registry import ToolRegistry


SessionFactory = Callable[
    [str, MCPServerSettings], Awaitable[object] | object
]


class MCPToolAdapter:
    """Component responsible for the mcp tool adapter."""
    def __init__(
        self,
        *,
        server_name: str,
        remote_name: str,
        session: object,
        spec: ToolSpec,
    ) -> None:
        self.server_name = server_name
        self.remote_name = remote_name
        self.session = session
        self.spec = spec

    @classmethod
    async def from_remote(
        cls, server_name: str, remote_tool: object, session: object
    ) -> "MCPToolAdapter":
        """Construct a value from remote data."""
        remote_name = _remote_name(remote_tool)
        annotations = _metadata(remote_tool, "annotations")
        mutates_workspace, permission_risk, requires_confirmation = _classify_metadata(annotations)
        spec = ToolSpec(
            name=f"mcp__{_safe_segment(server_name)}__{_safe_segment(remote_name)}",
            description=_description(remote_tool),
            input_schema=_input_schema(remote_tool),
            mutates_workspace=mutates_workspace,
            concurrency="exclusive" if mutates_workspace else "shared",
            permission_risk=permission_risk,
            requires_confirmation=requires_confirmation,
        )
        return cls(
            server_name=server_name,
            remote_name=remote_name,
            session=session,
            spec=spec,
        )

    async def execute(
        self, call: ToolCall, context: ToolContext
    ) -> ToolExecution:
        """Execute the requested tool call."""
        del context
        call_tool = getattr(self.session, "call_tool", None)
        if call_tool is None:
            raise ToolFailure("MCP session cannot call tools")
        try:
            result = call_tool(self.remote_name, arguments=call.arguments)
            resolved = await result if inspect.isawaitable(result) else result
        except Exception as error:
            raise ToolFailure("MCP tool call failed") from error
        content = normalize_mcp_content(_metadata(resolved, "content"))
        return ToolExecution.success(
            content,
            changed_workspace=self.spec.mutates_workspace,
            preview={"server": self.server_name, "tool": self.remote_name},
        )


class MCPConnectionManager:
    """Manager coordinating the mcp connection manager."""
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self.registry = registry
        self.session_factory = session_factory
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, object] = {}

    async def connect_all(
        self, servers: Mapping[str, MCPServerSettings]
    ) -> None:
        """Connect the all."""
        for name, settings in servers.items():
            await self.connect(name, settings)

    async def connect(self, name: str, settings: MCPServerSettings) -> None:
        """Connect the requested operation."""
        session = await self._create_session(name, settings)
        initialize = getattr(session, "initialize", None)
        if initialize is not None:
            initialized = initialize()
            if inspect.isawaitable(initialized):
                await initialized
        self.sessions[name] = session
        for remote in await _list_tools(session):
            self.registry.register(
                await MCPToolAdapter.from_remote(name, remote, session)
            )

    async def close_all(self) -> None:
        """Close the all."""
        failures: list[BaseException] = []
        if self.session_factory is not None:
            for session in tuple(self.sessions.values()):
                close = getattr(session, "aclose", None) or getattr(
                    session, "close", None
                )
                if close is None:
                    continue
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except BaseException as error:
                    failures.append(error)
        self.sessions.clear()
        try:
            await self.exit_stack.aclose()
        except BaseException as error:
            failures.append(error)
        if failures:
            raise failures[0]

    async def _create_session(
        self, name: str, settings: MCPServerSettings
    ) -> object:
        if self.session_factory is not None:
            created = self.session_factory(name, settings)
            return await created if inspect.isawaitable(created) else created
        return await self._create_sdk_session(settings)

    async def _create_sdk_session(self, settings: MCPServerSettings) -> object:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as error:
            raise RuntimeError(
                "MCP support requires installing litecoder with the mcp extra"
            ) from error

        if settings.transport == "stdio":
            assert settings.command is not None
            transport = stdio_client(
                StdioServerParameters(
                    command=settings.command,
                    args=list(settings.args),
                    env=_filtered_env(settings.env),
                )
            )
        else:
            assert settings.url is not None
            transport = streamablehttp_client(
                settings.url, headers=dict(settings.headers)
            )
        read, write, *_ = await self.exit_stack.enter_async_context(transport)
        return await self.exit_stack.enter_async_context(ClientSession(read, write))


def normalize_mcp_content(content: object) -> str:
    """Normalize the mcp content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_item_text(item) for item in content)
    if content is None:
        return ""
    return _json_text(content)


def _list_tools(session: object) -> Awaitable[tuple[object, ...]]:
    async def collect() -> tuple[object, ...]:
        list_tools = getattr(session, "list_tools", None)
        if list_tools is None:
            raise RuntimeError("MCP session cannot list tools")
        result = list_tools()
        resolved = await result if inspect.isawaitable(result) else result
        tools = _metadata(resolved, "tools")
        if not isinstance(tools, (list, tuple)):
            raise RuntimeError("MCP list_tools returned invalid payload")
        return tuple(tools)

    return collect()


def _classify_metadata(
    annotations: object,
) -> tuple[bool, PermissionRisk, bool]:
    """Only complete, typed MCP annotations may relax forced confirmation."""

    if not _has_complete_boolean_annotations(annotations):
        return True, "external", True
    read_only = _annotation(annotations, "readOnlyHint") is True
    destructive = _annotation(annotations, "destructiveHint") is True
    open_world = _annotation(annotations, "openWorldHint")
    if read_only and destructive:
        return True, "external", True
    mutates_workspace = destructive or not read_only
    if open_world is False:
        permission_risk: PermissionRisk = "workspace" if mutates_workspace else "safe"
    else:
        permission_risk = "external"
    return mutates_workspace, permission_risk, False


def _has_complete_boolean_annotations(annotations: object) -> bool:
    if annotations is None:
        return False
    for name in ("readOnlyHint", "destructiveHint", "openWorldHint"):
        if type(_annotation(annotations, name)) is not bool:
            return False
    return True

def _annotation(annotations: object, name: str) -> object:
    if isinstance(annotations, Mapping):
        return annotations.get(name)
    return getattr(annotations, name, None)


def _content_item_text(item: object) -> str:
    kind = _metadata(item, "type")
    text = _metadata(item, "text")
    if kind == "text" and isinstance(text, str):
        return text
    if isinstance(text, str):
        return text
    return _json_text(item)


def _remote_name(remote_tool: object) -> str:
    name = _metadata(remote_tool, "name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("MCP tool name must not be empty")
    return name


def _description(remote_tool: object) -> str:
    description = _metadata(remote_tool, "description")
    return description if isinstance(description, str) else ""


def _input_schema(remote_tool: object) -> dict[str, object]:
    schema = _metadata(remote_tool, "inputSchema")
    if schema is None:
        schema = _metadata(remote_tool, "input_schema")
    if isinstance(schema, dict):
        return schema
    return {"type": "object"}


def _metadata(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _safe_segment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip()).strip("_").lower()
    if not normalized:
        raise ValueError("MCP namespace segment must not be empty")
    return normalized


def _json_text(value: object) -> str:
    try:
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        elif hasattr(value, "__dict__"):
            value = vars(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _filtered_env(env: Mapping[str, str]) -> dict[str, str]:
    allowlist = (
        "PATH",
        "SYSTEMROOT",
        "COMSPEC",
        "PATHEXT",
        "HOME",
        "USERPROFILE",
        "TMP",
        "TEMP",
    )
    merged = {key: os.environ[key] for key in allowlist if key in os.environ}
    for key, value in env.items():
        if not isinstance(key, str) or not key:
            raise ValueError("MCP environment keys must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError("MCP environment values must be strings")
        merged[key] = value
    return merged
