"""Model Context Protocol tool integration."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from litecoder.paths import AppPaths
from litecoder.settings import MCPServerSettings, Settings
from litecoder.tools.mcp import MCPConnectionManager
from litecoder.tools.registry import ToolRegistry


app = typer.Typer(no_args_is_help=True)


@app.command("list")
def list_servers() -> None:
    """List configured MCP servers."""
    paths = AppPaths.discover(Path.cwd())
    settings = Settings.load(paths)
    console = Console()
    if not settings.mcp_servers:
        console.print("No MCP servers configured.")
        return
    table = Table("Name", "Transport", "Endpoint")
    for name, server in sorted(settings.mcp_servers.items()):
        table.add_row(name, server.transport, _endpoint(server))
    console.print(table)


@app.command("tools")
def list_tools(server: str | None = None) -> None:
    """List tools exposed by configured MCP servers."""
    _run_connection_command(server, render_tools=True)


@app.command("test")
def test_servers(server: str | None = None) -> None:
    """Connect to configured MCP servers and report whether they initialize."""
    _run_connection_command(server, render_tools=False)


def _run_connection_command(
    selected_server: str | None, *, render_tools: bool
) -> None:
    paths = AppPaths.discover(Path.cwd())
    settings = Settings.load(paths)
    servers = _select_servers(settings.mcp_servers, selected_server)
    registry = ToolRegistry()
    manager = MCPConnectionManager(registry)
    console = Console()
    try:
        asyncio.run(
            _connect_render_and_close(
                manager,
                servers,
                registry,
                console,
                render_tools=render_tools,
            )
        )
    except Exception as error:
        console.print(f"[red]{error}[/red]", stderr=True)
        raise typer.Exit(1) from error


async def _connect_render_and_close(
    manager: MCPConnectionManager,
    servers: dict[str, MCPServerSettings],
    registry: ToolRegistry,
    console: Console,
    *,
    render_tools: bool,
) -> None:
    """Connect the render and close."""
    try:
        await manager.connect_all(servers)
        if render_tools:
            table = Table("Tool", "Description")
            for tool in registry.list():
                table.add_row(tool.spec.name, tool.spec.description)
            console.print(table)
        else:
            for name in sorted(servers):
                console.print(f"{name}: ok")
    finally:
        await manager.close_all()


def _select_servers(
    servers: dict[str, MCPServerSettings], selected: str | None
) -> dict[str, MCPServerSettings]:
    if selected is None:
        return dict(servers)
    try:
        return {selected: servers[selected]}
    except KeyError:
        raise typer.BadParameter(f"unknown MCP server {selected!r}") from None


def _endpoint(server: MCPServerSettings) -> str:
    if server.transport == "stdio":
        return server.command or ""
    return server.url or ""