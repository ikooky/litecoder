from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from typer.testing import CliRunner

from litecoder.cli.app import app
from litecoder.cli import mcp as mcp_cli
from litecoder.paths import AppPaths


runner = CliRunner()


def make_paths(tmp_path: Path) -> AppPaths:
    return AppPaths(
        user_dir=tmp_path / ".litecoder",
        sessions_db=tmp_path / ".litecoder" / "sessions.db",
        project_id="project-1",
        project_dir=tmp_path / ".litecoder" / "projects" / "project-1",
        workspace_id="workspace-1",
        workspace_root=tmp_path,
    )


def use_paths(monkeypatch: pytest.MonkeyPatch, paths: AppPaths) -> None:
    monkeypatch.setattr(
        AppPaths,
        "discover",
        classmethod(lambda cls, cwd, home=None: paths),
    )


def test_mcp_list_renders_configured_servers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = make_paths(tmp_path)
    use_paths(monkeypatch, paths)
    paths.user_dir.mkdir(parents=True)
    paths.user_dir.joinpath("config.toml").write_text(
        "\n".join(
            [
                '[mcp_servers.docs]',
                'transport = "stdio"',
                'command = "python"',
                '',
                '[mcp_servers.remote]',
                'transport = "streamable-http"',
                'url = "https://example.invalid/mcp"',
                '',
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["mcp", "list"])

    assert result.exit_code == 0, result.output
    assert "docs" in result.output
    assert "stdio" in result.output
    assert "remote" in result.output
    assert "streamable-http" in result.output

def test_mcp_test_connects_and_closes_on_the_same_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loops: list[int] = []

    class ManagerDouble:
        def __init__(self, registry) -> None:
            self.registry = registry

        async def connect_all(self, servers) -> None:
            assert list(servers) == ["probe"]
            loops.append(id(asyncio.get_running_loop()))

        async def close_all(self) -> None:
            loops.append(id(asyncio.get_running_loop()))

    paths = make_paths(tmp_path)
    use_paths(monkeypatch, paths)
    paths.user_dir.mkdir(parents=True)
    paths.user_dir.joinpath("config.toml").write_text(
        "\n".join(
            [
                "[mcp_servers.probe]",
                'transport = "stdio"',
                'command = "probe-server"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_cli, "MCPConnectionManager", ManagerDouble)

    result = runner.invoke(app, ["mcp", "test"])

    assert result.exit_code == 0, result.output
    assert len(loops) == 2
    assert loops[0] == loops[1]
