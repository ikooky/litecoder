"""Public interfaces for the builtin package."""

from __future__ import annotations

from litecoder.tools.builtin.agents import SpawnSubagentTool, register_agent_tools
from litecoder.tools.builtin.filesystem import EditFileTool, ReadFileTool, WriteFileTool
from litecoder.tools.builtin.git import GitDiffTool, GitStatusTool
from litecoder.tools.builtin.search import GlobFilesTool, SearchTextTool
from litecoder.tools.builtin.shell import RunShellTool
from litecoder.tools.builtin.worktree import (
    WorktreeCreateTool,
    WorktreeListTool,
    WorktreeRemoveTool,
    register_worktree_tools,
)
from litecoder.tools.builtin.team import (
    TeamCreateTool,
    TeamListTool,
    TeamReceiveTool,
    TeamRequestPlanApprovalTool,
    TeamRequestShutdownTool,
    TeamRespondPlanApprovalTool,
    TeamRespondShutdownTool,
    TeamSendTool,
    register_team_tools,
)
from litecoder.tools.models import Tool
from litecoder.tools.registry import ToolRegistry


def builtin_tools() -> tuple[Tool, ...]:
    """Handle the builtin tools operation."""
    return (
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        GlobFilesTool(),
        SearchTextTool(),
        RunShellTool(),
        GitStatusTool(),
        GitDiffTool(),
    )


def register_builtin_tools(registry: ToolRegistry) -> None:
    """Register the builtin tools."""
    for tool in builtin_tools():
        registry.register(tool)


__all__ = [
    "EditFileTool",
    "GitDiffTool",
    "GitStatusTool",
    "GlobFilesTool",
    "ReadFileTool",
    "RunShellTool",
    "SearchTextTool",
    "SpawnSubagentTool",
    "WriteFileTool",
    "TeamCreateTool",
    "TeamListTool",
    "TeamReceiveTool",
    "TeamRequestPlanApprovalTool",
    "TeamRequestShutdownTool",
    "TeamRespondPlanApprovalTool",
    "TeamRespondShutdownTool",
    "TeamSendTool",
    "register_team_tools",
    "WorktreeCreateTool",
    "WorktreeListTool",
    "WorktreeRemoveTool",
    "register_worktree_tools",
    "builtin_tools",
    "register_agent_tools",
    "register_builtin_tools",
]
