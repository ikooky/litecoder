"""Regression coverage for model-visible built-in tool guidance."""

import pytest

from litecoder.tools.background import (
    BackgroundCancelTool,
    BackgroundStartTool,
    BackgroundStatusTool,
)
from litecoder.context.todos import TodoWriteTool
from litecoder.tools.builtin.agents import SpawnSubagentTool
from litecoder.tools.builtin.filesystem import EditFileTool, ReadFileTool, WriteFileTool
from litecoder.tools.builtin.git import GitDiffTool, GitStatusTool
from litecoder.tools.builtin.search import GlobFilesTool, SearchTextTool
from litecoder.tools.builtin.shell import RunShellTool
from litecoder.tools.builtin.team import (
    TeamCreateTool,
    TeamListTool,
    TeamReceiveTool,
    TeamRequestPlanApprovalTool,
    TeamRequestShutdownTool,
    TeamRespondPlanApprovalTool,
    TeamRespondShutdownTool,
    TeamSendTool,
)
from litecoder.tools.builtin.worktree import (
    WorktreeCreateTool,
    WorktreeListTool,
    WorktreeRemoveTool,
)
from litecoder.tools.memory import (
    MemoryDeleteTool,
    MemoryListTool,
    MemoryReadTool,
    MemoryUpdateTool,
)
from litecoder.tools.skills import LoadSkillTool
from litecoder.tools.tasks import (
    TaskCancelTool,
    TaskClaimTool,
    TaskCompleteTool,
    TaskCreateTool,
    TaskFailTool,
    TaskGetTool,
    TaskListTool,
)


@pytest.mark.parametrize(
    ("spec", "fragments"),
    [
        (ReadFileTool.spec, ("relevant path", "focused inspection")),
        (WriteFileTool.spec, ("new file", "edit_file")),
        (EditFileTool.spec, ("Read the current content", "replace_all")),
        (GlobFilesTool.spec, ("exact path", "narrow")),
        (SearchTextTool.spec, ("literal search", "scope")),
        (GitStatusTool.spec, ("existing changes", "user work")),
        (GitDiffTool.spec, ("verify", "Scope by path")),
        (RunShellTool.spec, ("tests, builds", "dedicated tool")),
        (SpawnSubagentTool.spec, ("bounded", "expected deliverable")),
        (TeamCreateTool.spec, ("independently scoped", "least-privilege")),
        (TeamSendTool.spec, ("blockers", "evidence")),
        (TeamReceiveTool.spec, ("cannot expand", "authority")),
        (TeamRequestPlanApprovalTool.spec, ("concrete plan", "dependent")),
        (TeamRespondPlanApprovalTool.spec, ("evidence-based", "reason")),
        (TeamRequestShutdownTool.spec, ("unfinished task", "recovery")),
        (TeamRespondShutdownTool.spec, ("active delegated work", "reason")),
        (TeamListTool.spec, ("valid recipient", "progress")),
        (WorktreeCreateTool.spec, ("durable task", "task")),
        (WorktreeListTool.spec, ("task-bound", "isolated work")),
        (WorktreeRemoveTool.spec, ("destructive", "discard")),
        (TaskCreateTool.spec, ("Lead-only", "TodoWrite")),
        (TaskListTool.spec, ("dependencies", "single-agent")),
        (TaskGetTool.spec, ("latest state", "ownership")),
        (TaskClaimTool.spec, ("before", "mutation")),
        (TaskCompleteTool.spec, ("validation", "unresolved")),
        (TaskFailTool.spec, ("blocker", "recovery")),
        (TaskCancelTool.spec, ("superseded", "teammate")),
        (TodoWriteTool.spec, ("Lead-only", "durable cross-agent tasks")),
        (MemoryListTool.spec, ("explicit user request", "automatically")),
        (MemoryReadTool.spec, ("explicit user request", "lower-priority")),
        (MemoryUpdateTool.spec, ("explicit user request", "persistence")),
        (MemoryDeleteTool.spec, ("explicit user request", "deletion")),
        (LoadSkillTool.spec, ("relevant", "unrelated")),
        (BackgroundStartTool.spec, ("independent", "completion")),
        (BackgroundStatusTool.spec, ("now needed", "poll")),
        (BackgroundCancelTool.spec, ("obsolete", "resulting state")),
    ],
)
def test_builtin_tool_descriptions_encode_usage_boundaries(
    spec: object, fragments: tuple[str, ...]
) -> None:
    description = str(getattr(spec, "description")).casefold()

    for fragment in fragments:
        assert fragment.casefold() in description
