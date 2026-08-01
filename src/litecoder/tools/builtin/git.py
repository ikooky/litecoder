"""Built-in Git tools."""

from __future__ import annotations

import os

from litecoder.tools.builtin._common import (
    canonical_workspace_root,
    optional_bool,
    resolve_workspace_path,
    workspace_relative,
)
from litecoder.tools.builtin.process import run_bounded_process
from litecoder.tools.builtin.shell import _safe_environment
from litecoder.tools.models import (
    ToolCall,
    ToolContext,
    ToolDenied,
    ToolExecution,
    ToolFailure,
    ToolPartialFailure,
    ToolSpec,
)


class GitStatusTool:
    """Component responsible for the git status tool."""
    spec = ToolSpec(
        "git_status",
        "Inspect the workspace Git baseline before a risky change, review, commit request, or worktree decision. Prefer this dedicated tool over shell Git inspection, and treat existing changes as user work unless evidence shows otherwise.",
        {
            "type": "object",
            "properties": {
                "porcelain": {"type": "string", "enum": ["v1", "v2"]}
            },
            "additionalProperties": False,
        },
        False,
        concurrency="shared",
        permission_risk="safe",
    )

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        porcelain = call.arguments.get("porcelain", "v1")
        if porcelain not in {"v1", "v2"}:
            raise ToolFailure(
                "Invalid tool arguments", metadata={"field": "porcelain"}
            )
        arguments = [
            "git",
            "--no-pager",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.quotepath=false",
            "status",
            f"--porcelain={porcelain}",
            "--untracked-files=all",
        ]
        return await _run_git(arguments, context)


class GitDiffTool:
    """Component responsible for the git diff tool."""
    spec = ToolSpec(
        "git_diff",
        "Inspect unstaged or staged workspace changes to understand existing work or verify the requested change. Prefer this dedicated tool over shell Git inspection and scope by path when only one area is relevant.",
        {
            "type": "object",
            "properties": {
                "staged": {"type": "boolean", "default": False},
                "path": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        False,
        concurrency="shared",
        permission_risk="safe",
    )

    def hard_guard(self, call: ToolCall, context: ToolContext) -> str | None:
        """Apply the tool safety guard before execution."""
        value = call.arguments.get("path")
        if value is None:
            return None
        try:
            resolve_workspace_path(context.workspace_root, value)
        except ToolDenied as error:
            return error.safe_message
        return None

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        staged = optional_bool(call.arguments, "staged", False)
        arguments = [
            "git",
            "--no-pager",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.quotepath=false",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
        ]
        if staged:
            arguments.append("--cached")
        value = call.arguments.get("path")
        if value is not None:
            path = resolve_workspace_path(context.workspace_root, value)
            root = canonical_workspace_root(context.workspace_root)
            relative = "." if path == root else workspace_relative(root, path)
            arguments.extend(("--", relative))
        return await _run_git(arguments, context)


_GIT_ENVIRONMENT = {
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_PAGER": "cat",
    "GIT_EDITOR": "true",
}


def _git_environment(context: ToolContext) -> dict[str, str]:
    environment = _safe_environment(context)
    environment.update(_GIT_ENVIRONMENT)
    return environment


async def _run_git(arguments: list[str], context: ToolContext) -> ToolExecution:
    """Run the git."""
    root = canonical_workspace_root(context.workspace_root)
    try:
        result = await run_bounded_process(
            arguments,
            workspace_root=root,
            cwd=".",
            env=_git_environment(context),
            timeout=30.0,
            redactor=context.redactor,
        )
    except ToolFailure:
        raise ToolPartialFailure(
            "Git is unavailable",
            changed_workspace=False,
            metadata={"exit_code": None, "changed_workspace": False},
        ) from None
    metadata = result.metadata(changed_workspace=False)
    if result.timed_out:
        raise ToolPartialFailure(
            "Git command timed out",
            changed_workspace=False,
            metadata=metadata,
        )
    if result.exit_code != 0:
        raise ToolPartialFailure(
            "Git command failed",
            changed_workspace=False,
            metadata=metadata,
        )
    return ToolExecution.success(
        result.stdout,
        metadata=metadata,
        changed_workspace=False,
        preview=result.stdout,
    )
