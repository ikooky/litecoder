"""Built-in shell execution tool."""

from __future__ import annotations

import os

from litecoder.tools.builtin._common import (
    has_reserved_memory_reference,
    optional_number,
    resolve_workspace_path,
)
from litecoder.tools.builtin.process import ProcessResult, run_bounded_process
from litecoder.tools.models import (
    ToolCall,
    ToolContext,
    ToolDenied,
    ToolExecution,
    ToolFailure,
    ToolPartialFailure,
    ToolSpec,
)


class RunShellTool:
    """Component responsible for the run shell tool."""
    spec = ToolSpec(
        "run_shell",
        "Run one bounded argv-only command inside the workspace, primarily for tests, builds, package operations, or commands without a dedicated tool. Prefer structured file, search, and Git tools for those operations; inspect failures before retrying.",
        {
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "cwd": {"type": "string", "minLength": 1},
                "timeout": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["argv"],
            "additionalProperties": False,
        },
        True,
        concurrency="exclusive",
        permission_risk="high",
        dedupe_policy="default",
    )

    def hard_guard(self, call: ToolCall, context: ToolContext) -> str | None:
        """Apply the tool safety guard before execution."""
        try:
            resolve_workspace_path(
                context.workspace_root, call.arguments.get("cwd", ".")
            )
        except ToolDenied as error:
            return error.safe_message
        argv = call.arguments.get("argv")
        if (
            not isinstance(argv, list)
            or any(has_reserved_memory_reference(item) for item in argv)
        ):
            return "Denied by workspace safety policy"
        return None

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        argv = _argv(call.arguments.get("argv"))
        cwd = call.arguments.get("cwd", ".")
        timeout = optional_number(
            call.arguments, "timeout", 60.0, minimum=0.001
        )
        result = await run_bounded_process(
            argv,
            workspace_root=context.workspace_root,
            cwd=cwd,
            env=_safe_environment(context),
            timeout=timeout,
            redactor=context.redactor,
        )
        metadata = result.metadata(changed_workspace=True)
        if result.timed_out:
            raise ToolPartialFailure(
                "Shell command timed out",
                changed_workspace=True,
                metadata=metadata,
            )
        if result.exit_code != 0:
            raise ToolPartialFailure(
                "Shell command failed",
                changed_workspace=True,
                metadata=metadata,
            )
        content = _combined_output(result)
        return ToolExecution.success(
            content,
            metadata={**metadata, "changed_workspace": True},
            changed_workspace=True,
            preview=content,
        )


def _argv(value: object) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ToolFailure("Invalid shell arguments", metadata={"field": "argv"})
    return list(value)


def _safe_environment(context: ToolContext) -> dict[str, str]:
    protected = {name.casefold() for name in context.secret_environment_names}
    return {
        key: value
        for key, value in os.environ.items()
        if key.casefold() not in protected
    }


def _combined_output(result: ProcessResult) -> str:
    if result.stdout and result.stderr:
        return f"{result.stdout}\n[stderr]\n{result.stderr}"
    return result.stdout or result.stderr
