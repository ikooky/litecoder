"""Evaluation policy definitions."""

from __future__ import annotations

from litecoder.eval.domain import EvalMode, ExecutionPolicy, validate_mode


BASE_EVAL_TOOLS = frozenset(
    {
        "read_file",
        "write_file",
        "edit_file",
        "glob_files",
        "search_text",
        "run_shell",
    }
)

MEMORY_EVAL_TOOLS = frozenset(
    {
        "memory_list",
        "memory_read",
        "memory_update",
        "memory_delete",
    }
)


def policy_for_mode(mode: str) -> ExecutionPolicy:
    """Handle the policy for mode operation."""
    selected = EvalMode(validate_mode(mode))
    if selected is EvalMode.MEMORY:
        return ExecutionPolicy(
            BASE_EVAL_TOOLS | MEMORY_EVAL_TOOLS,
            max_rounds=None,
            max_tokens=None,
        )
    return ExecutionPolicy(
        BASE_EVAL_TOOLS,
        max_rounds=None,
        max_tokens=None,
    )
