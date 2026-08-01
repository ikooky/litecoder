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


def policy_for_mode(mode: str) -> ExecutionPolicy:
    """Handle the policy for mode operation."""
    selected = EvalMode(validate_mode(mode))
    if selected is EvalMode.MEMORY:
        return ExecutionPolicy(BASE_EVAL_TOOLS | {"memory_update"})
    if selected is EvalMode.CONTEXT_MANAGER:
        # A compaction treatment can spend several turns recovering from a
        # denied or duplicate tool call. Give it a little more room than the
        # normal single-turn modes without making a stalled case unbounded.
        return ExecutionPolicy(
            BASE_EVAL_TOOLS,
            max_rounds=32,
            max_tokens=120_000,
        )
    if selected is EvalMode.MULTI_AGENT:
        # Multi-agent mode needs the runtime's collaboration and worktree tools.
        # It keeps the full registry, but still has explicit budgets so a stalled
        # team cannot make the suite run unboundedly.
        return ExecutionPolicy(
            frozenset({"*"}),
            max_rounds=96,
            # Team coordination repeats the lead context for every worker
            # result. The previous 250k cap stopped 11/12 cases before the
            # workers could close their durable tasks. Keep a finite cap, but
            # leave enough headroom for the complete workflow.
            max_tokens=400_000,
        )
    return ExecutionPolicy(BASE_EVAL_TOOLS)
