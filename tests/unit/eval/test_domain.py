from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.eval.domain import CaseSpec, ExecutionPolicy, RunSpec
from litecoder.eval.modes import mode_plugin
from litecoder.eval.policies import policy_for_mode


def test_case_spec_builds_restricted_prompt() -> None:
    spec = CaseSpec(
        "case-0001",
        "HumanEval/0",
        "humaneval",
        "answer",
        "def answer():\n",
        "agent-benchmark",
    )

    prompt = spec.prompt()

    assert "Only modify solution.py" in prompt
    assert "Do not create or edit tests" in prompt
    assert "HumanEval/0" in prompt


def test_run_spec_normalizes_mixed_datasets(tmp_path: Path) -> None:
    spec = RunSpec(
        "run-1",
        "memory",
        ("humaneval", "mbpp"),
        tmp_path,
    )

    assert spec.selected_datasets == ("humaneval", "mbpp")


def test_execution_policy_rejects_empty_tools() -> None:
    with pytest.raises(ValueError, match="allow at least one tool"):
        ExecutionPolicy(frozenset())


@pytest.mark.parametrize(
    "mode",
    (
        "agent-benchmark",
        "context-manager",
        "tools-hooks",
        "memory",
        "task-state",
    ),
)
def test_eval_modes_use_production_runtime_budgets(mode: str) -> None:
    policy = policy_for_mode(mode)

    assert policy.max_rounds is None
    assert policy.max_tokens is None


def test_memory_mode_keeps_full_memory_tool_surface() -> None:
    policy = policy_for_mode("memory")

    assert {
        "memory_list",
        "memory_read",
        "memory_update",
        "memory_delete",
    } <= policy.allowed_tools


def test_all_five_modes_are_registered() -> None:
    names = (
        "agent-benchmark",
        "context-manager",
        "tools-hooks",
        "memory",
        "task-state",
    )

    assert tuple(mode_plugin(name).name for name in names) == names
