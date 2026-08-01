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


def test_multi_agent_policy_has_explicit_runtime_budgets() -> None:
    policy = policy_for_mode("multi-agent")

    assert policy.max_rounds == 96
    assert policy.max_tokens == 250_000


def test_all_six_modes_are_registered() -> None:
    names = (
        "agent-benchmark",
        "context-manager",
        "tools-hooks",
        "memory",
        "task-state",
        "multi-agent",
    )

    assert tuple(mode_plugin(name).name for name in names) == names
