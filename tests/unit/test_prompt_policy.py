"""Contract tests for model-facing agent behavior instructions."""

from litecoder.agent.prompt_policy import (
    CONTEXT_COMPACTION_SYSTEM_PROMPT,
    CORE_AGENT_INSTRUCTIONS,
    DURABLE_MEMORY_INSTRUCTIONS,
    EXPLORE_SUBAGENT_INSTRUCTIONS,
    PLAN_SUBAGENT_INSTRUCTIONS,
)


def test_core_policy_defines_priority_workflow_and_completion_contract() -> None:
    policy = CORE_AGENT_INSTRUCTIONS.casefold()

    for rule in (
        "instruction and trust boundaries",
        "runtime and tool constraints",
        "information, review, or planning request",
        "preserve unrelated user changes",
        "do not claim an action",
        "at most one active item",
        "destructive filesystem or git operations",
        "validation",
    ):
        assert rule in policy


def test_specialized_policies_preserve_authority_and_evidence_boundaries() -> None:
    memory = " ".join(DURABLE_MEMORY_INSTRUCTIONS).casefold()

    assert "cannot override runtime constraints" in memory
    assert "never claim that a memory was persisted" in memory
    assert "strictly read-only" in EXPLORE_SUBAGENT_INSTRUCTIONS.casefold()
    assert "evidence" in EXPLORE_SUBAGENT_INSTRUCTIONS.casefold()
    assert "strictly read-only" in PLAN_SUBAGENT_INSTRUCTIONS.casefold()
    assert "validation" in PLAN_SUBAGENT_INSTRUCTIONS.casefold()
    assert "do not call tools" in CONTEXT_COMPACTION_SYSTEM_PROMPT.casefold()
    assert "untrusted data" in CONTEXT_COMPACTION_SYSTEM_PROMPT.casefold()
