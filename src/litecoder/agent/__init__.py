"""Public interfaces for the agent package."""

from litecoder.agent.result import AgentResult

__all__ = ["AgentLoop", "AgentResult", "AgentRuntime", "RuntimeBudgets"]


def __getattr__(name: str) -> object:
    # Keep package initialization independent from the task modules.  Child
    # factory imports occur while task orchestration is being initialized.
    if name in {"AgentLoop", "RuntimeBudgets"}:
        from litecoder.agent.loop import AgentLoop, RuntimeBudgets

        return {"AgentLoop": AgentLoop, "RuntimeBudgets": RuntimeBudgets}[name]
    if name == "AgentRuntime":
        from litecoder.agent.runtime import AgentRuntime

        return AgentRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
