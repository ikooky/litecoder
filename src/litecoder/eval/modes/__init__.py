"""Public interfaces for the modes package."""

from __future__ import annotations

from litecoder.eval.domain import EvalMode, validate_mode
from litecoder.eval.modes.agent_benchmark import AgentBenchmarkMode
from litecoder.eval.modes.base import EvalModePlugin
from litecoder.eval.modes.context_manager import ContextManagerMode
from litecoder.eval.modes.memory import MemoryMode
from litecoder.eval.modes.task_state import TaskStateMode
from litecoder.eval.modes.tools_hooks import ToolsHooksMode


_MODES: dict[EvalMode, EvalModePlugin] = {
    EvalMode.AGENT_BENCHMARK: AgentBenchmarkMode(),
    EvalMode.CONTEXT_MANAGER: ContextManagerMode(),
    EvalMode.TOOLS_HOOKS: ToolsHooksMode(),
    EvalMode.MEMORY: MemoryMode(),
    EvalMode.TASK_STATE: TaskStateMode(),
}


def mode_plugin(name: str) -> EvalModePlugin:
    """Handle the mode plugin operation."""
    return _MODES[EvalMode(validate_mode(name))]


__all__ = ["EvalModePlugin", "mode_plugin"]
