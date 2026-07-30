"""Data models for the surrounding subsystem."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import StrEnum


class HookPoint(StrEnum):
    """Enumeration of the hook point values."""
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_MODEL_CALL = "PreModelCall"
    POST_MODEL_CALL = "PostModelCall"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    TOOL_ERROR = "ToolError"
    AGENT_STOP = "AgentStop"
    SUBAGENT_START = "SubagentStart"
    SUBAGENT_STOP = "SubagentStop"


@dataclass(frozen=True, slots=True)
class HookDiagnostic:
    """Data model representing the hook diagnostic."""
    hook_id: str
    point: HookPoint
    phase: str
    kind: str
    code: str
    message: str

    def __post_init__(self) -> None:
        if not self.hook_id:
            raise ValueError("hook_id must not be empty")
        if not isinstance(self.point, HookPoint):
            raise TypeError("point must be a HookPoint")
        if self.phase not in {"pre", "post"}:
            raise ValueError("phase must be pre or post")
        if not self.kind or not self.code or not self.message:
            raise ValueError("diagnostic fields must not be empty")


@dataclass(frozen=True, slots=True)
class HookEnvelope:
    """Data model representing the hook envelope."""
    point: HookPoint
    payload: object
    hook_id: str
    dispatch_id: str
    phase: str

    def __post_init__(self) -> None:
        if not isinstance(self.point, HookPoint):
            raise TypeError("point must be a HookPoint")
        if not self.hook_id or not self.dispatch_id:
            raise ValueError("hook and dispatch identifiers must not be empty")
        if self.phase not in {"pre", "post"}:
            raise ValueError("phase must be pre or post")
        object.__setattr__(self, "payload", copy.deepcopy(self.payload))


@dataclass(frozen=True, slots=True)
class HookOutcome:
    """Data model representing the hook outcome."""
    payload: object
    blocked: bool = False
    diagnostics: tuple[HookDiagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.blocked, bool):
            raise TypeError("blocked must be a bool")
        diagnostics = tuple(self.diagnostics)
        if not all(isinstance(item, HookDiagnostic) for item in diagnostics):
            raise TypeError("diagnostics must contain HookDiagnostic values")
        object.__setattr__(self, "payload", copy.deepcopy(self.payload))
        object.__setattr__(self, "diagnostics", diagnostics)

    @classmethod
    def _from_trusted_snapshot(
        cls,
        payload: object,
        *,
        blocked: bool = False,
        diagnostics: tuple[HookDiagnostic, ...] = (),
    ) -> HookOutcome:
        """Build from a manager-owned snapshot without copying it again."""
        if not isinstance(blocked, bool):
            raise TypeError("blocked must be a bool")
        normalized = tuple(diagnostics)
        if not all(isinstance(item, HookDiagnostic) for item in normalized):
            raise TypeError("diagnostics must contain HookDiagnostic values")
        outcome = object.__new__(cls)
        object.__setattr__(outcome, "payload", payload)
        object.__setattr__(outcome, "blocked", blocked)
        object.__setattr__(outcome, "diagnostics", normalized)
        return outcome
