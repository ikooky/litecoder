"""Data models for the surrounding subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from litecoder.common.trace.redaction import SecretRedactor
from litecoder.providers._json import JsonValue, snapshot_json, snapshot_mapping


DedupePolicy = Literal["default", "none"]
ToolConcurrency = Literal["shared", "exclusive"]
PermissionRisk = Literal["safe", "workspace", "external", "high"]


def _identifier(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Data model representing the tool spec."""
    name: str
    description: str
    input_schema: dict[str, object]
    mutates_workspace: bool
    concurrency: ToolConcurrency | None = None
    permission_risk: PermissionRisk | None = None
    dedupe_policy: DedupePolicy = "default"
    requires_confirmation: bool = False
    workspace_lock: bool = True

    def __post_init__(self) -> None:
        _identifier(self.name, "name")
        if not isinstance(self.description, str):
            raise ValueError("description must be text")
        if not isinstance(self.mutates_workspace, bool):
            raise ValueError("mutates_workspace must be a bool")
        object.__setattr__(self, "input_schema", snapshot_mapping(self.input_schema, "input_schema"))
        concurrency = self.concurrency or ("exclusive" if self.mutates_workspace else "shared")
        if concurrency not in {"shared", "exclusive"}:
            raise ValueError("concurrency must be shared or exclusive")
        if self.mutates_workspace and concurrency != "exclusive":
            raise ValueError("mutating tools require exclusive concurrency")
        object.__setattr__(self, "concurrency", concurrency)
        if not isinstance(self.workspace_lock, bool):
            raise ValueError("workspace_lock must be a bool")
        if self.mutates_workspace and not self.workspace_lock:
            raise ValueError("mutating tools require a workspace lock")
        risk = self.permission_risk or ("workspace" if self.mutates_workspace else "safe")
        if risk not in {"safe", "workspace", "external", "high"}:
            raise ValueError("permission_risk is invalid")
        object.__setattr__(self, "permission_risk", risk)
        if self.dedupe_policy not in {"default", "none"}:
            raise ValueError("dedupe_policy must be default or none")
        if not isinstance(self.requires_confirmation, bool):
            raise ValueError("requires_confirmation must be a bool")


@dataclass(slots=True)
class ToolCall:
    """Data model representing the tool call."""
    id: str
    name: str
    arguments: dict[str, object]

    def __post_init__(self) -> None:
        _identifier(self.id, "id")
        _identifier(self.name, "name")
        self.arguments = snapshot_mapping(self.arguments, "arguments")


@dataclass(slots=True)
class ToolContext:
    """Data model representing the tool context."""
    agent_session_id: str
    workspace_id: str
    workspace_root: Path
    metadata: dict[str, object] = field(default_factory=dict)
    secret_environment_names: tuple[str, ...] = field(default_factory=tuple)
    secret_values: tuple[str, ...] = field(default_factory=tuple, repr=False)
    parent_permission_broker: object | None = field(
        default=None, repr=False, compare=False
    )
    ui_factory: object | None = field(default=None, repr=False, compare=False)
    _redactor: SecretRedactor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _identifier(self.agent_session_id, "agent_session_id")
        _identifier(self.workspace_id, "workspace_id")
        if not isinstance(self.workspace_root, Path):
            raise ValueError("workspace_root must be a Path")
        self.metadata = snapshot_mapping(self.metadata, "metadata")
        self.secret_environment_names = _secret_tuple(
            self.secret_environment_names, "secret_environment_names"
        )
        self.secret_values = _secret_tuple(
            self.secret_values, "secret_values", omit_empty=True
        )
        self._redactor = SecretRedactor.with_values(self.secret_values)

    @property
    def redactor(self) -> SecretRedactor:
        """Handle the redactor operation."""
        return self._redactor


def _secret_tuple(
    value: object, field_name: str, *, omit_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(f"{field_name} must be a collection of strings")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must contain only strings")
    return tuple(dict.fromkeys(item for item in value if item or not omit_empty))


@dataclass(slots=True)
class ToolResult:
    """Data model representing the tool result."""
    tool_call_id: str
    status: str
    content: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.tool_call_id, "tool_call_id")
        _identifier(self.status, "status")
        if not isinstance(self.content, str):
            raise ValueError("content must be text")
        self.metadata = snapshot_mapping(self.metadata, "metadata")


@dataclass(slots=True)
class ToolExecution:
    """Data model representing the tool execution."""
    status: str
    content: str
    metadata: dict[str, object] = field(default_factory=dict)
    changed_workspace: bool = False
    preview: JsonValue = None

    def __post_init__(self) -> None:
        _identifier(self.status, "status")
        if self.status != "success":
            raise ValueError("ToolExecution status must be success")
        if not isinstance(self.content, str):
            raise ValueError("content must be text")
        if not isinstance(self.changed_workspace, bool):
            raise ValueError("changed_workspace must be a bool")
        self.metadata = snapshot_mapping(self.metadata, "metadata")
        self.preview = snapshot_json(self.preview, "preview")

    @classmethod
    def success(
        cls,
        content: str,
        *,
        metadata: dict[str, object] | None = None,
        changed_workspace: bool = False,
        preview: object = None,
    ) -> ToolExecution:
        """Create a successful execution result."""
        return cls("success", content, metadata or {}, changed_workspace, preview)

    def to_result(self, tool_call_id: str) -> ToolResult:
        """Convert this object to a result value."""
        metadata = snapshot_mapping(self.metadata, "metadata")
        if self.preview is not None:
            metadata["preview"] = snapshot_json(self.preview, "preview")
        return ToolResult(tool_call_id, self.status, self.content, metadata)


class ToolPartialFailure(Exception):
    """Raised when the tool partial failure conditions occur."""
    def __init__(
        self,
        safe_message: str,
        *,
        changed_workspace: bool,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if not isinstance(safe_message, str) or not safe_message:
            raise ValueError("safe_message must not be empty")
        if not isinstance(changed_workspace, bool):
            raise ValueError("changed_workspace must be a bool")
        super().__init__(safe_message)
        self.safe_message = safe_message
        self.changed_workspace = changed_workspace
        self.metadata = snapshot_mapping(metadata or {}, "metadata")


class ToolFailure(Exception):
    """Raised when the tool failure conditions occur."""
    def __init__(
        self, safe_message: str, *, metadata: dict[str, object] | None = None
    ) -> None:
        if not isinstance(safe_message, str) or not safe_message:
            raise ValueError("safe_message must not be empty")
        super().__init__(safe_message)
        self.safe_message = safe_message
        self.metadata = snapshot_mapping(metadata or {}, "metadata")


class ToolDenied(Exception):
    """Raised when the tool denied conditions occur."""
    def __init__(
        self, safe_message: str = "Denied by workspace safety policy"
    ) -> None:
        if not isinstance(safe_message, str) or not safe_message:
            raise ValueError("safe_message must not be empty")
        super().__init__(safe_message)
        self.safe_message = safe_message


class Tool(Protocol):
    """Protocol describing the tool behavior."""
    spec: ToolSpec

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution: ...
