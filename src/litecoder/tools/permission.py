"""Tool permission and confirmation policies."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from litecoder.providers._json import JsonValue, snapshot_mapping
from litecoder.tools.models import ToolCall, ToolContext, ToolSpec


PERMISSION_CONFIRMATION_TIMEOUT_SECONDS = 60.0


class PermissionMode(StrEnum):
    """Enumeration of the permission mode values."""
    ASK = "ask"
    READ_ONLY = "read-only"
    BYPASS = "bypass"


class PromptChoice(StrEnum):
    """Enumeration of the prompt choice values."""
    ALLOW_ONCE = "Allow once"
    ALLOW_FOR_ROOT_SESSION = "Allow for root session"
    DENY = "Deny"


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    """Data model representing the permission decision."""
    allowed: bool
    action: Literal["allow", "prompt", "deny"]
    reason: str


@dataclass(frozen=True, slots=True)
class PermissionPrompt:
    """Data model representing the permission prompt."""
    tool_name: str
    risk: str
    scope: str
    arguments: dict[str, JsonValue] = field(default_factory=dict, repr=False)
    workspace_root: str = field(default="", repr=False)
    tool_call_id: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arguments",
            snapshot_mapping(self.arguments, "arguments"),
        )
        if not isinstance(self.workspace_root, str):
            raise ValueError("workspace_root must be text")
        if not isinstance(self.tool_call_id, str):
            raise ValueError("tool_call_id must be text")


@dataclass(frozen=True, slots=True)
class ChildPermissionRequest:
    """Data model representing the child permission request."""
    agent_session_id: str
    root_session_id: str
    tool_name: str
    risk: str
    scope: str
    spec: ToolSpec = field(repr=False)
    call: ToolCall = field(repr=False)
    context: ToolContext = field(repr=False)

    def root_context(self) -> ToolContext:
        """Handle the root context operation."""
        return ToolContext(
            self.context.agent_session_id,
            self.context.workspace_id,
            self.context.workspace_root,
            metadata=dict(self.context.metadata),
            secret_environment_names=self.context.secret_environment_names,
            secret_values=self.context.secret_values,
        )

Prompt = Callable[[PermissionPrompt], Awaitable[PromptChoice | str] | PromptChoice | str]
HardGuard = Callable[[ToolSpec, ToolCall, ToolContext], Awaitable[bool] | bool]


class PermissionService:
    """Service providing the permission service operations."""
    def __init__(self, *, prompt: Prompt | None = None, hard_guard: HardGuard | None = None) -> None:
        self._prompt = prompt
        self._hard_guard = hard_guard
        self._session_approvals: set[tuple[str, str, str, str, str, bool]] = set()

    def classify(
        self,
        mode: PermissionMode | str,
        spec: ToolSpec,
        context: ToolContext | None = None,
    ) -> PermissionDecision:
        """Classify the requested operation."""
        selected = PermissionMode(mode)
        external = spec.permission_risk in {"external", "high"}
        mutation = spec.mutates_workspace
        confirmation = spec.requires_confirmation
        if selected is PermissionMode.READ_ONLY:
            action = "deny" if mutation or external or confirmation else "allow"
        elif selected is PermissionMode.ASK:
            action = "prompt" if mutation or external or confirmation else "allow"
        elif selected is PermissionMode.BYPASS:
            authorized = bool(context and context.metadata.get("bypass_authorized") is True)
            if confirmation:
                action = "prompt"
            else:
                action = "allow" if authorized or not (mutation or external) else "prompt"
        return PermissionDecision(action == "allow", action, _reason(action))

    async def decide(self, spec: ToolSpec, call: ToolCall, context: ToolContext) -> PermissionDecision:
        """Handle the decide operation."""
        if self._hard_guard is not None:
            try:
                guarded = self._hard_guard(spec, call, context)
                allowed = await guarded if inspect.isawaitable(guarded) else guarded
            except Exception:
                return PermissionDecision(False, "deny", "Mandatory safety guard failed")
            if allowed is not True:
                return PermissionDecision(False, "deny", "Denied by mandatory safety guard")

        mode = context.metadata.get("permission_mode", PermissionMode.ASK)
        try:
            classified = self.classify(str(mode), spec, context)
        except (TypeError, ValueError):
            return PermissionDecision(False, "deny", "Invalid permission mode")
        if classified.action != "prompt":
            return classified

        key = self._approval_key(spec, call, context)
        broker = context.parent_permission_broker
        if broker is not None:
            request = ChildPermissionRequest(
                context.agent_session_id,
                key[0],
                spec.name,
                str(spec.permission_risk),
                key[4],
                spec,
                call,
                context,
            )
            try:
                decision = broker.request_from_child(request)
                resolved = await decision if inspect.isawaitable(decision) else decision
            except Exception:
                return PermissionDecision(False, "deny", "Parent permission broker failed")
            if not isinstance(resolved, PermissionDecision):
                return PermissionDecision(False, "deny", "Parent permission broker failed")
            return resolved

        if not spec.requires_confirmation and key in self._session_approvals:
            return PermissionDecision(True, "allow", "Allowed for root session")
        if self._prompt is None:
            return PermissionDecision(False, "deny", "Permission confirmation unavailable")
        try:
            response = self._prompt(
                PermissionPrompt(
                    spec.name,
                    str(spec.permission_risk),
                    key[4],
                    _prompt_arguments(call, context),
                    workspace_root=_prompt_workspace_root(context),
                    tool_call_id=call.id,
                )
            )
            choice_value = await response if inspect.isawaitable(response) else response
        except Exception:
            return PermissionDecision(False, "deny", "Permission prompt failed")
        try:
            choice = PromptChoice(choice_value)
        except (TypeError, ValueError):
            return PermissionDecision(False, "deny", "Invalid permission response")
        if choice is PromptChoice.DENY:
            return PermissionDecision(False, "deny", "Permission denied")
        if (
            choice is PromptChoice.ALLOW_FOR_ROOT_SESSION
            and not spec.requires_confirmation
        ):
            self._session_approvals.add(key)
            return PermissionDecision(True, "allow", "Allowed for root session")
        return PermissionDecision(True, "allow", "Allowed once")

    def clear_root_session(self, root_session_id: str) -> None:
        """Clear the root session."""
        if not isinstance(root_session_id, str) or not root_session_id.strip():
            raise ValueError("root_session_id must not be empty")
        self._session_approvals = {item for item in self._session_approvals if item[0] != root_session_id}

    @staticmethod
    def _approval_key(spec: ToolSpec, call: ToolCall, context: ToolContext) -> tuple[str, str, str, str, str, bool]:
        root = context.metadata.get("root_session_id", context.agent_session_id)
        if not isinstance(root, str) or not root.strip():
            root = context.agent_session_id
        scope_kind = "workspace" if spec.permission_risk in {"safe", "workspace"} else "external"
        payload = json.dumps(call.arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        scope = f"{scope_kind}:{digest}"
        return (root, context.workspace_id, spec.name, str(spec.permission_risk), scope, spec.mutates_workspace)


class PermissionBroker:
    """Component responsible for the permission broker."""
    def __init__(self, service: PermissionService) -> None:
        if not isinstance(service, PermissionService):
            raise ValueError("service must be a PermissionService")
        self.service = service

    async def request_from_child(
        self, request: ChildPermissionRequest
    ) -> PermissionDecision:
        """Handle the request from child operation."""
        if not isinstance(request, ChildPermissionRequest):
            return PermissionDecision(False, "deny", "Invalid child permission request")
        return await self.service.decide(
            request.spec,
            request.call,
            request.root_context(),
        )

def _reason(action: str) -> str:
    return {"allow": "Allowed by permission policy", "prompt": "Permission confirmation required", "deny": "Denied by permission policy"}[action]


def _prompt_arguments(call: ToolCall, context: ToolContext) -> dict[str, JsonValue]:
    redacted = context.redactor.redact_data(call.arguments)
    if not isinstance(redacted, dict):
        return {}
    return snapshot_mapping(redacted, "arguments")


def _prompt_workspace_root(context: ToolContext) -> str:
    try:
        return str(context.workspace_root.expanduser().resolve())
    except OSError:
        return str(context.workspace_root.expanduser())
