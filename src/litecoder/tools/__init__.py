"""Public interfaces for the tools package."""

from litecoder.tools.background import (
    BackgroundHandle,
    BackgroundManager,
    BackgroundState,
    BackgroundStatus,
    RuntimeNotification,
    register_background_tools,
)
from litecoder.tools.duplicate_guard import DuplicateGuard
from litecoder.tools.executor import ToolExecutor
from litecoder.tools.models import (
    Tool,
    ToolCall,
    ToolContext,
    ToolDenied,
    ToolExecution,
    ToolFailure,
    ToolPartialFailure,
    ToolResult,
    ToolSpec,
)
from litecoder.tools.permission import (
    ChildPermissionRequest,
    PermissionBroker,
    PermissionDecision,
    PermissionMode,
    PermissionService,
    PromptChoice,
)
from litecoder.tools.registry import ToolRegistry
from litecoder.tools.workspace_version import WorkspaceState, WorkspaceStateRegistry

__all__ = [
    "register_background_tools",
    "RuntimeNotification",
    "BackgroundStatus",
    "BackgroundState",
    "BackgroundManager",
    "BackgroundHandle",
    "DuplicateGuard",
    "ChildPermissionRequest",
    "PermissionBroker",
    "PermissionDecision",
    "PermissionMode",
    "PermissionService",
    "PromptChoice",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolDenied",
    "ToolExecution",
    "ToolExecutor",
    "ToolFailure",
    "ToolPartialFailure",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "WorkspaceState",
    "WorkspaceStateRegistry",
]
