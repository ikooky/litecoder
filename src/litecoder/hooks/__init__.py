"""Public interfaces for the hooks package."""

from litecoder.hooks.builtin import TraceHook
from litecoder.hooks.command import CommandHook
from litecoder.hooks.discovery import DiscoveredCommandHook, discover_command_hooks
from litecoder.hooks.manager import HookManager
from litecoder.hooks.models import (
    HookDiagnostic,
    HookEnvelope,
    HookOutcome,
    HookPoint,
)

__all__ = [
    "CommandHook",
    "DiscoveredCommandHook",
    "HookDiagnostic",
    "HookEnvelope",
    "HookManager",
    "HookOutcome",
    "HookPoint",
    "TraceHook",
    "discover_command_hooks",
]
