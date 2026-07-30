"""Configured hook discovery."""

from __future__ import annotations

from dataclasses import dataclass

from litecoder.hooks.command import CommandHook
from litecoder.hooks.models import HookPoint
from litecoder.settings import Settings


@dataclass(frozen=True, slots=True)
class DiscoveredCommandHook:
    """Data model representing the discovered command hook."""
    name: str
    point: HookPoint
    hook: CommandHook


def discover_command_hooks(settings: Settings) -> tuple[DiscoveredCommandHook, ...]:
    """Build hook registrations from the explicit user configuration."""

    if not isinstance(settings, Settings):
        raise TypeError("settings must be Settings")
    return tuple(
        DiscoveredCommandHook(
            name=configured.name,
            point=HookPoint(configured.point),
            hook=CommandHook(configured),
        )
        for configured in settings.hooks if configured.enabled
    )