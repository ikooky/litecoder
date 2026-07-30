"""Provider registration and lookup."""

from __future__ import annotations

from collections.abc import Iterable

from litecoder.tools.models import Tool


class ToolRegistry:
    """Registry for the tool registry."""
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register the requested operation."""
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"Tool {name!r} is already registered")
        self._tools[name] = tool

    def register_many(self, tools: Iterable[Tool]) -> None:
        """Register the many."""
        for tool in tools:
            self.register(tool)

    def require(self, name: str) -> Tool:
        """Return the registered tool or raise when it is missing."""
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Tool {name!r} is not registered") from None

    def list(self) -> tuple[Tool, ...]:
        """Return the available entries."""
        return tuple(self._tools[name] for name in sorted(self._tools))
