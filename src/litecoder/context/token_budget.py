"""Token budget calculations and limits."""

from __future__ import annotations

from dataclasses import dataclass


def _non_negative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class TokenAllocation:
    """Data model representing the token allocation."""
    system: int
    tools: int
    memories: int
    recent: int
    reserve: int
    truncated: bool

    def __post_init__(self) -> None:
        for field_name in ("system", "tools", "memories", "recent", "reserve"):
            _non_negative_integer(getattr(self, field_name), field_name)
        if not isinstance(self.truncated, bool):
            raise ValueError("truncated must be a bool")


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """Data model representing the token budget."""
    total: int
    reserve: int

    def __post_init__(self) -> None:
        total = _non_negative_integer(self.total, "total")
        reserve = _non_negative_integer(self.reserve, "reserve")
        if reserve > total:
            raise ValueError("reserve must not exceed total")

    def allocate(
        self, *, system: int, tools: int, memories: int, recent: int
    ) -> TokenAllocation:
        """Allocate the requested token budget."""
        system = _non_negative_integer(system, "system")
        tools = _non_negative_integer(tools, "tools")
        memories = _non_negative_integer(memories, "memories")
        recent = _non_negative_integer(recent, "recent")
        available = max(
            0, self.total - self.reserve - system - tools - memories
        )
        accepted_recent = min(recent, available)
        return TokenAllocation(
            system=system,
            tools=tools,
            memories=memories,
            recent=accepted_recent,
            reserve=self.reserve,
            truncated=accepted_recent < recent,
        )


def estimate_tokens(text: str) -> int:
    """Handle the estimate tokens operation."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    byte_count = len(text.encode("utf-8"))
    return (byte_count + 3) // 4
