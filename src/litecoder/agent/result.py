"""Result model returned by an agent run."""

from __future__ import annotations

from dataclasses import dataclass

from litecoder.providers.models import Usage


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Immutable summary of a completed, stopped, or failed agent session."""
    session_id: str
    status: str
    reason: str
    usage: Usage

    @property
    def completed(self) -> bool:
        """Return whether the agent finished normally."""
        return self.status == "completed"
