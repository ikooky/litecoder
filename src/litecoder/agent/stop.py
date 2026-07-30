"""Stop reasons and policies for agent turns."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from litecoder.providers.models import StopReason


@dataclass(frozen=True, slots=True)
class StopOutcome:
    """Data model representing the stop outcome."""
    status: str
    retry: bool = False
    consumes_continuation: bool = False


class StopPolicy:
    """Component responsible for the stop policy."""
    def __init__(self, accepted_stop_sequences: Iterable[str] = ("stop_sequence",)) -> None:
        self._accepted_stop_sequences = frozenset(accepted_stop_sequences)

    def decide(self, reason: StopReason, raw: str | None = None) -> StopOutcome:
        """Handle the decide operation."""
        if reason is StopReason.END_TURN:
            return StopOutcome("completed")
        if reason is StopReason.STOP_SEQUENCE:
            return StopOutcome(
                "completed" if raw in self._accepted_stop_sequences else "failed"
            )
        if reason is StopReason.TOOL_USE:
            return StopOutcome("continue_tools")
        if reason is StopReason.PAUSE_TURN:
            return StopOutcome("continue_provider", consumes_continuation=True)
        if reason is StopReason.MAX_TOKENS:
            return StopOutcome("continue_provider", consumes_continuation=True)
        if reason is StopReason.CONTEXT_EXHAUSTED:
            return StopOutcome("incomplete")
        if reason is StopReason.REFUSAL:
            return StopOutcome("refused")
        return StopOutcome("failed")

    def classify(self, reason: StopReason) -> StopOutcome:
        """Compatibility alias for early milestone callers."""
        return self.decide(reason, raw=reason.value)
