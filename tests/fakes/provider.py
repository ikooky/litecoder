from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace

from litecoder.providers.models import ModelRequest, ProviderEvent, ToolCallBlock


class FakeProvider:
    def __init__(self, script: list[list[ProviderEvent]]) -> None:
        self._script = tuple(
            tuple(_snapshot_event(event) for event in round_events)
            for round_events in script
        )
        self._next_round = 0
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ProviderEvent]:
        self.requests.append(request)
        if self._next_round >= len(self._script):
            raise RuntimeError("fake provider script exhausted")
        events = self._script[self._next_round]
        self._next_round += 1
        for event in events:
            yield event


def _snapshot_event(event: ProviderEvent) -> ProviderEvent:
    if not isinstance(event, ProviderEvent):
        raise ValueError("fake provider script entries must be ProviderEvent values")
    tool_call: ToolCallBlock | None = event.tool_call
    if tool_call is not None:
        tool_call = replace(tool_call)
    return replace(event, tool_call=tool_call)