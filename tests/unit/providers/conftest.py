from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from copy import deepcopy

import pytest


class StubStream:
    def __init__(
        self,
        events: Iterable[object],
        *,
        iteration_error: Exception | None = None,
    ) -> None:
        self._events = list(events)
        self._iteration_error = iteration_error
        self.entered = False
        self.closed = False

    async def __aenter__(self) -> StubStream:
        self.entered = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[object]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[object]:
        for event in self._events:
            yield event
        if self._iteration_error is not None:
            raise self._iteration_error


class StubCall:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.calls: list[dict[str, object]] = []
        self.setup_error: Exception | None = None
        self.iteration_error: Exception | None = None
        self.last_stream: StubStream | None = None

    async def __call__(self, **kwargs: object) -> StubStream:
        self.calls.append(deepcopy(kwargs))
        if self.setup_error is not None:
            raise self.setup_error
        self.last_stream = StubStream(
            self.events,
            iteration_error=self.iteration_error,
        )
        return self.last_stream


@pytest.fixture
def completion_call() -> StubCall:
    return StubCall()


@pytest.fixture
def responses_call() -> StubCall:
    return StubCall()
