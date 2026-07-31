"""Workspace version and lock helpers."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass(slots=True)
class _Waiter:
    """Data model representing the waiter."""
    mode: str
    future: asyncio.Future[None]
    granted: bool = False


class AsyncRWLock:
    """Component responsible for the async rw lock."""
    def __init__(self) -> None:
        self._mutex = asyncio.Lock()
        self._readers = 0
        self._writer = False
        self._waiters: deque[_Waiter] = deque()

    @asynccontextmanager
    async def read(self) -> AsyncIterator[None]:
        """Read the requested data."""
        waiter = await self._acquire("read")
        try:
            yield
        finally:
            await self._release_shielded(waiter)

    @asynccontextmanager
    async def write(self) -> AsyncIterator[None]:
        """Write the supplied data."""
        waiter = await self._acquire("write")
        try:
            yield
        finally:
            await self._release_shielded(waiter)

    async def _acquire(self, mode: str) -> _Waiter:
        loop = asyncio.get_running_loop()
        waiter = _Waiter(mode, loop.create_future())
        async with self._mutex:
            if mode == "read" and self._readers and not self._waiters:
                waiter.granted = True
                self._readers += 1
                waiter.future.set_result(None)
            else:
                self._waiters.append(waiter)
                self._wake_next()
        try:
            await waiter.future
            return waiter
        except asyncio.CancelledError:
            async with self._mutex:
                if waiter.granted:
                    self._release_owned(waiter)
                else:
                    try:
                        self._waiters.remove(waiter)
                    except ValueError:
                        pass
                self._wake_next()
            raise

    async def _release_shielded(self, waiter: _Waiter) -> None:
        task = asyncio.create_task(self._release(waiter))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
    async def _release(self, waiter: _Waiter) -> None:
        async with self._mutex:
            self._release_owned(waiter)
            self._wake_next()

    def _release_owned(self, waiter: _Waiter) -> None:
        """Release the owned."""
        if not waiter.granted:
            raise RuntimeError("lock waiter does not own the lock")
        waiter.granted = False
        if waiter.mode == "read":
            if self._readers <= 0:
                raise RuntimeError("reader ownership underflow")
            self._readers -= 1
        else:
            if not self._writer:
                raise RuntimeError("writer ownership underflow")
            self._writer = False

    def _wake_next(self) -> None:
        if self._writer or self._readers or not self._waiters:
            return
        first = self._waiters[0]
        if first.mode == "write":
            waiter = self._waiters.popleft()
            waiter.granted = True
            self._writer = True
            if not waiter.future.done():
                waiter.future.set_result(None)
            return
        while self._waiters and self._waiters[0].mode == "read":
            waiter = self._waiters.popleft()
            waiter.granted = True
            self._readers += 1
            if not waiter.future.done():
                waiter.future.set_result(None)


@dataclass(slots=True)
class WorkspaceState:
    """Data model representing the workspace state."""
    version: int = 0
    lock: AsyncRWLock = field(default_factory=AsyncRWLock)
    traversal_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class WorkspaceStateRegistry:
    """Registry for the workspace state registry."""
    def __init__(self) -> None:
        self._states: dict[str, WorkspaceState] = {}
        self._lock = threading.Lock()

    def get(self, workspace_id: str) -> WorkspaceState:
        """Return the requested value."""
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("workspace_id must not be empty")
        with self._lock:
            return self._states.setdefault(workspace_id, WorkspaceState())
