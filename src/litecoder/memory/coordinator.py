"""Background coordination for memory workflows."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from litecoder.context.session.models import MessageRecord
from litecoder.memory.consolidation import DREAM_THRESHOLD
from litecoder.memory.diagnostics import memory_diagnostic
from litecoder.memory.service import MemoryService

Diagnostic = Callable[[dict[str, object]], Awaitable[None]]


@dataclass(slots=True)
class _LifecycleProgress:
    """Data model representing the lifecycle progress."""
    operation: Literal["extract", "dream"] = "extract"
    events: list[dict[str, object]] = field(default_factory=list)


class MemoryCoordinator:
    """Run memory extraction and consolidation serially in the background."""

    def __init__(self, *, timeout: float = 30.0, close_timeout: float = 30.0) -> None:
        self.timeout = timeout
        self.close_timeout = close_timeout
        self._tail: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._closing = False

    def submit(
        self,
        service: MemoryService,
        session_id: str,
        messages: Sequence[MessageRecord],
        diagnostic: Diagnostic,
    ) -> None:
        """Submit a background memory job."""
        if self._closing:
            return

        previous = self._tail
        task = asyncio.create_task(
            self._run_after(previous, service, session_id, deepcopy(tuple(messages)), diagnostic)
        )
        self._tail = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_after(
        self,
        previous: asyncio.Task[None] | None,
        service: MemoryService,
        session_id: str,
        messages: tuple[MessageRecord, ...],
        diagnostic: Diagnostic,
    ) -> None:
        """Run the after."""
        if previous is not None:
            try:
                await previous
            except Exception:
                pass

        progress = _LifecycleProgress()
        job = asyncio.create_task(
            self._run_job(service, session_id, messages, progress)
        )
        try:
            done, _ = await asyncio.wait({job}, timeout=self.timeout)
        except asyncio.CancelledError:
            _cancel_and_consume(job)
            raise
        if not done:
            _cancel_and_consume(job)
            completed_events = tuple(progress.events)
            for event in completed_events:
                await self._emit(diagnostic, event)
            await self._emit(
                diagnostic,
                memory_diagnostic(progress.operation, "timeout"),
            )
            return

        try:
            job.result()
        except Exception:
            if not progress.events:
                await self._emit(
                    diagnostic,
                    memory_diagnostic("extract", "failed"),
                )
            return
        for event in progress.events:
            await self._emit(diagnostic, event)

    async def _run_job(
        self,
        service: MemoryService,
        session_id: str,
        messages: tuple[MessageRecord, ...],
        progress: _LifecycleProgress,
    ) -> None:
        """Run the job."""
        extracted = await service.extract_memories(session_id, messages)
        progress.events.append(memory_diagnostic(
            "extract",
            getattr(extracted, "status", "failed"),
            accepted=getattr(extracted, "accepted", None),
            rejected=getattr(extracted, "rejected", None),
            written=getattr(extracted, "written", None),
            code=getattr(extracted, "provider_code", None),
            limit=getattr(extracted, "limit", None),
        ))
        if not _should_dream(extracted):
            return

        progress.operation = "dream"
        try:
            dreamed = await service.consolidate_memories()
        except Exception:
            progress.events.append(memory_diagnostic("dream", "failed"))
            return

        status = getattr(dreamed, "status", "failed")
        if status == "skipped":
            return
        progress.events.append(memory_diagnostic(
            "dream",
            status,
            before=getattr(dreamed, "before", None),
            after=getattr(dreamed, "after", None),
        ))

    async def _emit(self, diagnostic: Diagnostic, event: dict[str, object]) -> None:
        try:
            await diagnostic(event)
        except Exception:
            pass

    async def close(self) -> None:
        """Close the managed resource and release any lock."""
        self._closing = True
        pending = tuple(self._tasks)
        if not pending:
            return

        _, unfinished = await asyncio.wait(pending, timeout=self.close_timeout)
        for task in unfinished:
            _cancel_and_consume(task)


def _should_dream(extracted: object) -> bool:
    status = getattr(extracted, "status", None)
    written = getattr(extracted, "written", 0)
    total = getattr(extracted, "total", 0)
    return (
        status in {"completed", "partial_rejected"}
        and isinstance(written, int)
        and not isinstance(written, bool)
        and written > 0
        and isinstance(total, int)
        and not isinstance(total, bool)
        and total >= DREAM_THRESHOLD
    )


def _cancel_and_consume(task: asyncio.Task[object]) -> None:
    task.cancel()
    task.add_done_callback(_consume_task)


def _consume_task(task: asyncio.Task[object]) -> None:
    try:
        task.result()
    except BaseException:
        pass
