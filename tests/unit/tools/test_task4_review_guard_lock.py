from __future__ import annotations

import asyncio

import pytest

from litecoder.tools import ToolCall, ToolSpec
from litecoder.tools.duplicate_guard import DuplicateGuard
from litecoder.tools.workspace_version import AsyncRWLock


def test_mutating_spec_rejects_shared_concurrency() -> None:
    with pytest.raises(ValueError, match="mutating tools require exclusive concurrency"):
        ToolSpec("write", "write", {}, True, concurrency="shared")

def test_mutating_spec_requires_workspace_lock() -> None:
    with pytest.raises(ValueError, match="require a workspace lock"):
        ToolSpec("write", "write", {}, True, workspace_lock=False)


@pytest.mark.asyncio
async def test_rw_lock_preserves_fifo_phases_and_batches_consecutive_readers() -> None:
    lock = AsyncRWLock()
    order: list[str] = []
    release_first = asyncio.Event()
    release_queued_readers = asyncio.Event()
    release_writer = asyncio.Event()
    first_entered = asyncio.Event()
    queued_readers_entered = asyncio.Event()
    writer_entered = asyncio.Event()
    queued_reader_count = 0

    async def first_reader() -> None:
        async with lock.write():
            order.append("r0")
            first_entered.set()
            await release_first.wait()

    async def queued_reader(name: str, started: asyncio.Event) -> None:
        nonlocal queued_reader_count
        started.set()
        async with lock.read():
            order.append(name)
            queued_reader_count += 1
            if queued_reader_count == 2:
                queued_readers_entered.set()
            await release_queued_readers.wait()

    async def writer(started: asyncio.Event) -> None:
        started.set()
        async with lock.write():
            order.append("w")
            writer_entered.set()
            await release_writer.wait()

    first = asyncio.create_task(first_reader())
    await first_entered.wait()
    r1_started = asyncio.Event()
    r2_started = asyncio.Event()
    writer_started = asyncio.Event()
    r1 = asyncio.create_task(queued_reader("r1", r1_started))
    await r1_started.wait()
    r2 = asyncio.create_task(queued_reader("r2", r2_started))
    await r2_started.wait()
    waiting_writer = asyncio.create_task(writer(writer_started))
    await writer_started.wait()
    release_first.set()
    await asyncio.wait_for(queued_readers_entered.wait(), timeout=0.1)
    assert order == ["r0", "r1", "r2"]
    release_queued_readers.set()
    await writer_entered.wait()
    release_writer.set()
    await asyncio.gather(first, r1, r2, waiting_writer)

@pytest.mark.asyncio
async def test_cancelled_middle_waiter_is_removed_without_barging() -> None:
    lock = AsyncRWLock()
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()
    order: list[str] = []

    async def holder() -> None:
        async with lock.write():
            holder_entered.set()
            await release_holder.wait()

    async def queued(name: str, started: asyncio.Event) -> None:
        started.set()
        async with lock.write():
            order.append(name)

    active = asyncio.create_task(holder())
    await holder_entered.wait()
    first_started = asyncio.Event()
    cancelled_started = asyncio.Event()
    last_started = asyncio.Event()
    first = asyncio.create_task(queued("first", first_started))
    await first_started.wait()
    cancelled = asyncio.create_task(queued("cancelled", cancelled_started))
    await cancelled_started.wait()
    last = asyncio.create_task(queued("last", last_started))
    await last_started.wait()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    release_holder.set()
    await asyncio.gather(active, first, last)
    assert order == ["first", "last"]


@pytest.mark.asyncio
async def test_duplicate_rounds_reject_decrease_prune_and_reset() -> None:
    guard = DuplicateGuard(annotation=lambda **_: None)
    spec = ToolSpec("read", "read", {}, False)
    for round_number in range(8):
        call = ToolCall(f"call-{round_number}", "read", {"round": round_number})
        await guard.record_success(
            "agent", "workspace", 0,
            round_number=round_number, call=call, preview=round_number, spec=spec,
        )
    assert guard.record_count == 5
    with pytest.raises(ValueError, match="round_number must be monotonic"):
        await guard.check(
            "agent", "workspace", 0,
            round_number=6, call=ToolCall("old", "read", {}), spec=spec,
        )
    await guard.start_user_message("agent")
    assert guard.record_count == 0
    assert await guard.check(
        "agent", "workspace", 0,
        round_number=0, call=ToolCall("reset", "read", {}), spec=spec,
    ) is None


@pytest.mark.asyncio
async def test_default_duplicate_annotation_requires_active_trace_context() -> None:
    guard = DuplicateGuard()
    call = ToolCall("one", "read", {})
    await guard.record_success("agent", "workspace", 0, round_number=1, call=call)
    with pytest.raises(RuntimeError, match="No active TraceContext"):
        await guard.check(
            "agent", "workspace", 0,
            round_number=1, call=ToolCall("two", "read", {}),
        )
