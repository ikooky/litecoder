from __future__ import annotations

import asyncio

import pytest

from litecoder.tools.workspace_version import AsyncRWLock, WorkspaceStateRegistry


@pytest.mark.asyncio
async def test_concurrent_readers_enter_together() -> None:
    lock = AsyncRWLock()
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    release = asyncio.Event()

    async def reader(entered: asyncio.Event) -> None:
        async with lock.read():
            entered.set()
            await release.wait()

    first = asyncio.create_task(reader(first_entered))
    await first_entered.wait()
    second = asyncio.create_task(reader(second_entered))
    await second_entered.wait()
    release.set()
    await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_writer_is_exclusive_and_waiting_writer_blocks_new_readers() -> None:
    lock = AsyncRWLock()
    initial_reader_entered = asyncio.Event()
    release_initial_reader = asyncio.Event()
    writer_started = asyncio.Event()
    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()
    late_reader_started = asyncio.Event()
    late_reader_entered = asyncio.Event()

    async def initial_reader() -> None:
        async with lock.read():
            initial_reader_entered.set()
            await release_initial_reader.wait()

    async def writer() -> None:
        writer_started.set()
        async with lock.write():
            writer_entered.set()
            await release_writer.wait()

    async def late_reader() -> None:
        late_reader_started.set()
        async with lock.read():
            late_reader_entered.set()

    first = asyncio.create_task(initial_reader())
    await initial_reader_entered.wait()
    waiting_writer = asyncio.create_task(writer())
    await writer_started.wait()
    late = asyncio.create_task(late_reader())
    await late_reader_started.wait()
    assert not writer_entered.is_set()
    assert not late_reader_entered.is_set()

    release_initial_reader.set()
    await writer_entered.wait()
    assert not late_reader_entered.is_set()
    release_writer.set()
    await late_reader_entered.wait()
    await asyncio.gather(first, waiting_writer, late)


@pytest.mark.asyncio
async def test_cancelling_waiting_writer_does_not_strand_readers() -> None:
    lock = AsyncRWLock()
    reader_entered = asyncio.Event()
    release_reader = asyncio.Event()
    writer_started = asyncio.Event()

    async def holding_reader() -> None:
        async with lock.read():
            reader_entered.set()
            await release_reader.wait()

    async def waiting_writer() -> None:
        writer_started.set()
        async with lock.write():
            raise AssertionError("cancelled writer entered")

    reader = asyncio.create_task(holding_reader())
    await reader_entered.wait()
    writer = asyncio.create_task(waiting_writer())
    await writer_started.wait()
    writer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writer
    release_reader.set()
    await reader

    entered_after_cancel = asyncio.Event()
    async with lock.read():
        entered_after_cancel.set()
    assert entered_after_cancel.is_set()


@pytest.mark.asyncio
async def test_cancelling_lock_holder_releases_ownership() -> None:
    lock = AsyncRWLock()
    writer_entered = asyncio.Event()

    async def holding_writer() -> None:
        async with lock.write():
            writer_entered.set()
            await asyncio.Event().wait()

    writer = asyncio.create_task(holding_writer())
    await writer_entered.wait()
    writer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writer

    async with lock.read():
        pass


def test_workspace_states_share_only_by_exact_workspace_id() -> None:
    registry = WorkspaceStateRegistry()
    first = registry.get("workspace-a")
    same = registry.get("workspace-a")
    other = registry.get("workspace-b")

    assert first is same
    assert first is not other
    assert first.lock is same.lock
    assert first.lock is not other.lock
    assert first.version == other.version == 0
    first.version += 1
    assert same.version == 1
    assert other.version == 0


def test_workspace_registry_rejects_empty_id_without_echoing_input() -> None:
    with pytest.raises(ValueError, match="workspace_id must not be empty"):
        WorkspaceStateRegistry().get("")
