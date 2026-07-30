from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from litecoder.common.locks import (
    LockAcquisitionCancelled,
    NamedFileLock,
    ResourceLockUnavailable,
)


@pytest.mark.asyncio
async def test_same_lock_sync_contender_waits_for_async_context_exit(
    tmp_path: Path,
) -> None:
    lock = NamedFileLock("shared", tmp_path, timeout=1.0)
    entered = asyncio.Event()
    release_holder = asyncio.Event()
    contender_started = threading.Event()
    contender_entered = threading.Event()

    async def hold_async() -> None:
        async with lock.acquired_async():
            entered.set()
            await release_holder.wait()

    def contend_sync() -> None:
        contender_started.set()
        with lock:
            contender_entered.set()

    holder = asyncio.create_task(hold_async())
    await entered.wait()
    contender = asyncio.create_task(asyncio.to_thread(contend_sync))
    await asyncio.to_thread(contender_started.wait, 1.0)

    entered_before_release = await asyncio.to_thread(
        contender_entered.wait, 0.2
    )
    release_holder.set()
    await asyncio.gather(holder, contender)

    assert entered_before_release is False
    assert contender_entered.is_set()


@pytest.mark.asyncio
async def test_cancelled_same_instance_contender_does_not_release_holder(
    tmp_path: Path,
) -> None:
    lock = NamedFileLock("shared", tmp_path, timeout=0.1)
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()

    async def hold() -> None:
        async with lock.acquired_async():
            holder_entered.set()
            await release_holder.wait()

    async def contend() -> None:
        async with lock.acquired_async():
            pytest.fail("same-instance contender unexpectedly acquired")

    holder = asyncio.create_task(hold())
    await holder_entered.wait()
    contender = asyncio.create_task(contend())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    contender.cancel()

    independent = NamedFileLock(
        "shared",
        tmp_path,
        timeout=0,
        fail_when_locked=True,
    )
    try:
        with pytest.raises(LockAcquisitionCancelled) as exc_info:
            await contender
        acquisition_error = exc_info.value.acquisition_error
        assert isinstance(acquisition_error, ResourceLockUnavailable)
        assert exc_info.value.__cause__ is acquisition_error

        with pytest.raises(ResourceLockUnavailable):
            with independent:
                pass
    finally:
        release_holder.set()
        await holder

    with independent:
        pass
