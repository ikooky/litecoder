from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from litecoder.common.locks import (
    LockAcquisitionCancelled,
    NamedFileLock,
    ProjectAlreadyActive,
    ProjectRuntimeLock,
    ResourceLockUnavailable,
    SessionAlreadyActive,
)
from litecoder.paths import canonical_path


def test_named_file_lock_fails_fast_when_configured(tmp_path: Path) -> None:
    with NamedFileLock("tasks-project-1", tmp_path, fail_when_locked=True):
        with pytest.raises(ResourceLockUnavailable) as error:
            with NamedFileLock("tasks-project-1", tmp_path, fail_when_locked=True):
                pass

    assert error.value.resource_type == "resource"
    assert error.value.path == tmp_path / "litecoder-tasks-project-1.lock"


@pytest.mark.asyncio
async def test_named_file_lock_waits_without_blocking_event_loop(
    tmp_path: Path,
) -> None:
    first = NamedFileLock("startup-project-1", tmp_path, timeout=1.0)
    second = NamedFileLock("startup-project-1", tmp_path, timeout=1.0)
    ticks: list[str] = []

    async with first.acquired_async():
        waiter = asyncio.create_task(second.acquire_async())
        ticker = asyncio.create_task(_tick_until(waiter, ticks))
        await asyncio.sleep(0.05)
        assert ticks
        first.release()
        await waiter
        await ticker

    second.release()


@pytest.mark.asyncio
async def test_named_file_lock_releases_after_cancelled_async_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    holder = NamedFileLock("startup-project-1", tmp_path, timeout=1.0)
    contender = NamedFileLock("startup-project-1", tmp_path, timeout=1.0)
    acquisition_started = threading.Event()
    acquisition_finished = threading.Event()
    original_acquire = contender.acquire

    def acquire_with_signals() -> NamedFileLock:
        acquisition_started.set()
        try:
            return original_acquire()
        finally:
            acquisition_finished.set()

    monkeypatch.setattr(contender, "acquire", acquire_with_signals)
    holder.acquire()
    task = asyncio.create_task(_acquire_lock(contender))

    try:
        await asyncio.to_thread(acquisition_started.wait)
        task.cancel()
        await asyncio.sleep(0)
        holder.release()

        with pytest.raises(asyncio.CancelledError):
            await task

        await asyncio.to_thread(acquisition_finished.wait)
        with NamedFileLock("startup-project-1", tmp_path, fail_when_locked=True):
            pass
    finally:
        holder.release()
        contender.release()


@pytest.mark.asyncio
async def test_named_file_lock_releases_after_repeated_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    holder = NamedFileLock("startup-project-1", tmp_path, timeout=1.0)
    contender = NamedFileLock("startup-project-1", tmp_path, timeout=1.0)
    acquisition_started = threading.Event()
    acquisition_finished = threading.Event()
    original_acquire = contender.acquire

    def acquire_with_signals() -> NamedFileLock:
        acquisition_started.set()
        try:
            return original_acquire()
        finally:
            acquisition_finished.set()

    monkeypatch.setattr(contender, "acquire", acquire_with_signals)
    holder.acquire()
    task = asyncio.create_task(_acquire_lock(contender))

    try:
        await asyncio.to_thread(acquisition_started.wait)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        holder.release()

        with pytest.raises(asyncio.CancelledError):
            await task

        await asyncio.to_thread(acquisition_finished.wait)
        with NamedFileLock("startup-project-1", tmp_path, fail_when_locked=True):
            pass
    finally:
        holder.release()
        contender.release()


@pytest.mark.asyncio
async def test_named_file_lock_releases_after_repeated_cancellation_on_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock = NamedFileLock("startup-project-1", tmp_path, timeout=1.0)
    contender = NamedFileLock(
        "startup-project-1", tmp_path, fail_when_locked=True
    )
    entered = asyncio.Event()
    worker_started = threading.Event()
    worker_release = threading.Event()
    release_started = threading.Event()
    original_release = lock.release
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    original_run_in_executor = loop.run_in_executor

    def run_in_executor(
        selected_executor: ThreadPoolExecutor | None,
        function: object,
        *args: object,
    ) -> asyncio.Future[object]:
        return original_run_in_executor(
            executor if selected_executor is None else selected_executor,
            function,
            *args,
        )

    def occupy_worker() -> None:
        worker_started.set()
        worker_release.wait()

    def release_with_signal() -> None:
        release_started.set()
        original_release()

    monkeypatch.setattr(loop, "run_in_executor", run_in_executor)
    monkeypatch.setattr(lock, "release", release_with_signal)
    task = asyncio.create_task(_hold_lock(lock, entered))
    blocker: asyncio.Task[None] | None = None

    try:
        await entered.wait()
        blocker = asyncio.create_task(asyncio.to_thread(occupy_worker))
        await _wait_for_thread_event(worker_started)

        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not release_started.is_set()

        task.cancel()
        worker_release.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert release_started.is_set()
        with contender:
            pass
    finally:
        worker_release.set()
        if blocker is not None:
            await blocker
        lock.release()
        contender.release()
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_cancellation_during_async_exit_propagates_after_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock = NamedFileLock("startup-project-1", tmp_path, timeout=1.0)
    contender = NamedFileLock(
        "startup-project-1", tmp_path, fail_when_locked=True
    )
    entered = asyncio.Event()
    leave_body = asyncio.Event()
    release_started = threading.Event()
    continue_release = threading.Event()
    original_release = lock.release

    def blocking_release() -> None:
        release_started.set()
        continue_release.wait()
        original_release()

    monkeypatch.setattr(lock, "release", blocking_release)
    task = asyncio.create_task(_leave_lock(lock, entered, leave_body))

    try:
        await entered.wait()
        leave_body.set()
        await _wait_for_thread_event(release_started)
        task.cancel()
        continue_release.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        with contender:
            pass
    finally:
        continue_release.set()
        lock.release()
        contender.release()


@pytest.mark.asyncio
async def test_cancelled_fail_fast_acquisition_preserves_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    holder = NamedFileLock("session-tree-session-1", tmp_path)
    contender = NamedFileLock(
        "session-tree-session-1",
        tmp_path,
        fail_when_locked=True,
    )
    acquisition_started = threading.Event()
    continue_acquisition = threading.Event()
    original_acquire = contender.acquire

    def acquire_after_signal() -> NamedFileLock:
        acquisition_started.set()
        continue_acquisition.wait()
        return original_acquire()

    monkeypatch.setattr(contender, "acquire", acquire_after_signal)
    holder.acquire()
    task = asyncio.create_task(_acquire_lock(contender))

    try:
        await _wait_for_thread_event(acquisition_started)
        task.cancel()
        continue_acquisition.set()

        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        continue_acquisition.set()
        holder.release()
        contender.release()


@pytest.mark.asyncio
async def test_repeated_cancellation_preserves_fail_fast_acquisition_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    holder = NamedFileLock("session-tree-session-1", tmp_path)
    contender = NamedFileLock.session_tree("session-1", tmp_path)
    acquisition_started = threading.Event()
    continue_acquisition = threading.Event()
    release_called = threading.Event()
    original_acquire = contender.acquire
    original_release = contender.release

    def acquire_after_signal() -> NamedFileLock:
        acquisition_started.set()
        continue_acquisition.wait()
        return original_acquire()

    def release_with_signal() -> None:
        release_called.set()

    monkeypatch.setattr(contender, "acquire", acquire_after_signal)
    monkeypatch.setattr(contender, "release", release_with_signal)
    holder.acquire()
    task = asyncio.create_task(_acquire_lock(contender))

    try:
        await _wait_for_thread_event(acquisition_started)
        task.cancel()
        continue_acquisition.set()

        with pytest.raises(asyncio.CancelledError) as cancellation:
            await task

        assert isinstance(cancellation.value, LockAcquisitionCancelled)
        assert isinstance(
            cancellation.value.acquisition_error, SessionAlreadyActive
        )
        assert not release_called.is_set()
    finally:
        continue_acquisition.set()
        holder.release()
        original_release()


async def _acquire_lock(lock: NamedFileLock) -> None:
    async with lock.acquired_async():
        pytest.fail("cancelled acquisition unexpectedly entered the context")


async def _leave_lock(
    lock: NamedFileLock,
    entered: asyncio.Event,
    leave_body: asyncio.Event,
) -> None:
    async with lock.acquired_async():
        entered.set()
        await leave_body.wait()


async def _hold_lock(lock: NamedFileLock, entered: asyncio.Event) -> None:
    async with lock.acquired_async():
        entered.set()
        await asyncio.Event().wait()


async def _wait_for_thread_event(event: threading.Event) -> None:
    await asyncio.wait_for(_poll_thread_event(event), timeout=1)


async def _poll_thread_event(event: threading.Event) -> None:
    while not event.is_set():
        await asyncio.sleep(0)


async def _tick_until(task: asyncio.Task[object], ticks: list[str]) -> None:
    while not task.done():
        ticks.append("tick")
        await asyncio.sleep(0)


def test_session_tree_lock_raises_specific_error(tmp_path: Path) -> None:
    with NamedFileLock.session_tree("session-1", tmp_path):
        with pytest.raises(SessionAlreadyActive):
            with NamedFileLock.session_tree("session-1", tmp_path):
                pass


def test_project_runtime_lock_remains_fail_fast_compatibility(
    tmp_path: Path,
) -> None:
    with ProjectRuntimeLock("project-1", tmp_path):
        with pytest.raises(ProjectAlreadyActive):
            with ProjectRuntimeLock("project-1", tmp_path):
                pass


@pytest.mark.asyncio
async def test_build_runtime_allows_two_active_runtimes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from litecoder.cli import app as app_module
    from litecoder.paths import AppPaths

    user_dir = tmp_path / ".litecoder"
    user_dir.mkdir()
    (user_dir / "config.toml").write_text(
        'default_provider = "fake"\n'
        'default_model = "model"\n'
        '[providers.fake]\n'
        'type = "openai-chat-completions"\n'
        'model = "model"\n'
        'api_key = "key"\n',
        encoding="utf-8",
    )
    paths = AppPaths(
        user_dir=user_dir,
        sessions_db=user_dir / "sessions.db",
        project_id="project-1",
        project_dir=user_dir / "projects" / "project-1",
        workspace_id="workspace-1",
        workspace_root=tmp_path,
    )
    monkeypatch.setattr(app_module.AppPaths, "discover", lambda _cwd: paths)

    first = await app_module.build_runtime(tmp_path)
    second = await app_module.build_runtime(tmp_path)
    try:
        assert first is not second
        assert first.worktree_manager.worktree_root == canonical_path(
            paths.workspace_root / ".worktrees"
        )
    finally:
        await second.close()
        await first.close()
