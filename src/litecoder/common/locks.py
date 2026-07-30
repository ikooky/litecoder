"""Cross-process locking helpers for shared runtime state."""

from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, IO

import portalocker


_SAFE_LOCK_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ResourceLockUnavailable(RuntimeError):
    """Component responsible for the resource lock unavailable."""
    def __init__(self, resource_type: str, path: Path) -> None:
        super().__init__(f"{resource_type} is already active: {path}")
        self.resource_type = resource_type
        self.path = path


class LockAcquisitionCancelled(asyncio.CancelledError):
    """Component responsible for the lock acquisition cancelled."""
    def __init__(
        self,
        cancellation: asyncio.CancelledError,
        acquisition_error: BaseException,
    ) -> None:
        super().__init__(*cancellation.args)
        self.acquisition_error = acquisition_error


class SessionAlreadyActive(ResourceLockUnavailable):
    """Component responsible for the session already active."""
    def __init__(self, path: Path) -> None:
        super().__init__("session", path)


class ProjectAlreadyActive(ResourceLockUnavailable):
    """Component responsible for the project already active."""
    def __init__(self, path: Path) -> None:
        super().__init__("project runtime", path)


class NamedFileLock:
    """Component responsible for the named file lock."""
    def __init__(
        self,
        name: str,
        lock_root: Path,
        *,
        resource_type: str = "resource",
        timeout: float = 10.0,
        fail_when_locked: bool = False,
        error_type: type[ResourceLockUnavailable] = ResourceLockUnavailable,
    ) -> None:
        if not isinstance(name, str) or not _SAFE_LOCK_NAME.fullmatch(name):
            raise ValueError("lock name is invalid")
        if not isinstance(lock_root, Path):
            raise ValueError("lock root must be a Path")
        if not isinstance(resource_type, str) or not resource_type.strip():
            raise ValueError("resource type is invalid")
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        self.name = name
        self.resource_type = resource_type
        self.path = lock_root / f"litecoder-{name}.lock"
        self.timeout = timeout
        self.fail_when_locked = fail_when_locked
        self.error_type = error_type
        self._lock: portalocker.Lock | None = None
        self._handle: IO[str] | None = None
        self._ownership_lock = threading.Lock()

    @classmethod
    def startup(cls, project_id: str, lock_root: Path) -> NamedFileLock:
        """Handle the startup operation."""
        return cls(f"startup-{project_id}", lock_root, resource_type="startup")

    @classmethod
    def session_tree(cls, root_session_id: str, lock_root: Path) -> NamedFileLock:
        """Handle the session tree operation."""
        return cls(
            f"session-tree-{root_session_id}",
            lock_root,
            resource_type="session",
            timeout=0,
            fail_when_locked=True,
            error_type=SessionAlreadyActive,
        )

    @classmethod
    def tasks(cls, project_id: str, lock_root: Path) -> NamedFileLock:
        """Handle the tasks operation."""
        return cls(f"tasks-{project_id}", lock_root, resource_type="tasks")

    @classmethod
    def memory(cls, project_id: str, lock_root: Path) -> NamedFileLock:
        """Handle the memory operation."""
        return cls(f"memory-{project_id}", lock_root, resource_type="memory")

    @classmethod
    def command_audit(cls, project_id: str, lock_root: Path) -> NamedFileLock:
        """Handle the command audit operation."""
        return cls(
            f"command-audit-{project_id}",
            lock_root,
            resource_type="command audit",
        )

    @classmethod
    def workspace(cls, workspace_id: str, lock_root: Path) -> NamedFileLock:
        """Handle the workspace operation."""
        return cls(
            f"workspace-{workspace_id}",
            lock_root,
            resource_type="workspace",
        )

    def _unavailable_error(self) -> ResourceLockUnavailable:
        if self.error_type is ResourceLockUnavailable:
            return self.error_type(self.resource_type, self.path)
        return self.error_type(self.path)

    def acquire(self) -> NamedFileLock:
        """Handle the acquire operation."""
        deadline = time.monotonic() + self.timeout
        if self.fail_when_locked or self.timeout == 0:
            ownership_acquired = self._ownership_lock.acquire(blocking=False)
        else:
            ownership_acquired = self._ownership_lock.acquire(
                timeout=self.timeout
            )
        if not ownership_acquired:
            raise self._unavailable_error()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            remaining = max(0.0, deadline - time.monotonic())
            lock = portalocker.Lock(
                self.path,
                mode="a",
                timeout=remaining,
                fail_when_locked=self.fail_when_locked,
            )
            handle = lock.acquire()
        except portalocker.exceptions.LockException as error:
            self._ownership_lock.release()
            raise self._unavailable_error() from error
        except BaseException:
            self._ownership_lock.release()
            raise
        self._lock = lock
        self._handle = handle
        return self

    async def acquire_async(self) -> NamedFileLock:
        """Handle the acquire async operation."""
        return await asyncio.to_thread(self.acquire)

    def release(self) -> None:
        """Release the managed resource."""
        lock = self._lock
        if lock is None:
            return
        try:
            lock.release()
        finally:
            self._lock = None
            self._handle = None
            self._ownership_lock.release()

    @asynccontextmanager
    async def acquired_async(self) -> AsyncIterator[NamedFileLock]:
        """Handle the acquired async operation."""
        acquisition = asyncio.create_task(self.acquire_async())
        acquired = False
        try:
            await asyncio.shield(acquisition)
            acquired = True
        except asyncio.CancelledError as cancellation:
            acquisition_error: BaseException | None = None
            try:
                await _await_cancellation_resilient(acquisition)
                acquired = True
            except BaseException as error:
                acquisition_error = error
            release_cancellation: asyncio.CancelledError | None = None
            if acquired:
                try:
                    await self._release_async()
                except asyncio.CancelledError as error:
                    release_cancellation = error
            propagated_cancellation = release_cancellation or cancellation
            if acquisition_error is not None:
                raise LockAcquisitionCancelled(
                    propagated_cancellation, acquisition_error
                ) from acquisition_error
            raise propagated_cancellation
        try:
            yield self
        finally:
            await self._release_async()

    async def _release_async(self) -> None:
        """Release the async."""
        release = asyncio.create_task(asyncio.to_thread(self.release))
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(release)
                break
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
                _clear_current_task_cancellation()
                if release.done():
                    release.result()
                    break
        if cancellation is not None:
            raise cancellation

    def __enter__(self) -> NamedFileLock:
        return self.acquire()

    def __exit__(self, *exc_info: object) -> None:
        self.release()


async def _await_cancellation_resilient(task: asyncio.Task[Any]) -> Any:
    # Shield the worker so cancellation does not abandon an operation already
    # handed to the background thread.
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()
            _clear_current_task_cancellation()


def _clear_current_task_cancellation() -> None:
    task = asyncio.current_task()
    if task is None:
        return
    while task.cancelling():
        task.uncancel()


class ProjectRuntimeLock(NamedFileLock):
    """Component responsible for the project runtime lock."""
    def __init__(self, project_id: str, lock_root: Path) -> None:
        super().__init__(
            project_id,
            lock_root,
            resource_type="project runtime",
            timeout=0,
            fail_when_locked=True,
            error_type=ProjectAlreadyActive,
        )
