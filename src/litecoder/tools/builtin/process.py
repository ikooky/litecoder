"""Built-in process lifecycle tools."""

from __future__ import annotations

import asyncio
import codecs
import ctypes
import functools
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from litecoder.common.trace.redaction import SecretRedactor
from litecoder.tools.builtin._common import (
    MAX_OUTPUT_BYTES,
    PROCESS_READ_CHUNK_BYTES,
    truncate_utf8,
)
from litecoder.tools.builtin.secure_path import secure_process_cwd
from litecoder.tools.models import ToolFailure


_CLEANUP_TIMEOUT_SECONDS = 5.0
_CREATE_SUSPENDED = 0x00000004
MAX_STREAM_SECRET_BYTES = MAX_OUTPUT_BYTES // 4  # UTF-8 worst-case overlap bound
_BEARER_TOKEN = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._~+/=-"
)


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Data model representing the process result."""
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool

    def metadata(self, *, changed_workspace: bool) -> dict[str, object]:
        """Handle the metadata operation."""
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "truncated": self.stdout_truncated or self.stderr_truncated,
            "timed_out": self.timed_out,
            "changed_workspace": changed_workspace,
        }


class _BoundedRedactedCapture:
    """Internal helper for the bounded redacted capture."""
    def __init__(self, redactor: SecretRedactor) -> None:
        self._redactor = redactor
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""
        self._in_bearer = False
        self._parts: list[str] = []
        self._size = 0
        self.truncated = False
        self._overlap = max(
            [len("Bearer ") - 1, *(max(0, len(value) - 1) for value in redactor.values)]
        )

    def feed(self, data: bytes) -> None:
        """Feed input to the managed process."""
        self._process(self._decoder.decode(data, final=False))

    def finish(self) -> str:
        """Finish the managed process and collect its result."""
        self._process(self._decoder.decode(b"", final=True))
        if not self._in_bearer:
            self._append(self._redactor.redact_text(self._pending))
        self._pending = ""
        return "".join(self._parts).replace("\r\n", "\n").replace("\r", "\n")

    def _process(self, text: str) -> None:
        """Process the supplied input."""
        if not text:
            return
        if self._in_bearer:
            index = 0
            while index < len(text) and text[index] in _BEARER_TOKEN:
                index += 1
            if index == len(text):
                return
            self._in_bearer = False
            text = text[index:]

        combined = self._pending + text
        self._pending = ""
        open_bearer = _open_bearer_start(combined)
        if open_bearer is not None:
            prefix = combined[:open_bearer]
            self._append(self._redactor.redact_text(prefix))
            self._append("[REDACTED]")
            self._in_bearer = True
            return

        keep = min(self._overlap, len(combined))
        safe_end = len(combined) - keep
        for secret in self._redactor.values:
            if not secret:
                continue
            search_start = max(0, safe_end - len(secret) + 1)
            occurrence = combined.find(secret, search_start)
            if occurrence >= 0 and occurrence < safe_end < occurrence + len(secret):
                safe_end = occurrence
        if safe_end:
            self._append(self._redactor.redact_text(combined[:safe_end]))
        self._pending = combined[safe_end:]

    def _append(self, value: str) -> None:
        if not value:
            return
        remaining = MAX_OUTPUT_BYTES - self._size
        if remaining <= 0:
            self.truncated = True
            return
        bounded, truncated = truncate_utf8(value, remaining)
        if bounded:
            self._parts.append(bounded)
            self._size += len(bounded.encode("utf-8"))
        if truncated:
            self.truncated = True


def _open_bearer_start(value: str) -> int | None:
    match = re.search(
        r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+$",
        value,
    )
    return None if match is None else match.start()


class _OwnedProcess:
    """Internal helper for the owned process."""
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        job: _WindowsJob | None,
    ) -> None:
        self.process = process
        self.job = job

    def terminate_tree(self) -> None:
        """Handle the terminate tree operation."""
        if self.job is not None:
            self.job.terminate()
        elif os.name != "nt":
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                self._kill_root()
        else:
            self._kill_root()
        self._kill_root()

    def close(self) -> None:
        """Close the managed resource and release any lock."""
        if self.job is not None:
            self.job.close()

    def _kill_root(self) -> None:
        if self.process.returncode is None:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass


async def run_bounded_process(
    argv: list[str],
    *,
    workspace_root: Path,
    cwd: object,
    env: dict[str, str],
    timeout: float,
    redactor: SecretRedactor,
) -> ProcessResult:
    """Run the bounded process."""
    _validate_redaction_bounds(redactor)
    owned = await _spawn_owned(argv, workspace_root=workspace_root, cwd=cwd, env=env)
    stdout_capture = _BoundedRedactedCapture(redactor)
    stderr_capture = _BoundedRedactedCapture(redactor)
    stdout_task = asyncio.create_task(
        _drain(owned.process.stdout, stdout_capture),
        name="litecoder-process-stdout",
    )
    stderr_task = asyncio.create_task(
        _drain(owned.process.stderr, stderr_capture),
        name="litecoder-process-stderr",
    )
    wait_task = asyncio.create_task(
        owned.process.wait(), name="litecoder-process-wait"
    )
    tasks = (wait_task, stdout_task, stderr_task)
    completion = asyncio.gather(*tasks)
    timed_out = False
    try:
        try:
            await asyncio.wait_for(asyncio.shield(completion), timeout=timeout)
        except TimeoutError:
            timed_out = True
            await _cleanup_owned(owned, tasks)
        except asyncio.CancelledError:
            await _cleanup_owned(owned, tasks)
            raise
        except Exception:
            await _cleanup_owned(owned, tasks)
            raise ToolFailure("Process output capture failed") from None
        stdout = stdout_capture.finish()
        stderr = stderr_capture.finish()
        return ProcessResult(
            exit_code=owned.process.returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_capture.truncated,
            stderr_truncated=stderr_capture.truncated,
            timed_out=timed_out,
        )
    finally:
        owned.close()


def _validate_redaction_bounds(redactor: SecretRedactor) -> None:
    """Validate the redaction bounds."""
    for secret in redactor.values:
        try:
            size = len(secret.encode("utf-8"))
        except UnicodeEncodeError:
            raise ToolFailure("Subprocess redaction bounds are invalid") from None
        if size > MAX_STREAM_SECRET_BYTES:
            raise ToolFailure("Subprocess redaction bounds are invalid")

async def _spawn_owned(
    argv: list[str],
    *,
    workspace_root: Path,
    cwd: object,
    env: dict[str, str],
) -> _OwnedProcess:
    try:
        job = _WindowsJob.create() if os.name == "nt" else None
    except OSError:
        raise ToolFailure("Process could not be contained") from None
    try:
        with secure_process_cwd(workspace_root, cwd) as pinned_cwd:
            kwargs: dict[str, object] = {}
            if os.name == "nt":
                kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP | _CREATE_SUSPENDED
                )
            else:
                kwargs["start_new_session"] = True
            if pinned_cwd.descriptor is not None:
                kwargs["preexec_fn"] = functools.partial(
                    os.fchdir, pinned_cwd.descriptor
                )
                kwargs["pass_fds"] = (pinned_cwd.descriptor,)
            else:
                kwargs["cwd"] = pinned_cwd.path
            spawn_task = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *argv,
                    env=env,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **kwargs,
                ),
                name="litecoder-process-spawn",
            )
            cancelled = False
            try:
                process = await asyncio.shield(spawn_task)
            except asyncio.CancelledError:
                cancelled = True
                deadline = (
                    asyncio.get_running_loop().time()
                    + _CLEANUP_TIMEOUT_SECONDS
                )
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        spawn_task.cancel()
                        await asyncio.gather(
                            spawn_task, return_exceptions=True
                        )
                        raise asyncio.CancelledError from None
                    try:
                        process = await asyncio.wait_for(
                            asyncio.shield(spawn_task), timeout=remaining
                        )
                        break
                    except asyncio.CancelledError:
                        continue
                    except TimeoutError:
                        spawn_task.cancel()
                        await asyncio.gather(
                            spawn_task, return_exceptions=True
                        )
                        raise asyncio.CancelledError from None
                    except (OSError, ValueError):
                        raise asyncio.CancelledError from None
            except (OSError, ValueError):
                raise ToolFailure("Process could not be started") from None

            owned = _OwnedProcess(process, job)
            try:
                if job is not None:
                    job.assign(process.pid)
                    job.resume(process.pid)
            except OSError:
                await _cleanup_unaccepted(owned)
                raise ToolFailure("Process could not be contained") from None
            if cancelled:
                await _cleanup_unaccepted(owned)
                raise asyncio.CancelledError
            return owned
    except BaseException:
        if job is not None:
            job.close()
        raise


async def _cleanup_unaccepted(owned: _OwnedProcess) -> None:
    stdout_capture = _BoundedRedactedCapture(SecretRedactor.with_values(()))
    stderr_capture = _BoundedRedactedCapture(SecretRedactor.with_values(()))
    tasks = (
        asyncio.create_task(owned.process.wait()),
        asyncio.create_task(_drain(owned.process.stdout, stdout_capture)),
        asyncio.create_task(_drain(owned.process.stderr, stderr_capture)),
    )
    await _cleanup_owned(owned, tasks)


async def _cleanup_owned(
    owned: _OwnedProcess,
    tasks: tuple[asyncio.Task[object], ...],
) -> None:
    owned.terminate_tree()
    cleanup = asyncio.gather(*tasks, return_exceptions=True)
    try:
        await asyncio.wait_for(
            asyncio.shield(cleanup), timeout=_CLEANUP_TIMEOUT_SECONDS
        )
    except TimeoutError:
        owned.terminate_tree()
        cleanup.cancel()
        await asyncio.gather(cleanup, return_exceptions=True)
    except asyncio.CancelledError:
        try:
            await asyncio.wait_for(
                asyncio.shield(cleanup), timeout=_CLEANUP_TIMEOUT_SECONDS
            )
        except (TimeoutError, asyncio.CancelledError):
            owned.terminate_tree()
            cleanup.cancel()
            await asyncio.gather(cleanup, return_exceptions=True)


async def _drain(
    stream: asyncio.StreamReader | None,
    capture: _BoundedRedactedCapture,
) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(PROCESS_READ_CHUNK_BYTES)
        if not chunk:
            return
        capture.feed(chunk)


if os.name == "nt":
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _PROCESS_SUSPEND_RESUME = 0x0800

    class _IO_COUNTERS(ctypes.Structure):
        """Internal helper for the io counters."""
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        """Internal helper for the jobobject basic limit information."""
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        """Internal helper for the jobobject extended limit information."""
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _kernel32.CreateJobObjectW.argtypes = (
        ctypes.c_void_p,
        wintypes.LPCWSTR,
    )
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.OpenProcess.argtypes = (
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.AssignProcessToJobObject.argtypes = (
        wintypes.HANDLE,
        wintypes.HANDLE,
    )
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _ntdll.NtResumeProcess.argtypes = (wintypes.HANDLE,)
    _ntdll.NtResumeProcess.restype = ctypes.c_long


class _WindowsJob:
    """Internal helper for the windows job."""
    def __init__(self, handle: int) -> None:
        self._handle = handle

    @classmethod
    def create(cls) -> _WindowsJob:
        """Create the requested object."""
        if os.name != "nt":
            raise OSError("Windows Job Objects are unavailable")
        handle = _kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError("Job Object creation failed")
        job = cls(handle)
        information = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        information.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not _kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            job.close()
            raise OSError("Job Object configuration failed")
        return job

    def assign(self, pid: int) -> None:
        """Assign the process to a task."""
        process = _kernel32.OpenProcess(
            _PROCESS_TERMINATE
            | _PROCESS_SET_QUOTA
            | _PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )
        if not process:
            raise OSError("Process ownership failed")
        try:
            if not _kernel32.AssignProcessToJobObject(self._handle, process):
                raise OSError("Process ownership failed")
        finally:
            _kernel32.CloseHandle(process)

    def resume(self, pid: int) -> None:
        """Resume a paused task or session."""
        process = _kernel32.OpenProcess(
            _PROCESS_SUSPEND_RESUME | _PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )
        if not process:
            raise OSError("Process resume failed")
        try:
            if _ntdll.NtResumeProcess(process) < 0:
                raise OSError("Process resume failed")
        finally:
            _kernel32.CloseHandle(process)

    def terminate(self) -> None:
        """Terminate the managed process."""
        if self._handle:
            _kernel32.TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        """Close the managed resource and release any lock."""
        if self._handle:
            _kernel32.CloseHandle(self._handle)
            self._handle = 0
