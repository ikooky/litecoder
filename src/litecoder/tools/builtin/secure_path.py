"""Secure path validation for built-in tools."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import os
import secrets
import stat
import sys
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from litecoder.tools.builtin._common import (
    MAX_DIRECTORY_ENTRIES,
    MAX_TRAVERSAL_ENTRIES,
    canonical_workspace_root,
    normalize_relative_path,
)
from litecoder.tools.models import ToolDenied, ToolFailure, ToolPartialFailure


_DENIED = "Denied by workspace safety policy"
_TRAVERSAL_YIELD_ENTRIES = 128

# Capture the platform os primitives at import time. Capability checks must
# reflect the platform, not whatever os.open currently resolves to, so runtime
# instrumentation (tests wrapping os.open to insert barriers) cannot flip them.
_platform_open = os.open
_platform_rename = os.rename
_platform_unlink = os.unlink
_platform_scandir = os.scandir


def secure_read_file(
    root: Path, value: object, *, max_bytes: int | None = None
) -> tuple[str, bytes]:
    """Handle the secure read file operation."""
    relative = _file_relative(value)
    if os.name == "nt":
        return relative.as_posix(), _windows_read(root, relative, max_bytes)
    return relative.as_posix(), _posix_read(root, relative, max_bytes)


def secure_write_file(root: Path, value: object, payload: bytes) -> tuple[str, bool]:
    """Handle the secure write file operation."""
    relative = _file_relative(value)
    if os.name == "nt":
        changed = _windows_write(root, relative, payload)
    else:
        changed = _posix_write(root, relative, payload)
    return relative.as_posix(), changed


@dataclass(slots=True)
class TraversalState:
    """Data model representing the traversal state."""
    traversed_entries: int = 0
    truncated: bool = False
    directory_entries_truncated: bool = False
    total_entries_truncated: bool = False
    checkpoint_entries: int = 0


@dataclass(frozen=True, slots=True)
class _PinnedProcessCwd:
    path: Path | None = None
    descriptor: int | None = None


async def secure_iter_files(
    root: Path, state: TraversalState | None = None
) -> AsyncIterator[str]:
    """Handle the secure iter files operation."""
    traversal = state if state is not None else TraversalState()
    if os.name == "nt":
        async for relative in _windows_iter(root, traversal):
            yield relative
    else:
        async for relative in _posix_iter(root, traversal):
            yield relative


def secure_read_chunks(
    root: Path,
    value: object,
    *,
    chunk_bytes: int,
    max_bytes: int,
) -> tuple[str, Iterator[bytes]]:
    """Handle the secure read chunks operation."""
    if chunk_bytes <= 0 or max_bytes < 0:
        raise ValueError("read bounds must be valid")
    relative = _file_relative(value)
    chunks = (
        _windows_read_chunks(root, relative, chunk_bytes, max_bytes)
        if os.name == "nt"
        else _posix_read_chunks(root, relative, chunk_bytes, max_bytes)
    )
    return relative.as_posix(), chunks


@contextmanager
def secure_process_cwd(root: Path, value: object) -> Iterator[_PinnedProcessCwd]:
    """Handle the secure process cwd operation."""
    relative = normalize_relative_path(value)
    if os.name == "nt":
        with _windows_process_cwd(root, relative) as path:
            yield _PinnedProcessCwd(path=path)
    else:
        with _posix_process_cwd(root, relative) as pinned_cwd:
            yield pinned_cwd


def _file_relative(value: object) -> PurePosixPath:
    relative = normalize_relative_path(value)
    if relative.as_posix() == ".":
        raise ToolDenied(_DENIED)
    return relative


def _posix_flags() -> int:
    # Descriptor-relative opens prevent symlink traversal outside the workspace.
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    if nofollow is None or directory is None or _platform_open not in supports_dir_fd:
        raise ToolDenied(_DENIED)
    return os.O_RDONLY | directory | nofollow


def _require_posix_write_capabilities() -> None:
    _posix_flags()
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    if any(
        operation not in supports_dir_fd
        for operation in (_platform_open, _platform_rename, _platform_unlink)
    ):
        raise ToolDenied(_DENIED)


def _require_posix_traversal_capabilities() -> None:
    _posix_flags()
    if _platform_scandir not in getattr(os, "supports_fd", ()):
        raise ToolDenied(_DENIED)


def _fd_path_matches(candidate: Path, descriptor: int) -> bool:
    try:
        probe = os.open(candidate, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return False
    try:
        return os.path.sameopenfile(descriptor, probe)
    except OSError:
        return False
    finally:
        with contextlib.suppress(OSError):
            os.close(probe)


def _select_posix_fd_path(descriptor: int) -> Path:
    for base in (Path("/proc/self/fd"), Path("/dev/fd")):
        candidate = base / str(descriptor)
        if _fd_path_matches(candidate, descriptor):
            return candidate
    raise ToolDenied(_DENIED)


@contextmanager
def _posix_parent(root: Path, relative: PurePosixPath) -> Iterator[tuple[int, str]]:
    root_path = canonical_workspace_root(root)
    flags = _posix_flags()
    descriptors: list[int] = []
    try:
        try:
            current = os.open(root_path, flags)
        except (OSError, TypeError, NotImplementedError):
            raise ToolDenied(_DENIED) from None
        descriptors.append(current)
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            raise ToolDenied(_DENIED)
        # Walk from the already-open root so each component is checked by the OS.
        for component in relative.parts[:-1]:
            try:
                current = os.open(component, flags, dir_fd=current)
            except OSError:
                raise ToolDenied(_DENIED) from None
            descriptors.append(current)
            if not stat.S_ISDIR(os.fstat(current).st_mode):
                raise ToolDenied(_DENIED)
        yield current, relative.parts[-1]
    finally:
        for descriptor in reversed(descriptors):
            with contextlib.suppress(OSError):
                os.close(descriptor)


@contextmanager
def _posix_process_cwd(
    root: Path, relative: PurePosixPath
) -> Iterator[_PinnedProcessCwd]:
    root_path = canonical_workspace_root(root)
    flags = _posix_flags()
    descriptors: list[int] = []
    try:
        try:
            current = os.open(root_path, flags)
        except (OSError, TypeError, NotImplementedError):
            raise ToolDenied(_DENIED) from None
        descriptors.append(current)
        for component in relative.parts:
            if component == ".":
                continue
            current = os.open(component, flags, dir_fd=current)
            descriptors.append(current)
        # macOS fdescfs paths cannot be used as a subprocess cwd. Keep the
        # validated directory descriptor so the child can fchdir instead.
        if sys.platform == "darwin":
            yield _PinnedProcessCwd(descriptor=current)
        else:
            yield _PinnedProcessCwd(path=_select_posix_fd_path(current))
    except (OSError, TypeError, NotImplementedError):
        raise ToolDenied(_DENIED) from None
    finally:
        for descriptor in reversed(descriptors):
            with contextlib.suppress(OSError):
                os.close(descriptor)


def _posix_read(
    root: Path, relative: PurePosixPath, max_bytes: int | None
) -> bytes:
    with _posix_parent(root, relative) as (parent, name):
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        except OSError:
            raise ToolDenied(_DENIED) from None
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ToolDenied(_DENIED)
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                return (
                    stream.read()
                    if max_bytes is None
                    else stream.read(max_bytes + 1)
                )
        finally:
            os.close(descriptor)


def _posix_read_chunks(
    root: Path,
    relative: PurePosixPath,
    chunk_bytes: int,
    max_bytes: int,
) -> Iterator[bytes]:
    with _posix_parent(root, relative) as (parent, name):
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        except OSError:
            raise ToolDenied(_DENIED) from None
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ToolDenied(_DENIED)
            remaining = max_bytes + 1
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                while remaining > 0:
                    chunk = stream.read(min(chunk_bytes, remaining))
                    if not chunk:
                        return
                    remaining -= len(chunk)
                    yield chunk
        finally:
            os.close(descriptor)


def _posix_write(root: Path, relative: PurePosixPath, payload: bytes) -> bool:
    _require_posix_write_capabilities()
    with _posix_parent(root, relative) as (parent, name):
        try:
            existing = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        except FileNotFoundError:
            existing = -1
        except OSError:
            raise ToolDenied(_DENIED) from None
        if existing >= 0:
            try:
                if not stat.S_ISREG(os.fstat(existing).st_mode):
                    raise ToolDenied(_DENIED)
                with os.fdopen(existing, "rb", closefd=False) as stream:
                    if stream.read() == payload:
                        return False
            finally:
                os.close(existing)

        temporary = f".{name}.litecoder-{secrets.token_hex(8)}.tmp"
        descriptor = -1
        replaced = False
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            replaced = True
            os.fsync(parent)
            return True
        except BaseException as error:
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
            cleanup_failed = False
            if not replaced:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
                except OSError:
                    cleanup_failed = True
            if replaced or cleanup_failed:
                raise ToolPartialFailure(
                    "Workspace write may have partially completed",
                    changed_workspace=True,
                    metadata={"phase": "durability" if replaced else "cleanup"},
                ) from None
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(error, ToolDenied):
                raise
            raise ToolFailure("Workspace file could not be written") from None


async def _posix_iter(root: Path, state: TraversalState) -> AsyncIterator[str]:
    _require_posix_traversal_capabilities()
    root_path = canonical_workspace_root(root)
    flags = _posix_flags()
    try:
        root_fd = os.open(root_path, flags)
    except (OSError, TypeError, NotImplementedError):
        raise ToolDenied(_DENIED) from None
    try:
        async for relative in _posix_walk(root_fd, PurePosixPath(), state):
            yield relative
    finally:
        os.close(root_fd)


async def _bounded_entries(
    source: object, state: TraversalState
) -> list[object] | None:
    entries: list[object] = []
    try:
        for entry in source:
            state.traversed_entries += 1
            if state.traversed_entries > MAX_TRAVERSAL_ENTRIES:
                state.truncated = True
                state.total_entries_truncated = True
                return None
            entries.append(entry)
            if len(entries) > MAX_DIRECTORY_ENTRIES:
                state.truncated = True
                state.directory_entries_truncated = True
                return None
            await _traversal_checkpoint(state)
    except (OSError, TypeError, NotImplementedError):
        return None
    return sorted(entries, key=lambda entry: entry.name)


async def _traversal_checkpoint(state: TraversalState) -> None:
    """Yield periodically so recursive scans cannot monopolize the event loop."""
    state.checkpoint_entries += 1
    if state.checkpoint_entries % _TRAVERSAL_YIELD_ENTRIES == 0:
        await asyncio.sleep(0)


async def _posix_walk(
    directory_fd: int, prefix: PurePosixPath, state: TraversalState
) -> AsyncIterator[str]:
    try:
        scanner = os.scandir(directory_fd)
    except (TypeError, NotImplementedError):
        raise ToolDenied(_DENIED) from None
    except OSError:
        return
    with scanner:
        entries = await _bounded_entries(scanner, state)
    if entries is None:
        return
    for entry in entries:
        if state.truncated:
            return
        await _traversal_checkpoint(state)
        if entry.name.casefold() == ".memory":
            continue
        try:
            information = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        relative = prefix / entry.name
        if stat.S_ISREG(information.st_mode):
            try:
                handle = os.open(
                    entry.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd
                )
            except OSError:
                continue
            try:
                if stat.S_ISREG(os.fstat(handle).st_mode):
                    yield relative.as_posix()
            finally:
                os.close(handle)
        elif stat.S_ISDIR(information.st_mode):
            try:
                child = os.open(entry.name, _posix_flags(), dir_fd=directory_fd)
            except OSError:
                continue
            try:
                async for child_relative in _posix_walk(child, relative, state):
                    yield child_relative
            finally:
                os.close(child)


if os.name == "nt":
    import msvcrt
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _DELETE = 0x00010000
    _FILE_SHARE_READ = 0x1
    _FILE_SHARE_WRITE = 0x2
    _CREATE_NEW = 1
    _OPEN_EXISTING = 3
    _FILE_ATTRIBUTE_DIRECTORY = 0x10
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x400
    _FILE_ATTRIBUTE_NORMAL = 0x80
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _MOVEFILE_REPLACE_EXISTING = 0x1
    _MOVEFILE_WRITE_THROUGH = 0x8
    _ERROR_FILE_NOT_FOUND = 2
    _ERROR_PATH_NOT_FOUND = 3
    _FILE_RENAME_INFO_CLASS = 3
    _FILE_DISPOSITION_INFO_CLASS = 13
    _DUPLICATE_SAME_ACCESS = 0x2

    class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        """Internal helper for the by handle file information."""
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    class _IO_STATUS_BLOCK(ctypes.Structure):
        """Internal helper for the io status block."""
        _fields_ = [
            ("Status", ctypes.c_ssize_t),
            ("Information", ctypes.c_size_t),
        ]
    class _FILE_RENAME_INFO(ctypes.Structure):
        """Internal helper for the file rename info."""
        _fields_ = [
            ("ReplaceIfExists", ctypes.c_ubyte),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", ctypes.c_wchar * 1),
        ]
    _kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.DuplicateHandle.argtypes = (
        wintypes.HANDLE,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    _kernel32.DuplicateHandle.restype = wintypes.BOOL
    _ntdll.NtSetInformationFile.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_IO_STATUS_BLOCK),
        ctypes.c_void_p,
        wintypes.ULONG,
        ctypes.c_int,
    )
    _ntdll.NtSetInformationFile.restype = ctypes.c_long
    _kernel32.GetFileInformationByHandle.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION),
    )
    _kernel32.GetFinalPathNameByHandleW.argtypes = (
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    )
    _kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
    _kernel32.MoveFileExW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
    _kernel32.MoveFileExW.restype = wintypes.BOOL
    _kernel32.DeleteFileW.argtypes = (wintypes.LPCWSTR,)
    _kernel32.DeleteFileW.restype = wintypes.BOOL


class _WinNotFound(Exception):
    """Raised when the win not found conditions occur."""
    pass


class _WinHandle:
    """Internal helper for the win handle."""
    def __init__(self, value: int) -> None:
        self.value = value

    def close(self) -> None:
        """Close the managed resource and release any lock."""
        if self.value:
            _kernel32.CloseHandle(self.value)
            self.value = 0

    def detach(self) -> int:
        """Detach the resource from its parent lifecycle."""
        value = self.value
        self.value = 0
        return value

    def __enter__(self) -> _WinHandle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@contextmanager
def _windows_process_cwd(
    root: Path, relative: PurePosixPath
) -> Iterator[Path]:
    root_path = canonical_workspace_root(root)
    handles: list[_WinHandle] = []
    try:
        root_handle = _win_open_existing(root_path, directory=True)
        handles.append(root_handle)
        root_final = _win_final_path(root_handle)
        _win_validate(root_handle, root_final, directory=True)
        current = root_path
        for component in relative.parts:
            if component == ".":
                continue
            current /= component
            handle = _win_open_existing(current, directory=True)
            handles.append(handle)
            _win_validate(handle, root_final, directory=True)
        yield current
    finally:
        for handle in reversed(handles):
            handle.close()


def _windows_read(
    root: Path, relative: PurePosixPath, max_bytes: int | None
) -> bytes:
    with _windows_parent(root, relative) as state:
        handle = _win_open_existing(state.path / relative.name, directory=False)
        try:
            _win_validate(handle, state.root_final, directory=False)
            descriptor = msvcrt.open_osfhandle(handle.detach(), os.O_RDONLY | os.O_BINARY)
            with os.fdopen(descriptor, "rb") as stream:
                return (
                    stream.read()
                    if max_bytes is None
                    else stream.read(max_bytes + 1)
                )
        except BaseException:
            handle.close()
            raise


def _windows_read_chunks(
    root: Path,
    relative: PurePosixPath,
    chunk_bytes: int,
    max_bytes: int,
) -> Iterator[bytes]:
    with _windows_parent(root, relative) as state:
        handle = _win_open_existing(state.path / relative.name, directory=False)
        try:
            _win_validate(handle, state.root_final, directory=False)
            descriptor = msvcrt.open_osfhandle(
                handle.detach(), os.O_RDONLY | os.O_BINARY
            )
            remaining = max_bytes + 1
            with os.fdopen(descriptor, "rb") as stream:
                while remaining > 0:
                    chunk = stream.read(min(chunk_bytes, remaining))
                    if not chunk:
                        return
                    remaining -= len(chunk)
                    yield chunk
        except BaseException:
            handle.close()
            raise


def _windows_write(root: Path, relative: PurePosixPath, payload: bytes) -> bool:
    with _windows_parent(root, relative) as state:
        target = state.path / relative.name
        try:
            existing = _win_open_existing(target, directory=False)
        except _WinNotFound:
            existing = None
        if existing is not None:
            try:
                _win_validate(existing, state.root_final, directory=False)
                descriptor = msvcrt.open_osfhandle(
                    existing.detach(), os.O_RDONLY | os.O_BINARY
                )
                with os.fdopen(descriptor, "rb") as stream:
                    if stream.read() == payload:
                        return False
            finally:
                existing.close()

        temporary = state.path / (
            f".{relative.name}.litecoder-{secrets.token_hex(8)}.tmp"
        )
        created: _WinHandle | None = None
        replaced = False
        try:
            created = _win_create_new(temporary)
            _win_validate(created, state.root_final, directory=False)
            writer = _win_duplicate_handle(created)
            try:
                descriptor = msvcrt.open_osfhandle(
                    writer.detach(), os.O_WRONLY | os.O_BINARY
                )
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                writer.close()

            _win_validate_parent_state(state)
            _win_rename_handle(
                created, state.handles[-1], relative.name
            )
            replaced = True
            created.close()
            created = None
            _win_validate_parent_state(state)
            result = _win_open_existing(target, directory=False)
            try:
                _win_validate(result, state.root_final, directory=False)
            finally:
                result.close()
            return True
        except BaseException as error:
            cleanup_failed = False
            if created is not None:
                if not replaced:
                    try:
                        _win_mark_delete(created)
                    except OSError:
                        cleanup_failed = True
                created.close()
            if replaced or cleanup_failed:
                raise ToolPartialFailure(
                    "Workspace write may have partially completed",
                    changed_workspace=True,
                    metadata={
                        "phase": "containment" if replaced else "cleanup"
                    },
                ) from None
            if isinstance(error, (ToolDenied, KeyboardInterrupt, SystemExit)):
                raise
            raise ToolFailure("Workspace file could not be written") from None


def _win_duplicate_handle(handle: _WinHandle) -> _WinHandle:
    current = _kernel32.GetCurrentProcess()
    duplicate = wintypes.HANDLE()
    if not _kernel32.DuplicateHandle(
        current,
        handle.value,
        current,
        ctypes.byref(duplicate),
        0,
        False,
        _DUPLICATE_SAME_ACCESS,
    ):
        raise ToolFailure("Workspace file could not be written")
    return _WinHandle(duplicate.value)


def _win_mark_delete(handle: _WinHandle) -> None:
    disposition = ctypes.c_ubyte(1)
    status = _IO_STATUS_BLOCK()
    result = _ntdll.NtSetInformationFile(
        handle.value,
        ctypes.byref(status),
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
        _FILE_DISPOSITION_INFO_CLASS,
    )
    if result < 0:
        raise OSError("Handle-based deletion failed")


def _win_rename_handle(
    handle: _WinHandle, parent: _WinHandle, filename: str
) -> None:
    filename_bytes = filename.encode("utf-16-le")
    size = _FILE_RENAME_INFO.FileName.offset + len(filename_bytes)
    buffer = ctypes.create_string_buffer(size)
    information = ctypes.cast(
        buffer, ctypes.POINTER(_FILE_RENAME_INFO)
    ).contents
    information.ReplaceIfExists = True
    information.RootDirectory = parent.value
    information.FileNameLength = len(filename_bytes)
    ctypes.memmove(
        ctypes.addressof(buffer) + _FILE_RENAME_INFO.FileName.offset,
        filename_bytes,
        len(filename_bytes),
    )
    status = _IO_STATUS_BLOCK()
    result = _ntdll.NtSetInformationFile(
        handle.value,
        ctypes.byref(status),
        buffer,
        size,
        10,  # FileRenameInformation
    )
    if result < 0:
        raise OSError("Handle-based rename failed")

class _WindowsParentState:
    """Internal helper for the windows parent state."""
    def __init__(
        self, path: Path, handles: list[_WinHandle], root_final: str
    ) -> None:
        self.path = path
        self.handles = handles
        self.root_final = root_final


@contextmanager
def _windows_parent(
    root: Path, relative: PurePosixPath
) -> Iterator[_WindowsParentState]:
    root_path = canonical_workspace_root(root)
    handles: list[_WinHandle] = []
    try:
        root_handle = _win_open_existing(root_path, directory=True)
        handles.append(root_handle)
        root_final = _win_final_path(root_handle)
        _win_validate(root_handle, root_final, directory=True)
        current = root_path
        for component in relative.parts[:-1]:
            current /= component
            handle = _win_open_existing(current, directory=True)
            handles.append(handle)
            _win_validate(handle, root_final, directory=True)
        yield _WindowsParentState(current, handles, root_final)
    finally:
        for handle in reversed(handles):
            handle.close()


async def _windows_iter(root: Path, state: TraversalState) -> AsyncIterator[str]:
    root_path = canonical_workspace_root(root)
    with _windows_parent(root_path, PurePosixPath(".")) as parent:
        async for relative in _windows_walk(
            parent.path,
            parent.handles[-1],
            parent.root_final,
            PurePosixPath(),
            state,
        ):
            yield relative


async def _windows_walk(
    directory: Path,
    directory_handle: _WinHandle,
    root_final: str,
    prefix: PurePosixPath,
    state: TraversalState,
) -> AsyncIterator[str]:
    _win_validate(directory_handle, root_final, directory=True)
    try:
        scanner = os.scandir(directory)
    except (OSError, TypeError, NotImplementedError):
        return
    with scanner:
        entries = await _bounded_entries(scanner, state)
    if entries is None:
        return
    for entry in entries:
        if state.truncated:
            return
        await _traversal_checkpoint(state)
        if entry.name.casefold() == ".memory":
            continue
        child_path = directory / entry.name
        relative = prefix / entry.name
        try:
            information = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        attributes = getattr(information, "st_file_attributes", 0)
        if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            continue
        if stat.S_ISDIR(information.st_mode):
            try:
                handle = _win_open_existing(child_path, directory=True)
                _win_validate(handle, root_final, directory=True)
            except (ToolDenied, _WinNotFound):
                continue
            try:
                async for child_relative in _windows_walk(
                    child_path, handle, root_final, relative, state
                ):
                    yield child_relative
            finally:
                handle.close()
        elif stat.S_ISREG(information.st_mode):
            try:
                handle = _win_open_existing(child_path, directory=False)
                _win_validate(handle, root_final, directory=False)
            except (ToolDenied, _WinNotFound):
                continue
            else:
                handle.close()
                yield relative.as_posix()


def _win_open_existing(path: Path, *, directory: bool) -> _WinHandle:
    flags = _FILE_FLAG_OPEN_REPARSE_POINT
    if directory:
        flags |= _FILE_FLAG_BACKUP_SEMANTICS
    handle = _kernel32.CreateFileW(
        _extended(path),
        _GENERIC_READ,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        None,
        _OPEN_EXISTING,
        flags,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        error = ctypes.get_last_error()
        if error in {_ERROR_FILE_NOT_FOUND, _ERROR_PATH_NOT_FOUND}:
            raise _WinNotFound
        raise ToolDenied(_DENIED)
    return _WinHandle(handle)


def _win_create_new(path: Path) -> _WinHandle:
    handle = _kernel32.CreateFileW(
        _extended(path),
        _GENERIC_READ | _GENERIC_WRITE | _DELETE,
        _FILE_SHARE_READ,
        None,
        _CREATE_NEW,
        _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        raise ToolFailure("Workspace file could not be written")
    return _WinHandle(handle)


def _win_validate(handle: _WinHandle, root_final: str, *, directory: bool) -> None:
    information = _BY_HANDLE_FILE_INFORMATION()
    if not _kernel32.GetFileInformationByHandle(handle.value, ctypes.byref(information)):
        raise ToolDenied(_DENIED)
    attributes = information.dwFileAttributes
    if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise ToolDenied(_DENIED)
    if bool(attributes & _FILE_ATTRIBUTE_DIRECTORY) != directory:
        raise ToolDenied(_DENIED)
    final = _win_final_path(handle)
    if not _win_within(root_final, final):
        raise ToolDenied(_DENIED)


def _win_validate_parent_state(state: _WindowsParentState) -> None:
    for handle in state.handles:
        _win_validate(handle, state.root_final, directory=True)


def _win_final_path(handle: _WinHandle) -> str:
    needed = _kernel32.GetFinalPathNameByHandleW(handle.value, None, 0, 0)
    if not needed:
        raise ToolDenied(_DENIED)
    buffer = ctypes.create_unicode_buffer(needed + 1)
    if not _kernel32.GetFinalPathNameByHandleW(handle.value, buffer, len(buffer), 0):
        raise ToolDenied(_DENIED)
    return _normal_final(buffer.value)


def _normal_final(value: str) -> str:
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return os.path.normcase(os.path.normpath(value))


def _win_within(root: str, candidate: str) -> bool:
    try:
        return os.path.commonpath((root, candidate)) == root
    except ValueError:
        return False


def _extended(path: Path) -> str:
    value = str(path.absolute())
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value
