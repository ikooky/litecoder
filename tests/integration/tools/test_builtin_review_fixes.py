from __future__ import annotations

import asyncio
import os
import re
import stat
import sys
from pathlib import Path

import pytest

from litecoder.common.trace.redaction import SecretRedactor
from litecoder.tools import ToolCall, ToolContext, ToolDenied, ToolFailure
from litecoder.tools.builtin import RunShellTool, WriteFileTool
from litecoder.tools.builtin._common import (
    MAX_FILE_BYTES,
    MAX_OUTPUT_BYTES,
    MAX_SEARCH_LINE_BYTES,
    PROCESS_READ_CHUNK_BYTES,
)


def context(root: Path, *, secrets: tuple[str, ...] = ()) -> ToolContext:
    return ToolContext(
        "agent",
        "workspace",
        root,
        metadata={"round_number": 1, "permission_mode": "ask"},
        secret_values=secrets,
    )


def pid_exists(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, 0, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            return bool(
                kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ) and exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


async def wait_for_exit(pid: int) -> None:
    for _ in range(500):
        if not pid_exists(pid):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"process remains alive: {pid}")


@pytest.mark.asyncio
async def test_repeated_cancellation_during_pending_spawn_still_reaps_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.process as process_runner

    real_spawn = asyncio.create_subprocess_exec
    entered = asyncio.Event()
    release = asyncio.Event()
    spawned_pid: list[int] = []

    async def delayed_spawn(*argv: str, **kwargs: object):
        entered.set()
        await release.wait()
        process = await real_spawn(*argv, **kwargs)
        spawned_pid.append(process.pid)
        return process

    monkeypatch.setattr(
        process_runner.asyncio, "create_subprocess_exec", delayed_spawn
    )
    task = asyncio.create_task(
        RunShellTool().execute(
            ToolCall(
                "repeated-spawn-cancel",
                "run_shell",
                {
                    "argv": [
                        sys.executable,
                        "-c",
                        "import threading; threading.Event().wait()",
                    ]
                },
            ),
            context(tmp_path),
        )
    )
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(spawned_pid) == 1
    await wait_for_exit(spawned_pid[0])


def test_search_keeps_only_bounded_prefix_for_no_newline_file(tmp_path: Path) -> None:
    from litecoder.tools.builtin.search import _search_chunks

    secret = "bounded-line-secret"
    observed_lengths: list[int] = []

    class Pattern:
        def finditer(self, value: str):
            observed_lengths.append(len(value.encode("utf-8")))
            return re.compile("needle").finditer(value)

    prefix = f"needle {secret} ".encode("utf-8")
    payload = prefix + b"x" * (MAX_FILE_BYTES - len(prefix))
    chunks = (
        payload[index : index + PROCESS_READ_CHUNK_BYTES]
        for index in range(0, len(payload), PROCESS_READ_CHUNK_BYTES)
    )

    matches, oversized, has_more = _search_chunks(
        chunks,
        relative="huge.txt",
        compiled=Pattern(),  # type: ignore[arg-type]
        context=context(tmp_path, secrets=(secret,)),
        remaining_limit=10,
    )

    assert oversized is False
    assert has_more is False
    assert len(matches) == 1
    assert matches[0]["line_truncated"] is True
    assert secret not in repr(matches)
    assert observed_lengths
    assert max(observed_lengths) <= MAX_SEARCH_LINE_BYTES


@pytest.mark.skipif(os.name != "nt", reason="Windows handle rename behavior")
@pytest.mark.asyncio
async def test_windows_write_keeps_validated_temp_identity_through_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    attacker = tmp_path / "attacker.txt"
    attacker.write_text("attacker", encoding="utf-8")
    original = secure_path._win_rename_handle
    attempted = False

    def racing_rename(handle, parent, filename: str) -> None:
        nonlocal attempted
        attempted = True
        temporary = next(tmp_path.glob(".target.txt.litecoder-*.tmp"))
        with pytest.raises(PermissionError):
            os.replace(attacker, temporary)
        original(handle, parent, filename)

    monkeypatch.setattr(secure_path, "_win_rename_handle", racing_rename)
    result = await WriteFileTool().execute(
        ToolCall(
            "temp-identity",
            "write_file",
            {"path": "target.txt", "content": "trusted"},
        ),
        context(tmp_path),
    )

    assert attempted is True
    assert result.status == "success"
    assert target.read_text(encoding="utf-8") == "trusted"


def test_posix_cwd_selects_dev_fd_when_proc_fd_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    monkeypatch.setattr(
        secure_path,
        "_fd_path_matches",
        lambda candidate, descriptor: candidate.as_posix().startswith("/dev/fd/"),
    )

    assert secure_path._select_posix_fd_path(17) == Path("/dev/fd/17")


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory descriptor behavior")
def test_macos_cwd_keeps_validated_directory_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    monkeypatch.setattr(secure_path.sys, "platform", "darwin")
    with secure_path.secure_process_cwd(tmp_path, ".") as pinned_cwd:
        assert pinned_cwd.path is None
        assert pinned_cwd.descriptor is not None
        assert stat.S_ISDIR(os.fstat(pinned_cwd.descriptor).st_mode)


def test_posix_write_preflights_dir_fd_capabilities_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    opened = False

    def forbidden_open(*_: object, **__: object) -> int:
        nonlocal opened
        opened = True
        raise AssertionError("open must not run")

    monkeypatch.setattr(secure_path, "_posix_flags", lambda: os.O_RDONLY)
    monkeypatch.setattr(secure_path.os, "open", forbidden_open)
    monkeypatch.setattr(
        secure_path.os,
        "supports_dir_fd",
        {forbidden_open, secure_path.os.unlink},
    )

    with pytest.raises(ToolDenied, match="workspace safety policy"):
        secure_path._posix_write(
            tmp_path, secure_path.PurePosixPath("target.txt"), b"content"
        )

    assert opened is False
    assert not list(tmp_path.glob(".target.txt.litecoder-*.tmp"))


def test_posix_write_accepts_rename_dir_fd_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    monkeypatch.setattr(secure_path, "_posix_flags", lambda: os.O_RDONLY)
    monkeypatch.setattr(
        secure_path.os,
        "supports_dir_fd",
        {secure_path.os.open, secure_path.os.rename, secure_path.os.unlink},
    )

    secure_path._require_posix_write_capabilities()


@pytest.mark.asyncio
async def test_posix_traversal_preflights_scandir_fd_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    opened = False

    def forbidden_open(*_: object, **__: object) -> int:
        nonlocal opened
        opened = True
        raise AssertionError("open must not run")

    monkeypatch.setattr(secure_path, "_posix_flags", lambda: os.O_RDONLY)
    monkeypatch.setattr(secure_path.os, "open", forbidden_open)
    monkeypatch.setattr(secure_path.os, "supports_fd", set())

    with pytest.raises(ToolDenied, match="workspace safety policy"):
        [
            path
            async for path in secure_path._posix_iter(
                tmp_path, secure_path.TraversalState()
            )
        ]

    assert opened is False


@pytest.mark.asyncio
async def test_huge_configured_secret_fails_before_subprocess_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.process as process_runner

    spawned = False

    async def forbidden_spawn(*_: object, **__: object):
        nonlocal spawned
        spawned = True
        raise AssertionError("subprocess must not spawn")

    monkeypatch.setattr(
        process_runner.asyncio, "create_subprocess_exec", forbidden_spawn
    )
    secret = "s" * (MAX_OUTPUT_BYTES + 1)

    with pytest.raises(ToolFailure, match="redaction bounds"):
        await RunShellTool().execute(
            ToolCall(
                "huge-secret",
                "run_shell",
                {"argv": [sys.executable, "-c", "print('never')"]},
            ),
            context(tmp_path, secrets=(secret,)),
        )

    assert spawned is False


def test_streaming_capture_pending_state_is_bounded() -> None:
    from litecoder.tools.builtin.process import (
        MAX_STREAM_SECRET_BYTES,
        _BoundedRedactedCapture,
    )

    secret = "s" * MAX_STREAM_SECRET_BYTES
    capture = _BoundedRedactedCapture(SecretRedactor.with_values((secret,)))
    payload = (secret + ("x" * MAX_STREAM_SECRET_BYTES)).encode("utf-8")
    observed: list[int] = []
    for index in range(0, len(payload), PROCESS_READ_CHUNK_BYTES):
        capture.feed(payload[index : index + PROCESS_READ_CHUNK_BYTES])
        observed.append(len(capture._pending.encode("utf-8")))

    assert observed
    assert max(observed) <= MAX_OUTPUT_BYTES


@pytest.mark.asyncio
async def test_pending_spawn_failure_preserves_prior_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.process as process_runner

    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_failure(*_: object, **__: object):
        entered.set()
        await release.wait()
        raise OSError("spawn failed")

    monkeypatch.setattr(
        process_runner.asyncio, "create_subprocess_exec", delayed_failure
    )
    task = asyncio.create_task(
        RunShellTool().execute(
            ToolCall(
                "cancelled-spawn-failure",
                "run_shell",
                {"argv": [sys.executable, "-c", "print('never')"]},
            ),
            context(tmp_path),
        )
    )
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_spawn_failure_without_cancellation_stays_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.process as process_runner

    async def fail_spawn(*_: object, **__: object):
        raise ValueError("sensitive spawn detail")

    monkeypatch.setattr(process_runner.asyncio, "create_subprocess_exec", fail_spawn)

    with pytest.raises(ToolFailure, match="Process could not be started") as captured:
        await RunShellTool().execute(
            ToolCall(
                "plain-spawn-failure",
                "run_shell",
                {"argv": [sys.executable, "-c", "print('never')"]},
            ),
            context(tmp_path),
        )

    assert "sensitive spawn detail" not in str(captured.value)


@pytest.mark.asyncio
async def test_literal_search_finds_late_and_cross_chunk_matches(
    tmp_path: Path,
) -> None:
    from litecoder.tools.builtin import SearchTextTool

    late_query = "late-needle"
    cross_query = "cross-boundary"
    late_offset = MAX_SEARCH_LINE_BYTES + 1005
    cross_offset = PROCESS_READ_CHUNK_BYTES - 3
    (tmp_path / "late.txt").write_text(
        ("x" * late_offset) + late_query + ("y" * 20), encoding="utf-8"
    )
    (tmp_path / "cross.txt").write_text(
        ("x" * cross_offset) + cross_query + ("y" * 20), encoding="utf-8"
    )

    late = await SearchTextTool().execute(
        ToolCall("late", "search_text", {"query": late_query, "glob": "late.txt"}),
        context(tmp_path),
    )
    cross = await SearchTextTool().execute(
        ToolCall(
            "cross",
            "search_text",
            {"query": cross_query, "glob": "cross.txt"},
        ),
        context(tmp_path),
    )

    assert late.metadata["matches"][0]["column"] == late_offset + 1
    assert late.metadata["matches"][0]["line_truncated"] is True
    assert cross.metadata["matches"][0]["column"] == cross_offset + 1
    assert cross.metadata["matches"][0]["line_truncated"] is True


@pytest.mark.asyncio
async def test_oversized_regex_line_without_prefix_match_reports_incomplete(
    tmp_path: Path,
) -> None:
    from litecoder.tools.builtin import SearchTextTool

    (tmp_path / "huge.txt").write_text(
        "x" * (MAX_SEARCH_LINE_BYTES + 5000), encoding="utf-8"
    )

    result = await SearchTextTool().execute(
        ToolCall(
            "regex-incomplete",
            "search_text",
            {"query": "needle", "regex": True, "glob": "huge.txt"},
        ),
        context(tmp_path),
    )

    assert result.metadata["matches"] == []
    assert result.metadata["search_incomplete"] is True
    assert result.metadata["truncated"] is True


def test_streaming_capture_reserves_utf8_worst_case_pending_bytes() -> None:
    from litecoder.tools.builtin.process import (
        MAX_STREAM_SECRET_BYTES,
        _BoundedRedactedCapture,
    )

    assert MAX_STREAM_SECRET_BYTES <= MAX_OUTPUT_BYTES // 4
    secret = "s" * MAX_STREAM_SECRET_BYTES
    capture = _BoundedRedactedCapture(SecretRedactor.with_values((secret,)))
    payload = ("😀" * (MAX_OUTPUT_BYTES // 2)).encode("utf-8")
    for index in range(0, len(payload), PROCESS_READ_CHUNK_BYTES):
        capture.feed(payload[index : index + PROCESS_READ_CHUNK_BYTES])
        assert len(capture._pending.encode("utf-8")) <= MAX_OUTPUT_BYTES


@pytest.mark.skipif(os.name != "nt", reason="Windows handle cleanup behavior")
@pytest.mark.asyncio
async def test_windows_failed_write_marks_validated_temp_handle_for_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    marked_handles: list[int] = []
    original_mark = secure_path._win_mark_delete

    def fail_rename(*_: object) -> None:
        raise OSError("rename failed")

    def record_mark(handle) -> None:
        assert handle.value
        marked_handles.append(handle.value)
        original_mark(handle)

    def forbid_path_delete(*_: object) -> bool:
        raise AssertionError("path-based temp deletion must not run")

    monkeypatch.setattr(secure_path, "_win_rename_handle", fail_rename)
    monkeypatch.setattr(secure_path, "_win_mark_delete", record_mark)
    monkeypatch.setattr(secure_path._kernel32, "DeleteFileW", forbid_path_delete)

    with pytest.raises(ToolFailure, match="could not be written"):
        await WriteFileTool().execute(
            ToolCall(
                "held-temp-cleanup",
                "write_file",
                {"path": "target.txt", "content": "new"},
            ),
            context(tmp_path),
        )

    assert marked_handles
    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".target.txt.litecoder-*.tmp"))


@pytest.mark.skipif(os.name != "nt", reason="Windows handle cleanup behavior")
@pytest.mark.asyncio
async def test_windows_unconfirmed_handle_cleanup_is_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path
    from litecoder.tools import ToolPartialFailure

    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")

    def fail_rename(*_: object) -> None:
        raise OSError("rename failed")

    def fail_mark(*_: object) -> None:
        raise OSError("disposition failed")

    monkeypatch.setattr(secure_path, "_win_rename_handle", fail_rename)
    monkeypatch.setattr(secure_path, "_win_mark_delete", fail_mark)

    with pytest.raises(ToolPartialFailure) as captured:
        await WriteFileTool().execute(
            ToolCall(
                "unconfirmed-cleanup",
                "write_file",
                {"path": "target.txt", "content": "new"},
            ),
            context(tmp_path),
        )

    assert captured.value.changed_workspace is True
    assert captured.value.metadata["phase"] == "cleanup"
