from __future__ import annotations

import asyncio
import os
import sys
import threading
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from litecoder.tools import ToolCall, ToolContext, ToolFailure
from litecoder.tools.builtin import RunShellTool


def context(root: Path) -> ToolContext:
    return ToolContext(
        "agent",
        "workspace",
        root,
        metadata={"round_number": 1, "permission_mode": "ask"},
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


async def wait_for_file(path: Path, task: asyncio.Task[object]) -> None:
    for _ in range(500):
        if path.exists():
            return
        if task.done():
            await task
            raise AssertionError(f"shell command exited before creating {path.name}")
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path.name}")


async def wait_for_exit(*pids: int) -> None:
    for _ in range(500):
        if all(not pid_exists(pid) for pid in pids):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"processes remain alive: {pids}")


@pytest.mark.asyncio
async def test_cancellation_kills_root_and_grandchild(tmp_path: Path) -> None:
    root_pid = tmp_path / "root.pid"
    child_pid = tmp_path / "child.pid"
    child_code = (
        "import os,pathlib,threading;"
        "pathlib.Path('child.pid').write_text(str(os.getpid()));"
        "threading.Event().wait()"
    )
    root_code = (
        "import os,pathlib,subprocess,sys,threading;"
        "pathlib.Path('root.pid').write_text(str(os.getpid()));"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        "threading.Event().wait()"
    )
    task = asyncio.create_task(
        RunShellTool().execute(
            ToolCall(
                "tree-cancel",
                "run_shell",
                {"argv": [sys.executable, "-c", root_code]},
            ),
            context(tmp_path),
        )
    )
    try:
        await wait_for_file(root_pid, task)
        await wait_for_file(child_pid, task)
        pids = (
            int(root_pid.read_text(encoding="utf-8")),
            int(child_pid.read_text(encoding="utf-8")),
        )

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await wait_for_exit(*pids)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_of_stalled_spawn_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.process as process_runner

    entered = asyncio.Event()
    never = asyncio.Event()

    async def stalled_spawn(*_: object, **__: object) -> object:
        entered.set()
        await never.wait()
        raise AssertionError("stalled spawn unexpectedly completed")

    monkeypatch.setattr(
        process_runner.asyncio, "create_subprocess_exec", stalled_spawn
    )
    monkeypatch.setattr(process_runner, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    task = asyncio.create_task(
        RunShellTool().execute(
            ToolCall(
                "stalled-spawn",
                "run_shell",
                {"argv": [sys.executable, "-c", "pass"]},
            ),
            context(tmp_path),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory descriptor behavior")
@pytest.mark.asyncio
async def test_macos_spawn_fchdirs_to_validated_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim = workspace / "victim"
    victim.mkdir()
    marker = victim / "marker.txt"

    monkeypatch.setattr(secure_path.sys, "platform", "darwin")
    result = await RunShellTool().execute(
        ToolCall(
            "darwin-fchdir",
            "run_shell",
            {
                "argv": [
                    sys.executable,
                    "-c",
                    "import pathlib; pathlib.Path('marker.txt').write_text('inside')",
                ],
                "cwd": "victim",
            },
        ),
        context(workspace),
    )

    assert result.status == "success"
    assert marker.read_text(encoding="utf-8") == "inside"


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object behavior")
async def test_windows_job_assignment_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.process as process_runner

    pid_path = tmp_path / "job-failure.pid"

    def fail_job() -> object:
        raise OSError("job-secret")

    monkeypatch.setattr(process_runner._WindowsJob, "create", fail_job)
    script = (
        "import os,pathlib,threading;"
        "pathlib.Path('job-failure.pid').write_text(str(os.getpid()));"
        "threading.Event().wait()"
    )
    with pytest.raises(ToolFailure, match="could not be contained") as raised:
        await RunShellTool().execute(
            ToolCall(
                "job-failure",
                "run_shell",
                {"argv": [sys.executable, "-c", script]},
            ),
            context(tmp_path),
        )

    assert "job-secret" not in str(raised.value)
    if pid_path.exists():
        await wait_for_exit(int(pid_path.read_text(encoding="utf-8")))

def _directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        if os.name != "nt":
            raise
    completed = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        pytest.skip("account cannot create a directory link")


@pytest.mark.asyncio
async def test_shell_cwd_remains_pinned_during_ancestor_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim = workspace / "victim"
    victim.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    entered = threading.Event()
    release = threading.Event()

    if os.name == "nt":
        original = secure_path._win_open_existing

        def open_with_barrier(path: Path, *, directory: bool):
            handle = original(path, directory=directory)
            if directory and path.name == "victim":
                entered.set()
                assert release.wait(timeout=5)
            return handle

        monkeypatch.setattr(secure_path, "_win_open_existing", open_with_barrier)
    else:
        original = secure_path.os.open

        def open_with_barrier(path, flags, mode=0o777, *, dir_fd=None):
            descriptor = original(path, flags, mode, dir_fd=dir_fd)
            if path == "victim" and dir_fd is not None:
                entered.set()
                assert release.wait(timeout=5)
            return descriptor

        monkeypatch.setattr(secure_path.os, "open", open_with_barrier)

    script = "import pathlib; pathlib.Path('marker.txt').write_text('inside')"
    operation = asyncio.create_task(
        asyncio.to_thread(
            lambda: asyncio.run(
                RunShellTool().execute(
                    ToolCall(
                        "cwd-race",
                        "run_shell",
                        {"argv": [sys.executable, "-c", script], "cwd": "victim"},
                    ),
                    context(workspace),
                )
            )
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    swapped = False
    try:
        victim.rename(workspace / "parked")
    except PermissionError:
        assert os.name == "nt"
    else:
        swapped = True
        _directory_link(victim, outside)
    release.set()
    result = await operation

    assert result.status == "success"
    pinned = workspace / "parked" if swapped else victim
    assert (pinned / "marker.txt").read_text(encoding="utf-8") == "inside"
    assert not (outside / "marker.txt").exists()

@pytest.mark.asyncio
async def test_output_drain_failure_terminates_and_reaps_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.process as process_runner

    real_spawn = process_runner.asyncio.create_subprocess_exec
    spawned_pid: list[int] = []

    async def tracked_spawn(*argv: str, **kwargs: object):
        process = await real_spawn(*argv, **kwargs)
        spawned_pid.append(process.pid)
        return process

    async def failing_drain(*_: object) -> None:
        raise OSError("drain-secret")

    monkeypatch.setattr(
        process_runner.asyncio, "create_subprocess_exec", tracked_spawn
    )
    monkeypatch.setattr(process_runner, "_drain", failing_drain)
    script = "import threading; threading.Event().wait()"

    with pytest.raises(ToolFailure, match="output capture failed") as raised:
        await RunShellTool().execute(
            ToolCall(
                "drain-failure",
                "run_shell",
                {"argv": [sys.executable, "-c", script]},
            ),
            context(tmp_path),
        )

    assert "drain-secret" not in str(raised.value)
    assert len(spawned_pid) == 1
    await wait_for_exit(spawned_pid[0])
