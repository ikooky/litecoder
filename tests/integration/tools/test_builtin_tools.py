from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from litecoder.hooks import HookManager
from litecoder.tools import (
    DuplicateGuard,
    PermissionService,
    PromptChoice,
    ToolCall,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    WorkspaceStateRegistry,
)
from litecoder.tools.builtin import (
    EditFileTool,
    GitDiffTool,
    GitStatusTool,
    GlobFilesTool,
    ReadFileTool,
    RunShellTool,
    SearchTextTool,
    WriteFileTool,
    builtin_tools,
)


class NullTrace:
    async def record(self, fact: Mapping[str, object]) -> None:
        return None


def context(root: Path, **kwargs: object) -> ToolContext:
    return ToolContext(
        "agent",
        "workspace",
        root,
        metadata={"round_number": 1, "permission_mode": "ask"},
        **kwargs,
    )


def executor_for(tool: object, *, prompt=None) -> tuple[ToolExecutor, WorkspaceStateRegistry]:
    registry = ToolRegistry()
    registry.register(tool)  # type: ignore[arg-type]
    workspaces = WorkspaceStateRegistry()
    executor = ToolExecutor(
        registry,
        HookManager(trace_hook=NullTrace()),
        DuplicateGuard(annotation=lambda **_: None),
        PermissionService(prompt=prompt or (lambda _: PromptChoice.ALLOW_ONCE)),
        workspaces,
    )
    return executor, workspaces


def call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(f"call-{name}", name, arguments)


def git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=True
    )


def test_builtin_tool_module_exports_exact_tool_set_and_specs() -> None:
    tools = builtin_tools()
    assert [tool.spec.name for tool in tools] == [
        "read_file", "write_file", "edit_file", "glob_files",
        "search_text", "run_shell", "git_status", "git_diff",
    ]
    assert [tool.spec.concurrency for tool in tools] == [
        "shared", "exclusive", "exclusive", "shared",
        "shared", "exclusive", "shared", "shared",
    ]
    assert [tool.spec.permission_risk for tool in tools] == [
        "safe", "workspace", "workspace", "safe",
        "safe", "high", "safe", "safe",
    ]
    assert all(tool.spec.input_schema["type"] == "object" for tool in tools)


def test_tool_context_snapshots_secrets_without_repr_leak(tmp_path: Path) -> None:
    names = ["API_TOKEN"]
    values = ["top-secret"]
    tool_context = context(
        tmp_path, secret_environment_names=names, secret_values=values
    )
    names.append("LATE")
    values[0] = "changed"
    assert tool_context.secret_environment_names == ("API_TOKEN",)
    assert tool_context.secret_values == ("top-secret",)
    assert tool_context.redactor.redact_text("Bearer abc top-secret") == "[REDACTED] [REDACTED]"
    assert "top-secret" not in repr(tool_context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (ReadFileTool(), {"path": "../outside.txt"}),
        (WriteFileTool(), {"path": "/absolute.txt", "content": "x"}),
        (WriteFileTool(), {"path": "C:\\outside.txt", "content": "x"}),
        (GlobFilesTool(), {"pattern": "../*.py"}),
        (SearchTextTool(), {"query": "x", "glob": "../*.py"}),
        (RunShellTool(), {"argv": [sys.executable, "-c", "print(1)"], "cwd": ".."}),
        (GitDiffTool(), {"path": "../outside.txt"}),
    ],
)
async def test_workspace_escape_is_denied_before_prompt(
    tmp_path: Path, tool: object, arguments: dict[str, object]
) -> None:
    prompts = 0

    async def prompt(_):
        nonlocal prompts
        prompts += 1
        return PromptChoice.ALLOW_ONCE

    executor, workspaces = executor_for(tool, prompt=prompt)
    result = await executor.execute(
        ToolCall("unsafe", tool.spec.name, arguments),  # type: ignore[attr-defined]
        context(tmp_path),
    )
    assert result.status == "denied"
    assert result.tool_call_id == "unsafe"
    assert prompts == 0
    assert workspaces.get("workspace").version == 0


@pytest.mark.asyncio
async def test_symlink_escape_is_denied_and_not_read(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("do-not-read", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink unavailable: {error.__class__.__name__}")
    executor, _ = executor_for(ReadFileTool())
    result = await executor.execute(
        call("read_file", path="link/secret.txt"), context(tmp_path)
    )
    assert result.status == "denied"
    assert "do-not-read" not in result.content


@pytest.mark.asyncio
async def test_read_file_lines_unicode_and_redaction(tmp_path: Path) -> None:
    (tmp_path / "space ü.txt").write_text(
        "zero\none top-secret\ntwo\nthree\n", encoding="utf-8", newline=""
    )
    executor, _ = executor_for(ReadFileTool())
    result = await executor.execute(
        call("read_file", path="space ü.txt", offset=1, limit=2),
        context(tmp_path, secret_values=["top-secret"]),
    )
    assert result.status == "success"
    assert result.content == "one [REDACTED]\ntwo\n"
    assert result.metadata == {
        "path": "space ü.txt", "size": 30, "line_offset": 1,
        "line_start": 2, "line_end": 3, "total_lines": 4,
        "truncated": True, "changed_workspace": False,
        "preview": "one [REDACTED]\ntwo\n",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"a\x00b", b"\xff\xfe"])
async def test_read_file_rejects_binary_or_invalid_utf8(
    tmp_path: Path, payload: bytes
) -> None:
    (tmp_path / "bad.dat").write_bytes(payload)
    executor, _ = executor_for(ReadFileTool())
    result = await executor.execute(call("read_file", path="bad.dat"), context(tmp_path))
    assert result.status == "tool_error"
    assert result.metadata["changed_workspace"] is False
    assert payload.hex() not in result.content


@pytest.mark.asyncio
async def test_write_file_noop_and_exact_atomic_replacement(tmp_path: Path) -> None:
    target = tmp_path / "same.txt"
    target.write_bytes(b"same\r\n")
    executor, workspaces = executor_for(WriteFileTool())
    unchanged = await executor.execute(
        call("write_file", path="same.txt", content="same\r\n"), context(tmp_path)
    )
    assert unchanged.status == "success"
    assert unchanged.metadata["changed_workspace"] is False
    assert workspaces.get("workspace").version == 0
    changed = await executor.execute(
        ToolCall("write-2", "write_file", {"path": "same.txt", "content": "new\n"}),
        context(tmp_path),
    )
    assert changed.status == "success"
    assert target.read_bytes() == b"new\n"
    assert changed.metadata["changed_workspace"] is True
    assert workspaces.get("workspace").version == 1
    assert not list(tmp_path.glob(".same.txt.litecoder-*.tmp"))


@pytest.mark.asyncio
async def test_atomic_write_replace_failure_cleans_temp_and_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")

    if os.name == "nt":
        def fail_rename(*_args, **_kwargs) -> None:
            raise OSError("outside-secret")

        monkeypatch.setattr(secure_path, "_win_rename_handle", fail_rename)
    else:
        def fail_replace(*_args, **_kwargs) -> None:
            raise OSError("outside-secret")

        monkeypatch.setattr(secure_path.os, "replace", fail_replace)
    executor, workspaces = executor_for(WriteFileTool())
    result = await executor.execute(
        call("write_file", path="target.txt", content="new"), context(tmp_path)
    )
    assert result.status == "tool_error"
    assert result.metadata["changed_workspace"] is False
    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".target.txt.litecoder-*.tmp"))
    assert workspaces.get("workspace").version == 0
    assert "outside-secret" not in result.content


@pytest.mark.asyncio
async def test_post_replace_durability_failure_is_partial_and_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.secure_path as secure_path

    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    calls = 0
    if os.name == "nt":
        original_validate = secure_path._win_validate_parent_state

        def fail_after_replace(state) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("durability-secret")
            original_validate(state)

        monkeypatch.setattr(secure_path, "_win_validate_parent_state", fail_after_replace)
    else:
        original_fsync = secure_path.os.fsync

        def fail_parent_fsync(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("durability-secret")
            original_fsync(descriptor)

        monkeypatch.setattr(secure_path.os, "fsync", fail_parent_fsync)
    executor, workspaces = executor_for(WriteFileTool())
    result = await executor.execute(
        call("write_file", path="target.txt", content="new"), context(tmp_path)
    )
    assert result.status == "partial_failure"
    assert result.metadata["changed_workspace"] is True
    assert result.metadata["automatic_retry"] is False
    assert target.read_text(encoding="utf-8") == "new"
    assert workspaces.get("workspace").version == 1
    assert "durability-secret" not in result.content


@pytest.mark.asyncio
async def test_edit_file_requires_one_match_unless_replace_all(tmp_path: Path) -> None:
    target = tmp_path / "edit.txt"
    target.write_text("old old", encoding="utf-8")
    executor, workspaces = executor_for(EditFileTool())
    ambiguous = await executor.execute(
        call("edit_file", path="edit.txt", old_text="old", new_text="new"),
        context(tmp_path),
    )
    assert ambiguous.status == "tool_error"
    assert ambiguous.metadata["occurrences"] == 2
    assert target.read_text(encoding="utf-8") == "old old"
    replaced = await executor.execute(
        ToolCall("edit-2", "edit_file", {
            "path": "edit.txt", "old_text": "old", "new_text": "new",
            "replace_all": True,
        }),
        context(tmp_path),
    )
    assert replaced.status == "success"
    assert replaced.metadata["replacements"] == 2
    assert target.read_text(encoding="utf-8") == "new new"
    assert workspaces.get("workspace").version == 1


@pytest.mark.asyncio
async def test_glob_is_sorted_limited_and_does_not_follow_symlink_dirs(tmp_path: Path) -> None:
    (tmp_path / "z.py").write_text("z", encoding="utf-8")
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "b.py").write_text("b", encoding="utf-8")
    try:
        (tmp_path / "linked").symlink_to(tmp_path / "dir", target_is_directory=True)
    except OSError:
        pass
    executor, _ = executor_for(GlobFilesTool())
    result = await executor.execute(
        call("glob_files", pattern="**/*.py", limit=2), context(tmp_path)
    )
    assert result.status == "success"
    assert result.content.splitlines() == ["a.py", "dir/b.py"]
    assert result.metadata["truncated"] is True
    assert result.metadata["count"] == 2
    assert "linked/b.py" not in result.content


@pytest.mark.asyncio
async def test_search_text_is_portable_deterministic_and_safe(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("Needle needle\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("zero\nneedle here\n", encoding="utf-8")
    (tmp_path / "binary.txt").write_bytes(b"needle\x00secret")
    executor, _ = executor_for(SearchTextTool())
    literal = await executor.execute(
        call("search_text", query="needle", glob="*.txt", case_sensitive=False, limit=2),
        context(tmp_path),
    )
    assert literal.status == "success"
    assert literal.content.splitlines() == [
        "a.txt:2:1:needle here", "b.txt:1:1:Needle needle",
    ]
    assert literal.metadata["truncated"] is True
    invalid = await executor.execute(
        ToolCall("regex", "search_text", {"query": "[", "regex": True}),
        context(tmp_path),
    )
    assert invalid.status == "tool_error"
    assert invalid.content == "Invalid search pattern"


@pytest.mark.asyncio
async def test_shell_uses_argv_without_injection_and_isolates_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITECODER_TEST_SECRET", "env-secret")
    marker = tmp_path / "injected.txt"
    script = (
        "import os,sys;print(sys.argv[1]);"
        "print(os.environ.get('LITECODER_TEST_SECRET'));"
        "print('Bearer bearer-secret output-secret')"
    )
    dangerous = f"hello; echo injected > {marker}"
    executor, workspaces = executor_for(RunShellTool())
    result = await executor.execute(
        call("run_shell", argv=[sys.executable, "-c", script, dangerous]),
        context(
            tmp_path, secret_environment_names=["LITECODER_TEST_SECRET"],
            secret_values=["output-secret"],
        ),
    )
    assert result.status == "success"
    assert dangerous in result.content
    assert "None" in result.content
    assert "bearer-secret" not in result.content
    assert "output-secret" not in result.content
    assert "[REDACTED]" in result.content
    assert not marker.exists()
    assert result.metadata["exit_code"] == 0
    assert workspaces.get("workspace").version == 1


@pytest.mark.asyncio
async def test_shell_nonzero_and_timeout_are_uncertain_partial_failures(tmp_path: Path) -> None:
    executor, workspaces = executor_for(RunShellTool())
    failed = await executor.execute(
        call("run_shell", argv=[
            sys.executable, "-c",
            "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)",
        ]), context(tmp_path),
    )
    assert failed.status == "partial_failure"
    assert failed.metadata["exit_code"] == 7
    assert failed.metadata["stdout"] == "out\n"
    assert failed.metadata["stderr"] == "err\n"
    assert failed.metadata["automatic_retry"] is False
    assert workspaces.get("workspace").version == 1
    timed_out = await executor.execute(
        ToolCall("timeout", "run_shell", {
            "argv": [sys.executable, "-c", "import threading; threading.Event().wait()"],
            "timeout": 0.05,
        }), context(tmp_path),
    )
    assert timed_out.status == "partial_failure"
    assert timed_out.metadata["timed_out"] is True
    assert workspaces.get("workspace").version == 2


@pytest.mark.asyncio
async def test_shell_cancellation_kills_and_reaps_process(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid.txt"
    tool = RunShellTool()
    task = asyncio.create_task(tool.execute(
        call("run_shell", argv=[
            sys.executable, "-c",
            "import os,pathlib,threading; pathlib.Path('pid.txt').write_text(str(os.getpid())); threading.Event().wait()",
        ]), context(tmp_path),
    ))
    for _ in range(1000):
        if pid_file.exists():
            break
        await asyncio.sleep(0.01)
    assert pid_file.exists()
    pid = int(pid_file.read_text(encoding="utf-8"))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(1000):
        if not _pid_exists(pid):
            break
        await asyncio.sleep(0.01)
    assert not _pid_exists(pid)


@pytest.mark.asyncio
async def test_git_status_and_diff_support_staged_unicode_and_path(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "Lite Coder")
    first = tmp_path / "space ü.txt"
    other = tmp_path / "other.txt"
    first.write_text("one\n", encoding="utf-8")
    other.write_text("base\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "initial")
    first.write_text("two\n", encoding="utf-8")
    other.write_text("changed\n", encoding="utf-8")
    git(tmp_path, "add", str(first.name))
    status_executor, _ = executor_for(GitStatusTool())
    status = await status_executor.execute(call("git_status"), context(tmp_path))
    assert status.status == "success"
    assert "space ü.txt" in status.content
    assert "other.txt" in status.content
    diff_executor, _ = executor_for(GitDiffTool())
    staged = await diff_executor.execute(
        call("git_diff", staged=True, path="space ü.txt"), context(tmp_path)
    )
    assert staged.status == "success"
    assert "+two" in staged.content
    assert "other.txt" not in staged.content
    unstaged = await diff_executor.execute(
        ToolCall("diff-2", "git_diff", {"path": "other.txt"}), context(tmp_path)
    )
    assert "+changed" in unstaged.content
    assert "space ü.txt" not in unstaged.content


class _FakeStream:
    async def read(self, size: int) -> bytes:
        assert size == 16_384
        return b""


class _FakeGitProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.pid = 12345
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()

    async def communicate(self) -> tuple[bytes, bytes]:
        raise AssertionError("shared runner must not call communicate")

    async def wait(self) -> int:
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = -1


class _FakeJob:
    def assign(self, pid: int) -> None:
        assert pid == 12345

    def resume(self, pid: int) -> None:
        assert pid == 12345

    def terminate(self) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_git_uses_exact_inert_argv_and_offline_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litecoder.tools.builtin.process as process_runner

    protected_name = "LITECODER_GIT_PROTECTED"
    monkeypatch.setenv(protected_name, "protected-secret")
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "1")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "1")
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "0")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "0")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "malicious-global-config")
    monkeypatch.setenv("GIT_PAGER", "malicious-pager")
    monkeypatch.setenv("GIT_EDITOR", "malicious-editor")
    captured: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def spawn(*argv: str, **kwargs: object) -> _FakeGitProcess:
        captured.append((argv, kwargs))
        return _FakeGitProcess()

    monkeypatch.setattr(
        process_runner.asyncio, "create_subprocess_exec", spawn
    )
    if os.name == "nt":
        monkeypatch.setattr(
            process_runner._WindowsJob, "create", lambda: _FakeJob()
        )
    (tmp_path / "space ü.txt").write_text("content", encoding="utf-8")
    tool_context = context(
        tmp_path,
        secret_environment_names=(protected_name,),
    )

    assert (
        await GitStatusTool().execute(
            call("git_status", porcelain="v2"), tool_context
        )
    ).status == "success"
    assert (
        await GitDiffTool().execute(
            call("git_diff", staged=True, path="space ü.txt"),
            tool_context,
        )
    ).status == "success"

    assert [list(argv) for argv, _ in captured] == [
        [
            "git",
            "--no-pager",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.quotepath=false",
            "status",
            "--porcelain=v2",
            "--untracked-files=all",
        ],
        [
            "git",
            "--no-pager",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.quotepath=false",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--cached",
            "--",
            "space ü.txt",
        ],
    ]
    expected_git_environment = {
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_PAGER": "cat",
        "GIT_EDITOR": "true",
    }
    for _, kwargs in captured:
        if os.name == "nt":
            assert kwargs["creationflags"] & process_runner._CREATE_SUSPENDED
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert protected_name not in environment
        assert {
            key: environment[key] for key in expected_git_environment
        } == expected_git_environment


@pytest.mark.asyncio
async def test_git_reads_leave_index_unchanged_and_never_run_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "Lite Coder")
    tracked = tmp_path / "tracked.txt"
    attributes = tmp_path / ".gitattributes"
    tracked.write_text("base\n", encoding="utf-8")
    attributes.write_text("tracked.txt diff=malicious\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "initial")
    tracked.write_text("changed\n", encoding="utf-8")

    markers = {
        name: tmp_path / f"{name}.invoked"
        for name in (
            "repo-fsmonitor",
            "repo-diff",
            "repo-textconv",
            "global-fsmonitor",
            "global-diff",
        )
    }
    helpers = {
        name: _malicious_git_helper(tmp_path, name, marker)
        for name, marker in markers.items()
    }
    git(tmp_path, "config", "core.fsmonitor", helpers["repo-fsmonitor"])
    git(
        tmp_path,
        "config",
        "diff.malicious.command",
        helpers["repo-diff"],
    )
    git(
        tmp_path,
        "config",
        "diff.malicious.textconv",
        helpers["repo-textconv"],
    )
    global_config = tmp_path / "malicious-global.gitconfig"
    git(
        tmp_path,
        "config",
        "--file",
        str(global_config),
        "core.fsmonitor",
        helpers["global-fsmonitor"],
    )
    git(
        tmp_path,
        "config",
        "--file",
        str(global_config),
        "diff.external",
        helpers["global-diff"],
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))

    index = tmp_path / ".git" / "index"
    before_bytes = index.read_bytes()
    before_mtime_ns = index.stat().st_mtime_ns

    status = await GitStatusTool().execute(call("git_status"), context(tmp_path))
    diff = await GitDiffTool().execute(
        call("git_diff", path="tracked.txt"), context(tmp_path)
    )

    assert status.status == "success"
    assert diff.status == "success"
    assert "+changed" in diff.content
    assert index.read_bytes() == before_bytes
    assert index.stat().st_mtime_ns == before_mtime_ns
    assert all(not marker.exists() for marker in markers.values())


def _malicious_git_helper(
    root: Path, name: str, marker: Path
) -> str:
    if os.name == "nt":
        helper = root / f"{name}.bat"
        helper.write_text(
            f'@echo off\r\n> "{marker}" echo invoked\r\n',
            encoding="utf-8",
            newline="",
        )
        return subprocess.list2cmdline([str(helper)])

    helper = root / f"{name}.sh"
    helper.write_text(
        "#!/bin/sh\n"
        f"printf invoked > {shlex.quote(str(marker))}\n",
        encoding="utf-8",
        newline="",
    )
    helper.chmod(0o700)
    return shlex.quote(str(helper))

@pytest.mark.asyncio
async def test_git_non_repo_is_safe_unchanged_failure(tmp_path: Path) -> None:
    executor, workspaces = executor_for(GitStatusTool())
    result = await executor.execute(call("git_status"), context(tmp_path))
    assert result.status == "partial_failure"
    assert result.metadata["changed_workspace"] is False
    assert result.metadata["automatic_retry"] is False
    assert workspaces.get("workspace").version == 0
    assert str(tmp_path.parent) not in result.content


def _pid_exists(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)
        )
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel32.OpenProcess(0x1000, 0, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
