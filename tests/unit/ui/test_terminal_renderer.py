from __future__ import annotations

from contextlib import contextmanager
from io import StringIO
from pathlib import Path

from rich.console import Console

from litecoder.ui import terminal_state
from litecoder.ui.events import RuntimeUIEvent, UIEventType
from litecoder.ui.renderers.terminal import TerminalRenderer


def console_and_renderer(
    width: int = 80,
    *,
    color_system: str | None = None,
) -> tuple[StringIO, TerminalRenderer]:
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        color_system=color_system,
        width=width,
    )
    return stream, TerminalRenderer(console)


def event(event_type: UIEventType, payload: dict[str, object]) -> RuntimeUIEvent:
    return RuntimeUIEvent(event_type, sequence=1, timestamp=1.0, payload=payload)


def test_terminal_renderer_renders_startup_banner_with_workspace_panel_only() -> None:
    stream, renderer = console_and_renderer(width=120)
    workspace = str((Path.cwd() / "project").resolve())

    renderer.startup_banner(workspace, "claude-sonnet-4")

    output = stream.getvalue()
    assert "Welcome to LiteCoder CLI!" in output
    assert f"Workspace: {workspace}" in output
    assert "`nUsing" not in output
    assert "Using claude-sonnet-4 (from .litecoder\\config.toml)" in output
    assert "LITE CODER" not in output
    assert "TTTTT" not in output


def test_terminal_renderer_streams_and_finalizes_assistant_markdown_in_bullet_column() -> None:
    stream, renderer = console_and_renderer()

    renderer.emit(event(UIEventType.ASSISTANT_DELTA, {"text": "## Title\n\n"}))
    renderer.emit(event(UIEventType.ASSISTANT_DELTA, {"text": "- item"}))
    renderer.emit(event(UIEventType.ASSISTANT_COMPLETED, {"text": "## Title\n\n- item"}))
    renderer.flush()

    output = stream.getvalue()
    assert "\u25cf" in output
    assert "Title" in output
    assert "item" in output
    assert "## Title" not in output


def test_terminal_renderer_prints_completed_assistant_once_after_repeated_flushes() -> None:
    stream = StringIO()
    renderer = TerminalRenderer(Console(file=stream, force_terminal=False, width=80))
    sentinel = "unique eval final sentinel"

    renderer.emit(event(UIEventType.TURN_STARTED, {"prompt": "evaluate"}))
    renderer.emit(event(UIEventType.ASSISTANT_DELTA, {"text": "unique eval "}))
    renderer.emit(event(UIEventType.ASSISTANT_DELTA, {"text": "final sentinel"}))
    renderer.emit(event(UIEventType.ASSISTANT_COMPLETED, {"text": sentinel}))
    renderer.emit(event(UIEventType.TURN_FINISHED, {
        "status": "completed",
        "reason": "end_turn",
        "elapsed_seconds": 1.0,
    }))
    renderer.flush()
    renderer.flush()

    assert stream.getvalue().count(sentinel) == 1


def test_terminal_renderer_folds_long_assistant_text_without_ellipsis() -> None:
    stream, renderer = console_and_renderer(width=16)
    message = "分析完成，下面是这个工作区的完整架构分析结果。"

    renderer.emit(event(UIEventType.ASSISTANT_COMPLETED, {"text": message}))

    output = stream.getvalue()
    assert "…" not in output
    assert message in "".join(output.split())

def test_terminal_renderer_folds_markdown_table_cells_without_ellipsis() -> None:
    stream, renderer = console_and_renderer(width=24)
    value = "x" * 60
    message = f"| Name | Value |\n| --- | --- |\n| long | {value} |"

    renderer.emit(event(UIEventType.ASSISTANT_COMPLETED, {"text": message}))

    output = stream.getvalue()
    assert "…" not in output
    assert output.count("x") == len(value)



def test_terminal_renderer_prints_assistant_output_while_live_input_is_suspended(
    monkeypatch,
) -> None:
    stream, renderer = console_and_renderer()
    snapshots: list[tuple[str, str]] = []

    @contextmanager
    def suspended(console):  # type: ignore[no-untyped-def]
        assert console is renderer.console
        snapshots.append(("enter", stream.getvalue()))
        yield
        snapshots.append(("exit", stream.getvalue()))

    monkeypatch.setattr(terminal_state, "suspend_waiting_status", suspended)

    renderer.emit(event(UIEventType.ASSISTANT_COMPLETED, {"text": "final answer"}))

    assert snapshots[0] == ("enter", "")
    assert snapshots[1][0] == "exit"
    assert "final answer" in snapshots[1][1]


def test_terminal_renderer_replaces_process_lines_with_single_elapsed_summary() -> None:
    stream, renderer = console_and_renderer(width=120)
    readme_path = str((Path.cwd() / "README.md").resolve())

    renderer.emit(event(UIEventType.TURN_STARTED, {"prompt": "list files"}))
    renderer.emit(event(UIEventType.MODEL_REQUESTED, {"memory_count": 2}))
    renderer.emit(event(UIEventType.THINKING_DELTA, {"text": "private reasoning text"}))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_STARTED,
        sequence=2,
        timestamp=1.1,
        tool_call_id="call-1",
        tool_name="read_file",
        payload={"arguments": {"path": "README.md"}},
    ))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_FINISHED,
        sequence=3,
        timestamp=1.2,
        tool_call_id="call-1",
        tool_name="read_file",
        payload={"status": "success", "preview": "42 lines"},
    ))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_STARTED,
        sequence=4,
        timestamp=1.3,
        tool_call_id="call-2",
        tool_name="run_shell",
        payload={"arguments": {"command": "git status"}},
    ))
    renderer.emit(event(UIEventType.ASSISTANT_COMPLETED, {"text": "final answer"}))
    renderer.emit(event(UIEventType.TURN_FINISHED, {
        "status": "completed",
        "reason": "stop",
        "elapsed_seconds": 12.34,
        "total_tokens": 1493,
    }))

    output = stream.getvalue()
    assert "final answer" in output
    assert f"Read({readme_path})" in output
    assert "└ 42 lines" in output
    assert "Elapsed 12.3s, Tools called: 2, memories recalled: 2" in output
    assert "☭" not in output
    assert "private reasoning text" not in output
    assert "completed" not in output
    assert "stop" not in output
    assert "tokens=1493" not in output


def test_terminal_renderer_resolves_tool_paths_from_explicit_workspace(
    tmp_path: Path,
) -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=160)
    workspace = tmp_path / "run" / "cases" / "case-0001" / "execution"
    renderer = TerminalRenderer(console, workspace_root=workspace)
    renderer.emit(
        RuntimeUIEvent(
            UIEventType.TOOL_EXECUTION_STARTED,
            sequence=1,
            timestamp=1.0,
            tool_call_id="call-1",
            tool_name="write_file",
            payload={"arguments": {"path": "solution.py"}},
        )
    )
    renderer.emit(
        RuntimeUIEvent(
            UIEventType.TOOL_EXECUTION_FINISHED,
            sequence=2,
            timestamp=1.1,
            tool_call_id="call-1",
            tool_name="write_file",
            payload={"metadata": {"path": "solution.py", "changed": True}},
        )
    )

    expected = str((workspace / "solution.py").resolve())
    output = stream.getvalue()
    assert expected in output
    assert str((Path.cwd() / "solution.py").resolve()) not in output


def test_terminal_renderer_omits_zero_tool_and_memory_counts() -> None:
    stream, renderer = console_and_renderer()

    renderer.emit(event(UIEventType.TURN_STARTED, {"prompt": "who are you"}))
    renderer.emit(event(UIEventType.ASSISTANT_COMPLETED, {"text": "final answer"}))
    renderer.emit(event(UIEventType.TURN_FINISHED, {
        "status": "completed",
        "reason": "stop",
        "elapsed_seconds": 6.0,
    }))

    output = stream.getvalue()
    assert "Elapsed 6.0s" in output
    assert "☭" not in output
    assert "Tools called" not in output
    assert "memories recalled" not in output


def test_terminal_renderer_renders_tool_results_with_status_colors_and_abs_paths() -> None:
    stream, renderer = console_and_renderer(width=140, color_system="standard")
    relative_path = "src/litecoder/ui/terminal_state.py"
    absolute_path = str((Path.cwd() / relative_path).resolve())

    renderer.emit(event(UIEventType.TURN_STARTED, {"prompt": "inspect"}))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_STARTED,
        sequence=2,
        timestamp=1.1,
        tool_call_id="call-1",
        tool_name="read_file",
        payload={"arguments": {"path": relative_path}},
    ))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_FINISHED,
        sequence=3,
        timestamp=1.2,
        tool_call_id="call-1",
        tool_name="read_file",
        payload={"status": "success", "preview": "line one\nline two\nline three"},
    ))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_STARTED,
        sequence=4,
        timestamp=1.3,
        tool_call_id="call-2",
        tool_name="run_shell",
        payload={"arguments": {"command": "curl -s https://api.ipify.org?format=json"}},
    ))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_FAILED,
        sequence=5,
        timestamp=1.4,
        tool_call_id="call-2",
        tool_name="run_shell",
        payload={
            "status": "partial_failure",
            "message": "Shell command failed",
            "metadata": {"exit_code": 35},
        },
    ))

    output = stream.getvalue()
    assert "Read(" in output
    assert absolute_path in output
    assert "└" in output
    assert "line one" in output
    assert "... +2 lines" in output
    assert "Bash(curl -s https://api.ipify.org?format=json)" in output
    assert "Error: Exit code 35" in output
    assert "\x1b[1;32m" in output or "\x1b[32m" in output or "\x1b[92m" in output
    assert "\x1b[1;31m" in output or "\x1b[31m" in output or "\x1b[91m" in output



def test_terminal_renderer_keeps_waiting_status_until_assistant_answer(monkeypatch) -> None:
    stream, renderer = console_and_renderer()
    stops: list[Console] = []
    monkeypatch.setattr(
        terminal_state,
        "stop_waiting_status",
        lambda console: stops.append(console),
    )

    renderer.emit(event(UIEventType.TURN_STARTED, {"prompt": "inspect"}))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_STARTED,
        sequence=2,
        timestamp=1.1,
        tool_call_id="call-1",
        tool_name="run_shell",
        payload={"arguments": {"command": "git status"}},
    ))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_FINISHED,
        sequence=3,
        timestamp=1.2,
        tool_call_id="call-1",
        tool_name="run_shell",
        payload={"status": "success", "preview": "clean"},
    ))

    assert stops == []

    renderer.emit(event(UIEventType.ASSISTANT_COMPLETED, {"text": "final answer"}))

    assert stops == [renderer.console]
    assert "Bash(git status)" in stream.getvalue()


def test_terminal_renderer_renders_glob_title_and_results_as_absolute_paths() -> None:
    stream, renderer = console_and_renderer(width=160)
    pattern = "**/*"
    absolute_pattern = str(Path.cwd() / pattern)
    absolute_result = str((Path.cwd() / "src/litecoder/ui/terminal_state.py").resolve())

    renderer.emit(event(UIEventType.TURN_STARTED, {"prompt": "list"}))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_STARTED,
        sequence=2,
        timestamp=1.1,
        tool_call_id="call-1",
        tool_name="glob_files",
        payload={"arguments": {"pattern": pattern}},
    ))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_FINISHED,
        sequence=3,
        timestamp=1.2,
        tool_call_id="call-1",
        tool_name="glob_files",
        payload={
            "status": "success",
            "preview": "",
            "metadata": {"preview": ["src/litecoder/ui/terminal_state.py"]},
        },
    ))

    output = stream.getvalue()
    assert f"Glob({absolute_pattern})" in output
    assert absolute_result in output
    assert "Glob(**/*)" not in output


def test_terminal_renderer_renders_denied_shell_with_original_arguments() -> None:
    stream, renderer = console_and_renderer(width=160)
    renderer.emit(event(UIEventType.TURN_STARTED, {"prompt": "run"}))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_DENIED,
        sequence=2,
        timestamp=1.1,
        tool_call_id="call-1",
        tool_name="run_shell",
        payload={
            "reason": "Denied by workspace safety policy",
            "arguments": {"argv": ["cmd", "/c", "dir"], "cwd": "."},
        },
    ))

    output = stream.getvalue()
    assert "Bash(cmd /c dir)" in output
    assert "Bash(shell)" not in output
def test_terminal_renderer_shows_provider_error_and_non_completed_elapsed_status() -> None:
    stream, renderer = console_and_renderer()

    renderer.emit(RuntimeUIEvent(
        UIEventType.PROVIDER_ERROR,
        sequence=1,
        timestamp=1.0,
        request_id="req-1",
        payload={"code": "provider_transient", "message": "temporary", "retryable": True},
    ))
    renderer.emit(event(UIEventType.TURN_FINISHED, {
        "status": "incomplete",
        "reason": "provider_transient retry budget exhausted",
        "elapsed_seconds": 5.0,
        "total_tokens": 3,
    }))

    output = stream.getvalue()
    assert output.splitlines()[0] == "Retrying..."
    assert "temporary" not in output
    assert "req-1" not in output
    assert "Incomplete provider_transient retry budget exhausted" in output
    assert "☭" not in output
    assert "Elapsed 5.0s" in output
    assert "tokens=3" not in output


def test_terminal_renderer_renders_recalled_memory_only() -> None:
    stream, renderer = console_and_renderer(width=160)
    diagnostics = [
        {"operation": "load", "status": "recalled", "count": 2},
        {"operation": "extract", "status": "completed", "written": 1},
        {
            "operation": "extract",
            "status": "partial_rejected",
            "accepted": 1,
            "rejected": 1,
            "written": 1,
        },
        {"operation": "extract", "status": "truncated", "limit": 1200},
        {
            "operation": "extract",
            "status": "provider_failed",
            "code": "provider_rate_limit",
            "message": "secret provider message",
        },
        {"operation": "extract", "status": "malformed"},
        {
            "operation": "dream",
            "status": "completed",
            "before": 10,
            "after": 5,
        },
        {"operation": "selection", "status": "fallback", "reason": "timeout"},
        {"operation": "load", "status": "partial", "skipped": 9},
        {"operation": "extract", "status": "timeout", "prompt": "secret prompt"},
        {"operation": "dream", "status": "timeout", "prompt": "secret prompt"},
    ]

    for diagnostic in diagnostics:
        renderer.emit(event(UIEventType.DIAGNOSTIC, {"memory": diagnostic}))

    output = stream.getvalue()
    assert output.strip() == "Memory load: recalled (count=2)"
    assert "secret provider message" not in output
    assert "secret prompt" not in output


def test_terminal_renderer_uses_completed_tool_call_arguments_and_releases_state() -> None:
    stream, renderer = console_and_renderer(width=120)
    renderer.emit(event(UIEventType.TURN_STARTED, {"prompt": "inspect"}))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_CALL_COMPLETED,
        sequence=2,
        timestamp=1.1,
        tool_call_id="call-1",
        tool_name="run_shell",
        payload={"arguments": {"command": "git status"}},
    ))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_STARTED,
        sequence=3,
        timestamp=1.2,
        tool_call_id="call-1",
        tool_name="run_shell",
        payload={},
    ))
    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_FINISHED,
        sequence=4,
        timestamp=1.3,
        tool_call_id="call-1",
        tool_name="run_shell",
        payload={"status": "success", "preview": "clean"},
    ))

    assert "Bash(git status)" in stream.getvalue()
    assert renderer._tool_starts == {}
    assert renderer._tool_call_completions == {}


def test_terminal_renderer_formats_todo_snapshot_without_live_surface() -> None:
    stream, renderer = console_and_renderer(width=120)
    todos = [
        {
            "content": "分析配置",
            "active_form": "正在分析配置",
            "status": "completed",
        },
        {
            "content": "分析日志",
            "active_form": "正在分析日志",
            "status": "in_progress",
        },
    ]

    renderer.emit(RuntimeUIEvent(
        UIEventType.TOOL_EXECUTION_FINISHED,
        sequence=1,
        timestamp=1.0,
        tool_call_id="todo-1",
        tool_name="todo_write",
        payload={"status": "success", "metadata": {"todos": todos}},
    ))

    output = stream.getvalue()
    assert "正在分析日志..." in output
    assert "✓ 分析配置" in output
    assert "□ 分析日志" in output
    assert "todo_write" not in output
    assert '"active_form"' not in output
