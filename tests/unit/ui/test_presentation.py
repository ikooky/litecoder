from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from litecoder.tools.permission import PermissionMode
from litecoder.ui.events import UIEventFactory, UIEventType
from litecoder.ui.presenters import compact_number, normalize_windows_drive_letters
from litecoder.ui.presentation import (
    BlockKind,
    TranscriptBlock,
    TodoViewItem,
    PresentationReducer,
    ToolVisualState,
)
from litecoder.ui.textual_widgets import (
    SPINNER_FRAMES,
    render_live_tail,
    render_footer,
    render_transcript_block,
    render_todos,
    startup_banner,
)


def test_streaming_content_stays_live_until_completion() -> None:
    reducer = PresentationReducer()
    events = UIEventFactory(session_id="s1")

    reducer.apply(events.next(UIEventType.TURN_STARTED))
    reducer.apply(events.next(UIEventType.THINKING_DELTA, payload={"text": "check "}))
    reducer.apply(events.next(UIEventType.THINKING_DELTA, payload={"text": "files"}))

    assert reducer.state.blocks == []
    assert reducer.state.live.thinking_text == "check files"

    reducer.apply(
        events.next(
            UIEventType.THINKING_COMPLETED,
            payload={"text": "check files"},
        )
    )
    reducer.apply(events.next(UIEventType.ASSISTANT_DELTA, payload={"text": "hello "}))
    reducer.apply(events.next(UIEventType.ASSISTANT_DELTA, payload={"text": "world"}))

    assert [block.kind for block in reducer.state.blocks] == [BlockKind.THINKING]
    assert reducer.state.live.assistant_text == "hello world"

    reducer.apply(
        events.next(
            UIEventType.ASSISTANT_COMPLETED,
            payload={"text": "hello world"},
        )
    )
    assert [block.kind for block in reducer.state.blocks] == [
        BlockKind.THINKING,
        BlockKind.ASSISTANT,
    ]
    assert reducer.state.live.assistant_text == ""


def test_command_output_has_no_notice_prefix() -> None:
    reducer = PresentationReducer()
    block = reducer.add_command_output("Local commands:")
    console = Console(width=80, record=True, force_terminal=False)

    console.print(render_transcript_block(block))

    assert block.kind is BlockKind.COMMAND_OUTPUT
    assert console.export_text().strip() == "Local commands:"
    notice = reducer.add_notice_block("Warning", level="warning")
    notice_console = Console(width=80, record=True, force_terminal=False)

    notice_console.print(render_transcript_block(notice))
    assert notice_console.export_text().strip() == "! Warning"


def test_tool_lifecycle_commits_only_after_real_completion() -> None:
    reducer = PresentationReducer(workspace_root=r"E:\repo")
    events = UIEventFactory(session_id="s1")
    reducer.apply(events.next(UIEventType.TURN_STARTED))

    reducer.apply(
        events.next(
            UIEventType.TOOL_CALL_STARTED,
            tool_call_id="call-1",
            tool_name="run_shell",
            payload={"arguments": {"command": "pytest -q"}},
        )
    )
    assert reducer.state.blocks == []
    tool = reducer.state.live.tools[0]

    reducer.apply(
        events.next(
            UIEventType.PERMISSION_REQUESTED,
            tool_call_id="call-1",
            tool_name="run_shell",
            payload={"reason": "Permission confirmation required"},
        )
    )
    assert tool.status == ToolVisualState.WAITING_PERMISSION
    reducer.apply(
        events.next(
            UIEventType.PERMISSION_RESOLVED,
            tool_call_id="call-1",
            tool_name="run_shell",
            payload={"allowed": True},
        )
    )
    reducer.apply(
        events.next(
            UIEventType.TOOL_EXECUTION_STARTED,
            tool_call_id="call-1",
            tool_name="run_shell",
        )
    )
    assert tool.status == ToolVisualState.RUNNING
    assert reducer.state.blocks == []

    reducer.apply(
        events.next(
            UIEventType.TOOL_EXECUTION_FINISHED,
            tool_call_id="call-1",
            tool_name="run_shell",
            payload={"preview": "12 passed"},
        )
    )

    committed = reducer.state.blocks[0]
    assert committed.kind is BlockKind.TOOL
    assert committed.title == "Bash(pytest -q)"
    assert committed.status == ToolVisualState.SUCCESS
    assert committed.detail == ("12 passed",)
    assert reducer.state.live.tools == ()

def test_todo_snapshot_moves_into_history_before_summary() -> None:
    reducer = PresentationReducer()
    events = UIEventFactory(session_id="s1")
    reducer.apply(events.next(UIEventType.TURN_STARTED))
    reducer.apply(
        events.next(
            UIEventType.TOOL_EXECUTION_STARTED,
            tool_call_id="todo-1",
            tool_name="todo_write",
        )
    )
    reducer.apply(
        events.next(
            UIEventType.TODO_UPDATED,
            tool_call_id="todo-1",
            tool_name="todo_write",
            payload={
                "todos": [
                    {
                        "content": "Inspect UI",
                        "active_form": "Inspecting UI",
                        "status": "completed",
                    },
                    {
                        "content": "Implement widgets",
                        "active_form": "Implementing widgets",
                        "status": "in_progress",
                    },
                ]
            },
        )
    )

    assert reducer.state.blocks == []
    assert len(reducer.state.live.todos) == 2
    reducer.apply(
        events.next(
            UIEventType.ASSISTANT_COMPLETED,
            payload={"text": "Finished the requested work."},
        )
    )
    reducer.apply(
        events.next(
            UIEventType.TURN_FINISHED,
            payload={"status": "completed", "elapsed_seconds": 1.2},
        )
    )

    assert [block.kind for block in reducer.state.blocks] == [
        BlockKind.ASSISTANT,
        BlockKind.TODO,
        BlockKind.SUMMARY,
    ]
    assert [item.status for item in reducer.state.blocks[1].todos] == [
        "completed",
        "in_progress",
    ]
    assert reducer.state.live.visible is False
    assert [item.status for item in reducer.state.current_todos] == [
        "completed",
        "in_progress",
    ]

    reducer.apply(events.next(UIEventType.TURN_STARTED))

    assert [item.status for item in reducer.state.live.todos] == ["in_progress"]
    assert reducer.state.live.todo_carried is True
    assert reducer.state.live.todo_dirty is False

    console = Console(width=100, record=True, force_terminal=False)
    console.print(
        render_live_tail(reducer.state, animation_frame=0, queued_prompts=())
    )
    rendered = console.export_text()
    assert "1 open · continuing" in rendered
    assert "Implement widgets" in rendered
    assert "Inspect UI" not in rendered

    reducer.apply(
        events.next(
            UIEventType.TODO_UPDATED,
            payload={
                "todos": [
                    {
                        "content": "Review replacement plan",
                        "active_form": "Reviewing replacement plan",
                        "status": "pending",
                    }
                ]
            },
        )
    )

    assert [item.content for item in reducer.state.live.todos] == ["Review replacement plan"]
    assert reducer.state.live.todo_carried is False
    assert reducer.state.live.todo_dirty is True

def test_all_completed_todos_commit_but_do_not_carry_forward() -> None:
    reducer = PresentationReducer()
    events = UIEventFactory(session_id="s1")
    reducer.apply(events.next(UIEventType.TURN_STARTED))
    reducer.apply(
        events.next(
            UIEventType.TODO_UPDATED,
            payload={
                "todos": [
                    {
                        "content": "Finish UI",
                        "active_form": "Finishing UI",
                        "status": "completed",
                    }
                ]
            },
        )
    )

    assert reducer.state.current_todos == ()
    assert [item.status for item in reducer.state.live.todos] == ["completed"]

    reducer.apply(
        events.next(
            UIEventType.TURN_FINISHED,
            payload={"status": "completed", "elapsed_seconds": 1.0},
        )
    )
    assert [block.kind for block in reducer.state.blocks[-2:]] == [
        BlockKind.TODO,
        BlockKind.SUMMARY,
    ]

    reducer.apply(events.next(UIEventType.TURN_STARTED))
    assert reducer.state.live.todos == ()
    assert reducer.state.live.todo_carried is False


def test_retry_state_is_live_and_completed_turn_keeps_statistics() -> None:
    reducer = PresentationReducer()
    events = UIEventFactory(session_id="s1")
    reducer.apply(events.next(UIEventType.TURN_STARTED))
    reducer.apply(events.next(UIEventType.MODEL_REQUESTED, payload={"memory_count": 3}))
    reducer.apply(
        events.next(
            UIEventType.USAGE_UPDATED,
            payload={"input_tokens": 100, "output_tokens": 25},
        )
    )
    reducer.apply(
        events.next(
            UIEventType.PROVIDER_ERROR,
            payload={
                "code": "rate_limit",
                "message": "Too many requests",
                "retryable": True,
            },
        )
    )
    reducer.apply(
        events.next(
            UIEventType.NOTICE_RAISED,
            payload={"message": "Context compacted", "level": "warning"},
        )
    )

    assert reducer.state.blocks == []
    assert reducer.state.live.provider_error is not None
    assert reducer.state.live.notice is not None
    reducer.apply(events.next(UIEventType.ASSISTANT_DELTA, payload={"text": "Recovered"}))
    assert reducer.state.live.provider_error is None
    reducer.apply(
        events.next(
            UIEventType.ASSISTANT_COMPLETED,
            payload={"text": "Recovered"},
        )
    )
    reducer.apply(
        events.next(
            UIEventType.TURN_FINISHED,
            payload={
                "status": "completed",
                "elapsed_seconds": 2.5,
                "total_tokens": 125,
            },
        )
    )

    assert not any(block.kind is BlockKind.ERROR for block in reducer.state.blocks)
    summary = reducer.state.blocks[-1]
    assert summary.kind is BlockKind.SUMMARY
    assert reducer.state.usage == {"input_tokens": 100, "output_tokens": 25}
    assert "Elapsed 2.5s" in summary.text
    assert "Tokens: 125" in summary.text
    assert "Memory: 3" in summary.text

def test_footer_keeps_all_fields_when_terminal_is_narrow() -> None:
    footer = render_footer(
        permission_mode=PermissionMode.ASK,
        model="deepseek-flash",
        workspace=r"E:\a\very\long\workspace\path",
        width=32,
        usage={"input_tokens": 100, "output_tokens": 25},
    )

    assert footer.plain.startswith("⏵⏵ ask (shift+tab: next mode)")
    assert "context 100" in footer.plain
    assert "25" not in footer.plain
    assert "deepseek-flash" in footer.plain
    assert r"E:\a\very\long\workspace\path" in footer.plain


def test_running_tool_uses_stable_neutral_icon_without_creating_history() -> None:
    reducer = PresentationReducer()
    events = UIEventFactory(session_id="s1")
    reducer.apply(events.next(UIEventType.TURN_STARTED))
    reducer.apply(
        events.next(
            UIEventType.TOOL_EXECUTION_STARTED,
            tool_call_id="call-1",
            tool_name="read_file",
            payload={"arguments": {"path": "README.md"}},
        )
    )
    block = reducer.state.live.tools[0]
    console = Console(width=80, record=True, force_terminal=False)
    console.print(render_transcript_block(block, animation_frame=0))
    first = console.export_text(clear=True)
    console.print(render_transcript_block(block, animation_frame=1))
    second = console.export_text(clear=True)

    assert reducer.state.blocks == []
    assert block.status == ToolVisualState.RUNNING
    assert first == second
    assert "● Read(" in first
    assert not any(symbol in first for symbol in "✢✳✶✻✽")

def test_live_tail_shows_compact_statistics() -> None:
    reducer = PresentationReducer()
    events = UIEventFactory(session_id="s1")
    reducer.apply(events.next(UIEventType.TURN_STARTED))
    reducer.apply(
        events.next(
            UIEventType.MODEL_REQUESTED,
            payload={"memory_count": 2},
        )
    )
    reducer.apply(
        events.next(
            UIEventType.TOOL_EXECUTION_STARTED,
            tool_call_id="call-1",
            tool_name="read_file",
        )
    )
    reducer.apply(
        events.next(
            UIEventType.USAGE_UPDATED,
            payload={"input_tokens": 100, "output_tokens": 25},
        )
    )
    console = Console(width=100, record=True, force_terminal=False)
    console.print(
        render_live_tail(
            reducer.state,
            animation_frame=0,
            queued_prompts=(),
        )
    )
    rendered = console.export_text()

    assert "Running tools" in rendered
    assert "1 tool" in rendered
    assert "125 tokens" in rendered
    assert "2 memories" in rendered

def test_footer_renders_natural_permission_control_and_colors_only_label() -> None:
    footers = [
        render_footer(
            permission_mode=mode,
            model="model",
            workspace=r"E:\repo",
            width=120,
            usage={},
        )
        for mode in PermissionMode
    ]

    labels = ["ask", "read-only", "bypass"]
    for footer, label in zip(footers, labels, strict=True):
        assert footer.plain.startswith(
            f"⏵⏵ {label} (shift+tab: next mode) · model · E:\\repo"
        )
        assert footer.plain[footer.spans[0].start : footer.spans[0].end] == "⏵⏵ "
        assert str(footer.spans[0].style) == "bold #d8b4fe"
    assert len({footer.plain.index(" · ") for footer in footers}) == 3
    colored = [
        [
            span
            for span in footer.spans
            if str(span.style) in {"#f9e2af", "#a6e3a1", "#f38ba8"}
        ]
        for footer in footers
    ]
    assert [str(spans[0].style) for spans in colored] == [
        "#f9e2af",
        "#a6e3a1",
        "#f38ba8",
    ]
    assert [
        [footer.plain[span.start : span.end] for span in spans]
        for footer, spans in zip(footers, colored, strict=True)
    ] == [["ask"], ["read-only"], ["bypass"]]


def test_custom_status_icons_and_welcome_panel() -> None:
    assert SPINNER_FRAMES == ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    banner = startup_banner(r"E:\repo", "test-model")

    assert isinstance(banner, Panel)
    assert str(banner.border_style) == "bright_black"
    assert "Welcome to LiteCoder CLI!" in banner.renderable.plain
    assert "Workspace: E:\\repo" in banner.renderable.plain
    assert "Model: test-model" in banner.renderable.plain

    console = Console(width=80, record=True, force_terminal=False)
    console.print(
        render_transcript_block(
            TranscriptBlock(
                "tool",
                BlockKind.TOOL,
                title="Tool",
                status=ToolVisualState.SUCCESS.value,
            )
        )
    )
    console.print(
        render_transcript_block(
            TranscriptBlock(
                "summary",
                BlockKind.SUMMARY,
                text="Completed",
                status="completed",
            )
        )
    )
    rendered = console.export_text()

    assert "● Tool" in rendered
    assert "☭" not in rendered
    assert "Completed" in rendered


def test_todo_icons_are_square_without_completed_strikethrough() -> None:
    console = Console(width=80, record=True, force_terminal=False)
    todos = (
        TodoViewItem("done", "doing done", "completed"),
        TodoViewItem("active", "doing active", "in_progress"),
        TodoViewItem("later", "doing later", "pending"),
    )
    renderable = render_todos(todos)
    console.print(renderable)
    rendered = console.export_text()

    assert "■ done" in rendered
    assert "■ active" in rendered
    assert "□ later" in rendered
    assert "✓" not in rendered
    done_segment = next(
        segment for segment in console.render(renderable) if "done" in segment.text
    )
    assert done_segment.style is not None
    assert done_segment.style.strike is not True


def test_todos_fold_around_active_work_and_expand_on_demand() -> None:
    todos = (
        tuple(
            TodoViewItem(f"done {index}", f"doing done {index}", "completed")
            for index in range(6)
        )
        + (TodoViewItem("active", "working now", "in_progress"),)
        + tuple(
            TodoViewItem(f"pending {index}", f"waiting {index}", "pending")
            for index in range(6)
        )
    )
    console = Console(width=100, record=True, force_terminal=False)
    console.print(render_todos(todos))
    collapsed = console.export_text()

    assert "3 earlier completed" in collapsed
    assert "done 0" not in collapsed
    assert "done 3" in collapsed
    assert collapsed.index("active") < collapsed.index("pending 0")
    assert "2 later pending" in collapsed
    assert "pending 5" not in collapsed

    expanded_console = Console(width=100, record=True, force_terminal=False)
    expanded_console.print(render_todos(todos, expanded=True))
    expanded = expanded_console.export_text()
    assert "earlier completed" not in expanded
    assert "later pending" not in expanded
    assert "done 0" in expanded
    assert "pending 5" in expanded


def test_compact_token_counts_and_uppercase_windows_drives() -> None:
    assert compact_number(999) == "999"
    assert compact_number(1_000) == "1k"
    assert compact_number(19_301) == "19.3k"
    assert compact_number(744_644) == "744.6k"
    assert compact_number(1_200_000) == "1.2m"
    assert (
        normalize_windows_drive_letters(
            r"read c:\repo and d:/cache, leave abc:\value alone"
        )
        == r"read C:\repo and D:/cache, leave abc:\value alone"
    )

    footer = render_footer(
        permission_mode=PermissionMode.ASK,
        model="model",
        workspace=r"e:\repo",
        width=100,
        usage={"input_tokens": 744_644, "output_tokens": 1_200_000},
    )
    assert "context 744.6k" in footer.plain
    assert "1.2m" not in footer.plain
    assert r"E:\repo" in footer.plain


def test_tool_details_use_second_line_as_fold_summary() -> None:
    block = TranscriptBlock(
        "tool",
        BlockKind.TOOL,
        title="Glob(E:\\repo\\*)",
        detail=("first", "second", "third", "fourth"),
        status=ToolVisualState.SUCCESS.value,
    )
    collapsed_console = Console(width=80, record=True, force_terminal=False)
    collapsed_console.print(render_transcript_block(block))
    collapsed = collapsed_console.export_text()

    assert "first" in collapsed
    assert "… +3 lines (ctrl+o to expand)" in collapsed
    assert "second" not in collapsed
    assert "third" not in collapsed
    assert "fourth" not in collapsed

    expanded_console = Console(width=80, record=True, force_terminal=False)
    expanded_console.print(render_transcript_block(block, expanded=True))
    expanded = expanded_console.export_text()

    assert "first" in expanded
    assert "second" in expanded
    assert "third" in expanded
    assert "fourth" in expanded
    assert "ctrl+o to expand" not in expanded


def test_provider_retries_update_one_live_error_and_commit_only_on_failure() -> None:
    reducer = PresentationReducer()
    events = UIEventFactory(session_id="s1")
    reducer.apply(events.next(UIEventType.TURN_STARTED))

    for attempt in (1, 2):
        reducer.apply(events.next(UIEventType.MODEL_REQUESTED))
        reducer.apply(
            events.next(
                UIEventType.PROVIDER_ERROR,
                payload={
                    "code": "provider_rate_limit",
                    "message": "Provider rate limit exceeded",
                    "retryable": True,
                    "retrying": True,
                    "attempt": attempt,
                    "max_attempts": 2,
                },
            )
        )
        assert not any(
            block.kind is BlockKind.ERROR for block in reducer.state.blocks
        )
        error = reducer.state.live.provider_error
        assert error is not None
        assert error.title == "Retrying"
        assert error.detail == (f"Retrying... ({attempt}/2)",)

    reducer.apply(
        events.next(
            UIEventType.PROVIDER_ERROR,
            payload={
                "code": "provider_rate_limit",
                "message": "Provider rate limit exceeded",
                "retryable": True,
                "retrying": False,
                "attempt": 2,
                "max_attempts": 2,
            },
        )
    )
    reducer.apply(
        events.next(
            UIEventType.TURN_FINISHED,
            payload={"status": "incomplete", "elapsed_seconds": 2.0},
        )
    )

    error = next(
        block for block in reducer.state.blocks if block.kind is BlockKind.ERROR
    )
    assert error.detail == (
        "Provider rate limit exceeded",
        "Retries exhausted (2/2)",
    )
    assert "provider_rate_limit" not in "\n".join(error.detail)
