from __future__ import annotations

import os
import asyncio
from types import SimpleNamespace

import pytest
import litecoder.ui.tui as tui_module
from textual.containers import Vertical, VerticalScroll
from textual.selection import SELECT_ALL
from textual.widgets import Static

from litecoder.agent.result import AgentResult
from litecoder.cli.local_commands import (
    LOCAL_COMMANDS,
    LocalCommandResult,
    LocalCommandSpec,
)
from litecoder.providers.models import Usage
from litecoder.tools.permission import PermissionPrompt, PromptChoice
from litecoder.ui.events import UIEventFactory, UIEventType
from litecoder.ui.textual_widgets import TranscriptBlockWidget
from litecoder.ui.presentation import BlockKind
from litecoder.ui.tui import (
    LiteCoderApp,
    PermissionPane,
    PromptEditor,
    TextualPermissionPrompt,
    TextualUISink,
)


class FakeRuntime:
    model = "test-model"
    permission_mode = "ask"
    paths = SimpleNamespace(workspace_root=r"E:\repo")

    def __init__(self, sink: TextualUISink) -> None:
        self.sink = sink
        self.prompts: list[str] = []

    async def run(self, prompt: str) -> AgentResult:
        self.prompts.append(prompt)
        factory = UIEventFactory(session_id="session-1")
        self.sink.emit(factory.next(UIEventType.TURN_STARTED))
        self.sink.emit(
            factory.next(
                UIEventType.ASSISTANT_DELTA,
                payload={"text": "## Answer\n\n"},
            )
        )
        self.sink.emit(
            factory.next(
                UIEventType.ASSISTANT_DELTA,
                payload={"text": "body"},
            )
        )
        self.sink.emit(
            factory.next(
                UIEventType.ASSISTANT_COMPLETED,
                payload={"text": "## Answer\n\nbody"},
            )
        )
        self.sink.emit(
            factory.next(
                UIEventType.TURN_FINISHED,
                payload={
                    "status": "completed",
                    "elapsed_seconds": 0.2,
                },
            )
        )
        return AgentResult(
            "session-1",
            "completed",
            "done",
            Usage(1, 2),
        )

    async def resume(
        self,
        session_id: str,
        prompt: str | None = None,
    ) -> AgentResult:
        assert session_id == "session-1"
        return await self.run(prompt or "")


def build_app() -> tuple[
    LiteCoderApp,
    FakeRuntime,
    TextualPermissionPrompt,
]:
    sink = TextualUISink()
    permission = TextualPermissionPrompt()
    runtime = FakeRuntime(sink)
    return (
        LiteCoderApp(
            runtime,  # type: ignore[arg-type]
            sink=sink,
            permission_prompt=permission,
        ),
        runtime,
        permission,
    )


@pytest.mark.parametrize("command", sorted(LOCAL_COMMANDS))
@pytest.mark.asyncio
async def test_every_local_command_stays_idle_and_refocuses_prompt(
    command: str,
) -> None:
    app, runtime, _ = build_app()
    dispatched: list[tuple[str, str | None]] = []

    async def dispatch(
        text: str,
        *,
        session_id: str | None,
    ) -> LocalCommandResult:
        dispatched.append((text, session_id))
        await asyncio.sleep(0)
        return LocalCommandResult(True, message=f"done: {text}")

    app.router.dispatch = dispatch  # type: ignore[method-assign]
    app.reducer.state.live.active = True
    app.reducer.state.live.phase = "tool"
    app.reducer.state.tool_invocations = 4
    app.reducer.state.usage = {"input_tokens": 7_500}
    async with app.run_test(size=(80, 24)) as pilot:
        app._start_prompt(command)
        assert app.reducer.state.live.active is False
        assert app.reducer.state.live.phase == "idle"
        assert app.query_one("#live-tail").display is False

        for _ in range(10):
            await pilot.pause()
            if app._turn_worker is None:
                break

        assert dispatched == [(command, None)]
        assert runtime.prompts == []
        assert app._turn_worker is None
        assert app.reducer.state.live.active is False
        assert app.reducer.state.live.phase == "idle"
        assert app.query_one("#live-tail").display is False
        assert app.focused is app.query_one("#prompt", PromptEditor)


@pytest.mark.asyncio
async def test_unhandled_runtime_error_does_not_expose_exception_text() -> None:
    app, runtime, _ = build_app()

    async def fail(_: str) -> AgentResult:
        raise RuntimeError("secret provider traceback detail")

    runtime.run = fail  # type: ignore[method-assign]
    async with app.run_test(size=(80, 24)) as pilot:
        app._start_prompt("trigger failure")
        for _ in range(10):
            await pilot.pause()
            if app._turn_worker is None:
                break

        errors = [
            block
            for block in app.reducer.state.blocks
            if block.kind is BlockKind.ERROR
        ]
        assert errors[-1].detail == ("Unexpected internal error",)
        assert "secret provider traceback detail" not in str(errors[-1])


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific driver")
@pytest.mark.asyncio
async def test_textual_app_uses_standard_windows_driver() -> None:
    from textual.drivers.windows_driver import WindowsDriver

    app, _, _ = build_app()

    driver = app._build_driver(headless=False, inline=True, mouse=False, size=(80, 24))

    assert isinstance(driver, WindowsDriver)
    assert driver.is_inline is False


@pytest.mark.asyncio
async def test_textual_layout_survives_repeated_terminal_resize() -> None:
    app, _, _ = build_app()

    async with app.run_test(size=(120, 40)) as pilot:
        for width, height in (
            (40, 18),
            (48, 20),
            (60, 24),
            (80, 28),
            (100, 32),
            (120, 40),
        ) * 8:
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            prompt = app.query_one("#prompt", PromptEditor)
            prompt_container = app.query_one("#prompt-container")
            footer = app.query_one("#footer")
            messages = app.query_one("#messages", Vertical)
            assert prompt.region.width == width - 3
            assert prompt_container.region.width == width
            assert footer.region.width == width
            assert messages.region.width == width
            assert messages.region.bottom == prompt_container.region.y
            assert prompt_container.region.bottom == footer.region.y

        transcript = app.query_one("#transcript", Vertical)
        assert len(transcript.children) == 1
        assert app.focused is app.query_one("#prompt", PromptEditor)
        assert app.query_one("#live-tail").display is False
        assert app.query_one("#permission-pane").display is False
        assert str(app.query_one("#prompt-prefix").render()) == "> "
        assert app.query_one("#prompt", PromptEditor).placeholder == ""


@pytest.mark.asyncio
async def test_prompt_cursor_only_blinks_for_blank_or_whitespace_input() -> None:
    app, _, _ = build_app()

    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one("#prompt", PromptEditor)
        assert prompt.cursor_blink is True

        await pilot.press("space", "space")
        assert prompt.cursor_blink is True

        await pilot.press("x")
        assert prompt.cursor_blink is False

        await pilot.press("backspace")
        assert prompt.cursor_blink is True


@pytest.mark.asyncio
async def test_slash_command_menu_lists_and_filters_commands_case_insensitively() -> None:
    app, _, _ = build_app()

    async with app.run_test(size=(100, 30)) as pilot:
        prompt = app.query_one("#prompt", PromptEditor)
        menu = app.query_one("#command-menu", Static)

        prompt.load_text("/")
        await pilot.pause()
        assert menu.display is True
        assert [command.name for command in app._command_matches] == sorted(
            LOCAL_COMMANDS
        )
        assert app._selected_command_name == "/clear"
        assert (
            app.query_one("#prompt-container").region.bottom == menu.region.y
        )
        assert menu.region.bottom == app.query_one("#footer").region.y
        screenshot = app.export_screenshot()
        assert "/clear" in screenshot
        assert "Start" in screenshot

        prompt.load_text("/MO")
        await pilot.pause()
        assert menu.display is True
        assert [command.name for command in app._command_matches] == ["/model"]

        prompt.load_text("/model provider")
        await pilot.pause()
        assert menu.display is False
        assert app._command_matches == ()

        prompt.load_text("hello")
        await pilot.pause()
        assert menu.display is False


@pytest.mark.asyncio
async def test_slash_command_menu_navigates_and_completes_with_tab() -> None:
    app, _, _ = build_app()

    async with app.run_test(size=(100, 30)) as pilot:
        prompt = app.query_one("#prompt", PromptEditor)
        prompt.load_text("/m")
        await pilot.pause()

        assert [command.name for command in app._command_matches] == [
            "/memory",
            "/model",
        ]
        assert app._selected_command_name == "/memory"

        await pilot.press("down", "tab")
        await pilot.pause()

        assert prompt.text == "/model"
        assert prompt.cursor_location == (0, len("/model"))
        assert app._selected_command_name == "/model"


@pytest.mark.asyncio
async def test_slash_command_menu_windows_large_match_sets() -> None:
    app, _, _ = build_app()
    commands = tuple(
        LocalCommandSpec(
            f"/command-{index:05d}",
            f"/command-{index:05d}",
            f"Command {index}",
        )
        for index in range(10_000)
    )
    app.router.command_specs = lambda: commands  # type: ignore[method-assign]

    async with app.run_test(size=(100, 30)) as pilot:
        prompt = app.query_one("#prompt", PromptEditor)
        prompt.load_text("/")
        await pilot.pause()

        assert len(app._command_matches) == 10_000
        assert [command.name for command in app._visible_command_matches()] == [
            f"/command-{index:05d}" for index in range(8)
        ]

        await pilot.press("pagedown")
        await pilot.pause()
        assert app._selected_command_index == 8
        assert app._command_window_start == 1
        assert [command.name for command in app._visible_command_matches()] == [
            f"/command-{index:05d}" for index in range(1, 9)
        ]

        await pilot.press("end")
        await pilot.pause()
        assert app._selected_command_index == 9_999
        assert app._command_window_start == 9_992
        assert [command.name for command in app._visible_command_matches()] == [
            f"/command-{index:05d}" for index in range(9_992, 10_000)
        ]

        await pilot.press("home")
        await pilot.pause()
        assert app._selected_command_index == 0
        assert app._command_window_start == 0


@pytest.mark.asyncio
async def test_textual_app_submits_and_renders_streamed_markdown() -> None:
    app, runtime, _ = build_app()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("h", "e", "l", "l", "o", "enter")
        for _ in range(5):
            await pilot.pause()

        transcript = app.query_one("#transcript", Vertical)
        classes = [child.classes for child in transcript.children]
        assert runtime.prompts == ["hello"]
        assert app.session_id == "session-1"
        assert any("user-message" in value for value in classes)
        assert any("assistant-message" in value for value in classes)
        assert any("turn-summary" in value for value in classes)


@pytest.mark.asyncio
async def test_output_flow_uses_single_page_scroll_container() -> None:
    app, _, _ = build_app()

    async with app.run_test(size=(80, 18)):
        page = app.query_one("#page", VerticalScroll)
        messages = app.query_one("#messages", Vertical)
        transcript = app.query_one("#transcript", Vertical)
        activity = app.query_one("#live-tail")
        permission = app.query_one("#permission-pane", PermissionPane)
        prompt_container = app.query_one("#prompt-container")
        bottom_dock = app.query_one("#bottom-dock", Vertical)

        assert messages.styles.height is not None
        assert messages.styles.height.is_auto
        assert messages.parent is page
        assert bottom_dock.parent is page
        assert transcript.parent is messages
        assert activity.parent is messages
        assert permission.parent is messages
        assert messages.region.bottom == prompt_container.region.y

        assert (
            messages.region.x == transcript.region.x == prompt_container.region.x == 0
        )
        assert (
            messages.region.width
            == transcript.region.width
            == prompt_container.region.width
        )
        assert str(app.screen.styles.overflow_y) == "hidden"
        assert str(page.styles.overflow_y) == "auto"
        assert page.styles.scrollbar_size_vertical == 1


@pytest.mark.asyncio
async def test_large_output_scrolls_to_tail_without_clipping_input() -> None:
    app, _, _ = build_app()

    async with app.run_test(size=(80, 18)) as pilot:
        for index in range(12):
            app.reducer.add_assistant_history(
                "\n".join(f"line {index}-{line}" for line in range(5))
            )
            await app._sync_transcript()
        await pilot.pause()

        page = app.query_one("#page", VerticalScroll)
        await pilot.pause()

        assert page.virtual_size.height > page.scrollable_content_region.height
        assert page.max_scroll_y > 0
        assert page.is_vertical_scroll_end
        assert page.vertical_scrollbar.display is True
        assert page.vertical_scrollbar.region.width == 1
        assert page.vertical_scrollbar.region.right == page.region.right
        assert app.query_one("#bottom-dock").region.bottom <= page.virtual_size.height


@pytest.mark.asyncio
async def test_runtime_events_commit_in_completion_order() -> None:
    app, runtime, _ = build_app()
    events = UIEventFactory(session_id="session-1")
    ordered_events = [
        events.next(UIEventType.TURN_STARTED),
        events.next(
            UIEventType.TOOL_CALL_STARTED,
            tool_call_id="first",
            tool_name="run_shell",
            payload={"arguments": {"command": "first"}},
        ),
        events.next(
            UIEventType.TOOL_EXECUTION_FINISHED,
            tool_call_id="first",
            tool_name="run_shell",
            payload={"preview": "first result"},
        ),
        events.next(
            UIEventType.TOOL_CALL_STARTED,
            tool_call_id="second",
            tool_name="run_shell",
            payload={"arguments": {"command": "second"}},
        ),
        events.next(
            UIEventType.TOOL_EXECUTION_FINISHED,
            tool_call_id="second",
            tool_name="run_shell",
            payload={"preview": "second result"},
        ),
        events.next(
            UIEventType.ASSISTANT_COMPLETED,
            payload={"text": "final answer"},
        ),
        events.next(
            UIEventType.TURN_FINISHED,
            payload={"status": "completed", "elapsed_seconds": 1.0},
        ),
    ]
    async with app.run_test(size=(80, 24)):
        for event in ordered_events:
            runtime.sink.emit(event)
        barrier = runtime.sink.flush()
        assert barrier is not None
        await barrier

        blocks = [
            child.block
            for child in app.query_one("#transcript", Vertical).children
            if isinstance(child, TranscriptBlockWidget)
        ]

    assert [block.title or block.text for block in blocks] == [
        "Bash(first)",
        "Bash(second)",
        "final answer",
        blocks[-1].text,
    ]
    assert [block.kind for block in blocks] == [
        BlockKind.TOOL,
        BlockKind.TOOL,
        BlockKind.ASSISTANT,
        BlockKind.SUMMARY,
    ]
    assert blocks[-1].text.startswith("Completed")


@pytest.mark.asyncio
async def test_runtime_flush_preserves_final_answer_before_real_completion() -> None:
    app, runtime, _ = build_app()
    factory = UIEventFactory(session_id="session-1")

    async def flush() -> None:
        barrier = runtime.sink.flush()
        assert barrier is not None
        await barrier

    async with app.run_test(size=(90, 28)):
        runtime.sink.emit(factory.next(UIEventType.TURN_STARTED))
        runtime.sink.emit(factory.next(UIEventType.THINKING_STARTED))
        runtime.sink.emit(
            factory.next(
                UIEventType.THINKING_COMPLETED,
                payload={"text": "Inspecting"},
            )
        )
        await flush()
        assert not app.query(".turn-summary")

        answer = "## Final answer\n\n| Item | Result |\n| --- | --- |\n| UI | fixed |"
        runtime.sink.emit(
            factory.next(
                UIEventType.ASSISTANT_COMPLETED,
                payload={"text": answer},
            )
        )
        await flush()

        assistant = app.query_one(
            ".assistant-message",
            TranscriptBlockWidget,
        )
        selected = assistant.get_selection(SELECT_ALL)
        assert selected is not None
        assert "Final answer" in selected[0]
        assert "UI" in selected[0]
        assert "fixed" in selected[0]
        assert all(line == line.rstrip() for line in selected[0].splitlines())
        assert not app.query(".turn-summary")

        runtime.sink.emit(
            factory.next(
                UIEventType.TURN_FINISHED,
                payload={
                    "status": "completed",
                    "elapsed_seconds": 1.0,
                },
            )
        )
        await flush()

        children = list(app.query_one("#transcript").children)
        summary = app.query_one(".turn-summary", TranscriptBlockWidget)
        assert children.index(assistant) < children.index(summary)


@pytest.mark.asyncio
async def test_todo_moves_from_live_tail_into_history_at_turn_end() -> None:
    app, runtime, _ = build_app()
    factory = UIEventFactory(session_id="session-1")
    todos = [
        {
            "content": f"task {index}",
            "active_form": f"doing {index}",
            "status": (
                "completed"
                if index < 6
                else "in_progress"
                if index == 6
                else "pending"
            ),
        }
        for index in range(10)
    ]

    async with app.run_test(size=(80, 18)):
        runtime.sink.emit(factory.next(UIEventType.TURN_STARTED))
        runtime.sink.emit(
            factory.next(UIEventType.TODO_UPDATED, payload={"todos": todos})
        )
        barrier = runtime.sink.flush()
        assert barrier is not None
        await barrier

        messages = app.query_one("#messages", Vertical)
        tail = app.query_one("#live-tail")
        prompt_container = app.query_one("#prompt-container")
        assert tail.display is True
        assert tail.parent is messages
        assert messages.region.bottom == prompt_container.region.y
        assert not app.query(".todo-message")

        runtime.sink.emit(
            factory.next(
                UIEventType.ASSISTANT_COMPLETED,
                payload={"text": "Final result"},
            )
        )
        runtime.sink.emit(
            factory.next(
                UIEventType.TURN_FINISHED,
                payload={"status": "completed", "elapsed_seconds": 1.0},
            )
        )
        barrier = runtime.sink.flush()
        assert barrier is not None
        await barrier

        blocks = [
            child.block
            for child in app.query_one("#transcript").children
            if isinstance(child, TranscriptBlockWidget)
        ]
        assert [block.kind for block in blocks[-3:]] == [
            BlockKind.ASSISTANT,
            BlockKind.TODO,
            BlockKind.SUMMARY,
        ]
        assert tail.display is False
        app.reducer.start_turn_preview()
        app._refresh_live_tail()

        assert len(app.reducer.state.live.todos) == 4
        assert all(
            item.status in {"in_progress", "pending"}
            for item in app.reducer.state.live.todos
        )
        assert app.reducer.state.live.todo_carried is True
        assert app.reducer.state.live.todo_dirty is False
        assert tail.display is True



@pytest.mark.asyncio
async def test_page_click_focuses_prompt_and_right_click_copies_selection() -> None:
    app, runtime, _ = build_app()
    factory = UIEventFactory(session_id="session-1")

    async with app.run_test(size=(80, 24)) as pilot:
        runtime.sink.emit(
            factory.next(
                UIEventType.ASSISTANT_COMPLETED,
                payload={"text": "Copy this answer"},
            )
        )
        barrier = runtime.sink.flush()
        assert barrier is not None
        await barrier

        assistant = app.query_one(
            ".assistant-message",
            TranscriptBlockWidget,
        )
        selected = assistant.get_selection(SELECT_ALL)
        assert selected is not None
        app.screen.selections[assistant] = SELECT_ALL

        await pilot.click(assistant, button=3)
        assert app.clipboard == selected[0]

        app.screen.selections.clear()
        app.screen.set_focus(None)
        await pilot.click(".banner-message")
        assert app.focused is app.query_one("#prompt", PromptEditor)

        await pilot.press("h", "i")
        assert app.query_one("#prompt", PromptEditor).text == "hi"


@pytest.mark.asyncio
async def test_assistant_markdown_table_cells_fold_without_ellipsis() -> None:
    app, runtime, _ = build_app()
    factory = UIEventFactory(session_id="session-1")
    value = "x" * 60
    message = f"| Name | Value |\n| --- | --- |\n| long | {value} |"

    async with app.run_test(size=(28, 24)):
        runtime.sink.emit(
            factory.next(
                UIEventType.ASSISTANT_COMPLETED,
                payload={"text": message},
            )
        )
        barrier = runtime.sink.flush()
        assert barrier is not None
        await barrier

        assistant = app.query_one(
            ".assistant-message",
            TranscriptBlockWidget,
        )
        selected = assistant.get_selection(SELECT_ALL)
        assert selected is not None
        rendered = selected[0]
        assert "…" not in rendered
        assert rendered.count("x") == len(value)


@pytest.mark.asyncio
async def test_output_click_focuses_prompt_without_scrolling_to_bottom() -> None:
    app, _, _ = build_app()
    for index in range(20):
        app.reducer.add_assistant_history(
            f"history {index}\n" + "\n".join(f"line {line}" for line in range(3))
        )

    async with app.run_test(size=(80, 18)) as pilot:
        await app._sync_transcript()
        await pilot.pause()
        page = app.query_one("#page", VerticalScroll)
        target = max(1, page.max_scroll_y // 2)
        page.scroll_to(y=target, animate=False)
        await pilot.pause()
        before = page.scroll_y
        assert before > 0

        await pilot.click(offset=(10, 5))
        await pilot.pause()


@pytest.mark.asyncio
async def test_textual_permission_prompt_is_inline_after_context() -> None:
    app, runtime, permission = build_app()
    result: list[PromptChoice] = []
    factory = UIEventFactory(session_id="session-1")

    async with app.run_test(size=(100, 30)) as pilot:
        runtime.sink.emit(
            factory.next(
                UIEventType.ASSISTANT_COMPLETED,
                payload={"text": "Visible context before permission"},
            )
        )
        runtime.sink.emit(
            factory.next(
                UIEventType.PERMISSION_REQUESTED,
                tool_call_id="permission-call",
                tool_name="run_shell",
                payload={"arguments": {"command": r"cd c:\repo"}},
            )
        )

        async def ask() -> None:
            result.append(
                await permission(
                    PermissionPrompt(
                        "run_shell",
                        "high",
                        "external:test",
                        {"command": r"cd c:\repo"},
                        r"e:\repo",
                    )
                )
            )

        app.run_worker(ask(), name="permission-test")
        await pilot.pause()
        pane = app.query_one("#permission-pane", PermissionPane)
        messages = app.query_one("#messages", Vertical)
        assert pane.display is True
        assert pane.parent is messages
        assert app.focused is pane
        assistant = app.query_one(".assistant-message", TranscriptBlockWidget)
        tail = app.query_one("#live-tail", Static)
        assert assistant.region.bottom <= tail.region.y
        assert tail.region.bottom <= pane.region.y
        screenshot = app.export_screenshot()
        assert "Visible" in screenshot
        assert "Bash" in screenshot
        assert not app.query(".tool-message")
        assert "C:\\repo" in str(pane.query_one("#permission-title").render())

        await pilot.press("enter")
        await pilot.pause()

        assert pane.display is False
        assert app.focused is app.query_one("#prompt", PromptEditor)

    assert result == [PromptChoice.ALLOW_ONCE]


@pytest.mark.asyncio
async def test_concurrent_permissions_are_resolved_in_fifo_order() -> None:
    app, _, permission = build_app()

    def prompt(label: str) -> PermissionPrompt:
        return PermissionPrompt(
            "run_shell",
            "high",
            f"external:{label}",
            {"command": label},
            r"e:\repo",
            label,
        )

    async with app.run_test(size=(100, 30)) as pilot:
        tasks = [
            asyncio.create_task(permission(prompt(label)))
            for label in ("one", "two", "three")
        ]
        pane = app.query_one("#permission-pane", PermissionPane)

        await pilot.pause()
        assert pane.prompt is not None
        assert pane.prompt.tool_call_id == "one"
        assert tasks[1].done() is False
        assert tasks[2].done() is False

        await pilot.press("1")
        await pilot.pause()
        assert pane.prompt is not None
        assert pane.prompt.tool_call_id == "two"

        await pilot.press("3")
        await pilot.pause()
        assert pane.prompt is not None
        assert pane.prompt.tool_call_id == "three"

        await pilot.press("2")
        choices = await asyncio.gather(*tasks)
        await pilot.pause()

        assert choices == [
            PromptChoice.ALLOW_ONCE,
            PromptChoice.DENY,
            PromptChoice.ALLOW_FOR_ROOT_SESSION,
        ]
        assert pane.display is False
        assert app.focused is app.query_one("#prompt", PromptEditor)


@pytest.mark.asyncio
async def test_permission_cancellation_removes_waiter_and_advances_queue() -> None:
    app, _, permission = build_app()

    def prompt(label: str) -> PermissionPrompt:
        return PermissionPrompt(
            "run_shell", "high", f"external:{label}",
            {"command": label}, r"e:\repo", label,
        )

    async with app.run_test(size=(100, 30)) as pilot:
        first = asyncio.create_task(permission(prompt("one")))
        second = asyncio.create_task(permission(prompt("two")))
        third = asyncio.create_task(permission(prompt("three")))
        pane = app.query_one("#permission-pane", PermissionPane)
        await pilot.pause()

        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await pilot.pause()

        assert pane.prompt is not None
        assert pane.prompt.tool_call_id == "three"
        await pilot.press("3")
        assert await third is PromptChoice.DENY
        await pilot.pause()
        assert pane.display is False
        assert app._permission_active is None
        assert not app._permission_queue


@pytest.mark.asyncio
async def test_active_permission_times_out_to_deny_and_advances_queue(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        tui_module, "PERMISSION_CONFIRMATION_TIMEOUT_SECONDS", 0.5
    )
    app, _, permission = build_app()

    def prompt(label: str) -> PermissionPrompt:
        return PermissionPrompt(
            "run_shell",
            "high",
            f"external:{label}",
            {"command": label},
            r"e:\repo",
            label,
        )

    async with app.run_test(size=(100, 30)) as pilot:
        first = asyncio.create_task(permission(prompt("one")))
        await asyncio.sleep(0)
        second = asyncio.create_task(permission(prompt("two")))
        pane = app.query_one("#permission-pane", PermissionPane)
        await pilot.pause()

        assert pane.prompt is not None
        assert pane.prompt.tool_call_id == "one"
        assert "Auto-deny after 0.5 seconds" in str(
            pane.query_one("#permission-help", Static).render()
        )

        assert await asyncio.wait_for(first, timeout=1.0) is PromptChoice.DENY
        assert pane.prompt is not None
        assert pane.prompt.tool_call_id == "two"
        assert second.done() is False

        await pilot.press("1")
        assert await second is PromptChoice.ALLOW_ONCE
        await pilot.pause()

        assert pane.display is False
        assert app._permission_active is None
        assert app._permission_timeout_task is None
        assert not app._permission_queue


@pytest.mark.asyncio
async def test_cancel_turn_denies_and_clears_all_permission_requests() -> None:
    app, _, permission = build_app()

    class WorkerStub:
        is_running = True
        cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    worker = WorkerStub()
    async with app.run_test(size=(100, 30)) as pilot:
        app._turn_worker = worker  # type: ignore[assignment]
        tasks = [
            asyncio.create_task(
                permission(
                    PermissionPrompt(
                        "run_shell", "high", f"external:{label}",
                        {"command": label}, r"e:\repo", label,
                    )
                )
            )
            for label in ("one", "two")
        ]
        await pilot.pause()

        app.action_cancel_turn()
        choices = await asyncio.gather(*tasks)
        await pilot.pause()

        assert worker.cancelled is True
        assert choices == [PromptChoice.DENY, PromptChoice.DENY]
        assert app.query_one("#permission-pane", PermissionPane).display is False
        assert app._permission_active is None
        assert not app._permission_queue


def test_permission_selection_uses_text_color_without_reverse_background() -> None:
    pane = PermissionPane()

    options = pane._render_options()
    assert options.plain.startswith("› 1. Yes")
    assert all("reverse" not in str(span.style) for span in options.spans)


@pytest.mark.asyncio
async def test_ctrl_c_copies_selection_and_ctrl_v_pastes_it() -> None:
    app, _, _ = build_app()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.press("shift+home")
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app.clipboard == "hello"

        await pilot.press("end")
        await pilot.press("ctrl+v")
        await pilot.pause()

        assert app.query_one("#prompt", PromptEditor).text == "hellohello"


@pytest.mark.asyncio
async def test_shift_tab_cycles_permission_mode_and_footer() -> None:
    app, runtime, _ = build_app()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("shift+tab")
        await pilot.pause()
        footer = app.query_one("#footer")
        assert runtime.permission_mode == "read-only"
        assert "read-only" in str(footer.render())


@pytest.mark.asyncio
async def test_live_prompt_queues_next_message_while_turn_runs() -> None:
    sink = TextualUISink()
    permission = TextualPermissionPrompt()
    started = asyncio.Event()
    release = asyncio.Event()

    class QueuedRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__(sink)
            self.calls: list[tuple[str, str]] = []

        async def run(self, prompt: str) -> AgentResult:
            self.calls.append(("run", prompt))
            started.set()
            await release.wait()
            return AgentResult(
                "session-1",
                "completed",
                "done",
                Usage(0, 0),
            )

        async def resume(
            self,
            session_id: str,
            prompt: str | None = None,
        ) -> AgentResult:
            self.calls.append(("resume", prompt or ""))
            return AgentResult(
                session_id,
                "completed",
                "done",
                Usage(0, 0),
            )

    runtime = QueuedRuntime()
    app = LiteCoderApp(
        runtime,  # type: ignore[arg-type]
        sink=sink,
        permission_prompt=permission,
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("o", "n", "e", "enter")
        await started.wait()
        await pilot.press("t", "w", "o", "enter")
        await pilot.pause()
        release.set()
        for _ in range(5):
            await pilot.pause()

    assert runtime.calls == [("run", "one"), ("resume", "two")]


@pytest.mark.asyncio
async def test_full_runtime_lifecycle_updates_stable_ui_regions() -> None:
    app, runtime, _ = build_app()
    factory = UIEventFactory(session_id="session-1")

    async with app.run_test(size=(100, 32)) as pilot:
        events = (
            factory.next(UIEventType.TURN_STARTED),
            factory.next(
                UIEventType.MODEL_REQUESTED,
                payload={"memory_count": 2},
            ),
            factory.next(UIEventType.THINKING_STARTED),
            factory.next(
                UIEventType.THINKING_DELTA,
                payload={"text": "Inspecting the repository"},
            ),
            factory.next(
                UIEventType.THINKING_COMPLETED,
                payload={"text": "Inspecting the repository"},
            ),
            factory.next(
                UIEventType.ASSISTANT_DELTA,
                payload={"text": "I found the issue."},
            ),
            factory.next(
                UIEventType.ASSISTANT_COMPLETED,
                payload={"text": "I found the issue."},
            ),
            factory.next(
                UIEventType.TOOL_CALL_STARTED,
                tool_call_id="call-1",
                tool_name="run_shell",
                payload={"arguments": {"command": "pytest -q"}},
            ),
            factory.next(
                UIEventType.PERMISSION_REQUESTED,
                tool_call_id="call-1",
                tool_name="run_shell",
                payload={"reason": "Permission confirmation required"},
            ),
            factory.next(
                UIEventType.PERMISSION_RESOLVED,
                tool_call_id="call-1",
                tool_name="run_shell",
                payload={"allowed": True},
            ),
            factory.next(
                UIEventType.TOOL_EXECUTION_STARTED,
                tool_call_id="call-1",
                tool_name="run_shell",
            ),
            factory.next(
                UIEventType.USAGE_UPDATED,
                payload={"input_tokens": 100, "output_tokens": 25},
            ),
            factory.next(
                UIEventType.TODO_UPDATED,
                payload={
                    "todos": [
                        {
                            "content": "Inspect UI",
                            "active_form": "Inspecting UI",
                            "status": "completed",
                        },
                        {
                            "content": "Verify resize",
                            "active_form": "Verifying resize",
                            "status": "in_progress",
                        },
                    ]
                },
            ),
            factory.next(
                UIEventType.TOOL_EXECUTION_FINISHED,
                tool_call_id="call-1",
                tool_name="run_shell",
                payload={"preview": "15 passed"},
            ),
            factory.next(
                UIEventType.NOTICE_RAISED,
                payload={
                    "message": "Context compacted",
                    "level": "warning",
                    "persistent": True,
                },
            ),
            factory.next(
                UIEventType.PROVIDER_ERROR,
                payload={
                    "code": "rate_limit",
                    "message": "Too many requests",
                    "retryable": True,
                },
            ),
        )
        for event in events:
            runtime.sink.emit(event)
        for _ in range(6):
            await pilot.pause()

        classes = [child.classes for child in app.query_one("#transcript").children]
        assert sum("tool-message" in value for value in classes) == 1
        assert any("thinking-message" in value for value in classes)
        assert any("assistant-message" in value for value in classes)
        assert not any("error-message" in value for value in classes)
        assert any("notice-message" in value for value in classes)
        tool = next(
            block for block in app.reducer.state.blocks if block.kind.value == "tool"
        )
        assert tool.title == "Bash(pytest -q)"
        assert tool.status == "success"
        assert tool.detail == ("15 passed",)
        assert app.reducer.state.live.phase == "retrying"
        assert app.reducer.state.live.provider_error is not None
        assert app.reducer.state.tool_invocations == 1
        assert app.reducer.state.memory_count == 2
        assert len(app.reducer.state.current_todos) == 2

        runtime.sink.emit(
            factory.next(
                UIEventType.TURN_FINISHED,
                payload={
                    "status": "incomplete",
                    "elapsed_seconds": 1.5,
                    "total_tokens": 125,
                },
            )
        )
        for _ in range(3):
            await pilot.pause()

        classes = [child.classes for child in app.query_one("#transcript").children]
        assert any("error-message" in value for value in classes)
        assert any("todo-message" in value for value in classes)
        assert any("turn-summary" in value for value in classes)
        assert app.reducer.state.live.visible is False
        assert "context 100" in str(app.query_one("#footer").render())


@pytest.mark.asyncio
async def test_resumed_session_restores_messages_tools_and_todos() -> None:
    app, runtime, _ = build_app()

    class Store:
        async def load_context(self, session_id: str) -> object:
            assert session_id == "restored"
            return SimpleNamespace(
                session=SimpleNamespace(model="restored-model"),
                messages=(
                    SimpleNamespace(
                        role="user",
                        content=[{"type": "text", "text": "original prompt"}],
                    ),
                    SimpleNamespace(
                        role="assistant",
                        content=[
                            {"type": "thinking", "thinking": "inspect"},
                            {
                                "type": "tool_call",
                                "id": "call-1",
                                "name": "read_file",
                                "input": {"path": "README.md"},
                            },
                            {"type": "text", "text": "restored answer"},
                        ],
                    ),
                    SimpleNamespace(
                        role="user",
                        content=[
                            {
                                "type": "tool_result",
                                "tool_call_id": "call-1",
                                "status": "success",
                                "content": "42 lines",
                            }
                        ],
                    ),
                ),
            )

        async def list_todos(self, session_id: str) -> list[dict[str, str]]:
            assert session_id == "restored"
            return [
                {
                    "content": "Resume work",
                    "active_form": "Resuming work",
                    "status": "in_progress",
                }
            ]

    runtime.store = Store()
    app.session_id = "restored"

    async with app.run_test(size=(90, 28)) as pilot:
        await pilot.pause()
        await pilot.pause()
        kinds = [block.kind.value for block in app.reducer.state.blocks]
        tool = next(
            block for block in app.reducer.state.blocks if block.kind.value == "tool"
        )
        assert app.model_name == "restored-model"
        assert kinds.count("user") == 1
        assert kinds.count("thinking") == 1
        assert kinds.count("assistant") == 1
        assert kinds.count("tool") == 1
        assert kinds.count("todo") == 1
        assert tool.status == "success"
        assert tool.detail == ("42 lines",)
        assert app.reducer.state.current_todos[0].status == "in_progress"


@pytest.mark.asyncio
async def test_footer_wraps_without_dropping_model_or_workspace() -> None:
    app, _, _ = build_app()
    app.workspace_path = r"E:\projects\a-very-long-workspace-directory\litecoder"
    app.reducer.state.usage = {"input_tokens": 123_456, "output_tokens": 7_890}

    async with app.run_test(size=(48, 24)) as pilot:
        app.model_name = "provider/very-long-model-name-with-context-window"
        app._refresh_footer()
        await pilot.pause()
        footer = app.query_one("#footer")
        rendered = str(footer.render())

        assert footer.region.height > 1
        assert "provider/very-long-model-name-with-context-window" in rendered
        assert r"E:\projects\a-very-long-workspace-directory\litecoder" in rendered
        assert "context 123.5k" in rendered
        assert "7.9k" not in rendered
