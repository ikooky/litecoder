"""Textual terminal user interface."""

from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from rich.console import Group
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.css.query import NoMatches
from textual.widgets import Static, TextArea
from textual.worker import Worker

from litecoder.cli.local_commands import LocalCommandRouter
from litecoder.tools.permission import (
    PERMISSION_CONFIRMATION_TIMEOUT_SECONDS,
    PermissionMode,
    PermissionPrompt,
    PromptChoice,
)
from litecoder.ui.events import RuntimeUIEvent, UIEventType
from litecoder.ui.presentation import PresentationReducer
from litecoder.ui.presenters import (
    optional_text,
    normalize_windows_drive_letters,
    permission_detail_lines,
    permission_title,
)
from litecoder.ui.textual_widgets import (
    TranscriptBlockWidget,
    render_footer,
    render_live_tail,
    startup_banner,
)

if TYPE_CHECKING:
    from litecoder.agent.runtime import AgentRuntime


class PromptEditor(TextArea):
    """Multiline composer: Enter submits, Shift+Enter inserts a newline."""

    class Submitted(Message):
        """Component responsible for the submitted."""
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return
        if event.key in {"shift+enter", "ctrl+j"}:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)


class RuntimeEventsReady(Message):
    """Signals that the sink has one FIFO batch ready for the UI loop."""


class FlushRuntimeOutput(Message):
    """Component responsible for the flush runtime output."""
    def __init__(self, completed: asyncio.Future[None]) -> None:
        super().__init__()
        self.completed = completed


class TextualUISink:
    """Thread-safe-enough bridge from runtime callbacks to the Textual message loop."""

    def __init__(self) -> None:
        self.app: LiteCoderApp | None = None
        self._pending: deque[RuntimeUIEvent] = deque()
        self._lock = Lock()
        self._drain_posted = False

    def bind(self, app: LiteCoderApp) -> None:
        """Temporarily bind this context for nested operations."""
        with self._lock:
            self.app = app
        self._request_drain()

    def emit(self, event: RuntimeUIEvent) -> None:
        """Emit the supplied event."""
        with self._lock:
            self._pending.append(event)
        self._request_drain()

    def flush(self) -> asyncio.Future[None] | None:
        """Flush pending output."""
        with self._lock:
            app = self.app
        if app is None or not app.is_running:
            return None
        self._request_drain()
        completed = asyncio.get_running_loop().create_future()
        if not app.post_message(FlushRuntimeOutput(completed)):
            completed.set_result(None)
        return completed

    def take_pending(self) -> tuple[RuntimeUIEvent, ...]:
        """Atomically take the next FIFO batch without reordering events."""

        with self._lock:
            events = tuple(self._pending)
            self._pending.clear()
            self._drain_posted = False
        return events

    def _request_drain(self) -> None:
        with self._lock:
            app = self.app
            if app is None or self._drain_posted or not self._pending:
                return
            self._drain_posted = True
        if app.post_message(RuntimeEventsReady()):
            return
        with self._lock:
            self._drain_posted = False


class TextualPermissionPrompt:
    """Component responsible for the textual permission prompt."""
    def __init__(self) -> None:
        self.app: LiteCoderApp | None = None

    def bind(self, app: LiteCoderApp) -> None:
        """Temporarily bind this context for nested operations."""
        self.app = app

    async def __call__(self, prompt: PermissionPrompt) -> PromptChoice:
        app = self.app
        if app is None or not app.is_running:
            return PromptChoice.DENY
        return await app.request_permission(prompt)


@dataclass(slots=True)
class _PermissionRequest:
    """Data model representing the permission request."""
    prompt: PermissionPrompt
    future: asyncio.Future[PromptChoice]


class PermissionPane(Vertical, can_focus=True):
    """Inline permission prompt rendered at the end of the output flow."""

    class Resolved(Message):
        """Component responsible for the resolved."""
        def __init__(self, choice: PromptChoice) -> None:
            super().__init__()
            self.choice = choice

    BINDINGS = [
        Binding("up,k", "previous", "Previous", show=False, priority=True),
        Binding("down,j,tab", "next", "Next", show=False, priority=True),
        Binding("enter", "choose", "Choose", show=False, priority=True),
        Binding("escape", "deny", "Deny", show=False, priority=True),
        Binding("1", "allow_once", "Allow once", show=False, priority=True),
        Binding("2", "allow_session", "Allow session", show=False, priority=True),
        Binding("3", "deny", "Deny", show=False, priority=True),
    ]

    _choices = (
        PromptChoice.ALLOW_ONCE,
        PromptChoice.ALLOW_FOR_ROOT_SESSION,
        PromptChoice.DENY,
    )

    def __init__(self) -> None:
        super().__init__(id="permission-pane")
        self.prompt: PermissionPrompt | None = None
        self.selected = 0
        self.display = False

    def compose(self) -> ComposeResult:
        """Compose the current UI view."""
        yield Static("", id="permission-title")
        yield Static("", id="permission-detail")
        yield Static("", id="permission-options")
        yield Static(
            Text(
                "Enter to confirm · Esc to deny · "
                f"Auto-deny after {PERMISSION_CONFIRMATION_TIMEOUT_SECONDS:g} seconds",
                style="dim italic",
            ),
            id="permission-help",
        )

    def show_prompt(self, prompt: PermissionPrompt) -> None:
        """Handle the show prompt operation."""
        self.prompt = prompt
        self.selected = 0
        self.query_one("#permission-title", Static).update(
            Text(
                f"Allow {normalize_windows_drive_letters(permission_title(prompt))}?",
                style="bold yellow",
            )
        )
        self.query_one("#permission-detail", Static).update(
            Group(
                *(
                    Text(normalize_windows_drive_letters(line), style="dim")
                    for line in permission_detail_lines(prompt)
                )
            )
        )
        self._refresh_options()

    def action_previous(self) -> None:
        """Handle the action previous operation."""
        self.selected = (self.selected - 1) % len(self._choices)
        self._refresh_options()

    def action_next(self) -> None:
        """Handle the action next operation."""
        self.selected = (self.selected + 1) % len(self._choices)
        self._refresh_options()

    def action_choose(self) -> None:
        """Handle the action choose operation."""
        self._resolve(self._choices[self.selected])

    def action_allow_once(self) -> None:
        """Handle the action allow once operation."""
        self._resolve(PromptChoice.ALLOW_ONCE)

    def action_allow_session(self) -> None:
        """Handle the action allow session operation."""
        self._resolve(PromptChoice.ALLOW_FOR_ROOT_SESSION)

    def action_deny(self) -> None:
        """Handle the action deny operation."""
        self._resolve(PromptChoice.DENY)

    def _resolve(self, choice: PromptChoice) -> None:
        self.post_message(self.Resolved(choice))

    def _refresh_options(self) -> None:
        self.query_one("#permission-options", Static).update(self._render_options())

    def _render_options(self) -> Text:
        labels = (
            "Yes",
            "Yes, and don't ask again for this session",
            "No",
        )
        text = Text()
        for index, label in enumerate(labels):
            if index:
                text.append("\n")
            selected = index == self.selected
            selected_style = "bold yellow" if selected else ""
            text.append("› " if selected else "  ", style=selected_style)
            text.append(f"{index + 1}. {label}", style=selected_style)
        return text


def _windows_inline_driver_class():  # type: ignore[no-untyped-def]
    if os.name != "nt":
        return None
    from litecoder.ui.windows_inline_driver import WindowsInlineDriver

    return WindowsInlineDriver


class LiteCoderApp(App[str | None]):
    """Component responsible for the lite coder app."""
    TITLE = "LiteCoder"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding(
            "shift+tab",
            "cycle_permission_mode",
            "Permission mode",
            show=False,
            priority=True,
        ),
        Binding("ctrl+o", "toggle_details", "Toggle details", show=False),
        Binding("escape", "cancel_turn", "Cancel", show=False, priority=True),
        Binding("ctrl+c", "interrupt", "Interrupt", show=False),
    ]

    CSS = """
    Screen {
        layout: vertical;
        background: #0d0d0d;
        color: #eeeeee;
        height: 100%;
        overflow-y: hidden;
    }

    #page {
        width: 100%;
        height: 100%;
        overflow-y: auto;
        scrollbar-size: 1 1;
        scrollbar-background: #0d0d0d;
        scrollbar-background-hover: #0d0d0d;
        scrollbar-background-active: #0d0d0d;
        scrollbar-color: #555555;
        scrollbar-color-hover: #777777;
        scrollbar-color-active: #888888;
    }

    #permission-pane {
        width: 100%;
        height: auto;
        padding: 0 0 1 0;
        margin-top: 1;
        border-top: solid #d7af5f;
        background: #0d0d0d;
    }

    #permission-title, #permission-detail, #permission-options, #permission-help {
        width: 100%;
        height: auto;
    }

    #permission-title {
        margin-top: 1;
    }

    #permission-options, #permission-help {
        margin-top: 1;
    }

    #permission-help {
        color: #777777;
    }

    #messages {
        height: auto;
        padding: 0;
        overflow-y: hidden;
    }

    #transcript {
        width: 100%;
        height: auto;
    }

    .message {
        width: 100%;
        height: auto;
        margin-bottom: 1;
        padding-right: 0;
    }

    .banner-message {
        margin-bottom: 1;
    }

    .user-message {
        padding: 0 2 0 1;
        background: #303030;
        color: white;
    }

    .thinking-message, .summary-message, .notice-message {
        color: #808080;
    }

    #bottom-dock {
        width: 100%;
        height: auto;
        background: #0d0d0d;
        margin-bottom: 0;
    }

    #live-tail {
        width: 100%;
        height: auto;
        padding: 0;
        background: #0d0d0d;
    }


    #prompt-container {
        width: 100%;
        height: 3;
        min-height: 3;
        max-height: 8;
        border-top: solid #666666;
        border-bottom: solid #666666;
        background: #0d0d0d;
    }

    #prompt-container:focus-within {
        border-top: solid #858585;
        border-bottom: solid #858585;
    }

    #prompt-prefix {
        width: 3;
        height: 1;
        padding-left: 1;
        color: white;
        background: #0d0d0d;
    }

    PromptEditor {
        width: 1fr;
        height: 100%;
        min-height: 1;
        max-height: 6;
        padding: 0;
        border: none;
        background: #0d0d0d;
        color: white;
        scrollbar-size: 1 1;
    }

    PromptEditor .text-area--cursor-line {
        background: transparent;
    }

    #footer {
        width: 100%;
        height: auto;
        min-height: 1;
        padding: 0 1;
        background: #0d0d0d;
        color: #777777;
    }
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        sink: TextualUISink,
        permission_prompt: TextualPermissionPrompt,
        session_id: str | None = None,
    ) -> None:
        super().__init__(driver_class=_windows_inline_driver_class())
        self.runtime = runtime
        self.sink = sink
        self.permission_prompt = permission_prompt
        self.session_id = session_id
        self.router = LocalCommandRouter(runtime)
        self.workspace_path = _runtime_workspace_path(runtime)
        self.model_name = _runtime_model(runtime)
        self.permission_mode = _runtime_permission_mode(runtime)
        self.reducer = PresentationReducer(workspace_root=self.workspace_path)
        self._block_widgets: dict[str, TranscriptBlockWidget] = {}
        self._turn_worker: Worker[object] | None = None
        self._queued_prompts: deque[str] = deque()
        self._animation_frame = 0
        self._interrupt_count = 0
        self._details_expanded = False
        self._permission_active: _PermissionRequest | None = None
        self._permission_queue: deque[_PermissionRequest] = deque()
        self._permission_timeout_task: asyncio.Task[None] | None = None
        sink.bind(self)
        permission_prompt.bind(self)

    def compose(self) -> ComposeResult:
        """Compose the current UI view."""
        with VerticalScroll(id="page"):
            with Vertical(id="messages"):
                yield Vertical(id="transcript")
                yield Static("", id="live-tail")
                yield PermissionPane()
            with Vertical(id="bottom-dock"):
                with Horizontal(id="prompt-container"):
                    yield Static(Text("> ", style="bold white"), id="prompt-prefix")
                    yield PromptEditor(
                        "",
                        id="prompt",
                        show_line_numbers=False,
                        soft_wrap=True,
                        compact=True,
                    )
                yield Static("", id="footer")

    async def on_mount(self) -> None:
        """Handle the on mount operation."""
        self.model_name = await _initial_model(self.runtime, self.session_id)
        _apply_runtime_permission_mode(self.runtime, self.permission_mode)
        await self._mount_banner()
        await self._restore_session()
        await self._sync_transcript()
        self._refresh_bottom()
        self.set_interval(0.08, self._tick_animation)
        self.query_one("#prompt", PromptEditor).focus()

    async def request_permission(self, prompt: PermissionPrompt) -> PromptChoice:
        """Handle the request permission operation."""
        if not self.is_running:
            return PromptChoice.DENY
        future = asyncio.get_running_loop().create_future()
        request = _PermissionRequest(prompt, future)
        self._permission_queue.append(request)
        try:
            if self._permission_active is None:
                await self._activate_next_permission()
            return await future
        finally:
            if self._permission_active is request:
                self._cancel_permission_timeout()
                self._permission_active = None
                self._hide_permission_pane()
                await self._activate_next_permission()
            else:
                with suppress(ValueError):
                    self._permission_queue.remove(request)

    async def _activate_next_permission(self) -> None:
        if self._permission_active is not None:
            return
        while self._permission_queue:
            request = self._permission_queue.popleft()
            if request.future.done():
                continue
            self._permission_active = request
            await self._drain_runtime_events()
            if self._permission_active is not request:
                return
            if request.future.done():
                self._permission_active = None
                continue
            pane = self.query_one("#permission-pane", PermissionPane)
            pane.show_prompt(request.prompt)
            pane.display = True
            pane.focus()
            self._start_permission_timeout(request)
            return
        self._focus_prompt()

    def _start_permission_timeout(self, request: _PermissionRequest) -> None:
        self._cancel_permission_timeout()
        self._permission_timeout_task = asyncio.create_task(
            self._deny_permission_after_timeout(request),
            name="litecoder-permission-timeout",
        )

    def _cancel_permission_timeout(self) -> None:
        task = self._permission_timeout_task
        self._permission_timeout_task = None
        if (
            task is not None
            and task is not asyncio.current_task()
            and not task.done()
        ):
            task.cancel()

    async def _deny_permission_after_timeout(
        self, request: _PermissionRequest
    ) -> None:
        try:
            await asyncio.sleep(PERMISSION_CONFIRMATION_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            return
        if self._permission_active is request and not request.future.done():
            request.future.set_result(PromptChoice.DENY)

    def _hide_permission_pane(self) -> None:
        with suppress(NoMatches):
            pane = self.query_one("#permission-pane", PermissionPane)
            pane.display = False
            pane.prompt = None

    def _resolve_all_permissions(self, choice: PromptChoice) -> None:
        """Resolve the all permissions."""
        self._cancel_permission_timeout()
        active = self._permission_active
        if active is not None and not active.future.done():
            active.future.set_result(choice)
        while self._permission_queue:
            request = self._permission_queue.popleft()
            if not request.future.done():
                request.future.set_result(choice)
        self._hide_permission_pane()

    def _focus_prompt(self) -> None:
        if not self.is_running:
            return
        with suppress(NoMatches):
            self.query_one("#prompt", PromptEditor).focus()

    def on_unmount(self) -> None:
        """Handle the on unmount operation."""
        self._resolve_all_permissions(PromptChoice.DENY)

    async def on_prompt_editor_submitted(
        self,
        message: PromptEditor.Submitted,
    ) -> None:
        """Handle the on prompt editor submitted operation."""
        value = message.value
        editor = self.query_one("#prompt", PromptEditor)
        editor.clear()
        self._resize_editor()
        if not value.strip():
            return
        self._interrupt_count = 0
        if self._turn_worker is not None and self._turn_worker.is_running:
            self._queued_prompts.append(value)
            self._refresh_bottom()
            return
        self._start_prompt(value)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle the on text area changed operation."""
        if event.text_area.id == "prompt":
            event.text_area.cursor_blink = not bool(event.text_area.text.strip())
            self._resize_editor()

    def on_resize(self, _: events.Resize) -> None:
        """Handle the on resize operation."""
        self._refresh_live_tail()
        self._refresh_footer()

    def on_permission_pane_resolved(self, message: PermissionPane.Resolved) -> None:
        """Handle the on permission pane resolved operation."""
        message.stop()
        active = self._permission_active
        if active is not None and not active.future.done():
            active.future.set_result(message.choice)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Handle the on mouse down operation."""
        if event.button != 3:
            return
        selected_text = self.screen.get_selected_text()
        if selected_text:
            self.copy_to_clipboard(selected_text)
        event.stop()

    def on_click(self, event: events.Click) -> None:
        """Handle the on click operation."""
        if event.button != 1 or self._permission_active is not None:
            return
        try:
            self.query_one("#prompt", PromptEditor).focus(scroll_visible=False)
        except NoMatches:
            pass

    def _start_prompt(self, value: str) -> None:
        local_command = value.lstrip().startswith("/")
        self._turn_worker = self.run_worker(
            self._process_prompt(value),
            name="agent-turn",
            group="agent-turn",
            exclusive=True,
            exit_on_error=False,
        )
        if local_command:
            self.reducer.reset_live()
        else:
            self.reducer.start_turn_preview()
        self._refresh_bottom()

    async def _process_prompt(self, value: str) -> None:
        """Process the prompt."""
        try:
            self.reducer.add_user_prompt(value)
            await self._sync_transcript()
            local = await self.router.dispatch(value, session_id=self.session_id)
            if local.handled:
                if local.message:
                    self.reducer.add_command_output(local.message)
                    await self._sync_transcript()
                if local.replacement_session_id is not None:
                    self.session_id = local.replacement_session_id
                    self.model_name = await _initial_model(
                        self.runtime,
                        self.session_id,
                    )
                    await self._reload_session_view()
                if local.clear_requested:
                    await self._clear_session_view()
                if local.exit_requested:
                    self.exit(self.session_id)
                return

            result = (
                await self.runtime.run(value)
                if self.session_id is None
                else await self.runtime.resume(self.session_id, value)
            )
            self.session_id = result.session_id
        except asyncio.CancelledError:
            self.reducer.reset_live()
            self.reducer.add_notice_block("Interrupted by user")
            await self._sync_transcript()
            raise
        except Exception:
            self.reducer.reset_live(phase="error")
            self.reducer.add_error_block("Error", ("Unexpected internal error",))
            await self._sync_transcript()
        finally:
            if value.lstrip().startswith("/"):
                self.reducer.reset_live()
            self._turn_worker = None
            self._refresh_bottom()
            if self._queued_prompts and self.is_running:
                next_prompt = self._queued_prompts.popleft()
                self.call_later(self._start_prompt, next_prompt)

            if self._permission_active is None:
                self._focus_prompt()

    async def on_runtime_events_ready(self, message: RuntimeEventsReady) -> None:
        """Handle the on runtime events ready operation."""
        await self._drain_runtime_events()

    async def _drain_runtime_events(self) -> None:
        events = self.sink.take_pending()
        if not events:
            return
        transient_notice_key: str | None = None
        for event in events:
            self.reducer.apply(event)
            if event.type is UIEventType.NOTICE_RAISED:
                notice = self.reducer.state.live.notice
                if notice is not None and not notice.persistent:
                    transient_notice_key = notice.key
        await self._sync_transcript()
        self._refresh_bottom()
        if transient_notice_key is not None:
            self.set_timer(5.0, lambda: self._clear_notice(transient_notice_key))

    async def on_flush_runtime_output(self, message: FlushRuntimeOutput) -> None:
        """Handle the on flush runtime output operation."""
        try:
            await self._drain_runtime_events()
        finally:
            if not message.completed.done():
                message.completed.set_result(None)

    async def _sync_transcript(self) -> None:
        """Synchronize the transcript."""
        page = self.query_one("#page", VerticalScroll)
        follow_tail = page.is_vertical_scroll_end
        transcript = self.query_one("#transcript", Vertical)
        for block in self.reducer.state.blocks:
            widget = self._block_widgets.get(block.key)
            if widget is None:
                widget = TranscriptBlockWidget(
                    block,
                    expanded=self._details_expanded,
                )
                self._block_widgets[block.key] = widget
                await transcript.mount(widget)

        if follow_tail:
            page.scroll_end(animate=False, x_axis=False)

    async def _mount_banner(self) -> None:
        await self.query_one("#transcript", Vertical).mount(
            Static(
                startup_banner(self.workspace_path, self.model_name),
                classes="message banner-message",
            )
        )

    async def _restore_session(self) -> None:
        if self.session_id is None:
            return
        store = getattr(self.runtime, "store", None)
        load_context = getattr(store, "load_context", None)
        if callable(load_context):
            with suppress(Exception):
                context = await load_context(self.session_id)
                for message in getattr(context, "messages", ()):
                    self._restore_message(message)
        list_todos = getattr(store, "list_todos", None)
        if callable(list_todos):
            with suppress(Exception):
                todos = await list_todos(self.session_id)
                self.reducer.add_todo_history(todos)

    def _restore_message(self, message: object) -> None:
        role = getattr(message, "role", "")
        content = getattr(message, "content", ())
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            return
        if role == "assistant":
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = optional_text(block.get("text"))
                    if text:
                        self.reducer.add_assistant_history(text)
                elif block_type == "thinking":
                    text = optional_text(block.get("thinking"))
                    if text:
                        self.reducer.add_thinking_history(text)
                elif block_type in {"tool_call", "tool_use"}:
                    call_id = _first_text(
                        block.get("id"),
                        block.get("call_id"),
                        block.get("tool_call_id"),
                    )
                    name = optional_text(block.get("name")) or "tool"
                    arguments = block.get("input", block.get("arguments", {}))
                    if call_id and isinstance(arguments, Mapping):
                        self.reducer.add_tool_history(call_id, name, arguments)
        elif role == "user":
            text_parts: list[str] = []
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = optional_text(block.get("text"))
                    if text:
                        text_parts.append(text)
                elif block_type == "tool_result":
                    call_id = _first_text(
                        block.get("tool_call_id"),
                        block.get("tool_use_id"),
                    )
                    result = block.get("content")
                    result_text = (
                        result if isinstance(result, str) else str(result or "")
                    )
                    status = optional_text(block.get("status"))
                    failed = block.get("is_error") is True or status not in {
                        "",
                        "success",
                    }
                    if call_id:
                        self.reducer.finish_tool_history(
                            call_id,
                            content=result_text,
                            failed=failed,
                        )
            if text_parts:
                self.reducer.add_user_prompt("\n\n".join(text_parts))

    async def _reload_session_view(self) -> None:
        transcript = self.query_one("#transcript", Vertical)
        await transcript.remove_children()
        self._block_widgets.clear()
        self.reducer.clear()
        await self._mount_banner()
        await self._restore_session()
        await self._sync_transcript()
        self._refresh_bottom()

    async def _clear_session_view(self) -> None:
        self.session_id = None
        self.model_name = _runtime_model(self.runtime)
        await self._reload_session_view()

    def _tick_animation(self) -> None:
        self._animation_frame += 1
        self._refresh_live_tail()

    def _refresh_bottom(self) -> None:
        if not self.is_mounted:
            return
        self._refresh_live_tail()
        self._refresh_footer()

    def _refresh_live_tail(self) -> None:
        if not self.is_mounted:
            return
        try:
            tail = self.query_one("#live-tail", Static)
        except NoMatches:
            return
        tail.display = bool(self.reducer.state.live.visible or self._queued_prompts)
        tail.update(
            render_live_tail(
                self.reducer.state,
                animation_frame=self._animation_frame,
                queued_prompts=tuple(self._queued_prompts),
                expanded=self._details_expanded,
            )
        )
    def _refresh_footer(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#footer", Static).update(
            render_footer(
                permission_mode=self.permission_mode,
                model=self.model_name,
                workspace=self.workspace_path,
                width=max(1, self.size.width),
                usage=self.reducer.state.usage,
            )
        )

    def _resize_editor(self) -> None:
        if not self.is_mounted:
            return
        editor = self.query_one("#prompt", PromptEditor)
        height = max(3, min(8, editor.text.count("\n") + 3))
        self.query_one("#prompt-container", Horizontal).styles.height = height

    def _clear_notice(self, key: str) -> None:
        self.reducer.clear_live_notice(key)
        self._refresh_bottom()

    def action_cycle_permission_mode(self) -> None:
        """Handle the action cycle permission mode operation."""
        order = (
            PermissionMode.ASK,
            PermissionMode.READ_ONLY,
            PermissionMode.BYPASS,
        )
        self.permission_mode = order[
            (order.index(self.permission_mode) + 1) % len(order)
        ]
        _apply_runtime_permission_mode(self.runtime, self.permission_mode)
        self._refresh_footer()

    def action_toggle_details(self) -> None:
        """Handle the action toggle details operation."""
        self._details_expanded = not self._details_expanded
        for block in self.reducer.state.blocks:
            widget = self._block_widgets.get(block.key)
            if widget is not None:
                widget.update_block(
                    block,
                    expanded=self._details_expanded,
                    animation_frame=self._animation_frame,
                )

        self._refresh_live_tail()

    def action_cancel_turn(self) -> None:
        """Handle the action cancel turn operation."""
        worker = self._turn_worker
        if worker is not None and worker.is_running:
            self._resolve_all_permissions(PromptChoice.DENY)
            worker.cancel()
            return
        self.query_one("#prompt", PromptEditor).clear()

    def action_interrupt(self) -> None:
        """Handle the action interrupt operation."""
        self._interrupt_count += 1
        worker = self._turn_worker
        if worker is not None and worker.is_running:
            self._resolve_all_permissions(PromptChoice.DENY)
            worker.cancel()
        if self._interrupt_count >= 2:
            self._resolve_all_permissions(PromptChoice.DENY)
            self.exit(self.session_id)


def _runtime_workspace_path(runtime: object) -> str:
    paths = getattr(runtime, "paths", None)
    workspace_root = getattr(paths, "workspace_root", None)
    return str(workspace_root) if workspace_root is not None else str(Path.cwd())


def _runtime_model(runtime: object) -> str:
    model = getattr(runtime, "model", None)
    return model.strip() if isinstance(model, str) and model.strip() else "unknown"


async def _initial_model(runtime: object, session_id: str | None) -> str:
    if session_id is not None:
        store = getattr(runtime, "store", None)
        load_context = getattr(store, "load_context", None)
        if callable(load_context):
            with suppress(Exception):
                context = await load_context(session_id)
                session = getattr(context, "session", None)
                model = getattr(session, "model", None)
                if isinstance(model, str) and model.strip():
                    return model.strip()
    return _runtime_model(runtime)


def _runtime_permission_mode(runtime: object) -> PermissionMode:
    value = getattr(runtime, "permission_mode", PermissionMode.ASK.value)
    with suppress(TypeError, ValueError):
        return PermissionMode(str(value))
    return PermissionMode.ASK


def _apply_runtime_permission_mode(runtime: object, mode: PermissionMode) -> None:
    with suppress(Exception):
        setattr(runtime, "permission_mode", mode.value)


def _first_text(*values: object) -> str:
    for value in values:
        text = optional_text(value)
        if text:
            return text
    return ""
