"""Interactive terminal input handling."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass

from rich.cells import cell_len
from rich.console import Console

# prompt_toolkit has no native Shift+Enter key, so patched inputs use this sentinel.
_SHIFT_ENTER_KEY = "\U000f0000"
_SHIFT_ENTER_SEQUENCES = (
    "\x1b[27;2;13~",
    "\x1b[13;2u",
)
_WAIT_STATUS_ATTRIBUTE = "_litecoder_waiting_status"
_LIVE_DRAFT_ATTRIBUTE = "_litecoder_live_draft"
_STDOUT_PATCH_ATTRIBUTE = "_litecoder_stdout_patch"
_TODO_PROGRESS_ATTRIBUTE = "_litecoder_todo_progress"
_FINISHING_MESSAGE = "Finishing response..."
_TODO_VISIBLE_ITEMS = 6
_WAITING_MESSAGE = "Waiting for response..."
_SUBMITTED_STYLE = "white on #3a3a3a"
FooterText = str | Callable[[], str]
PermissionModeToggle = Callable[[], None]

_WAITING_FRAMES = (
    "\u2838",
    "\u283c",
    "\u2834",
    "\u2826",
    "\u2827",
    "\u2807",
    "\u280f",
    "\u280b",
    "\u2819",
    "\u2839",
)


@dataclass(frozen=True, slots=True)
class TodoProgressItem:
    """Data model representing the todo progress item."""
    content: str
    active_form: str
    status: str


def replace_todo_progress(console: Console, value: object) -> bool:
    """Handle the replace todo progress operation."""
    items = _parse_todo_progress(value)
    if items is None:
        return False
    setattr(console, _TODO_PROGRESS_ATTRIBUTE, items)
    _refresh_waiting_surface(console)
    return True


def clear_todo_progress(console: Console) -> None:
    """Clear the todo progress."""
    if getattr(console, _TODO_PROGRESS_ATTRIBUTE, ()):
        setattr(console, _TODO_PROGRESS_ATTRIBUTE, ())
        _refresh_waiting_surface(console)


def todo_progress_items(console: Console) -> tuple[TodoProgressItem, ...]:
    """Handle the todo progress items operation."""
    value = getattr(console, _TODO_PROGRESS_ATTRIBUTE, ())
    if not isinstance(value, tuple) or any(
        not isinstance(item, TodoProgressItem) for item in value
    ):
        return ()
    return value


def has_live_waiting_surface(console: Console) -> bool:
    """Return whether the live waiting surface condition holds."""
    return (
        getattr(console, _LIVE_DRAFT_ATTRIBUTE, None) is not None
        or getattr(console, _WAIT_STATUS_ATTRIBUTE, None) is not None
    )


class InputInterrupt(KeyboardInterrupt):
    """Component responsible for the input interrupt."""
    def __init__(self, source: str) -> None:
        super().__init__(source)
        self.source = source


class LiveInputInterrupt:
    """Component responsible for the live input interrupt."""
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.source: str | None = None

    @property
    def requested(self) -> bool:
        """Return whether the interrupt was requested."""
        return self._event.is_set()

    def request(self, source: str) -> None:
        """Handle the request operation."""
        if self.source is None:
            self.source = source
        self._event.set()

    async def wait(self) -> str:
        """Wait for the requested operation."""
        await self._event.wait()
        return self.source or "escape"


class _DraftState:
    """Internal helper for the draft state."""
    def __init__(self) -> None:
        self.app = None
        self.buffer = None
        self.submitted = False

    def text(self) -> str:
        """Return the current input text."""
        buffer = self.buffer
        value = getattr(buffer, "text", "") if buffer is not None else ""
        return value if isinstance(value, str) else ""

    def exit(self) -> None:
        """Exit the current input session."""
        app = self.app
        if app is None:
            return
        exit_method = getattr(app, "exit", None)
        if callable(exit_method):
            with suppress(Exception):
                exit_method(result=self.text())


class _StdoutPatchController:
    """Internal helper for the stdout patch controller."""
    def __init__(self, proxy: object, stdout: object, stderr: object) -> None:
        self.proxy = proxy
        self.stdout = stdout
        self.stderr = stderr

    @contextmanager
    def suspend(self):  # type: ignore[no-untyped-def]
        """Suspend terminal input handling."""
        flush = getattr(self.proxy, "flush", None)
        if callable(flush):
            with suppress(Exception):
                flush()
        stdout_was_proxy = sys.stdout is self.proxy
        stderr_was_proxy = sys.stderr is self.proxy
        if stdout_was_proxy:
            sys.stdout = self.stdout  # type: ignore[assignment]
        if stderr_was_proxy:
            sys.stderr = self.stderr  # type: ignore[assignment]
        try:
            yield
        finally:
            if stdout_was_proxy and sys.stdout is self.stdout:
                sys.stdout = self.proxy  # type: ignore[assignment]
            if stderr_was_proxy and sys.stderr is self.stderr:
                sys.stderr = self.proxy  # type: ignore[assignment]


class _LiveDraftController:
    """Internal helper for the live draft controller."""
    def __init__(
        self, terminal_input: "TerminalInput", draft_state: _DraftState
    ) -> None:
        self.terminal_input = terminal_input
        self.draft_state = draft_state

    def refresh(self) -> None:
        """Refresh the visible waiting surface."""
        _call_if_present(self.draft_state.app, "invalidate")

    @contextmanager
    def suspend(self):  # type: ignore[no-untyped-def]
        """Suspend terminal input handling."""
        if not self.draft_state.submitted:
            text = self.draft_state.text()
            if text:
                self.terminal_input._draft = text
        app = self.draft_state.app
        if app is None:
            yield
            return

        stack = ExitStack()
        try:
            _call_if_present(getattr(app, "renderer", None), "erase")
            with suppress(Exception):
                setattr(app, "_running_in_terminal", True)
            input_obj = getattr(app, "input", None)
            _enter_context_if_present(stack, input_obj, "detach")
            _enter_context_if_present(stack, input_obj, "cooked_mode")
            yield
        finally:
            stack.close()
            with suppress(Exception):
                setattr(app, "_running_in_terminal", False)
            _call_if_present(getattr(app, "renderer", None), "reset")
            _call_if_present(app, "_request_absolute_cursor_position")
            _call_if_present(app, "_redraw")


class _PromptPlaceholder:
    """Internal helper for the prompt placeholder."""
    def __init__(self, placeholder: str, style: str = "") -> None:
        self.placeholder = placeholder
        self.style = style

    def apply_transformation(self, transformation_input):  # type: ignore[no-untyped-def]
        """Handle the apply transformation operation."""
        from prompt_toolkit.layout.processors import Transformation

        if transformation_input.lineno == 0 and not transformation_input.document.text:
            return Transformation(
                fragments=transformation_input.fragments
                + [(self.style, self.placeholder)]
            )
        return Transformation(transformation_input.fragments)


def _call_if_present(target: object, name: str) -> None:
    method = getattr(target, name, None)
    if callable(method):
        with suppress(Exception):
            method()


def _enter_context_if_present(stack: ExitStack, target: object, name: str) -> None:
    factory = getattr(target, name, None)
    if not callable(factory):
        return
    with suppress(Exception):
        stack.enter_context(factory())


def stop_waiting_status(console: Console) -> None:
    """Stop the waiting status."""
    status = getattr(console, _WAIT_STATUS_ATTRIBUTE, None)
    if status is None:
        return
    if getattr(console, _WAIT_STATUS_ATTRIBUTE, None) is status:
        setattr(console, _WAIT_STATUS_ATTRIBUTE, None)
    stop = getattr(status, "stop", None)
    if callable(stop):
        stop()


@contextmanager
def suspend_waiting_status(console: Console):  # type: ignore[no-untyped-def]
    """Handle the suspend waiting status operation."""
    live_draft = getattr(console, _LIVE_DRAFT_ATTRIBUTE, None)
    stdout_patch = getattr(console, _STDOUT_PATCH_ATTRIBUTE, None)
    status = getattr(console, _WAIT_STATUS_ATTRIBUTE, None)
    with ExitStack() as stack:
        suspend = getattr(live_draft, "suspend", None)
        if callable(suspend):
            stack.enter_context(suspend())
        pause_stdout = getattr(stdout_patch, "suspend", None)
        if callable(pause_stdout):
            stack.enter_context(pause_stdout())
        start = None
        if status is not None:
            stop = getattr(status, "stop", None)
            start = getattr(status, "start", None)
            if callable(stop):
                stop()
        try:
            yield
        finally:
            if (
                status is not None
                and getattr(console, _WAIT_STATUS_ATTRIBUTE, None) is status
                and callable(start)
            ):
                start()


@contextmanager
def _patched_stdout_context(console: Console | None = None):  # type: ignore[no-untyped-def]
    from prompt_toolkit.patch_stdout import StdoutProxy

    proxy = StdoutProxy(raw=True)
    stdout = sys.stdout
    stderr = sys.stderr
    controller = _StdoutPatchController(proxy, stdout, stderr)
    previous = getattr(console, _STDOUT_PATCH_ATTRIBUTE, None) if console else None
    if console is not None:
        setattr(console, _STDOUT_PATCH_ATTRIBUTE, controller)
    sys.stdout = proxy  # type: ignore[assignment]
    sys.stderr = proxy  # type: ignore[assignment]
    try:
        yield
    finally:
        if sys.stdout is proxy:
            sys.stdout = stdout
        if sys.stderr is proxy:
            sys.stderr = stderr
        if console is not None and getattr(console, _STDOUT_PATCH_ATTRIBUTE, None) is controller:
            if previous is None:
                setattr(console, _STDOUT_PATCH_ATTRIBUTE, None)
            else:
                setattr(console, _STDOUT_PATCH_ATTRIBUTE, previous)
        proxy.close()


class TerminalInput:
    """Component responsible for the terminal input."""
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._draft = ""
        self._queued_submissions: list[str] = []
        self._queued_submission: str | None = None
        self._prompt_rendered_submission = False

    async def read_async(
        self,
        *,
        footer: FooterText | None = None,
        on_permission_mode_toggle: PermissionModeToggle | None = None,
    ) -> str:
        """Read the async."""
        should_echo = True
        self._prompt_rendered_submission = False
        if self._queued_submissions:
            value = self._queued_submissions.pop(0)
            self._queued_submission = (
                self._queued_submissions[0] if self._queued_submissions else None
            )
        elif self._queued_submission is not None:
            value = self._queued_submission
            self._queued_submission = None
        elif self._should_use_framed_prompt():
            default, self._draft = self._draft, ""
            if on_permission_mode_toggle is None:
                value = await self._read_framed_async(footer, default)
            else:
                value = await self._read_framed_async(
                    footer,
                    default,
                    on_permission_mode_toggle=on_permission_mode_toggle,
                )
            should_echo = not self._prompt_rendered_submission
        elif self._draft:
            value, self._draft = self._draft, ""
        else:
            value = self.console.input("> ")
            should_echo = False
        if should_echo and value.strip():
            self.render_submitted(value)
        return value

    def render_submitted(self, value: str) -> None:
        """Render the submitted."""
        if not value.strip():
            return
        if self._should_use_framed_prompt():
            self._render_submitted_lines(
                value,
                style=_SUBMITTED_STYLE,
                pad_width=_console_width(self.console),
            )
            self.console.print()
            return
        self._render_submitted_lines(value)

    def _render_submitted_lines(
        self,
        value: str,
        *,
        style: str | None = None,
        pad_width: int | None = None,
    ) -> None:
        lines = value.splitlines() or [""]
        for index, line in enumerate(lines):
            prefix = "> " if index == 0 else " " * cell_len("> ")
            output = f"{prefix}{line}"
            if pad_width is not None:
                output = _clip_cells(output, pad_width)
            self.console.print(output, style=style)

    @asynccontextmanager
    async def live_draft_area(
        self,
        footer: FooterText | None = None,
        on_permission_mode_toggle: PermissionModeToggle | None = None,
    ):  # type: ignore[no-untyped-def]
        """Handle the live draft area operation."""
        interrupt = LiveInputInterrupt()
        if self._should_start_live_draft_reader():
            draft_state = _DraftState()
            controller = _LiveDraftController(self, draft_state)
            setattr(self.console, _LIVE_DRAFT_ATTRIBUTE, controller)
            try:
                with _patched_stdout_context(self.console):
                    task = asyncio.create_task(
                        self._capture_live_draft(
                            footer,
                            draft_state,
                            interrupt,
                            on_permission_mode_toggle,
                        )
                    )
                    try:
                        yield interrupt
                    finally:
                        await self._finish_live_draft(task, draft_state)
            finally:
                if getattr(self.console, _LIVE_DRAFT_ATTRIBUTE, None) is controller:
                    setattr(self.console, _LIVE_DRAFT_ATTRIBUTE, None)
            return

        if not self._should_show_wait_status():
            yield interrupt
            return
        status = self.console.status(
            _waiting_status_message(self.console),
            spinner="dots",
            refresh_per_second=8,
        )
        setattr(self.console, _WAIT_STATUS_ATTRIBUTE, status)
        status.start()
        try:
            yield interrupt
        finally:
            if getattr(self.console, _WAIT_STATUS_ATTRIBUTE, None) is status:
                setattr(self.console, _WAIT_STATUS_ATTRIBUTE, None)
                status.stop()

    async def _capture_live_draft(
        self,
        footer: FooterText | None,
        draft_state: _DraftState,
        interrupt: LiveInputInterrupt,
        on_permission_mode_toggle: PermissionModeToggle | None,
    ) -> None:
        while True:
            draft_state.submitted = False
            draft_state.buffer = None
            draft_state.app = None
            try:
                kwargs: dict[str, object] = {
                    "mark_submitted": False,
                    "draft_state": draft_state,
                    "show_waiting": True,
                    "transient": True,
                    "queued_messages": tuple(self._queued_submissions),
                }
                if on_permission_mode_toggle is not None:
                    kwargs["on_permission_mode_toggle"] = on_permission_mode_toggle
                value = await self._read_framed_async(
                    footer,
                    self._draft,
                    **kwargs,
                )
            except asyncio.CancelledError:
                raise
            except InputInterrupt as error:
                self._draft = draft_state.text()
                interrupt.request(error.source)
                return
            except KeyboardInterrupt:
                self._draft = draft_state.text()
                interrupt.request("ctrl_c")
                return
            except EOFError:
                return
            except Exception:
                return

            if draft_state.submitted:
                if value.strip():
                    self._queue_submission(value)
                self._draft = ""
                continue
            self._draft = value or draft_state.text()
            return

    def _queue_submission(self, value: str) -> None:
        self._queued_submissions.append(value)
        self._queued_submission = self._queued_submissions[0]

    async def _finish_live_draft(
        self,
        task: asyncio.Task[None],
        draft_state: _DraftState,
    ) -> None:
        if not task.done():
            text = draft_state.text()
            if text and not draft_state.submitted:
                self._draft = text
            draft_state.exit()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.2)
        except TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def _read_framed_async(
        self,
        footer: FooterText | None,
        default: str = "",
        *,
        mark_submitted: bool = True,
        draft_state: _DraftState | None = None,
        show_waiting: bool = False,
        transient: bool = False,
        queued_messages: Sequence[str] = (),
        on_permission_mode_toggle: PermissionModeToggle | None = None,
    ) -> str:
        """Read the framed async."""
        from prompt_toolkit.application import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.cursor_shapes import CursorShape, SimpleCursorShapeConfig
        from prompt_toolkit.layout import HSplit, Layout, Window
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.layout.processors import BeforeInput
        from prompt_toolkit.styles import Style

        _install_shift_enter_support()

        width = _console_width(self.console)
        app_holder: dict[str, object] = {}

        def accept(buffer: Buffer) -> bool:
            if draft_state is not None:
                draft_state.submitted = True
            app = app_holder["app"]
            app.exit(result=buffer.text)  # type: ignore[attr-defined]
            return True

        buffer = Buffer(multiline=True, accept_handler=accept)
        if draft_state is not None:
            draft_state.buffer = buffer
        if default:
            buffer.text = default
            buffer.cursor_position = len(default)
        input_control = BufferControl(
            buffer=buffer,
            input_processors=[
                BeforeInput("> ", style="bold"),
                _PromptPlaceholder("Type here", style="class:placeholder"),
            ],
        )

        def line() -> list[tuple[str, str]]:
            return [("class:border", _input_border(width))]

        children = []
        if show_waiting:
            children.append(
                Window(
                    FormattedTextControl(_waiting_text_factory(self.console)),
                    height=_waiting_height_factory(self.console),
                    dont_extend_height=True,
                    wrap_lines=False,
                )
            )
        if queued_messages:
            children.append(
                Window(
                    FormattedTextControl(
                        _queued_message_text_factory(queued_messages, width)
                    ),
                    height=1,
                    dont_extend_height=True,
                    wrap_lines=False,
                )
            )
        children.extend([
            Window(
                FormattedTextControl(line),
                height=1,
                dont_extend_height=True,
                wrap_lines=False,
            ),
            Window(
                input_control,
                height=Dimension(min=1, max=6),
                dont_extend_height=True,
                wrap_lines=True,
            ),
            Window(
                FormattedTextControl(line),
                height=1,
                dont_extend_height=True,
                wrap_lines=False,
            ),
        ])
        if footer:
            children.append(
                Window(
                    FormattedTextControl(_footer_text_factory(footer, width)),
                    height=1,
                    dont_extend_height=True,
                    wrap_lines=False,
                )
            )
        root = HSplit(children)
        app = Application(
            layout=Layout(root, focused_element=input_control),
            key_bindings=_submission_key_bindings(
                on_permission_mode_toggle=on_permission_mode_toggle
            ),
            full_screen=False,
            erase_when_done=transient or mark_submitted,
            refresh_interval=0.12 if show_waiting else None,
            cursor=SimpleCursorShapeConfig(CursorShape.BLOCK),
            style=Style.from_dict(
                {
                    "border": "ansibrightblack",
                    "placeholder": "ansibrightblack",
                    "queued": "bg:ansibrightblack ansiwhite",
                    "waiting": "ansibrightblack",
                    "todo.title": "ansiyellow bold",
                    "todo.completed": "ansigreen",
                    "todo.active": "ansiwhite bold",
                    "todo.pending": "ansibrightblack",
                    "todo.summary": "ansibrightblack",
                    "footer": "ansibrightblack",
                    "permission.ask": "ansiyellow",
                    "permission.read_only": "ansigreen",
                    "permission.bypass": "ansired",
                }
            ),
        )
        app_holder["app"] = app
        if draft_state is not None:
            draft_state.app = app
        value = await app.run_async()
        return value or ""

    def _should_use_framed_prompt(self) -> bool:
        return bool(getattr(self.console, "is_terminal", False))

    def _should_start_live_draft_reader(self) -> bool:
        return self._should_use_framed_prompt() and hasattr(self.console, "input")

    def _should_show_wait_status(self) -> bool:
        return self._should_use_framed_prompt() and hasattr(self.console, "status")


def _footer_text_factory(footer: FooterText, width: int):  # type: ignore[no-untyped-def]
    def footer_text() -> list[tuple[str, str]]:
        text = _clip_cells_without_padding(
            _footer_value(footer).strip(), max(20, width)
        )
        return _footer_fragments(text)

    return footer_text


def _footer_value(footer: FooterText) -> str:
    value = footer() if callable(footer) else footer
    return value if isinstance(value, str) else ""


def _footer_fragments(text: str) -> list[tuple[str, str]]:
    for label, style in _PERMISSION_FOOTER_STYLES:
        if text == label or text.startswith(f"{label} "):
            padding = " " * (_PERMISSION_FOOTER_WIDTH - cell_len(label))
            return [(style, label), ("class:footer", f"{padding}{text[len(label):]}")]
    return [("class:footer", text)]


_PERMISSION_FOOTER_STYLES = (
    ("Ask", "class:permission.ask"),
    ("Read-only", "class:permission.read_only"),
    ("Bypass", "class:permission.bypass"),
)
_PERMISSION_FOOTER_WIDTH = max(
    cell_len(label) for label, _style in _PERMISSION_FOOTER_STYLES
)


def _queued_message_text_factory(
    queued_messages: Sequence[str], width: int
):  # type: ignore[no-untyped-def]
    text = _queued_message_line(queued_messages, width)

    def queued_text() -> list[tuple[str, str]]:
        return [("class:queued", text)]

    return queued_text


def _queued_message_line(queued_messages: Sequence[str], width: int) -> str:
    message = queued_messages[-1] if queued_messages else ""
    first_line = next((line for line in message.splitlines() if line.strip()), message)
    suffix = f"  +{len(queued_messages) - 1} queued" if len(queued_messages) > 1 else ""
    return _clip_cells(f"  > {first_line.strip()}{suffix}", max(20, width))


def _clip_cells_without_padding(value: str, width: int) -> str:
    if cell_len(value) <= width:
        return value
    ellipsis = "..."
    target = max(0, width - cell_len(ellipsis))
    output = ""
    for character in value:
        if cell_len(output + character) > target:
            break
        output += character
    return output + ellipsis

def _clip_cells(value: str, width: int) -> str:
    if cell_len(value) <= width:
        return value + " " * max(0, width - cell_len(value))
    ellipsis = "..."
    target = max(0, width - cell_len(ellipsis))
    output = ""
    for character in value:
        if cell_len(output + character) > target:
            break
        output += character
    return output + ellipsis


def _waiting_text_factory(console: Console):  # type: ignore[no-untyped-def]
    frame_index = -1

    def waiting_text() -> list[tuple[str, str]]:
        nonlocal frame_index
        frame_index = (frame_index + 1) % len(_WAITING_FRAMES)
        frame = _WAITING_FRAMES[frame_index]
        items = todo_progress_items(console)
        if not items:
            return [("class:waiting", f"{frame} {_WAITING_MESSAGE}")]
        fragments = [
            ("class:todo.title", f"{frame} {_todo_progress_heading(items)}")
        ]
        for index, (style, text) in enumerate(_todo_progress_lines(items)):
            prefix = "  └ " if index == 0 else "    "
            fragments.append((style, f"\n{prefix}{text}"))
        return fragments

    return waiting_text


def _waiting_height_factory(console: Console):  # type: ignore[no-untyped-def]
    def waiting_height() -> int:
        items = todo_progress_items(console)
        return 1 + len(_todo_progress_lines(items)) if items else 1

    return waiting_height


def _waiting_status_message(console: Console) -> str:
    items = todo_progress_items(console)
    if not items:
        return _WAITING_MESSAGE
    lines = [_todo_progress_heading(items)]
    for index, (_, text) in enumerate(_todo_progress_lines(items)):
        prefix = "└ " if index == 0 else "  "
        lines.append(f"{prefix}{text}")
    return "\n".join(lines)


def _todo_progress_heading(items: tuple[TodoProgressItem, ...]) -> str:
    active = next((item for item in items if item.status == "in_progress"), None)
    if active is not None:
        return _with_ellipsis(active.active_form)
    pending = next((item for item in items if item.status == "pending"), None)
    if pending is not None:
        return _with_ellipsis(pending.active_form)
    return _FINISHING_MESSAGE


def _with_ellipsis(value: str) -> str:
    stripped = value.rstrip()
    return stripped if stripped.endswith(("...", "…")) else f"{stripped}..."


def _todo_progress_lines(
    items: tuple[TodoProgressItem, ...],
) -> list[tuple[str, str]]:
    if not items:
        return []
    start, end = _todo_visible_range(items)
    lines: list[tuple[str, str]] = []
    if start:
        hidden = items[:start]
        completed = sum(item.status == "completed" for item in hidden)
        label = "completed" if completed == len(hidden) else "earlier"
        lines.append(("class:todo.summary", f"… +{len(hidden)} {label}"))
    for item in items[start:end]:
        symbol = "✓" if item.status == "completed" else "□"
        style = {
            "completed": "class:todo.completed",
            "in_progress": "class:todo.active",
            "pending": "class:todo.pending",
        }[item.status]
        lines.append((style, f"{symbol} {item.content}"))
    if end < len(items):
        hidden = items[end:]
        pending = sum(item.status == "pending" for item in hidden)
        label = "pending" if pending == len(hidden) else "remaining"
        lines.append(("class:todo.summary", f"… +{len(hidden)} {label}"))
    return lines


def _todo_visible_range(items: tuple[TodoProgressItem, ...]) -> tuple[int, int]:
    if len(items) <= _TODO_VISIBLE_ITEMS:
        return 0, len(items)
    focus = next(
        (index for index, item in enumerate(items) if item.status == "in_progress"),
        next(
            (index for index, item in enumerate(items) if item.status == "pending"),
            len(items) - 1,
        ),
    )
    start = 0 if focus < _TODO_VISIBLE_ITEMS else max(0, focus - 2)
    end = min(len(items), start + _TODO_VISIBLE_ITEMS)
    start = max(0, end - _TODO_VISIBLE_ITEMS)
    return start, end


def _parse_todo_progress(
    value: object,
) -> tuple[TodoProgressItem, ...] | None:
    """Parse the todo progress."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    items: list[TodoProgressItem] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            return None
        content = raw.get("content")
        active_form = raw.get("active_form")
        status = raw.get("status")
        if (
            not isinstance(content, str)
            or not content.strip()
            or not isinstance(active_form, str)
            or not active_form.strip()
            or status not in {"pending", "in_progress", "completed"}
        ):
            return None
        items.append(TodoProgressItem(content, active_form, str(status)))
    return tuple(items)


def _refresh_waiting_surface(console: Console) -> None:
    live_draft = getattr(console, _LIVE_DRAFT_ATTRIBUTE, None)
    refresh = getattr(live_draft, "refresh", None)
    if callable(refresh):
        with suppress(Exception):
            refresh()
    status = getattr(console, _WAIT_STATUS_ATTRIBUTE, None)
    update = getattr(status, "update", None)
    if callable(update):
        with suppress(Exception):
            update(status=_waiting_status_message(console))


def _submission_key_bindings(
    on_permission_mode_toggle: PermissionModeToggle | None = None,
):  # type: ignore[no-untyped-def]
    from prompt_toolkit.key_binding import KeyBindings

    bindings = KeyBindings()

    if on_permission_mode_toggle is not None:
        @bindings.add("s-tab", eager=True)
        def _(event):  # type: ignore[no-untyped-def]
            on_permission_mode_toggle()
            invalidate = getattr(event.app, "invalidate", None)
            if callable(invalidate):
                invalidate()

    @bindings.add("escape", eager=True)
    def _(event):  # type: ignore[no-untyped-def]
        event.app.exit(exception=InputInterrupt("escape"))

    @bindings.add("c-c", eager=True)
    def _(event):  # type: ignore[no-untyped-def]
        event.app.exit(exception=InputInterrupt("ctrl_c"))

    @bindings.add(_SHIFT_ENTER_KEY, eager=True)
    @bindings.add("c-j", eager=True)
    def _(event):  # type: ignore[no-untyped-def]
        event.app.current_buffer.insert_text("\n")

    @bindings.add("enter")
    def _(event):  # type: ignore[no-untyped-def]
        buffer = event.app.current_buffer
        if buffer.text.strip():
            buffer.validate_and_handle()

    return bindings


def _install_shift_enter_support() -> None:
    _install_vt100_shift_enter_support()
    _install_win32_shift_enter_support()


def _install_vt100_shift_enter_support() -> None:
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES

    for sequence in _SHIFT_ENTER_SEQUENCES:
        ANSI_SEQUENCES[sequence] = _SHIFT_ENTER_KEY  # type: ignore[assignment]

    try:
        from prompt_toolkit.input.vt100_parser import _IS_PREFIX_OF_LONGER_MATCH_CACHE
    except ImportError:
        return
    _IS_PREFIX_OF_LONGER_MATCH_CACHE.clear()


def _install_win32_shift_enter_support() -> None:
    _install_win32_console_shift_enter_support()
    _install_win32_vt100_shift_enter_support()


def _install_win32_console_shift_enter_support() -> None:
    try:
        from prompt_toolkit.input import win32 as win32_input
        from prompt_toolkit.key_binding.key_processor import KeyPress
        from prompt_toolkit.keys import Keys
    except (AssertionError, ImportError):
        return

    reader_cls = getattr(win32_input, "ConsoleInputReader", None)
    if reader_cls is None or getattr(reader_cls, "_litecoder_shift_enter_patch", False):
        return

    original = reader_cls._event_to_key_presses

    def patched(self, ev):  # type: ignore[no-untyped-def]
        key_presses = original(self, ev)
        if len(key_presses) != 1:
            return key_presses

        key_press = key_presses[0]
        shift_pressed = bool(ev.ControlKeyState & self.SHIFT_PRESSED)
        ctrl_pressed = bool(
            ev.ControlKeyState & self.LEFT_CTRL_PRESSED
            or ev.ControlKeyState & self.RIGHT_CTRL_PRESSED
        )
        if shift_pressed and not ctrl_pressed and key_press.key in {
            Keys.ControlM,
            Keys.ControlJ,
        }:
            return [KeyPress(_SHIFT_ENTER_KEY, key_press.data)]
        return key_presses

    reader_cls._event_to_key_presses = patched
    reader_cls._litecoder_shift_enter_original = original
    reader_cls._litecoder_shift_enter_patch = True


def _install_win32_vt100_shift_enter_support() -> None:
    try:
        from prompt_toolkit.input import win32 as win32_input
    except (AssertionError, ImportError):
        return

    reader_cls = getattr(win32_input, "Vt100ConsoleInputReader", None)
    if reader_cls is None or getattr(reader_cls, "_litecoder_shift_enter_patch", False):
        return

    original = reader_cls._get_keys

    def patched(self, read, input_records):  # type: ignore[no-untyped-def]
        """Return whether the output stream has been patched."""
        for index in range(read.value):
            input_record = input_records[index]
            if input_record.EventType not in win32_input.EventTypes:
                continue

            event_name = win32_input.EventTypes[input_record.EventType]
            event = getattr(input_record.Event, event_name)
            if not getattr(event, "KeyDown", False):
                continue

            char = event.uChar.UnicodeChar
            if char == "\x00":
                continue

            shift_pressed = bool(
                event.ControlKeyState & win32_input.ConsoleInputReader.SHIFT_PRESSED
            )
            ctrl_pressed = bool(
                event.ControlKeyState & win32_input.ConsoleInputReader.LEFT_CTRL_PRESSED
                or event.ControlKeyState & win32_input.ConsoleInputReader.RIGHT_CTRL_PRESSED
            )
            if shift_pressed and not ctrl_pressed and char in {"\r", "\n"}:
                yield _SHIFT_ENTER_KEY
            else:
                yield char

    reader_cls._get_keys = patched
    reader_cls._litecoder_shift_enter_original = original
    reader_cls._litecoder_shift_enter_patch = True


def _console_width(console: Console) -> int:
    width = getattr(console, "width", 80)
    return width if isinstance(width, int) and width > 0 else 80


def _input_border(width: int) -> str:
    return "\u2500" * max(20, width)
