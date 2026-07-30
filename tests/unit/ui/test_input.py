from __future__ import annotations

import asyncio
import sys
from io import StringIO
from types import SimpleNamespace

import pytest
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from rich.console import Console

from litecoder.ui import input as input_module
from litecoder.ui.input import TerminalInput


@pytest.mark.asyncio
async def test_terminal_input_uses_async_reader_and_renders_submitted_prompt() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=True, color_system=None, width=80)

    class ScriptedInput(TerminalInput):
        async def _read_framed_async(self, footer: str | None, default: str = "") -> str:
            return "first\nsecond"

    terminal_input = ScriptedInput(console)
    value = await terminal_input.read_async(footer="E:\\repo context: 1")

    output = stream.getvalue()
    assert value == "first\nsecond"
    assert "> first" in output
    assert "  second" in output
    assert "\u2500" * 80 not in output
    assert output.endswith("\n\n")
    assert "context: 1" not in output


def test_render_submitted_uses_compact_message_bar_with_spacing() -> None:
    class CaptureConsole:
        is_terminal = True
        width = 24

        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def print(self, value: str = "", *, style: str | None = None) -> None:
            self.calls.append((value, style))

    console = CaptureConsole()

    TerminalInput(console).render_submitted("sent\nagain")  # type: ignore[arg-type]

    assert console.calls == [
        ("> sent" + " " * 18, "white on #3a3a3a"),
        ("  again" + " " * 17, "white on #3a3a3a"),
        ("", None),
    ]


@pytest.mark.asyncio
async def test_live_draft_area_shows_waiting_status() -> None:
    events: list[str] = []

    class StatusContext:
        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

    class StatusConsole:
        width = 80
        is_terminal = True

        def status(
            self,
            message: str,
            *,
            spinner: str,
            refresh_per_second: int,
        ) -> StatusContext:
            events.append(message)
            events.append(spinner)
            events.append(str(refresh_per_second))
            return StatusContext()

    terminal_input = TerminalInput(StatusConsole())  # type: ignore[arg-type]

    async with terminal_input.live_draft_area("footer"):
        events.append("body")

    assert events == [
        "Waiting for response...",
        "dots",
        "8",
        "start",
        "body",
        "stop",
    ]


@pytest.mark.asyncio
async def test_stop_waiting_status_stops_active_status_once() -> None:
    events: list[str] = []

    class StatusContext:
        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

    class StatusConsole:
        width = 80
        is_terminal = True

        def status(
            self,
            message: str,
            *,
            spinner: str,
            refresh_per_second: int,
        ) -> StatusContext:
            events.append(message)
            events.append(spinner)
            events.append(str(refresh_per_second))
            return StatusContext()

    terminal_input = TerminalInput(StatusConsole())  # type: ignore[arg-type]

    async with terminal_input.live_draft_area("footer"):
        input_module.stop_waiting_status(terminal_input.console)
        events.append("body")

    assert events == [
        "Waiting for response...",
        "dots",
        "8",
        "start",
        "stop",
        "body",
    ]



def test_patched_stdout_context_restores_real_stdout_while_suspended(monkeypatch) -> None:
    class Proxy:
        def __init__(self, *, raw: bool) -> None:
            self.raw = raw
            self.flushed = False
            self.closed = False

        def flush(self) -> None:
            self.flushed = True

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("prompt_toolkit.patch_stdout.StdoutProxy", Proxy)

    console = Console(file=StringIO(), force_terminal=True, color_system=None)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with input_module._patched_stdout_context(console):
        proxy = sys.stdout
        assert proxy is not original_stdout
        assert proxy.raw is True
        assert getattr(console, input_module._STDOUT_PATCH_ATTRIBUTE) is not None
        with input_module.suspend_waiting_status(console):
            assert proxy.flushed is True
            assert sys.stdout is original_stdout
            assert sys.stderr is original_stderr
        assert sys.stdout is proxy
        assert sys.stderr is proxy

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert proxy.closed is True
    assert getattr(console, input_module._STDOUT_PATCH_ATTRIBUTE) is None

def test_suspend_waiting_status_suspends_active_live_draft_prompt() -> None:
    events: list[str] = []
    console = Console(file=StringIO(), force_terminal=True, color_system=None)
    terminal_input = TerminalInput(console)

    class Context:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            events.append(f"{self.name}-enter")

        def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
            events.append(f"{self.name}-exit")

    class AppInput:
        def detach(self) -> Context:
            return Context("detach")

        def cooked_mode(self) -> Context:
            return Context("cooked")

    class Renderer:
        def erase(self) -> None:
            events.append("erase")

        def reset(self) -> None:
            events.append("reset")

    class App:
        input = AppInput()
        renderer = Renderer()
        _running_in_terminal = False

        def _request_absolute_cursor_position(self) -> None:
            events.append("cursor")

        def _redraw(self) -> None:
            events.append("redraw")

    draft_state = input_module._DraftState()
    draft_state.app = App()
    draft_state.buffer = SimpleNamespace(text="typed while waiting")
    controller = input_module._LiveDraftController(terminal_input, draft_state)
    setattr(console, input_module._LIVE_DRAFT_ATTRIBUTE, controller)

    with input_module.suspend_waiting_status(console):
        events.append("permission")

    assert terminal_input._draft == "typed while waiting"
    assert App._running_in_terminal is False
    assert events == [
        "erase",
        "detach-enter",
        "cooked-enter",
        "permission",
        "cooked-exit",
        "detach-exit",
        "reset",
        "cursor",
        "redraw",
    ]


@pytest.mark.asyncio
async def test_live_draft_area_starts_next_prompt_inside_patched_stdout(monkeypatch) -> None:
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = asyncio.Event()
    release_second = asyncio.Event()
    events: list[object] = []

    class PatchStdout:
        def __enter__(self) -> None:
            events.append("patch-enter")

        def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
            events.append("patch-exit")

    class DraftInput(TerminalInput):
        def _should_use_framed_prompt(self) -> bool:
            return True

        async def _read_framed_async(
            self,
            footer: str | None,
            default: str = "",
            *,
            mark_submitted: bool = True,
            draft_state=None,
            show_waiting: bool = False,
            transient: bool = False,
            queued_messages=(),
        ) -> str:
            events.append(
                (
                    footer,
                    default,
                    mark_submitted,
                    show_waiting,
                    transient,
                    draft_state is not None,
                    tuple(queued_messages),
                )
            )
            if len([event for event in events if isinstance(event, tuple)]) == 1:
                first_started.set()
                await release_first.wait()
                if draft_state is not None:
                    draft_state.submitted = True
                return "next prompt"
            second_started.set()
            await release_second.wait()
            return ""

    monkeypatch.setattr(input_module, "_patched_stdout_context", lambda console=None: PatchStdout())
    stream = StringIO()
    terminal_input = DraftInput(
        Console(file=stream, force_terminal=True, color_system=None, width=80)
    )

    async with terminal_input.live_draft_area("footer text"):
        await asyncio.wait_for(first_started.wait(), timeout=1)
        release_first.set()
        await asyncio.wait_for(second_started.wait(), timeout=1)
        release_second.set()

    assert events == [
        "patch-enter",
        ("footer text", "", False, True, True, True, ()),
        ("footer text", "", False, True, True, True, ("next prompt",)),
        "patch-exit",
    ]
    assert terminal_input._queued_submission == "next prompt"
    assert await terminal_input.read_async(footer="footer text") == "next prompt"
    output = stream.getvalue()
    assert output.count("> next prompt") == 1
    assert "\u2500" * 80 not in output
    assert output.endswith("\n\n")


@pytest.mark.asyncio
async def test_terminal_input_prefills_retained_draft_in_framed_prompt() -> None:
    defaults: list[str] = []

    class DraftInput(TerminalInput):
        def _should_use_framed_prompt(self) -> bool:
            return True

        async def _read_framed_async(
            self,
            footer: str | None,
            default: str = "",
            *,
            mark_submitted: bool = True,
            draft_state=None,
            show_waiting: bool = False,
            transient: bool = False,
        ) -> str:
            defaults.append(default)
            return "submitted"

    terminal_input = DraftInput(
        Console(file=StringIO(), force_terminal=True, color_system=None, width=80)
    )
    terminal_input._draft = "partial prompt"

    assert await terminal_input.read_async(footer="footer text") == "submitted"
    assert defaults == ["partial prompt"]
    assert terminal_input._draft == ""


@pytest.mark.asyncio
async def test_live_waiting_prompt_renders_queued_message_row(monkeypatch: object) -> None:
    captured: dict[str, object] = {}

    class Application:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def run_async(self) -> str:
            return ""

    class TerminalOnlyInput(TerminalInput):
        def _should_use_framed_prompt(self) -> bool:
            return True

    monkeypatch.setattr("prompt_toolkit.application.Application", Application)

    await TerminalOnlyInput(
        Console(file=StringIO(), force_terminal=True, color_system=None, width=32)
    )._read_framed_async(
        "footer",
        mark_submitted=False,
        show_waiting=True,
        transient=True,
        queued_messages=("queued prompt",),
    )

    root = getattr(captured["layout"], "container")
    queued_window = root.children[1]
    fragments = queued_window.content.text()

    assert fragment_list_to_text(fragments).startswith("  > queued prompt")
    assert fragments[0][0] == "class:queued"

@pytest.mark.asyncio
async def test_framed_prompt_renders_footer_below_input_border(monkeypatch: object) -> None:
    captured: dict[str, object] = {}

    class Application:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def run_async(self) -> str:
            return ""

    class TerminalOnlyInput(TerminalInput):
        def _should_use_framed_prompt(self) -> bool:
            return True

    monkeypatch.setattr("prompt_toolkit.application.Application", Application)

    await TerminalOnlyInput(
        Console(file=StringIO(), force_terminal=True, color_system=None, width=48)
    )._read_framed_async("model: test-model  workspace: E:\\repo")

    root = getattr(captured["layout"], "container")
    footer_window = root.children[-1]
    fragments = footer_window.content.text()

    assert fragment_list_to_text(fragments) == "model: test-model  workspace: E:\\repo"
    assert fragments[0][0] == "class:footer"


@pytest.mark.parametrize(
    ("label", "style"),
    [
        ("Ask", "class:permission.ask"),
        ("Read-only", "class:permission.read_only"),
        ("Bypass", "class:permission.bypass"),
    ],
)
def test_footer_text_factory_styles_permission_mode_prefix(label: str, style: str) -> None:
    footer = f"{label} model: test-model  workspace: E:\\repo"
    permission_width = len("Read-only")
    permission_padding = " " * (permission_width - len(label))
    padded_suffix = (
        f"{permission_padding} model: test-model  workspace: E:\\repo"
    )

    fragments = input_module._footer_text_factory(footer, 80)()

    assert fragment_list_to_text(fragments) == f"{label}{padded_suffix}"
    assert fragment_list_to_text(fragments).index("model:") == permission_width + 1
    assert fragments[0] == (style, label)
    assert fragments[1] == ("class:footer", padded_suffix)


@pytest.mark.asyncio
async def test_live_waiting_prompt_is_transient_and_animated(monkeypatch: object) -> None:
    captured: dict[str, object] = {}

    class Application:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def run_async(self) -> str:
            return ""

    class TerminalOnlyInput(TerminalInput):
        def _should_use_framed_prompt(self) -> bool:
            return True

    monkeypatch.setattr("prompt_toolkit.application.Application", Application)

    await TerminalOnlyInput(
        Console(file=StringIO(), force_terminal=True, color_system=None, width=24)
    )._read_framed_async(
        "footer",
        mark_submitted=False,
        show_waiting=True,
        transient=True,
    )

    assert captured.get("erase_when_done") is True
    assert captured.get("refresh_interval") == pytest.approx(0.12)

    layout = captured.get("layout")
    root = getattr(layout, "container")
    waiting_window = root.children[0]
    first = fragment_list_to_text(waiting_window.content.text())
    second = fragment_list_to_text(waiting_window.content.text())

    assert first.endswith(" Waiting for response...")
    assert second.endswith(" Waiting for response...")
    assert first != second

@pytest.mark.asyncio
async def test_live_draft_area_preserves_draft_text() -> None:
    terminal_input = TerminalInput(Console(file=StringIO(), force_terminal=False))

    async with terminal_input.live_draft_area("footer"):
        terminal_input._draft = "typed while waiting"

    assert await terminal_input.read_async() == "typed while waiting"


@pytest.mark.asyncio
async def test_terminal_input_uses_framed_application_with_shift_enter_newline(
    monkeypatch: object,
) -> None:
    captured: dict[str, object] = {}

    class Application:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def run_async(self) -> str:
            return "hello"

    class TerminalOnlyInput(TerminalInput):
        def _should_use_framed_prompt(self) -> bool:
            return True

    monkeypatch.setattr("prompt_toolkit.application.Application", Application)

    value = await TerminalOnlyInput(
        Console(file=StringIO(), force_terminal=True, color_system=None, width=24)
    ).read_async()

    bindings = captured.get("key_bindings")
    assert value == "hello"
    assert captured.get("full_screen") is False
    assert captured.get("erase_when_done") is True
    assert captured.get("cursor") is not None
    assert bindings is not None
    key_sequences = {
        tuple(getattr(key, "value", str(key)) for key in binding.keys)
        for binding in getattr(bindings, "bindings", ())
    }
    assert ("c-m",) in key_sequences
    assert (input_module._SHIFT_ENTER_KEY,) in key_sequences
    assert ("c-j",) in key_sequences

    layout = captured.get("layout")
    root = getattr(layout, "container")
    assert len(root.children) == 3
    top, input_window, bottom = root.children
    assert fragment_list_to_text(top.content.text()) == "\u2500" * 24
    assert fragment_list_to_text(bottom.content.text()) == "\u2500" * 24
    assert input_window.content.buffer.multiline() is True
    processors = input_window.content.input_processors
    assert [type(processor).__name__ for processor in processors] == [
        "BeforeInput",
        "_PromptPlaceholder",
    ]
    assert processors[1].placeholder == "Type here"
    assert input_window.height.min == 1
    assert input_window.height.max == 6
    assert input_window.height.preferred_specified is False
    assert input_window.dont_extend_height() is True
    assert input_window.wrap_lines() is True

    empty_height = input_window.preferred_height(width=24, max_available_height=20)
    assert empty_height.preferred == 1
    assert empty_height.max == 1

    input_window.content.buffer.text = "first\nsecond\nthird"
    multiline_height = input_window.preferred_height(width=24, max_available_height=20)
    assert multiline_height.preferred == 3
    assert multiline_height.max == 3


@pytest.mark.parametrize(
    ("key", "source"),
    [("escape", "escape"), ("c-c", "ctrl_c")],
)
def test_interrupt_bindings_exit_with_source(key: str, source: str) -> None:
    bindings = input_module._submission_key_bindings()
    binding = next(
        binding
        for binding in bindings.bindings
        if tuple(binding.keys) == (key,)
    )

    class App:
        def __init__(self) -> None:
            self.exception: BaseException | None = None

        def exit(self, *, exception=None, result=None):  # type: ignore[no-untyped-def]
            del result
            self.exception = exception

    app = App()
    event = SimpleNamespace(app=app)
    binding.handler(event)

    assert binding.eager() is True
    assert isinstance(app.exception, input_module.InputInterrupt)
    assert app.exception.source == source



def test_shift_tab_binding_cycles_permission_mode_and_invalidates() -> None:
    events: list[str] = []
    bindings = input_module._submission_key_bindings(
        on_permission_mode_toggle=lambda: events.append("cycle")
    )
    binding = next(
        binding
        for binding in bindings.bindings
        if tuple(binding.keys) == ("s-tab",)
    )

    class App:
        def invalidate(self) -> None:
            events.append("invalidate")

    binding.handler(SimpleNamespace(app=App()))

    assert binding.eager() is True
    assert events == ["cycle", "invalidate"]


@pytest.mark.parametrize("key", [input_module._SHIFT_ENTER_KEY, "c-j"])
def test_newline_bindings_insert_newline_without_submitting(key: str) -> None:
    bindings = input_module._submission_key_bindings()
    newline_binding = next(
        binding
        for binding in bindings.bindings
        if tuple(binding.keys) == (key,)
    )

    class Buffer:
        def __init__(self) -> None:
            self.text = ""
            self.submitted = False

        def insert_text(self, text: str) -> None:
            self.text += text

        def validate_and_handle(self) -> None:
            self.submitted = True

    buffer = Buffer()
    event = SimpleNamespace(app=SimpleNamespace(current_buffer=buffer))
    newline_binding.handler(event)

    assert newline_binding.eager() is True
    assert buffer.text == "\n"
    assert buffer.submitted is False


def test_enter_binding_ignores_empty_buffer() -> None:
    bindings = input_module._submission_key_bindings()
    enter_binding = next(
        binding
        for binding in bindings.bindings
        if tuple(binding.keys) == ("c-m",)
    )

    class Buffer:
        def __init__(self) -> None:
            self.text = "  \n "
            self.submitted = False

        def validate_and_handle(self) -> None:
            self.submitted = True

    buffer = Buffer()
    event = SimpleNamespace(app=SimpleNamespace(current_buffer=buffer))
    enter_binding.handler(event)

    assert buffer.submitted is False


@pytest.mark.asyncio
async def test_live_draft_area_keeps_prompt_open_after_queueing_message(
    monkeypatch: object,
) -> None:
    second_started = asyncio.Event()
    release_second = asyncio.Event()
    calls: list[tuple[str, tuple[str, ...]]] = []

    class PatchStdout:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
            return None

    class DraftInput(TerminalInput):
        def _should_use_framed_prompt(self) -> bool:
            return True

        async def _read_framed_async(
            self,
            footer: str | None,
            default: str = "",
            *,
            mark_submitted: bool = True,
            draft_state=None,
            show_waiting: bool = False,
            transient: bool = False,
            queued_messages=(),
        ) -> str:
            del footer, mark_submitted, show_waiting, transient
            calls.append((default, tuple(queued_messages)))
            if len(calls) == 1:
                draft_state.submitted = True
                return "queued prompt"
            second_started.set()
            await release_second.wait()
            return ""

    monkeypatch.setattr(input_module, "_patched_stdout_context", lambda console=None: PatchStdout())
    terminal_input = DraftInput(
        Console(file=StringIO(), force_terminal=True, color_system=None, width=80)
    )

    async with terminal_input.live_draft_area("footer text"):
        await asyncio.wait_for(second_started.wait(), timeout=1)
        assert terminal_input._queued_submission == "queued prompt"
        assert calls == [("", ()), ("", ("queued prompt",))]
        release_second.set()

    assert await terminal_input.read_async(footer="footer text") == "queued prompt"

def test_ctrl_j_binding_wins_over_prompt_toolkit_default() -> None:
    from prompt_toolkit.key_binding.defaults import load_key_bindings
    from prompt_toolkit.key_binding.key_bindings import merge_key_bindings
    from prompt_toolkit.keys import Keys

    merged_bindings = merge_key_bindings([
        load_key_bindings(),
        input_module._submission_key_bindings(),
    ])
    matches = [
        binding
        for binding in merged_bindings.get_bindings_for_keys((Keys.ControlJ,))
        if binding.filter()
    ]
    eager_matches = [binding for binding in matches if binding.eager()]

    class Buffer:
        def __init__(self) -> None:
            self.text = ""
            self.submitted = False

        def insert_text(self, text: str) -> None:
            self.text += text

        def validate_and_handle(self) -> None:
            self.submitted = True

    buffer = Buffer()
    event = SimpleNamespace(app=SimpleNamespace(current_buffer=buffer))
    eager_matches[-1].handler(event)

    assert buffer.text == "\n"
    assert buffer.submitted is False


@pytest.mark.asyncio
@pytest.mark.parametrize("key_input", ["\n", *input_module._SHIFT_ENTER_SEQUENCES])
async def test_real_prompt_toolkit_newline_keys_do_not_submit(key_input: str) -> None:
    from prompt_toolkit.application import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import BufferControl
    from prompt_toolkit.output import DummyOutput

    input_module._install_shift_enter_support()

    with create_pipe_input() as pipe_input:
        app_holder: dict[str, object] = {}

        def accept(buffer: Buffer) -> bool:
            app_holder["app"].exit(result=buffer.text)  # type: ignore[attr-defined]
            return True

        buffer = Buffer(multiline=True, accept_handler=accept)
        input_control = BufferControl(buffer=buffer)
        app = Application(
            layout=Layout(Window(input_control), focused_element=input_control),
            key_bindings=input_module._submission_key_bindings(),
            input=pipe_input,
            output=DummyOutput(),
            full_screen=False,
        )
        app_holder["app"] = app
        task = asyncio.create_task(app.run_async())

        await asyncio.sleep(0.05)
        pipe_input.send_text(key_input)
        await asyncio.sleep(0.05)

        assert task.done() is False
        assert buffer.text == "\n"

        pipe_input.send_text("body\r")
        assert await asyncio.wait_for(task, timeout=2) == "\nbody"

def test_shift_enter_support_maps_terminal_sequences_to_dedicated_key() -> None:
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.input.vt100_parser import Vt100Parser

    input_module._install_vt100_shift_enter_support()

    for sequence in input_module._SHIFT_ENTER_SEQUENCES:
        assert ANSI_SEQUENCES[sequence] == input_module._SHIFT_ENTER_KEY
        key_presses = []
        parser = Vt100Parser(key_presses.append)
        parser.feed_and_flush(sequence)

        assert len(key_presses) == 1
        assert key_presses[0].key == input_module._SHIFT_ENTER_KEY


def test_shift_enter_support_maps_win32_vt100_events_to_dedicated_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from prompt_toolkit.input import win32 as win32_input
    except (AssertionError, ImportError):
        pytest.skip("Win32 input is only available on Windows")

    reader_cls = win32_input.Vt100ConsoleInputReader
    monkeypatch.setattr(
        reader_cls,
        "_litecoder_shift_enter_patch",
        False,
        raising=False,
    )

    input_module._install_win32_vt100_shift_enter_support()

    key_event = SimpleNamespace(
        KeyDown=True,
        ControlKeyState=win32_input.ConsoleInputReader.SHIFT_PRESSED,
        uChar=SimpleNamespace(UnicodeChar="\r"),
    )
    record = SimpleNamespace(
        EventType=1,
        Event=SimpleNamespace(KeyEvent=key_event),
    )
    keys = list(
        reader_cls._get_keys(
            object(),
            SimpleNamespace(value=1),
            [record],
        )
    )

    assert keys == [input_module._SHIFT_ENTER_KEY]


def test_shift_enter_support_maps_win32_events_to_dedicated_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from prompt_toolkit.input import win32 as win32_input
    except (AssertionError, ImportError):
        pytest.skip("Win32 input is only available on Windows")

    from prompt_toolkit.key_binding.key_processor import KeyPress
    from prompt_toolkit.keys import Keys

    reader_cls = win32_input.ConsoleInputReader
    monkeypatch.setattr(
        reader_cls,
        "_litecoder_shift_enter_patch",
        False,
        raising=False,
    )

    def original(self, ev):  # type: ignore[no-untyped-def]
        return [KeyPress(Keys.ControlM, "\r")]

    monkeypatch.setattr(reader_cls, "_event_to_key_presses", original)

    input_module._install_win32_shift_enter_support()

    class Event:
        ControlKeyState = reader_cls.SHIFT_PRESSED

    reader = object.__new__(reader_cls)
    key_presses = reader_cls._event_to_key_presses(reader, Event())

    assert len(key_presses) == 1
    assert key_presses[0].key == input_module._SHIFT_ENTER_KEY


@pytest.mark.asyncio
async def test_terminal_prompt_renders_submitted_text_once(monkeypatch: object) -> None:
    class Application:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def run_async(self) -> str:
            return "hello"

    class TerminalOnlyInput(TerminalInput):
        def _should_use_framed_prompt(self) -> bool:
            return True

    monkeypatch.setattr("prompt_toolkit.application.Application", Application)
    stream = StringIO()

    value = await TerminalOnlyInput(
        Console(file=stream, force_terminal=True, color_system=None, width=20)
    ).read_async()

    assert value == "hello"
    output = stream.getvalue()
    assert output.count("> hello") == 1
    assert "\u2500" * 20 not in output
    assert output.endswith("\n\n")


def test_todo_progress_waiting_text_advances_to_the_next_item() -> None:
    console = Console(file=StringIO(), force_terminal=True, color_system=None)
    waiting_text = input_module._waiting_text_factory(console)
    initial = [
        {
            "content": "分析配置",
            "active_form": "正在分析配置",
            "status": "in_progress",
        },
        {
            "content": "分析日志",
            "active_form": "正在分析日志",
            "status": "pending",
        },
        {
            "content": "分析验证码",
            "active_form": "正在分析验证码",
            "status": "pending",
        },
    ]

    assert input_module.replace_todo_progress(console, initial) is True
    first = fragment_list_to_text(waiting_text())
    assert "正在分析配置..." in first
    assert "□ 分析配置" in first
    assert "✓ 分析配置" not in first
    assert input_module._waiting_height_factory(console)() == 4

    updated = [
        {**initial[0], "status": "completed"},
        {**initial[1], "status": "in_progress"},
        initial[2],
    ]
    assert input_module.replace_todo_progress(console, updated) is True
    second = fragment_list_to_text(waiting_text())
    assert "正在分析日志..." in second
    assert "✓ 分析配置" in second
    assert "□ 分析日志" in second

    completed = [{**item, "status": "completed"} for item in initial]
    assert input_module.replace_todo_progress(console, completed) is True
    final = fragment_list_to_text(waiting_text())
    assert "Finishing response..." in final
    assert final.count("✓") == 3


def test_todo_progress_collapses_long_lists_around_the_active_item() -> None:
    console = Console(file=StringIO(), force_terminal=True, color_system=None)
    initial = [
        {
            "content": f"步骤 {index + 1}",
            "active_form": f"正在执行步骤 {index + 1}",
            "status": "in_progress" if index == 0 else "pending",
        }
        for index in range(10)
    ]

    assert input_module.replace_todo_progress(console, initial) is True
    initial_lines = [
        text for _, text in input_module._todo_progress_lines(
            input_module.todo_progress_items(console)
        )
    ]
    assert "□ 步骤 1" in initial_lines
    assert "… +4 pending" in initial_lines

    advanced = [
        {
            **item,
            "status": (
                "completed"
                if index < 7
                else "in_progress"
                if index == 7
                else "pending"
            ),
        }
        for index, item in enumerate(initial)
    ]
    assert input_module.replace_todo_progress(console, advanced) is True
    advanced_lines = [
        text for _, text in input_module._todo_progress_lines(
            input_module.todo_progress_items(console)
        )
    ]
    assert "… +4 completed" in advanced_lines
    assert "□ 步骤 8" in advanced_lines
    assert "□ 步骤 9" in advanced_lines
    assert "□ 步骤 10" in advanced_lines
    assert "✓ 步骤 1" not in advanced_lines
