from __future__ import annotations

import asyncio
import sys

import pytest
from typer.testing import CliRunner

from litecoder.cli.app import app


def test_run_and_resume_commands_are_registered() -> None:
    runner = CliRunner()
    run_result = runner.invoke(app, ["run", "hello"])
    resume_result = runner.invoke(app, ["resume", "session-1", "--prompt", "hello"])

    assert run_result.exit_code != 2
    assert resume_result.exit_code != 2
    assert "No such command" not in run_result.output + resume_result.output


def test_terminal_renderer_renders_non_completed_result_event() -> None:
    from io import StringIO

    from rich.console import Console

    from litecoder.ui.events import RuntimeUIEvent, UIEventType
    from litecoder.ui.renderers.terminal import TerminalRenderer

    stream = StringIO()
    renderer = TerminalRenderer(
        Console(file=stream, force_terminal=True, color_system=None, width=80)
    )
    renderer.emit(
        RuntimeUIEvent(
            UIEventType.TURN_FINISHED,
            sequence=1,
            timestamp=1.0,
            payload={
                "status": "incomplete",
                "reason": "provider_transient retry budget exhausted",
                "elapsed_seconds": 5.0,
                "total_tokens": 0,
            },
        )
    )

    output = stream.getvalue()
    assert "Incomplete provider_transient retry budget exhausted" in output
    assert "Elapsed 5.0s" in output
    assert "tokens=0" not in output


async def test_run_once_renders_result_event_through_terminal_renderer(
    monkeypatch: object,
) -> None:
    from io import StringIO

    from rich.console import Console

    from litecoder.agent.result import AgentResult
    import litecoder.cli.app as cli_app
    from litecoder.providers.models import Usage
    from litecoder.ui.events import UIEventFactory, UIEventType
    from litecoder.ui.renderers.terminal import TerminalRenderer

    stream = StringIO()
    renderer = TerminalRenderer(
        Console(file=stream, force_terminal=True, color_system=None, width=80)
    )

    class Runtime:
        def __init__(self, ui_sink) -> None:
            self.ui_sink = ui_sink
            self.closed = False

        async def run(self, prompt: str) -> AgentResult:
            assert prompt == "hello"
            factory = UIEventFactory(session_id="session-1")
            self.ui_sink.emit(
                factory.next(
                    UIEventType.TURN_FINISHED,
                    payload={
                        "status": "completed",
                        "reason": "done",
                        "total_tokens": 3,
                    },
                )
            )
            return AgentResult("session-1", "completed", "done", Usage(1, 2))

        async def close(self) -> None:
            self.closed = True

    runtime = Runtime(None)
    captured = {}

    async def fake_build_runtime(*args, ui_sink=None, **kwargs):
        captured["ui_sink"] = ui_sink
        runtime.ui_sink = ui_sink
        return runtime

    monkeypatch.setattr(cli_app, "TerminalRenderer", lambda console=None: renderer)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli_app, "build_runtime", fake_build_runtime)  # type: ignore[attr-defined]

    await cli_app._run_once("hello")

    assert captured["ui_sink"].renderer is renderer
    assert runtime.closed is True
    output = stream.getvalue()
    assert "Elapsed --" in output
    assert "completed" not in output
    assert "tokens=3" not in output


async def test_run_once_does_not_duplicate_runtime_turn_finished_event(
    monkeypatch: object,
) -> None:
    from io import StringIO

    from rich.console import Console

    from litecoder.agent.result import AgentResult
    import litecoder.cli.app as cli_app
    from litecoder.providers.models import Usage
    from litecoder.ui.events import UIEventFactory, UIEventType
    from litecoder.ui.renderers.terminal import TerminalRenderer

    stream = StringIO()
    renderer = TerminalRenderer(
        Console(file=stream, force_terminal=True, color_system=None, width=80)
    )

    class Runtime:
        def __init__(self, ui_sink) -> None:
            self.ui_sink = ui_sink

        async def run(self, prompt: str) -> AgentResult:
            factory = UIEventFactory(session_id="session-1")
            self.ui_sink.emit(
                factory.next(
                    UIEventType.TURN_FINISHED,
                    payload={
                        "status": "completed",
                        "reason": "done",
                        "total_tokens": 3,
                    },
                )
            )
            return AgentResult("session-1", "completed", "done", Usage(1, 2))

        async def close(self) -> None:
            return None

    async def fake_build_runtime(*args, ui_sink=None, **kwargs):
        return Runtime(ui_sink)

    monkeypatch.setattr(cli_app, "TerminalRenderer", lambda console=None: renderer)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli_app, "build_runtime", fake_build_runtime)  # type: ignore[attr-defined]

    await cli_app._run_once("hello")

    output = stream.getvalue()
    assert output.count("Elapsed --") == 1
    assert "tokens=3" not in output


async def test_interactive_renders_startup_banner_without_diagnostic_guidance() -> None:
    from contextlib import asynccontextmanager
    from io import StringIO
    from types import SimpleNamespace

    from rich.console import Console

    from litecoder.cli.interactive import interactive_session
    from litecoder.ui.events import UIEventType
    from litecoder.ui.renderers.terminal import TerminalRenderer
    from litecoder.ui.sink import RecordingUISink

    class Runtime:
        model = "test-model"
        paths = SimpleNamespace(workspace_root="E:\\Codex_workspace\\litecoder")

    class ScriptedInput:
        async def read_async(self, *, footer=None):  # type: ignore[no-untyped-def]
            return "/exit"

        def render_submitted(self, value):  # type: ignore[no-untyped-def]
            return None

        @asynccontextmanager
        async def live_draft_area(self, footer=None):  # type: ignore[no-untyped-def]
            yield

    sink = RecordingUISink()
    stream = StringIO()
    renderer = TerminalRenderer(
        Console(file=stream, force_terminal=True, color_system=None, width=100)
    )

    await interactive_session(
        Runtime(),  # type: ignore[arg-type]
        terminal_input=ScriptedInput(),  # type: ignore[arg-type]
        renderer=renderer,
        ui_sink=sink,
    )

    assert [
        event for event in sink.events if event.type is UIEventType.DIAGNOSTIC
    ] == []
    output = stream.getvalue()
    assert "Welcome to LiteCoder CLI!" in output
    assert "Workspace: E:\\Codex_workspace\\litecoder" in output
    assert "Using test-model (from .litecoder\\config.toml)" in output


async def test_interactive_reuses_session_after_first_prompt() -> None:
    from contextlib import asynccontextmanager

    from litecoder.agent.result import AgentResult
    from litecoder.cli.interactive import interactive_session
    from litecoder.providers.models import Usage
    from litecoder.ui.renderers.terminal import TerminalRenderer
    from litecoder.ui.sink import RecordingUISink

    class Runtime:
        def __init__(self) -> None:
            self.run_prompts: list[str] = []
            self.resume_prompts: list[tuple[str, str | None]] = []

        async def run(self, prompt: str) -> AgentResult:
            self.run_prompts.append(prompt)
            return AgentResult("session-1", "completed", "done", Usage(0, 0))

        async def resume(
            self, session_id: str, prompt: str | None = None
        ) -> AgentResult:
            self.resume_prompts.append((session_id, prompt))
            return AgentResult(session_id, "completed", "done", Usage(0, 0))

    class ScriptedInput:
        def __init__(self) -> None:
            self.inputs = iter(["first", "second", "/exit"])

        async def read_async(self, *, footer=None):  # type: ignore[no-untyped-def]
            return next(self.inputs)

        def render_submitted(self, value):  # type: ignore[no-untyped-def]
            return None

        @asynccontextmanager
        async def live_draft_area(self, footer=None):  # type: ignore[no-untyped-def]
            yield

    runtime = Runtime()

    await interactive_session(
        runtime,
        terminal_input=ScriptedInput(),  # type: ignore[arg-type]
        renderer=TerminalRenderer(),
        ui_sink=RecordingUISink(),
    )

    assert runtime.run_prompts == ["first"]
    assert runtime.resume_prompts == [("session-1", "second")]


async def test_interactive_clear_starts_next_prompt_in_new_session() -> None:
    from contextlib import asynccontextmanager
    from io import StringIO

    from rich.console import Console

    from litecoder.agent.result import AgentResult
    from litecoder.cli.interactive import interactive_session
    from litecoder.providers.models import Usage
    from litecoder.ui.renderers.terminal import TerminalRenderer
    from litecoder.ui.sink import RecordingUISink

    class Runtime:
        model = "model-a"

        def __init__(self) -> None:
            self.run_prompts: list[str] = []
            self.resume_prompts: list[tuple[str, str | None]] = []

        async def run(self, prompt: str) -> AgentResult:
            self.run_prompts.append(prompt)
            return AgentResult(
                f"session-{len(self.run_prompts)}",
                "completed",
                "done",
                Usage(0, 0),
            )

        async def resume(
            self, session_id: str, prompt: str | None = None
        ) -> AgentResult:
            self.resume_prompts.append((session_id, prompt))
            pytest.fail("resume not expected after /clear")

    class ScriptedInput:
        def __init__(self) -> None:
            self.inputs = iter(["first", "/clear", "second", "/exit"])

        async def read_async(self, *, footer=None):  # type: ignore[no-untyped-def]
            return next(self.inputs)

        def render_submitted(self, value):  # type: ignore[no-untyped-def]
            return None

        @asynccontextmanager
        async def live_draft_area(self, footer=None):  # type: ignore[no-untyped-def]
            yield

    class ClearCountingConsole(Console):
        def __init__(self) -> None:
            super().__init__(
                file=StringIO(),
                force_terminal=True,
                color_system=None,
            )
            self.clear_calls = 0

        def clear(self, home: bool = True) -> None:
            self.clear_calls += 1

    runtime = Runtime()
    console = ClearCountingConsole()

    await interactive_session(
        runtime,
        terminal_input=ScriptedInput(),  # type: ignore[arg-type]
        renderer=TerminalRenderer(console),
        ui_sink=RecordingUISink(),
    )

    assert runtime.run_prompts == ["first", "second"]
    assert runtime.resume_prompts == []
    assert console.clear_calls == 1


async def test_interactive_model_switch_resumes_the_derived_session() -> None:
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from litecoder.agent.result import AgentResult
    from litecoder.cli.interactive import interactive_session
    from litecoder.providers.models import Usage
    from litecoder.ui.renderers.terminal import TerminalRenderer
    from litecoder.ui.sink import RecordingUISink

    class Runtime:
        provider_name = "fake"
        model = "model-a"
        provider_models = {"fake": "model-a", "other": "model-b"}
        paths = SimpleNamespace(workspace_root="workspace")

        def __init__(self) -> None:
            self.run_prompts: list[str] = []
            self.resume_prompts: list[tuple[str, str | None]] = []
            self.switches: list[tuple[str, str, str | None]] = []

        async def run(self, prompt: str) -> AgentResult:
            self.run_prompts.append(prompt)
            return AgentResult("root-session", "completed", "done", Usage(0, 0))

        async def switch_provider(
            self,
            session_id: str,
            provider: str,
            model: str | None = None,
        ) -> AgentResult:
            self.switches.append((session_id, provider, model))
            self.provider_name = provider
            self.model = model or ""
            return AgentResult(
                "derived-session",
                "ready",
                "provider switched",
                Usage(0, 0),
            )

        async def resume(
            self, session_id: str, prompt: str | None = None
        ) -> AgentResult:
            self.resume_prompts.append((session_id, prompt))
            return AgentResult(session_id, "completed", "done", Usage(0, 0))

    class ScriptedInput:
        def __init__(self) -> None:
            self.inputs = iter(["first", "/model other", "second", "/exit"])
            self.footers: list[str] = []

        async def read_async(self, *, footer=None):  # type: ignore[no-untyped-def]
            self.footers.append(footer() if callable(footer) else footer or "")
            return next(self.inputs)

        def render_submitted(self, value):  # type: ignore[no-untyped-def]
            return None

        @asynccontextmanager
        async def live_draft_area(self, footer=None):  # type: ignore[no-untyped-def]
            yield

    runtime = Runtime()
    terminal_input = ScriptedInput()

    await interactive_session(
        runtime,
        terminal_input=terminal_input,  # type: ignore[arg-type]
        renderer=TerminalRenderer(),
        ui_sink=RecordingUISink(),
    )

    assert runtime.run_prompts == ["first"]
    assert runtime.switches == [
        ("root-session", "other", "model-b"),
    ]
    assert runtime.resume_prompts == [("derived-session", "second")]
    assert terminal_input.footers == [
        "Ask model: model-a  workspace: workspace",
        "Ask model: model-a  workspace: workspace",
        "Ask model: model-b  workspace: workspace",
        "Ask model: model-b  workspace: workspace",
    ]


async def test_resumed_interactive_uses_persisted_model_for_command_and_footer() -> None:
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from litecoder.cli.interactive import interactive_session
    from litecoder.context.session.models import SessionRecord, SessionStatus
    from litecoder.ui.events import UIEventType
    from litecoder.ui.renderers.terminal import TerminalRenderer
    from litecoder.ui.sink import RecordingUISink

    persisted = SessionRecord.new(
        "derived-session",
        "project",
        "workspace",
        "other",
        "model-b",
        workspace_path="workspace",
        status=SessionStatus.IDLE,
    )

    class Store:
        async def load_context(self, session_id: str) -> object:
            assert session_id == "derived-session"
            return SimpleNamespace(session=persisted, messages=[])

    class Runtime:
        provider_name = "fake"
        model = "model-a"
        provider_models = {"fake": "model-a", "other": "model-b"}
        paths = SimpleNamespace(workspace_root="workspace")
        store = Store()

    class ScriptedInput:
        def __init__(self) -> None:
            self.inputs = iter(["/model", "/clear", "/exit"])
            self.footers: list[str] = []

        async def read_async(self, *, footer=None):  # type: ignore[no-untyped-def]
            self.footers.append(footer() if callable(footer) else footer or "")
            return next(self.inputs)

        def render_submitted(self, value):  # type: ignore[no-untyped-def]
            return None

        @asynccontextmanager
        async def live_draft_area(self, footer=None):  # type: ignore[no-untyped-def]
            yield

    terminal_input = ScriptedInput()
    sink = RecordingUISink()

    await interactive_session(
        Runtime(),  # type: ignore[arg-type]
        terminal_input=terminal_input,  # type: ignore[arg-type]
        renderer=TerminalRenderer(),
        ui_sink=sink,
        session_id="derived-session",
    )

    diagnostic = next(
        event for event in sink.events if event.type is UIEventType.DIAGNOSTIC
    )
    assert diagnostic.payload["message"].startswith("Current: other model-b\n")
    assert terminal_input.footers == [
        "Ask model: model-b  workspace: workspace",
        "Ask model: model-b  workspace: workspace",
        "Ask model: model-a  workspace: workspace",
    ]


async def test_interactive_keeps_input_area_live_while_agent_runs() -> None:
    from contextlib import asynccontextmanager

    import pytest

    from litecoder.agent.result import AgentResult
    from litecoder.cli.interactive import interactive_session
    from litecoder.providers.models import Usage
    from litecoder.ui.renderers.terminal import TerminalRenderer
    from litecoder.ui.sink import RecordingUISink

    class Runtime:
        model = "test-model"

        async def run(self, prompt: str) -> AgentResult:
            assert terminal_input.input_area_live is True
            return AgentResult("session-1", "completed", "done", Usage(0, 0))

        async def resume(
            self, session_id: str, prompt: str | None = None
        ) -> AgentResult:
            pytest.fail("resume not expected")

    class ScriptedInput:
        def __init__(self) -> None:
            self.inputs = iter(["hello", "/exit"])
            self.input_area_live = False
            self.live_footers: list[str] = []

        async def read_async(self, *, footer=None):  # type: ignore[no-untyped-def]
            return next(self.inputs)

        def render_submitted(self, value):  # type: ignore[no-untyped-def]
            return None

        @asynccontextmanager
        async def live_draft_area(self, footer=None):  # type: ignore[no-untyped-def]
            self.live_footers.append(footer() if callable(footer) else footer or "")
            self.input_area_live = True
            try:
                yield
            finally:
                self.input_area_live = False

    terminal_input = ScriptedInput()

    await interactive_session(
        Runtime(),
        terminal_input=terminal_input,  # type: ignore[arg-type]
        renderer=TerminalRenderer(),
        ui_sink=RecordingUISink(),
    )

    assert len(terminal_input.live_footers) == 1
    assert "Ask model: test-model" in terminal_input.live_footers[0]
    assert "workspace:" in terminal_input.live_footers[0]
    assert "context:" not in terminal_input.live_footers[0]


async def test_interactive_cancels_running_turn_on_live_escape_interrupt() -> None:
    from contextlib import asynccontextmanager

    from litecoder.cli.interactive import interactive_session
    from litecoder.ui.input import LiveInputInterrupt
    from litecoder.ui.renderers.terminal import TerminalRenderer
    from litecoder.ui.sink import RecordingUISink

    started = asyncio.Event()

    class Runtime:
        def __init__(self) -> None:
            self.cancelled = False

        async def run(self, prompt: str):  # type: ignore[no-untyped-def]
            assert prompt == "work"
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        async def resume(self, session_id: str, prompt: str | None = None):  # type: ignore[no-untyped-def]
            pytest.fail("resume not expected")

    class ScriptedInput:
        def __init__(self) -> None:
            self.inputs = iter(["work", "/exit"])

        async def read_async(self, *, footer=None):  # type: ignore[no-untyped-def]
            return next(self.inputs)

        def render_submitted(self, value):  # type: ignore[no-untyped-def]
            return None

        @asynccontextmanager
        async def live_draft_area(self, footer=None):  # type: ignore[no-untyped-def]
            interrupt = LiveInputInterrupt()

            async def trigger() -> None:
                await started.wait()
                interrupt.request("escape")

            trigger_task = asyncio.create_task(trigger())
            try:
                yield interrupt
            finally:
                trigger_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await trigger_task

    import contextlib

    runtime = Runtime()

    await interactive_session(
        runtime,  # type: ignore[arg-type]
        terminal_input=ScriptedInput(),  # type: ignore[arg-type]
        renderer=TerminalRenderer(),
        ui_sink=RecordingUISink(),
    )

    assert runtime.cancelled is True


async def test_interactive_exits_after_two_consecutive_ctrl_c_interrupts() -> None:
    from contextlib import asynccontextmanager

    from litecoder.cli.interactive import interactive_session
    from litecoder.ui.input import InputInterrupt
    from litecoder.ui.renderers.terminal import TerminalRenderer
    from litecoder.ui.sink import RecordingUISink

    class Runtime:
        async def run(self, prompt: str):  # type: ignore[no-untyped-def]
            pytest.fail("run not expected")

        async def resume(self, session_id: str, prompt: str | None = None):  # type: ignore[no-untyped-def]
            pytest.fail("resume not expected")

    class ScriptedInput:
        def __init__(self) -> None:
            self.reads = 0

        async def read_async(self, *, footer=None):  # type: ignore[no-untyped-def]
            self.reads += 1
            raise InputInterrupt("ctrl_c")

        def render_submitted(self, value):  # type: ignore[no-untyped-def]
            return None

        @asynccontextmanager
        async def live_draft_area(self, footer=None):  # type: ignore[no-untyped-def]
            yield

    terminal_input = ScriptedInput()

    await interactive_session(
        Runtime(),  # type: ignore[arg-type]
        terminal_input=terminal_input,  # type: ignore[arg-type]
        renderer=TerminalRenderer(),
        ui_sink=RecordingUISink(),
    )

    assert terminal_input.reads == 2


async def test_interactive_single_ctrl_c_interrupt_keeps_session_open() -> None:
    from contextlib import asynccontextmanager

    from litecoder.agent.result import AgentResult
    from litecoder.cli.interactive import interactive_session
    from litecoder.providers.models import Usage
    from litecoder.ui.input import InputInterrupt
    from litecoder.ui.renderers.terminal import TerminalRenderer
    from litecoder.ui.sink import RecordingUISink

    class Runtime:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def run(self, prompt: str) -> AgentResult:
            self.prompts.append(prompt)
            return AgentResult("session-1", "completed", "done", Usage(0, 0))

        async def resume(
            self, session_id: str, prompt: str | None = None
        ) -> AgentResult:
            pytest.fail("resume not expected")

    class ScriptedInput:
        def __init__(self) -> None:
            self.inputs = iter([InputInterrupt("ctrl_c"), "hello", "/exit"])

        async def read_async(self, *, footer=None):  # type: ignore[no-untyped-def]
            item = next(self.inputs)
            if isinstance(item, BaseException):
                raise item
            return item

        def render_submitted(self, value):  # type: ignore[no-untyped-def]
            return None

        @asynccontextmanager
        async def live_draft_area(self, footer=None):  # type: ignore[no-untyped-def]
            yield

    runtime = Runtime()

    await interactive_session(
        runtime,
        terminal_input=ScriptedInput(),  # type: ignore[arg-type]
        renderer=TerminalRenderer(),
        ui_sink=RecordingUISink(),
    )

    assert runtime.prompts == ["hello"]


async def test_interactive_escape_interrupt_does_not_count_as_ctrl_c() -> None:
    from contextlib import asynccontextmanager

    from litecoder.agent.result import AgentResult
    from litecoder.cli.interactive import interactive_session
    from litecoder.providers.models import Usage
    from litecoder.ui.input import InputInterrupt
    from litecoder.ui.renderers.terminal import TerminalRenderer
    from litecoder.ui.sink import RecordingUISink

    class Runtime:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def run(self, prompt: str) -> AgentResult:
            self.prompts.append(prompt)
            return AgentResult("session-1", "completed", "done", Usage(0, 0))

        async def resume(
            self, session_id: str, prompt: str | None = None
        ) -> AgentResult:
            pytest.fail("resume not expected")

    class ScriptedInput:
        def __init__(self) -> None:
            self.inputs = iter(
                [InputInterrupt("escape"), InputInterrupt("ctrl_c"), "hello", "/exit"]
            )

        async def read_async(self, *, footer=None):  # type: ignore[no-untyped-def]
            item = next(self.inputs)
            if isinstance(item, BaseException):
                raise item
            return item

        def render_submitted(self, value):  # type: ignore[no-untyped-def]
            return None

        @asynccontextmanager
        async def live_draft_area(self, footer=None):  # type: ignore[no-untyped-def]
            yield

    runtime = Runtime()

    await interactive_session(
        runtime,
        terminal_input=ScriptedInput(),  # type: ignore[arg-type]
        renderer=TerminalRenderer(),
        ui_sink=RecordingUISink(),
    )

    assert runtime.prompts == ["hello"]


async def test_interactive_passes_model_and_workspace_footer() -> None:
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from litecoder.agent.result import AgentResult
    from litecoder.cli.interactive import interactive_session
    from litecoder.providers.models import Usage
    from litecoder.ui.renderers.terminal import TerminalRenderer
    from litecoder.ui.sink import RecordingUISink

    class Runtime:
        model = "test-model"
        paths = SimpleNamespace(workspace_root="E:\\Codex_workspace\\litecoder")

        async def run(self, prompt: str) -> AgentResult:
            return AgentResult("session-1", "completed", "done", Usage(2, 3))

        async def resume(
            self, session_id: str, prompt: str | None = None
        ) -> AgentResult:
            pytest.fail("resume not expected")

    class ScriptedInput:
        def __init__(self) -> None:
            self.inputs = iter(["hello", "/exit"])
            self.footers: list[str] = []

        async def read_async(self, *, footer=None):  # type: ignore[no-untyped-def]
            self.footers.append(footer() if callable(footer) else footer or "")
            return next(self.inputs)

        def render_submitted(self, value):  # type: ignore[no-untyped-def]
            return None

        @asynccontextmanager
        async def live_draft_area(self, footer=None):  # type: ignore[no-untyped-def]
            yield

    terminal_input = ScriptedInput()

    await interactive_session(
        Runtime(),
        terminal_input=terminal_input,  # type: ignore[arg-type]
        renderer=TerminalRenderer(),
        ui_sink=RecordingUISink(),
    )

    assert (
        terminal_input.footers[0]
        == "Ask model: test-model  workspace: E:\\Codex_workspace\\litecoder"
    )
    assert (
        terminal_input.footers[1]
        == "Ask model: test-model  workspace: E:\\Codex_workspace\\litecoder"
    )


async def test_interactive_cycles_permission_mode_with_shift_tab_callback() -> None:
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from litecoder.agent.result import AgentResult
    from litecoder.cli.interactive import interactive_session
    from litecoder.providers.models import Usage
    from litecoder.ui.renderers.terminal import TerminalRenderer
    from litecoder.ui.sink import RecordingUISink

    class Runtime:
        model = "test-model"
        paths = SimpleNamespace(workspace_root="E:\\Codex_workspace\\litecoder")
        permission_mode = "ask"

        def __init__(self) -> None:
            self.prompts: list[tuple[str, str]] = []

        async def run(self, prompt: str) -> AgentResult:
            self.prompts.append((prompt, self.permission_mode))
            return AgentResult("session-1", "completed", "done", Usage(2, 3))

        async def resume(
            self, session_id: str, prompt: str | None = None
        ) -> AgentResult:
            pytest.fail("resume not expected")

    class ScriptedInput:
        def __init__(self) -> None:
            self.reads = 0
            self.footers: list[str] = []

        def footer_text(self, footer) -> str:  # type: ignore[no-untyped-def]
            return footer() if callable(footer) else footer or ""

        async def read_async(
            self,
            *,
            footer=None,
            on_permission_mode_toggle=None,
        ):  # type: ignore[no-untyped-def]
            self.reads += 1
            self.footers.append(self.footer_text(footer))
            assert on_permission_mode_toggle is not None
            on_permission_mode_toggle()
            self.footers.append(self.footer_text(footer))
            return "hello" if self.reads == 1 else "/exit"

        def render_submitted(self, value):  # type: ignore[no-untyped-def]
            return None

        @asynccontextmanager
        async def live_draft_area(self, footer=None):  # type: ignore[no-untyped-def]
            self.footers.append(self.footer_text(footer))
            yield

    runtime = Runtime()
    terminal_input = ScriptedInput()

    await interactive_session(
        runtime,
        terminal_input=terminal_input,  # type: ignore[arg-type]
        renderer=TerminalRenderer(),
        ui_sink=RecordingUISink(),
    )

    assert runtime.prompts == [("hello", "read-only")]
    assert terminal_input.footers == [
        "Ask model: test-model  workspace: E:\\Codex_workspace\\litecoder",
        "Read-only model: test-model  workspace: E:\\Codex_workspace\\litecoder",
        "Read-only model: test-model  workspace: E:\\Codex_workspace\\litecoder",
        "Read-only model: test-model  workspace: E:\\Codex_workspace\\litecoder",
        "Bypass model: test-model  workspace: E:\\Codex_workspace\\litecoder",
    ]
    assert runtime.permission_mode == "bypass"


async def test_interactive_emits_local_command_diagnostics_and_exit_summary() -> None:
    from contextlib import asynccontextmanager

    from litecoder.agent.result import AgentResult
    from litecoder.cli.interactive import interactive_session
    from litecoder.providers.models import Usage
    from litecoder.ui.events import UIEventType
    from litecoder.ui.renderers.terminal import TerminalRenderer
    from litecoder.ui.sink import RecordingUISink

    class Runtime:
        async def run(self, prompt: str) -> AgentResult:
            return AgentResult("session-1", "completed", "done", Usage(2, 3))

        async def resume(
            self, session_id: str, prompt: str | None = None
        ) -> AgentResult:
            pytest.fail("resume not expected")

    class ScriptedInput:
        def __init__(self) -> None:
            self.inputs = iter(["hello", "/help", "/exit"])

        async def read_async(self, *, footer=None):  # type: ignore[no-untyped-def]
            return next(self.inputs)

        def render_submitted(self, value):  # type: ignore[no-untyped-def]
            return None

        @asynccontextmanager
        async def live_draft_area(self, footer=None):  # type: ignore[no-untyped-def]
            yield

    sink = RecordingUISink()

    await interactive_session(
        Runtime(),
        terminal_input=ScriptedInput(),  # type: ignore[arg-type]
        renderer=TerminalRenderer(),
        ui_sink=sink,
    )

    diagnostics = [
        event.payload["message"]
        for event in sink.events
        if event.type is UIEventType.DIAGNOSTIC
    ]
    assert any("Local commands:" in message for message in diagnostics)
    assert diagnostics[-1] == "session=session-1"


async def test_interactive_entrypoint_runs_textual_app_and_closes_runtime(
    monkeypatch: object,
) -> None:
    import litecoder.cli.app as cli_app

    captured: dict[str, object] = {}

    class Runtime:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    runtime = Runtime()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    async def fake_build_runtime(
        *args,
        ui_sink=None,
        permission_prompt=None,
        **kwargs,
    ):
        assert ui_sink is not None
        assert permission_prompt is not None
        captured["sink"] = ui_sink
        captured["permission_prompt"] = permission_prompt
        return runtime

    class FakeTextualApp:
        def __init__(
            self,
            selected_runtime,
            *,
            sink,
            permission_prompt,
            session_id=None,
        ) -> None:
            assert selected_runtime is runtime
            assert sink is captured["sink"]
            assert permission_prompt is captured["permission_prompt"]
            assert session_id is None
            captured["app"] = self

        async def run_async(self, **kwargs: object) -> str:
            assert kwargs == {"inline": True, "inline_no_clear": True}
            assert sys.stdout is not original_stdout
            assert sys.stderr is not original_stderr
            print("hidden third-party stdout")
            print("hidden third-party stderr", file=sys.stderr)
            captured["run_kwargs"] = kwargs
            captured["ran"] = True
            return "session-1"

    class FakeConsole:
        def print(self, value, **kwargs) -> None:
            captured["summary"] = value
            captured["summary_style"] = kwargs.get("style")

    monkeypatch.setattr(cli_app, "build_runtime", fake_build_runtime)
    monkeypatch.setattr(cli_app, "LiteCoderApp", FakeTextualApp)
    monkeypatch.setattr(cli_app, "Console", FakeConsole)

    await cli_app._interactive()

    assert captured["ran"] is True
    assert runtime.closed is True
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert captured["summary"] == "session=session-1"
    assert captured["summary_style"] == "yellow"


async def test_interactive_ignores_blank_input() -> None:
    from contextlib import asynccontextmanager

    from litecoder.agent.result import AgentResult
    from litecoder.cli.interactive import interactive_session
    from litecoder.providers.models import Usage
    from litecoder.ui.renderers.terminal import TerminalRenderer
    from litecoder.ui.sink import RecordingUISink

    class Runtime:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def run(self, prompt: str) -> AgentResult:
            self.prompts.append(prompt)
            return AgentResult("session-1", "completed", "done", Usage(0, 0))

        async def resume(
            self, session_id: str, prompt: str | None = None
        ) -> AgentResult:
            pytest.fail("resume not expected")

    class ScriptedInput:
        def __init__(self) -> None:
            self.inputs = iter(["   ", "first", "/exit"])

        async def read_async(self, *, footer=None):  # type: ignore[no-untyped-def]
            return next(self.inputs)

        def render_submitted(self, value):  # type: ignore[no-untyped-def]
            return None

        @asynccontextmanager
        async def live_draft_area(self, footer=None):  # type: ignore[no-untyped-def]
            yield

    runtime = Runtime()

    await interactive_session(
        runtime,
        terminal_input=ScriptedInput(),  # type: ignore[arg-type]
        renderer=TerminalRenderer(),
        ui_sink=RecordingUISink(),
    )

    assert runtime.prompts == ["first"]


def test_streaming_renderer_redacts_secret_split_across_deltas_without_loss() -> None:
    from litecoder.common.trace import SecretRedactor
    from litecoder.ui.redaction import StreamingEventTextRedactor

    secret = "split-runtime-secret"
    emitted: list[str] = []
    stream = StreamingEventTextRedactor(
        SecretRedactor.with_values((secret,)), emitted.append
    )

    stream.write("before split-runtime-")
    stream.write("secret after")
    stream.flush()
    rendered = "".join(emitted)

    assert secret not in rendered
    assert rendered.startswith("before ")
    assert rendered.endswith(" after")
    assert rendered.count("before ") == 1
    assert rendered.count(" after") == 1


def test_streaming_renderer_redacts_split_bearer_without_configured_values() -> None:
    from litecoder.common.trace import SecretRedactor
    from litecoder.ui.redaction import StreamingEventTextRedactor

    emitted: list[str] = []
    stream = StreamingEventTextRedactor(SecretRedactor.with_values(()), emitted.append)

    stream.write("Authorization: Bear")
    stream.write("er abc123 end")
    stream.flush()
    rendered = "".join(emitted)

    assert "abc123" not in rendered
    assert rendered.endswith(" end")


def test_streaming_renderer_preserves_harmless_secret_prefix_suffix() -> None:
    from litecoder.common.trace import SecretRedactor
    from litecoder.ui.redaction import StreamingEventTextRedactor

    emitted: list[str] = []
    stream = StreamingEventTextRedactor(
        SecretRedactor.with_values(("secret",)), emitted.append
    )

    stream.write("harmless")
    stream.flush()

    assert "".join(emitted) == "harmless"


def test_runtime_secrets_include_configured_and_environment_keys(
    monkeypatch: object,
) -> None:
    from pydantic import SecretStr

    from litecoder.cli.app import _runtime_secrets
    from litecoder.settings import ProviderSettings, Settings

    monkeypatch.setenv("LITECODER_TEST_KEY", "environment-secret")  # type: ignore[attr-defined]
    settings = Settings(
        default_provider="test",
        providers={
            "test": ProviderSettings(
                type="openai-chat-completions",
                model="model",
                api_key=SecretStr("configured-secret"),
                api_key_env="LITECODER_TEST_KEY",
            )
        },
    )

    names, values = _runtime_secrets(settings)

    assert names == ("LITECODER_TEST_KEY",)
    assert set(values) == {"configured-secret", "environment-secret"}


async def test_completed_turn_bounds_hook_cleanup_latency(tmp_path: Path) -> None:
    from litecoder.agent.loop import AgentLoop
    from litecoder.common.errors import RecoveryPolicy
    from litecoder.context.manager import ContextManager
    from litecoder.context.session.models import SessionRecord, SessionStatus
    from litecoder.context.session.store import SQLiteSessionStore
    from litecoder.hooks import HookOutcome, HookPoint
    from litecoder.providers.models import ProviderEvent, StopReason, Usage
    from litecoder.tools.duplicate_guard import DuplicateGuard
    from litecoder.tools.models import ToolCall, ToolContext, ToolResult
    from litecoder.tools.registry import ToolRegistry

    class Provider:
        async def stream(self, request):
            yield ProviderEvent.text_delta(0, "done")
            yield ProviderEvent.content_block_completed(
                0, {"type": "text", "text": "done"}
            )
            yield ProviderEvent.response_completed(
                StopReason.END_TURN, "end_turn", usage=Usage(1, 1)
            )

    class Executor:
        async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
            pytest.fail("tool execution not expected")

    post_cancelled = asyncio.Event()

    class SlowHooks:
        async def dispatch_pre(self, point, payload):
            return HookOutcome(payload)

        async def dispatch_post(self, point, payload):
            if point == HookPoint.AGENT_STOP:
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    post_cancelled.set()
                    raise
            return []

    class Trace:
        async def record(self, payload):
            return None

    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    session = SessionRecord(
        id="session-1",
        project_id="project",
        parent_session_id=None,
        session_type="root",
        title=None,
        provider="provider",
        model="model",
        status=SessionStatus.IDLE,
        workspace_path=str(tmp_path),
        workspace_id="workspace",
        metadata={},
    )
    await store.create_session(session)
    loop = AgentLoop(
        store=store,
        provider=Provider(),
        context=ContextManager(store, model="model"),
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=DuplicateGuard(),
        recovery_policy=RecoveryPolicy(),
        hooks=SlowHooks(),
        trace_recorder=Trace(),
        cleanup_timeout=0.05,
    )

    result = await loop.run_turn("session-1", "hello")
    await store.close()

    assert result.status == "completed"
    await asyncio.wait_for(post_cancelled.wait(), timeout=1)


async def test_default_permission_prompt_uses_terminal_renderer_console(
    monkeypatch: object,
) -> None:
    from io import StringIO
    from types import SimpleNamespace

    from rich.console import Console

    import litecoder.cli.app as cli_app
    from litecoder.tools.permission import PermissionPrompt, PromptChoice

    console = Console(file=StringIO(), force_terminal=True, color_system=None)
    seen: dict[str, object] = {}

    def fake_select(prompt, **kwargs):  # type: ignore[no-untyped-def]
        seen["prompt"] = prompt
        seen.update(kwargs)
        return PromptChoice.DENY

    monkeypatch.setattr(cli_app, "_select_permission_choice", fake_select)

    prompt_callback = cli_app._default_permission_prompt_for_sink(
        SimpleNamespace(renderer=SimpleNamespace(console=console))
    )
    choice = await prompt_callback(PermissionPrompt("run_shell", "high", "scope"))

    assert choice is PromptChoice.DENY
    assert seen["console"] is console


def test_permission_prompt_selects_once_with_down_arrow() -> None:
    from io import StringIO

    from rich.console import Console

    from litecoder.cli.app import _select_permission_choice
    from litecoder.tools.permission import PermissionPrompt, PromptChoice

    keys = iter(("down", "enter"))
    console = Console(file=StringIO(), force_terminal=True, color_system=None)

    choice = _select_permission_choice(
        PermissionPrompt("run_shell", "external", "external:opaque"),
        console=console,
        read_key=lambda: next(keys),
    )

    output = console.file.getvalue()
    assert choice is PromptChoice.ALLOW_ONCE
    assert "allow" in output
    assert "always" in output
    assert "deny" in output
    assert "Allow once" not in output
    assert "Allow for root session" not in output
    assert "Deny" not in output


def test_permission_prompt_selects_always_with_up_arrow() -> None:
    from io import StringIO

    from rich.console import Console

    from litecoder.cli.app import _select_permission_choice
    from litecoder.tools.permission import PermissionPrompt, PromptChoice

    keys = iter(("up", "enter"))
    console = Console(file=StringIO(), force_terminal=True, color_system=None)

    choice = _select_permission_choice(
        PermissionPrompt("run_shell", "external", "external:opaque"),
        console=console,
        read_key=lambda: next(keys),
    )

    assert choice is PromptChoice.ALLOW_FOR_ROOT_SESSION


def test_permission_prompt_fails_closed_without_input() -> None:
    from io import StringIO

    from rich.console import Console

    from litecoder.cli.app import _select_permission_choice
    from litecoder.tools.permission import PermissionPrompt, PromptChoice

    def no_input() -> str:
        raise EOFError

    choice = _select_permission_choice(
        PermissionPrompt("run_shell", "external", "external:opaque"),
        console=Console(file=StringIO(), force_terminal=True, color_system=None),
        read_key=no_input,
    )

    assert choice is PromptChoice.DENY


def test_streaming_renderer_never_cuts_inside_completed_bearer_match() -> None:
    from litecoder.common.trace import SecretRedactor
    from litecoder.ui.redaction import StreamingEventTextRedactor

    redactor = SecretRedactor.with_values(("secret-token", "abc123"))
    emitted: list[str] = []
    stream = StreamingEventTextRedactor(redactor, emitted.append)

    chunks = ["Bearer h", "ello ", " abc123 \n", "zzBearer", " "]
    for chunk in chunks:
        stream.write(chunk)
    stream.flush()

    rendered = "".join(emitted)
    expected = redactor.redact_text("".join(chunks))
    assert rendered == expected
    assert "Bearer hello" not in rendered


def test_resume_without_session_id_is_a_safe_cli_diagnostic() -> None:
    result = CliRunner().invoke(app, ["resume"])

    assert result.exit_code != 2
    assert "session id" in result.output.lower()


async def test_streaming_redactor_propagates_async_emit_awaitable() -> None:
    import inspect

    from litecoder.common.trace import SecretRedactor
    from litecoder.ui.redaction import StreamingEventTextRedactor

    emitted: list[str] = []

    async def emit(text: str) -> None:
        emitted.append(text)

    stream = StreamingEventTextRedactor(SecretRedactor.with_values(()), emit)
    outcome = stream.write("hello")

    assert inspect.isawaitable(outcome)
    await outcome
    assert emitted == ["hello"]


async def test_build_runtime_wires_ui_sink_to_tool_executor(
    monkeypatch: object, tmp_path: Path
) -> None:
    from pydantic import SecretStr

    import litecoder.cli.app as cli_app
    from litecoder.paths import AppPaths
    from litecoder.settings import ProviderSettings, Settings
    from litecoder.tools.models import ToolContext
    from litecoder.ui.events import UIEventType
    from litecoder.ui.sink import RecordingUISink
    from tests.fakes.provider import FakeProvider

    paths = AppPaths(
        user_dir=tmp_path / ".litecoder",
        sessions_db=tmp_path / ".litecoder" / "sessions.db",
        project_id="project",
        project_dir=tmp_path / ".litecoder" / "projects" / "project",
        workspace_id="workspace",
        workspace_root=tmp_path,
    )
    settings = Settings(
        default_provider="fake",
        default_model="model",
        providers={
            "fake": ProviderSettings(
                type="openai-chat-completions",
                model="model",
                api_key=SecretStr("runtime-secret"),
            )
        },
    )
    captured: dict[str, object] = {}
    original_executor = cli_app.ToolExecutor

    class CapturingExecutor(original_executor):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            captured["ui_sink"] = kwargs.get("ui_sink")
            captured["ui_factory_resolver"] = kwargs.get("ui_factory_resolver")
            captured["workspace_lock_resolver"] = kwargs.get("workspace_lock_resolver")
            super().__init__(*args, **kwargs)

    class FakeMCPManager:
        def __init__(self, registry) -> None:
            self.registry = registry
            captured["tools"] = registry

        async def connect_all(self, servers) -> None:
            return None

        async def close_all(self) -> None:
            return None

    monkeypatch.setattr(cli_app.AppPaths, "discover", staticmethod(lambda cwd: paths))  # type: ignore[attr-defined]

    def load_settings(discovered_paths: AppPaths) -> Settings:
        config_path = discovered_paths.user_dir / "config.toml"
        assert config_path.exists()
        content = config_path.read_text(encoding="utf-8")
        assert 'base_url = "https://api.openai.com/v1"' in content
        assert 'model = "gpt-5.6-sol"' in content
        return settings

    monkeypatch.setattr(cli_app.Settings, "load", staticmethod(load_settings))  # type: ignore[attr-defined]
    monkeypatch.setattr(cli_app, "ToolExecutor", CapturingExecutor)
    monkeypatch.setattr(cli_app, "MCPConnectionManager", FakeMCPManager)
    monkeypatch.setattr(
        cli_app.ProviderRegistry,
        "create",
        lambda _self, _name, _settings: FakeProvider([]),
    )

    sink = RecordingUISink()
    runtime = await cli_app.build_runtime(tmp_path, ui_sink=sink)
    try:
        assert not (paths.workspace_root / ".memory").exists()
        assert runtime.provider_models == {"fake": "model"}
        tool_names = {tool.spec.name for tool in captured["tools"].list()}
        assert {
            "memory_list",
            "memory_read",
            "memory_update",
            "memory_delete",
        } <= tool_names
        assert runtime.manual_compactor is not None
        compact_manager = runtime.manual_compactor.manager_factory(
            "fake", "model", 64
        )
        assert compact_manager.context_budget_tokens == 64
        assert compact_manager.summarizer is not None
        assert compact_manager.summarizer.max_tokens == 8_000
        assert runtime.startup_lock is not None
        assert runtime.startup_lock.path.parent == paths.lock_dir
        assert runtime.task_manager is not None
        assert runtime.task_manager.file_lock is not None
        assert runtime.task_manager.file_lock.path.parent == paths.lock_dir
        assert runtime.session_lock_factory is not None
        assert runtime.session_lock_factory("root-1").path.parent == paths.lock_dir

        workspace_lock_resolver = captured["workspace_lock_resolver"]
        assert workspace_lock_resolver is not None
        workspace_lock = workspace_lock_resolver(  # type: ignore[operator]
            ToolContext("session-1", "workspace", tmp_path)
        )
        assert workspace_lock.path.parent == paths.lock_dir

        assert (paths.lock_dir / f"litecoder-memory-{paths.project_id}.lock").exists()
        assert not list(paths.user_dir.glob("*.lock"))

        assert captured["ui_sink"] is not sink
        resolver = captured["ui_factory_resolver"]
        assert resolver is not None
        factory = resolver(  # type: ignore[operator]
            ToolContext(
                "session-1",
                "workspace",
                tmp_path,
                metadata={"root_session_id": "root-1"},
            )
        )
        event = factory.next(UIEventType.DIAGNOSTIC, payload={"message": "ok"})
        assert event.session_id == "session-1"
        assert event.root_session_id == "root-1"
        captured_sink = captured["ui_sink"]
        captured_sink.emit(  # type: ignore[attr-defined]
            factory.next(
                UIEventType.ASSISTANT_COMPLETED,
                payload={"text": "runtime-secret Bearer abc123"},
            )
        )
        assert sink.events[-1].payload["text"] == "[REDACTED] [REDACTED]"
    finally:
        await runtime.close()
        assert not (paths.workspace_root / ".memory").exists()


async def test_build_runtime_registers_and_closes_configured_mcp_servers(
    monkeypatch: object, tmp_path: Path
) -> None:
    from litecoder.cli.app import build_runtime
    from litecoder.paths import AppPaths
    from litecoder.tools.models import ToolExecution, ToolSpec

    class FakeTool:
        spec = ToolSpec("mcp__docs__lookup", "Lookup docs", {}, False)

        async def execute(self, call, context):
            return ToolExecution.success("ok")

    class FakeMCPManager:
        instances: list["FakeMCPManager"] = []

        def __init__(self, registry) -> None:
            self.registry = registry
            self.servers = None
            self.closed = False
            self.instances.append(self)

        async def connect_all(self, servers) -> None:
            self.servers = dict(servers)
            self.registry.register(FakeTool())

        async def close_all(self) -> None:
            self.closed = True

    user_dir = tmp_path / ".litecoder"
    user_dir.mkdir()
    (user_dir / "config.toml").write_text(
        "\n".join(
            [
                'default_provider = "fake"',
                'default_model = "model"',
                "",
                "[providers.fake]",
                'type = "openai-chat-completions"',
                'model = "model"',
                'api_key = "key"',
                "",
                "[mcp_servers.docs]",
                'transport = "stdio"',
                'command = "fake-mcp"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    paths = AppPaths(
        user_dir=user_dir,
        sessions_db=user_dir / "sessions.db",
        project_id="project",
        project_dir=user_dir / "projects" / "project",
        workspace_id="workspace",
        workspace_root=tmp_path,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "litecoder.cli.app.AppPaths.discover", staticmethod(lambda cwd: paths)
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "litecoder.cli.app.MCPConnectionManager", FakeMCPManager, raising=False
    )

    runtime = await build_runtime(tmp_path)
    try:
        assert FakeMCPManager.instances
        assert "docs" in FakeMCPManager.instances[0].servers
    finally:
        await runtime.close()

    assert FakeMCPManager.instances[0].closed is True
