"""Supporting implementation for interactive."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import suppress
from pathlib import Path

from litecoder.agent.runtime import AgentRuntime
from litecoder.cli.local_commands import LocalCommandRouter
from litecoder.tools.permission import PermissionMode
from litecoder.ui.events import UIEventFactory, UIEventType
from litecoder.ui.input import InputInterrupt, TerminalInput
from litecoder.ui.renderers.terminal import TerminalRenderer
from litecoder.ui.sink import RuntimeUISink, emit_ui


async def interactive_session(
    runtime: AgentRuntime,
    *,
    terminal_input: TerminalInput | None = None,
    renderer: TerminalRenderer | None = None,
    ui_sink: RuntimeUISink | None = None,
    session_id: str | None = None,
) -> None:
    """Handle the interactive session operation."""
    renderer = renderer or TerminalRenderer()
    terminal_input = terminal_input or TerminalInput(renderer.console)
    sink = ui_sink
    router = LocalCommandRouter(runtime)
    factory = UIEventFactory(session_id=session_id)
    workspace_path = _runtime_workspace_path(runtime)
    model_name = await _initial_model(runtime, session_id)
    permission_mode = _runtime_permission_mode(runtime)
    _apply_runtime_permission_mode(runtime, permission_mode)

    def footer() -> str:
        return _footer(model_name, workspace_path, permission_mode)

    def cycle_permission_mode() -> None:
        nonlocal permission_mode
        permission_mode = _next_permission_mode(permission_mode)
        _apply_runtime_permission_mode(runtime, permission_mode)

    renderer.startup_banner(workspace_path, model_name)
    consecutive_ctrl_c = 0
    while True:
        try:
            text = await _read_input(
                terminal_input,
                footer=footer,
                on_permission_mode_toggle=cycle_permission_mode,
            )
        except InputInterrupt as error:
            consecutive_ctrl_c = _next_interrupt_count(
                consecutive_ctrl_c,
                error.source,
            )
            if consecutive_ctrl_c >= 2:
                await _emit_exit_summary(sink, factory, session_id)
                return
            continue
        except KeyboardInterrupt:
            consecutive_ctrl_c = _next_interrupt_count(consecutive_ctrl_c, "ctrl_c")
            if consecutive_ctrl_c >= 2:
                await _emit_exit_summary(sink, factory, session_id)
                return
            continue
        except EOFError:
            await _emit_exit_summary(sink, factory, session_id)
            return
        consecutive_ctrl_c = 0
        if not text.strip():
            continue
        local = await router.dispatch(text, session_id=session_id)
        if local.handled:
            if local.message:
                await emit_ui(
                    sink,
                    factory.next(
                        UIEventType.DIAGNOSTIC,
                        payload={"message": local.message},
                    ),
                )
            if local.replacement_session_id is not None:
                session_id = local.replacement_session_id
                factory.session_id = session_id
                model_name = _runtime_model(runtime)
            if local.clear_requested:
                renderer.console.clear()
                session_id = None
                model_name = _runtime_model(runtime)
                factory = UIEventFactory(session_id=None)
            if local.exit_requested:
                await _emit_exit_summary(sink, factory, session_id)
                return
            continue
        async with _live_draft_area(
            terminal_input,
            footer=footer,
            on_permission_mode_toggle=cycle_permission_mode,
        ) as interrupt:
            result, interrupt_source = await _run_turn_with_interrupt(
                runtime,
                session_id,
                text,
                interrupt,
            )
        if result is None:
            renderer.flush()
            consecutive_ctrl_c = _next_interrupt_count(
                consecutive_ctrl_c,
                interrupt_source or "escape",
            )
            if consecutive_ctrl_c >= 2:
                await _emit_exit_summary(sink, factory, session_id)
                return
            continue
        consecutive_ctrl_c = 0
        session_id = result.session_id
        factory.session_id = session_id


async def _run_turn_with_interrupt(
    runtime: AgentRuntime,
    session_id: str | None,
    text: str,
    interrupt: object,
):
    """Run the turn with interrupt."""
    turn_task = asyncio.create_task(
        runtime.run(text) if session_id is None else runtime.resume(session_id, text)
    )
    wait_task = _interrupt_wait_task(interrupt)
    if wait_task is None:
        return await turn_task, None

    try:
        done, _ = await asyncio.wait(
            {turn_task, wait_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if turn_task in done:
            wait_task.cancel()
            with suppress(asyncio.CancelledError):
                await wait_task
            return await turn_task, None

        source = await wait_task
        turn_task.cancel()
        with suppress(asyncio.CancelledError):
            await turn_task
        return None, source
    finally:
        if not wait_task.done():
            wait_task.cancel()
        if not turn_task.done():
            turn_task.cancel()
        with suppress(asyncio.CancelledError):
            await turn_task


def _interrupt_wait_task(interrupt: object) -> asyncio.Task[str] | None:
    wait = getattr(interrupt, "wait", None)
    if not callable(wait):
        return None
    return asyncio.create_task(wait())


def _next_interrupt_count(current: int, source: str) -> int:
    return current + 1 if source == "ctrl_c" else 0


async def _emit_exit_summary(
    sink: RuntimeUISink | None,
    factory: UIEventFactory,
    session_id: str | None,
) -> None:
    if session_id:
        await emit_ui(
            sink,
            factory.next(
                UIEventType.DIAGNOSTIC,
                payload={"message": f"session={session_id}"},
            ),
        )

def _runtime_workspace_path(runtime: AgentRuntime) -> str:
    paths = getattr(runtime, "paths", None)
    workspace_root = getattr(paths, "workspace_root", None)
    if workspace_root is None:
        return str(Path.cwd())
    return str(workspace_root)


async def _initial_model(
    runtime: AgentRuntime,
    session_id: str | None,
) -> str:
    if session_id is not None:
        context = await runtime.store.load_context(session_id)
        model = getattr(context.session, "model", None)
        if isinstance(model, str) and model.strip():
            return model.strip()
    return _runtime_model(runtime)


def _runtime_model(runtime: AgentRuntime) -> str:
    model = getattr(runtime, "model", None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    return "unknown"


def _footer(
    model_name: str,
    workspace_path: str,
    permission_mode: PermissionMode | str = PermissionMode.ASK,
) -> str:
    mode = PermissionMode(str(permission_mode))
    return (
        f"{_permission_mode_label(mode)} model: {model_name}  "
        f"workspace: {workspace_path}"
    )


def _permission_mode_label(mode: PermissionMode) -> str:
    return {
        PermissionMode.ASK: "Ask",
        PermissionMode.READ_ONLY: "Read-only",
        PermissionMode.BYPASS: "Bypass",
    }[mode]


def _runtime_permission_mode(runtime: AgentRuntime) -> PermissionMode:
    value = getattr(runtime, "permission_mode", PermissionMode.ASK.value)
    try:
        return PermissionMode(str(value))
    except (TypeError, ValueError):
        return PermissionMode.ASK


def _apply_runtime_permission_mode(runtime: AgentRuntime, mode: PermissionMode) -> None:
    try:
        setattr(runtime, "permission_mode", mode.value)
    except Exception:
        return


def _next_permission_mode(mode: PermissionMode) -> PermissionMode:
    order = (PermissionMode.ASK, PermissionMode.READ_ONLY, PermissionMode.BYPASS)
    return order[(order.index(mode) + 1) % len(order)]


async def _read_input(
    terminal_input: TerminalInput,
    *,
    footer: object,
    on_permission_mode_toggle: object,
) -> str:
    read_async = terminal_input.read_async
    if _supports_keyword(read_async, "on_permission_mode_toggle"):
        return await read_async(
            footer=footer,
            on_permission_mode_toggle=on_permission_mode_toggle,
        )
    return await read_async(footer=footer)


def _live_draft_area(
    terminal_input: TerminalInput,
    *,
    footer: object,
    on_permission_mode_toggle: object,
):  # type: ignore[no-untyped-def]
    live_draft_area = terminal_input.live_draft_area
    if _supports_keyword(live_draft_area, "on_permission_mode_toggle"):
        return live_draft_area(
            footer=footer,
            on_permission_mode_toggle=on_permission_mode_toggle,
        )
    return live_draft_area(footer=footer)


def _supports_keyword(callable_object: object, name: str) -> bool:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == name:
            return True
    return False
