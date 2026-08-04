"""Terminal event rendering."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.segment import Segment, Segments
from rich.table import Table
from rich.text import Text

from litecoder.ui.events import RuntimeUIEvent, UIEventType
from litecoder.ui import terminal_state

from litecoder.ui.markdown import WrappingMarkdown
from litecoder.ui.presenters import normalize_windows_drive_letters

_ASSISTANT_ICON = "●"
_TOOL_ICON = "●"
_MEMORY_DIAGNOSTIC_SPECS = {
    "load": ("load", frozenset({"recalled"}), ("count",)),
    "extract": (
        "extract",
        frozenset({
            "empty",
            "provider_failed",
            "truncated",
            "malformed",
            "partial_rejected",
            "failed",
            "timeout",
        }),
        ("accepted", "rejected", "written"),
    ),
}
_MAX_MEMORY_DIAGNOSTIC_COUNT = 1_000_000


@dataclass(slots=True)
class TerminalRenderer:
    """Data model representing the terminal renderer."""
    console: Console
    workspace_root: Path | None
    _assistant_buffer: list[str]
    _tool_invocation_keys: set[str]
    _memory_count: int
    _tool_starts: dict[str, RuntimeUIEvent]
    _tool_call_completions: dict[str, RuntimeUIEvent]

    def __init__(
        self,
        console: Console | None = None,
        *,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.console = console or Console()
        self.workspace_root = (
            Path(workspace_root) if workspace_root is not None else None
        )
        _configure_unicode_output(self.console)
        self._assistant_buffer = []
        self._tool_invocation_keys = set()
        self._memory_count = 0
        self._tool_starts = {}
        self._tool_call_completions = {}

    def emit(self, event: RuntimeUIEvent) -> None:
        """Emit the supplied event."""
        if event.type is UIEventType.TURN_STARTED:
            self._turn_started()
        elif event.type is UIEventType.MODEL_REQUESTED:
            self._model_requested(event)
        elif event.type is UIEventType.ASSISTANT_DELTA:
            self._assistant_delta(event)
        elif event.type is UIEventType.ASSISTANT_COMPLETED:
            self._assistant_completed(event)
        elif event.type is UIEventType.TOOL_CALL_COMPLETED:
            self._tool_call_completed(event)
        elif event.type is UIEventType.TOOL_EXECUTION_STARTED:
            self._tool_execution_started(event)
        elif event.type is UIEventType.TOOL_EXECUTION_FINISHED:
            self._tool_execution_finished(event)
        elif event.type is UIEventType.TOOL_EXECUTION_FAILED:
            self._tool_execution_failed(event)
        elif event.type is UIEventType.TOOL_EXECUTION_DENIED:
            self._tool_execution_denied(event)
        elif event.type is UIEventType.PROVIDER_ERROR:
            self._provider_error(event)
        elif event.type is UIEventType.DIAGNOSTIC:
            self._diagnostic(event)
        elif event.type is UIEventType.TURN_FINISHED:
            self._turn_finished(event)

    def flush(self) -> None:
        """Flush pending output."""
        if not self._assistant_buffer:
            return
        text = "".join(self._assistant_buffer)
        self._assistant_buffer.clear()
        self._assistant_message(text)

    def startup_banner(self, workspace_path: str, model_name: str = "unknown") -> None:
        """Handle the startup banner operation."""
        workspace = normalize_windows_drive_letters(
            workspace_path.strip() or str(Path.cwd())
        )
        welcome = Text()
        welcome.append("Welcome to LiteCoder CLI!", style="bold #c084fc")
        welcome.append("\n\n")
        model = model_name.strip() or "unknown"
        welcome.append(f"Workspace: {workspace}", style="dim")
        welcome.append("\n")
        welcome.append(f"Using {model} (from .litecoder\\config.toml)", style="dim")

        self.console.print(Panel(welcome, border_style="bright_black", padding=(1, 2)))
        self.console.print()

    def user_prompt(self, value: str) -> None:
        """Handle the user prompt operation."""
        self.console.print(
            Text(f"> {normalize_windows_drive_letters(value)}", style="bold")
        )

    def _turn_started(self) -> None:
        self._assistant_buffer.clear()
        self._tool_invocation_keys.clear()
        self._memory_count = 0
        self._tool_starts.clear()
        self._tool_call_completions.clear()
        terminal_state.clear_todo_progress(self.console)

    def _model_requested(self, event: RuntimeUIEvent) -> None:
        count = _memory_count_from_payload(event.payload)
        if count is not None:
            self._memory_count = max(self._memory_count, count)

    def _assistant_delta(self, event: RuntimeUIEvent) -> None:
        text = _string_payload(event, "text")
        if text:
            self._assistant_buffer.append(text)

    def _assistant_completed(self, event: RuntimeUIEvent) -> None:
        self._assistant_buffer.clear()
        text = _string_payload(event, "text")
        self._assistant_message(text)

    def _assistant_message(self, text: str) -> None:
        if not text.strip():
            return
        terminal_state.stop_waiting_status(self.console)
        grid = Table.grid(expand=True)
        grid.add_column(width=1)
        grid.add_column(width=1)
        grid.add_column(ratio=1, overflow="fold")
        grid.add_row(
            Text(_ASSISTANT_ICON, style="bold white"),
            "",
            WrappingMarkdown(normalize_windows_drive_letters(text)),
        )
        self._print_atomic(grid)

    def _print_atomic(self, *renderables: object) -> None:
        options = self.console.options
        segments: list[Segment] = []
        for renderable in renderables:
            rendered = list(self.console.render(renderable, options))
            segments.extend(rendered)
            if not rendered or not rendered[-1].text.endswith("\n"):
                segments.append(Segment.line())
        if not segments:
            return
        with terminal_state.suspend_waiting_status(self.console):
            self.console.print(Segments(segments), end="", soft_wrap=True)

    def _tool_call_completed(self, event: RuntimeUIEvent) -> None:
        if event.tool_call_id is not None:
            self._tool_call_completions[event.tool_call_id] = event

    def _tool_execution_started(self, event: RuntimeUIEvent) -> None:
        self._record_tool_invocation(event)

    def _tool_execution_finished(self, event: RuntimeUIEvent) -> None:
        key = self._record_tool_invocation(event)
        started = self._tool_starts.pop(key, event)
        completed = self._tool_call_completions.pop(key, None)
        self._render_tool_result(event, started, completed, success=True)

    def _tool_execution_failed(self, event: RuntimeUIEvent) -> None:
        key = self._record_tool_invocation(event)
        started = self._tool_starts.pop(key, event)
        completed = self._tool_call_completions.pop(key, None)
        self._render_tool_result(event, started, completed, success=False)

    def _tool_execution_denied(self, event: RuntimeUIEvent) -> None:
        key = self._record_tool_invocation(event)
        started = self._tool_starts.pop(key, event)
        completed = self._tool_call_completions.pop(key, None)
        self._render_tool_result(event, started, completed, success=False, denied=True)

    def _record_tool_invocation(self, event: RuntimeUIEvent) -> str:
        key = _tool_key(event)
        self._tool_invocation_keys.add(key)
        if (
            event.type is UIEventType.TOOL_EXECUTION_STARTED
            or key not in self._tool_starts
        ):
            self._tool_starts[key] = event
        return key

    def _render_tool_result(
        self,
        event: RuntimeUIEvent,
        started: RuntimeUIEvent,
        completed: RuntimeUIEvent | None,
        *,
        success: bool,
        denied: bool = False,
    ) -> None:
        arguments = (
            _tool_arguments(started)
            or (_tool_arguments(completed) if completed is not None else {})
            or _tool_arguments(event)
        )
        tool_name = event.tool_name or started.tool_name or "tool"
        if (
            success
            and tool_name == "todo_write"
            and terminal_state.replace_todo_progress(
                self.console, _todo_values(event, arguments)
            )
        ):
            if not terminal_state.has_live_waiting_surface(self.console):
                self._todo_progress_block()
            return
        title = _tool_title(tool_name, arguments, self.workspace_root)
        detail = (
            _success_detail_lines(event, self.workspace_root)
            if success
            else _failure_detail_lines(event, denied=denied)
        )
        self._tool_block(title, detail, success=success)

    def _todo_progress_block(self) -> None:
        items = terminal_state.todo_progress_items(self.console)
        if not items:
            return
        title = terminal_state._todo_progress_heading(items)
        detail = [text for _, text in terminal_state._todo_progress_lines(items)]
        self._tool_block(title, detail, success=True)

    def _tool_block(
        self, title: str, detail_lines: list[str], *, success: bool
    ) -> None:
        color = "green" if success else "red"
        detail_lines = detail_lines or (["Done"] if success else ["Error"])
        title = normalize_windows_drive_letters(title)
        detail_lines = [normalize_windows_drive_letters(line) for line in detail_lines]
        grid = Table.grid(expand=True)
        grid.add_column(width=1)
        grid.add_column(width=1)
        grid.add_column(ratio=1, overflow="fold")
        grid.add_row(
            Text(_TOOL_ICON, style=f"bold {color}"), "", Text(title, style="bold white")
        )
        grid.add_row("", "", _tool_detail_text(detail_lines, success=success))
        self._print_atomic(grid)

    def _provider_error(self, event: RuntimeUIEvent) -> None:
        code = _string_payload(event, "code")
        message = _string_payload(event, "message")
        request = f" request={event.request_id}" if event.request_id else ""
        retryable = event.payload.get("retryable") is True
        retrying_value = event.payload.get("retrying")
        retrying = retryable if retrying_value is None else bool(retrying_value)
        attempt = event.payload.get("attempt")
        max_attempts = event.payload.get("max_attempts")
        has_progress = (
            isinstance(attempt, int)
            and not isinstance(attempt, bool)
            and isinstance(max_attempts, int)
            and not isinstance(max_attempts, bool)
        )
        progress = f" ({attempt}/{max_attempts})" if has_progress else ""
        if retrying:
            self._print_atomic(Text(f"Retrying...{progress}", style="yellow"))
            return
        detail = message or code or "Provider request failed"
        if retryable and has_progress:
            detail += f" · Retries exhausted{progress}"
        self._print_atomic(
            Text(f"Provider error{request} {detail}".rstrip(), style="red")
        )
    def _diagnostic(self, event: RuntimeUIEvent) -> None:
        message = _string_payload(event, "message")
        if not message:
            message = _memory_diagnostic_text(event.payload)
        if message:
            self._print_atomic(Text(message, style="yellow"))

    def _turn_finished(self, event: RuntimeUIEvent) -> None:
        self.flush()
        terminal_state.stop_waiting_status(self.console)
        self._print_atomic(
            "",
            Text(
                _turn_finished_text(
                    event, len(self._tool_invocation_keys), self._memory_count
                ),
                style="dim",
            ),
        )
        self._turn_started()


class TerminalUISink:
    """Component responsible for the terminal ui sink."""
    def __init__(self, renderer: TerminalRenderer) -> None:
        self.renderer = renderer

    def emit(self, event: RuntimeUIEvent) -> None:
        """Emit the supplied event."""
        self.renderer.emit(event)

    def flush(self) -> None:
        """Flush pending output."""
        self.renderer.flush()


def _configure_unicode_output(console: Console) -> None:
    """Configure the unicode output."""
    stream = getattr(console, "file", None)
    encoding = getattr(stream, "encoding", None)
    if not isinstance(encoding, str) or _can_encode(_ASSISTANT_ICON, encoding):
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (OSError, TypeError, ValueError):
        return


def _can_encode(value: str, encoding: str) -> bool:
    try:
        value.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _string_payload(event: RuntimeUIEvent, key: str) -> str:
    value = event.payload.get(key)
    return normalize_windows_drive_letters(value) if isinstance(value, str) else ""


def _memory_diagnostic_text(payload: Mapping[str, object]) -> str:
    memory = payload.get("memory")
    if not isinstance(memory, Mapping):
        return ""
    operation = memory.get("operation")
    status = memory.get("status")
    if not isinstance(operation, str) or not isinstance(status, str):
        return ""
    if operation == "extract" and memory.get("visible") is not True:
        return ""
    spec = _MEMORY_DIAGNOSTIC_SPECS.get(operation)
    if spec is None:
        return ""
    label, statuses, fields = spec
    if status not in statuses:
        return ""

    details: list[str] = []
    for field in fields:
        value = memory.get(field)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= _MAX_MEMORY_DIAGNOSTIC_COUNT
        ):
            details.append(f"{field}={value}")

    title = "Memory" if not label else f"Memory {label}"
    rendered_status = status.replace("_", " ")
    detail_text = f" ({', '.join(details)})" if details else ""
    return f"{title}: {rendered_status}{detail_text}"


def _memory_count_from_payload(payload: Mapping[str, object]) -> int | None:
    for key in ("memory_count", "retrieved_memory_count"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    memories = payload.get("retrieved_memories")
    if isinstance(memories, Sequence) and not isinstance(memories, (str, bytes)):
        return len(memories)
    return None


def _tool_key(event: RuntimeUIEvent) -> str:
    return event.tool_call_id or f"{event.sequence}:{event.tool_name or 'tool'}"


def _tool_arguments(event: RuntimeUIEvent) -> Mapping[str, object]:
    arguments = event.payload.get("arguments")
    return arguments if isinstance(arguments, Mapping) else {}


def _todo_values(event: RuntimeUIEvent, arguments: Mapping[str, object]) -> object:
    metadata = event.payload.get("metadata")
    if isinstance(metadata, Mapping) and "todos" in metadata:
        return metadata["todos"]
    return arguments.get("todos")


def _tool_title(
    tool_name: str,
    arguments: Mapping[str, object],
    workspace_root: Path | None = None,
) -> str:
    if tool_name == "run_shell":
        command = _shell_command(arguments) or "shell"
        cwd = _optional_string(arguments.get("cwd"))
        if cwd and cwd != ".":
            command = f"{command} cwd={_absolute_path(cwd, workspace_root)}"
        return f"Bash({command})"
    if tool_name == "read_file":
        return _path_title("Read", arguments, workspace_root)
    if tool_name == "write_file":
        return _path_title("Write", arguments, workspace_root)
    if tool_name == "edit_file":
        return _path_title("Edit", arguments, workspace_root)
    if tool_name == "glob_files":
        pattern = _optional_string(arguments.get("pattern")) or "*"
        return f"Glob({_absolute_pattern(pattern, workspace_root)})"
    if tool_name == "search_text":
        query = _optional_string(arguments.get("query")) or ""
        glob = _optional_string(arguments.get("glob"))
        suffix = f" in {_absolute_pattern(glob, workspace_root)}" if glob else ""
        return f"Search({query}{suffix})"
    if tool_name == "git_status":
        return "Git(status)"
    if tool_name == "git_diff":
        path = _optional_string(arguments.get("path"))
        staged = arguments.get("staged") is True
        parts = ["diff"]
        if staged:
            parts.append("--cached")
        if path:
            parts.append(_absolute_path(path, workspace_root))
        return f"Git({' '.join(parts)})"
    summary = _argument_summary(arguments, workspace_root)
    return f"{tool_name}({summary})" if summary else tool_name


def _path_title(
    label: str,
    arguments: Mapping[str, object],
    workspace_root: Path | None = None,
) -> str:
    value = _optional_string(arguments.get("path"))
    return f"{label}({_absolute_path(value, workspace_root)})" if value else label


def _shell_command(arguments: Mapping[str, object]) -> str:
    command = _optional_string(arguments.get("command"))
    if command:
        return command
    argv = arguments.get("argv")
    if isinstance(argv, Sequence) and not isinstance(argv, (str, bytes)):
        values = [item for item in argv if isinstance(item, str)]
        if values:
            return subprocess.list2cmdline(values)
    return ""


def _argument_summary(
    arguments: Mapping[str, object], workspace_root: Path | None = None
) -> str:
    for key in ("path", "query", "pattern", "command", "name"):
        value = _optional_string(arguments.get(key))
        if value:
            return _absolute_path(value, workspace_root) if key == "path" else value
    argv = arguments.get("argv")
    if isinstance(argv, Sequence) and not isinstance(argv, (str, bytes)):
        values = [item for item in argv if isinstance(item, str)]
        if values:
            return subprocess.list2cmdline(values)
    return ""


def _success_detail_lines(
    event: RuntimeUIEvent, workspace_root: Path | None = None
) -> list[str]:
    preview = event.payload.get("preview")
    lines = _preview_lines(preview, event.tool_name, workspace_root)
    if lines:
        return lines
    metadata = event.payload.get("metadata")
    if isinstance(metadata, Mapping):
        lines = _metadata_detail_lines(metadata, event.tool_name, workspace_root)
        if lines:
            return lines
    artifact = event.payload.get("artifact")
    if isinstance(artifact, Mapping):
        path = _optional_string(artifact.get("path"))
        if path:
            return [f"Output saved to {_absolute_path(path, workspace_root)}"]
    return ["Done"]


def _failure_detail_lines(event: RuntimeUIEvent, *, denied: bool) -> list[str]:
    metadata = event.payload.get("metadata")
    if isinstance(metadata, Mapping):
        exit_code = metadata.get("exit_code")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            return [f"Error: Exit code {exit_code}"]
    if denied:
        reason = _string_payload(event, "reason") or _string_payload(event, "message")
        return [f"Denied: {reason}" if reason else "Denied"]
    message = _string_payload(event, "message")
    status = _string_payload(event, "status")
    detail = message or status or "Tool execution failed"
    return [f"Error: {detail}"]


def _metadata_detail_lines(
    metadata: Mapping[str, object],
    tool_name: str | None,
    workspace_root: Path | None = None,
) -> list[str]:
    preview = metadata.get("preview")
    lines = _preview_lines(preview, tool_name, workspace_root)
    if lines:
        return lines
    path = _optional_string(metadata.get("path"))
    if path:
        parts = [_absolute_path(path, workspace_root)]
        for key in ("changed", "replacements", "occurrences", "total_lines", "count"):
            value = metadata.get(key)
            if isinstance(value, (str, int, bool)) and not (
                isinstance(value, bool) and key not in {"changed"}
            ):
                parts.append(f"{key}={value}")
        return [", ".join(parts)]
    count = metadata.get("count")
    if isinstance(count, int) and not isinstance(count, bool):
        return [f"{count} result{'s' if count != 1 else ''}"]
    return []


def _preview_lines(
    value: object,
    tool_name: str | None,
    workspace_root: Path | None = None,
) -> list[str]:
    if isinstance(value, str):
        if not value.strip():
            return []
        return _string_preview_lines(value, tool_name, workspace_root)
    if isinstance(value, Mapping):
        return [_mapping_preview(value, workspace_root)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return ["No results"]
        first = _preview_item(value[0], tool_name, workspace_root)
        lines = [first]
        remaining = len(value) - 1
        if remaining > 0:
            lines.append(f"... +{remaining} entr{'y' if remaining == 1 else 'ies'}")
        return lines
    return []


def _string_preview_lines(
    value: str,
    tool_name: str | None,
    workspace_root: Path | None = None,
) -> list[str]:
    raw_lines = [line.rstrip() for line in value.splitlines() if line.strip()]
    if not raw_lines:
        return ["No output"]
    first = _preview_string_line(raw_lines[0], tool_name, workspace_root)
    lines = [first]
    remaining = len(raw_lines) - 1
    if remaining > 0:
        lines.append(f"... +{remaining} line{'s' if remaining != 1 else ''}")
    return lines


def _preview_item(
    value: object,
    tool_name: str | None,
    workspace_root: Path | None = None,
) -> str:
    if isinstance(value, str):
        return _preview_string_line(value, tool_name, workspace_root)
    if isinstance(value, Mapping):
        return _mapping_preview(value, workspace_root)
    return _json_preview(value)


def _preview_string_line(
    value: str,
    tool_name: str | None,
    workspace_root: Path | None = None,
) -> str:
    if tool_name == "glob_files":
        return _absolute_path(value, workspace_root)
    if tool_name == "search_text":
        parsed = _search_result_line(value, workspace_root)
        if parsed:
            return parsed
    return value


def _search_result_line(value: str, workspace_root: Path | None = None) -> str:
    path, line, column, text = _split_search_line(value)
    if not path:
        return ""
    return f"{_absolute_path(path, workspace_root)}:{line}:{column}:{text}"


def _split_search_line(value: str) -> tuple[str, str, str, str]:
    first, sep, rest = value.partition(":")
    if not sep:
        return "", "", "", ""
    second, sep, rest = rest.partition(":")
    if not sep:
        return "", "", "", ""
    third, sep, text = rest.partition(":")
    if not sep or not second.isdigit() or not third.isdigit():
        return "", "", "", ""
    return first, second, third, text


def _mapping_preview(
    value: Mapping[str, object], workspace_root: Path | None = None
) -> str:
    path = _optional_string(value.get("path"))
    if path:
        line = value.get("line")
        column = value.get("column")
        text = _optional_string(value.get("text"))
        if isinstance(line, int) and isinstance(column, int):
            suffix = f":{line}:{column}"
            return (
                f"{_absolute_path(path, workspace_root)}{suffix}: {text}"
                if text
                else f"{_absolute_path(path, workspace_root)}{suffix}"
            )
        parts = [f"path={_absolute_path(path, workspace_root)}"]
        for key, item in value.items():
            if key == "path":
                continue
            if isinstance(item, (str, int, float, bool)):
                parts.append(f"{key}={item}")
        return ", ".join(parts)
    return _json_preview(value)


def _tool_detail_text(lines: list[str], *, success: bool) -> Text:
    text = Text()
    for index, line in enumerate(lines):
        if index == 0:
            text.append("└ ", style="dim")
        else:
            text.append("\n  ", style="dim")
        text.append(line, style="red" if not success else "dim")
    return text


def _turn_summary(tool_invocations: int, memory_count: int) -> str:
    parts: list[str] = []
    if tool_invocations > 0:
        parts.append(f"Tools called: {tool_invocations}")
    if memory_count > 0:
        parts.append(f"memories recalled: {memory_count}")
    return ", ".join(parts)


def _turn_finished_text(
    event: RuntimeUIEvent,
    tool_invocations: int = 0,
    memory_count: int = 0,
) -> str:
    status = _string_payload(event, "status")
    reason = _string_payload(event, "reason")
    elapsed = event.payload.get("elapsed_seconds")
    elapsed_text = _elapsed_text(elapsed)
    if status and status != "completed":
        label = _status_label(status)
        if reason:
            label = f"{label} {reason}"
        base = f"{label} · {elapsed_text}" if elapsed_text else label
    else:
        base = elapsed_text or "Elapsed --"
    summary = _turn_summary(tool_invocations, memory_count)
    return f"{base}, {summary}" if summary else base


def _elapsed_text(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return f"Elapsed {value:.1f}s"
    return ""


def _status_label(status: str) -> str:
    return {
        "incomplete": "Incomplete",
        "failed": "Failed",
        "cancelled": "Cancelled",
    }.get(status, status)


def _absolute_path(value: str, workspace_root: Path | None = None) -> str:
    path = Path(value)
    if path.is_absolute():
        return normalize_windows_drive_letters(str(path))
    root = workspace_root if workspace_root is not None else Path.cwd()
    return normalize_windows_drive_letters(str((root / path).resolve()))


def _absolute_pattern(value: str, workspace_root: Path | None = None) -> str:
    path = Path(value)
    if path.is_absolute():
        return normalize_windows_drive_letters(str(path))
    root = workspace_root if workspace_root is not None else Path.cwd()
    return normalize_windows_drive_letters(str(root / path))


def _optional_string(value: object) -> str:
    return value if isinstance(value, str) and value else ""


def _json_preview(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    except TypeError:
        return str(value)
