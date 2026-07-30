"""UI presentation state and formatting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from litecoder.ui.events import RuntimeUIEvent, UIEventType
from litecoder.ui.presenters import (
    event_tool_arguments,
    event_tool_key,
    failure_detail_lines,
    memory_count_from_payload,
    memory_diagnostic_text,
    optional_text,
    success_detail_lines,
    tool_title,
    turn_finished_text,
)


class BlockKind(StrEnum):
    """Enumeration of the block kind values."""
    USER = "user"
    ASSISTANT = "assistant"
    THINKING = "thinking"
    TOOL = "tool"
    TODO = "todo"
    COMMAND_OUTPUT = "command_output"
    ERROR = "error"
    NOTICE = "notice"
    SUMMARY = "summary"


class ToolVisualState(StrEnum):
    """Enumeration of the tool visual state values."""
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_PERMISSION = "waiting_permission"
    SUCCESS = "success"
    ERROR = "error"
    DENIED = "denied"


class NoticeLevel(StrEnum):
    """Enumeration of the notice level values."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class TodoViewItem:
    """Data model representing the todo view item."""
    content: str
    active_form: str
    status: str


@dataclass(slots=True)
class TranscriptBlock:
    """Data model representing the transcript block."""
    key: str
    kind: BlockKind
    text: str = ""
    title: str = ""
    detail: tuple[str, ...] = ()
    todos: tuple[TodoViewItem, ...] = ()
    status: str = ""
    streaming: bool = False
    visible: bool = True


@dataclass(slots=True)
class NoticeView:
    """Data model representing the notice view."""
    key: str
    text: str
    level: NoticeLevel = NoticeLevel.INFO
    persistent: bool = False


@dataclass(slots=True)
class LiveTurnState:
    """Data model representing the live turn state."""
    active: bool = False
    phase: str = "idle"
    assistant_text: str = ""
    thinking_text: str = ""
    tools: tuple[TranscriptBlock, ...] = ()
    todos: tuple[TodoViewItem, ...] = ()
    todo_dirty: bool = False
    todo_carried: bool = False
    provider_error: TranscriptBlock | None = None
    notice: NoticeView | None = None

    @property
    def visible(self) -> bool:
        """Return whether the item is visible."""
        return bool(
            self.active
            or self.assistant_text
            or self.thinking_text
            or self.tools
            or self.todos
            or self.provider_error
            or self.notice
        )


@dataclass(slots=True)
class SessionViewState:
    """Data model representing the session view state."""
    blocks: list[TranscriptBlock] = field(default_factory=list)
    live: LiveTurnState = field(default_factory=LiveTurnState)
    current_todos: tuple[TodoViewItem, ...] = ()
    tool_invocations: int = 0
    memory_count: int = 0
    usage: dict[str, object] = field(default_factory=dict)


class PresentationReducer:
    """Commit completed output to history; keep in-flight output at the tail."""

    def __init__(self, *, workspace_root: str = "") -> None:
        self.workspace_root = workspace_root
        self.state = SessionViewState()
        self._live_tools: dict[str, TranscriptBlock] = {}
        self._tool_names: dict[str, str] = {}
        self._tool_arguments: dict[str, dict[str, object]] = {}
        self._restore_tools: dict[str, TranscriptBlock] = {}
        self._turn_number = 0
        self._serial = 0

    def clear(self, *, clear_todos: bool = True) -> None:
        """Clear the requested operation."""
        todos = () if clear_todos else self.state.current_todos
        self.state = SessionViewState(current_todos=todos)
        self._live_tools.clear()
        self._tool_names.clear()
        self._tool_arguments.clear()
        self._restore_tools.clear()
        self._turn_number = 0
        self._serial = 0

    def start_turn_preview(self) -> None:
        """Start the turn preview."""
        self._start_turn(increment=False)

    def reset_live(self, *, phase: str = "idle") -> None:
        """Handle the reset live operation."""
        self._live_tools.clear()
        self.state.live = LiveTurnState(phase=phase)

    def clear_live_notice(self, key: str) -> None:
        """Clear the live notice."""
        notice = self.state.live.notice
        if notice is not None and notice.key == key and not notice.persistent:
            self.state.live.notice = None

    def add_user_prompt(self, text: str) -> TranscriptBlock:
        """Add the user prompt."""
        return self._append(TranscriptBlock(self._next_key("user"), BlockKind.USER, text=text))

    def add_assistant_history(self, text: str) -> TranscriptBlock:
        """Add the assistant history."""
        return self._append(
            TranscriptBlock(self._next_key("assistant-history"), BlockKind.ASSISTANT, text=text)
        )

    def add_thinking_history(self, text: str) -> TranscriptBlock:
        """Add the thinking history."""
        return self._append(
            TranscriptBlock(
                self._next_key("thinking-history"),
                BlockKind.THINKING,
                title="Thinking",
                detail=tuple(text.splitlines()),
                status="completed",
            )
        )

    def add_todo_history(self, value: object) -> TranscriptBlock | None:
        """Add the todo history."""
        parsed = (
            tuple(value)
            if isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and all(isinstance(item, TodoViewItem) for item in value)
            else parse_todos(value)
        )
        if not parsed:
            return None
        self.state.current_todos = parsed if _has_open_todos(parsed) else ()
        return self._append(
            TranscriptBlock(
                self._next_key("todo-history"),
                BlockKind.TODO,
                todos=parsed,
                status="completed",
            )
        )

    def add_notice_block(self, text: str, *, level: str = "info") -> TranscriptBlock:
        """Add the notice block."""
        return self._append(
            TranscriptBlock(
                self._next_key("notice"),
                BlockKind.NOTICE,
                text=text,
                status=level,
            )
        )

    def add_command_output(self, text: str) -> TranscriptBlock:
        """Add the command output."""
        return self._append(
            TranscriptBlock(
                self._next_key("command-output"),
                BlockKind.COMMAND_OUTPUT,
                text=text,
            )
        )

    def add_error_block(self, title: str, detail: Sequence[str]) -> TranscriptBlock:
        """Add the error block."""
        return self._append(
            TranscriptBlock(
                self._next_key("error"),
                BlockKind.ERROR,
                title=title,
                detail=tuple(detail),
                status="error",
            )
        )

    def add_tool_history(
        self, call_id: str, name: str, arguments: Mapping[str, object]
    ) -> TranscriptBlock:
        """Add the tool history."""
        block = TranscriptBlock(
            f"tool-history:{call_id}",
            BlockKind.TOOL,
            title=tool_title(name, arguments, workspace_root=self.workspace_root),
            status=ToolVisualState.QUEUED.value,
            visible=name != "todo_write",
        )
        self._restore_tools[call_id] = block
        return self._append(block)

    def finish_tool_history(
        self, call_id: str, *, content: str, failed: bool
    ) -> TranscriptBlock | None:
        """Handle the finish tool history operation."""
        block = self._restore_tools.get(call_id)
        if block is None:
            return None
        block.status = (
            ToolVisualState.ERROR.value if failed else ToolVisualState.SUCCESS.value
        )
        block.detail = tuple(content.splitlines()) or (("Done",) if not failed else ())
        return block

    def apply(self, event: RuntimeUIEvent) -> SessionViewState:
        """Apply the presentation update."""
        event_type = event.type
        live = self.state.live
        if event_type is UIEventType.TURN_STARTED:
            self._start_turn(increment=True)
        elif event_type is UIEventType.MODEL_REQUESTED:
            live.phase = "waiting"
            count = memory_count_from_payload(event.payload)
            if count is not None:
                self.state.memory_count = max(self.state.memory_count, count)
        elif event_type is UIEventType.THINKING_STARTED:
            self._mark_recovered("thinking")
        elif event_type is UIEventType.THINKING_DELTA:
            self._mark_recovered("thinking")
            live.thinking_text += _payload_text(event.payload.get("text"))
        elif event_type is UIEventType.THINKING_COMPLETED:
            text = _payload_text(event.payload.get("text")) or live.thinking_text
            if text:
                self._append(
                    TranscriptBlock(
                        self._event_key("thinking", event),
                        BlockKind.THINKING,
                        title="Thinking",
                        detail=tuple(text.splitlines()),
                        status="completed",
                    )
                )
            live.thinking_text = ""
        elif event_type is UIEventType.ASSISTANT_DELTA:
            self._mark_recovered("responding")
            live.assistant_text += _payload_text(event.payload.get("text"))
        elif event_type is UIEventType.ASSISTANT_COMPLETED:
            self._mark_recovered("responding")
            text = _payload_text(event.payload.get("text")) or live.assistant_text
            if text:
                self._append(
                    TranscriptBlock(
                        self._event_key("assistant", event),
                        BlockKind.ASSISTANT,
                        text=text,
                    )
                )
            live.assistant_text = ""
        elif event_type in {
            UIEventType.TOOL_CALL_STARTED,
            UIEventType.TOOL_CALL_INPUT_DELTA,
            UIEventType.TOOL_CALL_COMPLETED,
        }:
            self._mark_recovered("tool")
            self._ensure_live_tool(event)
        elif event_type is UIEventType.PERMISSION_REQUESTED:
            block = self._ensure_live_tool(event)
            block.status = ToolVisualState.WAITING_PERMISSION.value
            reason = optional_text(event.payload.get("reason"))
            block.detail = (reason,) if reason else ("Waiting for permission…",)
            live.phase = "permission"
            self._sync_live_tools()
        elif event_type is UIEventType.PERMISSION_RESOLVED:
            block = self._ensure_live_tool(event)
            allowed = event.payload.get("allowed") is True
            block.status = (
                ToolVisualState.QUEUED.value
                if allowed
                else ToolVisualState.DENIED.value
            )
            reason = optional_text(event.payload.get("reason"))
            block.detail = (
                ("Permission granted",)
                if allowed
                else (reason or "Permission denied",)
            )
            live.phase = "tool"
            self._sync_live_tools()
        elif event_type is UIEventType.TOOL_EXECUTION_STARTED:
            block = self._ensure_live_tool(event)
            block.status = ToolVisualState.RUNNING.value
            block.detail = ("Running…",)
            live.phase = "tool"
            self._sync_live_tools()
        elif event_type is UIEventType.TOOL_EXECUTION_FINISHED:
            self._finish_live_tool(event, ToolVisualState.SUCCESS, success_detail_lines(event))
        elif event_type is UIEventType.TOOL_EXECUTION_FAILED:
            self._finish_live_tool(event, ToolVisualState.ERROR, failure_detail_lines(event))
        elif event_type is UIEventType.TOOL_EXECUTION_DENIED:
            self._finish_live_tool(
                event,
                ToolVisualState.DENIED,
                failure_detail_lines(event, denied=True),
            )
        elif event_type is UIEventType.TODO_UPDATED:
            parsed = parse_todos(event.payload.get("todos"))
            if parsed is not None:
                self.state.current_todos = parsed if _has_open_todos(parsed) else ()
                if live.active:
                    live.todos = parsed
                    live.todo_dirty = True
                    live.todo_carried = False
        elif event_type is UIEventType.PROVIDER_ERROR:
            self._provider_error(event)
        elif event_type is UIEventType.NOTICE_RAISED:
            self._notice(event)
        elif event_type is UIEventType.DIAGNOSTIC:
            text = optional_text(event.payload.get("message")) or memory_diagnostic_text(
                event.payload
            )
            if text:
                self.add_notice_block(text)
        elif event_type is UIEventType.USAGE_UPDATED:
            self.state.usage = dict(event.payload)
        elif event_type is UIEventType.MODEL_COMPLETED:
            live.phase = "finalizing"
        elif event_type is UIEventType.TURN_FINISHED:
            self._finish_turn(event)
        return self.state

    def _start_turn(self, *, increment: bool) -> None:
        if increment:
            self._turn_number += 1
        self._live_tools.clear()
        self._tool_names.clear()
        self._tool_arguments.clear()
        carried_todos = tuple(
            todo for todo in self.state.current_todos if todo.status != "completed"
        )
        self.state.live = LiveTurnState(
            active=True,
            phase="waiting",
            todos=carried_todos,
            todo_carried=bool(carried_todos),
        )
        self.state.tool_invocations = 0
        self.state.memory_count = 0
        self.state.usage = {}

    def _mark_recovered(self, phase: str) -> None:
        self.state.live.provider_error = None
        self.state.live.phase = phase

    def _ensure_live_tool(self, event: RuntimeUIEvent) -> TranscriptBlock:
        key = event_tool_key(event)
        name = event.tool_name or self._tool_names.get(key, "tool")
        if name != "tool":
            self._tool_names[key] = name
        arguments = event_tool_arguments(event)
        if arguments:
            self._tool_arguments[key] = dict(arguments)
        block = self._live_tools.get(key)
        if block is None:
            block = TranscriptBlock(
                f"live-tool:{key}",
                BlockKind.TOOL,
                status=ToolVisualState.QUEUED.value,
                visible=name != "todo_write",
            )
            self._live_tools[key] = block
            self.state.tool_invocations += 1
        known_name = self._tool_names.get(key, name)
        block.title = tool_title(
            known_name,
            self._tool_arguments.get(key, {}),
            workspace_root=self.workspace_root,
        )
        block.visible = known_name != "todo_write"
        self._sync_live_tools()
        return block

    def _finish_live_tool(
        self,
        event: RuntimeUIEvent,
        status: ToolVisualState,
        detail: Sequence[str],
    ) -> None:
        key = event_tool_key(event)
        block = self._ensure_live_tool(event)
        if block.visible:
            self._append(
                TranscriptBlock(
                    self._event_key("tool", event),
                    BlockKind.TOOL,
                    title=block.title,
                    detail=tuple(detail),
                    status=status.value,
                )
            )
        self._live_tools.pop(key, None)
        self._tool_names.pop(key, None)
        self._tool_arguments.pop(key, None)
        self._sync_live_tools()
        self.state.live.phase = "tool"

    def _sync_live_tools(self) -> None:
        self.state.live.tools = tuple(
            block for block in self._live_tools.values() if block.visible
        )

    def _provider_error(self, event: RuntimeUIEvent) -> None:
        code = optional_text(event.payload.get("code"))
        message = optional_text(event.payload.get("message"))
        retryable = event.payload.get("retryable") is True
        retrying_value = event.payload.get("retrying")
        retrying = retryable if retrying_value is None else retrying_value is True
        attempt = _non_negative_int(event.payload.get("attempt"))
        maximum = _non_negative_int(event.payload.get("max_attempts"))
        if retrying:
            progress = (
                f" ({attempt}/{maximum})"
                if attempt is not None and maximum is not None
                else ""
            )
            detail = [f"Retrying...{progress}"]
        else:
            detail = [message or code or "Provider request failed"]
            if retryable and attempt is not None and maximum is not None:
                detail.append(f"Retries exhausted ({attempt}/{maximum})")
        self.state.live.provider_error = TranscriptBlock(
            self._next_key("live-provider-error"),
            BlockKind.ERROR,
            title="Retrying" if retrying else "API Error",
            detail=tuple(detail),
            status="retrying" if retrying else "error",
        )
        self.state.live.phase = "retrying" if retrying else "error"

    def _notice(self, event: RuntimeUIEvent) -> None:
        text = optional_text(event.payload.get("message"))
        if not text:
            return
        try:
            level = NoticeLevel(optional_text(event.payload.get("level")) or "info")
        except ValueError:
            level = NoticeLevel.INFO
        persistent = event.payload.get("persistent") is True
        notice = NoticeView(self._next_key("notice"), text, level, persistent)
        if persistent:
            self._append(
                TranscriptBlock(
                    notice.key,
                    BlockKind.NOTICE,
                    text=text,
                    status=level.value,
                )
            )
        else:
            self.state.live.notice = notice

    def _finish_turn(self, event: RuntimeUIEvent) -> None:
        live = self.state.live
        if live.thinking_text:
            self.add_thinking_history(live.thinking_text)
        if live.assistant_text:
            self.add_assistant_history(live.assistant_text)
        status = optional_text(event.payload.get("status"))
        if live.provider_error is not None and status != "completed":
            error = live.provider_error
            self.add_error_block(error.title, error.detail)
        if live.todo_dirty and live.todos:
            self._append(
                TranscriptBlock(
                    self._next_key("todo"),
                    BlockKind.TODO,
                    todos=live.todos,
                    status="completed",
                )
            )
        self._append(
            TranscriptBlock(
                self._next_key("turn-summary"),
                BlockKind.SUMMARY,
                text=turn_finished_text(
                    event,
                    self.state.tool_invocations,
                    self.state.memory_count,
                ),
                status=status,
            )
        )
        self.reset_live()

    def _event_key(self, prefix: str, event: RuntimeUIEvent) -> str:
        identity = (
            event.request_id
            or event.tool_call_id
            or f"{self._turn_number}:{event.sequence}"
        )
        return f"{prefix}:{identity}:{self._next_key('event')}"

    def _next_key(self, prefix: str) -> str:
        self._serial += 1
        return f"{prefix}:{self._serial}"


    def _append(self, block: TranscriptBlock) -> TranscriptBlock:
        self.state.blocks.append(block)
        return block


def _has_open_todos(todos: Sequence[TodoViewItem]) -> bool:
    return any(todo.status != "completed" for todo in todos)


def _payload_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def parse_todos(value: object) -> tuple[TodoViewItem, ...] | None:
    """Parse the todos."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    items: list[TodoViewItem] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            return None
        content = optional_text(raw.get("content"))
        active_form = optional_text(raw.get("active_form"))
        status = optional_text(raw.get("status"))
        if (
            not content
            or not active_form
            or status not in {"pending", "in_progress", "completed"}
        ):
            return None
        items.append(TodoViewItem(content, active_form, status))
    return tuple(items)
