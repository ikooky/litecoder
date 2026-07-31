"""Textual UI widgets."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from litecoder.cli.local_commands import LocalCommandSpec
from litecoder.ui.markdown import WrappingMarkdown
from litecoder.ui.presenters import (
    compact_number,
    normalize_windows_drive_letters,
)

from litecoder.tools.permission import PermissionMode
from litecoder.ui.presentation import (
    BlockKind,
    NoticeView,
    SessionViewState,
    TodoViewItem,
    ToolVisualState,
    TranscriptBlock,
)


SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

TODO_COMPLETED_VISIBLE = 3
TODO_PENDING_VISIBLE = 4


class TranscriptBlockWidget(Static):
    """Component responsible for the transcript block widget."""
    def __init__(self, block: TranscriptBlock, *, expanded: bool = False) -> None:
        extra_class = " turn-summary" if block.kind is BlockKind.SUMMARY else ""
        super().__init__(
            "",
            classes=f"message {block.kind.value}-message{extra_class}",
        )
        self.block = block
        self.expanded = expanded
        self.animation_frame = 0
        self.display = block.visible

    def update_block(
        self,
        block: TranscriptBlock,
        *,
        expanded: bool,
        animation_frame: int = 0,
    ) -> None:
        """Update the block."""
        self.block = block
        self.expanded = expanded
        self.animation_frame = animation_frame
        self.display = block.visible
        self.refresh(layout=True)

    def render(self) -> RenderableType:
        """Render the requested operation."""
        renderable = render_transcript_block(
            self.block,
            expanded=self.expanded,
            animation_frame=self.animation_frame,
        )
        if isinstance(renderable, Text):
            return renderable
        width = max(
            1,
            self.content_size.width or self.container_size.width or self.app.size.width,
        )
        return _selectable_text(renderable, self.app.console, width)


def render_command_suggestions(
    commands: Sequence[LocalCommandSpec],
    *,
    selected_name: str | None,
    first_index: int,
    total_count: int,
) -> RenderableType:
    """Render the local-command completion list below the prompt."""
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=28, no_wrap=True)
    table.add_column(ratio=1, overflow="fold")
    for command in commands:
        selected = command.name == selected_name
        command_style = "bold #cdd6f4" if selected else "#b8b8b8"
        detail_style = "#cdd6f4" if selected else "dim"
        label = Text("› " if selected else "  ", style=command_style)
        label.append(command.usage, style=command_style)
        table.add_row(label, Text(command.description, style=detail_style))
    if total_count <= len(commands):
        return table
    last_index = first_index + len(commands)
    range_text = Text(
        f"  {first_index + 1}-{last_index} / {total_count:,}", style="dim"
    )
    return Group(table, range_text)


def render_transcript_block(
    block: TranscriptBlock,
    *,
    expanded: bool = False,
    animation_frame: int = 0,
) -> RenderableType:
    """Render the transcript block."""
    if block.kind is BlockKind.USER:
        text = Text("❯ ", style="bold white")
        text.append(normalize_windows_drive_letters(block.text), style="white")
        return text
    if block.kind is BlockKind.ASSISTANT:
        grid = _message_grid()
        grid.add_row(
            Text("●", style="bold white"),
            WrappingMarkdown(normalize_windows_drive_letters(block.text) or " "),
        )
        return grid
    if block.kind is BlockKind.THINKING:
        suffix = "…" if block.streaming else ""
        header = Text(f"∴ Thinking{suffix}", style="dim italic")
        if not block.detail or not (expanded or block.streaming):
            if block.detail:
                header.append("  ctrl+o to expand", style="dim not italic")
            return header
        body = Text(
            normalize_windows_drive_letters("\n".join(block.detail)), style="dim"
        )
        return Group(header, Text("  ").append_text(body))
    if block.kind is BlockKind.TOOL:
        return _render_tool(block, expanded=expanded)
    if block.kind is BlockKind.TODO:
        return render_todos(block.todos, expanded=expanded)
    if block.kind is BlockKind.COMMAND_OUTPUT:
        return Text(normalize_windows_drive_letters(block.text), style="yellow")
    if block.kind is BlockKind.ERROR:
        detail = _limited_detail(block.detail, expanded=expanded, limit=8)
        title = Text("✗ ", style="bold red")
        title.append(
            normalize_windows_drive_letters(block.title or "Error"), style="bold red"
        )
        return Group(title, _detail_text(detail, style="red"))
    if block.kind is BlockKind.NOTICE:
        style = {
            "warning": "yellow",
            "error": "red",
        }.get(block.status, "dim")
        return Text.assemble(
            ("! ", style), (normalize_windows_drive_letters(block.text), style)
        )
    if block.kind is BlockKind.SUMMARY:
        failed = block.status in {"failed", "cancelled"}
        prefix = "✗ " if failed else ""
        style = "red" if failed else "dim"
        return Text(
            f"{prefix}{normalize_windows_drive_letters(block.text)}", style=style
        )
    return Text(normalize_windows_drive_letters(block.text))


def render_live_tail(
    state: SessionViewState,
    *,
    animation_frame: int,
    queued_prompts: Sequence[str],
    expanded: bool = False,
) -> RenderableType:
    """Render the live tail."""
    rows: list[RenderableType] = []
    live = state.live
    if live.assistant_text:
        rows.append(
            render_transcript_block(
                TranscriptBlock(
                    "live-assistant",
                    BlockKind.ASSISTANT,
                    text=live.assistant_text,
                    streaming=True,
                ),
                expanded=expanded,
                animation_frame=animation_frame,
            )
        )
    if live.thinking_text:
        rows.append(
            render_transcript_block(
                TranscriptBlock(
                    "live-thinking",
                    BlockKind.THINKING,
                    title="Thinking",
                    detail=tuple(live.thinking_text.splitlines()),
                    status="streaming",
                    streaming=True,
                ),
                expanded=expanded,
                animation_frame=animation_frame,
            )
        )
    for tool in live.tools:
        rows.append(
            render_transcript_block(
                tool,
                expanded=expanded,
                animation_frame=animation_frame,
            )
        )
    if live.provider_error is not None:
        rows.append(
            render_transcript_block(
                live.provider_error,
                expanded=expanded,
                animation_frame=animation_frame,
            )
        )
    if live.active:
        frame = SPINNER_FRAMES[animation_frame % len(SPINNER_FRAMES)]
        label = {
            "waiting": "Working…",
            "thinking": "Thinking…",
            "responding": "Responding…",
            "tool": "Running tools…",
            "permission": "Waiting for permission…",
            "retrying": "Retrying...",
            "error": "Request failed",
            "finalizing": "Finishing…",
        }.get(live.phase, "Working…")
        style = "red" if live.phase == "error" else "dim"
        activity = Text(f"{frame} {label}", style=style)
        stats = _activity_stats(state)
        if stats:
            activity.append(f"  ·  {' · '.join(stats)}", style="dim")
        rows.append(activity)
    if live.todos:
        rows.append(
            render_todos(
                live.todos,
                expanded=expanded,
                continuing=live.todo_carried,
            )
        )
    if live.notice is not None:
        rows.append(render_notice(live.notice))
    if queued_prompts:
        rows.append(render_queued_prompts(queued_prompts))
    return Group(*rows) if rows else Text("")

def render_todos(
    items: Sequence[TodoViewItem],
    *,
    expanded: bool = False,
    continuing: bool = False,
) -> RenderableType:
    """Render the todos."""
    completed_count = sum(item.status == "completed" for item in items)
    active_form = next(
        (item.active_form for item in items if item.status == "in_progress"),
        None,
    )
    heading = Text()
    heading.append("Tasks", style="bold")
    if continuing:
        heading.append(f"  {len(items)} open · continuing", style="dim")
    else:
        heading.append(f"  {completed_count}/{len(items)} complete", style="dim")
    if active_form:
        heading.append(
            f" · {normalize_windows_drive_letters(active_form)}", style="dim"
        )

    lines: list[RenderableType] = [heading]
    for row in _visible_todo_rows(items, expanded=expanded):
        if isinstance(row, Text):
            lines.append(row)
            continue
        icon, style = {
            "completed": ("■", "green dim"),
            "in_progress": ("■", "bold white"),
            "pending": ("□", "dim"),
        }[row.status]
        lines.append(
            Text(
                f"  {icon} {normalize_windows_drive_letters(row.content)}", style=style
            )
        )
    return Group(*lines)


def _visible_todo_rows(
    items: Sequence[TodoViewItem],
    *,
    expanded: bool,
) -> tuple[TodoViewItem | Text, ...]:
    completed = [item for item in items if item.status == "completed"]
    active = [item for item in items if item.status == "in_progress"]
    pending = [item for item in items if item.status == "pending"]
    if expanded:
        return tuple((*completed, *active, *pending))

    rows: list[TodoViewItem | Text] = []
    hidden_completed = max(0, len(completed) - TODO_COMPLETED_VISIBLE)
    if hidden_completed:
        rows.append(
            Text(
                f"  … {hidden_completed} earlier completed (ctrl+o to expand)",
                style="dim italic",
            )
        )
    rows.extend(completed[-TODO_COMPLETED_VISIBLE:])
    rows.extend(active)
    rows.extend(pending[:TODO_PENDING_VISIBLE])
    hidden_pending = max(0, len(pending) - TODO_PENDING_VISIBLE)
    if hidden_pending:
        rows.append(
            Text(
                f"  … {hidden_pending} later pending (ctrl+o to expand)",
                style="dim italic",
            )
        )
    return tuple(rows)


def render_queued_prompts(prompts: Sequence[str]) -> RenderableType:
    """Render the queued prompts."""
    if not prompts:
        return Text("")
    lines: list[Text] = []
    for prompt in prompts[:3]:
        one_line = normalize_windows_drive_letters(" ".join(prompt.splitlines()))
        if len(one_line) > 100:
            one_line = one_line[:97] + "..."
        lines.append(Text(f"  ↳ {one_line}", style="dim"))
    remaining = len(prompts) - len(lines)
    if remaining:
        lines.append(Text(f"  … and {remaining} more", style="dim"))
    return Group(*lines)


def render_notice(notice: NoticeView | None) -> RenderableType:
    """Render the notice."""
    if notice is None:
        return Text("")
    style = {
        "warning": "yellow",
        "error": "red",
        "info": "dim",
    }[notice.level.value]
    return Text(normalize_windows_drive_letters(notice.text), style=style)


def render_footer(
    *,
    permission_mode: PermissionMode,
    model: str,
    workspace: str,
    width: int,
    usage: dict[str, object],
) -> Text:
    """Render the footer."""
    label, style = {
        PermissionMode.ASK: ("ask", "#f9e2af"),
        PermissionMode.READ_ONLY: ("read-only", "#a6e3a1"),
        PermissionMode.BYPASS: ("bypass", "#f38ba8"),
    }[permission_mode]
    left = Text()
    left.append("⏵⏵ ", style="bold #d8b4fe")
    left.append(label, style=style)
    left.append(" (shift+tab: next mode)", style="dim")
    input_tokens = _integer(usage.get("input_tokens"))
    context_text = (
        f"context {compact_number(input_tokens)}" if input_tokens is not None else ""
    )
    candidates = [
        context_text,
        normalize_windows_drive_letters(model),
        normalize_windows_drive_letters(workspace),
    ]
    for value in candidates:
        if not value:
            continue
        left.append(" · ", style="dim")
        left.append(value, style="dim")
    return left


def startup_banner(workspace: str, model: str) -> RenderableType:
    """Handle the startup banner operation."""
    welcome = Text()
    welcome.append("Welcome to LiteCoder CLI!", style="bold #c084fc")
    welcome.append("\n\n")
    welcome.append(
        f"Workspace: {normalize_windows_drive_letters(workspace)}", style="dim"
    )
    welcome.append("\n")
    welcome.append(f"Model: {normalize_windows_drive_letters(model)}", style="dim")
    return Panel(welcome, border_style="bright_black", padding=(0, 1))


def _render_tool(
    block: TranscriptBlock,
    *,
    expanded: bool,
) -> RenderableType:
    status = block.status or ToolVisualState.QUEUED.value
    icon, style = {
        ToolVisualState.QUEUED.value: ("○", "dim"),
        ToolVisualState.RUNNING.value: ("●", "yellow"),
        ToolVisualState.WAITING_PERMISSION.value: ("●", "yellow"),
        ToolVisualState.SUCCESS.value: ("●", "green"),
        ToolVisualState.ERROR.value: ("●", "red"),
        ToolVisualState.DENIED.value: ("●", "red dim"),
    }.get(status, ("○", "dim"))
    title = Text(f"{icon} ", style=style)
    title.append(normalize_windows_drive_letters(block.title or "Tool"), style="bold")
    if not block.detail:
        return title
    limit = 2
    detail = _limited_detail(block.detail, expanded=expanded, limit=limit)
    detail_style = "red" if status in {"error", "denied"} else "dim"
    return Group(title, _detail_text(detail, style=detail_style))


def _limited_detail(
    lines: Sequence[str],
    *,
    expanded: bool,
    limit: int,
) -> tuple[str, ...]:
    if expanded or len(lines) <= limit:
        return tuple(lines)
    visible_count = max(1, limit - 1)
    hidden = len(lines) - visible_count
    return tuple(lines[:visible_count]) + (f"… +{hidden} lines (ctrl+o to expand)",)


def _detail_text(lines: Sequence[str], *, style: str) -> Text:
    text = Text()
    for index, line in enumerate(lines):
        text.append("  ⎿  " if index == 0 else "\n     ", style="dim")
        text.append(normalize_windows_drive_letters(line), style=style)
    return text


def _message_grid() -> Table:
    grid = Table.grid(expand=True)
    grid.add_column(width=2)
    grid.add_column(ratio=1, overflow="fold")
    return grid


def _selectable_text(
    renderable: RenderableType,
    console: Console,
    width: int,
) -> Text:
    """Flatten a Rich renderable to styled text so Textual can select it."""
    options = console.options.update(width=width, height=None, highlight=False)
    rendered_lines = console.render_lines(renderable, options, pad=False)
    text = Text()
    for line_index, line in enumerate(rendered_lines):
        line_text = Text()
        for segment in line:
            if not segment.control:
                line_text.append(segment.text, style=segment.style)
        line_text.rstrip()
        if line_index:
            text.append("\n")
        text.append_text(line_text)
    return text


def _integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _activity_stats(state: SessionViewState) -> tuple[str, ...]:
    stats: list[str] = []
    if state.tool_invocations:
        stats.append(
            f"{state.tool_invocations} tool"
            f"{'s' if state.tool_invocations != 1 else ''}"
        )
    input_tokens = _integer(state.usage.get("input_tokens")) or 0
    output_tokens = _integer(state.usage.get("output_tokens")) or 0
    total_tokens = input_tokens + output_tokens
    if total_tokens:
        stats.append(f"{compact_number(total_tokens)} tokens")
    if state.memory_count:
        noun = "memory" if state.memory_count == 1 else "memories"
        stats.append(f"{state.memory_count} {noun}")
    return tuple(stats)
