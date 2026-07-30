"""Built-in filesystem tools."""

from __future__ import annotations

from litecoder.tools.builtin._common import (
    MAX_FILE_BYTES,
    decode_utf8_text,
    optional_bool,
    optional_int,
    require_string,
    resolve_workspace_path,
    truncate_utf8,
)
from litecoder.tools.builtin.secure_path import secure_read_file, secure_write_file
from litecoder.tools.models import (
    ToolCall,
    ToolContext,
    ToolDenied,
    ToolExecution,
    ToolFailure,
    ToolSpec,
)


_PATH_SCHEMA = {"type": "string", "minLength": 1}


class ReadFileTool:
    """Component responsible for the read file tool."""
    spec = ToolSpec(
        "read_file",
        "Read a workspace-relative UTF-8 file after identifying a relevant path; use offsets for focused inspection.",
        {
            "type": "object",
            "properties": {
                "path": _PATH_SCHEMA,
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        False,
        concurrency="shared",
        permission_risk="safe",
    )

    def hard_guard(self, call: ToolCall, context: ToolContext) -> str | None:
        """Apply the tool safety guard before execution."""
        return _path_guard(context, call.arguments.get("path"), require_leaf=True)

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        relative, data = secure_read_file(
            context.workspace_root,
            call.arguments.get("path"),
            max_bytes=MAX_FILE_BYTES,
        )
        size = len(data)
        if size > MAX_FILE_BYTES:
            raise ToolFailure(
                "Workspace file exceeds the safe read limit",
                metadata={"size": size, "max_size": MAX_FILE_BYTES},
            )
        text = decode_utf8_text(data, safe_message="Workspace file is not UTF-8 text")
        offset = optional_int(call.arguments, "offset", 0, minimum=0)
        limit_value = call.arguments.get("limit")
        limit = (
            None
            if limit_value is None
            else optional_int(
                call.arguments, "limit", 1, minimum=1, maximum=100_000
            )
        )
        lines = text.splitlines(keepends=True)
        selected = lines[offset:] if limit is None else lines[offset : offset + limit]
        rendered = context.redactor.redact_text("".join(selected))
        rendered, output_truncated = truncate_utf8(rendered)
        line_end = offset + len(selected)
        metadata = {
            "path": relative,
            "size": size,
            "line_offset": offset,
            "line_start": offset + 1 if selected else 0,
            "line_end": line_end if selected else 0,
            "total_lines": len(lines),
            "truncated": offset > 0 or line_end < len(lines) or output_truncated,
            "changed_workspace": False,
        }
        return ToolExecution.success(rendered, metadata=metadata, preview=rendered)


class WriteFileTool:
    """Component responsible for the write file tool."""
    spec = ToolSpec(
        "write_file",
        "Create or replace a workspace-relative UTF-8 file atomically. Use for a new file or deliberate full replacement; inspect existing files first and prefer edit_file for a localized change.",
        {
            "type": "object",
            "properties": {"path": _PATH_SCHEMA, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        True,
        concurrency="exclusive",
        permission_risk="workspace",
    )

    def hard_guard(self, call: ToolCall, context: ToolContext) -> str | None:
        """Apply the tool safety guard before execution."""
        return _path_guard(context, call.arguments.get("path"), require_leaf=True)

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        content = require_string(call.arguments, "content", allow_empty=True)
        payload = content.encode("utf-8")
        relative, changed = secure_write_file(
            context.workspace_root, call.arguments.get("path"), payload
        )
        metadata = {
            "path": relative,
            "size": len(payload),
            "changed_workspace": changed,
        }
        return ToolExecution.success(
            "Wrote workspace file" if changed else "Workspace file is unchanged",
            metadata=metadata,
            changed_workspace=changed,
            preview={"path": relative, "changed": changed},
        )


class EditFileTool:
    """Component responsible for the edit file tool."""
    spec = ToolSpec(
        "edit_file",
        "Make a targeted exact replacement in an existing workspace-relative UTF-8 file. Read the current content first, preserve surrounding conventions, and use replace_all only when every occurrence is intended.",
        {
            "type": "object",
            "properties": {
                "path": _PATH_SCHEMA,
                "old_text": {"type": "string", "minLength": 1},
                "new_text": {"type": "string"},
                "replace_all": {"type": "boolean", "default": False},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
        True,
        concurrency="exclusive",
        permission_risk="workspace",
    )

    def hard_guard(self, call: ToolCall, context: ToolContext) -> str | None:
        """Apply the tool safety guard before execution."""
        return _path_guard(context, call.arguments.get("path"), require_leaf=True)

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolExecution:
        """Execute the requested tool call."""
        relative, data = secure_read_file(
            context.workspace_root,
            call.arguments.get("path"),
            max_bytes=MAX_FILE_BYTES,
        )
        old_text = require_string(call.arguments, "old_text")
        new_text = require_string(call.arguments, "new_text", allow_empty=True)
        replace_all = optional_bool(call.arguments, "replace_all", False)
        if len(data) > MAX_FILE_BYTES:
            raise ToolFailure(
                "Workspace file exceeds the safe edit limit",
                metadata={"size": len(data), "max_size": MAX_FILE_BYTES},
            )
        text = decode_utf8_text(data, safe_message="Workspace file is not UTF-8 text")
        occurrences = text.count(old_text)
        if occurrences == 0:
            raise ToolFailure(
                "Edit text was not found",
                metadata={"occurrences": 0, "changed_workspace": False},
            )
        if not replace_all and occurrences != 1:
            raise ToolFailure(
                "Edit text is ambiguous",
                metadata={"occurrences": occurrences, "changed_workspace": False},
            )
        replacements = occurrences if replace_all else 1
        updated = text.replace(old_text, new_text, -1 if replace_all else 1)
        _, changed = secure_write_file(
            context.workspace_root, call.arguments.get("path"), updated.encode("utf-8")
        )
        metadata = {
            "path": relative,
            "occurrences": occurrences,
            "replacements": replacements,
            "changed_workspace": changed,
        }
        return ToolExecution.success(
            "Edited workspace file" if changed else "Workspace file is unchanged",
            metadata=metadata,
            changed_workspace=changed,
            preview={"path": relative, "replacements": replacements},
        )


def _path_guard(
    context: ToolContext, value: object, *, require_leaf: bool
) -> str | None:
    try:
        resolve_workspace_path(context.workspace_root, value, require_leaf=require_leaf)
    except ToolDenied as error:
        return error.safe_message
    return None
