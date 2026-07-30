"""Supporting implementation for command audit."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from litecoder.common.locks import NamedFileLock
from litecoder.common.trace import SecretRedactor, TraceRecorder
from litecoder.paths import AppPaths


class CommandAuditRecorder:
    """Component responsible for the command audit recorder."""
    def __init__(
        self,
        paths: AppPaths,
        redactor: SecretRedactor,
    ) -> None:
        self.paths = paths
        self.redactor = redactor

    @property
    def path(self) -> Path:
        """Handle the path operation."""
        return self.paths.command_audit_path

    def operation(
        self,
        command: str,
        arguments: list[str],
        session_id: str | None,
        root_session_id: str | None,
    ) -> CommandAuditOperation:
        """Handle the operation operation."""
        return CommandAuditOperation(
            recorder=self,
            command_id=uuid.uuid4().hex,
            command=command,
            session_id=session_id,
            root_session_id=root_session_id,
            attributes=_command_attributes(command, arguments),
            started=time.monotonic(),
        )

    async def record(self, payload: Mapping[str, object]) -> None:
        """Record the supplied event or payload."""
        write = asyncio.create_task(self._record(payload))
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(write)
                break
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
                if write.done():
                    write.result()
                    break
        if cancellation is not None:
            raise cancellation

    async def _record(self, payload: Mapping[str, object]) -> None:
        lock = NamedFileLock.command_audit(
            self.paths.project_id,
            self.paths.lock_dir,
        )
        async with lock.acquired_async():
            recorder = TraceRecorder(self.path, self.redactor)
            await recorder.start()
            try:
                await recorder.record_and_flush(payload)
            finally:
                await recorder.close()

    def status(self) -> str:
        """Return the current status."""
        path = self.path
        if not path.exists():
            return f"Command audit: path={path} status=missing"
        try:
            with path.open("r", encoding="utf-8") as handle:
                events = sum(1 for _ in handle)
        except (OSError, UnicodeError) as error:
            raise RuntimeError("Command audit is unavailable") from error
        return f"Command audit: path={path} status=present events={events}"


@dataclass(slots=True)
class CommandAuditOperation:
    """Data model representing the command audit operation."""
    recorder: CommandAuditRecorder
    command_id: str
    command: str
    session_id: str | None
    root_session_id: str | None
    attributes: dict[str, object]
    started: float

    async def start(self) -> None:
        """Start the managed runtime."""
        await self.recorder.record(
            self._event(
                "local.command.start",
                attributes=self.attributes,
            )
        )

    async def finish(
        self,
        *,
        status: str,
        code: str,
        outcome: str | None,
        message: str,
        exit_requested: bool,
        clear_requested: bool,
        replacement_session_id: str | None,
    ) -> None:
        """Finish the managed process and collect its result."""
        session_id_after = (
            None
            if clear_requested
            else replacement_session_id or self.session_id
        )
        attributes: dict[str, object] = {
            "code": code,
            "exit_requested": exit_requested,
            "clear_requested": clear_requested,
        }
        if outcome is not None:
            attributes["outcome"] = outcome
        if status in {"failed", "rejected"} and message:
            attributes["message"] = _bounded_audit_message(
                self.recorder.redactor.redact_text(message)
            )
        if replacement_session_id is not None:
            attributes["replacement_session_id"] = replacement_session_id
        await self.recorder.record(
            self._event(
                "local.command.end",
                status=status,
                session_id_after=session_id_after,
                attributes=attributes,
            )
        )

    async def cancel(self) -> None:
        """Cancel the pending operation."""
        await self._record_best_effort(
            status="cancelled",
            attributes={"code": "cancelled"},
        )

    async def fail(self, error: Exception) -> None:
        """Mark the task as failed."""
        await self._record_best_effort(
            status="failed",
            attributes={
                "code": "unexpected_error",
                "error_type": type(error).__name__,
            },
        )

    async def _record_best_effort(
        self,
        *,
        status: str,
        attributes: dict[str, object],
    ) -> None:
        try:
            await self.recorder.record(
                self._event(
                    "local.command.end",
                    status=status,
                    session_id_after=self.session_id,
                    attributes=attributes,
                )
            )
        except Exception:
            return

    def _event(
        self,
        event: str,
        *,
        status: str | None = None,
        session_id_after: str | None = None,
        attributes: dict[str, object],
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "event": event,
            "timestamp": datetime.now(UTC).isoformat(),
            "command_id": self.command_id,
            "command": self.command,
            "session_id": self.session_id,
            "root_session_id": self.root_session_id,
            "project_id": self.recorder.paths.project_id,
            "workspace_id": self.recorder.paths.workspace_id,
            "attributes": attributes,
        }
        if status is not None:
            payload["status"] = status
            payload["duration_ms"] = max(
                0,
                round((time.monotonic() - self.started) * 1_000),
            )
            payload["session_id_after"] = session_id_after
        return payload


def _bounded_audit_message(value: str, limit: int = 1_000) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="ignore")


def _command_attributes(
    command: str,
    arguments: list[str],
) -> dict[str, object]:
    attributes: dict[str, object] = {"argument_count": len(arguments)}
    if command == "/model":
        if arguments:
            attributes["provider"] = arguments[0]
        if len(arguments) > 1:
            attributes["model"] = arguments[1]
    elif command == "/memory" and arguments:
        attributes["memory_name"] = arguments[0]
    elif command == "/tasks" and arguments:
        attributes["task_id"] = arguments[0]
    return attributes
