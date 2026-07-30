"""Durable trace recording and serialization."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from litecoder.common.trace.redaction import SecretRedactor


_STOP: Final = object()
_RECOVERY_EVENT: Final = "trace.recovery"
_RECORD_ERROR: Final = "trace record must be UTF-8 JSON-compatible"


@dataclass(slots=True)
class _QueuedRecord:
    """Data model representing the queued record."""
    data: bytes
    durable: bool = False
    completion: asyncio.Future[None] | None = None


def recover_last_sequence(path: Path) -> tuple[int, bytes | None]:
    """Handle the recover last sequence operation."""
    if not path.exists():
        return 0, None

    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    if not lines:
        return 0, None

    for index in range(len(lines) - 1, -1, -1):
        try:
            sequence = _read_sequence(lines[index])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue

        retained = b"".join(lines[: index + 1])
        incomplete = b"".join(lines[index + 1 :])
        if incomplete:
            _truncate_to(path, retained)
        elif not retained.endswith((b"\n", b"\r")):
            with path.open("ab") as stream:
                stream.write(b"\n")
        return sequence, incomplete or None

    _truncate_to(path, b"")
    return 0, content


def _truncate_to(path: Path, retained: bytes) -> None:
    with path.open("r+b") as stream:
        stream.seek(0)
        stream.write(retained)
        stream.truncate()


def _read_sequence(line: bytes) -> int:
    row = json.loads(line)
    if not isinstance(row, dict):
        raise TypeError("trace row must be an object")
    sequence = row["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("trace sequence must be a non-negative integer")
    return sequence


class TraceRecorder:
    """Component responsible for the trace recorder."""
    def __init__(self, path: Path, redactor: SecretRedactor) -> None:
        self.path = path
        self.redactor = redactor
        self._queue: asyncio.Queue[_QueuedRecord | object] = asyncio.Queue()
        self._sequence = 0
        self._worker: asyncio.Task[None] | None = None
        self._started = False
        self._closed = False

    async def start(self) -> None:
        """Start the managed runtime."""
        if self._started:
            raise RuntimeError("TraceRecorder is already started")
        if self._closed:
            raise RuntimeError("TraceRecorder is closed")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence, incomplete = recover_last_sequence(self.path)
        self._started = True
        self._worker = asyncio.create_task(self._run())

        if incomplete is not None:
            await self.record(
                {
                    "event": _RECOVERY_EVENT,
                    "reason": "incomplete_trailing_line",
                    "discarded_tail": incomplete.decode("utf-8", errors="replace"),
                }
            )

    async def record(self, payload: Mapping[str, object]) -> None:
        """Record the supplied event or payload."""
        await self._enqueue(payload)

    async def record_and_flush(self, payload: Mapping[str, object]) -> None:
        """Record the and flush."""
        loop = asyncio.get_running_loop()
        completion = loop.create_future()
        await self._enqueue(payload, durable=True, completion=completion)
        worker = self._worker
        if worker is None:
            raise RuntimeError("TraceRecorder worker is unavailable")
        done, _ = await asyncio.wait(
            {completion, worker},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if completion in done:
            completion.result()
            return
        self._raise_if_worker_failed()

    async def _enqueue(
        self,
        payload: Mapping[str, object],
        *,
        durable: bool = False,
        completion: asyncio.Future[None] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("TraceRecorder is closed")
        if not self._started:
            raise RuntimeError("TraceRecorder is not started")

        self._raise_if_worker_failed()
        sequence = self._sequence + 1
        serialized = self._serialize(sequence, payload)
        self._sequence = sequence
        await self._queue.put(
            _QueuedRecord(serialized, durable=durable, completion=completion)
        )

    async def close(self) -> None:
        """Close the managed resource and release any lock."""
        if not self._started:
            raise RuntimeError("TraceRecorder is not started")
        worker = self._worker
        if worker is None:
            raise RuntimeError("TraceRecorder worker is unavailable")

        if not self._closed:
            self._closed = True
            if not worker.done():
                await self._queue.put(_STOP)
        await worker

    def _raise_if_worker_failed(self) -> None:
        worker = self._worker
        if worker is None or not worker.done():
            return
        try:
            failure = worker.exception()
        except asyncio.CancelledError as failure:
            raise RuntimeError("TraceRecorder worker failed") from failure
        if failure is not None:
            raise RuntimeError("TraceRecorder worker failed") from failure
        raise RuntimeError("TraceRecorder worker failed")

    def _serialize(self, sequence: int, payload: Mapping[str, object]) -> bytes:
        row = self.redactor.redact_data({**dict(payload), "sequence": sequence})
        try:
            rendered = json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
            return rendered.encode("utf-8") + b"\n"
        except (TypeError, ValueError, UnicodeEncodeError):
            raise ValueError(_RECORD_ERROR) from None

    async def _run(self) -> None:
        """Run the requested operation."""
        with self.path.open("ab") as stream:
            while True:
                item = await self._queue.get()
                if item is _STOP:
                    return
                assert isinstance(item, _QueuedRecord)
                try:
                    stream.write(item.data)
                    stream.flush()
                    if item.durable:
                        os.fsync(stream.fileno())
                except BaseException as error:
                    if item.completion is not None and not item.completion.done():
                        item.completion.set_exception(error)
                    raise
                if item.completion is not None and not item.completion.done():
                    item.completion.set_result(None)
