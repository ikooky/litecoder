"""Command hook execution and protocol handling."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum

from litecoder.common.trace.redaction import current_secret_redactor
from litecoder.hooks.models import HookDiagnostic, HookEnvelope, HookOutcome, HookPoint
from litecoder.settings import HookCommandSettings


_MAX_COMMAND_INPUT_BYTES = 1024 * 1024
_MAX_COMMAND_OUTPUT_BYTES = 64 * 1024
_READ_CHUNK_BYTES = 4096
_MAX_DIAGNOSTICS = 16
_MAX_DIAGNOSTIC_TEXT = 512


class _CommandHookError(Exception):
    """Raised when the command hook error conditions occur."""
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _CommandOutputTooLarge(Exception):
    """Raised when the command output too large conditions occur."""
    pass


@dataclass(frozen=True, slots=True)
class CommandHook:
    """Run one configured hook command with a bounded JSON-only protocol."""

    settings: HookCommandSettings

    def __post_init__(self) -> None:
        if not isinstance(self.settings, HookCommandSettings):
            raise TypeError("settings must be HookCommandSettings")

    async def __call__(self, envelope: HookEnvelope) -> HookOutcome:
        try:
            request = _request_bytes(envelope)
            stdout = await _run_command(self.settings, request)
            response = _response_object(stdout)
            return _outcome_from_response(envelope, response)
        except _CommandHookError as error:
            return _failed_outcome(envelope, error.code)


def _request_bytes(envelope: HookEnvelope) -> bytes:
    try:
        request = {
            "version": 1,
            "point": envelope.point.value,
            "phase": envelope.phase,
            "hook_id": envelope.hook_id,
            "dispatch_id": envelope.dispatch_id,
            "payload": _json_value(envelope.payload),
        }
        redacted = current_secret_redactor().redact_data(request)
        encoded = json.dumps(
            redacted,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise _CommandHookError("input_invalid") from error
    if len(encoded) > _MAX_COMMAND_INPUT_BYTES:
        raise _CommandHookError("input_too_large")
    return encoded

async def _run_command(settings: HookCommandSettings, request: bytes) -> bytes:
    """Run the command."""
    try:
        process = await asyncio.create_subprocess_exec(
            settings.command,
            *settings.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as error:
        raise _CommandHookError("unavailable") from error

    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    tasks: tuple[asyncio.Task[object], ...] = ()
    try:
        process.stdin.write(request)
        await asyncio.wait_for(process.stdin.drain(), timeout=settings.timeout_seconds)
        process.stdin.close()
        tasks = (
            asyncio.create_task(
                _read_limited(process.stdout, _MAX_COMMAND_OUTPUT_BYTES)
            ),
            asyncio.create_task(
                _read_limited(process.stderr, _MAX_COMMAND_OUTPUT_BYTES)
            ),
            asyncio.create_task(process.wait()),
        )
        stdout, _stderr, returncode = await asyncio.wait_for(
            asyncio.gather(*tasks), timeout=settings.timeout_seconds
        )
    except asyncio.TimeoutError as error:
        await _stop_process(process)
        await _cancel_tasks(tasks)
        raise _CommandHookError("timed_out") from error
    except _CommandOutputTooLarge as error:
        await _stop_process(process)
        await _cancel_tasks(tasks)
        raise _CommandHookError("output_too_large") from error
    except asyncio.CancelledError:
        # A command hook is an external process, so task cancellation alone
        # would otherwise leave it running after the owning agent has stopped.
        await _stop_process(process)
        await _cancel_tasks(tasks)
        raise
    except (BrokenPipeError, ConnectionResetError, OSError) as error:
        await _stop_process(process)
        await _cancel_tasks(tasks)
        raise _CommandHookError("execution_failed") from error
    except Exception as error:
        await _stop_process(process)
        await _cancel_tasks(tasks)
        raise _CommandHookError("execution_failed") from error

    if not isinstance(stdout, bytes) or not isinstance(returncode, int):
        raise _CommandHookError("execution_failed")
    if returncode != 0:
        raise _CommandHookError("execution_failed")
    return stdout


async def _read_limited(stream: asyncio.StreamReader, limit: int) -> bytes:
    """Read the limited."""
    data = bytearray()
    while True:
        chunk = await stream.read(min(_READ_CHUNK_BYTES, limit - len(data) + 1))
        if not chunk:
            return bytes(data)
        data.extend(chunk)
        if len(data) > limit:
            raise _CommandOutputTooLarge


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    """Stop the process."""
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
        return
    except asyncio.TimeoutError:
        pass
    try:
        process.kill()
    except ProcessLookupError:
        return
    await process.wait()


async def _cancel_tasks(tasks: tuple[asyncio.Task[object], ...]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _response_object(stdout: bytes) -> dict[str, object]:
    try:
        text = stdout.decode("utf-8")
        redacted = current_secret_redactor().redact_text(text)
        parsed = json.loads(redacted, parse_constant=_invalid_json_constant)
    except (UnicodeDecodeError, ValueError) as error:
        raise _CommandHookError("invalid_output") from error
    if not isinstance(parsed, dict):
        raise _CommandHookError("invalid_output")
    return parsed


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _outcome_from_response(
    envelope: HookEnvelope, response: Mapping[str, object]
) -> HookOutcome:
    blocked = response.get("blocked", False)
    if type(blocked) is not bool:
        raise _CommandHookError("invalid_output")
    payload = response.get("payload", envelope.payload)
    if envelope.phase == "post" or envelope.point in {
        HookPoint.SUBAGENT_START,
        HookPoint.SUBAGENT_STOP,
    }:
        payload = envelope.payload
    diagnostics = _response_diagnostics(envelope, response.get("diagnostics", []))
    return HookOutcome(payload=payload, blocked=blocked, diagnostics=diagnostics)


def _response_diagnostics(
    envelope: HookEnvelope, value: object
) -> tuple[HookDiagnostic, ...]:
    if not isinstance(value, list) or len(value) > _MAX_DIAGNOSTICS:
        raise _CommandHookError("invalid_output")
    diagnostics: list[HookDiagnostic] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise _CommandHookError("invalid_output")
        kind = _diagnostic_text(item.get("kind"), "notice")
        code = _diagnostic_text(item.get("code"), "command_notice")
        message = _diagnostic_text(
            item.get("message"), "External hook reported a diagnostic."
        )
        diagnostics.append(
            HookDiagnostic(
                hook_id=envelope.hook_id,
                point=envelope.point,
                phase=envelope.phase,
                kind=kind,
                code=code,
                message=message,
            )
        )
    return tuple(diagnostics)


def _diagnostic_text(value: object, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise _CommandHookError("invalid_output")
    return current_secret_redactor().redact_text(value)[:_MAX_DIAGNOSTIC_TEXT]


def _failed_outcome(envelope: HookEnvelope, code: str) -> HookOutcome:
    messages = {
        "input_invalid": "External hook input was invalid.",
        "input_too_large": "External hook input exceeded the size limit.",
        "unavailable": "External hook command could not be started.",
        "execution_failed": "External hook command failed.",
        "timed_out": "External hook command timed out.",
        "output_too_large": "External hook output exceeded the size limit.",
        "invalid_output": "External hook returned invalid JSON output.",
    }
    diagnostic = HookDiagnostic(
        hook_id=envelope.hook_id,
        point=envelope.point,
        phase=envelope.phase,
        kind="error",
        code=f"command_{code}",
        message=messages.get(code, "External hook command failed."),
    )
    return HookOutcome(
        payload=envelope.payload,
        blocked=envelope.phase == "pre",
        diagnostics=(diagnostic,),
    )


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("hook payload contains a non-finite number")
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
            if item.repr
        }
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("hook payload object keys must be strings")
            result[key] = _json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError("hook payload must be JSON compatible")
