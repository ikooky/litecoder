"""Provider adapter compatibility helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import inspect
import json
import re
from typing import Any

from litecoder.common.errors import ErrorCode, LiteCoderError
from litecoder.providers._json import JsonValue, snapshot_mapping
from litecoder.providers.models import Usage


class InvalidProviderData(ValueError):
    """Component responsible for the invalid provider data."""
    pass


def field_value(value: object, name: str, default: object = None) -> object:
    """Handle the field value operation."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def plain_mapping(value: object, field_name: str) -> dict[str, JsonValue]:
    """Handle the plain mapping operation."""
    if isinstance(value, Mapping):
        candidate = value
    else:
        model_dump = getattr(value, "model_dump", None)
        if not callable(model_dump):
            raise InvalidProviderData(f"{field_name} is not publicly serializable")
        try:
            candidate = model_dump(mode="json")
        except TypeError:
            candidate = model_dump()
    try:
        return snapshot_mapping(candidate, field_name)
    except ValueError as error:
        raise InvalidProviderData(f"{field_name} contains unsupported values") from error


def require_index(value: object) -> int:
    """Handle the require index operation."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidProviderData("provider content index is invalid")
    return value


def require_text(value: object, field_name: str) -> str:
    """Handle the require text operation."""
    if not isinstance(value, str):
        raise InvalidProviderData(f"{field_name} is invalid")
    return value


def require_identity(value: object, field_name: str) -> str:
    """Handle the require identity operation."""
    text = require_text(value, field_name)
    if not text.strip():
        raise InvalidProviderData(f"{field_name} is invalid")
    return text


def parse_tool_input(value: str) -> dict[str, JsonValue]:
    """Parse the tool input."""
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as error:
        raise InvalidProviderData("tool arguments are malformed") from error
    if not isinstance(parsed, dict):
        raise InvalidProviderData("tool arguments must be a JSON object")
    try:
        return snapshot_mapping(parsed, "tool arguments")
    except ValueError as error:
        raise InvalidProviderData("tool arguments contain unsupported values") from error


@asynccontextmanager
async def managed_async_stream(stream: Any) -> AsyncIterator[Any]:
    """Handle the managed async stream operation."""
    enter = getattr(stream, "__aenter__", None)
    exit_stream = getattr(stream, "__aexit__", None)
    if callable(enter) and callable(exit_stream):
        active = await enter()
        try:
            yield active
        except BaseException as primary:
            try:
                await exit_stream(type(primary), primary, primary.__traceback__)
            except BaseException:
                pass
            raise
        else:
            await exit_stream(None, None, None)
        return

    try:
        yield stream
    except BaseException:
        try:
            await _close_stream(stream)
        except BaseException:
            pass
        raise
    else:
        await _close_stream(stream)


async def _close_stream(stream: Any) -> None:
    """Close the stream."""
    close = getattr(stream, "aclose", None)
    if not callable(close):
        close = getattr(stream, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def invalid_tool_arguments_error(reason: str | None = None) -> LiteCoderError:
    """Handle the invalid tool arguments error operation."""
    details: dict[str, JsonValue] = {
        "provider_error_type": "invalid_tool_arguments",
    }
    if reason:
        details["provider_data_reason"] = reason
    return LiteCoderError(
        ErrorCode.PROVIDER_INVALID_RESPONSE,
        "Provider returned invalid tool arguments",
        retryable=True,
        details=details,
    )


def invalid_stream_error(
    kind: str = "invalid_provider_data", reason: str | None = None
) -> LiteCoderError:
    """Handle the invalid stream error operation."""
    details: dict[str, JsonValue] = {"provider_error_type": kind}
    if reason:
        details["provider_data_reason"] = reason
    return LiteCoderError(
        ErrorCode.PROVIDER_INVALID_RESPONSE,
        "Provider returned invalid streaming data",
        retryable=True,
        details=details,
    )


def classify_provider_error(error: Exception) -> LiteCoderError:
    """Classify the provider error."""
    structured = _structured_error_facts(error)
    status = structured.status
    fingerprint = " ".join([type(error).__name__, *structured.labels]).lower()
    details: dict[str, JsonValue] = {
        "provider_error_type": _safe_exception_kind(type(error).__name__)
    }
    if status is not None:
        details["status"] = status

    if status == 429 or "rate_limit" in fingerprint or "ratelimit" in fingerprint:
        return LiteCoderError(
            ErrorCode.PROVIDER_RATE_LIMIT,
            "Provider rate limit exceeded",
            retryable=True,
            details=details,
        )
    if _is_context_error(fingerprint):
        return LiteCoderError(
            ErrorCode.CONTEXT_OVERFLOW,
            "Provider context window exceeded",
            retryable=False,
            details=details,
        )
    if (
        status is not None
        and 500 <= status <= 599
        or any(part in fingerprint for part in ("timeout", "connection", "connecterror"))
    ):
        return LiteCoderError(
            ErrorCode.PROVIDER_TRANSIENT,
            "Provider request failed temporarily",
            retryable=True,
            details=details,
        )
    return LiteCoderError(
        ErrorCode.INTERNAL,
        "Provider request failed",
        retryable=False,
        details=details,
    )


@dataclass(slots=True)
class _ErrorFacts:
    """Data model representing the error facts."""
    status: int | None = None
    labels: list[str] = field(default_factory=list)


def _structured_error_facts(error: Exception) -> _ErrorFacts:
    facts = _ErrorFacts(status=_safe_status(field_value(error, "status_code")))
    direct_code = field_value(error, "code")
    if isinstance(direct_code, str):
        facts.labels.append(direct_code[:256])
    sources: list[object] = [
        field_value(error, "error"),
        field_value(error, "body"),
        field_value(error, "response"),
    ]
    response = field_value(error, "response")
    response_status = _safe_status(field_value(response, "status_code"))
    if facts.status is None:
        facts.status = response_status
    response_json = getattr(response, "json", None)
    if callable(response_json):
        try:
            parsed = response_json()
        except Exception:
            parsed = None
        if not inspect.isawaitable(parsed):
            sources.append(parsed)
    seen: set[int] = set()
    for source in sources:
        _collect_error_facts(source, facts, seen, depth=0, budget=[64])
    return facts


def _collect_error_facts(
    value: object,
    facts: _ErrorFacts,
    seen: set[int],
    *,
    depth: int,
    budget: list[int],
) -> None:
    if value is None or depth > 4 or budget[0] <= 0:
        return
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        for key, item in value.items():
            if budget[0] <= 0:
                break
            budget[0] -= 1
            key_text = key.lower() if isinstance(key, str) else ""
            if facts.status is None and key_text in {"status", "status_code", "http_status"}:
                facts.status = _safe_status(item)
            if key_text in {"code", "type", "error_type", "kind"} and isinstance(item, str):
                facts.labels.append(item[:256])
            if isinstance(item, (Mapping, list)):
                _collect_error_facts(
                    item, facts, seen, depth=depth + 1, budget=budget
                )
        return
    if isinstance(value, list):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        for item in value[:16]:
            if budget[0] <= 0:
                break
            budget[0] -= 1
            _collect_error_facts(item, facts, seen, depth=depth + 1, budget=budget)


def _safe_status(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


_SAFE_KIND = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


def _safe_exception_kind(value: str) -> str:
    lowered = value.lower()
    if (
        not _SAFE_KIND.fullmatch(value)
        or any(part in lowered for part in ("secret", "api_key", "apikey", "token", "sk_"))
    ):
        return "ProviderError"
    return value


def _is_context_error(fingerprint: str) -> bool:
    return (
        "context_length" in fingerprint
        or "context_window" in fingerprint
        or "context_overflow" in fingerprint
        or "too_many_tokens" in fingerprint
        or "max_tokens_exceeded" in fingerprint
    )


@dataclass(slots=True)
class UsageAccumulator:
    """Data model representing the usage accumulator."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    extensions: dict[str, int] = field(default_factory=dict)
    seen: bool = False

    def update(
        self,
        *,
        input_tokens: object = None,
        output_tokens: object = None,
        cache_read_tokens: object = None,
        cache_creation_tokens: object = None,
        extensions: Mapping[str, object] | None = None,
    ) -> Usage | None:
        """Update the stored state."""
        before = self.current if self.seen else None
        supplied = False
        for name, value in (
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
            ("cache_read_tokens", cache_read_tokens),
            ("cache_creation_tokens", cache_creation_tokens),
        ):
            if value is None:
                continue
            supplied = True
            count = _count(value, name)
            current = getattr(self, name)
            setattr(self, name, count if current is None else max(current, count))
        if extensions:
            for name, value in extensions.items():
                if len(self.extensions) >= 32 and name not in self.extensions:
                    break
                count = _optional_count(value)
                if count is None or not name:
                    continue
                supplied = True
                self.extensions[name] = max(self.extensions.get(name, 0), count)
        if not supplied:
            return None
        self.seen = True
        current = self.current
        return current if before != current else None

    @property
    def current(self) -> Usage:
        """Return the active context."""
        return Usage(
            self.input_tokens,
            self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens,
            extensions=self.extensions,
        )


def _count(value: object, name: str) -> int:
    count = _optional_count(value)
    if count is None:
        raise InvalidProviderData(f"provider usage {name} is invalid")
    return count


def _optional_count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None
