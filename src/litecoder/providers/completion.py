"""Shared text completion collection and bounded retry handling."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace

from litecoder.common.errors import ErrorCode, LiteCoderError
from litecoder.common.errors.classifier import ErrorClassifier
from litecoder.common.errors.retry import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MODEL_CONTINUATION_MAX_ATTEMPTS,
    MODEL_RETRY_BASE_DELAY,
    MODEL_RETRY_MAX_ATTEMPTS,
    MODEL_RETRY_MAX_DELAY,
    RetryBudget,
    next_output_max_tokens,
)
from litecoder.providers.base import ModelProvider
from litecoder.providers.models import ModelRequest, StopReason


RequestFactory = Callable[[int], ModelRequest]
RetrySleep = Callable[[float], Awaitable[object]]

_DEFAULT_CONTINUATION_PROMPT = (
    "Continue the previous response from exactly where it stopped. "
    "Return only the remaining continuation; do not repeat text already emitted, "
    "and preserve the original response format."
)


@dataclass(frozen=True, slots=True)
class TextCompletionResult:
    """Collected text and terminal provider state for one logical call."""

    text: str = field(repr=False)
    stop_reason: StopReason | None
    provider_error: LiteCoderError | None = None
    attempts: int = 1
    requested_max_tokens: int = 0


async def complete_text_with_retry(
    provider: ModelProvider,
    request_factory: RequestFactory,
    *,
    initial_max_tokens: int,
    max_output_tokens: int | None = None,
    retry_budget: RetryBudget | None = None,
    retry_empty: bool = True,
    retry_incomplete: bool = True,
    continuation_prompt: str = _DEFAULT_CONTINUATION_PROMPT,
    sleep: RetrySleep = asyncio.sleep,
) -> TextCompletionResult:
    """Complete a text-only model request with shared retry semantics.

    Provider transport errors marked retryable and empty/incomplete responses
    retry with the current output limit.  A max-token response preserves the
    emitted prefix, asks the provider for only the continuation, and retries
    with a larger limit once, bounded by ``max_output_tokens``.  The final
    result is returned to the caller so each domain can preserve its own
    fallback or validation behavior.
    """
    _validate_positive_int(initial_max_tokens, "initial_max_tokens")
    if max_output_tokens is None:
        max_output_tokens = max(DEFAULT_MAX_OUTPUT_TOKENS, initial_max_tokens)
    _validate_positive_int(max_output_tokens, "max_output_tokens")
    if max_output_tokens < initial_max_tokens:
        raise ValueError("max_output_tokens must not be below initial_max_tokens")
    if not isinstance(retry_empty, bool) or not isinstance(retry_incomplete, bool):
        raise ValueError("retry flags must be boolean")
    if not isinstance(continuation_prompt, str) or not continuation_prompt.strip():
        raise ValueError("continuation_prompt must be a non-empty string")

    budget = retry_budget or RetryBudget(
        max_attempts=MODEL_RETRY_MAX_ATTEMPTS,
        base_delay=MODEL_RETRY_BASE_DELAY,
        max_delay=MODEL_RETRY_MAX_DELAY,
    )
    current_max_tokens = initial_max_tokens
    max_tokens_expanded = False
    max_token_retries = 0
    attempts = 0
    accumulated_text = ""

    while True:
        attempts += 1
        try:
            request = request_factory(current_max_tokens)
            if accumulated_text:
                request = _with_continuation(
                    request,
                    accumulated_text,
                    continuation_prompt,
                )
            result = await _collect_text(
                provider,
                request,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            provider_error = ErrorClassifier().classify(error)
            if not provider_error.retryable:
                raise
            result = TextCompletionResult("", None, provider_error)
        result = replace(
            result,
            attempts=attempts,
            requested_max_tokens=current_max_tokens,
        )

        next_max_tokens = current_max_tokens
        should_retry = False
        if result.provider_error is not None:
            if accumulated_text and result.text:
                accumulated_text += result.text
                result = replace(result, text=accumulated_text)
            should_retry = result.provider_error.retryable
        elif result.stop_reason is StopReason.MAX_TOKENS:
            accumulated_text += result.text
            result = replace(result, text=accumulated_text)
            if max_token_retries < MODEL_CONTINUATION_MAX_ATTEMPTS:
                if not max_tokens_expanded:
                    next_max_tokens = next_output_max_tokens(
                        current_max_tokens,
                        cap=max_output_tokens,
                    )
                    should_retry = next_max_tokens > current_max_tokens
                    if should_retry:
                        max_tokens_expanded = True
                else:
                    should_retry = True
                if should_retry:
                    max_token_retries += 1
        elif result.stop_reason is StopReason.END_TURN:
            empty_response = not result.text.strip()
            if accumulated_text and not empty_response:
                result = replace(
                    result,
                    text=accumulated_text + result.text,
                )
            should_retry = retry_empty and empty_response
        elif result.stop_reason in {None, StopReason.UNKNOWN}:
            if accumulated_text:
                result = replace(
                    result,
                    text=accumulated_text + result.text,
                )
            should_retry = retry_incomplete

        if not should_retry:
            return result

        decision = budget.consume("model_call")
        if not decision.allowed:
            return result
        if decision.delay_seconds:
            await sleep(decision.delay_seconds)
        current_max_tokens = next_max_tokens


def _with_continuation(
    request: ModelRequest,
    accumulated_text: str,
    continuation_prompt: str,
) -> ModelRequest:
    """Resume a truncated text response without discarding its prefix."""
    messages = [
        *request.messages,
        {
            "role": "assistant",
            "content": [{"type": "text", "text": accumulated_text}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": continuation_prompt}],
        },
    ]
    return replace(request, messages=messages)


async def _collect_text(
    provider: ModelProvider,
    request: ModelRequest,
) -> TextCompletionResult:
    completed: dict[int, str] = {}
    deltas: list[str] = []
    stop_reason: StopReason | None = None

    async for event in provider.stream(request):
        if event.type == "provider.error":
            return TextCompletionResult(
                _render_text(completed, deltas),
                stop_reason,
                event.error
                or LiteCoderError(
                    ErrorCode.INTERNAL,
                    "Provider error",
                    retryable=False,
                ),
            )
        if event.type == "text.delta" and isinstance(event.delta, str):
            deltas.append(event.delta)
        elif (
            event.type == "content.completed"
            and event.index is not None
            and isinstance(event.block, dict)
            and event.block.get("type") == "text"
            and isinstance(event.block.get("text"), str)
        ):
            completed[event.index] = event.block["text"]
        elif event.type == "response.completed":
            stop_reason = event.stop_reason
    return TextCompletionResult(_render_text(completed, deltas), stop_reason)


def _render_text(completed: dict[int, str], deltas: list[str]) -> str:
    if completed:
        return "".join(completed[index] for index in sorted(completed))
    return "".join(deltas)


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
