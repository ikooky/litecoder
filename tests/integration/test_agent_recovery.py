from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.agent.loop import AgentLoop, RuntimeBudgets
from litecoder.common.errors import ErrorCode, LiteCoderError
from litecoder.common.errors.recovery import RecoveryPolicy
from litecoder.common.errors.retry import RetryBudget
from litecoder.context.compaction import CompactionResult
from litecoder.context.manager import ContextManager
from litecoder.context.session.models import MessageRecord, SessionRecord
from litecoder.context.session.store import SQLiteSessionStore
from litecoder.hooks import HookManager
from litecoder.providers.models import ProviderEvent, StopReason, Usage
from litecoder.tools.models import ToolCall, ToolResult
from litecoder.tools.registry import ToolRegistry
from tests.fakes.provider import FakeProvider


class Duplicates:
    async def start_user_message(self, agent_session_id: str) -> None:
        pass


class Executor:
    async def execute(self, call: ToolCall, context: object) -> ToolResult:
        raise AssertionError("no tool call expected")


class RecordingTrace:
    def __init__(self) -> None:
        self.facts: list[dict[str, object]] = []

    async def record(self, fact: object) -> None:
        assert isinstance(fact, dict)
        self.facts.append(dict(fact))


class CompactingContext:
    can_compact = True

    def __init__(self, store: SQLiteSessionStore) -> None:
        self.store = store
        self.compacted: list[str] = []

    async def build_request(self, session_id: str, tools: ToolRegistry):
        from litecoder.context.manager import ContextManager

        return await ContextManager(self.store, model="model").build_request(
            session_id, tools
        )

    async def compact(self, session_id: str) -> object:
        self.compacted.append(session_id)
        return object()

    async def compact_reactively(self, session_id: str) -> object:
        return await self.compact(session_id)


class FailingDiagnosticsContext(CompactingContext):
    def consume_memory_diagnostics(self) -> tuple[dict[str, object], ...]:
        raise RuntimeError("diagnostics failed")


def answer_round(text: str = "ack") -> list[ProviderEvent]:
    return [
        ProviderEvent.content_block_completed(
            0, {"type": "text", "text": text}
        ),
        ProviderEvent.response_completed(
            StopReason.END_TURN,
            "end_turn",
            usage=Usage(3, 1),
        ),
    ]


def provider_error(code: ErrorCode, *, retryable: bool) -> list[ProviderEvent]:
    return [
        ProviderEvent.provider_error(
            LiteCoderError(code, code.value, retryable=retryable)
        )
    ]


def max_tokens_round(text: str) -> list[ProviderEvent]:
    return [
        ProviderEvent.content_block_completed(
            0, {"type": "text", "text": text}
        ),
        ProviderEvent.response_completed(
            StopReason.MAX_TOKENS,
            "max_tokens",
            usage=Usage(3, 1),
        ),
    ]

async def make_store(tmp_path: Path) -> SQLiteSessionStore:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await store.create_session(SessionRecord.new(
        "session-1",
        "project-1",
        "workspace-1",
        "fake",
        "model",
        workspace_path=str(tmp_path),
    ))
    return store


@pytest.mark.asyncio
async def test_memory_diagnostics_are_optional_but_failures_propagate(
    tmp_path: Path,
) -> None:
    store = await make_store(tmp_path)
    optional_provider = FakeProvider([answer_round("without diagnostics")])
    optional_loop = AgentLoop(
        store=store,
        provider=optional_provider,
        context=CompactingContext(store),
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=Duplicates(),
        budgets=RuntimeBudgets(max_rounds=1, max_tokens=100),
    )

    try:
        result = await optional_loop.run_turn("session-1", "optional diagnostics")
        failing_provider = FakeProvider([answer_round("unused")])
        failing_loop = AgentLoop(
            store=store,
            provider=failing_provider,
            context=FailingDiagnosticsContext(store),
            tools=ToolRegistry(),
            executor=Executor(),
            duplicates=Duplicates(),
            budgets=RuntimeBudgets(max_rounds=1, max_tokens=100),
        )
        with pytest.raises(RuntimeError, match="diagnostics failed"):
            await failing_loop.run_turn("session-1", "broken diagnostics")
    finally:
        await store.close()

    assert result.status == "completed"
    assert len(optional_provider.requests) == 1
    assert failing_provider.requests == []


@pytest.mark.asyncio
async def test_retryable_provider_error_retries_with_bounded_budget(
    tmp_path: Path,
) -> None:
    store = await make_store(tmp_path)
    provider = FakeProvider([
        provider_error(ErrorCode.PROVIDER_RATE_LIMIT, retryable=True),
        answer_round("recovered"),
    ])
    sleeps: list[float] = []
    loop = AgentLoop(
        store=store,
        provider=provider,
        context=CompactingContext(store),
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=Duplicates(),
        recovery_policy=RecoveryPolicy(
            RetryBudget(max_attempts=1, base_delay=0.0)
        ),
        recovery_sleep=sleeps.append,
        budgets=RuntimeBudgets(max_rounds=3, max_tokens=100),
    )

    try:
        result = await loop.run_turn("session-1", "recover")
    finally:
        await store.close()

    assert result.status == "completed"
    assert result.reason == "end_turn"
    assert len(provider.requests) == 2
    assert [message["role"] for message in provider.requests[1].messages] == [
        "user"
    ]
    assert sleeps == [0.0]


@pytest.mark.parametrize(
    "code",
    [ErrorCode.PROVIDER_TRANSIENT, ErrorCode.PROVIDER_RATE_LIMIT],
)
@pytest.mark.asyncio
async def test_provider_transport_defaults_retry_five_times_with_backoff(
    tmp_path: Path,
    code: ErrorCode,
) -> None:
    store = await make_store(tmp_path)
    provider = FakeProvider([
        *[provider_error(code, retryable=True) for _ in range(5)],
        answer_round("recovered"),
    ])
    sleeps: list[float] = []
    loop = AgentLoop(
        store=store,
        provider=provider,
        context=CompactingContext(store),
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=Duplicates(),
        recovery_policy=RecoveryPolicy(),
        recovery_sleep=sleeps.append,
        budgets=RuntimeBudgets(max_rounds=1, max_tokens=100),
    )

    try:
        result = await loop.run_turn("session-1", "recover with backoff")
    finally:
        await store.close()

    assert result.status == "completed"
    assert len(provider.requests) == 6
    assert sleeps == [0.5, 1.0, 2.0, 4.0, 8.0]


@pytest.mark.asyncio
async def test_invalid_provider_response_retries_with_ephemeral_feedback_and_trace(
    tmp_path: Path,
) -> None:
    store = await make_store(tmp_path)
    invalid = LiteCoderError(
        ErrorCode.PROVIDER_INVALID_RESPONSE,
        "Provider returned invalid tool arguments",
        retryable=True,
        details={
            "provider_error_type": "invalid_tool_arguments",
            "provider_data_reason": "tool arguments are malformed",
        },
    )
    provider = FakeProvider([
        [ProviderEvent.provider_error(invalid, request_id="bad-request")],
        answer_round("repaired"),
    ])
    trace = RecordingTrace()
    loop = AgentLoop(
        store=store,
        provider=provider,
        context=CompactingContext(store),
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=Duplicates(),
        recovery_policy=RecoveryPolicy(),
        hooks=HookManager(trace_hook=trace),
        trace_recorder=trace,
        budgets=RuntimeBudgets(max_rounds=1, max_tokens=100),
    )

    try:
        result = await loop.run_turn("session-1", "repair the tool call")
        restored = await store.load_context("session-1")
    finally:
        await store.close()

    assert result.status == "completed"
    assert len(provider.requests) == 2
    assert "previous model response could not be processed" in str(
        provider.requests[1].system
    )
    assert "tool arguments are malformed" not in str(provider.requests[1])
    assert [message.role for message in restored.messages] == ["user", "assistant"]
    retry_fact = next(
        fact for fact in trace.facts
        if fact.get("event") == "provider.runtime"
        and fact.get("status") == "retrying"
    )
    assert retry_fact["failure_origin"] == "provider_response"
    assert retry_fact["failure_code"] == "malformed_tool_arguments"
    assert retry_fact["recovery_strategy"] == "retry_with_feedback"
    assert retry_fact["request_id"] == "bad-request"
    assert retry_fact["attempt"] == 1
    assert retry_fact["max_attempts"] == 2
    recovered_fact = next(
        fact for fact in trace.facts
        if fact.get("event") == "provider.runtime"
        and fact.get("status") == "recovered"
    )
    assert recovered_fact["failure_code"] == "malformed_tool_arguments"
    assert recovered_fact["recovery_strategy"] == "retry_with_feedback"


@pytest.mark.asyncio
async def test_repeated_invalid_response_fingerprint_stops_after_one_repair(
    tmp_path: Path,
) -> None:
    store = await make_store(tmp_path)
    provider = FakeProvider([
        [ProviderEvent.response_completed(StopReason.END_TURN, "end_turn")],
        [ProviderEvent.response_completed(StopReason.END_TURN, "end_turn")],
    ])
    loop = AgentLoop(
        store=store,
        provider=provider,
        context=CompactingContext(store),
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=Duplicates(),
        recovery_policy=RecoveryPolicy(),
        budgets=RuntimeBudgets(max_rounds=1, max_tokens=100),
    )

    try:
        result = await loop.run_turn("session-1", "do not return empty")
    finally:
        await store.close()

    assert result.status == "incomplete"
    assert result.reason == "provider response repair budget exhausted"
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_retry_does_not_persist_partial_failed_assistant(
    tmp_path: Path,
) -> None:
    store = await make_store(tmp_path)
    provider = FakeProvider([
        [
            ProviderEvent.content_block_completed(
                0, {"type": "text", "text": "partial before error"}
            ),
            *provider_error(ErrorCode.PROVIDER_TRANSIENT, retryable=True),
        ],
        answer_round("recovered"),
    ])
    loop = AgentLoop(
        store=store,
        provider=provider,
        context=CompactingContext(store),
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=Duplicates(),
        recovery_policy=RecoveryPolicy(
            RetryBudget(max_attempts=1, base_delay=0.0)
        ),
        recovery_sleep=lambda delay: None,
        budgets=RuntimeBudgets(max_rounds=3, max_tokens=100),
    )

    try:
        result = await loop.run_turn("session-1", "recover")
        restored = await store.load_context("session-1")
    finally:
        await store.close()

    assert result.status == "completed"
    assert len(provider.requests) == 2
    assert [message["role"] for message in provider.requests[1].messages] == [
        "user"
    ]
    assert "partial before error" not in str(provider.requests[1].messages)
    assert [message.role for message in restored.messages] == [
        "user",
        "assistant",
    ]
    assert restored.messages[-1].content == [
        {"type": "text", "text": "recovered"}
    ]


@pytest.mark.asyncio
async def test_context_overflow_compacts_before_retry(tmp_path: Path) -> None:
    store = await make_store(tmp_path)
    context = CompactingContext(store)
    provider = FakeProvider([
        provider_error(ErrorCode.CONTEXT_OVERFLOW, retryable=False),
        answer_round("after compact"),
    ])
    loop = AgentLoop(
        store=store,
        provider=provider,
        context=context,
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=Duplicates(),
        recovery_policy=RecoveryPolicy(
            RetryBudget(max_attempts=1, base_delay=0.0)
        ),
        recovery_sleep=lambda delay: None,
        budgets=RuntimeBudgets(max_rounds=3, max_tokens=100),
    )

    result = await loop.run_turn("session-1", "compact")

    assert result.status == "completed"
    assert context.compacted == ["session-1"]
    assert len(provider.requests) == 2
    await store.close()


@pytest.mark.asyncio
async def test_runtime_context_manager_enables_compaction_defaults(
    tmp_path: Path,
) -> None:
    store = await make_store(tmp_path)
    context = ContextManager(store, model="model")
    provider = FakeProvider([
        provider_error(ErrorCode.CONTEXT_OVERFLOW, retryable=False),
        answer_round("after reactive compact"),
    ])
    loop = AgentLoop(
        store=store,
        provider=provider,
        context=context,
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=Duplicates(),
        recovery_policy=RecoveryPolicy(
            RetryBudget(max_attempts=1, base_delay=0.0)
        ),
        recovery_sleep=lambda delay: None,
        budgets=RuntimeBudgets(max_rounds=3, max_tokens=100),
    )

    try:
        result = await loop.run_turn("session-1", "too much context")
    finally:
        await store.close()

    assert result.status == "completed"
    assert context.can_compact is True
    assert context.context_budget_tokens == 128_000
    assert context.max_tokens == 8_000
    assert [request.max_tokens for request in provider.requests] == [8_000, 8_000]
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_retry_budget_exhaustion_stops_incomplete(tmp_path: Path) -> None:
    store = await make_store(tmp_path)
    provider = FakeProvider([
        provider_error(ErrorCode.PROVIDER_TRANSIENT, retryable=True),
        provider_error(ErrorCode.PROVIDER_TRANSIENT, retryable=True),
    ])
    loop = AgentLoop(
        store=store,
        provider=provider,
        context=CompactingContext(store),
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=Duplicates(),
        recovery_policy=RecoveryPolicy(
            RetryBudget(max_attempts=1, base_delay=0.0)
        ),
        recovery_sleep=lambda delay: None,
        budgets=RuntimeBudgets(max_rounds=4, max_tokens=100),
    )

    result = await loop.run_turn("session-1", "recover once")
    restored = await store.load_context("session-1")

    assert result.status == "incomplete"
    assert result.reason == "provider_transient retry budget exhausted"
    assert len(provider.requests) == 2
    assert [message.role for message in restored.messages] == ["user"]
    await store.close()

@pytest.mark.asyncio
async def test_repeated_context_overflow_stops_when_recovery_budget_exhausts(
    tmp_path: Path,
) -> None:
    store = await make_store(tmp_path)
    context = CompactingContext(store)
    provider = FakeProvider([
        [ProviderEvent.response_completed(
            StopReason.CONTEXT_EXHAUSTED, "context_exhausted"
        )],
        [ProviderEvent.response_completed(
            StopReason.CONTEXT_EXHAUSTED, "context_exhausted"
        )],
    ])
    loop = AgentLoop(
        store=store,
        provider=provider,
        context=context,
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=Duplicates(),
        recovery_policy=RecoveryPolicy(
            RetryBudget(max_attempts=1, base_delay=0.0)
        ),
        recovery_sleep=lambda delay: None,
        budgets=RuntimeBudgets(max_rounds=5, max_tokens=100),
    )

    result = await loop.run_turn("session-1", "compact once")

    assert result.status == "incomplete"
    assert result.reason == "context_overflow retry budget exhausted"
    assert context.compacted == ["session-1"]
    assert len(provider.requests) == 2
    await store.close()
@pytest.mark.asyncio
async def test_max_tokens_resubmits_within_one_logical_round(tmp_path: Path) -> None:
    store = await make_store(tmp_path)
    provider = FakeProvider([
        max_tokens_round("discarded initial truncation"),
        answer_round("complete after expansion"),
    ])
    loop = AgentLoop(
        store=store,
        provider=provider,
        context=CompactingContext(store),
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=Duplicates(),
        budgets=RuntimeBudgets(max_rounds=1, max_tokens=100),
    )

    try:
        result = await loop.run_turn("session-1", "answer fully")
        restored = await store.load_context("session-1")
    finally:
        await store.close()

    assert result.status == "completed"
    assert [request.max_tokens for request in provider.requests] == [8_000, 64_000]
    assert [message.role for message in restored.messages] == ["user", "assistant"]
    assert "discarded initial truncation" not in str(restored.messages)

@pytest.mark.asyncio
async def test_max_tokens_stops_after_three_atomic_continuations(
    tmp_path: Path,
) -> None:
    store = await make_store(tmp_path)
    provider = FakeProvider([
        max_tokens_round("discarded initial truncation"),
        *[max_tokens_round("continued truncation") for _ in range(4)],
    ])
    loop = AgentLoop(
        store=store,
        provider=provider,
        context=CompactingContext(store),
        tools=ToolRegistry(),
        executor=Executor(),
        duplicates=Duplicates(),
        budgets=RuntimeBudgets(max_rounds=1, max_tokens=100),
    )

    try:
        result = await loop.run_turn("session-1", "answer fully")
        restored = await store.load_context("session-1")
    finally:
        await store.close()

    assert result.status == "incomplete"
    assert result.reason == "continuation budget exhausted"
    assert [request.max_tokens for request in provider.requests] == [8_000] + [64_000] * 4
    assert [message.role for message in restored.messages] == [
        "user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant"
    ]


@pytest.mark.asyncio
async def test_reactive_compaction_uses_reduced_context_budget(tmp_path: Path) -> None:
    class RecordingPolicy:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int, bool]] = []

        async def compact(
            self,
            messages: list[MessageRecord],
            budget_tokens: int,
            summarizer: object,
            *,
            summary_budget_tokens: int,
            force_summary: bool,
        ) -> CompactionResult:
            self.calls.append((budget_tokens, summary_budget_tokens, force_summary))
            return CompactionResult(messages)

    store = await make_store(tmp_path)
    policy = RecordingPolicy()
    manager = ContextManager(
        store, model="model", compaction_policy=policy, context_budget_tokens=50_000
    )
    try:
        await manager.compact_reactively("session-1")
    finally:
        await store.close()

    assert policy.calls == [(33_333, 33_333, True)]
