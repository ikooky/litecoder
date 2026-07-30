from __future__ import annotations

from pathlib import Path

import pytest

from litecoder.common.trace import SecretRedactor, bind_secret_redactor
from litecoder.context.compaction import (
    CompactionPolicy,
    SummaryRequest,
    estimate_message_tokens,
)
from litecoder.context.manual_compaction import ManualCompactor
from litecoder.context.manager import ContextManager, ContextStatistics
from litecoder.context.session.models import MessageRecord, SessionRecord
from litecoder.context.session.store import SQLiteSessionStore
from litecoder.tools.registry import ToolRegistry


async def _session(store: SQLiteSessionStore, workspace: Path) -> None:
    await store.create_session(
        SessionRecord.new(
            "session",
            "project",
            "workspace",
            "fake",
            "model",
            workspace_path=str(workspace),
        )
    )


async def _append_text(
    store: SQLiteSessionStore, role: str, text: str
) -> None:
    await store.append_message(
        MessageRecord("session", role, [{"type": "text", "text": text}])
    )


@pytest.mark.asyncio
async def test_context_statistics_count_only_latest_summary_and_uncovered_rows(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await _session(store, tmp_path)
    await _append_text(store, "user", "covered user")
    await _append_text(store, "assistant", "covered assistant")
    await store.append_message(MessageRecord(
        "session",
        "system",
        [{
            "type": "context_summary",
            "covered_through_sequence": 2,
            "text": "retained facts",
        }],
    ))
    await _append_text(store, "user", "uncovered user")

    manager = ContextManager(store, model="model")
    statistics = await manager.statistics("session")
    rows = (await store.load_context("session")).messages

    assert statistics == ContextStatistics(
        persisted_messages=4,
        effective_tokens=estimate_message_tokens([rows[2], rows[3]]),
    )
    await store.close()


@pytest.mark.asyncio
async def test_context_statistics_without_summary_count_all_rows(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await _session(store, tmp_path)
    await _append_text(store, "user", "hello")
    await _append_text(store, "assistant", "world")

    statistics = await ContextManager(store, model="model").statistics("session")
    rows = (await store.load_context("session")).messages

    assert statistics.persisted_messages == 2
    assert statistics.effective_tokens == estimate_message_tokens(rows)
    await store.close()

@pytest.mark.asyncio
async def test_summary_persists_without_rewriting_original_rows(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await _session(store, tmp_path)
    await _append_text(store, "user", "old user " * 100)
    await _append_text(store, "assistant", "old assistant " * 100)
    await _append_text(store, "user", "recent user " * 100)
    before = await store.load_context("session")
    requests: list[SummaryRequest] = []

    async def summarize(request: SummaryRequest) -> str:
        requests.append(request)
        return "facts from old history"

    manager = ContextManager(
        store,
        model="model",
        compaction_policy=CompactionPolicy(keep_recent_tool_rounds=0),
        context_budget_tokens=60,
        summarizer=summarize,
    )
    request = await manager.build_request("session", ToolRegistry())
    after = await store.load_context("session")

    assert len(requests) == 1
    assert requests[0].covered_through_sequence == 3
    assert len(after.messages) == len(before.messages) + 1
    assert [message.content for message in after.messages[:3]] == [
        message.content for message in before.messages
    ]
    summary = after.messages[-1]
    assert summary.role == "system"
    assert summary.content == [{
        "type": "context_summary",
        "covered_through_sequence": 3,
        "text": "facts from old history",
    }]
    assert request.messages == []
    assert any(
        block.get("text") == "facts from old history" for block in request.system
    )
    await store.close()


@pytest.mark.asyncio
async def test_context_manager_exposes_explicit_compaction_entrypoint(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await _session(store, tmp_path)
    await _append_text(store, "user", "old " * 100)
    await _append_text(store, "assistant", "latest " * 100)

    async def summarize(request: SummaryRequest) -> str:
        return "explicit summary"

    manager = ContextManager(
        store,
        model="model",
        compaction_policy=CompactionPolicy(keep_recent_tool_rounds=0),
        context_budget_tokens=60,
        summarizer=summarize,
    )
    result = await manager.compact("session")
    rows = (await store.load_context("session")).messages

    assert result.summary == "explicit summary"
    assert rows[-1].content[0]["type"] == "context_summary"
    assert rows[-1].content[0]["text"] == "explicit summary"
    await store.close()


@pytest.mark.asyncio
async def test_manual_compactor_persists_summary_without_rewriting_original_rows(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await _session(store, tmp_path)
    await _append_text(store, "user", "old user " * 120)
    await _append_text(store, "assistant", "old assistant " * 120)
    await _append_text(store, "user", "recent user " * 20)
    before_context = await store.load_context("session")
    summary_requests: list[SummaryRequest] = []

    async def summarize(request: SummaryRequest) -> str:
        summary_requests.append(request)
        return "retained facts"

    def manager_factory(
        provider: str,
        model: str,
        budget: int,
    ) -> ContextManager:
        assert (provider, model) == ("fake", "model")
        return ContextManager(
            store,
            model=model,
            compaction_policy=CompactionPolicy(keep_recent_tool_rounds=0),
            context_budget_tokens=budget,
            summarizer=summarize,
        )

    report = await ManualCompactor(store, manager_factory).compact("session")
    after_context = await store.load_context("session")
    after_statistics = await ContextManager(
        store, model="model"
    ).statistics("session")

    assert report.before_tokens > report.after_tokens
    assert report.saved_tokens == report.before_tokens - report.after_tokens
    assert report.summary_created is True
    assert report.reason == "compacted"
    assert report.after_tokens == after_statistics.effective_tokens
    assert len(summary_requests) == 2
    assert len(after_context.messages) == len(before_context.messages) + 1
    assert [message.sequence for message in after_context.messages[:-1]] == [
        message.sequence for message in before_context.messages
    ]
    assert [message.role for message in after_context.messages[:-1]] == [
        message.role for message in before_context.messages
    ]
    assert [message.content for message in after_context.messages[:-1]] == [
        message.content for message in before_context.messages
    ]
    assert after_context.messages[-1].content == [{
        "type": "context_summary",
        "covered_through_sequence": summary_requests[-1].covered_through_sequence,
        "text": "retained facts",
    }]
    await store.close()


@pytest.mark.asyncio
async def test_manual_compactor_persists_summary_when_tool_preview_fits_target(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await _session(store, tmp_path)
    await store.append_message(MessageRecord(
        "session",
        "assistant",
        [{
            "type": "tool_call",
            "call_id": "large-result",
            "name": "read",
            "input": {"path": "large.txt"},
        }],
    ))
    await store.append_message(MessageRecord(
        "session",
        "user",
        [{
            "type": "tool_result",
            "tool_call_id": "large-result",
            "status": "success",
            "content": "tool output " * 2_000,
        }],
    ))
    before_context = await store.load_context("session")
    requests: list[SummaryRequest] = []

    async def summarize(request: SummaryRequest) -> str:
        requests.append(request)
        return "persisted tool summary"

    def manager_factory(
        provider: str,
        model: str,
        budget: int,
    ) -> ContextManager:
        return ContextManager(
            store,
            model=model,
            compaction_policy=CompactionPolicy(),
            context_budget_tokens=budget,
            summarizer=summarize,
        )

    report = await ManualCompactor(store, manager_factory).compact("session")
    after_context = await store.load_context("session")

    assert report.before_tokens > report.after_tokens
    assert report.summary_created is True
    assert len(requests) == 1
    assert [message["role"] for message in requests[0].messages] == [
        "assistant",
        "user",
    ]
    assert len(after_context.messages) == len(before_context.messages) + 1
    assert [message.content for message in after_context.messages[:-1]] == [
        message.content for message in before_context.messages
    ]
    assert after_context.messages[-1].content == [{
        "type": "context_summary",
        "covered_through_sequence": 2,
        "text": "persisted tool summary",
    }]
    await store.close()


@pytest.mark.asyncio
async def test_manual_compactor_budgets_against_original_persisted_tool_suffix(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await _session(store, tmp_path)
    await _append_text(store, "user", "short prefix")
    await store.append_message(MessageRecord(
        "session",
        "assistant",
        [{
            "type": "tool_call",
            "call_id": "large-result",
            "name": "read",
            "input": {"path": "large.txt"},
        }],
    ))
    await store.append_message(MessageRecord(
        "session",
        "user",
        [{
            "type": "tool_result",
            "tool_call_id": "large-result",
            "status": "success",
            "content": "tool output " * 2_000,
        }],
    ))
    before_context = await store.load_context("session")
    before_statistics = await ContextManager(
        store, model="model"
    ).statistics("session")
    target = max(64, before_statistics.effective_tokens * 2 // 3)
    requests: list[SummaryRequest] = []

    async def summarize(request: SummaryRequest) -> str:
        requests.append(request)
        return "persisted tool summary"

    def manager_factory(
        provider: str,
        model: str,
        budget: int,
    ) -> ContextManager:
        return ContextManager(
            store,
            model=model,
            compaction_policy=CompactionPolicy(),
            context_budget_tokens=budget,
            summarizer=summarize,
        )

    report = await ManualCompactor(store, manager_factory).compact("session")
    after_context = await store.load_context("session")

    assert [request.covered_through_sequence for request in requests] == [1, 3]
    assert report.before_tokens == before_statistics.effective_tokens
    assert report.after_tokens < report.before_tokens
    assert report.after_tokens <= target
    assert len(after_context.messages) == len(before_context.messages) + 1
    assert [message.content for message in after_context.messages[:-1]] == [
        message.content for message in before_context.messages
    ]
    summary = after_context.messages[-1].content[0]
    assert summary["covered_through_sequence"] == 3
    await store.close()


@pytest.mark.asyncio
async def test_manual_compactor_leaves_short_history_unchanged_without_provider(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await _session(store, tmp_path)
    await _append_text(store, "user", "short")
    before_context = await store.load_context("session")

    def manager_factory(
        provider: str,
        model: str,
        budget: int,
    ) -> ContextManager:
        raise AssertionError("short history must not create a provider manager")

    report = await ManualCompactor(store, manager_factory).compact("session")
    after_context = await store.load_context("session")

    assert report.before_tokens == report.after_tokens
    assert report.summary_created is False
    assert report.reason == "history_too_small"
    assert after_context.messages == before_context.messages
    await store.close()


@pytest.mark.asyncio
async def test_manual_compactor_reports_oversized_existing_summary_unchanged(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await _session(store, tmp_path)
    await _append_text(store, "user", "covered")
    await store.append_message(MessageRecord(
        "session",
        "system",
        [{
            "type": "context_summary",
            "covered_through_sequence": 1,
            "text": "oversized summary " * 200,
        }],
    ))
    before_context = await store.load_context("session")

    def manager_factory(
        provider: str,
        model: str,
        budget: int,
    ) -> ContextManager:
        return ContextManager(
            store,
            model=model,
            compaction_policy=CompactionPolicy(),
            context_budget_tokens=budget,
            summarizer=lambda request: pytest.fail("summarizer not expected"),
        )

    report = await ManualCompactor(store, manager_factory).compact("session")
    after_context = await store.load_context("session")

    assert report.before_tokens == report.after_tokens
    assert report.summary_created is False
    assert report.reason == "existing_summary_over_budget"
    assert after_context.messages == before_context.messages
    await store.close()


@pytest.mark.asyncio
async def test_restore_uses_latest_valid_summary_sequence_and_later_rows(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await _session(store, tmp_path)
    await _append_text(store, "user", "covered one")  # 1
    await _append_text(store, "assistant", "covered two")  # 2
    await store.append_message(MessageRecord(
        "session",
        "system",
        [{
            "type": "context_summary",
            "covered_through_sequence": 2,
            "text": "older summary with higher cutoff",
        }],
    ))  # 3
    await _append_text(store, "user", "still covered by newest")  # 4
    await store.append_message(MessageRecord(
        "session",
        "system",
        [{
            "type": "context_summary",
            "covered_through_sequence": 1,
            "text": "newest valid summary",
        }],
    ))  # 5
    await store.append_message(MessageRecord(
        "session",
        "system",
        [{
            "type": "context_summary",
            "covered_through_sequence": "bad",
            "text": "invalid newest row",
        }],
    ))  # 6
    await _append_text(store, "assistant", "later answer")  # 7

    request = await ContextManager(store, model="model").build_request(
        "session", ToolRegistry()
    )

    system_text = "\n".join(str(block["text"]) for block in request.system)
    replay_text = [
        block["text"]
        for message in request.messages
        for block in message["content"]
        if block.get("type") == "text"
    ]
    assert "newest valid summary" in system_text
    assert "older summary" not in system_text
    assert "invalid newest row" not in system_text
    assert replay_text == ["covered two", "still covered by newest", "later answer"]
    await store.close()


@pytest.mark.asyncio
async def test_restore_after_reopen_keeps_original_rows_and_order(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    first = SQLiteSessionStore(path)
    await first.open()
    await _session(first, tmp_path)
    await _append_text(first, "user", "covered")  # 1
    await _append_text(first, "assistant", "also covered")  # 2
    await first.append_message(MessageRecord(
        "session",
        "system",
        [{
            "type": "context_summary",
            "covered_through_sequence": 2,
            "text": "persisted summary",
        }],
    ))  # 3
    await _append_text(first, "user", "later one")  # 4
    await _append_text(first, "assistant", "later two")  # 5
    await first.close()

    second = SQLiteSessionStore(path)
    await second.open()
    request = await ContextManager(second, model="model").build_request(
        "session", ToolRegistry()
    )
    rows = (await second.load_context("session")).messages

    assert [row.sequence for row in rows] == [1, 2, 3, 4, 5]
    assert [message["role"] for message in request.messages] == [
        "user",
        "assistant",
    ]
    assert [message["content"][0]["text"] for message in request.messages] == [
        "later one",
        "later two",
    ]
    await second.close()


@pytest.mark.asyncio
async def test_malformed_summary_cutoff_is_not_replayed_as_system_text(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await _session(store, tmp_path)
    await store.append_message(MessageRecord(
        "session",
        "system",
        [{
            "type": "context_summary",
            "covered_through_sequence": "not-an-integer",
            "text": "must not replay",
        }],
    ))
    await _append_text(store, "user", "safe later message")

    request = await ContextManager(store, model="model").build_request(
        "session", ToolRegistry()
    )

    assert all(
        "must not replay" not in str(block.get("text", ""))
        for block in request.system
    )
    assert request.messages[0]["content"][0]["text"] == "safe later message"
    await store.close()

@pytest.mark.asyncio
async def test_summary_is_redacted_before_persistence_and_replay(tmp_path: Path) -> None:
    secret = "configured-secret"
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await _session(store, tmp_path)
    await _append_text(store, "user", "old " * 200)
    await _append_text(store, "assistant", "recent " * 100)

    async def summarize(request: SummaryRequest) -> str:
        return f"secret={secret} Authorization: Bearer abc123"

    manager = ContextManager(
        store,
        model="model",
        compaction_policy=CompactionPolicy(keep_recent_tool_rounds=0),
        context_budget_tokens=60,
        summarizer=summarize,
    )
    with bind_secret_redactor(SecretRedactor.with_values((secret,))):
        result = await manager.compact("session")
        request = await manager.build_request("session", ToolRegistry())

    stored = (await store.load_context("session")).messages[-1].content[0]["text"]
    replayed = "\n".join(str(block["text"]) for block in request.system)
    assert result.summary is not None
    assert secret not in result.summary
    assert "abc123" not in result.summary
    assert secret not in stored
    assert "abc123" not in stored
    assert secret not in replayed
    assert "abc123" not in replayed
    assert "[REDACTED]" in stored
    await store.close()


@pytest.mark.asyncio
async def test_repeated_build_does_not_nest_stale_summaries(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await _session(store, tmp_path)
    await _append_text(store, "user", "first " * 100)
    await _append_text(store, "assistant", "second " * 100)
    summaries = iter(("summary one", "summary two"))
    requests: list[SummaryRequest] = []

    async def summarize(request: SummaryRequest) -> str:
        requests.append(request)
        return next(summaries)

    manager = ContextManager(
        store,
        model="model",
        compaction_policy=CompactionPolicy(keep_recent_tool_rounds=0),
        context_budget_tokens=60,
        summarizer=summarize,
    )
    await manager.build_request("session", ToolRegistry())
    await _append_text(store, "user", "third " * 100)
    request = await manager.build_request("session", ToolRegistry())

    system_text = "\n".join(str(block["text"]) for block in request.system)
    assert "summary two" in system_text
    assert "summary one" not in system_text
    assert "summary one" in repr(requests[1].messages)
    assert sum(
        "summary" in str(block.get("text", "")) for block in request.system
    ) == 1
    assert all(message["role"] != "system" for message in request.messages)
    await store.close()


@pytest.mark.asyncio
async def test_restore_preserves_later_text_only_context_summary(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    await store.open()
    await _session(store, tmp_path)
    await _append_text(store, "user", "covered")  # 1
    await store.append_message(MessageRecord(
        "session",
        "system",
        [{
            "type": "context_summary",
            "covered_through_sequence": 1,
            "text": "compaction summary",
        }],
    ))  # 2
    await store.append_message(MessageRecord(
        "session",
        "system",
        [{
            "type": "context_summary",
            "text": "provider-neutral parent summary",
        }],
    ))  # 3
    await _append_text(store, "user", "later")  # 4

    request = await ContextManager(store, model="model").build_request(
        "session", ToolRegistry()
    )
    system_text = "\n".join(str(block["text"]) for block in request.system)

    assert "compaction summary" in system_text
    assert "provider-neutral parent summary" in system_text
    assert request.messages[0]["content"][0]["text"] == "later"
    await store.close()
