from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

import litecoder.memory.selection as selection_module
from litecoder.common.errors import ErrorCode, LiteCoderError
from litecoder.context.session.models import MessageRecord
from litecoder.memory.models import MemoryEntry
from litecoder.memory.selection import select_relevant_memories
from litecoder.memory.store import MemoryStore
from litecoder.providers import ProviderEvent, StopReason
from tests.fakes.provider import FakeProvider


def seeded_store(
    tmp_path: Path,
    *,
    names: tuple[str, ...] = (),
    entries: list[MemoryEntry] | None = None,
) -> MemoryStore:
    store = MemoryStore(tmp_path / "memory")
    store.replace_all(
        entries
        if entries is not None
        else [
            MemoryEntry(name, f"{name} memory", "project", f"{name} body.")
            for name in names
        ]
    )
    return store


def text_message(role: str, text: str) -> MessageRecord:
    return MessageRecord("session", role, [{"type": "text", "text": text}])


def side_query_round(text: str) -> list[ProviderEvent]:
    return [
        ProviderEvent.content_block_completed(0, {"type": "text", "text": text}),
        ProviderEvent.response_completed(StopReason.END_TURN, "end_turn"),
    ]


def transient_error() -> LiteCoderError:
    return LiteCoderError(ErrorCode.PROVIDER_TRANSIENT, "side query failed")


async def test_model_selects_memory_indices_from_recent_user_messages(
    tmp_path: Path,
) -> None:
    store = seeded_store(tmp_path, names=("tabs", "build", "feedback"))
    provider = FakeProvider([side_query_round("[0, 2, 0, true, 99]")])
    messages = [
        text_message("user", "Earlier preference"),
        text_message("assistant", "Noted"),
        text_message("user", "当前构建方式是什么？"),
    ]

    selected = await select_relevant_memories(
        store, provider, "model", messages, max_items=2
    )

    assert selected == ["build.md", "tabs.md"]
    assert "当前构建方式是什么？" in provider.requests[0].messages[0]["content"][0]["text"]


async def test_valid_empty_model_selection_does_not_fallback(tmp_path: Path) -> None:
    store = seeded_store(tmp_path, names=("python-style",))
    provider = FakeProvider([side_query_round("[]")])

    selected = await select_relevant_memories(
        store, provider, "model", [text_message("user", "python style")]
    )

    assert selected == []


async def test_unicode_keyword_fallback_runs_only_after_model_failure(
    tmp_path: Path,
) -> None:
    store = seeded_store(
        tmp_path,
        entries=[MemoryEntry("reply-style", "每次回答先说喵", "user", "先说喵。")],
    )
    provider = FakeProvider([[ProviderEvent.provider_error(transient_error())]])

    selected = await select_relevant_memories(
        store, provider, "model", [text_message("user", "回答时记得先说喵")]
    )

    assert selected == ["reply-style.md"]


@pytest.mark.asyncio
async def test_selection_timeout_falls_back_without_waiting_for_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = seeded_store(tmp_path, names=("python-style",))
    started = asyncio.Event()

    class HangingProvider:
        async def stream(self, request):
            del request
            started.set()
            await asyncio.Event().wait()
            yield

    monkeypatch.setattr(selection_module, "SELECTION_TIMEOUT_SECONDS", 0.02)
    provider = HangingProvider()
    task = asyncio.create_task(
        select_relevant_memories(
            store,
            provider,
            "model",
            [text_message("user", "python style")],
        )
    )
    await started.wait()
    started_at = time.monotonic()
    selected = await asyncio.wait_for(task, timeout=0.2)

    assert time.monotonic() - started_at < 0.15
    assert selected == ["python-style.md"]


@pytest.mark.asyncio
async def test_cancellation_resistant_selection_provider_cannot_block_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = seeded_store(tmp_path, names=("python-style",))
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    class CancellationResistantProvider:
        async def stream(self, request):
            del request
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                raise
            yield

    monkeypatch.setattr(selection_module, "SELECTION_TIMEOUT_SECONDS", 0.02)
    task = asyncio.create_task(
        select_relevant_memories(
            store,
            CancellationResistantProvider(),
            "model",
            [text_message("user", "python style")],
        )
    )
    await started.wait()
    selected = await asyncio.wait_for(task, timeout=0.2)

    assert selected == ["python-style.md"]
    # Cancellation is delivered asynchronously; wait for the provider to
    # observe it before asserting, then release the suspended cleanup.
    await asyncio.wait_for(cancelled.wait(), timeout=0.2)
    assert cancelled.is_set()
    release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_outer_selection_cancellation_propagates_and_cancels_side_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = seeded_store(tmp_path, names=("python-style",))
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class HangingProvider:
        async def stream(self, request):
            del request
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            yield

    monkeypatch.setattr(selection_module, "SELECTION_TIMEOUT_SECONDS", 30.0)
    task = asyncio.create_task(
        select_relevant_memories(
            store,
            HangingProvider(),
            "model",
            [text_message("user", "python style")],
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await cancelled.wait()


@pytest.mark.asyncio
async def test_selection_detailed_result_records_fallback_reason_and_model_empty(
    tmp_path: Path,
) -> None:
    store = seeded_store(tmp_path, names=("python-style",))
    failed = FakeProvider([[ProviderEvent.provider_error(transient_error())]])
    fallback = await selection_module._select_relevant_memories_result(
        store,
        failed,
        "model",
        [text_message("user", "python style")],
    )
    empty = FakeProvider([side_query_round("[]")])
    model_empty = await selection_module._select_relevant_memories_result(
        store,
        empty,
        "model",
        [text_message("user", "python style")],
    )

    assert fallback.reason == "provider_failed"
    assert fallback.source == "fallback"
    assert model_empty.reason is None
    assert model_empty.source == "model"
    assert model_empty.filenames == ()

async def test_non_array_model_text_uses_unicode_keyword_fallback(tmp_path: Path) -> None:
    store = seeded_store(tmp_path, names=("python-style",))
    provider = FakeProvider([side_query_round("not an array")])

    selected = await select_relevant_memories(
        store, provider, "model", [text_message("user", "python style")]
    )

    assert selected == ["python-style.md"]
