from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from litecoder.common.trace import SecretRedactor
from litecoder.context.session.models import MessageRecord
from litecoder.memory.coordinator import MemoryCoordinator
from litecoder.memory.service import MemoryService
from litecoder.memory.store import MemoryStore
from litecoder.providers import ProviderEvent, StopReason


def messages(text: str) -> list[MessageRecord]:
    return [MessageRecord("session", "user", [{"type": "text", "text": text}])]


class RecordingDiagnostics:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def emit(self, event: dict[str, object]) -> None:
        self.events.append(event)


class RecordingMemoryService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.message_batches: list[tuple[MessageRecord, ...]] = []

    async def extract_memories(
        self, session_id: str, batch: Sequence[MessageRecord]
    ) -> SimpleNamespace:
        self.calls.append(("extract", session_id))
        self.message_batches.append(tuple(batch))
        await asyncio.sleep(0)
        return SimpleNamespace(
            status="completed",
            accepted=1,
            rejected=0,
            written=1,
            total=10,
            provider_code=None,
            limit=None,
        )

    async def consolidate_memories(self) -> SimpleNamespace:
        session_id = self.calls[-1][1]
        self.calls.append(("dream", session_id))
        await asyncio.sleep(0)
        return SimpleNamespace(status="completed", before=10, after=4)


class HangingMemoryService:
    async def extract_memories(
        self, session_id: str, batch: Sequence[MessageRecord]
    ) -> SimpleNamespace:
        del session_id, batch
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def consolidate_memories(self) -> SimpleNamespace:
        raise AssertionError("dream must not run after a timeout")


class FailingMemoryService:
    async def extract_memories(
        self, session_id: str, batch: Sequence[MessageRecord]
    ) -> SimpleNamespace:
        del session_id, batch
        raise RuntimeError("secret-value from memory body and prompt")

    async def consolidate_memories(self) -> SimpleNamespace:
        raise AssertionError("dream must not run after extraction failure")


class CancellationRecordingService:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False
        self._cancelled_event = asyncio.Event()

    async def extract_memories(
        self, session_id: str, batch: Sequence[MessageRecord]
    ) -> SimpleNamespace:
        del session_id, batch
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            self._cancelled_event.set()
            raise
        raise AssertionError("unreachable")

    async def _wait_for_cancelled(self) -> None:
        await self._cancelled_event.wait()

    async def consolidate_memories(self) -> SimpleNamespace:
        raise AssertionError("dream must not run while extraction is pending")


class CancellationResistantMemoryService:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def extract_memories(
        self,
        session_id: str,
        batch: Sequence[MessageRecord],
    ) -> SimpleNamespace:
        del session_id, batch
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()
        return SimpleNamespace(
            status="completed",
            accepted=1,
            rejected=0,
            written=1,
        )

    async def consolidate_memories(self) -> SimpleNamespace:
        return SimpleNamespace(status="completed", before=1, after=1)


class FailingDiagnostics:
    async def emit(self, event: dict[str, object]) -> None:
        del event
        raise RuntimeError("diagnostic backend failed")


async def no_op_diagnostic(event: dict[str, object]) -> None:
    del event


@pytest.mark.asyncio
async def test_jobs_run_serially_and_dream_follows_extraction() -> None:
    service = RecordingMemoryService()
    diagnostics = RecordingDiagnostics()
    coordinator = MemoryCoordinator(timeout=0.5)

    coordinator.submit(service, "s1", messages("one"), diagnostics.emit)
    coordinator.submit(service, "s2", messages("two"), diagnostics.emit)
    await coordinator.close()

    assert service.calls == [
        ("extract", "s1"),
        ("dream", "s1"),
        ("extract", "s2"),
        ("dream", "s2"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "written", "total"),
    [
        pytest.param("completed", 0, 10, id="no-writes"),
        pytest.param("completed", 1, 9, id="below-threshold"),
        pytest.param("completed", True, 10, id="boolean-written"),
        pytest.param("completed", 1, True, id="boolean-total"),
        pytest.param("unknown", 1, 10, id="unsupported-status"),
    ],
)
async def test_dream_is_not_called_without_a_qualifying_extraction(
    status: str,
    written: int | bool,
    total: int | bool,
) -> None:
    diagnostics = RecordingDiagnostics()

    class Service(RecordingMemoryService):
        async def extract_memories(
            self,
            session_id: str,
            batch: Sequence[MessageRecord],
        ) -> SimpleNamespace:
            self.calls.append(("extract", session_id))
            return SimpleNamespace(
                status=status,
                accepted=written,
                rejected=0,
                written=written,
                total=total,
                provider_code=None,
                limit=None,
            )

    service = Service()
    coordinator = MemoryCoordinator(timeout=0.5)
    coordinator.submit(service, "s1", messages("one"), diagnostics.emit)
    await coordinator.close()

    assert service.calls == [("extract", "s1")]
    assert all(event["operation"] != "dream" for event in diagnostics.events)


@pytest.mark.asyncio
async def test_dream_runs_after_partial_success_at_threshold() -> None:
    diagnostics = RecordingDiagnostics()

    class Service(RecordingMemoryService):
        async def extract_memories(
            self,
            session_id: str,
            batch: Sequence[MessageRecord],
        ) -> SimpleNamespace:
            self.calls.append(("extract", session_id))
            return SimpleNamespace(
                status="partial_rejected",
                accepted=1,
                rejected=1,
                written=1,
                total=10,
                provider_code=None,
                limit=None,
            )

    service = Service()
    coordinator = MemoryCoordinator(timeout=0.5)
    coordinator.submit(service, "s1", messages("one"), diagnostics.emit)
    await coordinator.close()

    assert service.calls == [("extract", "s1"), ("dream", "s1")]


@pytest.mark.asyncio
async def test_completed_job_emits_allowlisted_outcome_counts() -> None:
    service = RecordingMemoryService()
    diagnostics = RecordingDiagnostics()
    coordinator = MemoryCoordinator(timeout=0.5)

    coordinator.submit(service, "s1", messages("one"), diagnostics.emit)
    await coordinator.close()

    assert diagnostics.events == [
        {
            "operation": "extract",
            "status": "completed",
            "accepted": 1,
            "rejected": 0,
            "written": 1,
        },
        {
            "operation": "dream",
            "status": "completed",
            "before": 10,
            "after": 4,
        },
    ]


@pytest.mark.asyncio
async def test_non_completed_outcomes_keep_status_and_bounded_counts() -> None:
    diagnostics = RecordingDiagnostics()
    coordinator = MemoryCoordinator(timeout=0.5)

    class OutcomeService:
        async def extract_memories(
            self,
            session_id: str,
            batch: Sequence[MessageRecord],
        ) -> SimpleNamespace:
            del session_id, batch
            return SimpleNamespace(
                status="partial_rejected",
                accepted=2,
                rejected=1,
                written=2,
                total=10,
                provider_output="private provider output",
            )

        async def consolidate_memories(self) -> SimpleNamespace:
            return SimpleNamespace(
                status="conflict",
                before=10,
                after=11,
                filename="private-memory.md",
            )

    coordinator.submit(
        OutcomeService(),  # type: ignore[arg-type]
        "s1",
        messages("one"),
        diagnostics.emit,
    )
    await coordinator.close()

    assert diagnostics.events == [
        {
            "operation": "extract",
            "status": "partial_rejected",
            "accepted": 2,
            "rejected": 1,
            "written": 2,
        },
        {
            "operation": "dream",
            "status": "conflict",
            "before": 10,
            "after": 11,
        },
    ]
    assert "private" not in str(diagnostics.events)


@pytest.mark.asyncio
async def test_submit_freezes_messages_before_the_background_job_runs() -> None:
    service = RecordingMemoryService()
    coordinator = MemoryCoordinator(timeout=0.5)
    batch = messages("one")

    coordinator.submit(service, "s1", batch, no_op_diagnostic)
    batch.append(MessageRecord("session", "user", [{"type": "text", "text": "two"}]))
    await coordinator.close()

    assert [item.content[0]["text"] for item in service.message_batches[0]] == ["one"]


@pytest.mark.asyncio
async def test_timeout_emits_minimal_diagnostic_and_does_not_escape() -> None:
    diagnostics = RecordingDiagnostics()
    coordinator = MemoryCoordinator(timeout=0.02)

    coordinator.submit(HangingMemoryService(), "s1", messages("one"), diagnostics.emit)
    await coordinator.close()

    assert diagnostics.events == [{"operation": "extract", "status": "timeout"}]


@pytest.mark.asyncio
async def test_dream_timeout_retains_the_completed_extraction_event() -> None:
    diagnostics = RecordingDiagnostics()

    class Service(RecordingMemoryService):
        async def consolidate_memories(self) -> SimpleNamespace:
            session_id = self.calls[-1][1]
            self.calls.append(("dream", session_id))
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    service = Service()
    coordinator = MemoryCoordinator(timeout=0.02)
    coordinator.submit(service, "s1", messages("one"), diagnostics.emit)
    await coordinator.close()

    assert diagnostics.events == [
        {
            "operation": "extract",
            "status": "completed",
            "accepted": 1,
            "rejected": 0,
            "written": 1,
        },
        {"operation": "dream", "status": "timeout"},
    ]


@pytest.mark.asyncio
async def test_dream_timeout_does_not_emit_late_completion_while_diagnostics_yield() -> None:
    class Service(RecordingMemoryService):
        def __init__(self) -> None:
            super().__init__()
            self.dream_started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()

        async def consolidate_memories(self) -> SimpleNamespace:
            session_id = self.calls[-1][1]
            self.calls.append(("dream", session_id))
            self.dream_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
            return SimpleNamespace(status="completed", before=10, after=4)

    class YieldingDiagnostics:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []
            self.extraction_emitted = asyncio.Event()
            self.resume = asyncio.Event()

        async def emit(self, event: dict[str, object]) -> None:
            self.events.append(event)
            if event["operation"] == "extract":
                self.extraction_emitted.set()
                await self.resume.wait()

    service = Service()
    diagnostics = YieldingDiagnostics()
    coordinator = MemoryCoordinator(timeout=0.02)
    coordinator.submit(service, "s1", messages("one"), diagnostics.emit)

    await service.dream_started.wait()
    await diagnostics.extraction_emitted.wait()
    await service.cancelled.wait()
    service.release.set()
    await asyncio.sleep(0)
    diagnostics.resume.set()
    await coordinator.close()

    assert diagnostics.events == [
        {
            "operation": "extract",
            "status": "completed",
            "accepted": 1,
            "rejected": 0,
            "written": 1,
        },
        {"operation": "dream", "status": "timeout"},
    ]


@pytest.mark.asyncio
async def test_service_exception_does_not_stop_the_next_job() -> None:
    diagnostics = RecordingDiagnostics()
    succeeding_service = RecordingMemoryService()
    coordinator = MemoryCoordinator(timeout=0.5)

    coordinator.submit(FailingMemoryService(), "failed", messages("one"), diagnostics.emit)
    coordinator.submit(succeeding_service, "next", messages("two"), diagnostics.emit)
    await coordinator.close()

    assert diagnostics.events[0] == {"operation": "extract", "status": "failed"}
    assert succeeding_service.calls == [("extract", "next"), ("dream", "next")]


@pytest.mark.asyncio
async def test_diagnostic_exception_does_not_interrupt_the_job_or_queue() -> None:
    first_service = RecordingMemoryService()
    second_service = RecordingMemoryService()
    coordinator = MemoryCoordinator(timeout=0.5)

    coordinator.submit(first_service, "s1", messages("one"), FailingDiagnostics().emit)
    coordinator.submit(second_service, "s2", messages("two"), FailingDiagnostics().emit)
    await coordinator.close()

    assert first_service.calls == [("extract", "s1"), ("dream", "s1")]
    assert second_service.calls == [("extract", "s2"), ("dream", "s2")]


@pytest.mark.asyncio
async def test_close_waits_for_running_work() -> None:
    service = CancellationRecordingService()
    coordinator = MemoryCoordinator(timeout=0.5, close_timeout=0.5)
    coordinator.submit(service, "s1", messages("one"), no_op_diagnostic)

    close_task = asyncio.create_task(coordinator.close())
    await service.started.wait()
    assert not close_task.done()
    close_task.cancel()
    await asyncio.gather(close_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_timeout_cancels_and_drains_pending_work() -> None:
    service = CancellationRecordingService()
    coordinator = MemoryCoordinator(timeout=30.0, close_timeout=0.02)
    coordinator.submit(service, "s1", messages("one"), no_op_diagnostic)
    await service.started.wait()

    await coordinator.close()
    await asyncio.wait_for(service._wait_for_cancelled(), timeout=1)

    assert service.cancelled is True


@pytest.mark.asyncio
async def test_job_timeout_is_hard_bound_for_cancellation_resistant_service() -> None:
    service = CancellationResistantMemoryService()
    diagnostics = RecordingDiagnostics()
    coordinator = MemoryCoordinator(timeout=0.02, close_timeout=0.1)
    coordinator.submit(service, "s1", messages("one"), diagnostics.emit)
    await service.started.wait()

    job = coordinator._tail
    assert job is not None
    done, _ = await asyncio.wait({job}, timeout=0.2)
    if not done:
        service.release.set()
        await asyncio.wait({job}, timeout=1)

    assert job.done()
    assert service.cancelled.is_set()
    assert diagnostics.events == [
        {"operation": "extract", "status": "timeout"}
    ]
    service.release.set()
    await coordinator.close()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_close_timeout_does_not_await_cancellation_resistant_service() -> None:
    service = CancellationResistantMemoryService()
    coordinator = MemoryCoordinator(timeout=30.0, close_timeout=0.02)
    coordinator.submit(service, "s1", messages("one"), no_op_diagnostic)
    await service.started.wait()

    close_task = asyncio.create_task(coordinator.close())
    done, _ = await asyncio.wait({close_task}, timeout=0.2)
    if not done:
        service.release.set()
        await asyncio.wait({close_task}, timeout=1)

    assert done
    assert close_task.done()
    assert service.cancelled.is_set()
    service.release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_submit_is_ignored_after_closing_starts() -> None:
    first_service = CancellationRecordingService()
    second_service = RecordingMemoryService()
    coordinator = MemoryCoordinator(timeout=0.5, close_timeout=0.5)
    coordinator.submit(first_service, "s1", messages("one"), no_op_diagnostic)

    close_task = asyncio.create_task(coordinator.close())
    await first_service.started.wait()
    coordinator.submit(second_service, "s2", messages("two"), no_op_diagnostic)
    close_task.cancel()
    await asyncio.gather(close_task, return_exceptions=True)

    assert second_service.calls == []


@pytest.mark.asyncio
async def test_failed_diagnostic_has_no_exception_message_or_memory_content() -> None:
    diagnostics = RecordingDiagnostics()
    coordinator = MemoryCoordinator(timeout=0.5)

    coordinator.submit(FailingMemoryService(), "s1", messages("user secret"), diagnostics.emit)
    await coordinator.close()

    event = diagnostics.events[-1]
    assert event == {"operation": "extract", "status": "failed"}
    assert "secret-value" not in str(event)
    assert "memory body" not in str(event)
    assert "prompt" not in str(event)


@pytest.mark.asyncio
async def test_submit_deep_freezes_message_content_before_background_job_runs() -> None:
    service = RecordingMemoryService()
    coordinator = MemoryCoordinator(timeout=0.5)
    batch = [
        MessageRecord("session", "user", [{"type": "text", "text": "one", "metadata": {"key": "before"}}])
    ]

    coordinator.submit(service, "s1", batch, no_op_diagnostic)
    batch[0].content[0]["text"] = "after"
    batch[0].content[0]["metadata"]["key"] = "after"
    await coordinator.close()

    assert service.message_batches[0][0].content == [
        {"type": "text", "text": "one", "metadata": {"key": "before"}}
    ]


@pytest.mark.asyncio
async def test_timeout_cannot_leave_memory_lifecycle_running_or_write_late(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / ".memory")
    assert not store.root.exists()

    class CancellationResistantProvider:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = asyncio.Event()

        async def stream(self, request):
            del request
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
            yield ProviderEvent.content_block_completed(
                0,
                {
                    "type": "text",
                    "text": json.dumps([{
                        "name": "late-write",
                        "type": "project",
                        "description": "Late write",
                        "body": "Must never be installed after timeout.",
                    }]),
                },
            )
            yield ProviderEvent.response_completed(
                StopReason.END_TURN,
                "end_turn",
            )
            self.finished.set()

    class TrackingMemoryService(MemoryService):
        def __init__(self, provider: CancellationResistantProvider) -> None:
            super().__init__(
                store,
                provider,  # type: ignore[arg-type]
                "model",
                SecretRedactor.with_values(()),
            )
            self.lifecycle_active = False
            self.dream_started = False

        async def extract_memories(
            self,
            session_id: str,
            batch: Sequence[MessageRecord],
        ):
            self.lifecycle_active = True
            try:
                return await super().extract_memories(session_id, batch)
            finally:
                self.lifecycle_active = False

        async def consolidate_memories(self) -> SimpleNamespace:
            self.dream_started = True
            return SimpleNamespace(status="completed", before=1, after=1)

    class LaterService(RecordingMemoryService):
        def __init__(self, first: TrackingMemoryService) -> None:
            super().__init__()
            self.first = first
            self.overlapped = False

        async def extract_memories(
            self,
            session_id: str,
            batch: Sequence[MessageRecord],
        ) -> SimpleNamespace:
            self.overlapped = self.first.lifecycle_active
            return await super().extract_memories(session_id, batch)

    provider = CancellationResistantProvider()
    first = TrackingMemoryService(provider)
    later = LaterService(first)
    diagnostics = RecordingDiagnostics()
    coordinator = MemoryCoordinator(timeout=0.02, close_timeout=0.2)
    coordinator.submit(first, "first", messages("one"), diagnostics.emit)
    coordinator.submit(later, "later", messages("two"), diagnostics.emit)
    await provider.started.wait()

    await coordinator.close()

    assert provider.cancelled.is_set()
    assert first.lifecycle_active is False
    assert later.overlapped is False
    assert later.calls == [("extract", "later"), ("dream", "later")]
    provider.release.set()
    await asyncio.wait_for(provider.finished.wait(), timeout=1)
    await asyncio.sleep(0)

    assert first.dream_started is False
    assert not store.root.exists()
    assert diagnostics.events[0] == {"operation": "extract", "status": "timeout"}
