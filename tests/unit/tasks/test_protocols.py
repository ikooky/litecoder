from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from litecoder.tasks.message_bus import MessageBus
from litecoder.tasks.protocols import (
    ProtocolManager,
    ProtocolResponse,
    ProtocolViolation,
)


@pytest.fixture
def protocols(tmp_path: Path) -> ProtocolManager:
    return ProtocolManager(
        MessageBus(tmp_path / "mailboxes"),
        requester="lead",
        id_factory=lambda: "request-1",
    )


@pytest.mark.asyncio
async def test_plan_response_must_match_request_and_recipient(
    protocols: ProtocolManager,
) -> None:
    request = await protocols.request_plan_approval(
        "agent-1", {"tasks": ["t1"]}
    )

    with pytest.raises(ProtocolViolation, match="unexpected responder"):
        await protocols.respond_plan_approval(
            request.id, responder="agent-2", approved=True
        )

    assert protocols.pending_requests == {request.id: request}
    assert not request.future.done()


def test_new_protocol_manager_has_no_pending_requests() -> None:
    assert ProtocolManager().pending_requests == {}


@pytest.mark.asyncio
async def test_plan_request_uses_explicit_requester_and_mailbox_notification(
    tmp_path: Path,
) -> None:
    bus = MessageBus(tmp_path / "mailboxes")
    protocols = ProtocolManager(bus, requester="default-lead")

    request = await protocols.request_plan_approval(
        "reviewer", {"tasks": ["t1"]}, requester="lead-agent"
    )
    messages = await bus.receive("reviewer")
    body = json.loads(messages[0].body)

    assert request.requester == "lead-agent"
    assert request.responder == "reviewer"
    assert body == {
        "kind": "plan_approval",
        "payload": {"tasks": ["t1"]},
        "phase": "request",
        "request_id": request.id,
        "requester": "lead-agent",
        "responder": "reviewer",
    }


@pytest.mark.asyncio
async def test_terminal_response_is_single_use_and_notifies_requester(
    tmp_path: Path,
) -> None:
    bus = MessageBus(tmp_path / "mailboxes")
    protocols = ProtocolManager(bus, requester="lead")
    request = await protocols.request_plan_approval(
        "reviewer", {"tasks": ["t1"]}
    )

    await protocols.respond_plan_approval(
        request.id, responder="reviewer", approved=False, reason="revise"
    )

    assert await request == ProtocolResponse(False, "revise")
    assert request.id not in protocols.pending_requests
    response_message = (await bus.receive("lead"))[0]
    assert json.loads(response_message.body) == {
        "approved": False,
        "kind": "plan_approval",
        "phase": "response",
        "reason": "revise",
        "request_id": request.id,
        "requester": "lead",
        "responder": "reviewer",
    }
    with pytest.raises(ProtocolViolation, match="unknown or completed"):
        await protocols.respond_plan_approval(
            request.id, responder="reviewer", approved=True
        )


@pytest.mark.asyncio
async def test_response_kind_must_match_without_corrupting_other_requests(
    protocols: ProtocolManager,
) -> None:
    request = await protocols.request_shutdown("agent-1", "maintenance")

    with pytest.raises(ProtocolViolation, match="unexpected protocol kind"):
        await protocols.respond_plan_approval(
            request.id, responder="agent-1", approved=True
        )

    assert protocols.pending_requests == {request.id: request}
    await protocols.respond_shutdown(
        request.id, responder="agent-1", approved=True
    )
    assert await request == ProtocolResponse(True, None)


@pytest.mark.asyncio
async def test_timeout_removes_pending_request() -> None:
    protocols = ProtocolManager(requester="lead")
    request = await protocols.request_shutdown(
        "agent-1", timeout=0.01
    )

    with pytest.raises(TimeoutError, match="protocol request timed out"):
        await request

    assert request.id not in protocols.pending_requests
    with pytest.raises(ProtocolViolation, match="unknown or completed"):
        await protocols.respond_shutdown(
            request.id, responder="agent-1", approved=True
        )


@pytest.mark.asyncio
async def test_cancellation_removes_only_cancelled_pending_request() -> None:
    ids = iter(("request-1", "request-2"))
    protocols = ProtocolManager(requester="lead", id_factory=lambda: next(ids))
    cancelled = await protocols.request_shutdown("agent-1")
    remaining = await protocols.request_shutdown("agent-2")

    waiter = asyncio.create_task(cancelled.wait())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await asyncio.sleep(0)

    assert protocols.pending_requests == {remaining.id: remaining}


@pytest.mark.asyncio
async def test_restart_does_not_restore_pending_requests(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path / "mailboxes")
    first = ProtocolManager(bus, requester="lead")
    request = await first.request_plan_approval("agent-1", {"tasks": []})

    restarted = ProtocolManager(bus, requester="lead")

    assert request.id in first.pending_requests
    assert restarted.pending_requests == {}
    with pytest.raises(ProtocolViolation, match="unknown or completed"):
        await restarted.respond_plan_approval(
            request.id, responder="agent-1", approved=True
        )

def test_protocol_types_are_exported_from_tasks_package() -> None:
    from litecoder.tasks import (
        PendingRequest,
        ProtocolManager as ExportedProtocolManager,
        ProtocolResponse as ExportedProtocolResponse,
        ProtocolViolation as ExportedProtocolViolation,
    )

    assert ExportedProtocolManager is ProtocolManager
    assert ExportedProtocolResponse is ProtocolResponse
    assert ExportedProtocolViolation is ProtocolViolation
    assert PendingRequest.__name__ == "PendingRequest"

@pytest.mark.asyncio
async def test_any_wrong_responder_is_a_protocol_violation(
    protocols: ProtocolManager,
) -> None:
    request = await protocols.request_shutdown("agent-1")

    with pytest.raises(ProtocolViolation, match="unexpected responder"):
        await protocols.respond_shutdown(
            request.id, responder="../agent-1", approved=True
        )

    assert protocols.pending_requests == {request.id: request}

@pytest.mark.asyncio
async def test_reading_protocol_notification_does_not_complete_request(
    tmp_path: Path,
) -> None:
    bus = MessageBus(tmp_path / "mailboxes")
    protocols = ProtocolManager(bus, requester="lead")
    request = await protocols.request_shutdown("agent-1")

    messages = await bus.receive("agent-1")

    assert len(messages) == 1
    assert protocols.pending_requests == {request.id: request}
    assert not request.future.done()

@pytest.mark.asyncio
async def test_terminal_request_id_cannot_be_reused_by_a_later_request() -> None:
    protocols = ProtocolManager(requester="lead", id_factory=lambda: "same-id")
    first = await protocols.request_shutdown("agent-1")
    await protocols.respond_shutdown(
        first.id, responder="agent-1", approved=True
    )

    with pytest.raises(ProtocolViolation, match="duplicate request id"):
        await protocols.request_shutdown("agent-1")

    assert protocols.pending_requests == {}

@pytest.mark.asyncio
async def test_invalid_request_id_is_a_protocol_violation_without_corruption(
    protocols: ProtocolManager,
) -> None:
    request = await protocols.request_shutdown("agent-1")

    with pytest.raises(ProtocolViolation, match="unknown or completed"):
        await protocols.respond_shutdown(  # type: ignore[arg-type]
            [], responder="agent-1", approved=True
        )

    assert protocols.pending_requests == {request.id: request}

class ControlledProtocolBus:
    def __init__(self) -> None:
        self.messages = []
        self.request_started = asyncio.Event()
        self.response_started = asyncio.Event()
        self.release_request = asyncio.Event()
        self.release_request.set()
        self.release_response = asyncio.Event()
        self.release_response.set()
        self.request_failures = 0
        self.response_failures = 0

    async def send(self, agent_id, message) -> None:
        phase = json.loads(message.body)["phase"]
        if phase == "request":
            self.request_started.set()
            await self.release_request.wait()
            if self.request_failures:
                self.request_failures -= 1
                raise OSError("request notification unavailable")
        else:
            self.response_started.set()
            await self.release_response.wait()
            if self.response_failures:
                self.response_failures -= 1
                raise OSError("response notification unavailable")
        self.messages.append(message)


@pytest.mark.asyncio
async def test_timeout_starts_only_after_request_notification_completes() -> None:
    bus = ControlledProtocolBus()
    bus.release_request.clear()
    protocols = ProtocolManager(bus, requester="lead")
    creating = asyncio.create_task(
        protocols.request_shutdown("agent-1", timeout=0.01)
    )
    await bus.request_started.wait()
    request = next(iter(protocols.pending_requests.values()))

    await asyncio.sleep(0.03)

    assert request._timeout is None
    assert not request.future.done()
    bus.release_request.set()
    created = await creating
    assert created is request
    assert request._timeout is not None
    with pytest.raises(TimeoutError, match="protocol request timed out"):
        await request


@pytest.mark.asyncio
async def test_request_notification_failure_has_no_pending_or_timeout() -> None:
    bus = ControlledProtocolBus()
    bus.request_failures = 1
    protocols = ProtocolManager(bus, requester="lead")

    with pytest.raises(OSError, match="request notification unavailable"):
        await protocols.request_shutdown("agent-1", timeout=1)

    assert protocols.pending_requests == {}


@pytest.mark.asyncio
async def test_request_notification_cancellation_has_no_pending_or_timeout() -> None:
    bus = ControlledProtocolBus()
    bus.release_request.clear()
    protocols = ProtocolManager(bus, requester="lead")
    creating = asyncio.create_task(
        protocols.request_shutdown("agent-1", timeout=1)
    )
    await bus.request_started.wait()
    request = next(iter(protocols.pending_requests.values()))

    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    assert protocols.pending_requests == {}
    assert request._timeout is None
    assert request.future.cancelled()


@pytest.mark.asyncio
async def test_response_notification_failure_is_pending_and_retryable() -> None:
    bus = ControlledProtocolBus()
    protocols = ProtocolManager(bus, requester="lead")
    request = await protocols.request_shutdown("agent-1", timeout=1)
    bus.response_failures = 1

    with pytest.raises(RuntimeError, match="protocol response notification failed"):
        await protocols.respond_shutdown(
            request.id, responder="agent-1", approved=True
        )

    assert protocols.pending_requests == {request.id: request}
    assert not request.future.done()
    assert request._timeout is not None
    await protocols.respond_shutdown(
        request.id, responder="agent-1", approved=True
    )
    assert await request == ProtocolResponse(True, None)


@pytest.mark.asyncio
async def test_response_notification_cancellation_is_pending_and_retryable() -> None:
    bus = ControlledProtocolBus()
    protocols = ProtocolManager(bus, requester="lead")
    request = await protocols.request_shutdown("agent-1", timeout=1)
    bus.release_response.clear()
    responding = asyncio.create_task(
        protocols.respond_shutdown(
            request.id, responder="agent-1", approved=True
        )
    )
    await bus.response_started.wait()

    responding.cancel()
    with pytest.raises(asyncio.CancelledError):
        await responding

    assert protocols.pending_requests == {request.id: request}
    assert not request.future.done()
    assert request._timeout is not None
    bus.release_response.set()
    await protocols.respond_shutdown(
        request.id, responder="agent-1", approved=True
    )
    assert await request == ProtocolResponse(True, None)


@pytest.mark.asyncio
async def test_only_one_response_notification_can_be_in_flight() -> None:
    bus = ControlledProtocolBus()
    protocols = ProtocolManager(bus, requester="lead")
    request = await protocols.request_shutdown("agent-1")
    bus.release_response.clear()
    first = asyncio.create_task(
        protocols.respond_shutdown(
            request.id, responder="agent-1", approved=True
        )
    )
    await bus.response_started.wait()

    with pytest.raises(ProtocolViolation, match="already in progress"):
        await protocols.respond_shutdown(
            request.id, responder="agent-1", approved=False
        )

    bus.release_response.set()
    await first
    assert await request == ProtocolResponse(True, None)
    assert sum(json.loads(item.body)["phase"] == "response" for item in bus.messages) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf")])
async def test_protocol_manager_rejects_non_finite_timeout(timeout: float) -> None:
    protocols = ProtocolManager(requester="lead")

    with pytest.raises(ValueError, match="finite positive number"):
        await protocols.request_shutdown("agent-1", timeout=timeout)

    assert protocols.pending_requests == {}


@pytest.mark.asyncio
async def test_timeout_exception_is_observed_but_remains_awaitable() -> None:
    protocols = ProtocolManager(requester="lead")
    request = await protocols.request_shutdown("agent-1", timeout=0.01)

    await asyncio.sleep(0.03)

    assert request.future.done()
    assert request.future._log_traceback is False
    with pytest.raises(TimeoutError, match="protocol request timed out"):
        await request
