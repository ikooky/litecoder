"""Protocols used by task orchestration."""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any, Literal

from litecoder.tasks.message_bus import (
    MessageBus,
    TeamMessage,
    validate_agent_id,
)

ProtocolKind = Literal["plan_approval", "shutdown"]


class ProtocolViolation(RuntimeError):
    """Component responsible for the protocol violation."""
    pass


class ProtocolNotificationError(RuntimeError):
    """Raised when the protocol notification error conditions occur."""
    pass


@dataclass(frozen=True, slots=True)
class ProtocolResponse:
    """Data model representing the protocol response."""
    approved: bool
    reason: str | None = None


@dataclass(slots=True)
class PendingRequest:
    """Data model representing the pending request."""
    id: str
    kind: ProtocolKind
    requester: str
    responder: str
    payload: dict[str, object]
    future: asyncio.Future[ProtocolResponse] = field(repr=False)
    _timeout: asyncio.TimerHandle | None = field(
        default=None, repr=False, compare=False
    )
    _responding: bool = field(default=False, repr=False, compare=False)

    async def wait(self) -> ProtocolResponse:
        """Wait for the requested operation."""
        return await self.future

    def __await__(self) -> Generator[Any, None, ProtocolResponse]:
        return self.wait().__await__()


class ProtocolManager:
    """Manager coordinating the protocol manager."""
    def __init__(
        self,
        message_bus: MessageBus | None = None,
        *,
        requester: str = "lead",
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.message_bus = message_bus
        self.requester = validate_agent_id(requester)
        self.id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self.pending_requests: dict[str, PendingRequest] = {}
        self._issued_request_ids: set[str] = set()

    async def request_plan_approval(
        self,
        agent_id: str,
        plan: dict[str, object],
        *,
        requester: str | None = None,
        timeout: float | None = None,
    ) -> PendingRequest:
        """Handle the request plan approval operation."""
        if not isinstance(plan, dict):
            raise ValueError("plan must be an object")
        return await self._request(
            "plan_approval",
            agent_id,
            dict(plan),
            requester=requester,
            timeout=timeout,
        )

    async def respond_plan_approval(
        self,
        request_id: str,
        *,
        responder: str,
        approved: bool,
        reason: str | None = None,
    ) -> None:
        """Handle the respond plan approval operation."""
        await self._respond(
            request_id,
            kind="plan_approval",
            responder=responder,
            approved=approved,
            reason=reason,
        )

    async def request_shutdown(
        self,
        agent_id: str,
        reason: str | None = None,
        *,
        requester: str | None = None,
        timeout: float | None = None,
    ) -> PendingRequest:
        """Handle the request shutdown operation."""
        _optional_reason(reason)
        payload: dict[str, object] = {}
        if reason is not None:
            payload["reason"] = reason
        return await self._request(
            "shutdown",
            agent_id,
            payload,
            requester=requester,
            timeout=timeout,
        )

    async def respond_shutdown(
        self,
        request_id: str,
        *,
        responder: str,
        approved: bool,
        reason: str | None = None,
    ) -> None:
        """Handle the respond shutdown operation."""
        await self._respond(
            request_id,
            kind="shutdown",
            responder=responder,
            approved=approved,
            reason=reason,
        )

    async def _request(
        self,
        kind: ProtocolKind,
        responder: str,
        payload: dict[str, object],
        *,
        requester: str | None,
        timeout: float | None,
    ) -> PendingRequest:
        responder = validate_agent_id(responder)
        selected_requester = (
            self.requester
            if requester is None
            else validate_agent_id(requester)
        )
        timeout = _optional_timeout(timeout)
        request_id = self.id_factory()
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request id must be a non-empty string")
        if request_id in self._issued_request_ids:
            raise ProtocolViolation(f"duplicate request id {request_id!r}")
        self._issued_request_ids.add(request_id)
        future = asyncio.get_running_loop().create_future()
        request = PendingRequest(
            request_id,
            kind,
            selected_requester,
            responder,
            payload,
            future,
        )
        self.pending_requests[request_id] = request
        future.add_done_callback(
            lambda completed, identifier=request_id: self._future_done(
                identifier, completed
            )
        )
        try:
            await self._notify_request(request)
        except BaseException:
            self._remove(request_id, request)
            if not future.done():
                future.cancel()
            raise
        if timeout is not None:
            self._arm_timeout(request, timeout)
        return request

    async def _respond(
        self,
        request_id: str,
        *,
        kind: ProtocolKind,
        responder: str,
        approved: bool,
        reason: str | None,
    ) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise ProtocolViolation("unknown or completed protocol request")
        request = self.pending_requests.get(request_id)
        if request is None or request.future.done():
            raise ProtocolViolation("unknown or completed protocol request")
        if request.kind != kind:
            raise ProtocolViolation("unexpected protocol kind")
        if request.responder != responder:
            raise ProtocolViolation("unexpected responder")
        if request._responding:
            raise ProtocolViolation("protocol response already in progress")
        if not isinstance(approved, bool):
            raise ValueError("approved must be a boolean")
        _optional_reason(reason)
        response = ProtocolResponse(approved, reason)
        request._responding = True
        remaining_timeout = self._pause_timeout(request)
        try:
            await self._notify_response(request, response)
        except asyncio.CancelledError:
            request._responding = False
            self._arm_timeout(request, remaining_timeout)
            raise
        except Exception as error:
            request._responding = False
            self._arm_timeout(request, remaining_timeout)
            raise ProtocolNotificationError(
                "protocol response notification failed"
            ) from error
        request._responding = False
        if (
            self.pending_requests.get(request_id) is not request
            or request.future.done()
        ):
            raise ProtocolViolation("protocol request completed during response")
        self._remove(request_id, request)
        request.future.set_result(response)

    async def _notify_request(self, request: PendingRequest) -> None:
        await self._notify(
            request.requester,
            request.responder,
            {
                "kind": request.kind,
                "payload": request.payload,
                "phase": "request",
                "request_id": request.id,
                "requester": request.requester,
                "responder": request.responder,
            },
        )

    async def _notify_response(
        self, request: PendingRequest, response: ProtocolResponse
    ) -> None:
        await self._notify(
            request.responder,
            request.requester,
            {
                "approved": response.approved,
                "kind": request.kind,
                "phase": "response",
                "reason": response.reason,
                "request_id": request.id,
                "requester": request.requester,
                "responder": request.responder,
            },
        )

    async def _notify(
        self, sender: str, recipient: str, payload: dict[str, object]
    ) -> None:
        if self.message_bus is None:
            return
        await self.message_bus.send(
            recipient,
            TeamMessage(
                sender,
                recipient,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )

    def _expire(
        self, request_id: str, future: asyncio.Future[ProtocolResponse]
    ) -> None:
        request = self.pending_requests.get(request_id)
        if request is None or request.future is not future or future.done():
            return
        self._remove(request_id, request)
        future.set_exception(TimeoutError("protocol request timed out"))

    def _future_done(
        self,
        request_id: str,
        future: asyncio.Future[ProtocolResponse],
    ) -> None:
        if future.cancelled():
            request = self.pending_requests.get(request_id)
            if request is not None and request.future is future:
                self._remove(request_id, request)
            return
        future.exception()

    def _arm_timeout(
        self, request: PendingRequest, timeout: float | None
    ) -> None:
        if timeout is None:
            return
        if (
            self.pending_requests.get(request.id) is not request
            or request.future.done()
        ):
            return
        request._timeout = asyncio.get_running_loop().call_later(
            timeout, self._expire, request.id, request.future
        )

    def _pause_timeout(self, request: PendingRequest) -> float | None:
        handle = request._timeout
        if handle is None:
            return None
        remaining = max(0.0, handle.when() - asyncio.get_running_loop().time())
        handle.cancel()
        request._timeout = None
        return remaining

    def _remove(self, request_id: str, request: PendingRequest) -> None:
        if self.pending_requests.get(request_id) is not request:
            return
        del self.pending_requests[request_id]
        if request._timeout is not None:
            request._timeout.cancel()
            request._timeout = None


def _optional_timeout(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a finite positive number")
    converted = float(timeout)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError("timeout must be a finite positive number")
    return converted


def _optional_reason(reason: str | None) -> None:
    if reason is not None and not isinstance(reason, str):
        raise ValueError("reason must be text or None")
