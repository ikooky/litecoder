"""Duplicate tool-call suppression."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from litecoder.common.trace import trace_annotation
from litecoder.providers._json import JsonValue, snapshot_json
from litecoder.tools.models import ToolCall, ToolResult, ToolSpec


DUPLICATE_CALL_WINDOW_ROUNDS = 5
Annotation = Callable[..., Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class DuplicateKey:
    """Data model representing the duplicate key."""
    agent_session_id: str
    generation: int
    workspace_id: str
    workspace_version: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class PreparedDuplicate:
    """Data model representing the prepared duplicate."""
    workspace_id: str
    fingerprint: str | None
    generation: int


@dataclass(frozen=True, slots=True)
class _Record:
    """Data model representing the record."""
    round_number: int
    preview: JsonValue


@dataclass(slots=True)
class _Lease:
    """Data model representing the lease."""
    lock: asyncio.Lock
    users: int = 0


def fingerprint(call: ToolCall, workspace_id: str) -> str:
    """Return a stable fingerprint for the tool call."""
    if not isinstance(workspace_id, str) or not workspace_id.strip():
        raise ValueError("workspace_id must not be empty")
    payload = json.dumps(
        call.arguments,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    source = f"{call.name}\0{workspace_id}\0{payload}".encode("utf-8")
    return hashlib.sha256(source).hexdigest()


class DuplicateGuard:
    """Component responsible for the duplicate guard."""
    def __init__(self, *, annotation: Annotation | None = None) -> None:
        self._annotation = annotation or trace_annotation
        self._records: dict[DuplicateKey, _Record] = {}
        self._rounds: dict[str, int] = {}
        self._generations: dict[str, int] = {}
        self._leases: dict[tuple[str, int, str, str], _Lease] = {}
        self._lock = asyncio.Lock()

    @property
    def record_count(self) -> int:
        """Record the count."""
        return len(self._records)

    @property
    def lease_count(self) -> int:
        """Handle the lease count operation."""
        return len(self._leases)

    def prepare(
        self,
        call: ToolCall,
        workspace_id: str,
        spec: ToolSpec | None = None,
        *,
        agent_session_id: str | None = None,
    ) -> PreparedDuplicate | None:
        """Prepare the requested operation."""
        generation = (
            self._generations.get(agent_session_id, 0)
            if agent_session_id is not None
            else 0
        )
        if spec is not None and spec.dedupe_policy == "none":
            return PreparedDuplicate(workspace_id, None, generation)
        return PreparedDuplicate(
            workspace_id, fingerprint(call, workspace_id), generation
        )

    @staticmethod
    def freeze_preview(preview: object) -> JsonValue:
        """Handle the freeze preview operation."""
        return snapshot_json(preview, "preview")

    @asynccontextmanager
    async def execution_lease(
        self,
        agent_session_id: str,
        workspace_id: str,
        *,
        call: ToolCall,
        spec: ToolSpec | None = None,
        prepared: PreparedDuplicate | None = None,
    ) -> AsyncIterator[PreparedDuplicate | None]:
        """Handle the execution lease operation."""
        if not isinstance(agent_session_id, str) or not agent_session_id.strip():
            raise ValueError("agent_session_id must not be empty")
        prepared = prepared if prepared is not None else self.prepare(
            call, workspace_id, spec, agent_session_id=agent_session_id
        )
        if prepared is None or prepared.fingerprint is None:
            yield prepared
            return
        key = (
            agent_session_id,
            prepared.generation,
            workspace_id,
            prepared.fingerprint,
        )
        async with self._lock:
            lease = self._leases.get(key)
            if lease is None:
                lease = _Lease(asyncio.Lock())
                self._leases[key] = lease
            lease.users += 1
        acquired = False
        try:
            await lease.lock.acquire()
            acquired = True
            yield prepared
        finally:
            if acquired:
                lease.lock.release()
            async with self._lock:
                lease.users -= 1
                if lease.users == 0 and not lease.lock.locked():
                    self._leases.pop(key, None)

    async def check(
        self,
        agent_session_id: str,
        workspace_id: str,
        workspace_version: int,
        *,
        round_number: int,
        call: ToolCall,
        spec: ToolSpec | None = None,
        prepared: PreparedDuplicate | None = None,
    ) -> ToolResult | None:
        """Check the requested operation."""
        _validate_coordinates(agent_session_id, workspace_id, workspace_version, round_number)
        prepared = prepared if prepared is not None else self.prepare(
            call, workspace_id, spec, agent_session_id=agent_session_id
        )
        async with self._lock:
            generation = self._generations.get(agent_session_id, 0)
            self._purge_stale_generation(agent_session_id, generation)
            if prepared is not None and prepared.generation != generation:
                return None
            self._observe_round(agent_session_id, round_number)
        if prepared is None or prepared.fingerprint is None:
            return None
        key = DuplicateKey(
            agent_session_id,
            generation,
            workspace_id,
            workspace_version,
            prepared.fingerprint,
        )
        async with self._lock:
            record = self._records.get(key)
            if record is None or round_number - record.round_number >= DUPLICATE_CALL_WINDOW_ROUNDS:
                return None
            preview = snapshot_json(record.preview, "preview")
        result = self._annotation(
            intent="avoid repeated successful tool call",
            reason="duplicate-tool-call",
            attributes={"tool": call.name, "fingerprint": key.fingerprint},
        )
        if inspect.isawaitable(result):
            await result
        return ToolResult(
            call.id,
            "duplicate_blocked",
            "Duplicate tool call blocked",
            {"preview": preview},
        )

    async def record_prepared_success(
        self,
        agent_session_id: str,
        workspace_id: str,
        workspace_version: int,
        *,
        round_number: int,
        prepared: PreparedDuplicate | None,
        preview: JsonValue,
        post_workspace_version: int | None = None,
        round_prevalidated: bool = False,
    ) -> None:
        """Record the prepared success."""
        _validate_coordinates(agent_session_id, workspace_id, workspace_version, round_number)
        if prepared is None:
            async with self._lock:
                self._observe_round(agent_session_id, round_number)
            return
        if prepared.fingerprint is None:
            async with self._lock:
                generation = self._generations.get(agent_session_id, 0)
                if prepared.generation != generation:
                    return
                if not round_prevalidated:
                    self._observe_round(agent_session_id, round_number)
            return
        versions = {workspace_version}
        if post_workspace_version is not None:
            _validate_version(post_workspace_version)
            versions.add(post_workspace_version)
        async with self._lock:
            generation = self._generations.get(agent_session_id, 0)
            self._purge_stale_generation(agent_session_id, generation)
            if prepared.generation != generation:
                return
            if not round_prevalidated:
                self._observe_round(agent_session_id, round_number)
            for version in versions:
                key = DuplicateKey(
                    agent_session_id,
                    generation,
                    workspace_id,
                    version,
                    prepared.fingerprint,
                )
                self._records[key] = _Record(round_number, preview)

    async def record_success(
        self,
        agent_session_id: str,
        workspace_id: str,
        workspace_version: int,
        *,
        round_number: int,
        call: ToolCall,
        preview: object = None,
        post_workspace_version: int | None = None,
        spec: ToolSpec | None = None,
    ) -> None:
        """Record the success."""
        prepared = self.prepare(
            call, workspace_id, spec, agent_session_id=agent_session_id
        )
        saved_preview = self.freeze_preview(preview)
        await self.record_prepared_success(
            agent_session_id,
            workspace_id,
            workspace_version,
            round_number=round_number,
            prepared=prepared,
            preview=saved_preview,
            post_workspace_version=post_workspace_version,
        )

    async def start_user_message(self, agent_session_id: str) -> None:
        """Start the user message."""
        if not isinstance(agent_session_id, str) or not agent_session_id.strip():
            raise ValueError("agent_session_id must not be empty")
        async with self._lock:
            self._records = {
                key: value
                for key, value in self._records.items()
                if key.agent_session_id != agent_session_id
            }
            self._rounds.pop(agent_session_id, None)
            self._generations[agent_session_id] = (
                self._generations.get(agent_session_id, 0) + 1
            )

    async def clear_for_new_user_message(self, agent_session_id: str) -> None:
        """Clear the for new user message."""
        await self.start_user_message(agent_session_id)

    def _purge_stale_generation(
        self, agent_session_id: str, generation: int
    ) -> None:
        self._records = {
            key: record
            for key, record in self._records.items()
            if key.agent_session_id != agent_session_id
            or key.generation == generation
        }
    def _observe_round(self, agent_session_id: str, round_number: int) -> None:
        previous = self._rounds.get(agent_session_id)
        if previous is not None and round_number < previous:
            raise ValueError("round_number must be monotonic within an Agent Session")
        if previous is None or round_number > previous:
            self._rounds[agent_session_id] = round_number
        cutoff = round_number - DUPLICATE_CALL_WINDOW_ROUNDS + 1
        if cutoff <= 0:
            return
        self._records = {
            key: record
            for key, record in self._records.items()
            if key.agent_session_id != agent_session_id or record.round_number >= cutoff
        }


def _validate_coordinates(
    agent_session_id: str, workspace_id: str, workspace_version: int, round_number: int
) -> None:
    """Validate the coordinates."""
    if not isinstance(agent_session_id, str) or not agent_session_id.strip():
        raise ValueError("agent_session_id must not be empty")
    if not isinstance(workspace_id, str) or not workspace_id.strip():
        raise ValueError("workspace_id must not be empty")
    _validate_version(workspace_version)
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 0:
        raise ValueError("round_number must be a non-negative integer")


def _validate_version(version: int) -> None:
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("workspace_version must be a non-negative integer")
