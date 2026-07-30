"""Task and agent message bus."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from litecoder.common.locks import NamedFileLock

_AGENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_MAILBOX_POLL_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class TeamMessage:
    """Data model representing the team message."""
    sender: str
    recipient: str
    body: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        _non_empty(self.sender, "sender")
        _non_empty(self.recipient, "recipient")
        if not isinstance(self.body, str):
            raise ValueError("body must be text")
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id must be a non-empty string")

    @property
    def message_id(self) -> str:
        """Handle the message id operation."""
        return self.id

    def to_dict(self) -> dict[str, str]:
        """Convert this object to a dictionary."""
        return {"id": self.id, "sender": self.sender, "recipient": self.recipient, "body": self.body}

    @classmethod
    def from_dict(cls, value: object) -> "TeamMessage":
        """Construct a value from dict data."""
        if not isinstance(value, dict):
            raise ValueError("message must be an object")
        fields = (value.get("id", value.get("message_id")), value.get("sender"), value.get("recipient"), value.get("body"))
        if not all(isinstance(item, str) for item in fields):
            raise ValueError("message fields are invalid")
        return cls(fields[1], fields[2], fields[3], fields[0])


class MessageBus:
    """Component responsible for the message bus."""
    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise ValueError("root must be a Path")
        self.root = root
        self.lock = asyncio.Lock()
        self._events: dict[str, asyncio.Event] = {}
        self.sent_count = 0
        self.received_count = 0
        self.file_lock = NamedFileLock(
            f"mailbox-{_mailbox_lock_name(root)}", root
        )

    @asynccontextmanager
    async def _locked(self):
        async with self.lock:
            async with self.file_lock.acquired_async():
                yield

    def _event(self, agent_id: str) -> asyncio.Event:
        event = self._events.get(agent_id)
        if event is None:
            event = self._events[agent_id] = asyncio.Event()
        return event

    async def send(self, agent_id: str, message: TeamMessage) -> None:
        """Send the requested operation."""
        validate_agent_id(agent_id)
        if not isinstance(message, TeamMessage):
            raise ValueError("message must be a TeamMessage")
        if message.recipient != agent_id:
            raise ValueError("message recipient does not match mailbox")
        async with self._locked():
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / f"{agent_id}.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.sent_count += 1
            self._event(agent_id).set()

    async def receive(self, agent_id: str) -> list[TeamMessage]:
        """Wait for and return the next message."""
        validate_agent_id(agent_id)
        async with self._locked():
            path = self.root / f"{agent_id}.jsonl"
            if not path.exists():
                self._event(agent_id).clear()
                return []
            messages: list[TeamMessage] = []
            malformed: list[str] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    messages.append(TeamMessage.from_dict(json.loads(line)))
                except (TypeError, ValueError, json.JSONDecodeError):
                    malformed.append(line)
            if malformed:
                self._quarantine_malformed(path, malformed)
            path.unlink()
            self.received_count += len(messages)
            self._event(agent_id).clear()
            return messages

    @staticmethod
    def _quarantine_malformed(path: Path, lines: list[str]) -> None:
        quarantine = path.with_name(
            f"{path.stem}.corrupt-{uuid.uuid4().hex}{path.suffix}"
        )
        with quarantine.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    async def wait_for_messages(self, agent_id: str) -> None:
        """Wait until the durable mailbox has at least one message."""
        validate_agent_id(agent_id)
        while True:
            async with self._locked():
                if (self.root / f"{agent_id}.jsonl").exists():
                    return
                event = self._event(agent_id)
                event.clear()
            try:
                await asyncio.wait_for(event.wait(), timeout=_MAILBOX_POLL_SECONDS)
            except TimeoutError:
                continue


def validate_agent_id(agent_id: object) -> str:
    """Validate the agent id."""
    if not isinstance(agent_id, str) or _AGENT_ID.fullmatch(agent_id) is None:
        raise ValueError("agent_id must be a safe generated identifier")
    return agent_id


def _non_empty(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")

def _mailbox_lock_name(root: Path) -> str:
    try:
        identity = str(root.expanduser().resolve())
    except OSError:
        identity = str(root.expanduser())
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
