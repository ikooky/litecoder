"""Data models for the surrounding subsystem."""

from __future__ import annotations

import re
from dataclasses import dataclass

MEMORY_TYPES = frozenset({"user", "feedback", "project", "reference"})
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class MemoryMetadata:
    """Data model representing the memory metadata."""
    filename: str
    name: str
    description: str
    type: str


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """Data model representing the memory entry."""
    name: str
    description: str
    type: str
    body: str

    def __post_init__(self) -> None:
        validate_memory_name(self.name)
        _validate_text(self.description, "description", allow_empty=False)
        if self.type not in MEMORY_TYPES:
            raise ValueError("memory type is invalid")
        _validate_text(self.body, "body", allow_empty=True)

    @property
    def filename(self) -> str:
        """Handle the filename operation."""
        return f"{self.name}.md"

    def metadata(self) -> MemoryMetadata:
        """Handle the metadata operation."""
        return MemoryMetadata(self.filename, self.name, self.description, self.type)

    def render(self) -> str:
        """Render the requested operation."""
        return (
            "---\n"
            f"name: {self.name}\n"
            f"description: {self.description}\n"
            f"type: {self.type}\n"
            "---\n\n"
            f"{self.body.rstrip()}\n"
        )


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Data model representing the memory snapshot."""
    index: str
    entries: tuple[MemoryEntry, ...]


def validate_memory_name(name: str) -> None:
    """Validate the memory name."""
    if (
        not isinstance(name, str)
        or not _SAFE_NAME.fullmatch(name)
        or name.casefold() == "memory"
    ):
        raise ValueError("memory name is invalid")


def _validate_text(value: object, field_name: str, *, allow_empty: bool) -> None:
    """Validate the text."""
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError(f"memory {field_name} is invalid")
    if field_name == "description" and any(char in value for char in "\r\n"):
        raise ValueError(f"memory {field_name} is invalid")
    if not allow_empty and not value.strip():
        raise ValueError(f"memory {field_name} is invalid")
