"""Memory service lifecycle and operations."""

from __future__ import annotations

from collections.abc import Sequence
import asyncio

from litecoder.common.trace import SecretRedactor
from litecoder.agent.prompt_policy import DURABLE_MEMORY_INSTRUCTIONS
from litecoder.context.session.models import MessageRecord
from litecoder.memory.consolidation import (
    MemoryConsolidationResult,
    consolidate_memories,
)
from litecoder.memory.extraction import MemoryExtractionResult, extract_memories
from litecoder.memory.loading import LoadedMemories, load_memories
from litecoder.memory.store import MemoryStore
from litecoder.providers import ModelProvider


class MemoryService:
    """Small facade shared by prompt construction and later memory lifecycle work."""

    def __init__(
        self,
        store: MemoryStore,
        provider: ModelProvider,
        model: str,
        redactor: SecretRedactor,
    ) -> None:
        self.store = store
        self.provider = provider
        self.model = model
        self.redactor = redactor

    def system_payload(self) -> dict[str, object]:
        """Handle the system payload operation."""
        try:
            index = self.store.read_index()
        except Exception:
            index = ""
        return {
            "index": index,
            "instructions": list(DURABLE_MEMORY_INSTRUCTIONS),
        }

    async def system_payload_async(self) -> dict[str, object]:
        """Read lock-backed memory state without blocking the event loop."""
        return await asyncio.to_thread(self.system_payload)

    async def load_memories(
        self, messages: Sequence[MessageRecord]
    ) -> LoadedMemories:
        """Load the memories."""
        return await load_memories(
            self.store,
            self.provider,
            self.model,
            messages,
        )

    async def extract_memories(
        self,
        session_id: str,
        messages: Sequence[MessageRecord],
    ) -> MemoryExtractionResult:
        """Extract the memories."""
        return await extract_memories(
            self.store,
            self.provider,
            self.model,
            self.redactor,
            session_id,
            messages,
        )

    async def consolidate_memories(self) -> MemoryConsolidationResult:
        """Handle the consolidate memories operation."""
        return await consolidate_memories(
            self.store,
            self.provider,
            self.model,
            self.redactor,
        )
