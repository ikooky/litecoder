"""Public interfaces for the memory package."""

from litecoder.memory.consolidation import (
    MemoryConsolidationResult,
    consolidate_memories,
)
from litecoder.memory.coordinator import MemoryCoordinator
from litecoder.memory.extraction import MemoryExtractionResult, extract_memories
from litecoder.memory.loading import LoadedMemories, load_memories
from litecoder.memory.models import MemoryEntry, MemoryMetadata, MemorySnapshot
from litecoder.memory.selection import select_relevant_memories
from litecoder.memory.service import MemoryService
from litecoder.memory.store import MemoryStore, write_memory_file

__all__ = [
    "LoadedMemories",
    "MemoryConsolidationResult",
    "MemoryCoordinator",
    "MemoryEntry",
    "MemoryExtractionResult",
    "MemoryMetadata",
    "MemoryService",
    "MemorySnapshot",
    "MemoryStore",
    "consolidate_memories",
    "extract_memories",
    "load_memories",
    "select_relevant_memories",
    "write_memory_file",
]
