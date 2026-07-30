from __future__ import annotations

import pytest

from litecoder.memory.models import MemoryEntry, validate_memory_name


@pytest.mark.parametrize("reserved_name", ["memory", "Memory", "MEMORY"])
def test_reserved_index_stem_is_rejected_by_unified_name_validation(
    reserved_name: str,
) -> None:
    with pytest.raises(ValueError, match="memory name is invalid"):
        validate_memory_name(reserved_name)

    with pytest.raises(ValueError, match="memory name is invalid"):
        MemoryEntry(reserved_name, "Reserved index", "project", "body")