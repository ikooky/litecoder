from litecoder.memory.prompts import MEMORY_EXTRACTION_SYSTEM_PROMPT


def test_memory_extraction_prompt_prefers_precision_without_fixed_exclusions() -> None:
    normalized = MEMORY_EXTRACTION_SYSTEM_PROMPT.casefold()

    assert "high precision" in normalized
    assert "prefer returning no memory" in normalized
    assert "explicitly stated or directly evidenced" in normalized
    assert "future utility in context" in normalized
    assert "not by its topic or category" in normalized
    assert "does not clearly satisfy every extraction criterion" in normalized
    assert "empty json array" in normalized
