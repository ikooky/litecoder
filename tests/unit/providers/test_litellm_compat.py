from __future__ import annotations

from typing import Any

from litecoder.providers._litellm_compat import (
    install_stream_chunk_builder_compat,
)


def vulnerable_processor_type() -> type[Any]:
    class VulnerableProcessor:
        def __init__(self, chunks: list[dict[str, Any]]) -> None:
            self.first_chunk = chunks[0]
            self.seen_chunks: list[dict[str, Any]] = []

        def build_base_response(
            self, chunks: list[dict[str, Any]]
        ) -> dict[str, Any]:
            self.seen_chunks = chunks
            first = next(
                (chunk for chunk in chunks if chunk.get("choices")),
                self.first_chunk,
            )
            role = first["choices"][0]["delta"]["role"]
            return {"role": role, "first": self.first_chunk}

    return VulnerableProcessor


class FixedProcessor:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.first_chunk = chunks[0]
        self.seen_chunks: list[dict[str, Any]] = []

    def build_base_response(
        self, chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.seen_chunks = chunks
        return {"role": "assistant"}


def chunk(*, choices: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "request-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "model",
        "choices": choices,
    }


def test_compat_defaults_usage_only_stream_role_without_mutating_input() -> None:
    processor_type = vulnerable_processor_type()
    assert install_stream_chunk_builder_compat(processor_type) is True
    usage_only = chunk(choices=[])
    processor = processor_type([usage_only])

    response = processor.build_base_response([usage_only])

    assert response["role"] == "assistant"
    assert processor.first_chunk is usage_only
    assert usage_only["choices"] == []
    assert processor.seen_chunks[0]["choices"][0]["delta"]["role"] == "assistant"


def test_compat_preserves_provider_role_and_is_idempotent() -> None:
    processor_type = vulnerable_processor_type()
    install_stream_chunk_builder_compat(processor_type)
    assert install_stream_chunk_builder_compat(processor_type) is False
    provider_chunk = chunk(
        choices=[{"delta": {"role": "tool"}, "finish_reason": None}]
    )
    processor = processor_type([provider_chunk])

    response = processor.build_base_response([provider_chunk])

    assert response["role"] == "tool"
    assert processor.seen_chunks == [provider_chunk]


def test_compat_does_not_patch_fixed_processor() -> None:
    assert install_stream_chunk_builder_compat(FixedProcessor) is False
