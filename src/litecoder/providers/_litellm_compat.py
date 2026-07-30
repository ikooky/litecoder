"""LiteLLM compatibility helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from litecoder.providers._adapter import field_value


_PATCH_MARKER = "_litecoder_empty_choices_compat"


def install_stream_chunk_builder_compat(processor_type: type[Any]) -> bool:
    """Patch vulnerable LiteLLM chunk processors without pinning a private fork."""

    original = getattr(processor_type, "build_base_response", None)
    if not callable(original) or getattr(original, _PATCH_MARKER, False):
        return False
    if not _has_empty_choices_bug(processor_type):
        return False

    @wraps(original)
    def build_base_response(processor: Any, chunks: list[Any]) -> Any:
        if _first_stream_role(chunks) is not None:
            return original(processor, chunks)

        synthetic = _assistant_role_chunk(getattr(processor, "first_chunk", None))
        previous_first = getattr(processor, "first_chunk", None)
        processor.first_chunk = synthetic
        try:
            return original(processor, [synthetic, *chunks])
        finally:
            processor.first_chunk = previous_first

    setattr(build_base_response, _PATCH_MARKER, True)
    processor_type.build_base_response = build_base_response
    return True


def _has_empty_choices_bug(processor_type: type[Any]) -> bool:
    chunk = {
        "id": "litecoder-compat-probe",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "compat-probe",
        "choices": [],
    }
    try:
        processor = processor_type([chunk])
        processor.build_base_response([chunk])
    except IndexError:
        return True
    except Exception:
        return False
    return False


def _first_stream_role(chunks: Sequence[Any]) -> str | None:
    for chunk in chunks:
        choices = field_value(chunk, "choices", ())
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            continue
        if not choices:
            continue
        delta = field_value(choices[0], "delta", None)
        role = field_value(delta, "role", None)
        if isinstance(role, str) and role:
            return role
    return None


def _assistant_role_chunk(first_chunk: Any) -> dict[str, Any]:
    hidden = field_value(first_chunk, "_hidden_params", None)
    chunk: dict[str, Any] = {
        "id": field_value(first_chunk, "id", "") or "",
        "object": field_value(first_chunk, "object", "chat.completion.chunk")
        or "chat.completion.chunk",
        "created": field_value(first_chunk, "created", 0) or 0,
        "model": field_value(first_chunk, "model", "") or "",
        "system_fingerprint": field_value(first_chunk, "system_fingerprint", None),
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": None},
                "finish_reason": None,
            }
        ],
    }
    if isinstance(hidden, Mapping):
        chunk["_hidden_params"] = dict(hidden)
    return chunk
