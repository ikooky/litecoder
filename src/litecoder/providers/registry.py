"""Provider registration and lookup."""

from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any

from litecoder.providers.base import ModelProvider
from litecoder.settings import ProviderSettings


ProviderCall = Callable[..., Any]


async def close_default_async_clients() -> None:
    """Release LiteLLM's process-wide async HTTP clients when available."""
    try:
        import litellm
    except ModuleNotFoundError as error:
        if error.name == "litellm":
            return
        raise
    close = getattr(litellm, "close_litellm_async_clients", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


class ProviderRegistry:
    """Registry for the provider registry."""
    def __init__(
        self,
        *,
        completion: ProviderCall | None = None,
        responses: ProviderCall | None = None,
    ) -> None:
        self._completion = completion
        self._responses = responses

    def create(self, name: str, settings: ProviderSettings) -> ModelProvider:
        """Create the requested object."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Provider name must be a non-empty string")
        if settings.type not in {
            "anthropic-messages",
            "openai-chat-completions",
            "openai-responses",
        }:
            raise ValueError(f"Unsupported provider type for {name!r}")
        model = settings.model
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Provider {name!r} requires a non-empty model")
        if (
            settings.api_key is None
            or not settings.api_key.get_secret_value().strip()
        ):
            raise ValueError(f"Provider {name!r} requires an API key")
        if settings.base_url is not None and not settings.base_url.strip():
            raise ValueError(f"Provider {name!r} has an invalid base URL")
        from litecoder.providers.compatible import CompatibleProvider

        completion = self._completion
        responses = self._responses
        if completion is None or responses is None:
            loaded_completion, loaded_responses = _load_provider_calls()
            completion = completion or loaded_completion
            responses = responses or loaded_responses
        return CompatibleProvider(
            completion=completion,
            responses=responses,
            model=model,
            api_style=settings.type,
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
        )


def _load_provider_calls() -> tuple[ProviderCall, ProviderCall]:
    """Load the provider calls."""
    try:
        import litellm
    except ModuleNotFoundError as error:
        if error.name != "litellm":
            raise
        raise RuntimeError(
            "Provider support is optional; install it with pip install 'litecoder[providers]'"
        ) from None
    from litecoder.providers._litellm_compat import (
        install_stream_chunk_builder_compat,
    )

    try:
        from litellm.litellm_core_utils.streaming_chunk_builder_utils import (
            ChunkProcessor,
        )
    except (ImportError, AttributeError):
        pass
    else:
        install_stream_chunk_builder_compat(ChunkProcessor)
    return litellm.acompletion, litellm.aresponses
