"""Base interfaces for evaluation modes or providers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from litecoder.providers.models import ModelRequest, ProviderEvent


@runtime_checkable
class ModelProvider(Protocol):
    """Protocol describing the model provider behavior."""
    def stream(self, request: ModelRequest) -> AsyncIterator[ProviderEvent]: ...
