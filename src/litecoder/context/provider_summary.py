"""Provider-specific context summarization adapters."""

from __future__ import annotations

import json

from litecoder.context.compaction import CompactionUnavailable, SummaryRequest
from litecoder.agent.prompt_policy import CONTEXT_COMPACTION_SYSTEM_PROMPT
from litecoder.providers.base import ModelProvider
from litecoder.providers.completion import complete_text_with_retry
from litecoder.providers.models import ModelRequest, StopReason


class ProviderContextSummarizer:
    """Component responsible for the provider context summarizer."""
    def __init__(
        self,
        provider: ModelProvider,
        model: str,
        *,
        max_tokens: int = 8_000,
    ) -> None:
        if not model.strip():
            raise ValueError("summary model must not be empty")
        if max_tokens <= 0:
            raise ValueError("summary max_tokens must be positive")
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens

    async def __call__(self, request: SummaryRequest) -> str:
        """Handle the value."""
        def request_factory(max_tokens: int) -> ModelRequest:
            return ModelRequest(
                model=self.model,
                system=[{"type": "text", "text": CONTEXT_COMPACTION_SYSTEM_PROMPT}],
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": json.dumps(
                            {
                                "covered_through_sequence": request.covered_through_sequence,
                                "messages": request.messages,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }],
                }],
                tools=[],
                max_tokens=max_tokens,
            )

        result = await complete_text_with_retry(
            self.provider,
            request_factory,
            initial_max_tokens=self.max_tokens,
        )
        if result.provider_error is not None:
            raise RuntimeError("summary generation failed") from result.provider_error
        if result.stop_reason is StopReason.MAX_TOKENS:
            raise CompactionUnavailable("summary generation exceeded output limit")
        if result.stop_reason is not StopReason.END_TURN:
            raise RuntimeError("summary generation did not complete")
        text = result.text.strip()
        if not text:
            raise RuntimeError("summary generation returned empty text")
        return text
