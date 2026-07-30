"""Provider-specific context summarization adapters."""

from __future__ import annotations

import json

from litecoder.context.compaction import CompactionUnavailable, SummaryRequest
from litecoder.agent.prompt_policy import CONTEXT_COMPACTION_SYSTEM_PROMPT
from litecoder.providers.base import ModelProvider
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
        for attempt in range(2):
            model_request = ModelRequest(
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
                max_tokens=self.max_tokens,
            )
            blocks: dict[int, str] = {}
            deltas: list[str] = []
            stop_reason: StopReason | None = None
            async for event in self.provider.stream(model_request):
                if event.type == "provider.error":
                    raise RuntimeError("summary generation failed")
                if event.type == "text.delta" and isinstance(event.delta, str):
                    deltas.append(event.delta)
                elif (
                    event.type == "content.completed"
                    and event.index is not None
                    and isinstance(event.block, dict)
                    and event.block.get("type") == "text"
                    and isinstance(event.block.get("text"), str)
                ):
                    blocks[event.index] = event.block["text"]
                elif event.type == "response.completed":
                    stop_reason = event.stop_reason
            if stop_reason is StopReason.MAX_TOKENS and attempt == 0:
                continue
            if stop_reason is StopReason.MAX_TOKENS:
                raise CompactionUnavailable(
                    "summary generation exceeded output limit"
                )
            if stop_reason is not StopReason.END_TURN:
                raise RuntimeError("summary generation did not complete")
            text = (
                "".join(blocks[index] for index in sorted(blocks))
                if blocks
                else "".join(deltas)
            ).strip()
            if not text:
                raise RuntimeError("summary generation returned empty text")
            return text
        raise AssertionError("summary retry loop exhausted")
