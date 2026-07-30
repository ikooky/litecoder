"""Supporting implementation for manual compaction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from litecoder.context.compaction import CompactionUnavailable
from litecoder.context.manager import ContextManager
from litecoder.context.session.store import SQLiteSessionStore


@dataclass(frozen=True, slots=True)
class ManualCompactionReport:
    """Data model representing the manual compaction report."""
    before_tokens: int
    after_tokens: int
    summary_created: bool
    reason: str = "unspecified"

    @property
    def saved_tokens(self) -> int:
        """Handle the saved tokens operation."""
        return max(0, self.before_tokens - self.after_tokens)


ContextManagerFactory = Callable[[str, str, int], ContextManager]


class ManualCompactor:
    """Component responsible for the manual compactor."""
    def __init__(
        self,
        store: SQLiteSessionStore,
        manager_factory: ContextManagerFactory,
    ) -> None:
        self.store = store
        self.manager_factory = manager_factory

    async def compact(self, session_id: str) -> ManualCompactionReport:
        """Compact the selected context or session."""
        loaded = await self.store.load_context(session_id)
        probe = ContextManager(self.store, model=loaded.session.model)
        before = await probe.statistics(session_id)
        if before.effective_tokens == 0:
            return ManualCompactionReport(0, 0, False, "empty_context")
        target = max(64, before.effective_tokens * 2 // 3)
        if target >= before.effective_tokens:
            return ManualCompactionReport(
                before.effective_tokens,
                before.effective_tokens,
                False,
                "history_too_small",
            )
        manager = self.manager_factory(
            loaded.session.provider,
            loaded.session.model,
            target,
        )
        try:
            result = await manager.compact(session_id, force_summary=True)
        except CompactionUnavailable as error:
            return ManualCompactionReport(
                before.effective_tokens,
                before.effective_tokens,
                False,
                _compaction_unavailable_reason(error),
            )
        after = await probe.statistics(session_id)
        return ManualCompactionReport(
            before.effective_tokens,
            after.effective_tokens,
            result.summary is not None,
            (
                "compacted"
                if after.effective_tokens < before.effective_tokens
                else "no_reduction"
            ),
        )


def _compaction_unavailable_reason(error: CompactionUnavailable) -> str:
    message = str(error).casefold()
    if "existing summary" in message:
        return "existing_summary_over_budget"
    if "generation exceeded output limit" in message:
        return "summary_generation_limit"
    if "summary output" in message:
        return "summary_over_budget"
    return "compaction_unavailable"
