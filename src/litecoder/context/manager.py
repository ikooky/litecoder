"""Context assembly, persistence, and compaction coordination."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from litecoder.common.trace import current_secret_redactor
from litecoder.context.compaction import (
    CompactionPolicy,
    CompactionResult,
    CompactionUnavailable,
    SummaryRequest,
    Summarizer,
    estimate_message_tokens,
)
from litecoder.context.prompt import (
    PromptAssembler,
    PromptInputs,
    load_project_instructions,
)
from litecoder.context.prompt_state import PromptState, PromptStateProvider
from litecoder.context.session.models import MessageRecord
from litecoder.context.token_budget import estimate_tokens
from litecoder.context.session.store import SQLiteSessionStore
from litecoder.agent.prompt_policy import (
    CORE_AGENT_INSTRUCTIONS,
    DURABLE_MEMORY_INSTRUCTIONS,
)
from litecoder.memory.diagnostics import memory_diagnostic
from litecoder.memory.loading import LoadedMemories
from litecoder.memory.service import MemoryService
from litecoder.providers.models import ModelRequest
from litecoder.tools.registry import ToolRegistry
from litecoder.tools.skills import (
    DEFAULT_SKILL_CATALOG_CHARS,
    SKILL_CATALOG_CHARS_PER_TOKEN,
    SKILL_CATALOG_CONTEXT_PERCENT,
    SkillCatalog,
)


SkillCatalogResolver = Callable[[Path], SkillCatalog]



_TOTAL_BUDGET_ENFORCEMENT_MINIMUM = 4_096
_MINIMUM_PROMPT_SECTION_BYTES = 128 * 8

@dataclass(frozen=True, slots=True)
class ContextStatistics:
    """Data model representing the context statistics."""
    persisted_messages: int
    effective_tokens: int


class ContextManager:
    """Manager coordinating the context manager."""
    def __init__(
        self,
        store: SQLiteSessionStore,
        *,
        model: str,
        max_tokens: int = 32_000,
        compaction_policy: CompactionPolicy | None = None,
        context_budget_tokens: int | None = None,
        summarizer: Summarizer | None = None,
        agent_instructions: str | None = None,
        skill_catalog: SkillCatalog | None = None,
        skill_catalog_resolver: SkillCatalogResolver | None = None,
        memory_service: MemoryService | None = None,
        prompt_assembler: PromptAssembler | None = None,
        prompt_state_provider: PromptStateProvider | None = None,
        prompt_task_ids: frozenset[str] | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if context_budget_tokens is not None and (
            isinstance(context_budget_tokens, bool)
            or not isinstance(context_budget_tokens, int)
            or context_budget_tokens < 0
        ):
            raise ValueError("context_budget_tokens must be a non-negative integer")
        if (compaction_policy is None) != (context_budget_tokens is None):
            raise ValueError(
                "compaction_policy and context_budget_tokens must be configured together"
            )
        if skill_catalog is not None and skill_catalog_resolver is not None:
            raise ValueError(
                "skill_catalog and skill_catalog_resolver are mutually exclusive"
            )
        self.store = store
        self.model = model
        self.max_tokens = max_tokens
        self.compaction_policy = compaction_policy
        self.context_budget_tokens = context_budget_tokens
        self.summarizer = summarizer
        self.skill_catalog = skill_catalog or SkillCatalog({})
        self.skill_catalog_resolver = skill_catalog_resolver
        self.memory_service = memory_service
        self._loaded_memories: LoadedMemories | None = None
        self._memory_load_task: asyncio.Task[LoadedMemories] | None = None
        self._loaded_memory_count = 0
        self._pending_memory_diagnostics: list[dict[str, object]] = []
        self._prompt_section_tokens: dict[str, int] = {}
        self._durable_memory_section_tokens = 0
        self._all_memory_tokens = 0
        self._memory_index_tokens = 0
        self._recalled_memory_tokens = 0
        self._optimized_memory_tokens = 0
        self._memory_context_tokens = 0
        self._memory_catalog_reduction = 0.0
        self._memory_recalled_ids: tuple[str, ...] = ()
        self.prompt_assembler = prompt_assembler or PromptAssembler()
        self.prompt_state_provider = prompt_state_provider
        if prompt_task_ids is not None and any(
            not isinstance(task_id, str) or not task_id
            for task_id in prompt_task_ids
        ):
            raise ValueError("prompt_task_ids must contain non-empty strings")
        if agent_instructions is not None and (
            not isinstance(agent_instructions, str) or not agent_instructions.strip()
        ):
            raise ValueError("agent_instructions must be non-empty text when provided")
        self.prompt_task_ids = prompt_task_ids
        self.agent_instructions = agent_instructions

    @property
    def loaded_memory_count(self) -> int:
        """Handle the loaded memory count operation."""
        return self._loaded_memory_count

    def consume_memory_diagnostics(self) -> tuple[dict[str, object], ...]:
        """Handle the consume memory diagnostics operation."""
        diagnostics = tuple(self._pending_memory_diagnostics)
        self._pending_memory_diagnostics.clear()
        return diagnostics

    def prompt_telemetry(self) -> dict[str, object]:
        """Return token telemetry for the most recently built request."""
        return {
            "prompt_section_tokens": dict(self._prompt_section_tokens),
            "durable_memory_section_tokens": self._durable_memory_section_tokens,
            "all_memory_tokens": self._all_memory_tokens,
            "memory_index_tokens": self._memory_index_tokens,
            "recalled_memory_tokens": self._recalled_memory_tokens,
            "optimized_memory_tokens": self._optimized_memory_tokens,
            "memory_context_tokens": self._memory_context_tokens,
            "memory_catalog_reduction": self._memory_catalog_reduction,
            "memory_recalled_ids": list(self._memory_recalled_ids),
        }

    @property
    def can_compact(self) -> bool:
        """Return whether the compact condition holds."""
        return (
            self.compaction_policy is not None
            and self.context_budget_tokens is not None
        )

    def configure_runtime_compaction(self, summarizer: Summarizer) -> None:
        """Connect safe runtime defaults without overriding explicit settings."""
        if self.compaction_policy is None:
            self.compaction_policy = CompactionPolicy()
            self.context_budget_tokens = 256_000
        if self.summarizer is None:
            self.summarizer = summarizer

    async def build_request(
        self, session_id: str, tools: ToolRegistry
    ) -> ModelRequest:
        """Build the request."""
        context = await self.store.load_context(session_id)
        tool_schemas = [
            {
                "name": tool.spec.name,
                "description": tool.spec.description,
                "input_schema": tool.spec.input_schema,
            }
            for tool in tools.list()
        ]
        workspace_root = Path(context.session.workspace_path)
        skill_catalog = self._skill_catalog_for(workspace_root)
        loaded = await self._load_memories(context.messages)
        memory_payload: object = []
        if self.memory_service is not None:
            memory_payload = await _safe_system_memory_payload(self.memory_service)
        if self.prompt_state_provider is not None:
            prompt_state = await self.prompt_state_provider.snapshot(
                session_id, task_ids=self.prompt_task_ids
            )
            prompt_todos: list[dict[str, object]] | None = prompt_state.todos
        else:
            prompt_state = PromptState(todos=[], tasks=[], team=[])
            prompt_todos = None
        prompt_total_bytes = _prompt_total_bytes(self.context_budget_tokens, tool_schemas)
        system = self.prompt_assembler.build(
            PromptInputs(
                identity=_identity_with_agent_instructions(self.agent_instructions),
                runtime={
                    "project_id": context.session.project_id,
                    "workspace_id": context.session.workspace_id,
                    "workspace_root": str(workspace_root),
                },
                project_instructions=load_project_instructions(workspace_root),
                skill_catalog=skill_catalog.prompt_metadata(
                    max_chars=_skill_catalog_budget_chars(self.context_budget_tokens)
                ),
                memories=memory_payload,
                todos=prompt_todos,
                tasks=prompt_state.tasks,
                team=prompt_state.team,
            ),
            total_max_bytes=prompt_total_bytes,
        )
        latest_summary, restored_messages = _restore_messages(context.messages)
        if self.compaction_policy is not None:
            history_budget = _history_budget_tokens(
                self.context_budget_tokens,
                system,
                tool_schemas,
                loaded.rendered,
            )
            compacted, latest_summary = await self._compact_loaded(
                session_id,
                latest_summary,
                restored_messages,
                force_summary=False,
                target_budget_tokens=history_budget,
            )
            restored_messages = compacted.messages
        if latest_summary is not None:
            system.extend(_system_blocks(latest_summary.content))

        request_messages: list[dict[str, object]] = []
        for message in restored_messages:
            if message.role == "system":
                system.extend(_system_blocks(message.content))
            elif message.role in {"user", "assistant"}:
                request_messages.append(
                    {
                        "role": message.role,
                        "content": copy.deepcopy(message.content),
                    }
                )
        _inject_loaded_memories(request_messages, loaded.rendered)
        self._record_prompt_telemetry(system, loaded)
        return ModelRequest(
            model=self.model,
            system=system,
            messages=request_messages,
            tools=tool_schemas,
            max_tokens=self.max_tokens,
        )

    def _record_prompt_telemetry(
        self, system: list[dict[str, object]], loaded: LoadedMemories
    ) -> None:
        section_tokens = _prompt_section_tokens(system)
        recalled_text = loaded.rendered
        recalled_tokens = estimate_tokens(recalled_text) if recalled_text else 0
        durable_tokens = section_tokens.get("memories", 0)
        optimized_tokens = loaded.memory_index_tokens + recalled_tokens
        self._prompt_section_tokens = section_tokens
        self._durable_memory_section_tokens = durable_tokens
        self._all_memory_tokens = loaded.all_memory_tokens
        self._memory_index_tokens = loaded.memory_index_tokens
        self._recalled_memory_tokens = recalled_tokens
        self._optimized_memory_tokens = optimized_tokens
        self._memory_context_tokens = durable_tokens + recalled_tokens
        self._memory_catalog_reduction = (
            1 - optimized_tokens / loaded.all_memory_tokens
            if loaded.all_memory_tokens
            else 0.0
        )
        self._memory_recalled_ids = loaded.selected_names

    def _skill_catalog_for(self, workspace_root: Path) -> SkillCatalog:
        if self.skill_catalog_resolver is None:
            return self.skill_catalog
        return self.skill_catalog_resolver(workspace_root)

    async def _load_memories(
        self, messages: list[MessageRecord]
    ) -> LoadedMemories:
        """Load the memories."""
        if self.memory_service is None:
            self._loaded_memory_count = 0
            return LoadedMemories((), "")
        if self._loaded_memories is not None:
            return self._loaded_memories
        task = self._memory_load_task
        if task is None:
            task = asyncio.create_task(self._load_memories_once(messages))
            self._memory_load_task = task
        return await asyncio.shield(task)

    async def _load_memories_once(
        self,
        messages: list[MessageRecord],
    ) -> LoadedMemories:
        """Load the memories once."""
        current_task = asyncio.current_task()
        try:
            try:
                loaded = await self.memory_service.load_memories(messages)  # type: ignore[union-attr]
            except Exception:
                loaded = LoadedMemories((), "")

            self._loaded_memories = loaded
            self._loaded_memory_count = len(loaded.entries)
            if loaded.entries:
                self._pending_memory_diagnostics.append(
                    memory_diagnostic(
                        "load",
                        "recalled",
                        count=len(loaded.entries),
                    )
                )
            return loaded
        finally:
            if self._memory_load_task is current_task:
                self._memory_load_task = None

    async def statistics(self, session_id: str) -> ContextStatistics:
        """Return context usage statistics."""
        context = await self.store.load_context(session_id)
        latest_summary, restored_messages = _restore_messages(context.messages)
        effective_messages = list(restored_messages)
        if latest_summary is not None:
            effective_messages.insert(0, latest_summary)
        return ContextStatistics(
            persisted_messages=len(context.messages),
            effective_tokens=estimate_message_tokens(effective_messages),
        )

    async def compact(
        self,
        session_id: str,
        *,
        force_summary: bool = False,
    ) -> CompactionResult:
        """Compact the selected context or session."""
        if self.compaction_policy is None:
            raise RuntimeError("context compaction is not configured")
        context = await self.store.load_context(session_id)
        latest_summary, restored_messages = _restore_messages(context.messages)
        result, _ = await self._compact_loaded(
            session_id,
            latest_summary,
            restored_messages,
            force_summary=force_summary,
        )
        return result

    async def compact_reactively(self, session_id: str) -> CompactionResult:
        """Handle the compact reactively operation."""
        if self.context_budget_tokens is None:
            raise RuntimeError("context compaction is not configured")
        target_budget_tokens = max(1, self.context_budget_tokens * 2 // 3)
        context = await self.store.load_context(session_id)
        latest_summary, restored_messages = _restore_messages(context.messages)
        result, _ = await self._compact_loaded(
            session_id,
            latest_summary,
            restored_messages,
            force_summary=True,
            target_budget_tokens=target_budget_tokens,
        )
        return result

    async def _compact_loaded(
        self,
        session_id: str,
        latest_summary: MessageRecord | None,
        restored_messages: list[MessageRecord],
        *,
        force_summary: bool = False,
        target_budget_tokens: int | None = None,
    ) -> tuple[CompactionResult, MessageRecord | None]:
        if self.compaction_policy is None or self.context_budget_tokens is None:
            raise RuntimeError("context compaction is not configured")
        budget_tokens = self.context_budget_tokens
        if target_budget_tokens is not None:
            budget_tokens = target_budget_tokens

        previous_cutoff: int | None = None
        previous_text: str | None = None
        message_budget = budget_tokens
        if latest_summary is not None:
            previous_cutoff = _summary_cutoff(latest_summary)
            previous_text = _summary_text(latest_summary)
            if previous_cutoff is None or previous_text is None:
                raise RuntimeError("latest context summary is invalid")
            previous_tokens = estimate_message_tokens([latest_summary])
            if previous_tokens > budget_tokens:
                raise CompactionUnavailable(
                    "existing summary exceeds the context budget"
                )
            message_budget -= previous_tokens

        combined_request: SummaryRequest | None = None
        base_summarizer = self.summarizer
        active_summarizer = base_summarizer
        if (
            previous_cutoff is not None
            and previous_text is not None
            and base_summarizer is not None
        ):
            async def summarize_with_previous(
                request: SummaryRequest,
            ) -> str:
                """Summarize the with previous."""
                nonlocal combined_request
                combined_request = SummaryRequest(
                    covered_through_sequence=max(
                        previous_cutoff,
                        request.covered_through_sequence,
                    ),
                    messages=(
                        {
                            "role": "system",
                            "content": [{
                                "type": "text",
                                "text": previous_text,
                            }],
                        },
                        *request.messages,
                    ),
                )
                return await base_summarizer(combined_request)

            active_summarizer = summarize_with_previous

        compacted = await self.compaction_policy.compact(
            restored_messages,
            message_budget,
            active_summarizer,
            summary_budget_tokens=budget_tokens,
            force_summary=force_summary,
        )
        if compacted.summary is None:
            return compacted, latest_summary
        effective_request = combined_request or compacted.summary_request
        if effective_request is None:
            raise RuntimeError("summary result is missing its request")

        redacted = current_secret_redactor().redact_text(compacted.summary)
        if not redacted.strip():
            raise ValueError("redacted summary output must not be empty")
        summary_record = MessageRecord(
            session_id=session_id,
            role="system",
            content=[{
                "type": "context_summary",
                "covered_through_sequence": (
                    effective_request.covered_through_sequence
                ),
                "text": redacted,
            }],
        )
        persisted_suffix = [
            message
            for message in restored_messages
            if (
                isinstance(message.sequence, int)
                and not isinstance(message.sequence, bool)
                and message.sequence > effective_request.covered_through_sequence
            )
        ]
        if (
            estimate_message_tokens([summary_record, *persisted_suffix])
            > budget_tokens
        ):
            raise CompactionUnavailable(
                "summary output exceeds the context budget"
            )
        await self.store.append_message(summary_record)
        safe_result = CompactionResult(
            compacted.messages,
            redacted,
            effective_request,
        )
        return safe_result, summary_record


async def _safe_system_memory_payload(
    service: MemoryService,
) -> dict[str, object]:
    """Load lock-backed prompt memory without blocking the event loop."""
    try:
        loader = getattr(service, "system_payload_async", None)
        if callable(loader):
            payload = await loader()
        else:
            payload = await asyncio.to_thread(service.system_payload)
    except Exception:
        return _memory_prompt_fallback()
    return (
        payload if isinstance(payload, dict) else _memory_prompt_fallback()
    )


def _prompt_section_tokens(system: list[dict[str, object]]) -> dict[str, int]:
    """Estimate tokens for each named prompt section."""
    sections: dict[str, int] = {}
    for index, block in enumerate(system):
        text = block.get("text")
        if not isinstance(text, str):
            continue
        name = f"system_{index}"
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("name"), str):
            name = parsed["name"]
        sections[name] = sections.get(name, 0) + estimate_tokens(text)
    return sections


def _memory_prompt_fallback() -> dict[str, object]:
    return {
        "index": "",
        "instructions": list(DURABLE_MEMORY_INSTRUCTIONS),
    }

def _prompt_total_bytes(
    context_budget_tokens: int | None, tool_schemas: list[dict[str, object]]
) -> int | None:
    if (
        context_budget_tokens is None
        or context_budget_tokens < _TOTAL_BUDGET_ENFORCEMENT_MINIMUM
    ):
        return None
    available_tokens = context_budget_tokens - _json_token_count(tool_schemas)
    return max(_MINIMUM_PROMPT_SECTION_BYTES, available_tokens * 4)


def _identity_with_agent_instructions(agent_instructions: str | None) -> str:
    if agent_instructions is None:
        return CORE_AGENT_INSTRUCTIONS
    return f"{CORE_AGENT_INSTRUCTIONS}\n\n{agent_instructions}"


def _skill_catalog_budget_chars(context_budget_tokens: int | None) -> int:
    if context_budget_tokens is None:
        return DEFAULT_SKILL_CATALOG_CHARS
    return max(
        256,
        int(
            context_budget_tokens
            * SKILL_CATALOG_CHARS_PER_TOKEN
            * SKILL_CATALOG_CONTEXT_PERCENT
        ),
    )


def _history_budget_tokens(
    context_budget_tokens: int | None,
    system: list[dict[str, object]],
    tool_schemas: list[dict[str, object]],
    loaded_memories: str,
) -> int | None:
    if (
        context_budget_tokens is None
        or context_budget_tokens < _TOTAL_BUDGET_ENFORCEMENT_MINIMUM
    ):
        return None
    static_tokens = (
        _json_token_count(system)
        + _json_token_count(tool_schemas)
        + estimate_tokens(loaded_memories)
    )
    return max(1, context_budget_tokens - static_tokens)


def _json_token_count(value: object) -> int:
    rendered = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return estimate_tokens(rendered)


def _inject_loaded_memories(
    messages: list[dict[str, object]], rendered: str
) -> None:
    if not rendered:
        return
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if not any(
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            for block in content
        ):
            continue
        message["content"] = [
            {"type": "text", "text": rendered},
            *copy.deepcopy(content),
        ]
        return


def _restore_messages(
    messages: list[MessageRecord],
) -> tuple[MessageRecord | None, list[MessageRecord]]:
    latest: MessageRecord | None = None
    cutoff: int | None = None
    for message in reversed(messages):
        candidate = _summary_cutoff(message)
        if candidate is None:
            continue
        latest = message
        cutoff = candidate
        break
    if latest is None or cutoff is None:
        return None, [
            message
            for message in messages
            if not _is_malformed_context_summary(message)
        ]
    return latest, [
        message
        for message in messages
        if (
            isinstance(message.sequence, int)
            and not isinstance(message.sequence, bool)
            and message.sequence > cutoff
            and not _is_compaction_summary(message)
        )
    ]


def _summary_cutoff(message: MessageRecord) -> int | None:
    if message.role != "system" or len(message.content) != 1:
        return None
    block = message.content[0]
    if block.get("type") != "context_summary":
        return None
    cutoff = block.get("covered_through_sequence")
    sequence = message.sequence
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, int)
        or cutoff <= 0
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or cutoff >= sequence
        or _summary_text(message) is None
    ):
        return None
    return cutoff


def _summary_text(message: MessageRecord) -> str | None:
    if message.role != "system" or len(message.content) != 1:
        return None
    block = message.content[0]
    text = block.get("text")
    if block.get("type") != "context_summary" or not isinstance(text, str):
        return None
    return text if text.strip() else None


def _is_compaction_summary(message: MessageRecord) -> bool:
    return any(
        block.get("type") == "context_summary"
        and "covered_through_sequence" in block
        for block in message.content
    )


def _is_malformed_context_summary(message: MessageRecord) -> bool:
    return any(
        block.get("type") == "context_summary"
        and "covered_through_sequence" in block
        for block in message.content
    ) and _summary_cutoff(message) is None


def _system_blocks(content: list[dict[str, object]]) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    for block in content:
        if block.get("type") not in {"context_summary", "text"}:
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            blocks.append({"type": "text", "text": text})
    return blocks
