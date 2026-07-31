"""Supporting implementation for local commands."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from litecoder.cli.command_audit import CommandAuditRecorder
from litecoder.cli.tasks import render_task_detail, render_task_list
from litecoder.common.locks import NamedFileLock
from litecoder.common.trace import SecretRedactor
from litecoder.context.manager import ContextManager
from litecoder.memory.models import validate_memory_name
from litecoder.memory.store import MemoryStore
from litecoder.paths import AppPaths
from litecoder.tasks.store import TaskStore

if TYPE_CHECKING:
    from litecoder.agent.runtime import AgentRuntime


@dataclass(frozen=True, slots=True)
class LocalCommandSpec:
    """Display and usage metadata for one local command."""

    name: str
    usage: str
    description: str


LOCAL_COMMAND_SPECS = (
    LocalCommandSpec("/clear", "/clear", "Start a new session context."),
    LocalCommandSpec("/compact", "/compact", "Compact the active session context."),
    LocalCommandSpec("/context", "/context", "Show context and token usage."),
    LocalCommandSpec("/exit", "/exit", "Leave the interactive interface."),
    LocalCommandSpec("/help", "/help", "Show local command help."),
    LocalCommandSpec("/memory", "/memory [name]", "Inspect workspace memory."),
    LocalCommandSpec(
        "/model",
        "/model [provider] [model]",
        "Show or change the selected model.",
    ),
    LocalCommandSpec("/tasks", "/tasks [task-id]", "List or inspect tasks."),
    LocalCommandSpec("/trace", "/trace", "Show trace and command-audit locations."),
)

LOCAL_COMMANDS = frozenset(spec.name for spec in LOCAL_COMMAND_SPECS)

_HELP = "Local commands:\n" + "\n".join(
    f"  {spec.usage}" for spec in LOCAL_COMMAND_SPECS
)


@dataclass(frozen=True, slots=True)
class LocalCommandResult:
    """Data model representing the local command result."""
    handled: bool
    forward_to_model: bool = False
    message: str = ""
    exit_requested: bool = False
    clear_requested: bool = False
    replacement_session_id: str | None = None
    audit_status: str = "success"
    audit_code: str = "ok"
    audit_outcome: str | None = None


class LocalCommandRouter:
    """Component responsible for the local command router."""
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime
        paths = getattr(runtime, "paths", None)
        redactor = getattr(runtime, "trace_redactor", None)
        if not isinstance(redactor, SecretRedactor):
            redactor = SecretRedactor.with_values(())
        self.audit = (
            CommandAuditRecorder(paths, redactor)
            if isinstance(paths, AppPaths)
            else None
        )

    def names(self) -> list[str]:
        """Handle the names operation."""
        return sorted(LOCAL_COMMANDS)

    def command_specs(self) -> tuple[LocalCommandSpec, ...]:
        """Return the command metadata used by interactive command completion."""
        return LOCAL_COMMAND_SPECS

    async def dispatch(
        self,
        text: str,
        *,
        session_id: str | None,
    ) -> LocalCommandResult:
        """Handle the dispatch operation."""
        parts = text.strip().split()
        command = parts[0] if parts else ""
        arguments = parts[1:]
        if not command.startswith("/"):
            return LocalCommandResult(False, forward_to_model=True)
        if self.audit is None:
            return await self._execute(command, arguments, session_id)

        root_session_id = await self._root_session_id(session_id)
        operation = self.audit.operation(
            command,
            arguments,
            session_id,
            root_session_id,
        )
        try:
            await operation.start()
        except asyncio.CancelledError as cancellation:
            await operation.cancel()
            raise cancellation
        except Exception:
            return _failed(
                "Command audit is unavailable; command was not executed.",
                code="audit_unavailable",
            )

        try:
            result = await self._execute(command, arguments, session_id)
        except asyncio.CancelledError as cancellation:
            await operation.cancel()
            raise cancellation
        except Exception as error:
            await operation.fail(error)
            raise

        try:
            await operation.finish(
                status=result.audit_status,
                code=result.audit_code,
                outcome=result.audit_outcome,
                message=result.message,
                exit_requested=result.exit_requested,
                clear_requested=result.clear_requested,
                replacement_session_id=result.replacement_session_id,
            )
        except Exception:
            result = _with_audit_warning(result)
        if command == "/trace" and not arguments:
            result = self._with_audit_status(result)
        return result

    async def _execute(
        self,
        command: str,
        arguments: list[str],
        session_id: str | None,
    ) -> LocalCommandResult:
        if command not in LOCAL_COMMANDS:
            return _rejected(
                f"Unknown local command: {command}",
                code="unknown_command",
            )
        try:
            return await self._dispatch_known(command, arguments, session_id)
        except KeyError as error:
            return _rejected(_error_message(error), code="not_found")
        except ValueError as error:
            return _rejected(_error_message(error), code="invalid_argument")
        except RuntimeError as error:
            return _failed(_error_message(error), code="runtime_error")

    async def _root_session_id(self, session_id: str | None) -> str | None:
        if session_id is None:
            return None
        store = getattr(self.runtime, "store", None)
        resolver = getattr(store, "root_session_id", None)
        if not callable(resolver):
            return session_id
        try:
            resolved = await resolver(session_id)
        except Exception:
            return session_id
        return resolved if isinstance(resolved, str) else session_id

    def _with_audit_status(
        self,
        result: LocalCommandResult,
    ) -> LocalCommandResult:
        if self.audit is None:
            return result
        try:
            audit_status = self.audit.status()
        except (OSError, RuntimeError, ValueError):
            audit_status = "Command audit is unavailable"
        message = (
            f"{result.message}\n{audit_status}"
            if result.message
            else audit_status
        )
        return replace(result, message=message)

    async def _dispatch_known(
        self,
        command: str,
        arguments: list[str],
        session_id: str | None,
    ) -> LocalCommandResult:
        if command == "/clear":
            if arguments:
                return _rejected("Usage: /clear", code="usage")
            return LocalCommandResult(
                True,
                clear_requested=True,
                audit_outcome="cleared",
            )
        if command == "/exit":
            if arguments:
                return _rejected("Usage: /exit", code="usage")
            return LocalCommandResult(
                True,
                exit_requested=True,
                audit_outcome="exit_requested",
            )
        if command == "/help":
            if arguments:
                return _rejected("Usage: /help", code="usage")
            return _diagnostic(_HELP, outcome="shown")
        if command == "/compact":
            return await self._compact(arguments, session_id)
        if command == "/context":
            return await self._context(arguments, session_id)
        if command == "/memory":
            return await self._memory(arguments)
        if command == "/tasks":
            return await self._tasks(arguments)
        if command == "/trace":
            return await self._trace(arguments, session_id)
        return await self._model(arguments, session_id)

    async def _compact(
        self,
        arguments: list[str],
        session_id: str | None,
    ) -> LocalCommandResult:
        if arguments:
            return _rejected("Usage: /compact", code="usage")
        if session_id is None:
            return _rejected(
                "No active session to compact.",
                code="no_active_session",
            )
        report = await self.runtime.compact_session(session_id)
        before = report.before_tokens
        after = report.after_tokens
        saved = getattr(report, "saved_tokens", max(0, before - after))
        summary = "yes" if report.summary_created else "no"
        if after >= before and not report.summary_created:
            reason = getattr(report, "reason", "unspecified")
            if reason == "unspecified":
                reason = "no_reduction"
            return _diagnostic(
                "Context not compacted: "
                f"before={before} after={after} saved={saved} "
                f"summary=no reason={reason}",
                outcome="no_change",
            )
        return _diagnostic(
            "Context compacted: "
            f"before={before} after={after} saved={saved} summary={summary}",
            outcome="compacted",
        )

    async def _memory(self, arguments: list[str]) -> LocalCommandResult:
        if len(arguments) > 1:
            return _rejected("Usage: /memory [name]", code="usage")
        memory_root = self.runtime.paths.workspace_root / ".memory"
        if arguments:
            validate_memory_name(arguments[0])
        try:
            memory_root.lstat()
        except FileNotFoundError:
            memory_missing = True
        except OSError as error:
            raise ValueError("Memory is unavailable") from error
        else:
            memory_missing = False
        if memory_missing:
            if not arguments:
                return _diagnostic("No memory entries.", outcome="empty")
            raise KeyError(f"Unknown memory {arguments[0]!r}")
        memory = MemoryStore(
            memory_root,
            file_lock=NamedFileLock.memory(
                self.runtime.paths.project_id,
                self.runtime.paths.lock_dir,
            ),
        )
        if not arguments:
            index = memory.read_index().rstrip()
            if not index:
                return _diagnostic("No memory entries.", outcome="empty")
            return _diagnostic(
                index,
                outcome="listed",
            )
        name = arguments[0]
        index = memory.scan()
        selected = next(
            (entry for entry in index if entry.name.casefold() == name.casefold()),
            None,
        )
        if selected is None:
            raise KeyError(f"Unknown memory {name!r}")
        entry = memory.read(selected.name)
        return _diagnostic(
            entry.render().rstrip(),
            outcome="shown",
        )

    async def _tasks(self, arguments: list[str]) -> LocalCommandResult:
        if len(arguments) > 1:
            return _rejected("Usage: /tasks [task-id]", code="usage")
        records = TaskStore(self.runtime.paths.project_dir / "tasks").read_all()
        if not arguments:
            return _diagnostic(render_task_list(records), outcome="listed")
        return _diagnostic(
            render_task_detail(records, arguments[0]),
            outcome="shown",
        )

    async def _context(
        self,
        arguments: list[str],
        session_id: str | None,
    ) -> LocalCommandResult:
        if arguments:
            return _rejected("Usage: /context", code="usage")
        if session_id is None:
            return _rejected(
                "No active session to inspect.",
                code="no_active_session",
            )
        context = await self.runtime.store.load_context(session_id)
        statistics = await ContextManager(
            self.runtime.store,
            model=context.session.model,
        ).statistics(session_id)
        return _diagnostic(
            "Context: "
            f"session={context.session.id} "
            f"provider={context.session.provider} "
            f"model={context.session.model} "
            f"messages={statistics.persisted_messages} "
            f"context_tokens={statistics.effective_tokens}",
            outcome="inspected",
        )

    async def _trace(
        self,
        arguments: list[str],
        session_id: str | None,
    ) -> LocalCommandResult:
        if arguments:
            return _rejected("Usage: /trace", code="usage")
        if session_id is None:
            return _rejected(
                "No active session to trace.",
                code="no_active_session",
            )
        root_session_id = await self.runtime.store.root_session_id(session_id)
        try:
            path = self.runtime.paths.trace_path(root_session_id)
        except ValueError:
            return _failed("Trace is unavailable", code="trace_unavailable")
        if not path.exists():
            return _diagnostic(
                f"Trace: path={path} status=missing",
                outcome="inspected",
            )
        try:
            with path.open("r", encoding="utf-8") as handle:
                events = sum(1 for _ in handle)
        except (OSError, UnicodeError) as error:
            raise RuntimeError("Trace is unavailable") from error
        return _diagnostic(
            f"Trace: path={path} status=present events={events}",
            outcome="inspected",
        )

    async def _model(
        self,
        arguments: list[str],
        session_id: str | None,
    ) -> LocalCommandResult:
        if len(arguments) > 2:
            return _rejected(
                "Usage: /model [provider] [model]",
                code="usage",
            )
        if not arguments:
            current_provider = self.runtime.provider_name
            current_model = self.runtime.model
            if session_id is not None:
                context = await self.runtime.store.load_context(session_id)
                current_provider = context.session.provider
                current_model = context.session.model
            current_label = (
                current_model
                if current_model is not None and current_model.strip()
                else "(no model configured)"
            )
            configured = "\n".join(
                f"  {name} {model if model is not None and model.strip() else '(no model configured)'}"
                for name, model in sorted(
                    self.runtime.provider_models.items()
                )
            )
            return _diagnostic(
                f"Current: {current_provider} {current_label}\n"
                f"Configured:\n{configured}",
                outcome="inspected",
            )
        if session_id is None:
            return _rejected(
                "No active session to switch.",
                code="no_active_session",
            )
        provider = arguments[0]
        if provider not in self.runtime.provider_models:
            return _rejected(
                f"Unknown provider {provider!r}.",
                code="unknown_provider",
            )
        model = (
            arguments[1]
            if len(arguments) == 2
            else self.runtime.provider_models[provider]
        )
        if model is None or not model.strip():
            return _rejected(
                f"Provider {provider!r} has no configured model.",
                code="model_unavailable",
            )
        result = await self.runtime.switch_provider(session_id, provider, model)
        return LocalCommandResult(
            True,
            message=f"Switched to {provider} {model}; session={result.session_id}",
            replacement_session_id=result.session_id,
            audit_outcome="switched",
        )


def _diagnostic(
    message: str,
    *,
    outcome: str | None = None,
) -> LocalCommandResult:
    return LocalCommandResult(
        True,
        message=message,
        audit_outcome=outcome,
    )


def _rejected(
    message: str,
    *,
    code: str,
) -> LocalCommandResult:
    return LocalCommandResult(
        True,
        message=message,
        audit_status="rejected",
        audit_code=code,
    )


def _failed(
    message: str,
    *,
    code: str,
) -> LocalCommandResult:
    return LocalCommandResult(
        True,
        message=message,
        audit_status="failed",
        audit_code=code,
    )


def _with_audit_warning(result: LocalCommandResult) -> LocalCommandResult:
    warning = "Command completed, but its audit completion could not be recorded."
    message = f"{result.message}\n{warning}" if result.message else warning
    return replace(result, message=message)


def _error_message(error: BaseException) -> str:
    for argument in error.args:
        if isinstance(argument, str):
            return argument
    return type(error).__name__
