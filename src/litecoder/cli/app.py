"""Supporting implementation for app."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from rich.console import Console

import typer
from litecoder.agent.factory import DefaultAgentRuntimeFactory
from litecoder.agent.loop import AgentLoop, RuntimeBudgets
from litecoder.agent.runtime import AgentRuntime, RuntimeContext
from litecoder.cli import (
    config,
    mcp as mcp_cli,
    sessions as sessions_cli,
    tasks as tasks_cli,
)
from litecoder.cli.trace import trace_command
from litecoder.common.errors import RecoveryPolicy, RetryBudget
from litecoder.common.locks import NamedFileLock
from litecoder.common.trace import SecretRedactor
from litecoder.context.compaction import CompactionPolicy
from litecoder.context.manual_compaction import ManualCompactor
from litecoder.context.manager import ContextManager
from litecoder.context.provider_summary import ProviderContextSummarizer
from litecoder.context.prompt_state import PromptStateProvider
from litecoder.context.session.store import SQLiteSessionStore
from litecoder.context.todos import TodoService, register_todo_tools
from litecoder.hooks import HookManager, discover_command_hooks
from litecoder.memory import MemoryCoordinator, MemoryService, MemoryStore
from litecoder.paths import AppPaths
from litecoder.providers.base import ModelProvider
from litecoder.providers.registry import ProviderRegistry
from litecoder.settings import Settings, ensure_user_config
from litecoder.tasks import (
    AgentCaller,
    ChildAuthority,
    MessageBus,
    SubagentManager,
    TaskManager,
    TaskStore,
    TeamManager,
    WorktreeManager,
)
from litecoder.tools.artifacts import ProjectArtifactStores
from litecoder.tools.background import BackgroundManager, register_background_tools
from litecoder.tools.builtin import (
    register_agent_tools,
    register_builtin_tools,
    register_team_tools,
    register_worktree_tools,
)
from litecoder.tools.duplicate_guard import DuplicateGuard
from litecoder.tools.executor import ToolExecutor
from litecoder.tools.mcp import MCPConnectionManager
from litecoder.tools.memory import register_memory_tools
from litecoder.tools.models import ToolCall, ToolContext
from litecoder.tools.permission import (
    PermissionPrompt as ToolPermissionPrompt,
    PermissionService,
    Prompt as PermissionPromptCallback,
    PromptChoice,
)
from litecoder.tools.registry import ToolRegistry
from litecoder.tools.skills import LoadSkillTool, SkillCatalog
from litecoder.tools.tasks import register_task_tools
from litecoder.tools.workspace_version import WorkspaceStateRegistry
from litecoder.ui.events import UIEventFactory
from litecoder.ui.output_guard import TUIOutputGuard
from litecoder.ui.permissions import (
    select_permission_choice as _select_permission_choice,
)
from litecoder.ui.redaction import RedactingUISink
from litecoder.ui.renderers.terminal import TerminalRenderer, TerminalUISink
from litecoder.ui.sink import RuntimeUISink
from litecoder.ui.tui import (
    LiteCoderApp,
    TextualPermissionPrompt,
    TextualUISink,
)


app = typer.Typer(no_args_is_help=False)
app.add_typer(config.app, name="config")
app.add_typer(mcp_cli.app, name="mcp")
app.add_typer(sessions_cli.app, name="sessions")
app.add_typer(tasks_cli.app, name="tasks")
app.command("trace")(trace_command)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Run the command-line entry point."""
    if ctx.invoked_subcommand is None:
        asyncio.run(_interactive())


@app.command("run")
def run_command(prompt: str) -> None:
    """Run one prompt in a new LiteCoder session."""
    asyncio.run(_run_once(prompt))


@app.command("resume")
def resume_command(
    session_id: Annotated[str | None, typer.Argument()] = None,
    prompt: str | None = None,
) -> None:
    """Resume an existing LiteCoder session."""
    if session_id is None:
        typer.echo(
            "A session ID is required until recent-session selection is available.",
            err=True,
        )
        raise typer.Exit(1)
    asyncio.run(_resume_once(session_id, prompt))


async def build_runtime(
    cwd: Path | None = None,
    *,
    ui_sink: RuntimeUISink | None = None,
    permission_prompt: PermissionPromptCallback | None = None,
    hook_registrar: Callable[[HookManager], None] | None = None,
    isolated_workspace: bool = False,
    runtime_budgets: RuntimeBudgets | None = None,
    tool_allowlist: frozenset[str] | None = None,
    context_compaction: bool | None = None,
    context_budget_tokens: int | None = None,
    memory_recall: bool = True,
) -> AgentRuntime:
    """Build the runtime."""
    if context_compaction is not None and not isinstance(
        context_compaction, bool
    ):
        raise ValueError("context_compaction must be a bool or None")
    if context_budget_tokens is not None and (
        context_compaction is not True
        or isinstance(context_budget_tokens, bool)
        or not isinstance(context_budget_tokens, int)
        or context_budget_tokens <= 0
    ):
        raise ValueError(
            "context_budget_tokens requires enabled context_compaction"
        )
    if not isinstance(memory_recall, bool):
        raise ValueError("memory_recall must be a bool")
    runtime_cwd = cwd or Path.cwd()
    paths = (
        AppPaths.discover(runtime_cwd, isolated=True)
        if isolated_workspace
        else AppPaths.discover(runtime_cwd)
    )
    created_config = ensure_user_config(paths)
    if created_config is not None:
        Console(stderr=True).print(
            "[dim]Created default configuration at "
            f"{created_config}. Set OPENAI_API_KEY to use gpt-5.6-sol.[/dim]"
        )
    settings = Settings.load(paths)
    provider_name = settings.default_provider or next(iter(settings.providers), None)
    if provider_name is None:
        raise ValueError("No provider is configured")
    provider_settings = settings.providers[provider_name]
    model = settings.default_model or provider_settings.model
    if not model:
        raise ValueError(f"Provider {provider_name!r} has no model configured")
    secret_environment_names, secret_values = _runtime_secrets(settings)
    redactor = SecretRedactor.with_values(secret_values)
    runtime_ui_sink = RedactingUISink(ui_sink, redactor) if ui_sink else None
    provider_registry = ProviderRegistry()
    providers: dict[tuple[str, str], ModelProvider] = {}
    store = SQLiteSessionStore(paths.sessions_db)
    try:
        await store.open()
        tools = ToolRegistry()
        register_builtin_tools(tools)
        bundled_skills_root = Path(__file__).resolve().parents[1] / "skills"
        skill_catalog_cache: dict[str, SkillCatalog] = {}
        memory_lock = NamedFileLock.memory(paths.project_id, paths.lock_dir)
        memory_store = MemoryStore(
            paths.workspace_root / ".memory",
            file_lock=memory_lock,
        )
        register_memory_tools(tools, memory_store)
        memory_coordinator = MemoryCoordinator(timeout=30.0, close_timeout=30.0)
        task_manager = TaskManager(
            TaskStore(paths.project_dir / "tasks"),
            file_lock=NamedFileLock.tasks(paths.project_id, paths.lock_dir),
        )
        todo_service = TodoService(store)
        register_todo_tools(tools, todo_service)
        register_task_tools(tools, task_manager)
        worktree_manager = WorktreeManager(
            paths.workspace_root,
            paths.workspace_root / ".worktrees",
        )
        message_bus = MessageBus(paths.project_dir / "mailboxes")
        prompt_state_provider: PromptStateProvider | None = None
        team_manager: TeamManager | None = None

        def skill_catalog_for(workspace_root: Path) -> SkillCatalog:
            """Handle the skill catalog for operation."""
            key = _skill_catalog_cache_key(workspace_root)
            catalog = skill_catalog_cache.get(key)
            if catalog is None:
                catalog = SkillCatalog.discover(
                    workspace_root,
                    paths.user_dir,
                    bundled_skills_root,
                )
                skill_catalog_cache[key] = catalog
            return catalog

        tools.register(LoadSkillTool(catalog_resolver=skill_catalog_for))
        duplicates = DuplicateGuard()
        hooks = HookManager()
        for discovered_hook in discover_command_hooks(settings):
            hooks.register(
                discovered_hook.point,
                discovered_hook.hook,
                name=discovered_hook.name,
            )
        if hook_registrar is not None:
            hook_registrar(hooks)
        background = BackgroundManager()
        executor = ToolExecutor(
            tools,
            hooks,
            duplicates,
            PermissionService(
                prompt=permission_prompt or _default_permission_prompt_for_sink(ui_sink)
            ),
            WorkspaceStateRegistry(),
            artifact_store_resolver=ProjectArtifactStores(paths.user_dir, redactor),
            workspace_lock_resolver=lambda context: NamedFileLock.workspace(
                context.workspace_id,
                paths.lock_dir,
            ),
            ui_sink=runtime_ui_sink,
            ui_factory_resolver=_tool_ui_factory,
        )

        async def run_background_tool(
            tool_name: str,
            arguments: dict[str, object],
            context: ToolContext,
        ) -> object:
            """Run the background tool."""
            nested = ToolCall(f"background-{uuid.uuid4().hex}", tool_name, arguments)
            return await executor.execute(nested, context)

        register_background_tools(tools, background, run_background_tool)
        mcp_manager = MCPConnectionManager(tools)
        try:
            await mcp_manager.connect_all(settings.mcp_servers)
        except BaseException:
            await mcp_manager.close_all()
            raise

        def provider_for(
            selected_provider: str,
            selected_model: str,
        ) -> ModelProvider:
            """Handle the provider for operation."""
            if selected_provider not in settings.providers:
                raise ValueError(f"Provider {selected_provider!r} is not configured")
            key = (selected_provider, selected_model)
            provider = providers.get(key)
            if provider is None:
                selected_settings = settings.providers[selected_provider].model_copy(
                    update={
                        "api_key": settings.resolve_api_key(selected_provider),
                        "model": selected_model,
                    }
                )
                provider = provider_registry.create(
                    selected_provider, selected_settings
                )
                providers[key] = provider
            return provider

        def loop_factory(
            selected_provider: str, selected_model: str, turn: RuntimeContext
        ) -> AgentLoop:
            """Handle the loop factory operation."""
            provider = provider_for(selected_provider, selected_model)
            memory_service = MemoryService(
                memory_store,
                provider,
                selected_model,
                turn.redactor,
            )
            context_options: dict[str, object] = {}
            if context_compaction is True:
                context_options.update(
                    {
                        "compaction_policy": CompactionPolicy(),
                        "context_budget_tokens": context_budget_tokens or 8_192,
                        "summarizer": ProviderContextSummarizer(
                            provider,
                            selected_model,
                            max_tokens=2_000,
                        ),
                    }
                )
            context_type = (
                _CompactionDisabledContextManager
                if context_compaction is False
                else ContextManager
            )
            return AgentLoop(
                store=store,
                provider=provider,
                context=context_type(
                    store,
                    model=selected_model,
                    skill_catalog_resolver=skill_catalog_for,
                    memory_service=memory_service if memory_recall else None,
                    prompt_state_provider=prompt_state_provider,
                    **context_options,
                ),
                tools=tools,
                executor=executor,
                duplicates=duplicates,
                memory_service=memory_service,
                memory_coordinator=memory_coordinator,
                memory_eligible=turn.memory_eligible,
                background=background,
                team_inbox=team_manager,
                recovery_policy=RecoveryPolicy(
                    RetryBudget(max_attempts=5, base_delay=0.5)
                ),
                budgets=runtime_budgets,
                hooks=hooks,
                ui_sink=runtime_ui_sink,
                trace_recorder=turn.trace_recorder,
                trace_id=turn.trace_id,
                root_session_id=turn.root_session_id,
                span_id=turn.span_id,
                parent_span_id=turn.parent_span_id,
                agent_id=turn.agent_id,
                parent_permission_broker=turn.parent_permission_broker,
                permission_mode=turn.permission_mode,
                permission_mode_resolver=turn.permission_mode_resolver,
                redactor=turn.redactor,
                secret_environment_names=turn.secret_environment_names,
                secret_values=turn.secret_values,
            )

        def compaction_manager_for(
            selected_provider: str,
            selected_model: str,
            budget: int,
        ) -> ContextManager:
            """Handle the compaction manager for operation."""
            provider = provider_for(selected_provider, selected_model)
            return ContextManager(
                store,
                model=selected_model,
                compaction_policy=CompactionPolicy(),
                context_budget_tokens=budget,
                summarizer=ProviderContextSummarizer(
                    provider,
                    selected_model,
                    max_tokens=8_000,
                ),
            )

        runtime = AgentRuntime(
            store=store,
            paths=paths,
            provider_name=provider_name,
            model=model,
            loop_factory=loop_factory,
            trace_redactor=redactor,
            secret_environment_names=secret_environment_names,
            secret_values=secret_values,
            startup_lock=NamedFileLock.startup(paths.project_id, paths.lock_dir),
            session_lock_factory=lambda root_id: NamedFileLock.session_tree(
                root_id, paths.lock_dir
            ),
            task_manager=task_manager,
            background_manager=background,
            closeables=(memory_coordinator, mcp_manager),
            manual_compactor=ManualCompactor(store, compaction_manager_for),
            provider_models={
                name: provider.model for name, provider in settings.providers.items()
            },
        )
        runtime_factory = DefaultAgentRuntimeFactory(
            runtime,
            worktrees=worktree_manager,
            task_manager=task_manager,
            parent_session_resolver=lambda: runtime.active_session_id,
            lease_resolver=lambda: runtime.active_root_turn_lease,
        )
        subagent_manager = SubagentManager(runtime_factory, hooks=hooks)
        team_manager = TeamManager(
            runtime_factory,
            message_bus=message_bus,
            task_manager=task_manager,
        )
        prompt_state_provider = PromptStateProvider(
            todo_service=todo_service,
            task_manager=task_manager,
            team_manager=team_manager,
        )

        def caller_for(context: ToolContext) -> AgentCaller:
            """Handle the caller for operation."""
            configured_agent_id = context.metadata.get("agent_id")
            agent_id = (
                configured_agent_id
                if isinstance(configured_agent_id, str) and configured_agent_id.strip()
                else context.agent_session_id
            )
            raw_task_ids = context.metadata.get("task_ids", ())
            task_ids = (
                frozenset(raw_task_ids)
                if isinstance(raw_task_ids, (list, tuple, set, frozenset))
                and all(
                    isinstance(task_id, str) and task_id.strip()
                    for task_id in raw_task_ids
                )
                else frozenset()
            )
            permission_mode = context.metadata.get("permission_mode", "ask")
            return AgentCaller(
                "lead" if agent_id == "lead" else "child",
                context.agent_session_id,
                ChildAuthority(
                    tools=frozenset(tool.spec.name for tool in tools.list()),
                    workspace_id=context.workspace_id,
                    permission_mode=(
                        permission_mode
                        if isinstance(permission_mode, str) and permission_mode.strip()
                        else "ask"
                    ),
                    task_ids=task_ids,
                    max_rounds=32,
                    max_tool_calls=128,
                    task_workspaces=frozenset({context.workspace_id}),
                ),
                runtime=runtime,
            )

        register_worktree_tools(tools, worktree_manager, task_manager=task_manager)
        register_agent_tools(
            tools,
            subagent_manager,
            caller_resolver=caller_for,
            worktrees=worktree_manager,
            task_manager=task_manager,
        )
        register_team_tools(
            tools,
            team_manager,
            message_bus,
            caller_resolver=caller_for,
            worktrees=worktree_manager,
            task_manager=task_manager,
        )
        if tool_allowlist is not None and "*" not in tool_allowlist:
            tools = _filtered_tool_registry(tools, tool_allowlist)
            executor = executor.fork(registry=tools, duplicates=duplicates)
        runtime.register_turn_end_resource(team_manager)
        runtime.closeables = (*runtime.closeables, team_manager)
        runtime.todo_service = todo_service
        runtime.subagent_manager = subagent_manager
        runtime.team_manager = team_manager
        runtime.message_bus = message_bus
        runtime.worktree_manager = worktree_manager
        try:
            await runtime.start()
            async with memory_lock.acquired_async():
                pass
        except BaseException:
            await runtime.close()
            raise
        return runtime

    except BaseException:
        await store.close()
        raise


def _filtered_tool_registry(
    registry: ToolRegistry,
    allowlist: frozenset[str],
) -> ToolRegistry:
    available = {tool.spec.name: tool for tool in registry.list()}
    missing = sorted(allowlist - available.keys())
    if missing:
        raise ValueError(f"Unknown allowed tools: {', '.join(missing)}")
    selected = ToolRegistry()
    selected.register_many(available[name] for name in sorted(allowlist))
    return selected


class _CompactionDisabledContextManager(ContextManager):
    """Manager coordinating the compaction disabled context manager."""
    def configure_runtime_compaction(self, summarizer: object) -> None:
        """Configure the runtime compaction."""
        del summarizer


def _default_permission_prompt_for_sink(
    ui_sink: RuntimeUISink | None,
) -> PermissionPromptCallback:
    console = _console_from_ui_sink(ui_sink)

    async def prompt(prompt: ToolPermissionPrompt) -> PromptChoice:
        if console is None:
            return _select_permission_choice(prompt)
        return _select_permission_choice(prompt, console=console)

    return prompt


def _console_from_ui_sink(ui_sink: RuntimeUISink | None) -> Console | None:
    renderer = getattr(ui_sink, "renderer", None)
    console = getattr(renderer, "console", None)
    return console if isinstance(console, Console) else None


def _skill_catalog_cache_key(workspace_root: Path) -> str:
    try:
        resolved = workspace_root.expanduser().resolve()
    except OSError:
        resolved = workspace_root.expanduser()
    return os.path.normcase(str(resolved))


def _tool_ui_factory(context: ToolContext) -> UIEventFactory:
    root_session_id = context.metadata.get("root_session_id")
    return UIEventFactory(
        session_id=context.agent_session_id,
        root_session_id=root_session_id if isinstance(root_session_id, str) else None,
    )


def _runtime_secrets(settings: Settings) -> tuple[tuple[str, ...], tuple[str, ...]]:
    environment_names: list[str] = []
    values: list[str] = []
    for provider_name, provider in settings.providers.items():
        environment_name = provider.api_key_env
        if environment_name is None and provider.type == "anthropic":
            environment_name = "ANTHROPIC_API_KEY"
        if environment_name:
            environment_names.append(environment_name)
        if provider.api_key is not None:
            configured = provider.api_key.get_secret_value()
            if configured:
                values.append(configured)
        try:
            value = settings.resolve_api_key(provider_name).get_secret_value()
        except (KeyError, ValueError):
            continue
        if value:
            values.append(value)
    return tuple(dict.fromkeys(environment_names)), tuple(dict.fromkeys(values))


async def _run_once(prompt: str) -> None:
    renderer = TerminalRenderer()
    sink = TerminalUISink(renderer)
    runtime = await build_runtime(ui_sink=sink)
    try:
        await runtime.run(prompt)
        renderer.flush()
    finally:
        await runtime.close()


async def _resume_once(session_id: str, prompt: str | None) -> None:
    if prompt is None:
        await _run_textual(session_id=session_id)
        return
    renderer = TerminalRenderer()
    sink = TerminalUISink(renderer)
    runtime = await build_runtime(ui_sink=sink)
    try:
        await runtime.resume(session_id, prompt)
        renderer.flush()
    finally:
        await runtime.close()


async def _interactive() -> None:
    await _run_textual()


async def _run_textual(*, session_id: str | None = None) -> None:
    """Run the textual."""
    sink = TextualUISink()
    permission_prompt = TextualPermissionPrompt()
    runtime = await build_runtime(
        ui_sink=sink,
        permission_prompt=permission_prompt,
    )
    try:
        textual_app = LiteCoderApp(
            runtime,
            sink=sink,
            permission_prompt=permission_prompt,
            session_id=session_id,
        )
        with TUIOutputGuard():
            completed_session_id = await textual_app.run_async()
        if completed_session_id:
            Console().print(f"session={completed_session_id}", style="yellow")
    finally:
        await runtime.close()


def run() -> None:
    """Run the requested operation."""
    app()
