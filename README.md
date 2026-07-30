<h1 align="center">LiteCoder</h1>

<p align="center"><strong>A practical, cross-platform coding agent for terminal workflows.</strong></p>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">中文</a>
</p>

LiteCoder is a Python coding-agent CLI designed for repository-level software development. It combines a streaming terminal interface with built-in development tools, persistent sessions, context management, durable memory, multi-agent collaboration, and extensible integrations.

<p align="center">
  <img src="assets/litecoder-demo.gif" alt="LiteCoder real API workflow demo" width="960">
</p>

<p align="center"><sub>One continuous session recorded with a real API. Local paths and internal identifiers are sanitized.</sub></p>

## Highlights

- **Terminal-first workflow** — use the interactive interface, run a single prompt, or resume a previous session.
- **Flexible model providers** — connect through Anthropic Messages, OpenAI Chat Completions, or OpenAI Responses compatible APIs.
- **Controlled execution** — apply explicit permissions to workspace mutations and external side effects, with secret redaction throughout the runtime.
- **Context management** — persist conversations, track token budgets, and compact long sessions automatically or on demand.
- **Project memory** — store durable workspace knowledge as readable Markdown and load relevant memories into later turns.
- **Agent collaboration** — coordinate tasks, subagents, teams, background tools, message passing, and isolated Git worktrees.
- **Extensible runtime** — add MCP servers, lifecycle hooks, and project or user skills without changing the agent loop.

## Quick Start

LiteCoder requires Python 3.11 or later and an API key for a supported provider.

From the repository root, install LiteCoder with provider and MCP support:

```bash
python -m pip install ".[providers,mcp]"
```

Copy [`config.example.toml`](config.example.toml) to `~/.litecoder/config.toml`, then configure a provider and model:

```toml
default_provider = "custom"

[providers.custom]
type = "openai-chat-completions"
base_url = "https://your-provider.example/v1"
model = "your-model"
api_key = "your-api-key"
```

Set `type` to `anthropic-messages`, `openai-chat-completions`, or `openai-responses` to match the endpoint exposed by the service. Official and third-party compatible endpoints use the same configuration path. You can also store the API key after defining the provider:

```bash
litecoder config set-key custom
```

Start LiteCoder inside the repository you want to work on:

```bash
litecoder
```

## Usage

### CLI

```bash
# Run one prompt in a new session
litecoder run "inspect this repository and summarize its architecture"

# Resume an existing session
litecoder resume <session-id>

# Inspect persisted state
litecoder sessions list
litecoder sessions show <session-id>
litecoder tasks list
litecoder tasks show <task-id>

# Inspect traces and integrations
litecoder trace <session-id>
litecoder mcp list
litecoder mcp tools
litecoder mcp test
```

Use `litecoder --help` or `litecoder <command> --help` for the complete command syntax.

### Interactive Commands

- `/compact` compacts the active session context.
- `/context` shows effective context and token usage.
- `/memory [name]` inspects workspace memory.
- `/model [provider] [model]` shows or changes the selected model.
- `/tasks [task-id]` lists tasks or displays one task.
- `/trace` shows trace and command-audit locations.
- `/clear` starts a new persisted session context.
- `/help` and `/exit` show help or leave the interface.

## How It Works

**Permissions and traceability.** Safe reads can run automatically, while workspace mutations, external actions, and tools requiring confirmation are checked by the permission service. Root sessions produce JSONL traces, local commands are written to a durable audit log, and configured secrets are redacted from UI output, diagnostics, traces, and tool results.

**Context management and memory.** Sessions and messages are stored in `~/.litecoder/sessions.db`. LiteCoder manages context against token budgets and can compact older conversation state while preserving a working summary. Workspace memory is stored separately in `<workspace>/.memory/`, where it remains readable and editable as Markdown.

**Tasks and collaboration.** The runtime supports todos, persistent task state, explicitly delegated subagents and teams, team mailboxes, background tool execution, and Git worktree isolation. Project traces, task files, mailboxes, audits, and large tool outputs are stored under `~/.litecoder/projects/<project-id>/`.

<p align="center">
  <img src="assets/litecoder-permission.png" alt="LiteCoder permission confirmation" width="49%">
  <img src="assets/litecoder-context.png" alt="LiteCoder context status" width="49%">
</p>

## Extending LiteCoder

### MCP

Configure local stdio or remote Streamable HTTP servers in `~/.litecoder/config.toml`. Connected MCP tools are registered alongside LiteCoder's built-in tools and remain subject to the same execution pipeline.

### Hooks

LiteCoder supports user-defined external command hooks. Add one or more `[[hooks]]` entries to `~/.litecoder/config.toml`; each entry runs an executable at one lifecycle point.

```toml
[[hooks]]
name = "guard-writes"
enabled = true
point = "PreToolUse"
command = "python"
args = ["/absolute/path/to/guard_writes.py"]

[[hooks]]
name = "audit-tools"
enabled = true
point = "PostToolUse"
command = "python"
args = ["/absolute/path/to/audit_tools.py"]
```

For example, a minimal `guard_writes.py` hook can block a tool call based on its input:

```python
import json
import sys

request = json.load(sys.stdin)
call = request["payload"].get("call", {})
if call.get("name") == "shell":
    print(json.dumps({"blocked": True}))
else:
    print(json.dumps({}))
```

Pre-hooks run before an operation and may block it: `UserPromptSubmit`, `PreModelCall`, `PreToolUse`, and `SubagentStart`. Post-hooks run for observation after an operation: `PostModelCall`, `PostToolUse`, `ToolError`, `AgentStop`, and `SubagentStop`.

### Skills

Skills are discovered from `<workspace>/.litecoder/skills/` and `~/.litecoder/skills/`. LiteCoder exposes compact skill metadata to the model and loads full `SKILL.md` instructions only when requested.

### Project Instructions

Place repository-specific agent guidance in `<workspace>/LITECODER.md`. Keep it focused on project conventions, architecture, validation commands, and scoped delivery expectations. LiteCoder applies it after runtime constraints and the current user request: it cannot grant permissions, override tool policies, or turn repository content into higher-priority instructions.

See [`config.example.toml`](config.example.toml) for provider, MCP, and hook examples.

## Evaluation

LiteCoder includes an EvalPlus-based evaluation CLI covering agent execution, context management, tools and hooks, memory, task state, and multi-agent workflows.

```bash
python -m pip install -e ".[eval,providers,mcp,test]"
litecoder-eval run agent-benchmark --dataset humaneval --limit 15
litecoder-eval report <run.json>
```

Each run writes `run.json`, `report.md`, and one structured case directory with
the prompt, starter, solution, diff, runtime trace, local-test output, EvalPlus
validation details, and mode-specific evidence.

Run the consolidated cross-platform suite with:

```bash
python -m litecoder.eval.suite
```

Evaluation executes generated code. Use an isolated environment for untrusted or adversarial inputs; process deadlines on Windows are not a complete security sandbox.

## Development

Install the project in editable mode and run the deterministic checks:

```bash
python -m pip install -e ".[providers,mcp,test]"
python -m pytest -m "not real_model" -q
python -m litecoder --help
python -m build --no-isolation
```

CI runs on Windows, macOS, and Linux with Python 3.11 and 3.13.

## Project Structure

```text
src/litecoder/
├── agent/              # Agent runtime, loop, results, and stop handling
├── cli/                # CLI entry points and interactive commands
├── common/             # Shared locks, errors, and tracing
│   ├── errors/         # Error classification, recovery, and retry policy
│   └── trace/          # Trace context, events, recording, and redaction
├── context/            # Prompt assembly, token budgets, and compaction
│   └── session/        # Session models, migrations, and SQLite storage
├── eval/               # Evaluation orchestration, mode plugins, execution, validation, metrics, and reports
├── hooks/              # Built-in and external lifecycle hooks
├── memory/             # Memory loading, extraction, and consolidation
├── providers/          # Compatible model API adapter and registry
├── tasks/              # Tasks, agents, teams, messaging, and worktrees
├── tools/              # Tool registry, execution, permissions, MCP, and skills
│   └── builtin/        # Filesystem, search, shell, process, Git, agent, team, and worktree tools
├── ui/                 # Terminal UI events, presentation, and input
│   └── renderers/      # Terminal renderers
├── __init__.py         # Package marker
├── __main__.py         # `python -m litecoder` entry point
├── paths.py            # User, project, and workspace path resolution
└── settings.py         # Validated configuration models and key storage
```

## Community

Thanks to the [LINUX DO](https://linux.do/) community for its help and support.

## License

LiteCoder is released under the [MIT License](LICENSE).
