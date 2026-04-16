---
name: "self-modify"
description: "View and modify Ollama Code (you) settings, features, and customization"
trigger: "ollama.code|modify (this|the) (app|tool|agent|cli)|change (this|the) (app|tool|agent|cli)|add.*(mcp|hook|skill|tool plugin)|configure.*(mcp|hook|skill)|mcp server|modify (itself|yourself)|update (itself|yourself)|(edit|change|update|set).*(config|settings)"
---

# Self-Configuration

The user is asking about ollama-code configuration, settings, features, or customization. Use your tools to read and modify the relevant files as needed.

## Config Files

### User config: `{config_dir}/config.toml`

Global settings. Read it with `read` and modify with `edit`. All fields are optional:

```toml
model = "qwen2.5-coder:7b"        # Default model name (Ollama) or HuggingFace repo
context_size = 32768               # Context window size in tokens (default: 32768)
backend = "ollama"                 # "ollama" (default) or "llama-cpp"
ollama_url = "http://localhost:11434"  # Ollama API base URL

# llama-cpp backend options
llama_server_path = "/path/to/llama-server"   # Path to llama-server binary
llama_server_url = "http://host:8080"         # Remote llama-server URL (skips local spawn)
model_path = "/path/to/model.gguf"            # Local GGUF model file
hf_repo = "org/model-GGUF"                    # HuggingFace repo for model download
llama_server_args = ["-ngl", "99"]            # Extra llama-server arguments

# Behavior
bash_timeout = 120                 # Bash tool timeout in seconds (default: 120)
subagent_max_turns = 15            # Max turns for sub-agents (default: 15)
no_confirm = false                 # Skip tool confirmation prompts
bypass = false                     # Auto-approve all tool calls on startup
verbose = false                    # Enable debug output
show_cost_estimate = false         # Show estimated cost on status line
```

### Project config: `.ollama-code.toml`

Place in your project root (or any parent directory). Same format as user config. Project settings override user settings. Useful for per-project model or context size preferences.

## Slash Commands

These are available in the TUI:

- `/model` — Switch the active model (shows Ollama models + recent HuggingFace models)
- `/context` — Show context usage breakdown, or `/context <N>` to set context size
- `/bypass` — Toggle auto-approve for all tool calls
- `/clear` — Clear conversation history
- `/new` — Start a fresh conversation (same as /clear)
- `/rewind` — Pick a message to rewind to (or `/rewind N` for N turns)
- `/session` — Show the current session log directory
- `/resume` — List recent sessions (resume with `--resume <id>` on startup)

## CLI Flags

```
ollama-code [OPTIONS]
  -p, --prompt <PROMPT>        Run in pipe mode (non-interactive)
  -m, --model <MODEL>          Override model
  --backend <BACKEND>          "ollama" or "llama-cpp"
  --llama-server-path <PATH>   Path to llama-server binary
  --llama-server-url <URL>     Remote llama-server URL
  --model-path <PATH>          GGUF model file path
  --hf-repo <REPO>             HuggingFace model repo
  --context-size <N>           Context window size
  --no-confirm                 Skip tool confirmation prompts
  --verbose                    Enable debug output
  --resume [ID]                Resume a previous session
```

## Tools

The agent has these built-in tools:

- `bash(command)` — Run shell commands (timeout controlled by `bash_timeout`)
- `read(path, offset?, limit?)` — Read file contents
- `edit(path, old, new)` — Replace exact string in a file
- `write(path, content)` — Write/create a file
- `glob(pattern, path?)` — Find files by glob pattern
- `grep(pattern, path?, include?)` — Search file contents with regex
- `subagent(task)` — Spawn a sub-agent with fresh context for focused tasks

Tools that modify state (`bash`, `edit`, `write`, `subagent`) require user confirmation unless bypass mode is on.

## MCP Servers

MCP (Model Context Protocol) servers extend the agent with external tools. Configure in `{config_dir}/config.toml` or `.ollama-code.toml`.

### Stdio Transport (spawn a local process)

```toml
[mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
env = { "NODE_ENV" = "production" }
needs_confirm = true   # Require user confirmation for tool calls (default: true)
```

### Streamable HTTP Transport (connect to a remote server)

```toml
[mcp.remote-api]
url = "https://example.com/mcp"
headers = { "Authorization" = "Bearer tok_..." }
needs_confirm = false
```

### Configuration Fields

| Field | Transport | Description |
|-------|-----------|-------------|
| `command` | stdio | Command to spawn (mutually exclusive with `url`) |
| `args` | stdio | Arguments for the command |
| `env` | stdio | Environment variables for the process |
| `url` | HTTP | Server endpoint URL (mutually exclusive with `command`) |
| `headers` | HTTP | Extra HTTP headers (e.g., Authorization) |
| `needs_confirm` | both | Whether tool calls require user confirmation (default: true) |

### Runtime Behavior

- MCP servers connect at startup; tools are auto-discovered via the MCP protocol
- Tools are namespaced as `mcp__<server>__<tool>` (e.g., `mcp__filesystem__read_file`)
- `/mcp` slash command shows connected servers and their tools
- Disable a server: set `plugins.<server_name> = false` in config
- Disable a specific tool: set `plugins.mcp__<server>__<tool> = false` in config
- Stdio servers are managed as child processes; HTTP servers reconnect automatically

### Common MCP Servers

```toml
# Filesystem access
[mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allow"]

# GitHub
[mcp.github]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { "GITHUB_PERSONAL_ACCESS_TOKEN" = "<token>" }

# Brave Search
[mcp.brave-search]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-brave-search"]
env = { "BRAVE_API_KEY" = "<key>" }
```

## Hooks

Hooks are external scripts that run at specific lifecycle events in the agent loop. They can inspect, modify, or block tool calls and agent responses.

### Hook Files

- **User hooks**: `{config_dir}/hooks.toml`
- **Project hooks**: `.agents/hooks.toml` (searched walking up from cwd)

Project hooks override user hooks with the same name.

### Hook Entry Format

Each top-level key in `hooks.toml` is a named hook:

```toml
[block-dangerous-commands]
event = "pre_tool_execute"          # Required: lifecycle event
command = "./check.sh"              # Required: script to run (relative to hooks.toml dir)
tools = ["bash|shell_exec"]         # Optional: regex patterns for tool names (anchored)
if_args = "rm -rf"                  # Optional: regex searched against argument string values
timeout = 30                        # Optional: seconds (default: 30)
fail_closed = false                 # Optional: if true, hook errors = deny (default: false/fail-open)
priority = 50                       # Optional: execution order, lower = first (default: 50)
```

### Events

| Event | When | Hook Can |
|-------|------|----------|
| `pre_tool_execute` | Before a tool runs | Deny, modify arguments, or proceed |
| `post_tool_execute` | After a tool completes | Modify output/success flag |
| `agent_start` | Before the agent processes user input | Inject extra system context |
| `agent_done` | After the agent produces a final response | Rewrite the response |

### Tool Matching (`tools` field)

Each entry in the `tools` array is an anchored regex pattern (`^pattern$`):

- `"bash"` — exact match (backward compatible)
- `"file_.*"` — matches `file_read`, `file_write`, etc.
- `"bash|write"` — matches either `bash` or `write`
- `"[a-z]+_file"` — character class matching

If `tools` is omitted, the hook fires for all tools.

### Argument Filtering (`if_args` field)

A regex pattern searched (unanchored) against all string values in the tool arguments JSON. The hook only fires when at least one string value matches. Examples:

- `if_args = "^git push"` — only bash calls where a string arg starts with "git push"
- `if_args = "\\.rs$"` — only when an arg value ends with ".rs"
- `if_args = "secret|password"` — when any arg contains "secret" or "password"

If `if_args` is omitted, the hook fires regardless of arguments.

### Hook Input/Output Protocol

**Input**: Hooks receive JSON on stdin:
```json
{
  "hook": "pre_tool_execute",
  "data": { "tool_name": "bash", "arguments": {"command": "..."} },
  "config": { "key": "value" }
}
```

The `config` field contains per-hook config from `[hooks.<name>]` in `config.toml`.

**Output**: Hooks write JSON to stdout (empty = no-op):

- **pre_tool_execute**: `{"action": "proceed|modify|deny", "arguments": {...}, "message": "..."}`
- **post_tool_execute**: `{"action": "proceed|modify", "output": "...", "success": true}`
- **agent_start**: `{"system_context": "extra context to inject"}`
- **agent_done**: `{"action": "proceed|modify", "response": "rewritten response"}`

### Hook Management in config.toml

```toml
[hooks]
my-hook = false              # Disable a hook by name

[hooks.my-hook]
api_key = "secret123"        # Per-hook config (passed as JSON on stdin)
```

## System Prompt

The base system prompt is compiled from `SYSTEM_PROMPT.md` at the repo root. It defines the agent's persona, tool descriptions, and guidelines.

## Project Docs

The agent auto-loads these files (walking up from cwd, stops at first directory containing either):

- `CLAUDE.md` — Project instructions, coding conventions, architecture notes
- `AGENTS.md` — Multi-agent coordination instructions

These are appended to the system prompt and count toward context usage.

## Skills

Skills are slash-command extensions loaded from `SKILL.md` files:

- **Project skills**: `.agents/skills/<name>/SKILL.md` (searched walking up from cwd)
- **User skills**: `{config_dir}/skills/<name>/SKILL.md` (always available)

Project skills override user skills with the same name. Each `SKILL.md` has YAML frontmatter (`name`, `description`) and a body that becomes the prompt when invoked.

## How to Apply Changes

- **Config file changes**: Read the file, edit the relevant field, and tell the user. Changes take effect on next startup (or immediately for some fields like `context_size` via `/context`).
- **Runtime changes**: Use slash commands for immediate effect (model, context, bypass).
- **Project-specific overrides**: Create/edit `.ollama-code.toml` in the project root.
- **Custom skills**: Create a new directory under `{config_dir}/skills/<name>/` with a `SKILL.md` file containing frontmatter and instructions.
