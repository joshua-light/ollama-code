# Ollama Code

A terminal-based AI coding agent that runs locally via [Ollama](https://ollama.com) or [llama.cpp](https://github.com/ggerganov/llama.cpp). It connects to a local model, sends chat messages with tool definitions, and executes tool calls in an agentic loop until the model produces a final response.

## Features

- **Interactive TUI** with streaming token output, markdown rendering, syntax highlighting, and a spinner
- **Pipe mode** for non-interactive, scriptable use (`-p "prompt"`)
- **Tool use** — bash, read, edit, write, glob, grep, and sub-agent tools with confirmation prompts
- **Dual backend** — works with Ollama or llama-server (llama.cpp), including remote servers and HuggingFace model downloads
- **Runtime model switching** via the `/model` command (supports both Ollama and HuggingFace models)
- **MCP support** — connect external tools via the [Model Context Protocol](https://modelcontextprotocol.io)
- **Plugin system** — extend with custom tools via PLUGIN.toml manifests
- **Skills** — reusable prompt templates via SKILL.md files, activatable by user or model
- **Session management** — conversations saved as JSONL, with resume (`--resume`) and rewind (`/rewind`)
- **Project config** — per-project `.ollama-code.toml` layered over user config
- **Context window tracking** — segmented context bar with auto-trimming
- **Small-model tuning** — sampling params, dynamic tool scoping, tool validation, and task re-injection
- **Cancellation** — press Esc to cancel in-flight generation
- **Repetition detection** — detects and recovers from degenerate output loops

## Requirements

- [Rust](https://www.rust-lang.org/tools/install) (for building)
- [Ollama](https://ollama.com) running locally, **or** a [llama-server](https://github.com/ggerganov/llama.cpp) binary

A model with tool-calling support is recommended (e.g. `qwen2.5-coder`).

## Install

```bash
git clone https://github.com/joshua-light/ollama-code.git
cd ollama-code
cargo build --release
# Binary is at target/release/ollama-code
```

## Testing

```bash
cargo test               # run all tests (unit + integration)
cargo test --lib         # unit tests only
cargo test --test <name> # run a specific integration test
```

Integration tests use a mock backend and don't require Ollama to be running.

## Quick start

```bash
# Make sure Ollama is running
ollama serve

# Pull a model (if you haven't already)
ollama pull qwen2.5-coder:7b

# Launch the TUI
ollama-code

# Or run a one-shot prompt
ollama-code -p "list all TODO comments in this directory"
```

On first launch, a default `config.toml` is created automatically with all available options documented as comments. If no model is configured, you'll be prompted to pick from your locally available models. The config file location is `~/.config/ollama-code/config.toml` (or `$XDG_CONFIG_HOME/ollama-code/config.toml` if set).

## Usage

```
ollama-code [OPTIONS]

Options:
  -p, --prompt <PROMPT>                  Run in pipe mode with the given prompt
  -m, --model <MODEL>                    Model to use (overrides config)
      --backend <BACKEND>                "ollama" (default) or "llama-cpp"
      --llama-server-path <PATH>         Path to llama-server binary
      --llama-server-url <URL>           URL of a remote llama-server
      --model-path <PATH>                Path to GGUF model file
      --hf-repo <REPO>                   HuggingFace repo for model download
      --context-size <SIZE>              Context window size (overrides config)
      --no-confirm                       Auto-approve all tool calls
      --verbose                          Enable verbose/debug output
      --resume [<ID>]                    Resume a previous session
```

### TUI commands

| Command    | Description                                  |
|------------|----------------------------------------------|
| `/bypass`  | Toggle auto-approve for tool calls           |
| `/clear`   | Clear conversation history                   |
| `/context` | Show or set context window size              |
| `/help`    | Show available commands                      |
| `/mcp`     | Show connected MCP servers and their tools   |
| `/model`   | Switch the active model                      |
| `/new`     | Start a new conversation                     |
| `/resume`  | List recent sessions                         |
| `/rewind`  | Undo the last turn (or `/rewind N`)          |
| `/session` | Show session log directory                   |
| `/skills`  | List available skills                        |
| `/stats`   | Show session statistics                      |

### Tools

| Tool       | Description                                              |
|------------|----------------------------------------------------------|
| `bash`     | Execute shell commands                                   |
| `read`     | Read file contents (with optional offset/limit)          |
| `edit`     | Replace an exact string match in a file (returns a diff) |
| `write`    | Create a new file                                        |
| `glob`     | Find files by pattern                                    |
| `grep`     | Search file contents with regex                          |
| `subagent` | Spawn a sub-agent for a scoped task                      |
| `skill`    | Activate a skill by name (loads its instructions)        |

Bash, edit, and write require user confirmation by default (use `--no-confirm` or `/bypass` to skip).

## Configuration

Settings are stored in `~/.config/ollama-code/config.toml` (or `$XDG_CONFIG_HOME/ollama-code/config.toml` if set). You can also place a `.ollama-code.toml` in your project root — its values are layered over the user config.

```toml
model = "qwen2.5-coder:7b"
context_size = 32768

# General
# ollama_url = "http://localhost:11434"   # custom Ollama API URL
# no_confirm = false                      # auto-approve all tool calls
# verbose = false                         # enable debug output
# bypass = false                          # start with bypass mode on
# bash_timeout = 120                      # bash tool timeout in seconds
# subagent_max_turns = 15                 # max turns per sub-agent
# show_cost_estimate = false              # show estimated cost on status line

# Sampling
# temperature = 0.2                       # 0.0 = deterministic
# top_p = 0.9                             # nucleus sampling
# top_k = 40                              # top-k sampling

# Context management
# trim_threshold = 80                     # auto-trim at this % of context
# trim_target = 60                        # trim down to this %

# Small-model helpers
# tool_scoping = false                    # hide edit/write until a read/glob/grep is done
# task_reinjection = false                # periodically re-state the task objective
# reinjection_interval = 3               # re-inject every N agent turns

# llama-cpp backend
# backend = "llama-cpp"
# llama_server_path = "/path/to/llama-server"
# llama_server_url = "http://192.168.1.50:8080"  # remote server
# model_path = "/path/to/model.gguf"
# hf_repo = "google/gemma-3-27b-it-GGUF"
# llama_server_args = ["-ngl", "99"]

# MCP servers
# [mcp.filesystem]
# command = "npx"
# args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

# Plugins
# [plugins]
# my-tool = false          # disable a specific tool
# [plugins.my-plugin]      # plugin-specific config
# key = "value"
```

## llama.cpp backend

Instead of Ollama, you can use `llama-server` from llama.cpp directly. Ollama Code will spawn and manage the server process for you.

```bash
# With a local GGUF file
ollama-code --backend llama-cpp \
  --llama-server-path /path/to/llama-server \
  --model-path /path/to/model.gguf

# With a HuggingFace repo (downloaded automatically)
ollama-code --backend llama-cpp \
  --llama-server-path /path/to/llama-server \
  --hf-repo google/gemma-3-27b-it-GGUF

# Reuse a model already pulled by Ollama
ollama-code --backend llama-cpp \
  --llama-server-path /path/to/llama-server \
  -m qwen2.5-coder:7b
```

## MCP servers

Connect external tools via the [Model Context Protocol](https://modelcontextprotocol.io). Servers are configured in `config.toml` and their tools are discovered at startup:

```toml
[mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

Use `/mcp` in the TUI to see connected servers and their tools.

## Plugins

Extend the agent with custom tools via external plugins. Plugins are discovered from:
- `<CONFIG_DIR>/ollama-code/plugins/<name>/PLUGIN.toml` (user-global)
- `.agents/plugins/<name>/PLUGIN.toml` (project-local)

Each plugin exposes tools that are executed as subprocesses with JSON on stdin. See [PLUGIN.toml](https://github.com/joshua-light/ollama-code) for the manifest format.

## Skills

Skills are reusable prompt templates defined as SKILL.md files. They are discovered from:
- `<CONFIG_DIR>/ollama-code/skills/<name>/SKILL.md` (user-global)
- `.agents/skills/<name>/SKILL.md` (project-local)

> `<CONFIG_DIR>` is `~/.config` (or `$XDG_CONFIG_HOME` if set).

Skills can be activated in two ways:
- **By the user**: type `/<skill-name>` as a slash command in the TUI
- **By the model**: the agent can call the `skill` tool autonomously when a task matches an available skill

Use `/skills` in the TUI to list available skills.

## Session logs

Every conversation is logged to `~/.local/share/ollama-code/sessions/<id>/`:

- `messages.jsonl` — full message history (system, user, assistant, tool) with timestamps
- `debug.log` — agent events, timing, and diagnostics

## License

[MIT](LICENSE)
