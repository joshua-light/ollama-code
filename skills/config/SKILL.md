---
name: "config"
description: "View and modify ollama-code settings, features, and customization"
---

# Self-Configuration

The user is asking about ollama-code configuration, settings, features, or customization. Use your tools to read and modify the relevant files as needed.

## Config Files

### User config: `~/.config/ollama-code/config.toml`

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
- `/rewind` — Undo the last turn, or `/rewind N` for N turns
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
- **User skills**: `~/.config/ollama-code/skills/<name>/SKILL.md` (always available)

Project skills override user skills with the same name. Each `SKILL.md` has YAML frontmatter (`name`, `description`) and a body that becomes the prompt when invoked.

## How to Apply Changes

- **Config file changes**: Read the file, edit the relevant field, and tell the user. Changes take effect on next startup (or immediately for some fields like `context_size` via `/context`).
- **Runtime changes**: Use slash commands for immediate effect (model, context, bypass).
- **Project-specific overrides**: Create/edit `.ollama-code.toml` in the project root.
- **Custom skills**: Create a new directory under `~/.config/ollama-code/skills/<name>/` with a `SKILL.md` file containing frontmatter and instructions.
