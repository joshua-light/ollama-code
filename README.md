# Ollama Code

A terminal-based AI coding agent that runs locally via [Ollama](https://ollama.com) or [llama.cpp](https://github.com/ggerganov/llama.cpp). It connects to a local model, sends chat messages with tool definitions, and executes tool calls in an agentic loop until the model produces a final response.

## Features

- **Interactive TUI** with streaming token output, markdown rendering, and a spinner
- **Pipe mode** for non-interactive, scriptable use (`-p "prompt"`)
- **Tool use** — the agent can run bash commands, read files, edit files, and create new files
- **Dual backend** — works with Ollama or llama-server (llama.cpp), including HuggingFace model downloads
- **Runtime model switching** via the `/model` command (supports both Ollama and HuggingFace models)
- **Session logging** — every conversation is saved as JSONL for later review
- **Context window tracking** — shows prompt token usage in the TUI header

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

On first launch, if no model is configured, you'll be prompted to pick from your locally available models. The choice is saved to `~/.config/ollama-code/config.toml`.

## Usage

```
ollama-code [OPTIONS]

Options:
  -p, --prompt <PROMPT>                  Run in pipe mode with the given prompt
  -m, --model <MODEL>                    Model to use (overrides config)
      --backend <BACKEND>                "ollama" (default) or "llama-cpp"
      --llama-server-path <PATH>         Path to llama-server binary
      --model-path <PATH>                Path to GGUF model file
      --hf-repo <REPO>                   HuggingFace repo for model download
```

### TUI commands

| Command    | Description                  |
|------------|------------------------------|
| `/clear`   | Clear conversation history   |
| `/context` | Show context window usage    |
| `/model`   | Switch the active model      |
| `/session` | Show session log directory   |
| `/new`     | Start a new conversation     |

### Tools

The agent has access to four tools:

| Tool    | Description                                              |
|---------|----------------------------------------------------------|
| `bash`  | Execute shell commands                                   |
| `read`  | Read file contents (with optional offset/limit)          |
| `edit`  | Replace an exact string match in a file (returns a diff) |
| `write` | Create a new file                                        |

## Configuration

Settings are stored in `~/.config/ollama-code/config.toml`:

```toml
model = "qwen2.5-coder:7b"
context_size = 32768

# Optional: use llama-server instead of Ollama
# backend = "llama-cpp"
# llama_server_path = "/path/to/llama-server"
# model_path = "/path/to/model.gguf"
# hf_repo = "google/gemma-3-27b-it-GGUF"
# llama_server_args = ["-ngl", "99"]
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

## Session logs

Every conversation is logged to `~/.local/share/ollama-code/sessions/<id>/`:

- `messages.jsonl` — full message history (system, user, assistant, tool) with timestamps
- `debug.log` — agent events, timing, and diagnostics

## License

[MIT](LICENSE)
