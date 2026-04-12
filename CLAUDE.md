# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is this

Ollama Code is a CLI agent built on Ollama. It connects to a local Ollama instance, sends chat messages with tool definitions, and executes tool calls in a loop until the model produces a final response. It has two modes: an interactive TUI (default) and a non-interactive pipe mode (`-p "prompt"`). It also supports using `llama-server` (from llama.cpp) as an alternative backend.

## Build & Run

```bash
cargo build              # debug build
cargo build --release    # release build
cargo run                # run TUI mode (requires ollama running)
cargo run -- -p "prompt" # pipe mode
cargo run -- -m model    # override model
cargo clippy             # lint
cargo test               # no tests yet

# llama-cpp backend (uses llama-server instead of ollama)
cargo run -- --backend llama-cpp --llama-server-path /path/to/llama-server --model-path /path/to/model.gguf
cargo run -- --backend llama-cpp --llama-server-path /path/to/llama-server -m qwen2.5-coder:7b  # resolves model from Ollama's storage
```

Ollama must be running locally on port 11434 (`ollama serve`), unless using the `llama-cpp` backend.

## Architecture

The agent loop lives in `src/agent.rs`. `Agent::run()` sends messages to Ollama, streams the response, and if tool calls are present, executes them and loops back for the model's next response. Events are emitted via an `mpsc::UnboundedSender<AgentEvent>` channel — both the TUI and pipe mode consume from this channel.

**Ollama client** (`src/ollama.rs`): Streams responses from `POST /api/chat` line-by-line (newline-delimited JSON). Collects content tokens and tool calls from the streamed chunks. The `on_token` callback enables live token streaming to the UI.

**Tool system** (`src/tools.rs`): `Tool` trait with `definition()` and `execute()`. `ToolRegistry` holds registered tools and converts definitions to Ollama's JSON tool format. Currently only `BashTool` is registered — it runs commands via `bash -c` and returns combined stdout/stderr.

**TUI** (`src/tui.rs`): Ratatui alternate-screen app with three panels (header, chat, input). Uses `tokio::select!` over terminal events, agent events, and an 80ms tick for the spinner. Includes a basic markdown renderer for assistant messages (code blocks, headings, lists, inline bold/code). The `/model` command supports runtime switching between Ollama models and HuggingFace models (via llama-server). The TUI owns the llama-server lifecycle when using HF models.

**Message types** (`src/message.rs`): Serializable message structs matching Ollama's chat API format (system/user/assistant/tool roles).

**Config** (`src/config.rs`): Loads from `~/.config/ollama-code/config.toml`. Stores model, context_size, and optional llama-cpp backend settings (backend, llama_server_path, model_path, llama_server_args).

**llama-server manager** (`src/llama_server.rs`): Spawns and manages a `llama-server` child process for the llama-cpp backend. Handles startup, health-check polling, and cleanup. Can resolve GGUF model files from Ollama's local blob storage.
