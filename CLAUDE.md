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

The agent loop lives in `src/agent.rs`. `Agent::run()` sends messages to the model backend, streams the response, and if tool calls are present, executes them and loops back for the model's next response. Events are emitted via an `mpsc::UnboundedSender<AgentEvent>` channel — both the TUI and pipe mode consume from this channel. The agent holds a `Box<dyn ModelBackend>` and is backend-agnostic.

**Backend trait** (`src/backend.rs`): The `ModelBackend` trait defines the `chat()` method that all backends implement. Also contains the shared `ChatResponse` and `ModelInfo` types. Two implementations exist: `OllamaBackend` and `LlamaCppBackend`.

**Ollama backend** (`src/ollama.rs`): `OllamaBackend` implements `ModelBackend`. Streams responses from `POST /api/chat` line-by-line, auto-detecting Ollama (newline-delimited JSON) vs OpenAI SSE format. Collects content tokens and tool calls from the streamed chunks. Also provides Ollama-specific methods: `list_models()` and `unload_model()`.

**llama-cpp backend** (`src/llama_server.rs`): `LlamaCppBackend` implements `ModelBackend` by delegating to `OllamaBackend` pointed at the llama-server URL (both speak the same HTTP chat protocol). Also contains `LlamaServer`, which spawns and manages a `llama-server` child process — handling startup, health-check polling, and cleanup. Can resolve GGUF model files from Ollama's local blob storage.

**Tool system** (`src/tools.rs`): `Tool` trait with `definition()` and `execute()`. `ToolRegistry` holds registered tools and converts definitions to the JSON tool format.

**TUI** (`src/tui/`): Ratatui alternate-screen app with three panels (header, chat, input). Uses `tokio::select!` over terminal events, agent events, and an 80ms tick for the spinner. Includes a basic markdown renderer for assistant messages (code blocks, headings, lists, inline bold/code). The `/model` command supports runtime switching between Ollama models and HuggingFace models (via llama-server). The TUI owns the llama-server lifecycle when using HF models.

**Message types** (`src/message.rs`): Serializable message structs matching Ollama's chat API format (system/user/assistant/tool roles).

**Config** (`src/config.rs`): Loads from `~/.config/ollama-code/config.toml`. Stores model, context_size, and optional llama-cpp backend settings (backend, llama_server_path, model_path, llama_server_args).
