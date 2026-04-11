# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is this

Imp is a CLI agent built on Ollama. It connects to a local Ollama instance, sends chat messages with tool definitions, and executes tool calls in a loop until the model produces a final response. It has two modes: an interactive TUI (default) and a non-interactive pipe mode (`-p "prompt"`).

## Build & Run

```bash
cargo build              # debug build
cargo build --release    # release build
cargo run                # run TUI mode (requires ollama running)
cargo run -- -p "prompt" # pipe mode
cargo run -- -m model    # override model
cargo clippy             # lint
cargo test               # no tests yet
```

Ollama must be running locally on port 11434 (`ollama serve`).

## Architecture

The agent loop lives in `src/agent.rs`. `Agent::run()` sends messages to Ollama, streams the response, and if tool calls are present, executes them and loops back for the model's next response. Events are emitted via an `mpsc::UnboundedSender<AgentEvent>` channel — both the TUI and pipe mode consume from this channel.

**Ollama client** (`src/ollama.rs`): Streams responses from `POST /api/chat` line-by-line (newline-delimited JSON). Collects content tokens and tool calls from the streamed chunks. The `on_token` callback enables live token streaming to the UI.

**Tool system** (`src/tools.rs`): `Tool` trait with `definition()` and `execute()`. `ToolRegistry` holds registered tools and converts definitions to Ollama's JSON tool format. Currently only `BashTool` is registered — it runs commands via `bash -c` and returns combined stdout/stderr.

**TUI** (`src/tui.rs`): Ratatui alternate-screen app with three panels (header, chat, input). Uses `tokio::select!` over terminal events, agent events, and an 80ms tick for the spinner. Includes a basic markdown renderer for assistant messages (code blocks, headings, lists, inline bold/code).

**Message types** (`src/message.rs`): Serializable message structs matching Ollama's chat API format (system/user/assistant/tool roles).

**Config** (`src/config.rs`): Loads from `~/.config/imp/config.toml`. Currently only has an optional `model` field.
