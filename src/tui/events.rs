use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyModifiers};
use tokio::sync::mpsc;

use crate::agent::AgentEvent;
use crate::commands;
use crate::llama_server::{self, LlamaCppBackend, LlamaServer, ModelSource};
use crate::ollama::OllamaBackend;
use crate::session::Session;

use super::app::{AgentInput, App, ChatMessage, PendingConfirm, PendingServerStart, ToolResultData};
use super::render::{format_number, get_git_info_sync, pick_verb};

pub(super) fn handle_terminal_event(
    evt: Event,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    confirm_tx: &mpsc::UnboundedSender<bool>,
    model_tx: &mpsc::UnboundedSender<Result<Vec<String>>>,
    session: &Session,
) {
    if let Event::Key(key) = evt {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            if app.pending_confirm.is_some() {
                app.pending_confirm = None;
                let _ = confirm_tx.send(false);
                return;
            }
            if app.input.is_empty() {
                app.should_quit = true;
            } else {
                app.input.clear();
                app.cursor_pos = 0;
            }
            return;
        }

        // Tool confirmation mode — intercept y/n
        if app.pending_confirm.is_some() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                    app.pending_confirm = None;
                    let _ = confirm_tx.send(true);
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    app.pending_confirm = None;
                    let _ = confirm_tx.send(false);
                }
                _ => {} // ignore other keys
            }
            return;
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
            app.tools_expanded = !app.tools_expanded;
            app.needs_clear = true;
            return;
        }

        // Scroll keys work in all states
        let half_page = (app.max_scroll.max(10) / 2).max(5);
        match key.code {
            KeyCode::Up => { app.scroll_up(1); return; }
            KeyCode::Down => { app.scroll_down(1); return; }
            KeyCode::PageUp => { app.scroll_up(half_page); return; }
            KeyCode::PageDown => { app.scroll_down(half_page); return; }
            _ => {}
        }

        if app.is_processing {
            if key.code == KeyCode::Esc {
                app.cancel_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            return;
        }

        // Model selection mode
        if app.model_choices.is_some() {
            match key.code {
                KeyCode::Esc => {
                    app.dismiss_model_chooser();
                    app.messages.push(ChatMessage::Info("Model selection cancelled.".into()));
                }
                KeyCode::Enter => {
                    let raw = app.input.trim().to_string();
                    let parts: Vec<&str> = raw.split_whitespace().collect();
                    let choice = parts.first().and_then(|s| s.parse::<usize>().ok());
                    let ctx_override = parts.get(1).and_then(|s| s.parse::<u64>().ok());

                    // Resolve input to a model name: numbered selection or free text
                    let resolved = if let Some(num) = choice {
                        let models = app.model_choices.as_ref().unwrap();
                        if num >= 1 && num <= models.len() {
                            Some(models[num - 1].clone())
                        } else {
                            app.messages.push(ChatMessage::Info(format!(
                                "Invalid selection: {}", num
                            )));
                            app.dismiss_model_chooser();
                            None
                        }
                    } else if raw.contains('/') {
                        Some(raw.clone())
                    } else if !raw.is_empty() {
                        app.messages.push(ChatMessage::Info(
                            "Enter a number to select, or a HuggingFace repo (org/model).".into(),
                        ));
                        app.dismiss_model_chooser();
                        None
                    } else {
                        None
                    };

                    if let Some(model_name) = resolved {
                        if model_name.contains('/') {
                            // HuggingFace repo — defer start so event loop stops old server first
                            let hf_repo = model_name;
                            let server_path = app.config.llama_server_path.clone();
                            let extra_args = app.config.llama_server_args.clone().unwrap_or_default();
                            let ctx = ctx_override.unwrap_or(app.context_size);

                            if let Some(server_path) = server_path {
                                app.messages.push(ChatMessage::Info(format!(
                                    "Starting llama-server for {}...", hf_repo
                                )));
                                app.dismiss_model_chooser();
                                app.is_processing = true;
                                app.generation_start = Some(Instant::now());
                                app.generation_tokens = 0;
                                app.has_received_tokens = false;
                                app.generation_verb = "Starting server".to_string();

                                app.pending_server_start = Some(PendingServerStart {
                                    server_path,
                                    model_source: ModelSource::HuggingFace(hf_repo.clone()),
                                    ctx,
                                    extra_args,
                                    model_name: hf_repo,
                                    unload: Some((app.ollama.clone(), app.model.clone())),
                                });
                                app.stop_llama_server = true;
                            } else {
                                app.messages.push(ChatMessage::Error(
                                    "Set llama_server_path in ~/.config/ollama-code/config.toml to use HuggingFace models.".into(),
                                ));
                                app.dismiss_model_chooser();
                            }
                        } else {
                            // Ollama model selection
                            let ctx = ctx_override.unwrap_or(app.context_size);
                            if model_name == app.model && ctx == app.context_size {
                                app.messages.push(ChatMessage::Info(format!(
                                    "Already using {} (context: {}).", model_name, format_number(ctx)
                                )));
                            } else {
                                // Switch to Ollama backend (stop server in event loop)
                                app.stop_llama_server = true;
                                let backend = OllamaBackend::new(None);
                                let _ = input_tx.send(AgentInput::SetBackend(Arc::new(backend)));
                                app.model = model_name.clone();
                                let _ = input_tx.send(AgentInput::SetModel(model_name.clone()));
                                app.context_used = 0;
                                app.context_size = ctx;
                                let _ = input_tx.send(AgentInput::SetContextSize(ctx));
                                app.messages.push(ChatMessage::Info(format!(
                                    "Switched to {} (context: {}).", model_name, format_number(ctx)
                                )));

                                // Save to config
                                let mut config = app.config.clone();
                                config.model = Some(model_name);
                                config.context_size = Some(ctx);
                                config.backend = None; // back to Ollama
                                config.hf_repo = None;
                                if let Err(e) = config.save() {
                                    app.messages.push(ChatMessage::Error(format!(
                                        "Warning: could not save config: {}", e
                                    )));
                                }
                                app.config = config;
                            }
                            app.dismiss_model_chooser();
                        }
                    }
                }
                KeyCode::Char(c) => {
                    app.input.insert(app.cursor_pos, c);
                    app.cursor_pos += c.len_utf8();
                }
                KeyCode::Backspace => {
                    app.backspace();
                }
                _ => {}
            }
            return;
        }

        match key.code {
            KeyCode::Enter => {
                if key.modifiers.contains(KeyModifiers::SHIFT) {
                    app.input.insert(app.cursor_pos, '\n');
                    app.cursor_pos += 1;
                } else if let Some(cmd) = commands::parse(&app.input) {
                    let raw_input = app.input.clone();
                    handle_command(cmd, &raw_input, app, input_tx, model_tx, session);
                } else if let Some(msg) = app.submit() {
                    let _ = input_tx.send(AgentInput::Message(msg));
                }
            }
            KeyCode::Tab => {
                let matches = commands::completions(&app.input);
                if let Some(cmd) = matches.first() {
                    app.input = cmd.name.to_string();
                    app.cursor_pos = app.input.len();
                }
            }
            KeyCode::Char(c) => {
                app.input.insert(app.cursor_pos, c);
                app.cursor_pos += c.len_utf8();
            }
            KeyCode::Backspace => {
                app.backspace();
            }
            KeyCode::Left => {
                if app.cursor_pos > 0 {
                    app.cursor_pos = app.input[..app.cursor_pos]
                        .char_indices()
                        .next_back()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                }
            }
            KeyCode::Right => {
                if app.cursor_pos < app.input.len() {
                    app.cursor_pos = app.input[app.cursor_pos..]
                        .char_indices()
                        .nth(1)
                        .map(|(i, _)| app.cursor_pos + i)
                        .unwrap_or(app.input.len());
                }
            }
            KeyCode::Home => {
                app.cursor_pos = 0;
            }
            KeyCode::End => {
                app.cursor_pos = app.input.len();
            }
            KeyCode::Esc => {
                app.should_quit = true;
            }
            _ => {}
        }
    }
}

pub(super) fn handle_agent_event(event: AgentEvent, app: &mut App) {
    let was_at_bottom = app.is_at_bottom();

    match event {
        AgentEvent::Token(t) => {
            app.current_response.push_str(&t);
            app.generation_tokens += 1;
            app.has_received_tokens = true;
        }
        AgentEvent::ContentReplaced(content) => {
            // Tool calls were extracted from streamed text — replace the
            // raw JSON the user saw with the cleaned content.
            app.current_response = content;
        }
        AgentEvent::ToolCall { name, args } => {
            app.flush_streaming();
            app.tool_call_count += 1;
            app.has_received_tokens = false;
            app.generation_verb = pick_verb();
            app.messages
                .push(ChatMessage::ToolCall { name, args, result: None });
        }
        AgentEvent::ToolResult {
            name, output, success,
        } => {
            // Merge result into the last pending ToolCall
            let mut found = false;
            for msg in app.messages.iter_mut().rev() {
                if let ChatMessage::ToolCall { result, .. } = msg {
                    if result.is_none() {
                        *result = Some(ToolResultData { output, success });
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                app.messages.push(ChatMessage::Error(
                    format!("Orphaned tool result for '{}' (no pending tool call found)", name),
                ));
            }
        }
        AgentEvent::ContextUpdate { prompt_tokens } => {
            app.context_used = prompt_tokens;
        }
        AgentEvent::Done { prompt_tokens, .. } => {
            app.flush_streaming();
            if prompt_tokens > 0 {
                app.context_used = prompt_tokens;
            }
            if let Some(start) = app.generation_start.take() {
                let duration = start.elapsed();
                if duration.as_secs() >= 1 {
                    app.messages
                        .push(ChatMessage::GenerationSummary { duration });
                }
            }
            app.is_processing = false;
            // Refresh git status after agent finishes (bash tool may have changed things)
            let (branch, dirty) = get_git_info_sync();
            app.git_branch = branch;
            app.git_dirty = dirty;
        }
        AgentEvent::Error(e) => {
            app.flush_streaming();
            app.messages.push(ChatMessage::Error(e));
            app.is_processing = false;
        }
        AgentEvent::ToolConfirmRequest { name, args } => {
            app.pending_confirm = Some(PendingConfirm { name, args });
        }
        AgentEvent::ContextTrimmed {
            removed_messages,
            estimated_tokens_freed,
        } => {
            app.messages.push(ChatMessage::Info(format!(
                "Context trimmed: removed {} oldest messages (~{} tokens freed)",
                removed_messages, estimated_tokens_freed
            )));
        }
        AgentEvent::SubagentStart { .. } => {
            // The ToolCall event already shows "Subagent(task...)" in the UI.
        }
        AgentEvent::SubagentToolCall { name, args } => {
            app.messages.push(ChatMessage::SubagentToolCall {
                name,
                args,
                success: None,
            });
        }
        AgentEvent::SubagentToolResult { name, success } => {
            // Merge success into the last SubagentToolCall with matching name
            for msg in app.messages.iter_mut().rev() {
                if let ChatMessage::SubagentToolCall {
                    name: ref n,
                    success: ref mut s,
                    ..
                } = msg
                {
                    if *n == name && s.is_none() {
                        *s = Some(success);
                        break;
                    }
                }
            }
        }
        AgentEvent::SubagentEnd { .. } => {
            // The ToolResult event merges the final response into the ToolCall display.
        }
        AgentEvent::Cancelled => {
            app.flush_streaming();
            app.messages.push(ChatMessage::Info("Generation cancelled.".into()));
            app.is_processing = false;
            app.generation_start = None;
        }
        // MessageLogged and Debug are handled by the session logger in the event loop,
        // not by the app state.
        AgentEvent::MessageLogged(_) | AgentEvent::Debug(_) => {}
    }

    // Auto-scroll to bottom if user hadn't scrolled up
    if was_at_bottom {
        app.scroll_offset = 0;
    }
}

// ── Slash commands ────────────────────────────────────────────────────────

fn handle_command(
    cmd: commands::SlashCommand,
    raw_input: &str,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    model_tx: &mpsc::UnboundedSender<Result<Vec<String>>>,
    session: &Session,
) {
    let was_at_bottom = app.is_at_bottom();

    match cmd {
        commands::SlashCommand::Bypass => {
            app.auto_approve = !app.auto_approve;
            let status = if app.auto_approve { "on" } else { "off" };
            app.messages.push(ChatMessage::Info(format!(
                "Bypass permissions {}.", status
            )));
        }
        commands::SlashCommand::Clear => {
            app.reset_conversation(input_tx, "Conversation cleared.");
        }
        commands::SlashCommand::Context => {
            // `/context <size>` — set context size for current model
            let arg = raw_input.trim().strip_prefix("/context").unwrap_or("").trim();
            if let Ok(new_size) = arg.parse::<u64>() {
                let is_llama_cpp = app.config.backend.as_deref() == Some("llama-cpp");

                app.context_size = new_size;
                let _ = input_tx.send(AgentInput::SetContextSize(new_size));

                // Save to config
                let mut config = app.config.clone();
                config.context_size = Some(new_size);
                if let Err(e) = config.save() {
                    app.messages.push(ChatMessage::Error(format!(
                        "Warning: could not save config: {}", e
                    )));
                }
                app.config = config;

                if is_llama_cpp {
                    // Restart llama-server with the new context size
                    let model_source = if let Some(ref hf) = app.config.hf_repo {
                        Some(ModelSource::HuggingFace(hf.clone()))
                    } else if let Some(ref p) = app.config.model_path {
                        let path = std::path::PathBuf::from(p);
                        if path.exists() {
                            Some(ModelSource::File(path))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let (Some(server_path), Some(model_source)) =
                        (app.config.llama_server_path.clone(), model_source)
                    {
                        app.messages.push(ChatMessage::Info(format!(
                            "Restarting llama-server with context {}...",
                            format_number(new_size)
                        )));
                        app.is_processing = true;
                        app.generation_start = Some(Instant::now());
                        app.generation_tokens = 0;
                        app.has_received_tokens = false;
                        app.generation_verb = "Restarting server".to_string();

                        let extra_args = app.config.llama_server_args.clone().unwrap_or_default();
                        app.pending_server_start = Some(PendingServerStart {
                            server_path,
                            model_source,
                            ctx: new_size,
                            extra_args,
                            model_name: app.model.clone(),
                            unload: None,
                        });
                        app.stop_llama_server = true;
                    } else {
                        app.messages.push(ChatMessage::Error(
                            "Cannot restart llama-server: missing server path or model source.".into(),
                        ));
                    }
                } else {
                    app.messages.push(ChatMessage::Info(format!(
                        "Context size set to {} for {}.",
                        format_number(new_size),
                        app.model
                    )));
                }
            } else {
                // No argument — show context info
                let mut user_messages = 0u32;
                let mut assistant_messages = 0u32;
                let mut tool_calls = 0u32;
                let mut user_chars = 0usize;
                let mut assistant_chars = 0usize;
                let mut tool_chars = 0usize;

                for msg in &app.messages {
                    match msg {
                        ChatMessage::User(text) => {
                            user_messages += 1;
                            user_chars += text.len();
                        }
                        ChatMessage::Assistant(text) => {
                            assistant_messages += 1;
                            assistant_chars += text.len();
                        }
                        ChatMessage::ToolCall { result, .. } => {
                            tool_calls += 1;
                            if let Some(r) = result {
                                tool_chars += r.output.len();
                            }
                        }
                        _ => {}
                    }
                }

                app.messages.push(ChatMessage::ContextInfo {
                    context_used: app.context_used,
                    context_size: app.context_size,
                    user_messages,
                    assistant_messages,
                    tool_calls,
                    user_chars,
                    assistant_chars,
                    tool_chars,
                });
            }
        }
        commands::SlashCommand::Model => {
            app.messages.push(ChatMessage::Info("Fetching models...".into()));
            let ollama = app.ollama.clone();
            let tx = model_tx.clone();
            tokio::spawn(async move {
                let result = match ollama.list_models().await {
                    Ok(models) => Ok(models.into_iter().map(|m| m.name).collect()),
                    Err(e) => Err(e),
                };
                let _ = tx.send(result);
            });
        }
        commands::SlashCommand::Session => {
            app.messages.push(ChatMessage::Info(format!(
                "Session: {}",
                session.path().display()
            )));
        }
        commands::SlashCommand::New => {
            app.reset_conversation(input_tx, "New conversation started.");
        }
        commands::SlashCommand::Unknown(name) => {
            app.messages
                .push(ChatMessage::Info(format!("Unknown command: {}", name)));
        }
    }

    app.input.clear();
    app.cursor_pos = 0;

    if was_at_bottom {
        app.scroll_offset = 0;
    }
}

/// Shared logic for starting a llama-server and sending the result over a channel.
pub(super) async fn spawn_llama_server_inner(
    server_path: String,
    model_source: ModelSource,
    ctx: u64,
    extra_args: Vec<String>,
    model_name: String,
    tx: mpsc::UnboundedSender<super::BackendReady>,
) {
    // Try to reuse an existing llama-server for the same model.
    if let Some(server) = LlamaServer::connect_existing(&model_source).await {
        let backend = LlamaCppBackend::new(server.base_url());
        let _ = tx.send(Ok((Arc::new(backend), model_name, server)));
        return;
    }

    let server_binary = std::path::PathBuf::from(&server_path);
    if !server_binary.exists() {
        let _ = tx.send(Err(anyhow::anyhow!(
            "llama-server binary not found at: {}",
            server_binary.display()
        )));
        return;
    }
    let port = match llama_server::find_free_port() {
        Ok(p) => p,
        Err(e) => {
            let _ = tx.send(Err(e));
            return;
        }
    };
    match LlamaServer::start(&server_binary, &model_source, port, ctx, &extra_args).await {
        Ok(server) => {
            let backend = LlamaCppBackend::new(server.base_url());
            let _ = tx.send(Ok((Arc::new(backend), model_name, server)));
        }
        Err(e) => {
            let _ = tx.send(Err(e));
        }
    }
}
