mod agent_events;
mod commands;

use std::sync::Arc;

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyModifiers};
use tokio::sync::mpsc;

use crate::llama_server::{self, LlamaCppBackend, LlamaServer, ModelSource};
use crate::ollama::OllamaBackend;
use crate::session::Session;

use super::app::{AgentInput, App, ChatMessage, PendingServerStart};
use super::render::format_number;

pub(super) use agent_events::handle_agent_event;

pub(super) fn handle_terminal_event(
    evt: Event,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    confirm_tx: &mpsc::UnboundedSender<bool>,
    model_tx: &mpsc::UnboundedSender<Result<Vec<String>>>,
    session: &Session,
) {
    if let Event::Paste(text) = evt {
        if !app.is_processing && app.pending_confirm.is_none() && app.model_choices.is_none() {
            app.input.insert_str(app.cursor_pos, &text);
            app.cursor_pos += text.len();
        }
        return;
    }

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
            handle_confirm_keys(key, app, confirm_tx);
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
                if app.server.loading.is_some() {
                    // Quit during initial server loading (Drop kills the child process)
                    app.should_quit = true;
                } else {
                    app.server.cancel_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
            return;
        }

        // Model selection mode
        if app.model_choices.is_some() {
            handle_model_selection(key, app, input_tx);
            return;
        }

        handle_normal_input(key, app, input_tx, model_tx, session);
    }
}

fn handle_confirm_keys(
    key: crossterm::event::KeyEvent,
    app: &mut App,
    confirm_tx: &mpsc::UnboundedSender<bool>,
) {
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
}

fn handle_model_selection(
    key: crossterm::event::KeyEvent,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
) {
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
                if model_name.contains('/') && app.config.llama_server_url.is_some() {
                    // Remote server — cannot spawn a local llama-server
                    app.messages.push(ChatMessage::Error(
                        "Cannot start a local llama-server while connected to a remote server. \
                         Change the model on the remote server directly.".into(),
                    ));
                    app.dismiss_model_chooser();
                } else if model_name.contains('/') {
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
                        app.begin_processing("Starting server".to_string());

                        app.server.pending_server_start = Some(PendingServerStart {
                            server_path,
                            model_source: ModelSource::HuggingFace(hf_repo.clone()),
                            ctx,
                            extra_args,
                            model_name: hf_repo,
                            unload: Some((app.ollama.clone(), app.model.clone())),
                        });
                        app.server.stop_llama_server = true;
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
                        app.server.stop_llama_server = true;
                        let backend = OllamaBackend::new(app.config.ollama_url.clone());
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
}

fn handle_normal_input(
    key: crossterm::event::KeyEvent,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    model_tx: &mpsc::UnboundedSender<Result<Vec<String>>>,
    session: &Session,
) {
    match key.code {
        KeyCode::Enter => {
            if key.modifiers.contains(KeyModifiers::SHIFT) {
                app.input.insert(app.cursor_pos, '\n');
                app.cursor_pos += 1;
            } else if let Some(cmd) = crate::commands::parse(&app.input) {
                let raw_input = app.input.clone();
                commands::handle_command(cmd, &raw_input, app, input_tx, model_tx, session);
            } else if let Some(msg) = app.submit() {
                let _ = input_tx.send(AgentInput::Message(msg));
            }
        }
        KeyCode::Tab => {
            let matches = crate::commands::completions(&app.input);
            if let Some(cmd) = matches.first() {
                app.input = cmd.name.to_string();
                app.cursor_pos = app.input.len();
            } else {
                // Check skill name completions
                let prefix = app.input.trim();
                if let Some(skill_prefix) = prefix.strip_prefix('/') {
                    if let Some(skill) = app.skills.iter().find(|s| s.name.starts_with(skill_prefix)) {
                        app.input = format!("/{}", skill.name);
                        app.cursor_pos = app.input.len();
                    }
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
