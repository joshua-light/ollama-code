mod agent_events;
mod commands;

use std::sync::Arc;

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use tokio::sync::mpsc;

use crate::llama_server::{self, LlamaCppBackend, LlamaServer, ModelSource};
use crate::ollama::OllamaBackend;
use crate::session::Session;

use super::app::{AgentInput, App, ChatMessage, PendingServerStart, messages_to_chat_messages};
use super::picker::{PickerKind, PickerResult};
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
        if !app.is_processing && app.pending_confirm.is_none() && app.picker.is_none() {
            app.input.insert_str(app.cursor_pos, &text);
            app.cursor_pos += text.len();
        }
        return;
    }

    if let Event::Key(key) = evt {
        if key.kind != KeyEventKind::Press {
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            if app.pending_confirm.is_some() {
                app.pending_confirm = None;
                let _ = confirm_tx.send(false);
                return;
            }
            if app.picker.is_some() {
                app.dismiss_picker();
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

        // Picker mode — delegate all keys to the picker (except Ctrl combos above)
        if app.picker.is_some() {
            handle_picker_key(key, app, input_tx, session);
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

fn handle_picker_key(
    key: crossterm::event::KeyEvent,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    session: &Session,
) {
    let result = app.picker.as_mut().unwrap().handle_key(key);

    match result {
        PickerResult::Cancelled => {
            app.dismiss_picker();
        }
        PickerResult::Active => {}
        PickerResult::Selected(idx) => {
            let picker = app.picker.as_ref().unwrap();
            let label = picker.item_label(idx).to_string();
            let is_model = matches!(picker.kind, PickerKind::Model);
            app.dismiss_picker();

            if is_model {
                handle_model_picked(app, label, input_tx);
            } else {
                handle_resume_picked(app, &label, input_tx, session);
            }
        }
        PickerResult::FreeText(text) => {
            let is_model = matches!(app.picker.as_ref().unwrap().kind, PickerKind::Model);
            app.dismiss_picker();
            if is_model {
                handle_model_picked(app, text, input_tx);
            }
        }
    }
}

fn handle_model_picked(
    app: &mut App,
    model_name: String,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
) {
    let ctx = app.context_size;
    if model_name.contains('/') {
        if app.config.llama_server_url.is_some() {
            app.messages.push(ChatMessage::Error(
                "Cannot start a local llama-server while connected to a remote server. \
                 Change the model on the remote server directly."
                    .into(),
            ));
        } else {
            start_hf_model(app, model_name, ctx);
        }
    } else {
        switch_ollama_model(app, model_name, ctx, input_tx);
    }
}

fn handle_resume_picked(
    app: &mut App,
    session_id: &str,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    session: &Session,
) {
    if session_id == session.id() {
        app.messages
            .push(ChatMessage::Info("Already in this session.".into()));
        return;
    }

    match Session::load_messages(session_id) {
        Ok(msgs) => {
            app.reset_conversation(input_tx, &format!("Resumed session {}.", session_id));
            let chat_messages = messages_to_chat_messages(&msgs);
            // Insert loaded messages before the "Resumed" info message
            let info_msg = app.messages.pop();
            app.messages = chat_messages;
            if let Some(info) = info_msg {
                app.messages.push(info);
            }
            let _ = input_tx.send(AgentInput::RestoreMessages(msgs));
        }
        Err(e) => {
            app.messages.push(ChatMessage::Error(format!(
                "Failed to load session: {}",
                e
            )));
        }
    }
}

/// Start a HuggingFace model via llama-server.
fn start_hf_model(app: &mut App, hf_repo: String, ctx: u64) {
    let server_path = app.config.llama_server_path.clone();
    let extra_args = app.config.llama_server_args.clone().unwrap_or_default();

    if let Some(server_path) = server_path {
        app.messages.push(ChatMessage::Info(format!(
            "Starting llama-server for {}...", hf_repo
        )));
        app.begin_processing("Starting server".to_string());

        app.server.pending_server_start = Some(PendingServerStart {
            server_path,
            model_source: ModelSource::HuggingFace(hf_repo.clone()),
            ctx,
            extra_args,
            model_name: hf_repo,
            sampling: crate::ollama::SamplingParams {
                temperature: app.config.temperature,
                top_p: app.config.top_p,
                top_k: app.config.top_k,
            },
            unload: Some((app.ollama.clone(), app.model.clone())),
        });
        app.server.stop_llama_server = true;
    } else {
        app.messages.push(ChatMessage::Error(
            "Set llama_server_path in ~/.config/ollama-code/config.toml to use HuggingFace models.".into(),
        ));
    }
}

/// Switch to an Ollama model, update the agent backend, and persist to config.
fn switch_ollama_model(
    app: &mut App,
    model_name: String,
    ctx: u64,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
) {
    if model_name == app.model && ctx == app.context_size {
        app.messages.push(ChatMessage::Info(format!(
            "Already using {} (context: {}).", model_name, format_number(ctx)
        )));
        return;
    }

    app.server.stop_llama_server = true;
    let backend = OllamaBackend::with_sampling(
        app.config.ollama_url.clone(),
        crate::ollama::SamplingParams {
            temperature: app.config.temperature,
            top_p: app.config.top_p,
            top_k: app.config.top_k,
        },
    );
    let _ = input_tx.send(AgentInput::SetBackend(Arc::new(backend)));
    app.model = model_name.clone();
    let _ = input_tx.send(AgentInput::SetModel(model_name.clone()));
    app.context_used = 0;
    app.context_size = ctx;
    let _ = input_tx.send(AgentInput::SetContextSize(ctx));
    app.messages.push(ChatMessage::Info(format!(
        "Switched to {} (context: {}).", model_name, format_number(ctx)
    )));

    let mut config = app.config.clone();
    config.model = Some(model_name);
    config.context_size = Some(ctx);
    config.backend = None;
    config.hf_repo = None;
    if let Err(e) = config.save() {
        app.messages.push(ChatMessage::Error(format!(
            "Warning: could not save config: {}", e
        )));
    }
    app.config = config;
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
    sampling: crate::ollama::SamplingParams,
    tx: mpsc::UnboundedSender<super::BackendReady>,
) {
    // Try to reuse an existing llama-server for the same model.
    if let Some(server) = LlamaServer::connect_existing(&model_source).await {
        let backend = LlamaCppBackend::with_sampling(server.base_url(), sampling);
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
            let backend = LlamaCppBackend::with_sampling(server.base_url(), sampling);
            let _ = tx.send(Ok((Arc::new(backend), model_name, server)));
        }
        Err(e) => {
            let _ = tx.send(Err(e));
        }
    }
}
