mod agent_events;
mod commands;

use std::sync::Arc;

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind};
use tokio::sync::mpsc;

use crate::llama_server::{self, LlamaCppBackend, LlamaServer, ModelSource};
use crate::ollama::OllamaBackend;
use crate::session::Session;

use super::app::{AgentInput, App, ChatMessage, PendingServerStart, messages_to_chat_messages};
use super::picker::{PickerKind, PickerResult};
use super::render::format_number;
use super::settings::SettingsResult;
use super::tree_browser::TreeBrowserResult;

pub(super) use agent_events::handle_agent_event;

pub(super) fn handle_terminal_event(
    evt: Event,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    confirm_tx: &mpsc::UnboundedSender<bool>,
    model_tx: &mpsc::UnboundedSender<Result<Vec<String>>>,
    session: &mut Session,
) {
    if let Event::Paste(text) = evt {
        if app.pending_confirm.is_none() && app.picker.is_none() {
            app.input.insert_str(app.cursor_pos, &text);
            app.cursor_pos += text.len();
        }
        return;
    }

    if let Event::Mouse(mouse) = evt {
        let chat = app.chat_area;
        let in_chat = chat.height > 0
            && mouse.row >= chat.y
            && mouse.row < chat.y + chat.height
            && mouse.column >= chat.x
            && mouse.column < chat.x + chat.width;

        match mouse.kind {
            MouseEventKind::ScrollUp => {
                if app.picker.is_none() {
                    app.scroll_up(5);
                }
            }
            MouseEventKind::ScrollDown => {
                if app.picker.is_none() {
                    app.scroll_down(5);
                }
            }
            MouseEventKind::Down(MouseButton::Left) if in_chat => {
                let scroll_top = app.max_scroll.saturating_sub(app.scroll_offset) as usize;
                let content_line = scroll_top + (mouse.row - chat.y) as usize;
                let col = mouse.column.saturating_sub(chat.x);
                app.selection_line_cache.clear();
                app.selection = Some(super::app::TextSelection {
                    anchor: (content_line, col),
                    cursor: (content_line, col),
                });
            }
            MouseEventKind::Down(MouseButton::Left) => {
                // Click outside chat clears selection
                app.selection = None;
                app.selection_line_cache.clear();
            }
            MouseEventKind::Drag(MouseButton::Left) if app.selection.is_some() => {
                // Set continuous auto-scroll direction when near edges
                let edge = 2u16;
                if mouse.row <= chat.y.saturating_add(edge) {
                    app.auto_scroll = -1;
                    app.scroll_up(1);
                } else if mouse.row >= chat.y + chat.height.saturating_sub(edge) {
                    app.auto_scroll = 1;
                    app.scroll_down(1);
                } else {
                    app.auto_scroll = 0;
                }

                let scroll_top = app.max_scroll.saturating_sub(app.scroll_offset) as usize;
                let row_in_chat = mouse.row.clamp(chat.y, chat.y + chat.height.saturating_sub(1)) - chat.y;
                let content_line = scroll_top + row_in_chat as usize;
                let col = mouse.column.saturating_sub(chat.x).min(chat.width.saturating_sub(1));
                app.selection.as_mut().unwrap().cursor = (content_line, col);
            }
            MouseEventKind::Up(MouseButton::Left) if app.selection.is_some() => {
                app.auto_scroll = 0;
                let sel = app.selection.as_ref().unwrap();
                if sel.anchor == sel.cursor {
                    // Zero-width click — clear
                    app.selection = None;
                    app.selection_line_cache.clear();
                } else {
                    // Trigger clipboard copy on next render
                    app.copy_selection = true;
                }
            }
            _ => {}
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
            if app.settings.is_some() {
                app.dismiss_settings();
                return;
            }
            if app.tree_browser.is_some() {
                app.dismiss_tree_browser();
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

        // Settings panel — delegate all keys
        if app.settings.is_some() {
            handle_settings_key(key, app);
            return;
        }

        // Tree browser mode — delegate all keys to the tree browser
        if app.tree_browser.is_some() {
            handle_tree_browser_key(key, app, input_tx, session);
            return;
        }

        // Picker mode — delegate all keys to the picker (except Ctrl combos above)
        if app.picker.is_some() {
            handle_picker_key(key, app, input_tx, session);
            return;
        }

        // PageUp/PageDown always scroll
        let half_page = (app.max_scroll.max(10) / 2).max(5);
        match key.code {
            KeyCode::PageUp => { app.scroll_up(half_page); return; }
            KeyCode::PageDown => { app.scroll_down(half_page); return; }
            _ => {}
        }

        // Alt+Up/Down: always scroll
        if key.modifiers.contains(KeyModifiers::ALT) {
            match key.code {
                KeyCode::Up => { app.scroll_up(1); return; }
                KeyCode::Down => { app.scroll_down(1); return; }
                _ => {}
            }
        }

        // Esc: cancel prompt fill, cancel generation, or quit
        if key.code == KeyCode::Esc {
            if app.pending_prompt.is_some() {
                // Let handle_normal_input deal with it (prompt fill cancel)
            } else if app.is_processing {
                if app.server.loading.is_some() {
                    app.should_quit = true;
                } else {
                    app.server.cancel_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                return;
            } else {
                app.should_quit = true;
                return;
            }
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

fn handle_settings_key(key: crossterm::event::KeyEvent, app: &mut App) {
    let result = app.settings.as_mut().unwrap().handle_key(key);
    match result {
        SettingsResult::Active => {}
        SettingsResult::Dismissed => {
            app.dismiss_settings();
        }
        SettingsResult::Modified => {
            let config = app.settings.as_ref().unwrap().config().clone();
            if let Err(e) = config.save() {
                app.messages.push(ChatMessage::Error(format!(
                    "Warning: could not save config: {}", e
                )));
            }
            app.config = config;
        }
    }
}

fn handle_tree_browser_key(
    key: crossterm::event::KeyEvent,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    session: &mut Session,
) {
    let result = app.tree_browser.as_mut().unwrap().handle_key(key);
    match result {
        TreeBrowserResult::Cancelled => {
            app.dismiss_tree_browser();
        }
        TreeBrowserResult::Active => {}
        TreeBrowserResult::Selected(entry_id) => {
            app.dismiss_tree_browser();

            // Branch to the selected entry
            if let Err(e) = session.branch(&entry_id) {
                app.messages.push(ChatMessage::Error(format!("Branch failed: {}", e)));
                return;
            }

            let msgs = session.get_branch_path();
            let short_id = if entry_id.len() > 8 { &entry_id[..8] } else { &entry_id };
            restore_messages(app, input_tx, msgs, &format!("Switched to branch at {}.", short_id));
        }
    }
}

fn handle_picker_key(
    key: crossterm::event::KeyEvent,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    session: &mut Session,
) {
    let result = app.picker.as_mut().unwrap().handle_key(key);

    match result {
        PickerResult::Cancelled => {
            app.dismiss_picker();
        }
        PickerResult::Active => {}
        PickerResult::Selected(idx) => {
            let picker = app.picker.as_ref().unwrap();
            let kind = picker.kind;
            let label = picker.item_label(idx).to_string();
            let total = picker.items.len();
            app.dismiss_picker();

            match kind {
                PickerKind::Model => handle_model_picked(app, label, input_tx),
                PickerKind::Resume => handle_resume_picked(app, &label, input_tx, session),
                PickerKind::Rewind => {
                    // Picker items are oldest-first with the newest at the bottom,
                    // so the number of turns to rewind is (total - idx).
                    let turns = total.saturating_sub(idx);
                    commands::apply_rewind(
                        app,
                        input_tx,
                        session,
                        turns,
                        crate::agent::RewindMode::RewindTo,
                    );
                }
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

/// Reset conversation display, load messages, and send them to the agent.
fn restore_messages(
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    msgs: Vec<crate::message::Message>,
    info: &str,
) {
    app.reset_conversation(input_tx, info);
    let chat_messages = messages_to_chat_messages(&msgs);
    let info_msg = app.messages.pop();
    app.messages = chat_messages;
    if let Some(info) = info_msg {
        app.messages.push(info);
    }
    let _ = input_tx.send(AgentInput::RestoreMessages(msgs));
}

fn handle_resume_picked(
    app: &mut App,
    session_id: &str,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    session: &mut Session,
) {
    if session_id == session.id() {
        app.messages
            .push(ChatMessage::Info("Already in this session.".into()));
        return;
    }

    match Session::load_messages(session_id) {
        Ok(msgs) => {
            restore_messages(app, input_tx, msgs, &format!("Resumed session {}.", session_id));
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

/// Find the byte offset of the previous word boundary (for Ctrl+W, Alt+Left).
pub(in crate::tui) fn prev_word_boundary(input: &str, pos: usize) -> usize {
    let before = &input[..pos];
    // Skip trailing whitespace, then skip word chars
    let trimmed = before.trim_end();
    if trimmed.is_empty() {
        return 0;
    }
    // Find last whitespace before the word
    trimmed
        .rfind(|c: char| c.is_whitespace())
        .map(|i| {
            // i is byte offset of the whitespace char; step past it
            i + trimmed[i..].chars().next().unwrap().len_utf8()
        })
        .unwrap_or(0)
}

/// Find the byte offset of the next word boundary (for Alt+Right).
fn next_word_boundary(input: &str, pos: usize) -> usize {
    let after = &input[pos..];
    // Skip leading whitespace, then skip word chars
    let skipped_ws = after.trim_start().len();
    let ws_bytes = after.len() - skipped_ws;
    let rest = &after[ws_bytes..];
    // Find first whitespace in the remaining word
    rest.find(|c: char| c.is_whitespace())
        .map(|i| pos + ws_bytes + i)
        .unwrap_or(input.len())
}

fn handle_normal_input(
    key: crossterm::event::KeyEvent,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    model_tx: &mpsc::UnboundedSender<Result<Vec<String>>>,
    session: &mut Session,
) {
    // ── Prompt template variable fill mode ───────────────────────────
    if app.pending_prompt.is_some() {
        match key.code {
            KeyCode::Esc => {
                app.pending_prompt = None;
                app.input.clear();
                app.cursor_pos = 0;
                app.messages.push(ChatMessage::Info("Prompt cancelled.".into()));
                return;
            }
            KeyCode::Enter => {
                let fill = app.pending_prompt.as_mut().unwrap();
                let var = &fill.variables[fill.current];
                let value = if !app.input.trim().is_empty() {
                    app.input.trim().to_string()
                } else if let Some(ref default) = var.default {
                    default.clone()
                } else {
                    return;
                };
                let var_name = var.name.clone();
                fill.values.insert(var_name, value);
                fill.current += 1;
                app.input.clear();
                app.cursor_pos = 0;

                // Check if all variables are filled
                let fill = app.pending_prompt.as_ref().unwrap();
                if fill.current >= fill.variables.len() {
                    let fill = app.pending_prompt.take().unwrap();
                    let expanded = fill.template.expand(&fill.values);
                    let was_at_bottom = app.is_at_bottom();
                    app.messages.push(ChatMessage::SkillLoad {
                        name: format!("prompt:{}", fill.template.name),
                    });
                    app.begin_processing(super::render::pick_verb());
                    let _ = input_tx.send(AgentInput::Message(expanded));
                    if was_at_bottom {
                        app.scroll_offset = 0;
                    }
                }
                return;
            }
            // Allow normal text editing during fill mode — fall through
            _ => {}
        }
    }

    // Ctrl key combos
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            // Ctrl+A — beginning of line
            KeyCode::Char('a') => { app.cursor_pos = 0; return; }
            // Ctrl+E — end of line
            KeyCode::Char('e') => { app.cursor_pos = app.input.len(); return; }
            // Ctrl+W — delete word backward
            KeyCode::Char('w') => {
                app.delete_word_backward();
                return;
            }
            // Ctrl+K — kill to end of line
            KeyCode::Char('k') => {
                app.input.truncate(app.cursor_pos);
                return;
            }
            // Ctrl+U — kill to beginning of line
            KeyCode::Char('u') => {
                app.input.drain(..app.cursor_pos);
                app.cursor_pos = 0;
                return;
            }
            // Ctrl+D — delete char forward (or quit if empty)
            KeyCode::Char('d') => {
                if app.input.is_empty() {
                    app.should_quit = true;
                } else {
                    app.delete_forward();
                }
                return;
            }
            _ => {}
        }
    }

    // Alt key combos
    if key.modifiers.contains(KeyModifiers::ALT) {
        match key.code {
            // Alt+Enter — queue follow-up (sent after agent completes)
            KeyCode::Enter => {
                if app.is_processing {
                    app.submit_followup();
                } else {
                    // When idle, Alt+Enter behaves like Enter
                    if let Some(msg) = app.submit() {
                        let _ = input_tx.send(AgentInput::Message(msg));
                    }
                }
                return;
            }
            // Alt+Left — word backward
            KeyCode::Left => {
                app.cursor_pos = prev_word_boundary(&app.input, app.cursor_pos);
                return;
            }
            // Alt+Right — word forward
            KeyCode::Right => {
                app.cursor_pos = next_word_boundary(&app.input, app.cursor_pos);
                return;
            }
            // Alt+Backspace — delete word backward
            KeyCode::Backspace => {
                app.delete_word_backward();
                return;
            }
            _ => {}
        }
    }

    match key.code {
        KeyCode::Enter => {
            if key.modifiers.contains(KeyModifiers::SHIFT) {
                app.input.insert(app.cursor_pos, '\n');
                app.cursor_pos += 1;
            } else if let Some(cmd) = crate::commands::parse(&app.input) {
                // Commands are only executed when idle
                if !app.is_processing {
                    let raw_input = app.input.clone();
                    commands::handle_command(cmd, &raw_input, app, input_tx, model_tx, session);
                }
            } else if let Some(msg) = app.submit() {
                // submit() handles both idle (returns Some) and processing
                // (sends steering message via steer_tx, returns None)
                let _ = input_tx.send(AgentInput::Message(msg));
            }
        }
        KeyCode::Tab => {
            let matches = crate::commands::completions(&app.input);
            if let Some(cmd) = matches.first() {
                app.input = cmd.name.to_string();
                app.cursor_pos = app.input.len();
            } else {
                // Check skill and prompt name completions
                let prefix = app.input.trim();
                if let Some(skill_prefix) = prefix.strip_prefix('/') {
                    if let Some(skill) = app.skills.iter().find(|s| s.name.starts_with(skill_prefix)) {
                        app.input = format!("/{}", skill.name);
                        app.cursor_pos = app.input.len();
                    } else if let Some(prompt) = app.prompts.iter().find(|p| p.name.starts_with(skill_prefix)) {
                        app.input = format!("/{}", prompt.name);
                        app.cursor_pos = app.input.len();
                    }
                }
            }
        }
        // Up — input history (previous)
        KeyCode::Up => {
            if app.input_history.is_empty() {
                app.scroll_up(1);
                return;
            }
            match app.history_index {
                None => {
                    // Save current input as draft, jump to last history entry
                    app.history_draft = app.input.clone();
                    let idx = app.input_history.len() - 1;
                    app.history_index = Some(idx);
                    app.input = app.input_history[idx].clone();
                    app.cursor_pos = app.input.len();
                }
                Some(idx) if idx > 0 => {
                    let new_idx = idx - 1;
                    app.history_index = Some(new_idx);
                    app.input = app.input_history[new_idx].clone();
                    app.cursor_pos = app.input.len();
                }
                _ => {} // already at oldest entry
            }
        }
        // Down — input history (next)
        KeyCode::Down => {
            match app.history_index {
                Some(idx) if idx + 1 < app.input_history.len() => {
                    let new_idx = idx + 1;
                    app.history_index = Some(new_idx);
                    app.input = app.input_history[new_idx].clone();
                    app.cursor_pos = app.input.len();
                }
                Some(_) => {
                    // Past last entry — restore draft
                    app.history_index = None;
                    app.input = std::mem::take(&mut app.history_draft);
                    app.cursor_pos = app.input.len();
                }
                None => {
                    app.scroll_down(1);
                }
            }
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) && !key.modifiers.contains(KeyModifiers::ALT) => {
            app.input.insert(app.cursor_pos, c);
            app.cursor_pos += c.len_utf8();
        }
        KeyCode::Backspace => {
            app.backspace();
        }
        KeyCode::Delete => {
            app.delete_forward();
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
