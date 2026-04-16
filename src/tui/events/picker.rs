//! Handlers for the picker overlay (model / resume / rewind) and the model
//! switching side-effects.

use std::sync::Arc;

use crossterm::event::KeyEvent;
use tokio::sync::mpsc;

use crate::llama_server::ModelSource;
use crate::ollama::OllamaBackend;
use crate::session::Session;
use crate::tui::app::{
    messages_to_chat_messages, AgentInput, App, ChatMessage, PendingServerStart,
};
use crate::tui::picker::{PickerKind, PickerResult};
use crate::tui::render::format_number;

use super::commands;

pub(super) fn handle_picker_key(
    key: KeyEvent,
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
            restore_messages(
                app,
                input_tx,
                msgs,
                &format!("Resumed session {}.", session_id),
            );
        }
        Err(e) => {
            app.messages
                .push(ChatMessage::Error(format!("Failed to load session: {}", e)));
        }
    }
}

/// Reset conversation display, load messages, and send them to the agent.
pub(super) fn restore_messages(
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

/// Start a HuggingFace model via llama-server.
fn start_hf_model(app: &mut App, hf_repo: String, ctx: u64) {
    let server_path = app.config.llama_server_path.clone();
    let extra_args = app.config.llama_server_args.clone().unwrap_or_default();

    if let Some(server_path) = server_path {
        app.messages.push(ChatMessage::Info(format!(
            "Starting llama-server for {}...",
            hf_repo
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
            "Already using {} (context: {}).",
            model_name,
            format_number(ctx)
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
        "Switched to {} (context: {}).",
        model_name,
        format_number(ctx)
    )));

    let mut config = app.config.clone();
    config.model = Some(model_name);
    config.context_size = Some(ctx);
    config.backend = None;
    config.hf_repo = None;
    if let Err(e) = config.save() {
        app.messages.push(ChatMessage::Error(format!(
            "Warning: could not save config: {}",
            e
        )));
    }
    app.config = config;
}
