mod app;
mod events;
mod markdown;
pub(crate) mod render;

use std::io;
use std::sync::Arc;

use anyhow::Result;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::{FutureExt, StreamExt};
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentEvent};
use crate::backend::ModelBackend;
use crate::config::Config;
use crate::llama_server::LlamaServer;
use crate::ollama::OllamaBackend;
use crate::session::Session;

use app::{AgentInput, App, ChatMessage};
use render::format_number;

/// Handle model list results from Ollama.
fn handle_model_list_result(app: &mut App, result: Result<Vec<String>>) {
    let was_at_bottom = app.is_at_bottom();
    let recent_hf = app.config.recent_hf_models.clone().unwrap_or_default();

    match result {
        Ok(models) if models.is_empty() => {
            let mut info = String::from("No Ollama models found.\n");
            if recent_hf.is_empty() {
                info.push_str("\nEnter a HuggingFace repo to use llama.cpp\n");
                info.push_str("(e.g. \"bartowski/Qwen2.5-Coder-7B-Instruct-GGUF\").\n");
            } else {
                info.push_str("\nRecent HuggingFace models:\n");
                info.push_str(&format_model_list(&recent_hf, &app.model, 0));
                info.push_str("\nType a number to select, or enter a new HuggingFace repo\n");
            }
            info.push_str("Esc to cancel.");
            app.messages.push(ChatMessage::Info(info));
            app.model_choices = Some(recent_hf);
        }
        Ok(models) => {
            let mut info = String::from("Available models (Ollama):\n");
            info.push_str(&format_model_list(&models, &app.model, 0));
            let ollama_count = models.len();
            if !recent_hf.is_empty() {
                info.push_str("\nRecent HuggingFace models:\n");
                info.push_str(&format_model_list(&recent_hf, &app.model, ollama_count));
            }
            info.push_str("\nType a number to select, or enter a HuggingFace repo for llama.cpp\n");
            info.push_str("(e.g. \"bartowski/Qwen2.5-Coder-7B-Instruct-GGUF\").\nEsc to cancel.");
            app.messages.push(ChatMessage::Info(info));
            let mut all_models = models;
            all_models.extend(recent_hf);
            app.model_choices = Some(all_models);
        }
        Err(e) => {
            let mut info = format!("Could not reach Ollama: {}\n", e);
            if !recent_hf.is_empty() {
                info.push_str("\nRecent HuggingFace models:\n");
                info.push_str(&format_model_list(&recent_hf, &app.model, 0));
                info.push_str("\nType a number to select, or enter a new HuggingFace repo\n");
            } else {
                info.push_str("\nEnter a HuggingFace repo to use llama.cpp\n");
                info.push_str("(e.g. \"bartowski/Qwen2.5-Coder-7B-Instruct-GGUF\").\n");
            }
            info.push_str("Esc to cancel.");
            app.messages.push(ChatMessage::Info(info));
            app.model_choices = Some(recent_hf);
        }
    }
    if was_at_bottom {
        app.scroll_offset = 0;
    }
}

/// Handle a newly started llama-server backend.
fn handle_backend_ready(
    app: &mut App,
    result: Result<(Arc<dyn ModelBackend>, String, LlamaServer)>,
    llama_server: &mut Option<LlamaServer>,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
) {
    let was_at_bottom = app.is_at_bottom();
    match result {
        Ok((backend, model_name, server)) => {
            // Stop old server if any
            if let Some(mut old) = llama_server.take() {
                tokio::spawn(async move { old.stop().await; });
            }
            *llama_server = Some(server);
            let _ = input_tx.send(AgentInput::SetBackend(backend));
            let _ = input_tx.send(AgentInput::SetModel(model_name.clone()));
            app.model = model_name.clone();
            app.context_used = 0;
            app.messages.push(ChatMessage::Info(format!(
                "Switched to {} (llama.cpp, context: {}).",
                model_name, format_number(app.context_size)
            )));

            // Save to config
            let mut config = app.config.clone();
            config.model = Some(model_name.clone());
            config.backend = Some("llama-cpp".to_string());
            config.add_recent_hf_model(&model_name);
            config.hf_repo = Some(model_name);
            if let Err(e) = config.save() {
                app.messages.push(ChatMessage::Error(format!(
                    "Warning: could not save config: {}", e
                )));
            }
            app.config = config;
        }
        Err(e) => {
            app.messages.push(ChatMessage::Error(format!(
                "Failed to start llama-server: {}", e
            )));
        }
    }
    app.is_processing = false;
    app.generation_start = None;
    if was_at_bottom {
        app.scroll_offset = 0;
    }
}

/// Format a numbered model list with `(current)` marker for the active model.
fn format_model_list(models: &[String], current: &str, start_index: usize) -> String {
    let mut s = String::new();
    for (i, name) in models.iter().enumerate() {
        let marker = if *name == current { " (current)" } else { "" };
        s.push_str(&format!("  {}. {}{}\n", start_index + i + 1, name, marker));
    }
    s
}

/// Result type sent over the channel when a new llama-server backend is ready.
pub(super) type BackendReady = Result<(Arc<dyn ModelBackend>, String, LlamaServer)>;

pub async fn run(agent: Agent, context_size: u64, mut session: Session, config: Config, mut llama_server: Option<LlamaServer>, no_confirm: bool) -> Result<()> {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(info);
    }));

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let model = agent.model().to_string();
    let ollama = OllamaBackend::new(None); // always points to real Ollama for model listing
    let mut app = App::new(model.clone(), context_size, ollama, config);
    if no_confirm {
        app.auto_approve = true;
    }

    session.log_debug(&format!("TUI_START model={}", model));
    let session_path = session.path().display().to_string();

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<AgentEvent>();
    let (input_tx, mut input_rx) = mpsc::unbounded_channel::<AgentInput>();
    let (confirm_tx, mut confirm_rx) = mpsc::unbounded_channel::<bool>();
    let (model_tx, mut model_rx) = mpsc::unbounded_channel::<Result<Vec<String>>>();
    let (backend_tx, mut backend_rx) = mpsc::unbounded_channel::<BackendReady>();

    // Keep a clone for the panic handler
    let panic_event_tx = event_tx.clone();
    let cancel_flag = app.cancel_flag.clone();

    tokio::spawn(async move {
        let result = std::panic::AssertUnwindSafe(async {
            let mut agent = agent;
            let event_tx = event_tx;
            let cancel_flag = cancel_flag;
            while let Some(input) = input_rx.recv().await {
                match input {
                    AgentInput::Message(msg) => {
                        cancel_flag.store(false, std::sync::atomic::Ordering::Relaxed);
                        if let Err(e) = agent.run(&msg, &event_tx, &mut confirm_rx, cancel_flag.clone()).await {
                            let _ = event_tx.send(AgentEvent::Error(format!("Agent error: {}", e)));
                            let _ = event_tx.send(AgentEvent::Done { prompt_tokens: 0 });
                        }
                    }
                    AgentInput::ClearHistory => {
                        agent.clear_history();
                    }
                    AgentInput::SetModel(model) => {
                        agent.set_model(model);
                    }
                    AgentInput::SetContextSize(size) => {
                        agent.set_context_size(size);
                    }
                    AgentInput::SetBackend(backend) => {
                        agent.set_backend(backend);
                    }
                }
            }
        })
        .catch_unwind()
        .await;

        if let Err(panic_info) = result {
            let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "unknown panic".to_string()
            };
            let _ = panic_event_tx.send(AgentEvent::Error(format!("Agent panicked: {}", msg)));
            let _ = panic_event_tx.send(AgentEvent::Done { prompt_tokens: 0 });
        }
    });

    let mut reader = crossterm::event::EventStream::new();
    let mut tick = tokio::time::interval(std::time::Duration::from_millis(80));
    let mut needs_redraw = true;

    loop {
        if app.needs_clear {
            terminal.clear()?;
            app.needs_clear = false;
        }
        if needs_redraw || app.is_processing {
            terminal.draw(|f| render::render(f, &mut app))?;
            needs_redraw = false;
        }

        tokio::select! {
            _ = tick.tick() => {
                // Only redraw on tick when processing (spinner/progress needs animation)
            }
            Some(Ok(evt)) = reader.next() => {
                events::handle_terminal_event(evt, &mut app, &input_tx, &confirm_tx, &model_tx, &session);
                if app.stop_llama_server {
                    app.stop_llama_server = false;
                    let old_server = llama_server.take();
                    if let Some(pending) = app.pending_server_start.take() {
                        // Stop old server, then start new one (single task guarantees ordering)
                        let tx = backend_tx.clone();
                        tokio::spawn(async move {
                            if let Some(mut old) = old_server {
                                old.stop().await;
                            }
                            if let Some((ollama, model)) = pending.unload {
                                let _ = ollama.unload_model(&model).await;
                            }
                            events::spawn_llama_server_inner(
                                pending.server_path,
                                pending.model_source,
                                pending.ctx,
                                pending.extra_args,
                                pending.model_name,
                                tx,
                            ).await;
                        });
                    } else if let Some(mut old) = old_server {
                        // Just stop (e.g. switching back to Ollama)
                        tokio::spawn(async move { old.stop().await; });
                    }
                }
                needs_redraw = true;
            }
            event = event_rx.recv() => {
                match event {
                    Some(e) => {
                        session.log_agent_event(&e);
                        // Handle message logging separately
                        if let AgentEvent::MessageLogged(ref msg) = e {
                            session.log_message(msg);
                        }
                        events::handle_agent_event(e, &mut app);
                        // Auto-approve tool calls when bypass is enabled
                        if app.auto_approve && app.pending_confirm.is_some() {
                            app.pending_confirm = None;
                            let _ = confirm_tx.send(true);
                        }
                    }
                    None => {
                        session.log_debug("AGENT_DISCONNECTED");
                        app.flush_streaming();
                        app.messages.push(ChatMessage::Error("Agent disconnected".into()));
                        app.is_processing = false;
                        app.generation_start = None;
                    }
                }
                needs_redraw = true;
            }
            Some(result) = model_rx.recv() => {
                handle_model_list_result(&mut app, result);
                needs_redraw = true;
            }
            Some(result) = backend_rx.recv() => {
                handle_backend_ready(&mut app, result, &mut llama_server, &input_tx);
                needs_redraw = true;
            }
        }

        if app.should_quit {
            break;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    // Stop any running llama-server
    if let Some(mut server) = llama_server {
        server.stop().await;
    }

    eprintln!("Session: {}", session_path);

    Ok(())
}
