mod app;
mod events;
mod markdown;
pub(crate) mod render;

use anyhow::Result;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::{FutureExt, StreamExt};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentEvent};
use crate::config::Config;
use crate::llama_server::LlamaServer;
use crate::ollama::OllamaClient;
use crate::session::Session;

use app::{AgentInput, App, ChatMessage};
use render::format_number;

pub async fn run(agent: Agent, context_size: u64, mut session: Session, config: Config, mut llama_server: Option<LlamaServer>) -> Result<()> {
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
    let ollama = OllamaClient::new(None); // always points to real Ollama for model listing
    let mut app = App::new(model.clone(), context_size, ollama, config);

    session.log_debug(&format!("TUI_START model={}", model));
    let session_path = session.path().display().to_string();

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<AgentEvent>();
    let (input_tx, mut input_rx) = mpsc::unbounded_channel::<AgentInput>();
    let (confirm_tx, mut confirm_rx) = mpsc::unbounded_channel::<bool>();
    let (model_tx, mut model_rx) = mpsc::unbounded_channel::<Result<Vec<String>>>();
    let (backend_tx, mut backend_rx) = mpsc::unbounded_channel::<Result<(OllamaClient, String, LlamaServer)>>();

    // Keep a clone for the panic handler
    let panic_event_tx = event_tx.clone();

    tokio::spawn(async move {
        let result = std::panic::AssertUnwindSafe(async {
            let mut agent = agent;
            let event_tx = event_tx;
            while let Some(input) = input_rx.recv().await {
                match input {
                    AgentInput::Message(msg) => {
                        if let Err(e) = agent.run(&msg, &event_tx, &mut confirm_rx).await {
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
                    AgentInput::SetClient(client) => {
                        agent.set_client(client);
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
                events::handle_terminal_event(evt, &mut app, &input_tx, &confirm_tx, &model_tx, &backend_tx, &session);
                if app.stop_llama_server {
                    app.stop_llama_server = false;
                    if let Some(mut old) = llama_server.take() {
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
                let was_at_bottom = app.is_at_bottom();
                match result {
                    Ok(models) if models.is_empty() => {
                        let mut info = String::from("No Ollama models found.\n");
                        info.push_str("\nEnter a HuggingFace repo to use llama.cpp\n");
                        info.push_str("(e.g. \"bartowski/Qwen2.5-Coder-7B-Instruct-GGUF\").\n");
                        info.push_str("Esc to cancel.");
                        app.messages.push(ChatMessage::Info(info));
                        app.model_choices = Some(Vec::new());
                    }
                    Ok(models) => {
                        let mut info = String::from("Available models (Ollama):\n");
                        for (i, name) in models.iter().enumerate() {
                            let marker = if *name == app.model { " (current)" } else { "" };
                            info.push_str(&format!("  {}. {}{}\n", i + 1, name, marker));
                        }
                        info.push_str("\nType a number to select, or enter a HuggingFace repo for llama.cpp\n");
                        info.push_str("(e.g. \"bartowski/Qwen2.5-Coder-7B-Instruct-GGUF\").\nEsc to cancel.");
                        app.messages.push(ChatMessage::Info(info));
                        app.model_choices = Some(models);
                    }
                    Err(e) => {
                        // Ollama not reachable — still allow HF model selection
                        let mut info = format!("Could not reach Ollama: {}\n", e);
                        info.push_str("\nEnter a HuggingFace repo to use llama.cpp\n");
                        info.push_str("(e.g. \"bartowski/Qwen2.5-Coder-7B-Instruct-GGUF\").\n");
                        info.push_str("Esc to cancel.");
                        app.messages.push(ChatMessage::Info(info));
                        app.model_choices = Some(Vec::new());
                    }
                }
                if was_at_bottom {
                    app.scroll_offset = 0;
                }
                needs_redraw = true;
            }
            Some(result) = backend_rx.recv() => {
                let was_at_bottom = app.is_at_bottom();
                match result {
                    Ok((client, model_name, server)) => {
                        // Stop old server if any
                        if let Some(mut old) = llama_server.take() {
                            tokio::spawn(async move { old.stop().await; });
                        }
                        llama_server = Some(server);
                        let _ = input_tx.send(AgentInput::SetClient(client));
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
