mod app;
mod events;
mod markdown;
pub(crate) mod picker;
pub(crate) mod render;
pub(crate) mod settings;
mod syntax;
pub(crate) mod tree_browser;

use std::io;
use std::sync::Arc;

use anyhow::Result;
use crossterm::{
    execute,
    event::{EnableBracketedPaste, DisableBracketedPaste, EnableMouseCapture, DisableMouseCapture},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::{FutureExt, StreamExt};
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentEvent};
use crate::backend::ModelBackend;
use crate::config::Config;
use crate::llama_server::{LlamaServer, ModelSource, ServerStartEvent};
use crate::ollama::OllamaBackend;
use crate::session::Session;

use app::{AgentInput, App, ChatMessage, ServerLoadingState, messages_to_chat_messages};
use render::format_number;
use crate::message::Message;

/// Configuration for deferred initial server startup (TUI shows progress bar).
pub struct InitialServerStart {
    pub server: LlamaServer,
    pub model_source: ModelSource,
}

/// Handle model list results from Ollama — open a picker with the available models.
fn handle_model_list_result(app: &mut App, result: Result<Vec<String>>) {
    use picker::{Picker, PickerItem, PickerKind};

    let recent_hf = app.config.recent_hf_models.clone().unwrap_or_default();
    let current = app.model.clone();

    let build_items = |models: Vec<String>| -> Vec<PickerItem> {
        models
            .into_iter()
            .map(|name| {
                let hint = if name == current {
                    "(current)".to_string()
                } else {
                    String::new()
                };
                PickerItem { label: name, hint }
            })
            .collect()
    };

    match result {
        Ok(models) if models.is_empty() && recent_hf.is_empty() => {
            app.messages.push(ChatMessage::Info(
                "No Ollama models found. Type a HuggingFace repo in the picker filter \
                 (e.g. \"bartowski/Qwen2.5-Coder-7B-Instruct-GGUF\")."
                    .into(),
            ));
            app.picker = Some(Picker::new("Select Model", Vec::new(), PickerKind::Model));
        }
        Ok(models) => {
            let mut all = models;
            all.extend(recent_hf);
            let items = build_items(all);
            app.picker = Some(Picker::new("Select Model", items, PickerKind::Model));
        }
        Err(e) => {
            if recent_hf.is_empty() {
                app.messages.push(ChatMessage::Error(format!(
                    "Could not reach Ollama: {}",
                    e
                )));
                return;
            }
            app.messages.push(ChatMessage::Info(format!(
                "Could not reach Ollama: {}. Showing recent HuggingFace models.",
                e
            )));
            let items = build_items(recent_hf);
            app.picker = Some(Picker::new("Select Model", items, PickerKind::Model));
        }
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
    app.finish_processing();
    if was_at_bottom {
        app.scroll_offset = 0;
    }
}

/// Result type sent over the channel when a new llama-server backend is ready.
pub(super) type BackendReady = Result<(Arc<dyn ModelBackend>, String, LlamaServer)>;

pub async fn run(agent: Agent, context_size: u64, mut session: Session, config: Config, mut llama_server: Option<LlamaServer>, restored: Option<Vec<Message>>, initial_server_start: Option<InitialServerStart>) -> Result<()> {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), DisableMouseCapture, DisableBracketedPaste, LeaveAlternateScreen);
        original_hook(info);
    }));

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableBracketedPaste, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let model = agent.model().to_string();
    let skills = agent.skills().to_vec();
    let system_prompt = agent.system_prompt().to_string();
    let ollama = OllamaBackend::with_sampling(
        config.ollama_url.clone(),
        crate::ollama::SamplingParams {
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
        },
    );
    let no_confirm = config.no_confirm.unwrap_or(false);
    let bypass = config.bypass.unwrap_or(false);

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<AgentEvent>();
    let (input_tx, mut input_rx) = mpsc::unbounded_channel::<AgentInput>();
    let (confirm_tx, mut confirm_rx) = mpsc::unbounded_channel::<bool>();
    let (steer_tx, mut steer_rx) = mpsc::unbounded_channel::<String>();
    let (model_tx, mut model_rx) = mpsc::unbounded_channel::<Result<Vec<String>>>();
    let (backend_tx, mut backend_rx) = mpsc::unbounded_channel::<BackendReady>();

    let mut app = App::new(model.clone(), context_size, ollama, config, skills, system_prompt, steer_tx);
    // Discover prompt templates
    if let Ok(cwd) = std::env::current_dir() {
        app.prompts = crate::prompts::discover_prompts(&cwd.to_string_lossy());
    }
    if no_confirm || bypass {
        app.auto_approve = true;
    }

    session.log_debug(&format!("TUI_START model={}", model));
    let session_path = session.path().display().to_string();

    // Keep a clone for the panic handler
    let panic_event_tx = event_tx.clone();
    let cancel_flag = app.server.cancel_flag.clone();

    tokio::spawn(async move {
        let result = std::panic::AssertUnwindSafe(async {
            let mut agent = agent;
            let event_tx = event_tx;
            let cancel_flag = cancel_flag;
            while let Some(input) = input_rx.recv().await {
                match input {
                    AgentInput::Message(msg) => {
                        cancel_flag.store(false, std::sync::atomic::Ordering::Relaxed);
                        if let Err(e) = agent.run(&msg, &event_tx, &mut confirm_rx, &mut steer_rx, cancel_flag.clone()).await {
                            let _ = event_tx.send(AgentEvent::Error(format!("Agent error: {}", e)));
                            let _ = event_tx.send(AgentEvent::Done { prompt_tokens: 0, eval_count: 0 });
                        }
                    }
                    AgentInput::ClearHistory => {
                        agent.clear_history();
                    }
                    AgentInput::Rewind(n) => {
                        agent.rewind_turns(n);
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
                    AgentInput::RestoreMessages(msgs) => {
                        agent.restore_messages(msgs);
                    }
                    AgentInput::Reload(config) => {
                        let (summary, system_prompt) = agent.reload(*config);
                        let _ = event_tx.send(AgentEvent::ReloadComplete { summary, system_prompt });
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
            let _ = panic_event_tx.send(AgentEvent::Done { prompt_tokens: 0, eval_count: 0 });
        }
    });

    // Restore a previous session if provided
    if let Some(msgs) = restored {
        let session_id = session.id().to_string();
        let chat_messages = messages_to_chat_messages(&msgs);
        app.messages = chat_messages;
        app.messages.push(ChatMessage::Info(format!(
            "Resumed session {}.", session_id
        )));
        let _ = input_tx.send(AgentInput::RestoreMessages(msgs));
    }

    // Deferred initial server startup — spawn progress-reporting task
    let (server_progress_tx, mut server_progress_rx) = mpsc::unbounded_channel::<ServerStartEvent>();
    if let Some(initial) = initial_server_start {
        let model_name = app.model.clone();
        app.server.loading = Some(ServerLoadingState {
            progress: 0.0,
            start: std::time::Instant::now(),
            model_name,
        });
        app.begin_processing("Loading model".to_string());

        let tx = server_progress_tx;
        tokio::spawn(async move {
            initial.server.wait_until_ready_with_progress(&initial.model_source, tx).await;
        });
    }

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
        // Copy selected text to system clipboard (staged during render)
        if let Some(text) = app.clipboard_text.take() {
            copy_to_clipboard(&text);
        }

        tokio::select! {
            _ = tick.tick() => {
                // Continuous auto-scroll while mouse is held at edge
                if app.auto_scroll != 0 {
                    if app.auto_scroll < 0 {
                        app.scroll_up(3);
                    } else {
                        app.scroll_down(3);
                    }
                    // Extend selection cursor to follow the scroll
                    if let Some(ref mut sel) = app.selection {
                        let chat = app.chat_area;
                        let scroll_top = app.max_scroll.saturating_sub(app.scroll_offset) as usize;
                        let row_in_chat = if app.auto_scroll < 0 { 0 } else { chat.height.saturating_sub(1) };
                        sel.cursor.0 = scroll_top + row_in_chat as usize;
                    }
                    needs_redraw = true;
                }
            }
            Some(Ok(evt)) = reader.next() => {
                events::handle_terminal_event(evt, &mut app, &input_tx, &confirm_tx, &model_tx, &mut session);
                if app.server.stop_llama_server {
                    app.server.stop_llama_server = false;
                    let old_server = llama_server.take();
                    if let Some(pending) = app.server.pending_server_start.take() {
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
                                pending.sampling,
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
                        // Persist trim watermark so resumed sessions skip trimmed messages
                        match &e {
                            AgentEvent::ContextTrimmed { removed_messages, .. }
                            | AgentEvent::ContextCompacted { removed_messages, .. } => {
                                session.record_trim(*removed_messages);
                            }
                            _ => {}
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
                        app.finish_processing();
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
            Some(evt) = server_progress_rx.recv() => {
                match evt {
                    ServerStartEvent::Progress(pct) => {
                        if let Some(ref mut loading) = app.server.loading {
                            loading.progress = pct;
                        }
                    }
                    ServerStartEvent::Ready(server) => {
                        llama_server = Some(server);
                        app.server.loading = None;
                        app.finish_processing();
                        let model = app.model.clone();
                        app.messages.push(ChatMessage::Info(format!(
                            "Model {} loaded.", model
                        )));
                    }
                    ServerStartEvent::Failed(err) => {
                        app.server.loading = None;
                        app.finish_processing();
                        app.messages.push(ChatMessage::Error(format!(
                            "Failed to start llama-server: {}", err
                        )));
                    }
                }
                needs_redraw = true;
            }
        }

        // If processing just finished and a message was queued, send it now
        if let Some(msg) = app.drain_queued() {
            let _ = input_tx.send(AgentInput::Message(msg));
        }

        // Async git status refresh (non-blocking)
        if app.refresh_git_pending {
            app.refresh_git_pending = false;
            let (branch, dirty) = render::get_git_info_async().await;
            app.git_branch = branch;
            app.git_dirty = dirty;
            needs_redraw = true;
        }

        if app.should_quit {
            break;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), DisableMouseCapture, DisableBracketedPaste, LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    // Stop any running llama-server
    if let Some(mut server) = llama_server {
        server.stop().await;
    }

    eprintln!("Session: {}", session_path);

    Ok(())
}

/// Copy text to the system clipboard via platform-native commands.
fn copy_to_clipboard(text: &str) {
    use std::io::Write;

    let candidates: &[(&str, &[&str])] = if std::env::var("WAYLAND_DISPLAY").is_ok() {
        &[
            ("wl-copy", &[]),
            ("xclip", &["-selection", "clipboard"]),
        ]
    } else {
        &[
            ("xclip", &["-selection", "clipboard"]),
            ("xsel", &["--clipboard", "--input"]),
        ]
    };

    for (cmd, args) in candidates {
        if let Ok(mut child) = std::process::Command::new(cmd)
            .args(*args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(text.as_bytes());
            }
            let _ = child.wait();
            return;
        }
    }
}
