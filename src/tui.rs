use anyhow::Result;
use crossterm::{
    event::{Event, EventStream, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::{FutureExt, StreamExt};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io;
use std::time::Instant;
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentEvent};
use crate::commands;
use crate::config::Config;
use crate::format;
use crate::llama_server::{self, LlamaServer, ModelSource};
use crate::ollama::OllamaClient;
use crate::session::Session;

enum AgentInput {
    Message(String),
    ClearHistory,
    SetModel(String),
    SetContextSize(u64),
    SetClient(OllamaClient),
}

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const MD_INDENT: &str = "   ";
const VERBS: &[&str] = &[
    "Tempering",
    "Harmonizing",
    "Composing",
    "Weaving",
    "Distilling",
    "Forging",
    "Conjuring",
    "Sculpting",
    "Brewing",
    "Crystallizing",
    "Synthesizing",
    "Kindling",
    "Rendering",
    "Assembling",
    "Alchemizing",
];

fn patch_leading_circle(lines: &mut [Line]) {
    for line in lines.iter_mut() {
        if !line.spans.is_empty() && line.spans[0].content == MD_INDENT {
            line.spans[0] = Span::styled(" ● ", Style::default().fg(Color::White));
            break;
        }
    }
}

fn sep_span() -> Span<'static> {
    Span::styled(" │ ", Style::default().fg(Color::DarkGray))
}

fn context_bar_spans(used: u64, total: u64, bar_width: usize) -> (u64, Vec<Span<'static>>) {
    let pct = ((used as f64 / total as f64) * 100.0).min(100.0) as u64;
    let filled = ((pct as usize * bar_width) / 100).max(if pct > 0 { 1 } else { 0 });
    let empty = bar_width - filled;
    let bar_color = if pct > 80 {
        Color::Red
    } else if pct >= 50 {
        Color::Yellow
    } else {
        Color::Green
    };
    (
        pct,
        vec![
            Span::styled("━".repeat(filled), Style::default().fg(bar_color)),
            Span::styled("╌".repeat(empty), Style::default().fg(Color::DarkGray)),
        ],
    )
}

fn render_expanded_output(lines: &mut Vec<Line<'static>>, output: &str) {
    for formatted in format::format_tool_output(output) {
        lines.push(Line::from(Span::styled(
            formatted,
            Style::default().fg(Color::DarkGray),
        )));
    }
}

fn spinner_frame() -> &'static str {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    SPINNER[(ms / 80) as usize % SPINNER.len()]
}

fn pick_verb() -> String {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    VERBS[(ms as usize) % VERBS.len()].to_string()
}

fn format_elapsed(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    if secs >= 3600 {
        format!("{}h{}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}s", secs)
    }
}

fn format_token_count(n: usize) -> String {
    if n == 0 {
        return String::new();
    }
    if n >= 1000 {
        format!(" · ↓ {:.1}k tokens", n as f64 / 1000.0)
    } else {
        format!(" · ↓ {} tokens", n)
    }
}

#[derive(Clone)]
struct ToolResultData {
    output: String,
    success: bool,
}

#[derive(Clone)]
enum ChatMessage {
    User(String),
    Assistant(String),
    ToolCall {
        name: String,
        args: String,
        result: Option<ToolResultData>,
    },
    Error(String),
    Info(String),
    ContextInfo {
        context_used: u64,
        context_size: u64,
        user_messages: u32,
        assistant_messages: u32,
        tool_calls: u32,
        user_chars: usize,
        assistant_chars: usize,
        tool_chars: usize,
    },
    GenerationSummary { duration: std::time::Duration },
}

struct App {
    messages: Vec<ChatMessage>,
    current_response: String,
    input: String,
    cursor_pos: usize,
    is_processing: bool,
    model: String,
    should_quit: bool,
    scroll_offset: u16,
    max_scroll: u16,
    // Generation tracking
    generation_start: Option<Instant>,
    generation_tokens: usize,
    generation_verb: String,
    has_received_tokens: bool,
    // Status line
    dir_name: String,
    context_size: u64,
    context_used: u64,
    session_start: Instant,
    tool_call_count: usize,
    tools_expanded: bool,
    git_branch: Option<String>,
    git_dirty: bool,
    // Model selection
    ollama: OllamaClient,
    model_choices: Option<Vec<String>>,
    config: Config,
    /// Set by model selection to signal the event loop to stop the llama-server
    stop_llama_server: bool,
}

impl App {
    fn new(model: String, context_size: u64, ollama: OllamaClient, config: Config) -> Self {
        let (git_branch, git_dirty) = get_git_info_sync();
        let dir_name = std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .unwrap_or_else(|| ".".to_string());
        Self {
            messages: Vec::new(),
            current_response: String::new(),
            input: String::new(),
            cursor_pos: 0,
            is_processing: false,
            model,
            should_quit: false,
            scroll_offset: 0,
            max_scroll: 0,
            generation_start: None,
            generation_tokens: 0,
            generation_verb: pick_verb(),
            has_received_tokens: false,
            dir_name,
            context_size,
            context_used: 0,
            session_start: Instant::now(),
            tool_call_count: 0,
            tools_expanded: false,
            git_branch,
            git_dirty,
            ollama,
            model_choices: None,
            config,
            stop_llama_server: false,
        }
    }

    fn submit(&mut self) -> Option<String> {
        if self.input.trim().is_empty() || self.is_processing {
            return None;
        }
        let msg = self.input.clone();
        self.input.clear();
        self.cursor_pos = 0;
        self.messages.push(ChatMessage::User(msg.clone()));
        self.is_processing = true;
        self.generation_start = Some(Instant::now());
        self.generation_tokens = 0;
        self.generation_verb = pick_verb();
        self.has_received_tokens = false;
        Some(msg)
    }

    fn flush_streaming(&mut self) {
        if !self.current_response.is_empty() {
            self.messages.push(ChatMessage::Assistant(std::mem::take(
                &mut self.current_response,
            )));
        }
    }

    fn scroll_up(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_add(lines).min(self.max_scroll);
    }

    fn scroll_down(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
    }

    fn is_at_bottom(&self) -> bool {
        self.scroll_offset == 0
    }

    fn backspace(&mut self) {
        if self.cursor_pos > 0 {
            let prev = self.input[..self.cursor_pos]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.input.remove(prev);
            self.cursor_pos = prev;
        }
    }

    fn dismiss_model_chooser(&mut self) {
        self.model_choices = None;
        self.input.clear();
        self.cursor_pos = 0;
    }

    fn reset_conversation(&mut self, input_tx: &mpsc::UnboundedSender<AgentInput>, msg: &str) {
        self.messages.clear();
        self.current_response.clear();
        self.context_used = 0;
        self.tool_call_count = 0;
        self.scroll_offset = 0;
        self.max_scroll = 0;
        let _ = input_tx.send(AgentInput::ClearHistory);
        self.messages.push(ChatMessage::Info(msg.into()));
    }
}

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
                        if let Err(e) = agent.run(&msg, &event_tx).await {
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

    let mut reader = EventStream::new();
    let mut tick = tokio::time::interval(std::time::Duration::from_millis(80));
    let mut needs_redraw = true;

    loop {
        if needs_redraw || app.is_processing {
            terminal.draw(|f| render(f, &mut app))?;
            needs_redraw = false;
        }

        tokio::select! {
            _ = tick.tick() => {
                // Only redraw on tick when processing (spinner/progress needs animation)
            }
            Some(Ok(evt)) = reader.next() => {
                handle_terminal_event(evt, &mut app, &input_tx, &model_tx, &backend_tx, &session);
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
                        handle_agent_event(e, &mut app);
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

fn handle_terminal_event(
    evt: Event,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    model_tx: &mpsc::UnboundedSender<Result<Vec<String>>>,
    backend_tx: &mpsc::UnboundedSender<Result<(OllamaClient, String, LlamaServer)>>,
    session: &Session,
) {
    if let Event::Key(key) = evt {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            if app.input.is_empty() {
                app.should_quit = true;
            } else {
                app.input.clear();
                app.cursor_pos = 0;
            }
            return;
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
            app.tools_expanded = !app.tools_expanded;
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
                app.should_quit = true;
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

                    if raw.contains('/') {
                        // HuggingFace repo — start llama-server
                        let hf_repo = raw.clone();
                        let server_path = app.config.llama_server_path.clone();
                        let extra_args = app.config.llama_server_args.clone().unwrap_or_default();
                        let ctx = app.context_size;

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

                            let tx = backend_tx.clone();
                            let current_model = app.model.clone();
                            let ollama_for_unload = app.ollama.clone();
                            tokio::spawn(async move {
                                // Unload current Ollama model to free VRAM
                                let _ = ollama_for_unload.unload_model(&current_model).await;
                                let server_binary = std::path::PathBuf::from(&server_path);
                                if !server_binary.exists() {
                                    let _ = tx.send(Err(anyhow::anyhow!(
                                        "llama-server binary not found at: {}", server_binary.display()
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
                                let model_source = ModelSource::HuggingFace(hf_repo.clone());
                                match LlamaServer::start(
                                    &server_binary,
                                    &model_source,
                                    port,
                                    ctx,
                                    &extra_args,
                                ).await {
                                    Ok(server) => {
                                        let client = OllamaClient::new(Some(server.base_url()));
                                        let _ = tx.send(Ok((client, hf_repo, server)));
                                    }
                                    Err(e) => {
                                        let _ = tx.send(Err(e));
                                    }
                                }
                            });
                        } else {
                            app.messages.push(ChatMessage::Error(
                                "Set llama_server_path in ~/.config/ollama-code/config.toml to use HuggingFace models.".into(),
                            ));
                            app.dismiss_model_chooser();
                        }
                    } else if let Some(choice) = choice {
                        // Ollama model selection
                        let ctx_override = parts.get(1).and_then(|s| s.parse::<u64>().ok());
                        let models = app.model_choices.take().unwrap();
                        if choice >= 1 && choice <= models.len() {
                            let new_model = models[choice - 1].clone();
                            let ctx = ctx_override.unwrap_or(app.context_size);
                            if new_model == app.model && ctx == app.context_size {
                                app.messages.push(ChatMessage::Info(format!(
                                    "Already using {} (context: {}).", new_model, format_number(ctx)
                                )));
                            } else {
                                // Switch to Ollama backend (stop server in event loop)
                                app.stop_llama_server = true;
                                let ollama_client = OllamaClient::new(None);
                                let _ = input_tx.send(AgentInput::SetClient(ollama_client));
                                app.model = new_model.clone();
                                let _ = input_tx.send(AgentInput::SetModel(new_model.clone()));
                                app.context_used = 0;
                                app.context_size = ctx;
                                let _ = input_tx.send(AgentInput::SetContextSize(ctx));
                                app.messages.push(ChatMessage::Info(format!(
                                    "Switched to {} (context: {}).", new_model, format_number(ctx)
                                )));

                                // Save to config
                                let mut config = app.config.clone();
                                config.model = Some(new_model);
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
                        } else {
                            app.messages.push(ChatMessage::Info(format!(
                                "Invalid selection: {}", choice
                            )));
                        }
                        app.dismiss_model_chooser();
                    } else {
                        app.messages.push(ChatMessage::Info(
                            "Enter a number for Ollama, or a HuggingFace repo (org/model).".into(),
                        ));
                        app.dismiss_model_chooser();
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
                if let Some(cmd) = commands::parse(&app.input) {
                    handle_command(cmd, app, input_tx, model_tx, session);
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

fn handle_agent_event(event: AgentEvent, app: &mut App) {
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
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    model_tx: &mpsc::UnboundedSender<Result<Vec<String>>>,
    session: &Session,
) {
    let was_at_bottom = app.is_at_bottom();

    match cmd {
        commands::SlashCommand::Clear => {
            app.reset_conversation(input_tx, "Conversation cleared.");
        }
        commands::SlashCommand::Context => {
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

fn format_number(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut result = String::with_capacity(len + len / 3);
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(b as char);
    }
    result
}

// ── Rendering ──────────────────────────────────────────────────────────────

fn compute_input_height(input: &str, term_width: u16, term_height: u16, is_processing: bool) -> u16 {
    if is_processing || input.is_empty() {
        return 3; // border + 1 line + border
    }
    let content_width = term_width.saturating_sub(3).max(1) as usize; // 3 for " ❯ "
    let char_count = input.chars().count();
    let lines = ((char_count + content_width - 1) / content_width).max(1) as u16;
    // Cap so chat area stays usable (leave room for header + status + at least 3 chat lines)
    let max_lines = term_height.saturating_sub(6).max(3);
    lines.min(max_lines) + 2 // + top border + bottom border
}

fn render(f: &mut Frame, app: &mut App) {
    let area = f.area();
    let input_height = compute_input_height(&app.input, area.width, area.height, app.is_processing);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),            // Header
            Constraint::Min(1),              // Chat
            Constraint::Length(input_height), // Input (dynamic)
            Constraint::Length(1),            // Status line
        ])
        .split(f.area());

    render_header(f, app, chunks[0]);
    render_chat(f, app, chunks[1]);
    render_input(f, app, chunks[2]);
    render_status_line(f, app, chunks[3]);
}

fn get_git_info_sync() -> (Option<String>, bool) {
    let branch = std::process::Command::new("git")
        .args(["branch", "--show-current"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());

    let dirty = std::process::Command::new("git")
        .args(["diff-index", "--quiet", "HEAD", "--"])
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(false);

    (branch, dirty)
}

fn render_status_line(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    // Show command completions when input starts with /
    let matches = commands::completions(&app.input);
    if !matches.is_empty() && !app.is_processing {
        render_command_completions(f, area, &app.input, &matches);
        return;
    }

    let mut spans: Vec<Span> = Vec::new();

    // Model
    spans.push(Span::styled(" \u{F171A} ", Style::default().fg(Color::Magenta)));
    spans.push(Span::styled(
        app.model.clone(),
        Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD),
    ));

    // Directory
    spans.push(sep_span());
    spans.push(Span::styled(
        app.dir_name.clone(),
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD),
    ));

    // Git branch
    if let Some(branch) = &app.git_branch {
        spans.push(sep_span());
        spans.push(Span::styled("⎇ ", Style::default().fg(Color::Magenta)));
        spans.push(Span::styled(
            branch.clone(),
            Style::default().fg(Color::Yellow),
        ));
        if app.git_dirty {
            spans.push(Span::styled("*", Style::default().fg(Color::Yellow)));
        }
    }

    // Context bar
    if app.context_size > 0 {
        // Show live estimate during generation: last prompt tokens + tokens generated so far
        let effective_context = if app.is_processing {
            app.context_used + app.generation_tokens as u64
        } else {
            app.context_used
        };
        let (pct, bar) = context_bar_spans(effective_context, app.context_size, 10);

        spans.push(sep_span());
        spans.push(Span::styled("ctx ", Style::default().fg(Color::DarkGray)));
        spans.extend(bar);
        spans.push(Span::styled(
            format!(" {}%", pct),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));
    }

    // Session duration
    let time_str = format_elapsed(app.session_start.elapsed());
    spans.push(sep_span());
    spans.push(Span::styled("⏱ ", Style::default().fg(Color::Blue)));
    spans.push(Span::styled(time_str, Style::default().fg(Color::Blue)));

    // Tool call count
    if app.tool_call_count > 0 {
        spans.push(sep_span());
        spans.push(Span::styled(
            format!("⚙ {}", app.tool_call_count),
            Style::default().fg(Color::Yellow),
        ));
    }

    let line = Line::from(spans);
    let paragraph = Paragraph::new(line);
    f.render_widget(paragraph, area);
}

fn render_command_completions(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    input: &str,
    matches: &[&commands::CommandInfo],
) {
    let input = input.trim();
    let mut spans: Vec<Span> = Vec::new();

    for (i, cmd) in matches.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("  │  ", Style::default().fg(Color::DarkGray)));
        }

        // Highlight the matching prefix of the command name
        let prefix_len = input.len().min(cmd.name.len());
        let matched = &cmd.name[..prefix_len];
        let rest = &cmd.name[prefix_len..];

        spans.push(Span::styled(" ", Style::default()));
        spans.push(Span::styled(
            matched.to_string(),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            rest.to_string(),
            Style::default().fg(Color::Yellow),
        ));
        spans.push(Span::styled(
            format!(" {}", cmd.description),
            Style::default().fg(Color::DarkGray),
        ));
    }

    // Tab hint on the right side
    let hint = " tab ⏎";
    let content_width: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    let available = area.width as usize;
    if content_width + hint.len() < available {
        let pad = available - content_width - hint.len();
        spans.push(Span::raw(" ".repeat(pad)));
        spans.push(Span::styled(
            hint,
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        ));
    }

    let line = Line::from(spans);
    let paragraph = Paragraph::new(line);
    f.render_widget(paragraph, area);
}

fn render_header(f: &mut Frame, _app: &App, area: ratatui::layout::Rect) {
    let header = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(
                " \u{F06A9} ",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "Ollama Code",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" · ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("v{}", env!("CARGO_PKG_VERSION")),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(Span::styled(
            " ─".to_string() + &"─".repeat(area.width.saturating_sub(2) as usize),
            Style::default().fg(Color::DarkGray),
        )),
    ]);
    f.render_widget(header, area);
}

fn render_chat(f: &mut Frame, app: &mut App, area: ratatui::layout::Rect) {
    let lines = build_chat_lines(app, area.width);
    let text = Text::from(lines);
    let paragraph = Paragraph::new(text).wrap(Wrap { trim: false });

    let line_count = paragraph.line_count(area.width) as u16;
    let max_scroll = line_count.saturating_sub(area.height);
    app.max_scroll = max_scroll;
    app.scroll_offset = app.scroll_offset.min(max_scroll);

    let scroll = max_scroll.saturating_sub(app.scroll_offset);

    let paragraph = paragraph.scroll((scroll, 0));
    f.render_widget(paragraph, area);
}

fn render_input(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let input_block = Block::default()
        .borders(Borders::TOP | Borders::BOTTOM)
        .border_style(Style::default().fg(Color::DarkGray));

    if app.is_processing {
        let input_line = Line::from(Span::styled(
            " …",
            Style::default().fg(Color::DarkGray),
        ));
        let input = Paragraph::new(input_line).block(input_block);
        f.render_widget(input, area);
        return;
    }

    let prompt = if app.model_choices.is_some() { " # " } else { " ❯ " };

    let content_width = area.width.saturating_sub(3).max(1) as usize;
    let chars: Vec<char> = app.input.chars().collect();
    let mut lines: Vec<Line> = Vec::new();

    if chars.is_empty() {
        lines.push(Line::from(vec![
            Span::styled(
                prompt,
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(""),
        ]));
    } else {
        let mut i = 0;
        while i < chars.len() {
            let end = (i + content_width).min(chars.len());
            let chunk: String = chars[i..end].iter().collect();
            if i == 0 {
                lines.push(Line::from(vec![
                    Span::styled(
                        prompt,
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(chunk),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::raw("   "),
                    Span::raw(chunk),
                ]));
            }
            i = end;
        }
    }

    let input = Paragraph::new(lines).block(input_block);
    f.render_widget(input, area);

    let cursor_chars = app.input[..app.cursor_pos].chars().count();
    let cursor_line = cursor_chars / content_width;
    let cursor_col = cursor_chars % content_width;
    f.set_cursor_position((area.x + 3 + cursor_col as u16, area.y + 1 + cursor_line as u16));
}

// ── Chat content builder ───────────────────────────────────────────────────

fn build_chat_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = Vec::new();

    if app.messages.is_empty() && app.current_response.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Type a message to get started.",
            Style::default().fg(Color::DarkGray),
        )));
        return lines;
    }

    for msg in &app.messages {
        match msg {
            ChatMessage::User(text) => {
                let user_bg = Style::default().bg(Color::Indexed(236));
                lines.push(Line::styled("", user_bg));
                lines.push(
                    Line::from(vec![
                        Span::styled(
                            "  ❯ ",
                            Style::default()
                                .fg(Color::White)
                                .bg(Color::Indexed(236))
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            text.clone(),
                            Style::default()
                                .fg(Color::White)
                                .bg(Color::Indexed(236))
                                .add_modifier(Modifier::BOLD),
                        ),
                    ])
                    .style(user_bg),
                );
            }
            ChatMessage::Assistant(text) => {
                lines.push(Line::from(""));
                let mut md_lines = render_markdown(text.trim_end(), width);
                patch_leading_circle(&mut md_lines);
                lines.extend(md_lines);
            }
            ChatMessage::ToolCall { name, args, result } => {
                lines.push(Line::from(""));
                let circle_color = match result {
                    Some(ToolResultData { success: true, .. }) => Color::Green,
                    Some(ToolResultData { success: false, .. }) => Color::Red,
                    None => Color::White,
                };
                lines.push(Line::from(vec![
                    Span::styled(" ● ", Style::default().fg(circle_color)),
                    Span::styled(
                        format!("{}({})", format::capitalize_first(name), format::truncate_args(args, 77)),
                        Style::default().fg(Color::White),
                    ),
                ]));
                if let Some(result_data) = result {
                    match name.as_str() {
                        "read" => {
                            render_read_result(&mut lines, result_data, app.tools_expanded);
                        }
                        "edit" => {
                            render_edit_result(&mut lines, result_data);
                        }
                        _ => {
                            render_default_result(&mut lines, result_data, app.tools_expanded);
                        }
                    }
                } else {
                    // Tool is still running
                    lines.push(Line::from(vec![
                        Span::styled(format::PREFIX_FIRST, Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            format!("{} Running...", spinner_frame()),
                            Style::default().fg(Color::Yellow),
                        ),
                    ]));
                }
            }
            ChatMessage::Error(e) => {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled(
                        " \u{F16A1} ",
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        e.clone(),
                        Style::default().fg(Color::Red),
                    ),
                ]));
            }
            ChatMessage::Info(text) => {
                lines.push(Line::from(""));
                for (i, line) in text.lines().enumerate() {
                    if i == 0 {
                        lines.push(Line::from(vec![
                            Span::styled(
                                " ℹ ",
                                Style::default()
                                    .fg(Color::Yellow)
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(
                                line.to_string(),
                                Style::default().fg(Color::Yellow),
                            ),
                        ]));
                    } else {
                        lines.push(Line::from(Span::styled(
                            format!("   {}", line),
                            Style::default().fg(Color::DarkGray),
                        )));
                    }
                }
            }
            ChatMessage::ContextInfo {
                context_used,
                context_size,
                user_messages,
                assistant_messages,
                tool_calls,
                user_chars,
                assistant_chars,
                tool_chars,
            } => {
                lines.push(Line::from(""));
                // Header
                lines.push(Line::from(vec![
                    Span::styled(
                        " ℹ ",
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "Context",
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));

                // Token counts
                if *context_size > 0 {
                    let (pct, bar) = context_bar_spans(*context_used, *context_size, 40);
                    lines.push(Line::from(Span::styled(
                        format!(
                            "   {} / {} tokens ({}%)",
                            format_number(*context_used),
                            format_number(*context_size),
                            pct
                        ),
                        Style::default().fg(Color::White),
                    )));

                    let mut bar_line = vec![Span::raw("   ")];
                    bar_line.extend(bar);
                    lines.push(Line::from(bar_line));
                } else if *context_used > 0 {
                    lines.push(Line::from(Span::styled(
                        format!("   {} tokens used (window size unknown)", format_number(*context_used)),
                        Style::default().fg(Color::White),
                    )));
                } else {
                    lines.push(Line::from(Span::styled(
                        "   No tokens used yet",
                        Style::default().fg(Color::DarkGray),
                    )));
                }

                // Message counts
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled("   Messages  ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!(
                            "{} user · {} assistant · {} tool",
                            user_messages, assistant_messages, tool_calls
                        ),
                        Style::default().fg(Color::White),
                    ),
                ]));

                // Character counts
                let total = *user_chars + *assistant_chars + *tool_chars;
                if total > 0 {
                    lines.push(Line::from(vec![
                        Span::styled("   Chars     ", Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            format!(
                                "{} user · {} assistant · {} tool",
                                format_number(*user_chars as u64),
                                format_number(*assistant_chars as u64),
                                format_number(*tool_chars as u64),
                            ),
                            Style::default().fg(Color::White),
                        ),
                    ]));
                }
            }
            ChatMessage::GenerationSummary { duration } => {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled(" ✻ ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("Crunched for {}", format_elapsed(*duration)),
                        Style::default()
                            .fg(Color::DarkGray)
                            .add_modifier(Modifier::ITALIC),
                    ),
                ]));
            }
        }
    }

    // Streaming response (rendered with markdown)
    if !app.current_response.is_empty() {
        lines.push(Line::from(""));
        let mut md_lines = render_markdown(app.current_response.trim_end(), width);
        patch_leading_circle(&mut md_lines);
        lines.extend(md_lines);
    }

    // Progress indicator
    if app.is_processing
        && !matches!(
            app.messages.last(),
            Some(ChatMessage::ToolCall { result: None, .. })
        )
    {
        let elapsed = app
            .generation_start
            .map(|s| s.elapsed())
            .unwrap_or_default();
        let time_str = format_elapsed(elapsed);
        let token_str = format_token_count(app.generation_tokens);

        let (symbol, symbol_color) = if app.has_received_tokens {
            ("·", Color::DarkGray)
        } else {
            ("✻", Color::Yellow)
        };

        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(format!(" {} ", symbol), Style::default().fg(symbol_color)),
            Span::styled(
                format!("{}… ({}{})", app.generation_verb, time_str, token_str),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            ),
        ]));
    }

    lines
}

// ── Tool result rendering ─────────────────────────────────────────────────

fn render_read_result(lines: &mut Vec<Line<'static>>, result: &ToolResultData, expanded: bool) {
    if !result.success {
        lines.push(Line::from(Span::styled(
            format::format_tool_error(&result.output),
            Style::default().fg(Color::Red),
        )));
        return;
    }

    if !expanded {
        let line_count = result.output.lines().count();
        let summary = if result.output.trim() == "(empty file)" {
            "(empty file)".to_string()
        } else {
            format!("{} lines (ctrl+o to expand)", line_count)
        };
        lines.push(Line::from(Span::styled(
            format!("{}{}", format::PREFIX_FIRST, summary),
            Style::default().fg(Color::DarkGray),
        )));
        return;
    }

    render_expanded_output(lines, &result.output);
}

fn render_edit_result(lines: &mut Vec<Line<'static>>, result: &ToolResultData) {
    if !result.success {
        lines.push(Line::from(Span::styled(
            format::format_tool_error(&result.output),
            Style::default().fg(Color::Red),
        )));
        return;
    }

    let diff_lines: Vec<&str> = result.output.lines().collect();

    // Count removed/added lines for summary
    let removed = diff_lines.iter().filter(|l| l.starts_with('-')).count();
    let added = diff_lines.iter().filter(|l| l.starts_with('+')).count();

    let summary = if removed > 0 && added > 0 {
        format!(
            "Removed {} line{}, added {} line{}",
            removed,
            if removed == 1 { "" } else { "s" },
            added,
            if added == 1 { "" } else { "s" },
        )
    } else if removed > 0 {
        format!(
            "Removed {} line{}",
            removed,
            if removed == 1 { "" } else { "s" }
        )
    } else if added > 0 {
        format!(
            "Added {} line{}",
            added,
            if added == 1 { "" } else { "s" }
        )
    } else {
        "No changes".to_string()
    };

    lines.push(Line::from(vec![
        Span::styled(format::PREFIX_FIRST, Style::default().fg(Color::DarkGray)),
        Span::styled(summary, Style::default().fg(Color::DarkGray)),
    ]));

    // Render diff lines with colors
    for diff_line in &diff_lines {
        let style = if diff_line.starts_with('-') {
            Style::default().fg(Color::Red)
        } else if diff_line.starts_with('+') {
            Style::default().fg(Color::Green)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        lines.push(Line::from(Span::styled(
            format!("      {}", diff_line),
            style,
        )));
    }
}

fn render_default_result(lines: &mut Vec<Line<'static>>, result: &ToolResultData, expanded: bool) {
    let line_count = result.output.lines().count();

    if expanded {
        render_expanded_output(lines, &result.output);
    } else {
        let summary = if line_count <= 1 {
            result.output.trim().to_string()
        } else {
            format!("{} lines (ctrl+o to expand)", line_count)
        };
        lines.push(Line::from(Span::styled(
            format!("   ⎿  {}", summary),
            Style::default().fg(Color::DarkGray),
        )));
    }
}

// ── Markdown rendering ─────────────────────────────────────────────────────

fn render_markdown(text: &str, width: u16) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut in_code_block = false;
    let width = width as usize;

    let code_prefix = "   │ ";
    let code_prefix_len = 5; // "   │ " = 5 chars

    for raw_line in text.lines() {
        let trimmed = raw_line.trim();

        // Code block fences
        if trimmed.starts_with("```") {
            if !in_code_block {
                // Opening fence — extract language
                let lang = trimmed[3..].trim().to_string();
                in_code_block = true;

                let label = if lang.is_empty() {
                    String::new()
                } else {
                    format!(" {} ", lang)
                };
                let bar_len = width.saturating_sub(4 + label.len());
                lines.push(Line::from(vec![
                    Span::styled("   ┌", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        label,
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "─".repeat(bar_len),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
            } else {
                // Closing fence
                in_code_block = false;
                let bar_len = width.saturating_sub(4);
                lines.push(Line::from(Span::styled(
                    format!("   └{}", "─".repeat(bar_len)),
                    Style::default().fg(Color::DarkGray),
                )));
            }
            continue;
        }

        // Inside code block — truncate to prevent wrapping
        if in_code_block {
            let max_content = width.saturating_sub(code_prefix_len);
            let char_count = raw_line.chars().count();
            let display = if char_count > max_content && max_content > 1 {
                let truncated: String = raw_line.chars().take(max_content - 1).collect();
                format!("{}…", truncated)
            } else {
                raw_line.to_string()
            };
            let content_len = display.chars().count();
            let padding = max_content.saturating_sub(content_len);

            lines.push(Line::from(vec![
                Span::styled(code_prefix, Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{}{}", display, " ".repeat(padding)),
                    Style::default().fg(Color::White),
                ),
            ]));
            continue;
        }

        // Empty line
        if trimmed.is_empty() {
            lines.push(Line::from(""));
            continue;
        }

        // Headings: # ## ### ####
        let hash_count = trimmed.chars().take_while(|c| *c == '#').count();
        if hash_count > 0 && hash_count <= 4 {
            if let Some(rest) = trimmed.get(hash_count..) {
                if rest.starts_with(' ') {
                    let heading_text = rest.trim();
                    let heading_style = Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::BOLD);
                    let mut spans = vec![Span::raw(MD_INDENT.to_string())];
                    for mut span in parse_inline_markdown(heading_text) {
                        span.style = span.style.patch(heading_style);
                        spans.push(span);
                    }
                    lines.push(Line::from(spans));
                    continue;
                }
            }
        }

        // Unordered list: * or -
        if (trimmed.starts_with("* ") || trimmed.starts_with("- ")) && trimmed.len() > 2 {
            let rest = trimmed[2..].trim_start();
            let mut spans = vec![Span::raw("   • ".to_string())];
            spans.extend(parse_inline_markdown(rest));
            lines.push(Line::from(spans));
            continue;
        }

        // Numbered list: 1. 2. etc.
        if let Some(dot_pos) = trimmed.find(". ") {
            if dot_pos > 0
                && dot_pos <= 3
                && trimmed[..dot_pos].chars().all(|c| c.is_ascii_digit())
            {
                let num = &trimmed[..dot_pos];
                let rest = trimmed[dot_pos + 2..].trim_start();
                let mut spans = vec![Span::raw(format!("   {}. ", num))];
                spans.extend(parse_inline_markdown(rest));
                lines.push(Line::from(spans));
                continue;
            }
        }

        // Regular text with inline markdown
        let mut spans = vec![Span::raw(MD_INDENT.to_string())];
        spans.extend(parse_inline_markdown(trimmed));
        lines.push(Line::from(spans));
    }

    lines
}

fn parse_inline_markdown(text: &str) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Escaped characters: \_ \* \` \\
        if chars[i] == '\\' && i + 1 < len && matches!(chars[i + 1], '_' | '*' | '`' | '\\') {
            current.push(chars[i + 1]);
            i += 2;
            continue;
        }

        // Backtick — inline code
        if chars[i] == '`' {
            if !current.is_empty() {
                spans.push(Span::raw(std::mem::take(&mut current)));
            }
            i += 1;
            while i < len && chars[i] != '`' {
                current.push(chars[i]);
                i += 1;
            }
            if i < len {
                i += 1;
            }
            if !current.is_empty() {
                spans.push(Span::styled(
                    std::mem::take(&mut current),
                    Style::default().fg(Color::Yellow),
                ));
            }
            continue;
        }

        // ** — bold
        if chars[i] == '*' && i + 1 < len && chars[i + 1] == '*' {
            if !current.is_empty() {
                spans.push(Span::raw(std::mem::take(&mut current)));
            }
            i += 2;
            while i < len {
                if chars[i] == '*' && i + 1 < len && chars[i + 1] == '*' {
                    i += 2;
                    break;
                }
                current.push(chars[i]);
                i += 1;
            }
            if !current.is_empty() {
                spans.push(Span::styled(
                    std::mem::take(&mut current),
                    Style::default().add_modifier(Modifier::BOLD),
                ));
            }
            continue;
        }

        // Regular character
        current.push(chars[i]);
        i += 1;
    }

    if !current.is_empty() {
        spans.push(Span::raw(current));
    }

    spans
}
