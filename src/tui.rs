use anyhow::Result;
use crossterm::{
    event::{Event, EventStream, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::StreamExt;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io;
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentEvent};

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

fn spinner_frame() -> &'static str {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    SPINNER[(ms / 80) as usize % SPINNER.len()]
}

#[derive(Clone)]
enum ChatMessage {
    User(String),
    Assistant(String),
    ToolCall { name: String, args: String },
    ToolResult { output: String },
    Error(String),
}

struct App {
    messages: Vec<ChatMessage>,
    current_response: String,
    input: String,
    cursor_pos: usize,
    is_processing: bool,
    model: String,
    should_quit: bool,
}

impl App {
    fn new(model: String) -> Self {
        Self {
            messages: Vec::new(),
            current_response: String::new(),
            input: String::new(),
            cursor_pos: 0,
            is_processing: false,
            model,
            should_quit: false,
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
        Some(msg)
    }

    fn flush_streaming(&mut self) {
        if !self.current_response.is_empty() {
            self.messages.push(ChatMessage::Assistant(std::mem::take(
                &mut self.current_response,
            )));
        }
    }
}

pub async fn run(agent: Agent) -> Result<()> {
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
    let mut app = App::new(model);

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<AgentEvent>();
    let (input_tx, mut input_rx) = mpsc::unbounded_channel::<String>();

    tokio::spawn(async move {
        let mut agent = agent;
        while let Some(msg) = input_rx.recv().await {
            let _ = agent.run(&msg, &event_tx).await;
        }
    });

    let mut reader = EventStream::new();
    let mut tick = tokio::time::interval(std::time::Duration::from_millis(80));

    loop {
        terminal.draw(|f| render(f, &app))?;

        tokio::select! {
            _ = tick.tick() => {}
            Some(Ok(evt)) = reader.next() => {
                handle_terminal_event(evt, &mut app, &input_tx);
            }
            event = event_rx.recv() => {
                match event {
                    Some(e) => handle_agent_event(e, &mut app),
                    None => {
                        app.messages.push(ChatMessage::Error("Agent disconnected".into()));
                        app.is_processing = false;
                    }
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}

fn handle_terminal_event(evt: Event, app: &mut App, input_tx: &mpsc::UnboundedSender<String>) {
    if let Event::Key(key) = evt {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            app.should_quit = true;
            return;
        }

        if app.is_processing {
            if key.code == KeyCode::Esc {
                app.should_quit = true;
            }
            return;
        }

        match key.code {
            KeyCode::Enter => {
                if let Some(msg) = app.submit() {
                    let _ = input_tx.send(msg);
                }
            }
            KeyCode::Char(c) => {
                app.input.insert(app.cursor_pos, c);
                app.cursor_pos += 1;
            }
            KeyCode::Backspace => {
                if app.cursor_pos > 0 {
                    app.cursor_pos -= 1;
                    app.input.remove(app.cursor_pos);
                }
            }
            KeyCode::Left => {
                app.cursor_pos = app.cursor_pos.saturating_sub(1);
            }
            KeyCode::Right => {
                if app.cursor_pos < app.input.len() {
                    app.cursor_pos += 1;
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
    match event {
        AgentEvent::Token(t) => {
            app.current_response.push_str(&t);
        }
        AgentEvent::ToolCall { name, args } => {
            app.flush_streaming();
            app.messages.push(ChatMessage::ToolCall { name, args });
        }
        AgentEvent::ToolResult { output, .. } => {
            app.messages.push(ChatMessage::ToolResult { output });
        }
        AgentEvent::Done => {
            app.flush_streaming();
            app.is_processing = false;
        }
        AgentEvent::Error(e) => {
            app.flush_streaming();
            app.messages.push(ChatMessage::Error(e));
            app.is_processing = false;
        }
    }
}

// ── Rendering ──────────────────────────────────────────────────────────────

fn render(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Header
            Constraint::Min(1),   // Chat
            Constraint::Length(3), // Input
        ])
        .split(f.area());

    render_header(f, app, chunks[0]);
    render_chat(f, app, chunks[1]);
    render_input(f, app, chunks[2]);
}

fn render_header(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let header = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(
                " imp",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" · ", Style::default().fg(Color::DarkGray)),
            Span::styled(app.model.as_str(), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(Span::styled(
            " ─".to_string() + &"─".repeat(area.width.saturating_sub(2) as usize),
            Style::default().fg(Color::DarkGray),
        )),
    ]);
    f.render_widget(header, area);
}

fn render_chat(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let lines = build_chat_lines(app);
    let text = Text::from(lines);
    let paragraph = Paragraph::new(text).wrap(Wrap { trim: false });

    let line_count = paragraph.line_count(area.width) as u16;
    let scroll = line_count.saturating_sub(area.height);

    let paragraph = paragraph.scroll((scroll, 0));
    f.render_widget(paragraph, area);
}

fn render_input(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let input_block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray));

    let input_line = if app.is_processing {
        let frame = spinner_frame();
        let label = if app.current_response.is_empty() {
            "Thinking"
        } else {
            "Responding"
        };
        Line::from(vec![
            Span::styled(format!(" {} ", frame), Style::default().fg(Color::Cyan)),
            Span::styled(
                label,
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            ),
        ])
    } else {
        Line::from(vec![
            Span::styled(
                " › ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(app.input.as_str()),
        ])
    };

    let input = Paragraph::new(input_line).block(input_block);
    f.render_widget(input, area);

    if !app.is_processing {
        f.set_cursor_position((area.x + 3 + app.cursor_pos as u16, area.y + 1));
    }
}

// ── Chat content builder ───────────────────────────────────────────────────

fn build_chat_lines(app: &App) -> Vec<Line<'static>> {
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
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled(
                        " › ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        text.clone(),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]));
            }
            ChatMessage::Assistant(text) => {
                lines.push(Line::from(""));
                lines.extend(render_markdown(text));
            }
            ChatMessage::ToolCall { name, args } => {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("   ⚙ {}", name),
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::styled(format!(" $ {}", args), Style::default().fg(Color::DarkGray)),
                ]));
            }
            ChatMessage::ToolResult { output } => {
                let result_lines: Vec<&str> = output.lines().collect();
                let display = if result_lines.len() > 20 {
                    let mut d: Vec<&str> = result_lines[..20].to_vec();
                    d.push("...");
                    d
                } else {
                    result_lines
                };
                for line in display {
                    lines.push(Line::from(Span::styled(
                        format!("   ┃ {}", line),
                        Style::default().fg(Color::DarkGray),
                    )));
                }
            }
            ChatMessage::Error(e) => {
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    format!("   error: {}", e),
                    Style::default().fg(Color::Red),
                )));
            }
        }
    }

    // Streaming response (rendered with markdown)
    if !app.current_response.is_empty() {
        lines.push(Line::from(""));
        lines.extend(render_markdown(&app.current_response));
    }

    // Thinking spinner
    if app.is_processing
        && app.current_response.is_empty()
        && !matches!(
            app.messages.last(),
            Some(ChatMessage::ToolCall { .. }) | Some(ChatMessage::ToolResult { .. })
        )
    {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(
                format!("   {} ", spinner_frame()),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled(
                "Thinking",
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            ),
        ]));
    }

    lines
}

// ── Markdown rendering ─────────────────────────────────────────────────────

fn render_markdown(text: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut in_code_block = false;

    for raw_line in text.lines() {
        let trimmed = raw_line.trim();

        // Code block fences
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }

        // Inside code block — render with code style
        if in_code_block {
            lines.push(Line::from(Span::styled(
                format!   ("   │ {}", raw_line),
                Style::default().fg(Color::Green),
            )));
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
                    lines.push(Line::from(vec![
                        Span::raw("   ".to_string()),
                        Span::styled(
                            heading_text.to_string(),
                            Style::default()
                                .fg(Color::Blue)
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]));
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
        let mut spans = vec![Span::raw("   ".to_string())];
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
