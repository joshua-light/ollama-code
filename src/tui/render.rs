use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::commands;
use crate::format;

use super::app::{App, ChatMessage, ToolResultData};
use super::markdown::{render_markdown, MD_INDENT};

pub(super) const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
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

pub(super) fn patch_leading_circle(lines: &mut [Line]) {
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

pub(super) fn spinner_frame() -> &'static str {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    SPINNER[(ms / 80) as usize % SPINNER.len()]
}

pub(crate) fn pick_verb() -> String {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    VERBS[(ms as usize) % VERBS.len()].to_string()
}

pub(super) fn format_elapsed(d: std::time::Duration) -> String {
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

pub(crate) fn format_number(n: u64) -> String {
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

pub(crate) fn get_git_info_sync() -> (Option<String>, bool) {
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

// ── Rendering ──────────────────────────────────────────────────────────────

pub(super) fn compute_input_height(input: &str, term_width: u16, term_height: u16, is_processing: bool) -> u16 {
    if is_processing || input.is_empty() {
        return 3; // border + 1 line + border
    }
    let content_width = term_width.saturating_sub(3).max(1) as usize; // 3 for " ❯ "
    let mut visual_lines: u16 = 0;
    for logical_line in input.split('\n') {
        let char_count = logical_line.chars().count();
        visual_lines += ((char_count.max(1) + content_width - 1) / content_width).max(1) as u16;
    }
    // Cap so chat area stays usable (leave room for header + status + at least 3 chat lines)
    let max_lines = term_height.saturating_sub(6).max(3);
    visual_lines.min(max_lines) + 2 // + top border + bottom border
}

pub(super) fn render(f: &mut Frame, app: &mut App) {
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
        let mut first = true;
        for logical_line in app.input.split('\n') {
            let lchars: Vec<char> = logical_line.chars().collect();
            if lchars.is_empty() {
                // Empty logical line (just a newline)
                let prefix = if first { prompt } else { "   " };
                lines.push(Line::from(vec![
                    Span::styled(
                        prefix,
                        if first {
                            Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
                        } else {
                            Style::default()
                        },
                    ),
                    Span::raw(""),
                ]));
                first = false;
                continue;
            }
            let mut i = 0;
            while i < lchars.len() {
                let end = (i + content_width).min(lchars.len());
                let chunk: String = lchars[i..end].iter().collect();
                if first {
                    lines.push(Line::from(vec![
                        Span::styled(
                            prompt,
                            Style::default()
                                .fg(Color::White)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(chunk),
                    ]));
                    first = false;
                } else {
                    lines.push(Line::from(vec![
                        Span::raw("   "),
                        Span::raw(chunk),
                    ]));
                }
                i = end;
            }
        }
    }

    let input = Paragraph::new(lines).block(input_block);
    f.render_widget(input, area);

    // Compute cursor position accounting for newlines and wrapping
    let before_cursor = &app.input[..app.cursor_pos];
    let mut visual_line: u16 = 0;
    let mut col_in_line: usize = 0;
    for (li, logical_line) in before_cursor.split('\n').enumerate() {
        let char_count = logical_line.chars().count();
        if li < before_cursor.matches('\n').count() {
            // This logical line is fully before the cursor — count its visual lines
            visual_line += ((char_count.max(1) + content_width - 1) / content_width).max(1) as u16;
        } else {
            // Cursor is in this logical line
            visual_line += (char_count / content_width) as u16;
            col_in_line = char_count % content_width;
        }
    }
    f.set_cursor_position((area.x + 3 + col_in_line as u16, area.y + 1 + visual_line));
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

    // Bypass permissions indicator
    if app.auto_approve {
        spans.push(sep_span());
        spans.push(Span::styled(
            "⏵⏵ bypass",
            Style::default()
                .fg(Color::Yellow)
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

// ── Chat content builder ──────────────────────────────────────────────��────

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
                                "\u{F16A3} ",
                                Style::default()
                                    .fg(Color::Cyan)
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(
                                line.to_string(),
                                Style::default().fg(Color::Cyan),
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
                        " \u{F16A3} ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "Context",
                        Style::default()
                            .fg(Color::Cyan)
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
            ChatMessage::SubagentToolCall { name, args, success } => {
                let indicator = match success {
                    Some(true) => Span::styled("✓", Style::default().fg(Color::Green)),
                    Some(false) => Span::styled("✗", Style::default().fg(Color::Red)),
                    None => Span::styled("·", Style::default().fg(Color::DarkGray)),
                };
                lines.push(Line::from(vec![
                    Span::raw("     "),
                    Span::styled("↳ ", Style::default().fg(Color::DarkGray)),
                    indicator,
                    Span::styled(
                        format!(
                            " {}({})",
                            format::capitalize_first(name),
                            format::truncate_args(args, 60),
                        ),
                        Style::default().fg(Color::DarkGray),
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

    // Tool confirmation prompt
    if let Some(confirm) = &app.pending_confirm {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(
                " ? ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "Allow {}({})? ",
                    format::capitalize_first(&confirm.name),
                    format::truncate_args(&confirm.args, 60),
                ),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled("(y/n)", Style::default().fg(Color::DarkGray)),
        ]));
    }

    // Progress indicator
    if app.is_processing
        && app.pending_confirm.is_none()
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
            format!("{}{}", format::PREFIX_FIRST, summary),
            Style::default().fg(Color::DarkGray),
        )));
    }
}
