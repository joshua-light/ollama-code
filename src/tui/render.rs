use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::commands;
use crate::format;

use super::app::{App, ChatMessage, PendingConfirm, ServerLoadingState, ToolResultData};
use super::markdown::{render_markdown, MD_INDENT};

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

// ── Input wrapping helpers ────────────────────────────────────────────────

/// Content width available for input text (accounting for " ❯ " / "   " prefix).
fn input_content_width(available_width: u16) -> usize {
    available_width.saturating_sub(3).max(1) as usize
}

/// Count visual lines a logical line occupies when wrapped to `content_width` chars.
fn wrapped_line_count(char_count: usize, content_width: usize) -> usize {
    char_count.max(1).div_ceil(content_width)
}

// ── Rendering ──────────────────────────────────────────────────────────────

pub(super) fn compute_input_height(input: &str, term_width: u16, term_height: u16, is_processing: bool) -> u16 {
    if is_processing || input.is_empty() {
        return 3; // border + 1 line + border
    }
    let content_width = input_content_width(term_width);
    let mut visual_lines: u16 = 0;
    for logical_line in input.split('\n') {
        let char_count = logical_line.chars().count();
        visual_lines += wrapped_line_count(char_count, content_width) as u16;
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
    let content_width = input_content_width(area.width);
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
            visual_line += wrapped_line_count(char_count, content_width) as u16;
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
                .fg(Color::Red)
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

    // Estimated Opus 4.6 cost
    if app.config.show_cost_estimate.unwrap_or(false)
        && (app.total_input_tokens > 0 || app.total_output_tokens > 0)
    {
        let cost = app.total_input_tokens as f64 * 5.0 / 1_000_000.0
                 + app.total_output_tokens as f64 * 25.0 / 1_000_000.0;
        spans.push(sep_span());
        spans.push(Span::styled(
            format!("${:.4}", cost),
            Style::default().fg(Color::Cyan),
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

// ── Chat content builder ──────────────────────────────────────────────────

fn build_chat_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = Vec::new();

    if app.messages.is_empty() && app.current_response.is_empty() && app.server_loading.is_none() {
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
                render_chat_user(&mut lines, text);
            }
            ChatMessage::Assistant(text) => {
                render_chat_assistant(&mut lines, text, width);
            }
            ChatMessage::ToolCall { name, args, result } => {
                render_chat_tool_call(&mut lines, name, args, result.as_ref(), app.tools_expanded);
            }
            ChatMessage::Error(e) => {
                render_chat_error(&mut lines, e);
            }
            ChatMessage::Info(text) => {
                render_chat_info(&mut lines, text);
            }
            ChatMessage::ContextInfo {
                context_used,
                context_size,
                user_messages,
                assistant_messages,
                tool_calls,
                base_prompt_tokens,
                project_docs_tokens,
                skills_tokens,
                tool_defs_tokens,
            } => {
                render_chat_context_info(
                    &mut lines,
                    *context_used,
                    *context_size,
                    *user_messages,
                    *assistant_messages,
                    *tool_calls,
                    *base_prompt_tokens,
                    project_docs_tokens,
                    *skills_tokens,
                    *tool_defs_tokens,
                );
            }
            ChatMessage::GenerationSummary { duration } => {
                render_chat_generation_summary(&mut lines, *duration);
            }
            ChatMessage::SubagentToolCall { name, args, success } => {
                render_chat_subagent_tool_call(&mut lines, name, args, *success);
            }
        }
    }

    // Streaming response
    if !app.current_response.is_empty() {
        render_chat_streaming(&mut lines, &app.current_response, width);
    }

    // Tool confirmation prompt
    if let Some(confirm) = &app.pending_confirm {
        render_chat_confirm_prompt(&mut lines, confirm);
    }

    // Progress indicator
    if app.is_processing
        && app.pending_confirm.is_none()
        && !matches!(
            app.messages.last(),
            Some(ChatMessage::ToolCall { result: None, .. })
                | Some(ChatMessage::SubagentToolCall { .. })
        )
    {
        if let Some(ref loading) = app.server_loading {
            render_server_loading_progress(&mut lines, loading);
        } else {
            render_chat_progress(&mut lines, app);
        }
    }

    lines
}

// ── Per-variant chat renderers ────────────────────────────────────────────

fn render_chat_user(lines: &mut Vec<Line<'static>>, text: &str) {
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
                text.to_string(),
                Style::default()
                    .fg(Color::White)
                    .bg(Color::Indexed(236))
                    .add_modifier(Modifier::BOLD),
            ),
        ])
        .style(user_bg),
    );
}

fn render_chat_assistant(lines: &mut Vec<Line<'static>>, text: &str, width: u16) {
    lines.push(Line::from(""));
    let mut md_lines = render_markdown(text.trim_end(), width);
    patch_leading_circle(&mut md_lines);
    lines.extend(md_lines);
}

fn render_chat_tool_call(
    lines: &mut Vec<Line<'static>>,
    name: &str,
    args: &str,
    result: Option<&ToolResultData>,
    expanded: bool,
) {
    lines.push(Line::from(""));
    let circle_color = match result {
        Some(ToolResultData { success: true, .. }) => Color::Green,
        Some(ToolResultData { success: false, .. }) => Color::Red,
        None => Color::DarkGray,
    };
    lines.push(Line::from(vec![
        Span::styled(" ● ", Style::default().fg(circle_color)),
        Span::styled(
            format!("{}({})", format::capitalize_first(name), format::truncate_args(args, 77)),
            Style::default().fg(Color::White),
        ),
    ]));
    if let Some(result_data) = result {
        match name {
            "read" => {
                render_read_result(lines, result_data, expanded);
            }
            "edit" => {
                render_edit_result(lines, result_data);
            }
            "write" => {
                render_write_result(lines, result_data);
            }
            _ => {
                render_default_result(lines, result_data, expanded);
            }
        }
    }
}

fn render_chat_error(lines: &mut Vec<Line<'static>>, text: &str) {
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(
            " \u{F16A1} ",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            text.to_string(),
            Style::default().fg(Color::Red),
        ),
    ]));
}

fn render_chat_info(lines: &mut Vec<Line<'static>>, text: &str) {
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

#[allow(clippy::too_many_arguments)]
fn render_chat_context_info(
    lines: &mut Vec<Line<'static>>,
    context_used: u64,
    context_size: u64,
    user_messages: u32,
    assistant_messages: u32,
    tool_calls: u32,
    base_prompt_tokens: u64,
    project_docs_tokens: &[(String, u64)],
    skills_tokens: u64,
    tool_defs_tokens: u64,
) {
    lines.push(Line::from(""));

    // Header with inline summary
    let header_detail = if context_size > 0 {
        let pct = ((context_used as f64 / context_size as f64) * 100.0).min(100.0) as u64;
        format!(
            " — {} / {} tokens ({}%)",
            format_number(context_used),
            format_number(context_size),
            pct,
        )
    } else if context_used > 0 {
        format!(" — {} tokens", format_number(context_used))
    } else {
        String::new()
    };
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
        Span::styled(header_detail, Style::default().fg(Color::DarkGray)),
    ]));

    // Build segments for the breakdown.
    // Each segment: (label, estimated tokens, color).
    // "System" = base prompt; project docs, skills, tool schemas are separate.
    // "Messages" = everything else (context_used minus the known fixed costs).
    let dim = Style::default().fg(Color::DarkGray);
    let white = Style::default().fg(Color::White);

    // Colors for segments (chosen for visual distinctness in 256-color terminals).
    let c_system = Color::Indexed(67);   // steel blue
    let c_docs = Color::Indexed(109);    // light teal
    let c_skills = Color::Indexed(139);  // muted purple
    let c_tools = Color::Indexed(173);   // salmon
    let c_msgs = Color::Indexed(150);    // light green
    let c_free = Color::Indexed(238);    // dark gray

    let docs_total: u64 = project_docs_tokens.iter().map(|(_, t)| *t).sum();
    let system_overhead = base_prompt_tokens + docs_total + skills_tokens + tool_defs_tokens;
    // Messages tokens = total used minus the fixed overhead (clamped to 0).
    let messages_tokens = context_used.saturating_sub(system_overhead);
    let free_tokens = context_size.saturating_sub(context_used);

    // Segmented progress bar (only when context_size is known).
    if context_size > 0 {
        let bar_width: usize = 40;
        // Segments: system, docs, skills, tool schemas, messages, free.
        let segments: Vec<(u64, Color)> = vec![
            (base_prompt_tokens, c_system),
            (docs_total, c_docs),
            (skills_tokens, c_skills),
            (tool_defs_tokens, c_tools),
            (messages_tokens, c_msgs),
            (free_tokens, c_free),
        ];
        let mut bar_spans = vec![Span::raw("   ")];
        let total = context_size.max(1) as f64;
        let mut used_cols: usize = 0;
        for (i, (tokens, color)) in segments.iter().enumerate() {
            let is_last = i == segments.len() - 1;
            let cols = if is_last {
                bar_width - used_cols
            } else {
                let raw = ((*tokens as f64 / total) * bar_width as f64).round() as usize;
                // Ensure non-zero tokens get at least 1 column, but don't overflow.
                if *tokens > 0 && raw == 0 {
                    1usize.min(bar_width - used_cols)
                } else {
                    raw.min(bar_width - used_cols)
                }
            };
            if cols > 0 {
                let ch = if *color == c_free { "╌" } else { "━" };
                bar_spans.push(Span::styled(
                    ch.repeat(cols),
                    Style::default().fg(*color),
                ));
            }
            used_cols += cols;
        }
        lines.push(Line::from(bar_spans));
    }

    // Legend: one line per category, with colored bullet + token count.
    lines.push(Line::from(""));

    // Helper closure: push a legend line.
    let push_legend = |lines: &mut Vec<Line<'static>>, color: Color, label: &str, tokens: u64, detail: &str| {
        let mut spans = vec![
            Span::styled("   ━ ", Style::default().fg(color)),
            Span::styled(format!("{:<14}", label), dim),
            Span::styled(format!("~{}", format_number(tokens)), white),
        ];
        if !detail.is_empty() {
            spans.push(Span::styled(format!("  {}", detail), dim));
        }
        lines.push(Line::from(spans));
    };

    if base_prompt_tokens > 0 {
        push_legend(lines, c_system, "System", base_prompt_tokens, "");
    }
    for (name, tokens) in project_docs_tokens {
        push_legend(lines, c_docs, name, *tokens, "");
    }
    if skills_tokens > 0 {
        push_legend(lines, c_skills, "Skills", skills_tokens, "");
    }
    if tool_defs_tokens > 0 {
        push_legend(lines, c_tools, "Tool schemas", tool_defs_tokens, "");
    }
    let msg_detail = format!(
        "{} user · {} assistant · {} tool",
        user_messages, assistant_messages, tool_calls,
    );
    push_legend(lines, c_msgs, "Messages", messages_tokens, &msg_detail);
    if context_size > 0 {
        push_legend(lines, c_free, "Free", free_tokens, "");
    }
}

fn render_chat_generation_summary(lines: &mut Vec<Line<'static>>, duration: std::time::Duration) {
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(" ✻ ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("Crunched for {}", format_elapsed(duration)),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        ),
    ]));
}

fn render_chat_subagent_tool_call(
    lines: &mut Vec<Line<'static>>,
    name: &str,
    args: &str,
    success: Option<bool>,
) {
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

fn render_chat_streaming(lines: &mut Vec<Line<'static>>, response: &str, width: u16) {
    render_chat_assistant(lines, response, width);
}

fn render_chat_confirm_prompt(lines: &mut Vec<Line<'static>>, confirm: &PendingConfirm) {
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

fn render_chat_progress(lines: &mut Vec<Line<'static>>, app: &App) {
    let elapsed = app
        .generation_start
        .map(|s| s.elapsed())
        .unwrap_or_default();
    let time_str = format_elapsed(elapsed);
    let token_str = format_token_count(app.generation_tokens);

    let (symbol, symbol_color) = if app.has_received_tokens {
        ("·", Color::DarkGray)
    } else {
        ("✻", Color::DarkGray)
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

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

fn render_server_loading_progress(lines: &mut Vec<Line<'static>>, loading: &ServerLoadingState) {
    let elapsed = loading.start.elapsed();
    let time_str = format_elapsed(elapsed);
    let pct = (loading.progress * 100.0) as u64;

    let bar_width: usize = 30;
    let filled = ((loading.progress * bar_width as f32) as usize).min(bar_width);
    let empty = bar_width - filled;

    let bar_color = if pct > 80 {
        Color::Green
    } else {
        Color::Cyan
    };

    let spinner_idx = (elapsed.as_millis() / 80) as usize % SPINNER.len();

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(
            format!(" {} ", SPINNER[spinner_idx]),
            Style::default().fg(Color::Yellow),
        ),
        Span::styled(
            format!("Loading {}…", loading.model_name),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::raw("   "),
        Span::styled("━".repeat(filled), Style::default().fg(bar_color)),
        Span::styled("╌".repeat(empty), Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!(" {}%", pct),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  ({})", time_str),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        ),
    ]));
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
            Style::default()
                .fg(Color::White)
                .bg(Color::Indexed(52))
        } else if diff_line.starts_with('+') {
            Style::default()
                .fg(Color::White)
                .bg(Color::Indexed(22))
        } else {
            Style::default().fg(Color::DarkGray)
        };
        lines.push(Line::from(Span::styled(
            format!("      {}", diff_line),
            style,
        )));
    }
}

fn render_write_result(lines: &mut Vec<Line<'static>>, result: &ToolResultData) {
    if !result.success {
        lines.push(Line::from(Span::styled(
            format::format_tool_error(&result.output),
            Style::default().fg(Color::Red),
        )));
        return;
    }

    // First line is the summary "Created 'path' (N lines)", rest is the diff
    let mut output_lines = result.output.lines();
    let summary = output_lines.next().unwrap_or("Created file");

    lines.push(Line::from(vec![
        Span::styled(format::PREFIX_FIRST, Style::default().fg(Color::DarkGray)),
        Span::styled(summary.to_string(), Style::default().fg(Color::DarkGray)),
    ]));

    let max_diff_lines = 30;
    let mut shown = 0;
    let mut remaining = 0;
    for diff_line in output_lines {
        if diff_line.is_empty() {
            continue;
        }
        if shown < max_diff_lines {
            lines.push(Line::from(Span::styled(
                format!("      {}", diff_line),
                Style::default()
                    .fg(Color::White)
                    .bg(Color::Indexed(22)),
            )));
            shown += 1;
        } else {
            remaining += 1;
        }
    }
    if remaining > 0 {
        lines.push(Line::from(Span::styled(
            format!("      ... ({} more lines)", remaining),
            Style::default().fg(Color::DarkGray),
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
