mod chat_content;

use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::commands;
use crate::format;

use super::app::App;
use super::markdown::MD_INDENT;

pub(super) const VERBS: &[&str] = &[
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

pub(super) fn context_bar_spans(used: u64, total: u64, bar_width: usize) -> (u64, Vec<Span<'static>>) {
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

pub(super) fn render_expanded_output(lines: &mut Vec<Line<'static>>, output: &str) {
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

pub(super) fn format_token_count(n: usize) -> String {
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

    // Picker overlay (rendered on top of everything)
    if let Some(ref mut picker) = app.picker {
        picker.render(f, f.area());
    }
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
    let lines = chat_content::build_chat_lines(app, area.width);
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

    let prompt = " ❯ ";
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
            app.context_used + app.generation.tokens as u64
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
    let time_str = format_elapsed(app.stats.session_start.elapsed());
    spans.push(sep_span());
    spans.push(Span::styled("⏱ ", Style::default().fg(Color::Blue)));
    spans.push(Span::styled(time_str, Style::default().fg(Color::Blue)));

    // Tool call count
    if app.stats.tool_call_count > 0 {
        spans.push(sep_span());
        spans.push(Span::styled(
            format!("⚙ {}", app.stats.tool_call_count),
            Style::default().fg(Color::Yellow),
        ));
    }

    // Estimated Opus 4.6 cost
    if app.config.show_cost_estimate.unwrap_or(false)
        && (app.stats.input_tokens > 0 || app.stats.output_tokens > 0)
    {
        let cost = app.stats.input_tokens as f64 * 5.0 / 1_000_000.0
                 + app.stats.output_tokens as f64 * 25.0 / 1_000_000.0;
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
