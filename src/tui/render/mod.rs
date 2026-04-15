mod chat_content;

use ratatui::{
    layout::{Constraint, Direction, Layout, Position},
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

pub(crate) async fn get_git_info_async() -> (Option<String>, bool) {
    let (branch_out, dirty_out) = tokio::join!(
        tokio::process::Command::new("git")
            .args(["branch", "--show-current"])
            .output(),
        tokio::process::Command::new("git")
            .args(["diff-index", "--quiet", "HEAD", "--"])
            .output(),
    );

    let branch = branch_out
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());

    let dirty = dirty_out
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

pub(super) fn compute_input_height(input: &str, term_width: u16, term_height: u16) -> u16 {
    if input.is_empty() {
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
    let input_height = compute_input_height(&app.input, area.width, area.height);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),            // Header
            Constraint::Min(1),              // Chat
            Constraint::Length(input_height), // Input (dynamic)
            Constraint::Length(1),            // Status line
        ])
        .split(f.area());

    app.chat_area = chunks[1];
    render_header(f, app, chunks[0]);
    render_chat(f, app, chunks[1]);
    render_input(f, app, chunks[2]);
    render_status_line(f, app, chunks[3]);

    // Picker overlay (rendered on top of everything)
    if let Some(ref mut picker) = app.picker {
        picker.render(f, f.area());
    }

    // Tree browser overlay
    if let Some(ref mut tree) = app.tree_browser {
        tree.render(f, f.area());
    }

    // Settings panel overlay
    if let Some(ref mut settings) = app.settings {
        settings.render(f, f.area());
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

    // ── Selection highlight + text extraction ──
    if app.selection.is_none() {
        return;
    }
    let sel = app.selection.as_ref().unwrap();
    let (start, end) = sel.ordered();
    let scroll_usize = scroll as usize;

    let buf = f.buffer_mut();

    // Cache visible line texts (accumulates across frames while selecting)
    for row_off in 0..area.height as usize {
        let y = area.y + row_off as u16;
        let content_line = scroll_usize + row_off;
        let mut line_text = String::new();
        for col in 0..area.width {
            if let Some(cell) = buf.cell(Position { x: area.x + col, y }) {
                line_text.push_str(cell.symbol());
            }
        }
        app.selection_line_cache.insert(content_line, line_text.trim_end().to_string());
    }

    // Highlight selected cells
    for content_line in start.0..=end.0 {
        if content_line < scroll_usize || content_line >= scroll_usize + area.height as usize {
            continue;
        }
        let screen_row = (content_line - scroll_usize) as u16 + area.y;
        let col_start = if content_line == start.0 { start.1 } else { 0 };
        let col_end = if content_line == end.0 { end.1 } else { area.width.saturating_sub(1) };

        for col in col_start..=col_end {
            let x = area.x + col;
            if x >= area.x + area.width { break; }
            if let Some(cell) = buf.cell_mut(Position { x, y: screen_row }) {
                cell.set_style(Style::default().bg(Color::DarkGray).fg(Color::White));
            }
        }
    }

    // Extract selected text for clipboard when mouse was released
    if app.copy_selection {
        app.copy_selection = false;
        let mut result = String::new();
        for content_line in start.0..=end.0 {
            if content_line > start.0 { result.push('\n'); }
            if let Some(line) = app.selection_line_cache.get(&content_line) {
                let chars: Vec<char> = line.chars().collect();
                let cs = if content_line == start.0 { start.1 as usize } else { 0 };
                let ce = if content_line == end.0 { (end.1 as usize + 1).min(chars.len()) } else { chars.len() };
                let cs = cs.min(chars.len());
                let selected: String = chars[cs..ce].iter().collect();
                result.push_str(selected.trim_end());
            }
        }
        let trimmed = result.trim_end().to_string();
        if !trimmed.is_empty() {
            app.clipboard_text = Some(trimmed);
        }
    }
}

fn render_input(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let input_block = Block::default()
        .borders(Borders::TOP | Borders::BOTTOM)
        .border_style(Style::default().fg(Color::DarkGray));

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
    // Prompt template variable fill mode — show variable prompt
    if let Some(ref fill) = app.pending_prompt {
        let var = &fill.variables[fill.current];
        let mut spans: Vec<Span> = Vec::new();
        spans.push(Span::styled(
            format!(" /{} ", fill.template.name),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!("[{}/{}] ", fill.current + 1, fill.variables.len()),
            Style::default().fg(Color::DarkGray),
        ));
        spans.push(Span::styled(
            format!("{}:", var.name),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
        if let Some(ref default) = var.default {
            spans.push(Span::styled(
                format!(" (default: {})", default),
                Style::default().fg(Color::DarkGray),
            ));
        }
        // Esc hint on the right
        let hint = " esc to cancel";
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
        f.render_widget(Paragraph::new(line), area);
        return;
    }

    // Show command + skill + prompt completions when input starts with /
    if !app.is_processing {
        let builtin_matches = commands::completions(&app.input);
        let prefix = app.input.trim();
        let skill_prefix = prefix.strip_prefix('/').unwrap_or("");
        let mut skill_matches: Vec<(&str, &str)> = if !skill_prefix.is_empty() {
            app.skills
                .iter()
                .filter(|s| s.name.starts_with(skill_prefix))
                .map(|s| (s.name.as_str(), s.description.as_str()))
                .collect()
        } else if prefix == "/" {
            app.skills
                .iter()
                .map(|s| (s.name.as_str(), s.description.as_str()))
                .collect()
        } else {
            Vec::new()
        };

        // Also include prompt template completions
        let prompt_matches: Vec<(&str, &str)> = if !skill_prefix.is_empty() {
            app.prompts
                .iter()
                .filter(|p| p.name.starts_with(skill_prefix))
                .map(|p| (p.name.as_str(), p.description.as_str()))
                .collect()
        } else if prefix == "/" {
            app.prompts
                .iter()
                .map(|p| (p.name.as_str(), p.description.as_str()))
                .collect()
        } else {
            Vec::new()
        };
        skill_matches.extend(prompt_matches);

        if !builtin_matches.is_empty() || !skill_matches.is_empty() {
            render_command_completions(f, area, &app.input, &builtin_matches, &skill_matches);
            return;
        }
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

    // Queued follow-up messages indicator
    if !app.followup_queue.is_empty() {
        let count = app.followup_queue.len();
        spans.push(sep_span());
        spans.push(Span::styled(
            format!("{} queued", count),
            Style::default().fg(Color::Cyan),
        ));
    }

    // Estimated cost (Anthropic Opus 4.6 pricing: $5/$25 per 1M tokens).
    // Only meaningful when proxying to a paid API; local Ollama models are free.
    if app.config.show_cost_estimate.unwrap_or(false)
        && (app.stats.input_tokens > 0 || app.stats.output_tokens > 0)
    {
        let (input_rate, output_rate) = app.config.cost_per_million_tokens
            .unwrap_or((5.0, 25.0));
        let cost = app.stats.input_tokens as f64 * input_rate / 1_000_000.0
                 + app.stats.output_tokens as f64 * output_rate / 1_000_000.0;
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

fn push_completion_item(
    spans: &mut Vec<Span>,
    item_idx: &mut usize,
    input: &str,
    display_name: &str,
    description: &str,
    color: Color,
) {
    if *item_idx > 0 {
        spans.push(Span::styled("  │  ", Style::default().fg(Color::DarkGray)));
    }
    let prefix_len = input.len().min(display_name.len());
    let matched = &display_name[..prefix_len];
    let rest = &display_name[prefix_len..];
    spans.push(Span::styled(" ", Style::default()));
    spans.push(Span::styled(
        matched.to_string(),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::styled(
        rest.to_string(),
        Style::default().fg(color),
    ));
    spans.push(Span::styled(
        format!(" {}", description),
        Style::default().fg(Color::DarkGray),
    ));
    *item_idx += 1;
}

fn render_command_completions(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    input: &str,
    builtin_matches: &[&commands::CommandInfo],
    skill_matches: &[(&str, &str)],
) {
    let input = input.trim();
    let mut spans: Vec<Span> = Vec::new();
    let mut item_idx = 0;

    for cmd in builtin_matches {
        push_completion_item(&mut spans, &mut item_idx, input, cmd.name, cmd.description, Color::Yellow);
    }

    for (name, description) in skill_matches {
        let full_name = format!("/{}", name);
        push_completion_item(&mut spans, &mut item_idx, input, &full_name, description, Color::Cyan);
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
