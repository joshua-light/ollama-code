use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use crate::format;

use super::super::app::{App, ChatMessage, ContextInfoData, PendingConfirm, ServerLoadingState, StatsInfoData, ToolResultData};
use super::super::markdown::render_markdown;
use super::super::syntax;
use super::{
    format_elapsed, format_number, format_token_count,
    patch_leading_circle, render_expanded_output,
};

pub(super) fn build_chat_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = Vec::new();

    if app.messages.is_empty() && app.current_response.is_empty() && app.server.loading.is_none() {
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
            ChatMessage::ContextInfo(data) => {
                render_chat_context_info(&mut lines, data);
            }
            ChatMessage::StatsInfo(data) => {
                render_chat_stats_info(&mut lines, data);
            }
            ChatMessage::GenerationSummary { duration } => {
                render_chat_generation_summary(&mut lines, *duration);
            }
            ChatMessage::SubagentToolCall { name, args, success } => {
                render_chat_subagent_tool_call(&mut lines, name, args, *success);
            }
            ChatMessage::SkillLoad { name } => {
                render_chat_skill_load(&mut lines, name);
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
        if let Some(ref loading) = app.server.loading {
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
            format!("{}({})", format::format_tool_name(name), format::truncate_args(args, 77)),
            Style::default().fg(Color::White),
        ),
    ]));
    if let Some(result_data) = result {
        match name {
            "read" => {
                render_read_result(lines, result_data, expanded);
            }
            "edit" => {
                render_edit_result(lines, result_data, args);
            }
            "write" => {
                render_write_result(lines, result_data, args);
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

fn render_chat_context_info(lines: &mut Vec<Line<'static>>, data: &ContextInfoData) {
    let context_used = data.context_used;
    let context_size = data.context_size;
    let user_messages = data.user_messages;
    let assistant_messages = data.assistant_messages;
    let tool_calls = data.tool_calls;
    let base_prompt_tokens = data.base_prompt_tokens;
    let project_docs_tokens = &data.project_docs_tokens;
    let skills_tokens = data.skills_tokens;
    let tool_defs_breakdown = &data.tool_defs_breakdown;
    let tool_defs_tokens: u64 = tool_defs_breakdown.iter().map(|(_, t)| *t).sum();

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
            "Context Window",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(header_detail, Style::default().fg(Color::DarkGray)),
    ]));

    // Build segments for the breakdown.
    let dim = Style::default().fg(Color::DarkGray);
    let white = Style::default().fg(Color::White);

    // Colors for segments
    let c_system = Color::Indexed(67);   // steel blue
    let c_docs = Color::Indexed(109);    // light teal
    let c_skills = Color::Indexed(139);  // muted purple
    let c_tools = Color::Indexed(173);   // salmon
    let c_msgs = Color::Indexed(150);    // light green
    let c_free = Color::Indexed(238);    // dark gray

    let docs_total: u64 = project_docs_tokens.iter().map(|(_, t)| *t).sum();
    let system_overhead = base_prompt_tokens + docs_total + skills_tokens + tool_defs_tokens;
    let messages_tokens = context_used.saturating_sub(system_overhead);
    let free_tokens = context_size.saturating_sub(context_used);

    // Segmented progress bar
    if context_size > 0 {
        let bar_width: usize = 40;
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

    // Legend
    lines.push(Line::from(""));

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
        if tool_defs_breakdown.len() > 1 {
            // Show aggregate header then per-category sub-items.
            push_legend(lines, c_tools, "Tool schemas", tool_defs_tokens, "");
            for (label, tokens) in tool_defs_breakdown {
                lines.push(Line::from(vec![
                    Span::styled("     · ", Style::default().fg(c_tools)),
                    Span::styled(format!("{:<12}", label), dim),
                    Span::styled(format!("~{}", format_number(*tokens)), white),
                ]));
            }
        } else {
            // Single category — just show the flat line (label from breakdown or fallback).
            let label = tool_defs_breakdown.first()
                .map(|(l, _)| l.as_str())
                .unwrap_or("Tool schemas");
            push_legend(lines, c_tools, label, tool_defs_tokens, "");
        }
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

fn render_chat_stats_info(lines: &mut Vec<Line<'static>>, data: &StatsInfoData) {
    let dim = Style::default().fg(Color::DarkGray);
    let white = Style::default().fg(Color::White);

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
            "Session Stats",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" \u{2014} {}", format_elapsed(data.session_duration)),
            dim,
        ),
    ]));

    lines.push(Line::from(""));

    // Helper closure for label-value rows
    let push_row = |lines: &mut Vec<Line<'static>>, label: &str, value: String| {
        lines.push(Line::from(vec![
            Span::styled(format!("   {:<16}", label), dim),
            Span::styled(value, white),
        ]));
    };

    // General section
    push_row(lines, "Model", data.model.clone());
    push_row(lines, "Agent turns", data.agent_turns.to_string());
    if data.context_trims > 0 {
        push_row(lines, "Context trims", data.context_trims.to_string());
    }

    lines.push(Line::from(""));

    // Tool calls section
    let tool_summary = if data.failed_tool_call_count > 0 {
        let ok = data.tool_call_count.saturating_sub(data.failed_tool_call_count);
        format!(
            "{} ({} ok \u{00B7} {} failed)",
            data.tool_call_count, ok, data.failed_tool_call_count
        )
    } else {
        data.tool_call_count.to_string()
    };
    push_row(lines, "Tool calls", tool_summary);

    // Per-tool breakdown (already sorted by count descending)
    if !data.tool_call_breakdown.is_empty() {
        let display_names: Vec<(String, &usize)> = data.tool_call_breakdown.iter()
            .map(|(name, count)| (format::format_tool_name(name), count))
            .collect();
        let col_width = display_names.iter().map(|(n, _)| n.len()).max().unwrap_or(14).max(14) + 2;
        for (display_name, count) in &display_names {
            lines.push(Line::from(vec![
                Span::styled(format!("     {:<width$}", display_name, width = col_width), dim),
                Span::styled(count.to_string(), dim),
            ]));
        }
    }

    lines.push(Line::from(""));

    // Token section
    push_row(lines, "Tokens (in)", format_number(data.input_tokens));
    push_row(lines, "Tokens (out)", format_number(data.output_tokens));
    push_row(
        lines,
        "Tokens (total)",
        format_number(data.input_tokens + data.output_tokens),
    );
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
                format::format_tool_name(name),
                format::truncate_args(args, 60),
            ),
            Style::default().fg(Color::DarkGray),
        ),
    ]));
}

fn render_chat_skill_load(lines: &mut Vec<Line<'static>>, name: &str) {
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(" ● ", Style::default().fg(Color::Green)),
        Span::styled(
            format!("Skill(/{})", name),
            Style::default().fg(Color::White),
        ),
    ]));
    lines.push(Line::from(Span::styled(
        format!("{}Successfully loaded skill", format::PREFIX_FIRST),
        Style::default().fg(Color::DarkGray),
    )));
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
                format::format_tool_name(&confirm.name),
                format::truncate_args(&confirm.args, 60),
            ),
            Style::default().fg(Color::Yellow),
        ),
        Span::styled("(y/n)", Style::default().fg(Color::DarkGray)),
    ]));
}

fn render_chat_progress(lines: &mut Vec<Line<'static>>, app: &App) {
    let elapsed = app
        .generation.start
        .map(|s| s.elapsed())
        .unwrap_or_default();
    let time_str = format_elapsed(elapsed);
    let token_str = format_token_count(app.generation.tokens);

    let (symbol, symbol_color) = if app.generation.has_received_tokens {
        ("·", Color::DarkGray)
    } else {
        ("✻", Color::DarkGray)
    };

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(format!(" {} ", symbol), Style::default().fg(symbol_color)),
        Span::styled(
            format!("{}… ({}{})", app.generation.verb, time_str, token_str),
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

fn render_edit_result(lines: &mut Vec<Line<'static>>, result: &ToolResultData, file_path: &str) {
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

    // Render diff lines with syntax highlighting
    let mut hl = syntax::highlighter_for_path(file_path);

    for diff_line in &diff_lines {
        let bg = if diff_line.starts_with('-') {
            Some(Color::Indexed(52))
        } else if diff_line.starts_with('+') {
            Some(Color::Indexed(22))
        } else {
            None
        };

        // Strip the leading +/- for highlighting, keep it for display
        let (prefix, code) = if diff_line.starts_with('+') || diff_line.starts_with('-') {
            (&diff_line[..1], &diff_line[1..])
        } else {
            ("", *diff_line)
        };

        if let (Some(bg_color), Some(ref mut h)) = (bg, &mut hl) {
            let mut spans = vec![Span::styled(
                format!("      {}", prefix),
                Style::default().fg(Color::White).bg(bg_color),
            )];
            for mut span in syntax::highlight_line(h, code) {
                span.style = span.style.bg(bg_color);
                spans.push(span);
            }
            lines.push(Line::from(spans));
        } else {
            let style = match bg {
                Some(bg_color) => Style::default().fg(Color::White).bg(bg_color),
                None => Style::default().fg(Color::DarkGray),
            };
            lines.push(Line::from(Span::styled(
                format!("      {}", diff_line),
                style,
            )));
        }
    }
}

fn render_write_result(lines: &mut Vec<Line<'static>>, result: &ToolResultData, file_path: &str) {
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

    let bg = Color::Indexed(22);
    let mut hl = syntax::highlighter_for_path(file_path);
    let max_diff_lines = 30;
    let mut shown = 0;
    let mut remaining = 0;
    for diff_line in output_lines {
        if diff_line.is_empty() {
            continue;
        }
        if shown < max_diff_lines {
            if let Some(ref mut h) = hl {
                // Strip leading + for highlighting
                let code = diff_line.strip_prefix('+').unwrap_or(diff_line);
                let prefix_char = if diff_line.starts_with('+') { "+" } else { " " };
                let mut spans = vec![Span::styled(
                    format!("      {}", prefix_char),
                    Style::default().fg(Color::White).bg(bg),
                )];
                for mut span in syntax::highlight_line(h, code) {
                    span.style = span.style.bg(bg);
                    spans.push(span);
                }
                lines.push(Line::from(spans));
            } else {
                lines.push(Line::from(Span::styled(
                    format!("      {}", diff_line),
                    Style::default().fg(Color::White).bg(bg),
                )));
            }
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
