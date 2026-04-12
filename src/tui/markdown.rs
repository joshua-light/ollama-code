use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

pub(super) const MD_INDENT: &str = "   ";

pub(super) fn render_markdown(text: &str, width: u16) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut in_code_block = false;
    let width = width as usize;

    let code_prefix = "   │ ";
    let code_prefix_len = code_prefix.chars().count();

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

pub(super) fn parse_inline_markdown(text: &str) -> Vec<Span<'static>> {
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
