use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

pub(super) const MD_INDENT: &str = "   ";

// ── Helpers ──────────────────────────────────────────────────────────────

/// Check if a line is a markdown table separator (e.g. `|---|---|---|`).
fn is_table_separator(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.contains('|')
        && trimmed.contains('-')
        && trimmed
            .chars()
            .all(|c| c == '|' || c == '-' || c == ':' || c == ' ')
}

/// Check if a line is a horizontal rule (`---`, `***`, `___`, or with spaces).
fn is_horizontal_rule(line: &str) -> bool {
    let chars: Vec<char> = line.chars().filter(|c| !c.is_whitespace()).collect();
    chars.len() >= 3
        && matches!(chars[0], '-' | '*' | '_')
        && chars.iter().all(|&c| c == chars[0])
}

/// Parse a markdown table row into cells.
fn parse_table_row(line: &str) -> Vec<String> {
    let trimmed = line.trim();
    let trimmed = trimmed.strip_prefix('|').unwrap_or(trimmed);
    let trimmed = trimmed.strip_suffix('|').unwrap_or(trimmed);
    trimmed
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect()
}

/// Truncate a string to fit within a given character width.
fn truncate_to_width(s: &str, max_width: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_width {
        s.to_string()
    } else if max_width > 1 {
        let truncated: String = s.chars().take(max_width - 1).collect();
        format!("{}…", truncated)
    } else {
        s.chars().take(max_width).collect()
    }
}

// ── Table renderer ───────────────────────────────────────────────────────

/// Render a markdown table with box-drawing borders.
fn render_table(table_lines: &[&str], output: &mut Vec<Line<'static>>, max_width: usize) {
    if table_lines.len() < 2 {
        return;
    }

    let header = parse_table_row(table_lines[0]);
    // table_lines[1] is the separator — skip it
    let body_rows: Vec<Vec<String>> = table_lines[2..]
        .iter()
        .map(|line| parse_table_row(line))
        .collect();

    let num_cols = header.len();
    if num_cols == 0 {
        return;
    }

    // Calculate column widths from content
    let mut col_widths: Vec<usize> = header.iter().map(|c| c.chars().count()).collect();
    for row in &body_rows {
        for (j, cell) in row.iter().enumerate() {
            if j < col_widths.len() {
                col_widths[j] = col_widths[j].max(cell.chars().count());
            }
        }
    }
    for w in &mut col_widths {
        *w = (*w).max(1);
    }

    // Cap total width: indent(3) + borders(num_cols+1) + padding(num_cols*2)
    let border_chars = num_cols + 1;
    let padding_chars = num_cols * 2;
    let available_content = max_width.saturating_sub(3 + border_chars + padding_chars);
    let total_content: usize = col_widths.iter().sum();
    if total_content > available_content && available_content > 0 {
        let scale = available_content as f64 / total_content as f64;
        for w in &mut col_widths {
            *w = ((*w as f64 * scale) as usize).max(1);
        }
    }

    let border_style = Style::default().fg(Color::DarkGray);
    let header_style = Style::default()
        .fg(Color::White)
        .add_modifier(Modifier::BOLD);
    let cell_style = Style::default().fg(Color::White);

    // ┌──┬──┐
    let mut top = String::from("   ┌");
    for (j, &w) in col_widths.iter().enumerate() {
        top.push_str(&"─".repeat(w + 2));
        if j < num_cols - 1 {
            top.push('┬');
        }
    }
    top.push('┐');
    output.push(Line::from(Span::styled(top, border_style)));

    // Header row
    let mut spans = vec![Span::styled("   │", border_style)];
    for (j, cell) in header.iter().enumerate() {
        let w = col_widths.get(j).copied().unwrap_or(1);
        let display = truncate_to_width(cell, w);
        let pad = w.saturating_sub(display.chars().count());
        spans.push(Span::styled(
            format!(" {}{} ", display, " ".repeat(pad)),
            header_style,
        ));
        spans.push(Span::styled("│", border_style));
    }
    output.push(Line::from(spans));

    // ├──┼──┤
    let mut sep = String::from("   ├");
    for (j, &w) in col_widths.iter().enumerate() {
        sep.push_str(&"─".repeat(w + 2));
        if j < num_cols - 1 {
            sep.push('┼');
        }
    }
    sep.push('┤');
    output.push(Line::from(Span::styled(sep, border_style)));

    // Body rows
    for row in &body_rows {
        let mut spans = vec![Span::styled("   │", border_style)];
        for (j, w) in col_widths.iter().enumerate() {
            let cell = row.get(j).map(|s| s.as_str()).unwrap_or("");
            let display = truncate_to_width(cell, *w);
            let pad = w.saturating_sub(display.chars().count());
            spans.push(Span::styled(
                format!(" {}{} ", display, " ".repeat(pad)),
                cell_style,
            ));
            spans.push(Span::styled("│", border_style));
        }
        output.push(Line::from(spans));
    }

    // └──┴──┘
    let mut bot = String::from("   └");
    for (j, &w) in col_widths.iter().enumerate() {
        bot.push_str(&"─".repeat(w + 2));
        if j < num_cols - 1 {
            bot.push('┴');
        }
    }
    bot.push('┘');
    output.push(Line::from(Span::styled(bot, border_style)));
}

// ── Markdown renderer ────────────────────────────────────────────────────

pub(super) fn render_markdown(text: &str, width: u16) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut in_code_block = false;
    let width = width as usize;

    let code_prefix = "   │ ";
    let code_prefix_len = code_prefix.chars().count();

    let raw_lines: Vec<&str> = text.lines().collect();
    let mut i = 0;

    while i < raw_lines.len() {
        let raw_line = raw_lines[i];
        let trimmed = raw_line.trim();

        // Code block fences
        if let Some(fence_rest) = trimmed.strip_prefix("```") {
            if !in_code_block {
                // Opening fence — extract language
                let lang = fence_rest.trim().to_string();
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
            i += 1;
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
            i += 1;
            continue;
        }

        // Table detection: line has |, next line is a separator row
        if trimmed.contains('|')
            && i + 1 < raw_lines.len()
            && is_table_separator(raw_lines[i + 1])
        {
            let start = i;
            i += 2; // skip header + separator
            while i < raw_lines.len() {
                let next = raw_lines[i].trim();
                if next.is_empty() || !next.contains('|') || next.starts_with("```") {
                    break;
                }
                i += 1;
            }
            render_table(&raw_lines[start..i], &mut lines, width);
            continue;
        }

        // Empty line
        if trimmed.is_empty() {
            lines.push(Line::from(""));
            i += 1;
            continue;
        }

        // Horizontal rules: ---, ***, ___ (3+ chars, optionally with spaces)
        if is_horizontal_rule(trimmed) {
            let rule_len = width.saturating_sub(6);
            lines.push(Line::from(Span::styled(
                format!("   {}", "─".repeat(rule_len)),
                Style::default().fg(Color::DarkGray),
            )));
            i += 1;
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
                    i += 1;
                    continue;
                }
            }
        }

        // Blockquotes: > text, >> nested, etc.
        if trimmed.starts_with('>') {
            let mut s = trimmed;
            let mut depth = 0usize;
            while let Some(rest) = s.strip_prefix('>') {
                depth += 1;
                s = rest.strip_prefix(' ').unwrap_or(rest);
            }
            let content = s;
            let bar_style = Style::default().fg(Color::Blue);
            let mut spans = vec![
                Span::raw(MD_INDENT.to_string()),
                Span::styled("▌".repeat(depth) + " ", bar_style),
            ];
            if !content.is_empty() {
                for mut span in parse_inline_markdown(content) {
                    span.style = span.style.patch(Style::default().add_modifier(Modifier::ITALIC));
                    spans.push(span);
                }
            }
            lines.push(Line::from(spans));
            i += 1;
            continue;
        }

        // Task lists: - [ ], - [x], * [ ], * [x]
        {
            let indent = raw_line.chars().take_while(|c| *c == ' ').count();
            let after_indent = raw_line.trim_start();
            if (after_indent.starts_with("- [") || after_indent.starts_with("* ["))
                && after_indent.len() >= 6
            {
                let check_char = after_indent.as_bytes().get(3).copied();
                let close_bracket = after_indent.as_bytes().get(4).copied();
                if close_bracket == Some(b']')
                    && matches!(check_char, Some(b' ') | Some(b'x') | Some(b'X'))
                {
                    let checked = check_char != Some(b' ');
                    let rest = after_indent[5..].trim_start();
                    let nest_pad = " ".repeat((indent / 2).min(4) * 2);
                    let (marker, color) = if checked {
                        ("\u{2611} ", Color::Green) // ☑
                    } else {
                        ("\u{2610} ", Color::DarkGray) // ☐
                    };
                    let mut spans = vec![
                        Span::raw(format!("{}{}", MD_INDENT, nest_pad)),
                        Span::styled(marker.to_string(), Style::default().fg(color)),
                    ];
                    if checked {
                        for mut span in parse_inline_markdown(rest) {
                            span.style = span.style.patch(
                                Style::default()
                                    .fg(Color::DarkGray)
                                    .add_modifier(Modifier::CROSSED_OUT),
                            );
                            spans.push(span);
                        }
                    } else {
                        spans.extend(parse_inline_markdown(rest));
                    }
                    lines.push(Line::from(spans));
                    i += 1;
                    continue;
                }
            }
        }

        // Unordered list: * or - (with nesting)
        {
            let indent = raw_line.chars().take_while(|c| *c == ' ').count();
            let after_indent = raw_line.trim_start();
            if (after_indent.starts_with("* ") || after_indent.starts_with("- "))
                && after_indent.len() > 2
            {
                let rest = after_indent[2..].trim_start();
                let nest = indent / 2;
                let pad = " ".repeat(3 + nest.min(4) * 2);
                let bullet = match nest {
                    0 => "• ",
                    1 => "◦ ",
                    _ => "▪ ",
                };
                let mut spans = vec![Span::raw(pad), Span::raw(bullet.to_string())];
                spans.extend(parse_inline_markdown(rest));
                lines.push(Line::from(spans));
                i += 1;
                continue;
            }
        }

        // Numbered list: 1. 2. etc. (with nesting)
        {
            let indent = raw_line.chars().take_while(|c| *c == ' ').count();
            let after_indent = raw_line.trim_start();
            if let Some(dot_pos) = after_indent.find(". ") {
                if dot_pos > 0
                    && dot_pos <= 3
                    && after_indent[..dot_pos].chars().all(|c| c.is_ascii_digit())
                {
                    let num = &after_indent[..dot_pos];
                    let rest = after_indent[dot_pos + 2..].trim_start();
                    let nest = indent / 2;
                    let pad = " ".repeat(3 + nest.min(4) * 2);
                    let mut spans = vec![Span::raw(format!("{}{}. ", pad, num))];
                    spans.extend(parse_inline_markdown(rest));
                    lines.push(Line::from(spans));
                    i += 1;
                    continue;
                }
            }
        }

        // Regular text with inline markdown
        let mut spans = vec![Span::raw(MD_INDENT.to_string())];
        spans.extend(parse_inline_markdown(trimmed));
        lines.push(Line::from(spans));
        i += 1;
    }

    lines
}

// ── Inline markdown parser ───────────────────────────────────────────────

pub(super) fn parse_inline_markdown(text: &str) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Escaped characters: \_ \* \` \\ \~ \[ \!
        if chars[i] == '\\'
            && i + 1 < len
            && matches!(chars[i + 1], '_' | '*' | '`' | '\\' | '~' | '[' | ']' | '!')
        {
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
                    Style::default().fg(Color::Indexed(75)),
                ));
            }
            continue;
        }

        // Image: ![alt](url)
        if chars[i] == '!' && i + 1 < len && chars[i + 1] == '[' {
            if let Some((alt, _url, end)) = parse_link_at(&chars, i + 1) {
                if !current.is_empty() {
                    spans.push(Span::raw(std::mem::take(&mut current)));
                }
                let label = if alt.is_empty() {
                    "image".to_string()
                } else {
                    format!("image: {}", alt)
                };
                spans.push(Span::styled(
                    format!("[{}]", label),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::ITALIC),
                ));
                i = end;
                continue;
            }
        }

        // Link: [text](url)
        if chars[i] == '[' {
            if let Some((text, url, end)) = parse_link_at(&chars, i) {
                if !current.is_empty() {
                    spans.push(Span::raw(std::mem::take(&mut current)));
                }
                spans.push(Span::styled(
                    text,
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::UNDERLINED),
                ));
                if !url.is_empty() {
                    spans.push(Span::styled(
                        format!(" ({})", url),
                        Style::default().fg(Color::DarkGray),
                    ));
                }
                i = end;
                continue;
            }
        }

        // ~~ — strikethrough
        if chars[i] == '~' && i + 1 < len && chars[i + 1] == '~' {
            // Look ahead for closing ~~
            let mut j = i + 2;
            while j + 1 < len {
                if chars[j] == '~' && chars[j + 1] == '~' {
                    break;
                }
                j += 1;
            }
            if j + 1 < len {
                if !current.is_empty() {
                    spans.push(Span::raw(std::mem::take(&mut current)));
                }
                let inner: String = chars[i + 2..j].iter().collect();
                i = j + 2;
                let patch = Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::CROSSED_OUT);
                for mut span in parse_inline_markdown(&inner) {
                    span.style = span.style.patch(patch);
                    spans.push(span);
                }
                continue;
            }
        }

        // *** — bold + italic
        if chars[i] == '*'
            && i + 2 < len
            && chars[i + 1] == '*'
            && chars[i + 2] == '*'
        {
            // Look ahead for closing ***
            let mut j = i + 3;
            while j + 2 < len {
                if chars[j] == '*' && chars[j + 1] == '*' && chars[j + 2] == '*' {
                    break;
                }
                j += 1;
            }
            if j + 2 < len {
                if !current.is_empty() {
                    spans.push(Span::raw(std::mem::take(&mut current)));
                }
                let inner: String = chars[i + 3..j].iter().collect();
                i = j + 3;
                let patch = Style::default().add_modifier(Modifier::BOLD | Modifier::ITALIC);
                for mut span in parse_inline_markdown(&inner) {
                    span.style = span.style.patch(patch);
                    spans.push(span);
                }
                continue;
            }
        }

        // ** — bold
        if chars[i] == '*' && i + 1 < len && chars[i + 1] == '*' {
            // Look ahead for closing **
            let mut j = i + 2;
            while j + 1 < len {
                if chars[j] == '*' && chars[j + 1] == '*' {
                    break;
                }
                j += 1;
            }
            if j + 1 < len {
                if !current.is_empty() {
                    spans.push(Span::raw(std::mem::take(&mut current)));
                }
                let inner: String = chars[i + 2..j].iter().collect();
                i = j + 2;
                let patch = Style::default().add_modifier(Modifier::BOLD);
                for mut span in parse_inline_markdown(&inner) {
                    span.style = span.style.patch(patch);
                    spans.push(span);
                }
                continue;
            }
        }

        // * — italic (only if followed by non-space and a closing * exists)
        if chars[i] == '*' && i + 1 < len && !chars[i + 1].is_whitespace() {
            let mut j = i + 1;
            while j < len && chars[j] != '*' {
                j += 1;
            }
            if j < len && j > i + 1 {
                if !current.is_empty() {
                    spans.push(Span::raw(std::mem::take(&mut current)));
                }
                let inner: String = chars[i + 1..j].iter().collect();
                i = j + 1;
                let patch = Style::default().add_modifier(Modifier::ITALIC);
                for mut span in parse_inline_markdown(&inner) {
                    span.style = span.style.patch(patch);
                    spans.push(span);
                }
                continue;
            }
        }

        // _italic_ (only at word boundaries to avoid matching variable_names)
        if chars[i] == '_' {
            let at_start = i == 0 || !chars[i - 1].is_alphanumeric();
            if at_start && i + 1 < len && !chars[i + 1].is_whitespace() {
                let mut j = i + 1;
                while j < len && chars[j] != '_' {
                    j += 1;
                }
                if j < len && j > i + 1 {
                    let at_end = j + 1 >= len || !chars[j + 1].is_alphanumeric();
                    if at_end {
                        if !current.is_empty() {
                            spans.push(Span::raw(std::mem::take(&mut current)));
                        }
                        let inner: String = chars[i + 1..j].iter().collect();
                        i = j + 1;
                        let patch = Style::default().add_modifier(Modifier::ITALIC);
                        for mut span in parse_inline_markdown(&inner) {
                            span.style = span.style.patch(patch);
                            spans.push(span);
                        }
                        continue;
                    }
                }
            }
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

/// Try to parse a markdown link starting at `chars[start]` which must be `[`.
/// Returns `(text, url, end_index)` where end_index is the position after `)`.
fn parse_link_at(chars: &[char], start: usize) -> Option<(String, String, usize)> {
    if chars.get(start).copied() != Some('[') {
        return None;
    }
    // Find closing ]
    let mut j = start + 1;
    let len = chars.len();
    while j < len && chars[j] != ']' {
        j += 1;
    }
    if j >= len || j + 1 >= len || chars[j + 1] != '(' {
        return None;
    }
    let text: String = chars[start + 1..j].iter().collect();
    j += 2; // skip ](
    let url_start = j;
    // Handle nested parens in URLs (rare but possible)
    let mut paren_depth = 1;
    while j < len && paren_depth > 0 {
        match chars[j] {
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            _ => {}
        }
        if paren_depth > 0 {
            j += 1;
        }
    }
    if paren_depth != 0 {
        return None;
    }
    let url: String = chars[url_start..j].iter().collect();
    Some((text, url, j + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: extract (content, fg_color, bold, italic) from spans.
    fn span_info<'a>(spans: &'a [Span<'a>]) -> Vec<(&'a str, Option<Color>, bool, bool)> {
        spans
            .iter()
            .map(|s| {
                let bold = s.style.add_modifier.contains(Modifier::BOLD);
                let italic = s.style.add_modifier.contains(Modifier::ITALIC);
                (s.content.as_ref(), s.style.fg, bold, italic)
            })
            .collect()
    }

    /// Helper: collect all span text from rendered lines.
    fn all_text(lines: &[Line]) -> String {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Helper: check if any span in lines has a specific modifier.
    fn has_modifier_with_text(lines: &[Line], modifier: Modifier, needle: &str) -> bool {
        lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .any(|s| s.style.add_modifier.contains(modifier) && s.content.contains(needle))
    }

    /// Helper: check if any span has a specific fg color and contains text.
    fn has_fg_with_text(lines: &[Line], color: Color, needle: &str) -> bool {
        lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .any(|s| s.style.fg == Some(color) && s.content.contains(needle))
    }

    // ── Italic ───────────────────────────────────────────────────────────

    #[test]
    fn italic_single_asterisk() {
        let spans = parse_inline_markdown("this is *italic* text");
        let info = span_info(&spans);
        assert_eq!(info[0], ("this is ", None, false, false));
        assert_eq!(info[1], ("italic", None, false, true));
        assert_eq!(info[2], (" text", None, false, false));
    }

    #[test]
    fn italic_underscore() {
        let spans = parse_inline_markdown("this is _italic_ text");
        let info = span_info(&spans);
        assert_eq!(info[0], ("this is ", None, false, false));
        assert_eq!(info[1], ("italic", None, false, true));
        assert_eq!(info[2], (" text", None, false, false));
    }

    #[test]
    fn underscore_in_variable_name_not_italic() {
        let spans = parse_inline_markdown("some_variable_name here");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content.as_ref(), "some_variable_name here");
    }

    #[test]
    fn bold_italic_triple_asterisk() {
        let spans = parse_inline_markdown("this is ***bold italic*** text");
        let info = span_info(&spans);
        assert_eq!(info[0], ("this is ", None, false, false));
        assert_eq!(info[1], ("bold italic", None, true, true));
        assert_eq!(info[2], (" text", None, false, false));
    }

    #[test]
    fn bold_still_works() {
        let spans = parse_inline_markdown("this is **bold** text");
        let info = span_info(&spans);
        assert_eq!(info[0], ("this is ", None, false, false));
        assert_eq!(info[1], ("bold", None, true, false));
        assert_eq!(info[2], (" text", None, false, false));
    }

    #[test]
    fn italic_asterisk_not_triggered_by_space() {
        let spans = parse_inline_markdown("5 * 3 = 15");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content.as_ref(), "5 * 3 = 15");
    }

    #[test]
    fn mixed_inline_formatting() {
        let spans = parse_inline_markdown("**bold** and *italic* and `code`");
        let info = span_info(&spans);
        assert_eq!(info[0], ("bold", None, true, false));
        assert_eq!(info[1], (" and ", None, false, false));
        assert_eq!(info[2], ("italic", None, false, true));
        assert_eq!(info[3], (" and ", None, false, false));
        assert_eq!(info[4].0, "code");
    }

    // ── Inline code ──────────────────────────────────────────────────────

    #[test]
    fn inline_code_has_light_blue_fg() {
        let spans = parse_inline_markdown("use `foo()` here");
        assert_eq!(spans[0].content.as_ref(), "use ");
        assert_eq!(spans[1].content.as_ref(), "foo()");
        assert_eq!(spans[1].style.fg, Some(Color::Indexed(75)));
        assert_eq!(spans[1].style.bg, None);
        assert_eq!(spans[2].content.as_ref(), " here");
    }

    // ── Tables ───────────────────────────────────────────────────────────

    #[test]
    fn table_renders_with_borders() {
        let md = "\
| Name  | Age |
|-------|-----|
| Alice | 30  |
| Bob   | 25  |";
        let lines = render_markdown(md, 80);
        let text = all_text(&lines);
        assert!(text.contains('┌'));
        assert!(text.contains('┘'));
        assert!(text.contains("Alice"));
        assert!(text.contains("Name"));
    }

    #[test]
    fn table_separator_detection() {
        assert!(is_table_separator("|---|---|"));
        assert!(is_table_separator("| --- | --- |"));
        assert!(is_table_separator("|:---:|---:|"));
        assert!(!is_table_separator("| hello | world |"));
        assert!(!is_table_separator("not a table"));
    }

    // ── Horizontal rules ─────────────────────────────────────────────────

    #[test]
    fn horizontal_rule_dashes() {
        let lines = render_markdown("above\n---\nbelow", 40);
        let text = all_text(&lines);
        assert!(text.contains('─'));
        assert!(text.contains("above"));
        assert!(text.contains("below"));
    }

    #[test]
    fn horizontal_rule_asterisks() {
        assert!(is_horizontal_rule("***"));
        assert!(is_horizontal_rule("* * *"));
        assert!(is_horizontal_rule("___"));
        assert!(is_horizontal_rule("----"));
        assert!(!is_horizontal_rule("--")); // too short
        assert!(!is_horizontal_rule("--x")); // mixed chars
    }

    // ── Blockquotes ──────────────────────────────────────────────────────

    #[test]
    fn blockquote_single_level() {
        let lines = render_markdown("> quoted text", 60);
        let text = all_text(&lines);
        assert!(text.contains('▌'));
        assert!(text.contains("quoted text"));
        assert!(has_modifier_with_text(&lines, Modifier::ITALIC, "quoted text"));
    }

    #[test]
    fn blockquote_nested() {
        let lines = render_markdown(">> deeply quoted", 60);
        let text = all_text(&lines);
        assert!(text.contains("▌▌"));
        assert!(text.contains("deeply quoted"));
    }

    #[test]
    fn blockquote_with_inline() {
        let lines = render_markdown("> this is **important**", 60);
        // "important" should be both bold (from **) and italic (from blockquote)
        assert!(has_modifier_with_text(
            &lines,
            Modifier::BOLD | Modifier::ITALIC,
            "important"
        ));
    }

    // ── Links ────────────────────────────────────────────────────────────

    #[test]
    fn link_renders_text_and_url() {
        let spans = parse_inline_markdown("see [docs](https://example.com) here");
        assert_eq!(spans[0].content.as_ref(), "see ");
        assert_eq!(spans[1].content.as_ref(), "docs");
        assert_eq!(spans[1].style.fg, Some(Color::Blue));
        assert!(spans[1].style.add_modifier.contains(Modifier::UNDERLINED));
        assert!(spans[2].content.contains("https://example.com"));
        assert_eq!(spans[2].style.fg, Some(Color::DarkGray));
        assert_eq!(spans[3].content.as_ref(), " here");
    }

    #[test]
    fn link_no_false_positive_on_plain_brackets() {
        let spans = parse_inline_markdown("array[0] is fine");
        // Should not be parsed as a link since there's no ](url)
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content.as_ref(), "array[0] is fine");
    }

    // ── Images ───────────────────────────────────────────────────────────

    #[test]
    fn image_renders_label() {
        let spans = parse_inline_markdown("see ![screenshot](img.png) above");
        assert_eq!(spans[0].content.as_ref(), "see ");
        assert_eq!(spans[1].content.as_ref(), "[image: screenshot]");
        assert_eq!(spans[1].style.fg, Some(Color::DarkGray));
        assert!(spans[1].style.add_modifier.contains(Modifier::ITALIC));
        assert_eq!(spans[2].content.as_ref(), " above");
    }

    #[test]
    fn image_empty_alt() {
        let spans = parse_inline_markdown("![](img.png)");
        assert_eq!(spans[0].content.as_ref(), "[image]");
    }

    // ── Strikethrough ────────────────────────────────────────────────────

    #[test]
    fn strikethrough_renders() {
        let spans = parse_inline_markdown("this is ~~deleted~~ text");
        assert_eq!(spans[0].content.as_ref(), "this is ");
        assert_eq!(spans[1].content.as_ref(), "deleted");
        assert!(spans[1].style.add_modifier.contains(Modifier::CROSSED_OUT));
        assert_eq!(spans[1].style.fg, Some(Color::DarkGray));
        assert_eq!(spans[2].content.as_ref(), " text");
    }

    #[test]
    fn strikethrough_no_false_positive() {
        let spans = parse_inline_markdown("use ~home for tilde");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content.as_ref(), "use ~home for tilde");
    }

    // ── Task lists ───────────────────────────────────────────────────────

    #[test]
    fn task_list_unchecked() {
        let lines = render_markdown("- [ ] todo item", 60);
        let text = all_text(&lines);
        assert!(text.contains('\u{2610}')); // ☐
        assert!(text.contains("todo item"));
    }

    #[test]
    fn task_list_checked() {
        let lines = render_markdown("- [x] done item", 60);
        let text = all_text(&lines);
        assert!(text.contains('\u{2611}')); // ☑
        assert!(text.contains("done item"));
        assert!(has_modifier_with_text(
            &lines,
            Modifier::CROSSED_OUT,
            "done item"
        ));
    }

    // ── Nested lists ─────────────────────────────────────────────────────

    #[test]
    fn nested_unordered_list() {
        let md = "\
- top
  - nested
    - deep";
        let lines = render_markdown(md, 60);
        let text = all_text(&lines);
        assert!(text.contains('•')); // top level
        assert!(text.contains('◦')); // nested
        assert!(text.contains('▪')); // deep
    }

    #[test]
    fn nested_numbered_list() {
        let md = "\
1. first
  2. sub
    3. deep";
        let lines = render_markdown(md, 60);
        let text = all_text(&lines);
        assert!(text.contains("1."));
        assert!(text.contains("2."));
        assert!(text.contains("3."));
    }

    // ── Escape sequences ─────────────────────────────────────────────────

    #[test]
    fn escape_tilde_and_bracket() {
        let spans = parse_inline_markdown("\\~not strike\\~ and \\[not link\\]");
        assert_eq!(spans.len(), 1);
        assert_eq!(
            spans[0].content.as_ref(),
            "~not strike~ and [not link]"
        );
    }

    // ── Full visual dump ─────────────────────────────────────────────────

    #[test]
    fn full_render_visual_dump() {
        let md = "\
# Heading

This has *italic text* and **bold text** and ***bold italic***.

Here is `inline code` in a sentence.

| Feature       | Status | Notes     |
|---------------|--------|-----------|
| Italics       | Done   | *works*   |
| Code          | Done   | highlight |
| Tables        | Done   | neat      |

---

> This is a blockquote with **bold** inside.
>> Nested quote here.

- [x] Completed task
- [ ] Pending task

- top level
  - nested item
    - deep item

1. First
  2. Sub-item

See [the docs](https://example.com) for details.

This has ~~old text~~ in it.

![screenshot](image.png)

And _underscore italic_ too.";

        let lines = render_markdown(md, 70);

        // Dump each line with style annotations for visual inspection
        let mut output = String::new();
        for line in &lines {
            for span in &line.spans {
                let mut tags = Vec::new();
                if span.style.add_modifier.contains(Modifier::BOLD) {
                    tags.push("B");
                }
                if span.style.add_modifier.contains(Modifier::ITALIC) {
                    tags.push("I");
                }
                if span.style.add_modifier.contains(Modifier::UNDERLINED) {
                    tags.push("U");
                }
                if span.style.add_modifier.contains(Modifier::CROSSED_OUT) {
                    tags.push("S");
                }
                if span.style.fg == Some(Color::Indexed(75)) {
                    tags.push("code");
                }
                if span.style.fg == Some(Color::Blue) {
                    tags.push("blue");
                }
                if span.style.fg == Some(Color::Green) {
                    tags.push("green");
                }
                if span.style.fg == Some(Color::DarkGray) {
                    tags.push("dim");
                }
                if tags.is_empty() {
                    output.push_str(&span.content);
                } else {
                    output.push_str(&format!("[{}:{}]", tags.join("+"), span.content));
                }
            }
            output.push('\n');
        }

        // Print for visual review (visible with `cargo test -- --nocapture`)
        println!("\n--- RENDERED MARKDOWN ---\n{}\n--- END ---", output);

        let text = all_text(&lines);

        // All features present
        assert!(text.contains("Heading"));
        assert!(text.contains('─')); // horizontal rule
        assert!(text.contains('▌')); // blockquote
        assert!(text.contains('\u{2611}')); // ☑ checked task
        assert!(text.contains('\u{2610}')); // ☐ unchecked task
        assert!(text.contains('•')); // bullet
        assert!(text.contains('◦')); // nested bullet
        assert!(text.contains('▪')); // deep bullet
        assert!(text.contains("the docs")); // link text
        assert!(text.contains("example.com")); // link url
        assert!(text.contains("[image: screenshot]")); // image
        assert!(has_modifier_with_text(&lines, Modifier::CROSSED_OUT, "old text"));
        assert!(has_fg_with_text(&lines, Color::Blue, "the docs"));
    }
}
