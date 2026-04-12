/// Capitalize the first character of a string.
pub fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Truncate a string, appending "..." if longer than `max` chars.
pub fn truncate_args(s: &str, max: usize) -> String {
    let mut iter = s.chars();
    let truncated: String = iter.by_ref().take(max).collect();
    if iter.next().is_some() {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

const MAX_OUTPUT_LINES: usize = 30;
pub const PREFIX_FIRST: &str = "   ⎿  ";
pub const PREFIX_REST: &str = "      ";

/// Format tool output lines with ⎿ prefix, capped at 30 lines.
pub fn format_tool_output(output: &str) -> Vec<String> {
    let mut line_count = 0usize;
    let mut result = Vec::with_capacity(MAX_OUTPUT_LINES + 1);
    for (i, line) in output.lines().enumerate() {
        line_count += 1;
        if i < MAX_OUTPUT_LINES {
            let prefix = if i == 0 { PREFIX_FIRST } else { PREFIX_REST };
            result.push(format!("{}{}", prefix, line));
        }
    }
    if line_count > MAX_OUTPUT_LINES {
        result.push(format!(
            "{}... ({} more lines)",
            PREFIX_REST,
            line_count - MAX_OUTPUT_LINES
        ));
    }
    result
}

/// Format a single-line tool result (used for errors or short output).
pub fn format_tool_error(output: &str) -> String {
    format!("{}{}", PREFIX_FIRST, output.trim())
}
