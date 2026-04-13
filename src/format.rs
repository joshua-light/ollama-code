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

/// Format tool call arguments for display, extracting the most relevant field
/// per tool type (e.g. the command for bash, the file path for read/edit).
pub fn format_tool_args_display(name: &str, args: &serde_json::Value) -> String {
    match name {
        "bash" => args
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        "read" => {
            let path = args
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let offset = args.get("offset").and_then(|v| v.as_u64());
            let limit = args.get("limit").and_then(|v| v.as_u64());
            match (offset, limit) {
                (Some(o), Some(l)) => format!("{}, offset={}, limit={}", path, o, l),
                (Some(o), None) => format!("{}, offset={}", path, o),
                (None, Some(l)) => format!("{}, limit={}", path, l),
                (None, None) => path.to_string(),
            }
        }
        "edit" | "write" => args
            .get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string(),
        "subagent" => args
            .get("task")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string(),
        "glob" | "grep" => {
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("?");
            let path = args.get("path").and_then(|v| v.as_str());
            match path {
                Some(p) => format!("{} in {}", pattern, p),
                None => pattern.to_string(),
            }
        }
        _ => args.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── capitalize_first ────────────────────────────────────────────

    #[test]
    fn capitalize_first_normal() {
        assert_eq!(capitalize_first("hello"), "Hello");
    }

    #[test]
    fn capitalize_first_empty() {
        assert_eq!(capitalize_first(""), "");
    }

    #[test]
    fn capitalize_first_already_upper() {
        assert_eq!(capitalize_first("Hello"), "Hello");
    }

    #[test]
    fn capitalize_first_single_char() {
        assert_eq!(capitalize_first("a"), "A");
    }

    #[test]
    fn capitalize_first_unicode() {
        assert_eq!(capitalize_first("über"), "Über");
    }

    // ── truncate_args ───────────────────────────────────────────────

    #[test]
    fn truncate_args_short() {
        assert_eq!(truncate_args("hello", 10), "hello");
    }

    #[test]
    fn truncate_args_exact() {
        assert_eq!(truncate_args("hello", 5), "hello");
    }

    #[test]
    fn truncate_args_over() {
        assert_eq!(truncate_args("hello world", 5), "hello...");
    }

    #[test]
    fn truncate_args_empty() {
        assert_eq!(truncate_args("", 5), "");
    }

    #[test]
    fn truncate_args_multibyte() {
        // Should not panic or split codepoints
        let s = "こんにちは世界";
        let result = truncate_args(s, 3);
        assert_eq!(result, "こんに...");
    }

    // ── format_tool_output ──────────────────────────────────────────

    #[test]
    fn format_tool_output_single_line() {
        let result = format_tool_output("hello");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "   ⎿  hello");
    }

    #[test]
    fn format_tool_output_few_lines() {
        let input = "line1\nline2\nline3";
        let result = format_tool_output(input);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "   ⎿  line1");
        assert_eq!(result[1], "      line2");
        assert_eq!(result[2], "      line3");
    }

    #[test]
    fn format_tool_output_truncated() {
        let lines: Vec<String> = (0..50).map(|i| format!("line {}", i)).collect();
        let input = lines.join("\n");
        let result = format_tool_output(&input);
        assert_eq!(result.len(), 31); // 30 kept + 1 truncation notice
        assert!(result.last().unwrap().contains("20 more lines"));
    }

    #[test]
    fn format_tool_output_empty() {
        let result = format_tool_output("");
        // "".lines() yields nothing in Rust
        assert!(result.is_empty());
    }

    // ── format_tool_error ───────────────────────────────────────────

    #[test]
    fn format_tool_error_trims() {
        assert_eq!(format_tool_error("  oops  "), "   ⎿  oops");
    }

    // ── format_tool_args_display ────────────────────────────────────

    #[test]
    fn display_bash() {
        let args = json!({"command": "ls -la"});
        assert_eq!(format_tool_args_display("bash", &args), "ls -la");
    }

    #[test]
    fn display_bash_missing_command() {
        let args = json!({});
        assert_eq!(format_tool_args_display("bash", &args), "");
    }

    #[test]
    fn display_read_simple() {
        let args = json!({"file_path": "/tmp/foo.rs"});
        assert_eq!(format_tool_args_display("read", &args), "/tmp/foo.rs");
    }

    #[test]
    fn display_read_with_offset_and_limit() {
        let args = json!({"file_path": "/tmp/foo.rs", "offset": 10, "limit": 20});
        assert_eq!(
            format_tool_args_display("read", &args),
            "/tmp/foo.rs, offset=10, limit=20"
        );
    }

    #[test]
    fn display_read_with_offset_only() {
        let args = json!({"file_path": "/tmp/foo.rs", "offset": 10});
        assert_eq!(
            format_tool_args_display("read", &args),
            "/tmp/foo.rs, offset=10"
        );
    }

    #[test]
    fn display_edit() {
        let args = json!({"file_path": "/tmp/foo.rs", "old_string": "a", "new_string": "b"});
        assert_eq!(format_tool_args_display("edit", &args), "/tmp/foo.rs");
    }

    #[test]
    fn display_write() {
        let args = json!({"file_path": "/tmp/new.rs", "content": "fn main() {}"});
        assert_eq!(format_tool_args_display("write", &args), "/tmp/new.rs");
    }

    #[test]
    fn display_subagent() {
        let args = json!({"task": "find all TODOs"});
        assert_eq!(format_tool_args_display("subagent", &args), "find all TODOs");
    }

    #[test]
    fn display_glob_with_path() {
        let args = json!({"pattern": "**/*.rs", "path": "/src"});
        assert_eq!(format_tool_args_display("glob", &args), "**/*.rs in /src");
    }

    #[test]
    fn display_glob_without_path() {
        let args = json!({"pattern": "**/*.rs"});
        assert_eq!(format_tool_args_display("glob", &args), "**/*.rs");
    }

    #[test]
    fn display_grep_with_path() {
        let args = json!({"pattern": "TODO", "path": "/src"});
        assert_eq!(format_tool_args_display("grep", &args), "TODO in /src");
    }

    #[test]
    fn display_unknown_tool() {
        let args = json!({"key": "value"});
        assert_eq!(
            format_tool_args_display("custom_tool", &args),
            args.to_string()
        );
    }
}
