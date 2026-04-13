use anyhow::Result;
use serde_json::Value;
use std::fs;
use std::io::Write;

use super::{required_str, Tool, ToolDefinition};

enum FindResult {
    None,
    One(usize),
    Multiple(usize),
}

/// Exact substring search — current behavior.
fn find_exact(content: &str, needle: &str) -> FindResult {
    let count = content.matches(needle).count();
    match count {
        0 => FindResult::None,
        1 => FindResult::One(content.find(needle).unwrap()),
        n => FindResult::Multiple(n),
    }
}

/// Strip trailing whitespace from each line.
fn normalize_trailing(s: &str) -> String {
    s.lines()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Search for `needle` in `content` after stripping trailing whitespace from
/// every line in both. Returns the byte offset in the *original* content where
/// the matching region starts.
fn find_normalized(content: &str, needle: &str) -> FindResult {
    let norm_content = normalize_trailing(content);
    let norm_needle = normalize_trailing(needle);

    let count = norm_content.matches(&norm_needle).count();
    match count {
        0 => FindResult::None,
        1 => {
            // Map the char offset in the normalized string back to a byte
            // offset in the original content by finding which original line
            // the match starts on.
            let norm_pos = norm_content.find(&norm_needle).unwrap();
            let norm_line_idx = norm_content[..norm_pos].matches('\n').count();
            // The match starts at the beginning of this line in the original.
            let orig_pos = content
                .match_indices('\n')
                .nth(norm_line_idx.wrapping_sub(1))
                .map(|(i, _)| i + 1)
                .unwrap_or(0);
            // If norm_pos doesn't fall on a line boundary, add the column offset.
            let norm_line_start = if norm_line_idx == 0 {
                0
            } else {
                norm_content
                    .match_indices('\n')
                    .nth(norm_line_idx - 1)
                    .map(|(i, _)| i + 1)
                    .unwrap_or(0)
            };
            let col = norm_pos - norm_line_start;
            FindResult::One(orig_pos + col)
        }
        n => FindResult::Multiple(n),
    }
}

/// Replace the region in the original content that matches `needle` (after
/// trailing-whitespace normalization) at position `pos` with `replacement`.
fn replace_normalized(content: &str, needle: &str, replacement: &str, _pos: usize) -> String {
    let norm_needle = normalize_trailing(needle);
    let needle_line_count = norm_needle.matches('\n').count() + 1;

    // Find which original lines correspond to the normalized match.
    let norm_content = normalize_trailing(content);
    let norm_pos = norm_content.find(&norm_needle).unwrap();
    let start_line = norm_content[..norm_pos].matches('\n').count();
    let end_line = start_line + needle_line_count;

    // Rebuild: lines before + replacement + lines after
    let orig_lines: Vec<&str> = content.lines().collect();
    let mut result = String::new();
    for line in &orig_lines[..start_line] {
        result.push_str(line);
        result.push('\n');
    }
    result.push_str(replacement);
    if !replacement.ends_with('\n') && end_line < orig_lines.len() {
        result.push('\n');
    }
    for (i, line) in orig_lines[end_line..].iter().enumerate() {
        result.push_str(line);
        if end_line + i + 1 < orig_lines.len() || content.ends_with('\n') {
            result.push('\n');
        }
    }
    result
}

pub struct EditTool;

impl Tool for EditTool {
    fn name(&self) -> &str { "edit" }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit".to_string(),
            description: "Edit a file by replacing an exact string match with new content. The \
                          old_string must appear exactly once in the file (including whitespace \
                          and indentation). Returns a diff of the change."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace (must be unique in the file)"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string"
                    }
                },
                "required": ["file_path", "old_string", "new_string"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let file_path = required_str(arguments, "file_path")?;
        let old_string = required_str(arguments, "old_string")?;
        let new_string = required_str(arguments, "new_string")?;

        if old_string == new_string {
            anyhow::bail!("old_string and new_string are identical — nothing to change");
        }

        let content = fs::read_to_string(file_path)
            .map_err(|e| anyhow::anyhow!("Failed to read '{}': {}", file_path, e))?;

        // Try exact match first, then fall back to trailing-whitespace-normalized match.
        let (first_pos, new_content) = match find_exact(&content, old_string) {
            FindResult::One(pos) => {
                let new_content = content.replacen(old_string, new_string, 1);
                (pos, new_content)
            }
            FindResult::Multiple(count) => {
                anyhow::bail!(
                    "old_string found {} times in '{}'. Provide more surrounding \
                     context to make the match unique.",
                    count,
                    file_path
                );
            }
            FindResult::None => {
                // Fallback: try matching with trailing whitespace stripped from each line
                match find_normalized(&content, old_string) {
                    FindResult::One(pos) => {
                        let new_content = replace_normalized(&content, old_string, new_string, pos);
                        (pos, new_content)
                    }
                    FindResult::Multiple(count) => {
                        anyhow::bail!(
                            "old_string found {} times in '{}' (after normalizing \
                             trailing whitespace). Provide more surrounding context \
                             to make the match unique.",
                            count,
                            file_path
                        );
                    }
                    FindResult::None => {
                        anyhow::bail!(
                            "old_string not found in '{}'. Make sure it matches exactly, \
                             including whitespace and indentation.",
                            file_path
                        );
                    }
                }
            }
        };

        let mut file = fs::File::create(file_path)
            .map_err(|e| anyhow::anyhow!("Failed to write '{}': {}", file_path, e))?;
        file.write_all(new_content.as_bytes())
            .map_err(|e| anyhow::anyhow!("Failed to write '{}': {}", file_path, e))?;

        // Build contextual diff with line numbers
        let file_lines: Vec<&str> = content.lines().collect();
        let old_lines: Vec<&str> = old_string.lines().collect();
        let new_lines: Vec<&str> = new_string.lines().collect();
        let new_file_lines: Vec<&str> = new_content.lines().collect();

        let start_line = content[..first_pos].matches('\n').count();
        let old_end = start_line + old_lines.len();
        let new_end = start_line + new_lines.len();

        let ctx = 3;
        let ctx_start = start_line.saturating_sub(ctx);
        let ctx_after_end = (new_end + ctx).min(new_file_lines.len());
        let max_line_num = ctx_after_end.max(old_end);
        let num_width = format!("{}", max_line_num).len().max(3);

        let mut diff = String::new();

        // Context before
        for i in ctx_start..start_line {
            diff.push_str(&format!(
                " {:>width$}  {}\n",
                i + 1,
                file_lines[i],
                width = num_width
            ));
        }
        // Removed lines
        for (j, line) in old_lines.iter().enumerate() {
            diff.push_str(&format!(
                "-{:>width$}  {}\n",
                start_line + j + 1,
                line,
                width = num_width
            ));
        }
        // Added lines
        for (j, line) in new_lines.iter().enumerate() {
            diff.push_str(&format!(
                "+{:>width$}  {}\n",
                start_line + j + 1,
                line,
                width = num_width
            ));
        }
        // Context after (from new file)
        for i in new_end..ctx_after_end {
            diff.push_str(&format!(
                " {:>width$}  {}\n",
                i + 1,
                new_file_lines[i],
                width = num_width
            ));
        }

        Ok(diff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let content = "fn foo() {\n    let x = 1;\n}\n";
        match find_exact(content, "    let x = 1;") {
            FindResult::One(_) => {}
            _ => panic!("expected one match"),
        }
    }

    #[test]
    fn test_normalized_trailing_whitespace() {
        // File has empty line (just \n), model sends line with a space
        let content = "line1\n\nline3\n";
        let needle = "line1\n \nline3";
        assert!(matches!(find_exact(content, needle), FindResult::None));
        assert!(matches!(find_normalized(content, needle), FindResult::One(_)));
    }

    #[test]
    fn test_replace_normalized_empty_line() {
        let content = "before\n/// doc comment\n\n    let x = 1;\nafter\n";
        let needle = "/// doc comment\n \n    let x = 1;";
        let replacement = "/// doc comment\nfn foo() {\n    let x = 1;";

        let result = replace_normalized(content, needle, replacement, 0);
        assert_eq!(result, "before\n/// doc comment\nfn foo() {\n    let x = 1;\nafter\n");
    }

    #[test]
    fn test_normalize_trailing() {
        assert_eq!(normalize_trailing("a \nb\n  c  "), "a\nb\n  c");
    }
}
