use anyhow::Result;
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::Path;

use super::{expand_tilde, optional_str, required_str, Tool, ToolDefinition};

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

/// Find the closest matching region in `content` for `needle` by looking for
/// lines that contain the first non-empty line of `needle`. Returns
/// `Some((start_line_1indexed, end_line_1indexed, snippet))` or `None`.
fn find_closest_match(content: &str, needle: &str) -> Option<(usize, usize, String)> {
    let needle_lines: Vec<&str> = needle.lines().collect();
    let file_lines: Vec<&str> = content.lines().collect();

    if needle_lines.is_empty() || file_lines.is_empty() {
        return None;
    }

    // Find the first non-empty line from needle to use as anchor
    let anchor = needle_lines
        .iter()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim())?;

    // Find all file lines that contain the anchor (trimmed)
    let mut best_idx: Option<usize> = None;
    let mut best_score: usize = 0;

    for (i, file_line) in file_lines.iter().enumerate() {
        if file_line.trim().contains(anchor) {
            // Score: count how many subsequent needle lines match file lines
            let span = needle_lines.len().min(file_lines.len() - i);
            let mut score = 0;
            for j in 0..span {
                if file_lines[i + j].trim() == needle_lines[j].trim() {
                    score += 1;
                }
            }
            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }
    }

    // If exact anchor wasn't found, try substring match on trimmed first line
    if best_idx.is_none() {
        // Try each word token from the anchor (at least 4 chars) as fallback
        let tokens: Vec<&str> = anchor.split_whitespace().filter(|t| t.len() >= 4).collect();
        for (i, file_line) in file_lines.iter().enumerate() {
            let trimmed = file_line.trim();
            let matching_tokens = tokens.iter().filter(|t| trimmed.contains(**t)).count();
            if matching_tokens > best_score {
                best_score = matching_tokens;
                best_idx = Some(i);
            }
        }
        // Only use token match if we matched at least 2 tokens or 1 long token
        if best_score < 1 {
            best_idx = None;
        }
    }

    let idx = best_idx?;
    let span = needle_lines.len();
    let end = (idx + span).min(file_lines.len());
    let snippet: String = file_lines[idx..end]
        .iter()
        .enumerate()
        .map(|(j, line)| format!("{:>4} | {}", idx + j + 1, line))
        .collect::<Vec<_>>()
        .join("\n");

    Some((idx + 1, end, snippet))
}

/// Write `data` to `path` atomically via a temp file + rename.
/// This prevents data loss if the process crashes mid-write.
fn atomic_write(path: &str, data: &[u8]) -> Result<()> {
    let target = Path::new(path);
    let dir = target
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine parent directory for '{}'", path))?;

    let mut tmp = tempfile::NamedTempFile::new_in(dir)
        .map_err(|e| anyhow::anyhow!("Failed to create temp file in '{}': {}", dir.display(), e))?;
    tmp.write_all(data)
        .map_err(|e| anyhow::anyhow!("Failed to write temp file for '{}': {}", path, e))?;
    tmp.persist(target)
        .map_err(|e| anyhow::anyhow!("Failed to rename temp file to '{}': {}", path, e))?;
    Ok(())
}

pub struct EditTool;

impl Tool for EditTool {
    fn name(&self) -> &str { "edit" }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit".to_string(),
            description: "Edit a file. Two modes: (1) String mode — provide old_string and \
                          new_string to replace an exact unique match. (2) Line-range mode — \
                          provide start_line, end_line, and new_string to replace an inclusive \
                          range of lines. Returns a diff of the change."
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
                        "description": "The exact string to find and replace (must be unique in the file). Required for string mode."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "The first line to replace (1-based, inclusive). Used with end_line for line-range mode."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "The last line to replace (1-based, inclusive). Used with start_line for line-range mode."
                    }
                },
                "required": ["file_path", "new_string"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let raw_path = required_str(arguments, "file_path")?;
        let file_path = expand_tilde(raw_path);
        let file_path = file_path.as_ref();
        let new_string = required_str(arguments, "new_string")?;

        let start_line_arg = arguments.get("start_line").and_then(|v| v.as_u64());
        let end_line_arg = arguments.get("end_line").and_then(|v| v.as_u64());
        let old_string = optional_str(arguments, "old_string");

        // Decide which mode to use
        let use_line_range =
            start_line_arg.is_some() && end_line_arg.is_some() && old_string.is_none();

        if use_line_range {
            // --- Line-range mode ---
            let start = start_line_arg.unwrap() as usize;
            let end = end_line_arg.unwrap() as usize;

            if start == 0 {
                anyhow::bail!("start_line must be >= 1 (1-based)");
            }
            if end < start {
                anyhow::bail!(
                    "end_line ({}) must be >= start_line ({})",
                    end,
                    start
                );
            }

            let content = fs::read_to_string(file_path)
                .map_err(|e| anyhow::anyhow!("Failed to read '{}': {}", file_path, e))?;

            let file_lines: Vec<&str> = content.lines().collect();
            let total_lines = file_lines.len();

            if start > total_lines {
                anyhow::bail!(
                    "start_line {} is beyond end of file ({} lines)",
                    start,
                    total_lines
                );
            }
            if end > total_lines {
                anyhow::bail!(
                    "end_line {} is beyond end of file ({} lines)",
                    end,
                    total_lines
                );
            }

            // Convert to 0-based indices
            let start_idx = start - 1;
            let end_idx = end; // exclusive (end_line is inclusive, so end_idx = end)

            // Build the old lines being replaced
            let old_lines: Vec<&str> = file_lines[start_idx..end_idx].to_vec();

            // Rebuild file content
            let mut new_content = String::new();
            for line in &file_lines[..start_idx] {
                new_content.push_str(line);
                new_content.push('\n');
            }
            new_content.push_str(new_string);
            if !new_string.ends_with('\n') && end_idx < file_lines.len() {
                new_content.push('\n');
            }
            for (i, line) in file_lines[end_idx..].iter().enumerate() {
                new_content.push_str(line);
                if end_idx + i + 1 < file_lines.len() || content.ends_with('\n') {
                    new_content.push('\n');
                }
            }

            atomic_write(file_path, new_content.as_bytes())?;

            // Build contextual diff
            let new_lines: Vec<&str> = new_string.lines().collect();
            let new_file_lines: Vec<&str> = new_content.lines().collect();

            let new_end = start_idx + new_lines.len();

            let ctx = 3;
            let ctx_start = start_idx.saturating_sub(ctx);
            let ctx_after_end = (new_end + ctx).min(new_file_lines.len());
            let max_line_num = ctx_after_end.max(end_idx);
            let num_width = format!("{}", max_line_num).len().max(3);

            let mut diff = String::new();

            // Context before
            for i in ctx_start..start_idx {
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
                    start_idx + j + 1,
                    line,
                    width = num_width
                ));
            }
            // Added lines
            for (j, line) in new_lines.iter().enumerate() {
                diff.push_str(&format!(
                    "+{:>width$}  {}\n",
                    start_idx + j + 1,
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
        } else {
            // --- String mode (existing behavior) ---
            let old_string = old_string.ok_or_else(|| {
                anyhow::anyhow!(
                    "old_string is required when start_line/end_line are not both provided"
                )
            })?;

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
                            let new_content =
                                replace_normalized(&content, old_string, new_string, pos);
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
                            let hint = find_closest_match(&content, old_string);
                            match hint {
                                Some((start, end, snippet)) => {
                                    anyhow::bail!(
                                        "old_string not found in '{}'. Closest match at lines {}-{}:\n{}\n\n\
                                         Make sure old_string matches exactly, including whitespace and indentation.",
                                        file_path, start, end, snippet
                                    );
                                }
                                None => {
                                    anyhow::bail!(
                                        "old_string not found in '{}'. Make sure it matches exactly, \
                                         including whitespace and indentation.",
                                        file_path
                                    );
                                }
                            }
                        }
                    }
                }
            };

            atomic_write(file_path, new_content.as_bytes())?;

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
