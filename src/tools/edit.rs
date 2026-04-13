use anyhow::Result;
use serde_json::Value;
use std::fs;
use std::io::Write;

use super::{required_str, Tool, ToolDefinition};

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

        let first_pos = match content.find(old_string) {
            None => {
                anyhow::bail!(
                    "old_string not found in '{}'. Make sure it matches exactly, \
                     including whitespace and indentation.",
                    file_path
                );
            }
            Some(pos) => pos,
        };

        // Check for multiple occurrences
        let match_count = content.matches(old_string).count();
        if match_count > 1 {
            anyhow::bail!(
                "old_string found {} times in '{}'. Provide more surrounding \
                 context to make the match unique.",
                match_count,
                file_path
            );
        }

        let new_content = content.replacen(old_string, new_string, 1);

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
