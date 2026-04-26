use anyhow::Result;
use serde_json::Value;
use std::fs;

use super::{expand_tilde, required_str, Tool, ToolDefinition};

pub struct ReadTool;

impl Default for ReadTool {
    fn default() -> Self { Self }
}

impl ReadTool {
    pub fn new() -> Self { Self }
}

impl Tool for ReadTool {
    fn name(&self) -> &str { "read" }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read".to_string(),
            description: "Read a file from the filesystem. Returns file contents with line \
                          numbers. Use offset and limit to read specific portions of large files. \
                          The limit parameter is capped at 2000 lines; use grep to search large \
                          files instead of reading them entirely."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-based, default: 1)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default: 200, max: 2000). Use grep to search large files instead of reading them entirely."
                    }
                },
                "required": ["file_path"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let raw_path = required_str(arguments, "file_path")?;
        let file_path = expand_tilde(raw_path);

        let content = fs::read_to_string(file_path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to read '{}': {}", file_path, e))?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let offset = arguments
            .get("offset")
            .and_then(|v| v.as_u64())
            .map(|v| (v as usize).max(1))
            .unwrap_or(1);

        let start = (offset - 1).min(total_lines);

        const DEFAULT_LIMIT: usize = 200;
        const MAX_LIMIT: usize = 2000;

        let raw_limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let limit_capped = raw_limit.is_some_and(|l| l > MAX_LIMIT);
        let limit = raw_limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT);

        let end = (start + limit).min(total_lines);
        let truncated = end < total_lines && raw_limit.is_none();

        let mut result = String::new();

        for (i, line) in lines[start..end].iter().enumerate() {
            let line_num = start + i + 1;
            result.push_str(&format!("{:>4}\t{}\n", line_num, line));
        }

        if result.is_empty() {
            result = "(empty file)".to_string();
        } else if limit_capped && end < total_lines {
            result.push_str(&format!(
                "\n... (limit capped to {} lines, {} more lines not shown. \
                 If you are looking for something specific, use the grep tool instead \
                 of reading large portions of a file.)\n",
                MAX_LIMIT,
                total_lines - end,
            ));
        } else if truncated {
            result.push_str(&format!(
                "\n... ({} more lines not shown. Use offset and limit to read more.)\n",
                total_lines - end
            ));
        }

        Ok(result)
    }
}
