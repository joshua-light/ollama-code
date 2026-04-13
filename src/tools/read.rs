use anyhow::Result;
use serde_json::Value;
use std::fs;

use super::{required_str, Tool, ToolDefinition};

pub struct ReadTool;

impl Tool for ReadTool {
    fn name(&self) -> &str { "read" }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read".to_string(),
            description: "Read a file from the filesystem. Returns file contents with line \
                          numbers. Use offset and limit to read specific portions of large files."
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
                        "description": "Maximum number of lines to read (default: entire file)"
                    }
                },
                "required": ["file_path"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let file_path = required_str(arguments, "file_path")?;

        let content = fs::read_to_string(file_path)
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

        let limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_LIMIT);

        let end = (start + limit).min(total_lines);
        let truncated = end < total_lines && arguments.get("limit").is_none();

        let mut result = String::new();
        for (i, line) in lines[start..end].iter().enumerate() {
            let line_num = start + i + 1;
            result.push_str(&format!("{:>4}\t{}\n", line_num, line));
        }

        if result.is_empty() {
            result = "(empty file)".to_string();
        } else if truncated {
            result.push_str(&format!(
                "\n... ({} more lines not shown. Use offset and limit to read more.)\n",
                total_lines - end
            ));
        }

        Ok(result)
    }
}
