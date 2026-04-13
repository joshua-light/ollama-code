use anyhow::Result;
use serde_json::Value;
use std::fs;
use std::io::Write;

use super::{required_str, Tool, ToolDefinition};

pub struct WriteTool;

impl Tool for WriteTool {
    fn name(&self) -> &str { "write" }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write".to_string(),
            description: "Write content to a new file. Use this to create new files. The file \
                          must not already exist — use the edit tool to modify existing files. \
                          Parent directories are created automatically if needed."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the new file to create"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let file_path = required_str(arguments, "file_path")?;
        let content = required_str(arguments, "content")?;

        let path = std::path::Path::new(file_path);

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("Failed to create directories for '{}': {}", file_path, e))?;
        }

        // Atomically fail if the file already exists (no TOCTOU race)
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(file_path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    anyhow::anyhow!("File '{}' already exists. Use the edit tool to modify existing files.", file_path)
                } else {
                    anyhow::anyhow!("Failed to create '{}': {}", file_path, e)
                }
            })?;
        file.write_all(content.as_bytes())
            .map_err(|e| anyhow::anyhow!("Failed to write '{}': {}", file_path, e))?;

        let lines: Vec<&str> = content.lines().collect();
        let line_count = lines.len();
        let num_width = format!("{}", line_count).len().max(3);

        const MAX_DIFF_LINES: usize = 50;
        let mut diff = String::new();
        let shown = lines.len().min(MAX_DIFF_LINES);
        for (i, line) in lines.iter().take(MAX_DIFF_LINES).enumerate() {
            diff.push_str(&format!("+{:>width$}  {}\n", i + 1, line, width = num_width));
        }
        if line_count > MAX_DIFF_LINES {
            diff.push_str(&format!("... ({} more lines)\n", line_count - shown));
        }

        Ok(format!("Created '{}' ({} lines)\n{}", file_path, line_count, diff))
    }
}
