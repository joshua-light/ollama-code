use std::sync::OnceLock;

use anyhow::Result;
use serde_json::Value;

use super::{optional_str, required_str, Tool, ToolDefinition};

/// Cache `rg` availability — checked once per process.
fn has_ripgrep() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::process::Command::new("rg")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

pub struct GrepTool;

impl Tool for GrepTool {
    fn name(&self) -> &str { "grep" }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "grep".to_string(),
            description: "Search file contents using regex patterns. Returns matching lines \
                          with file paths and line numbers. More efficient than bash grep for \
                          codebase search. Prefer this over bash grep/rg commands."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in (default: current directory)"
                    },
                    "include": {
                        "type": "string",
                        "description": "File glob to filter (e.g. \"*.rs\", \"*.py\")"
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let pattern = required_str(arguments, "pattern")?;
        let path = optional_str(arguments, "path").unwrap_or(".");
        let include = optional_str(arguments, "include");

        let use_rg = has_ripgrep();

        let mut cmd;
        if use_rg {
            cmd = std::process::Command::new("rg");
            cmd.arg("-n")           // line numbers
               .arg("--color=never")
               .arg("-m").arg("101");  // stop after 101 matches per file

            if let Some(glob) = include {
                cmd.arg("--glob").arg(glob);
            }

            cmd.arg(pattern).arg(path);
        } else {
            cmd = std::process::Command::new("grep");
            cmd.arg("-rn")     // recursive, line numbers
               .arg("-E")      // extended regex
               .arg("--color=never")
               .arg("-m").arg("101");

            for dir in [".git", "node_modules", "target", ".venv", "venv",
                        "__pycache__", "dist", "build", ".next", ".cache",
                        "vendor", "bower_components"] {
                cmd.arg("--exclude-dir").arg(dir);
            }

            if let Some(glob) = include {
                cmd.arg("--include").arg(glob);
            }

            cmd.arg(pattern).arg(path);
        }

        let output = cmd.output()
            .map_err(|e| anyhow::anyhow!("Failed to run grep: {}", e))?;

        let result = String::from_utf8_lossy(&output.stdout);

        if result.trim().is_empty() {
            return Ok("No matches found.".to_string());
        }

        let lines: Vec<&str> = result.lines().collect();
        let count = lines.len();
        if count > 100 {
            let truncated: String = lines[..100].join("\n");
            Ok(format!(
                "{}\n\n... ({} total matches, showing first 100. Refine your pattern.)",
                truncated, count
            ))
        } else {
            Ok(format!("{}\n\n({} matches)", result.trim(), count))
        }
    }
}
