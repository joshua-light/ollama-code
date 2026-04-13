use anyhow::Result;
use serde_json::Value;
use std::path::Path;

use super::{optional_str, required_str, Tool, ToolDefinition};

pub struct GlobTool;

impl Tool for GlobTool {
    fn name(&self) -> &str { "glob" }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "glob".to_string(),
            description: "Find files matching a glob pattern. Returns matching file paths, \
                          most recent first. Use for finding files by name or extension. \
                          Prefer this over bash find commands."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. \"*.rs\", \"**/*.ts\", \"src/**/*.json\")"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)"
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let pattern = required_str(arguments, "pattern")?;
        let path = optional_str(arguments, "path").unwrap_or(".");

        let glob = globset::GlobBuilder::new(pattern)
            .literal_separator(false)
            .build()
            .map_err(|e| anyhow::anyhow!("Invalid glob pattern '{}': {}", pattern, e))?
            .compile_matcher();

        let walker = ignore::WalkBuilder::new(path)
            .hidden(false)       // don't skip dotfiles (except .gitignored ones)
            .git_ignore(true)    // respect .gitignore
            .git_global(true)    // respect global gitignore
            .git_exclude(true)   // respect .git/info/exclude
            .build();

        let base = Path::new(path);

        let mut entries: Vec<(std::time::SystemTime, String)> = Vec::new();
        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                continue;
            }
            let entry_path = entry.path();
            // Match against the relative path from the search root
            let rel = entry_path.strip_prefix(base).unwrap_or(entry_path);
            if !glob.is_match(rel) {
                continue;
            }
            let mtime = entry.metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            entries.push((mtime, entry_path.to_string_lossy().to_string()));
        }

        if entries.is_empty() {
            return Ok("No files found matching pattern.".to_string());
        }

        entries.sort_by(|a, b| b.0.cmp(&a.0));

        let count = entries.len();
        let paths: Vec<String> = entries.into_iter().map(|(_, p)| p).collect();
        let result = paths.join("\n");
        Ok(format!("{}\n\n({} files)", result, count))
    }
}
