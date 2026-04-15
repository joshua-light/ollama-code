use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::sync::Mutex;

use super::{expand_tilde, required_str, Tool, ToolDefinition};

/// Tracks which (path, start, end) ranges have been read.
struct ReadCache {
    /// path -> list of (start_line_0based, end_line_0based) ranges read.
    ranges: HashMap<String, Vec<(usize, usize)>>,
}

impl ReadCache {
    fn new() -> Self {
        Self {
            ranges: HashMap::new(),
        }
    }

    /// Check if the requested range is fully covered by a previous read.
    fn is_covered(&self, path: &str, start: usize, end: usize) -> bool {
        if let Some(ranges) = self.ranges.get(path) {
            ranges.iter().any(|&(s, e)| s <= start && e >= end)
        } else {
            false
        }
    }

    /// Record a read range.
    fn record(&mut self, path: &str, start: usize, end: usize) {
        self.ranges
            .entry(path.to_string())
            .or_default()
            .push((start, end));
    }

    /// Clear the cache (e.g. on conversation clear).
    #[allow(dead_code)]
    fn clear(&mut self) {
        self.ranges.clear();
    }
}

pub struct ReadTool {
    cache: Mutex<ReadCache>,
}

impl Default for ReadTool {
    fn default() -> Self {
        Self {
            cache: Mutex::new(ReadCache::new()),
        }
    }
}

impl ReadTool {
    pub fn new() -> Self {
        Self::default()
    }
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

        // Check read cache for duplicate reads
        let dedup_note = {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            let covered = cache.is_covered(file_path.as_ref(), start, end);
            cache.record(file_path.as_ref(), start, end);
            if covered {
                Some(format!(
                    "(Note: you have already read this section of '{}'. \
                     Consider using the information from the earlier read instead \
                     of re-reading the same content.)\n\n",
                    file_path
                ))
            } else {
                None
            }
        };

        let mut result = String::new();

        let has_dedup_note = dedup_note.is_some();
        if let Some(note) = dedup_note {
            result.push_str(&note);
        }

        for (i, line) in lines[start..end].iter().enumerate() {
            let line_num = start + i + 1;
            result.push_str(&format!("{:>4}\t{}\n", line_num, line));
        }

        if result.is_empty() || (has_dedup_note && lines[start..end].is_empty()) {
            if !has_dedup_note {
                result = "(empty file)".to_string();
            }
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
