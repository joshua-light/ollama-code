use anyhow::Result;
use serde_json::Value;
use std::fs;
use std::io::Write;

pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn definition(&self) -> ToolDefinition;
    fn execute(&self, arguments: &Value) -> Result<String>;
}

// --- Bash tool ---

/// Format the output of a bash command into a result string and success flag.
pub fn format_bash_output(output: &std::process::Output) -> (String, bool) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let mut result = String::new();
    if !stdout.is_empty() {
        result.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(&stderr);
    }
    if result.is_empty() {
        result.push_str("(no output)");
    }
    let success = output.status.success();
    if !success {
        result.push_str(&format!(
            "\n(exit code: {})",
            output.status.code().unwrap_or(-1)
        ));
    }
    (result, success)
}

pub struct BashTool;

impl BashTool {
    /// Async execution with timeout and kill-on-drop. This is the primary
    /// execution path — the sync `Tool::execute()` should never be called.
    pub async fn execute_async(
        &self,
        arguments: &Value,
        timeout: std::time::Duration,
    ) -> (String, bool) {
        let command = arguments
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        match tokio::process::Command::new("bash")
            .arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
        {
            Ok(child) => {
                match tokio::time::timeout(timeout, child.wait_with_output()).await {
                    Ok(Ok(output)) => format_bash_output(&output),
                    Ok(Err(e)) => (format!("Error: {}", e), false),
                    Err(_) => (
                        format!("Error: command timed out after {}s", timeout.as_secs()),
                        false,
                    ),
                }
            }
            Err(e) => (format!("Error: {}", e), false),
        }
    }
}

impl Tool for BashTool {
    fn name(&self) -> &str { "bash" }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "bash".to_string(),
            description: "Execute a bash command and return its output. Use this for running \
                          shell commands, installing packages, running programs, git operations, \
                          and other terminal tasks. Do NOT use this to read or edit files — use \
                          the read and edit tools instead."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    }
                },
                "required": ["command"]
            }),
        }
    }

    fn execute(&self, _arguments: &Value) -> Result<String> {
        anyhow::bail!("BashTool must be executed via execute_async()")
    }
}

// --- Read tool ---

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
        let file_path = arguments
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'file_path' argument"))?;

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

// --- Edit tool ---

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
        let file_path = arguments
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'file_path' argument"))?;
        let old_string = arguments
            .get("old_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'old_string' argument"))?;
        let new_string = arguments
            .get("new_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'new_string' argument"))?;

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

// --- Write tool ---

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
        let file_path = arguments
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'file_path' argument"))?;
        let content = arguments
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'content' argument"))?;

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

        let line_count = content.lines().count();
        Ok(format!("Created '{}' ({} lines)", file_path, line_count))
    }
}

// --- Glob tool ---

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
        let pattern = arguments
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'pattern' argument"))?;
        let path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        // Decompose the pattern: for "**/*.rs" search path with -name "*.rs"
        // For "src/**/*.rs" search src with -name "*.rs"
        let (search_path, name_pattern) = if let Some(pos) = pattern.rfind('/') {
            let dir_part = &pattern[..pos];
            let file_part = &pattern[pos + 1..];
            // If the dir part is just "**" or contains only wildcards, search from root path
            let dir = if dir_part == "**" || dir_part.chars().all(|c| c == '*') {
                path.to_string()
            } else {
                // Strip leading **/ from dir part
                let clean_dir = dir_part.trim_start_matches("**/");
                if clean_dir.is_empty() {
                    path.to_string()
                } else if path == "." {
                    clean_dir.to_string()
                } else {
                    format!("{}/{}", path, clean_dir)
                }
            };
            (dir, file_part.to_string())
        } else {
            (path.to_string(), pattern.to_string())
        };

        let output = std::process::Command::new("find")
            .arg(&search_path)
            .arg("-type")
            .arg("f")
            .arg("-name")
            .arg(&name_pattern)
            .arg("-not")
            .arg("-path")
            .arg("*/.git/*")
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to run find: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.trim().is_empty() {
            return Ok("No files found matching pattern.".to_string());
        }

        // Sort by modification time (most recent first)
        let mut entries: Vec<(std::time::SystemTime, String)> = stdout
            .lines()
            .filter_map(|line| {
                let path = line.trim();
                if path.is_empty() {
                    return None;
                }
                let mtime = fs::metadata(path)
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                Some((mtime, path.to_string()))
            })
            .collect();
        entries.sort_by(|a, b| b.0.cmp(&a.0));

        let count = entries.len();
        let paths: Vec<String> = entries.into_iter().map(|(_, p)| p).collect();
        let result = paths.join("\n");
        Ok(format!("{}\n\n({} files)", result, count))
    }
}

// --- Grep tool ---

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
        let pattern = arguments
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'pattern' argument"))?;
        let path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");
        let include = arguments
            .get("include")
            .and_then(|v| v.as_str());

        let mut cmd = std::process::Command::new("grep");
        cmd.arg("-rn")     // recursive, line numbers
           .arg("-E")      // extended regex
           .arg("--color=never")
           .arg("--exclude-dir=.git")
           .arg("-m").arg("101");  // stop after 101 matches per file

        if let Some(glob) = include {
            cmd.arg("--include").arg(glob);
        }

        cmd.arg(pattern).arg(path);

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

// --- Subagent tool (definition only — execution handled by agent loop) ---

pub struct SubagentToolDef;

impl Tool for SubagentToolDef {
    fn name(&self) -> &str { "subagent" }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "subagent".to_string(),
            description: "Spawn a sub-agent with a fresh, clean context to handle a focused task. \
                          The sub-agent has its own conversation history (only the task you give it), \
                          making it ideal for research, exploration, and self-contained coding tasks \
                          that benefit from a clean context window. Returns the sub-agent's final \
                          response. The sub-agent cannot see this conversation, so include all \
                          necessary context in the task description."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "A self-contained task description. Include all necessary context — the sub-agent cannot see the current conversation."
                    }
                },
                "required": ["task"]
            }),
        }
    }

    fn execute(&self, _arguments: &Value) -> Result<String> {
        anyhow::bail!("subagent tool must be executed by the agent loop, not the tool registry")
    }
}

// --- Tool registry ---

pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
    cached_definitions: Vec<Value>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            cached_definitions: Vec::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let def = tool.definition();
        self.cached_definitions.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": def.name,
                "description": def.description,
                "parameters": def.parameters,
            }
        }));
        self.tools.push(tool);
    }

    pub fn definitions(&self) -> Vec<Value> {
        self.cached_definitions.clone()
    }

    pub fn execute(&self, name: &str, arguments: &Value) -> Result<String> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == name)
            .ok_or_else(|| anyhow::anyhow!("Unknown tool: {}", name))?;
        tool.execute(arguments)
    }
}
