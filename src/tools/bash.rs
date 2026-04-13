use anyhow::Result;
use serde_json::Value;

use super::{format_bash_output, optional_str, Tool, ToolDefinition};

pub struct BashTool;

impl BashTool {
    /// Async execution with timeout and kill-on-drop. This is the primary
    /// execution path — the sync `Tool::execute()` should never be called.
    pub async fn execute_async(
        &self,
        arguments: &Value,
        timeout: std::time::Duration,
    ) -> (String, bool) {
        let command = optional_str(arguments, "command").unwrap_or("");
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
