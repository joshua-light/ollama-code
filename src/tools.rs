use anyhow::Result;
use serde_json::Value;
use std::process::Command;

pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn execute(&self, arguments: &Value) -> Result<String>;
}

// --- Bash tool ---

pub struct BashTool;

impl Tool for BashTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "bash".to_string(),
            description: "Execute a bash command and return its output. Use this to run shell \
                          commands, read files, list directories, and perform system operations."
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

    fn execute(&self, arguments: &Value) -> Result<String> {
        let command = arguments
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'command' argument"))?;

        let output = Command::new("bash").arg("-c").arg(command).output()?;

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
        if !output.status.success() {
            result.push_str(&format!(
                "\n(exit code: {})",
                output.status.code().unwrap_or(-1)
            ));
        }

        Ok(result)
    }
}

// --- Tool registry ---

pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn definitions(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|t| {
                let def = t.definition();
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": def.name,
                        "description": def.description,
                        "parameters": def.parameters,
                    }
                })
            })
            .collect()
    }

    pub fn execute(&self, name: &str, arguments: &Value) -> Result<String> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.definition().name == name)
            .ok_or_else(|| anyhow::anyhow!("Unknown tool: {}", name))?;
        tool.execute(arguments)
    }
}
