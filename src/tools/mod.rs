mod bash;
mod edit;
mod glob;
mod grep;
mod read;
mod subagent;
mod write;

pub use bash::BashTool;
pub use edit::EditTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use read::ReadTool;
pub use subagent::SubagentToolDef;
pub use write::WriteTool;

use anyhow::Result;
use serde_json::Value;

pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

pub(crate) fn required_str<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing '{}' argument", key))
}

pub(crate) fn optional_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(|v| v.as_str())
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn definition(&self) -> ToolDefinition;
    fn execute(&self, arguments: &Value) -> Result<String>;
}

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
