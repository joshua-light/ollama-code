mod bash;
mod edit;
mod glob;
mod grep;
mod read;
mod skill;
mod subagent;
mod write;

pub use bash::BashTool;
pub use edit::EditTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use read::ReadTool;
pub use skill::SkillTool;
pub use subagent::SubagentToolDef;
pub use write::WriteTool;

use anyhow::Result;
use serde_json::Value;
use std::borrow::Cow;

/// Expand a leading `~` or `~/` to the user's home directory.
/// Returns the original string unchanged if there is no leading tilde
/// or if the home directory cannot be determined.
pub(crate) fn expand_tilde(path: &str) -> Cow<'_, str> {
    if path == "~" || path.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            let home = home.to_string_lossy();
            return Cow::Owned(format!("{}{}", home, &path[1..]));
        }
    }
    Cow::Borrowed(path)
}

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

/// Validate tool call arguments against a tool's JSON schema.
/// Returns `Ok(())` if valid, or an error message describing what's wrong.
pub fn validate_tool_args(schema: &Value, args: &Value) -> std::result::Result<(), String> {
    let properties = schema.get("properties").and_then(|p| p.as_object());
    let required: Vec<&str> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    // Args should be an object
    let args_obj = match args.as_object() {
        Some(o) => o,
        None => return Err("Arguments must be a JSON object".to_string()),
    };

    // Check required fields
    for field in &required {
        if !args_obj.contains_key(*field) {
            return Err(format!(
                "Missing required argument '{}'. Required: [{}]",
                field,
                required.join(", ")
            ));
        }
    }

    // Type-check provided fields against schema
    if let Some(props) = properties {
        for (key, value) in args_obj {
            if let Some(prop_schema) = props.get(key) {
                if let Some(expected_type) = prop_schema.get("type").and_then(|t| t.as_str()) {
                    let type_ok = match expected_type {
                        "string" => value.is_string(),
                        "number" => value.is_number(),
                        "integer" => value.is_i64() || value.is_u64(),
                        "boolean" => value.is_boolean(),
                        "object" => value.is_object(),
                        "array" => value.is_array(),
                        _ => true,
                    };
                    if !type_ok {
                        return Err(format!(
                            "Argument '{}' should be {}, got {}",
                            key,
                            expected_type,
                            value_type_name(value),
                        ));
                    }
                }
            }
        }
    }

    Ok(())
}

fn value_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
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
        let name = tool.name().to_string();
        if self.tools.iter().any(|t| t.name() == name) {
            eprintln!("Warning: duplicate tool name '{}', skipping registration", name);
            return;
        }
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

    /// Return definitions for only the named tools (for dynamic scoping).
    pub fn definitions_filtered(&self, allowed: &[&str]) -> Vec<Value> {
        self.cached_definitions
            .iter()
            .filter(|def| {
                def.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .map(|name| allowed.contains(&name))
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    /// Return definitions excluding the named tools.
    pub fn definitions_excluding(&self, excluded: &[&str]) -> Vec<Value> {
        self.cached_definitions
            .iter()
            .filter(|def| {
                def.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .map(|name| !excluded.contains(&name))
                    .unwrap_or(true)
            })
            .cloned()
            .collect()
    }

    /// Validate arguments for a tool against its schema.
    /// Returns `None` if the tool is not found (external tools skip validation).
    pub fn validate(&self, name: &str, args: &Value) -> Option<std::result::Result<(), String>> {
        let idx = self.tools.iter().position(|t| t.name() == name)?;
        let schema = &self.cached_definitions[idx];
        let params = schema
            .get("function")
            .and_then(|f| f.get("parameters"))?;
        Some(validate_tool_args(params, args))
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
