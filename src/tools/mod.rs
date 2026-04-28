mod bash;
mod edit;
mod evidence;
mod glob;
mod grep;
mod plan;
mod read;
mod skill;
mod subagent;
mod write;

pub use bash::BashTool;
pub use edit::EditTool;
pub use evidence::{
    new_evidence_store, EvidenceAddTool, EvidenceGetTool, EvidenceListTool, EvidenceStore,
};
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use plan::{
    PlanAddStepTool, PlanListStepsTool, PlanMarkDoneTool, PlanMarkInProgressTool, PlanSkipStepTool,
};
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

/// Coerce loosely-typed tool arguments against a JSON schema before validation.
/// Local models routinely emit `"index": "1"` instead of `1` and `"~/foo"`
/// instead of `/home/u/foo` — both fail validation for nothing. We rewrite
/// these in place so the call goes through.
///
/// Coercions performed (idempotent on already-correct args):
/// - `"integer"` / `"number"` typed fields: `"1"` → `1`, `"-2"` → `-2`,
///   `"3.14"` → `3.14`. Booleans/objects are left alone.
/// - `"boolean"` typed fields: `"true"` / `"false"` (case-insensitive) → bool.
/// - `"string"` typed fields named `path`, `file_path`, `cwd`, or any field
///   ending in `_path`: leading `~` / `~/` expanded to `$HOME`.
///
/// No effect when schema is missing/invalid or args isn't an object.
pub fn coerce_arg_types(schema: &Value, args: &mut Value) {
    let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
        return;
    };
    let Some(args_obj) = args.as_object_mut() else {
        return;
    };
    for (key, value) in args_obj.iter_mut() {
        let Some(prop) = props.get(key) else { continue };
        let Some(expected) = prop.get("type").and_then(|t| t.as_str()) else {
            continue;
        };
        match expected {
            "integer" => {
                if let Value::String(s) = value {
                    let trimmed = s.trim();
                    if let Ok(n) = trimmed.parse::<i64>() {
                        *value = serde_json::Value::from(n);
                    } else if let Ok(n) = trimmed.parse::<f64>() {
                        if n.fract() == 0.0 && n.is_finite() {
                            *value = serde_json::Value::from(n as i64);
                        }
                    }
                }
            }
            "number" => {
                if let Value::String(s) = value {
                    if let Ok(n) = s.trim().parse::<f64>() {
                        if let Some(num) = serde_json::Number::from_f64(n) {
                            *value = Value::Number(num);
                        }
                    }
                }
            }
            "boolean" => {
                if let Value::String(s) = value {
                    match s.trim().to_ascii_lowercase().as_str() {
                        "true" | "yes" | "1" => *value = Value::Bool(true),
                        "false" | "no" | "0" => *value = Value::Bool(false),
                        _ => {}
                    }
                }
            }
            "string" => {
                let path_like = key == "path"
                    || key == "file_path"
                    || key == "cwd"
                    || key.ends_with("_path");
                if !path_like {
                    continue;
                }
                if let Value::String(s) = value {
                    if let Cow::Owned(expanded) = expand_tilde(s) {
                        *s = expanded;
                    }
                }
            }
            _ => {}
        }
    }
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

    /// Coerce stringly-typed integer/number/boolean/path arguments against the
    /// tool's JSON schema. Mutates `args` in place. No-op if the tool isn't
    /// in this registry (external tools skip coercion).
    pub fn coerce(&self, name: &str, args: &mut Value) {
        let Some(idx) = self.tools.iter().position(|t| t.name() == name) else {
            return;
        };
        let schema = &self.cached_definitions[idx];
        let Some(params) = schema
            .get("function")
            .and_then(|f| f.get("parameters"))
        else {
            return;
        };
        coerce_arg_types(params, args);
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

#[cfg(test)]
mod coerce_tests {
    use super::*;
    use serde_json::json;

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "index": { "type": "integer" },
                "ratio": { "type": "number" },
                "verbose": { "type": "boolean" },
                "file_path": { "type": "string" },
                "name": { "type": "string" }
            }
        })
    }

    #[test]
    fn integer_strings_are_coerced() {
        let mut args = json!({ "index": "1" });
        coerce_arg_types(&schema(), &mut args);
        assert_eq!(args["index"], json!(1));
        let mut args = json!({ "index": "-2" });
        coerce_arg_types(&schema(), &mut args);
        assert_eq!(args["index"], json!(-2));
    }

    #[test]
    fn integer_already_correct_is_idempotent() {
        let mut args = json!({ "index": 5 });
        coerce_arg_types(&schema(), &mut args);
        assert_eq!(args["index"], json!(5));
    }

    #[test]
    fn boolean_strings_are_coerced() {
        let mut args = json!({ "verbose": "true" });
        coerce_arg_types(&schema(), &mut args);
        assert_eq!(args["verbose"], json!(true));
        let mut args = json!({ "verbose": "FALSE" });
        coerce_arg_types(&schema(), &mut args);
        assert_eq!(args["verbose"], json!(false));
    }

    #[test]
    fn tilde_expansion_for_path_field() {
        let mut args = json!({ "file_path": "~/foo/bar.rs" });
        coerce_arg_types(&schema(), &mut args);
        let p = args["file_path"].as_str().unwrap();
        assert!(!p.starts_with("~"));
        assert!(p.ends_with("/foo/bar.rs"));
    }

    #[test]
    fn non_path_string_field_untouched() {
        let mut args = json!({ "name": "~/foo" });
        coerce_arg_types(&schema(), &mut args);
        assert_eq!(args["name"], json!("~/foo"));
    }

    #[test]
    fn unknown_field_left_alone() {
        let mut args = json!({ "extra": "1", "index": "2" });
        coerce_arg_types(&schema(), &mut args);
        assert_eq!(args["extra"], json!("1"));
        assert_eq!(args["index"], json!(2));
    }
}
