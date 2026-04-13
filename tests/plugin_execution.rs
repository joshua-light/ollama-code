use std::fs;
use std::os::unix::fs::PermissionsExt;

use ollama_code::plugin::ExternalTool;
use ollama_code::tools::Tool;
use tempfile::TempDir;

/// Create an executable script in `dir` with the given content.
fn write_script(dir: &std::path::Path, name: &str, content: &str) {
    let path = dir.join(name);
    fs::write(&path, content).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn external_tool_executes_successfully() {
    let dir = TempDir::new().unwrap();
    write_script(
        dir.path(),
        "run",
        "#!/bin/sh\necho \"hello from plugin\"",
    );

    let tool = ExternalTool::new(
        "test_tool".to_string(),
        "A test tool".to_string(),
        dir.path().join("run").to_str().unwrap().to_string(),
        dir.path().to_path_buf(),
        None,     // needs_confirm
        None,     // timeout
        Some(r#"{"type": "object", "properties": {"msg": {"type": "string"}}}"#.to_string()),
        None,     // plugin config
    );

    assert_eq!(tool.name(), "test_tool");

    let def = tool.definition();
    assert_eq!(def.name, "test_tool");
    assert_eq!(def.description, "A test tool");

    let args = serde_json::json!({"msg": "hi"});
    let result = tool.execute(&args).unwrap();
    assert_eq!(result.trim(), "hello from plugin");
}

#[test]
fn external_tool_receives_arguments_on_stdin() {
    let dir = TempDir::new().unwrap();
    // Script that reads stdin and echoes it
    write_script(
        dir.path(),
        "run",
        "#!/bin/sh\ncat",
    );

    let tool = ExternalTool::new(
        "echo_tool".to_string(),
        "Echoes stdin".to_string(),
        dir.path().join("run").to_str().unwrap().to_string(),
        dir.path().to_path_buf(),
        None,
        None,
        Some(r#"{"type": "object", "properties": {}}"#.to_string()),
        None,
    );

    let args = serde_json::json!({"key": "value"});
    let result = tool.execute(&args).unwrap();
    // The output should be a JSON object with "arguments"
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["arguments"]["key"], "value");
}

#[test]
fn external_tool_receives_plugin_config() {
    let dir = TempDir::new().unwrap();
    write_script(dir.path(), "run", "#!/bin/sh\ncat");

    let mut plugin_cfg = toml::map::Map::new();
    plugin_cfg.insert("api_key".to_string(), toml::Value::String("secret123".to_string()));

    let tool = ExternalTool::new(
        "cfg_tool".to_string(),
        "Receives config".to_string(),
        dir.path().join("run").to_str().unwrap().to_string(),
        dir.path().to_path_buf(),
        None,
        None,
        Some(r#"{"type": "object", "properties": {}}"#.to_string()),
        Some(plugin_cfg),
    );

    let result = tool.execute(&serde_json::json!({})).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["config"]["api_key"], "secret123");
}

#[test]
fn external_tool_failure_on_nonzero_exit() {
    let dir = TempDir::new().unwrap();
    write_script(
        dir.path(),
        "run",
        "#!/bin/sh\necho 'something went wrong' >&2\nexit 1",
    );

    let tool = ExternalTool::new(
        "fail_tool".to_string(),
        "Fails".to_string(),
        dir.path().join("run").to_str().unwrap().to_string(),
        dir.path().to_path_buf(),
        None,
        None,
        Some(r#"{"type": "object", "properties": {}}"#.to_string()),
        None,
    );

    let result = tool.execute(&serde_json::json!({}));
    // Should return an error or error message
    match result {
        Ok(output) => {
            // If it returns Ok with error info, it should indicate failure
            assert!(
                output.contains("something went wrong") || output.contains("exit"),
                "Expected error info in output: {}",
                output
            );
        }
        Err(e) => {
            assert!(
                e.to_string().contains("something went wrong")
                    || e.to_string().contains("exit"),
                "Expected error info: {}",
                e
            );
        }
    }
}

#[test]
fn external_tool_timeout() {
    let dir = TempDir::new().unwrap();
    write_script(
        dir.path(),
        "run",
        "#!/bin/sh\nsleep 30\necho done",
    );

    let tool = ExternalTool::new(
        "slow_tool".to_string(),
        "Takes too long".to_string(),
        dir.path().join("run").to_str().unwrap().to_string(),
        dir.path().to_path_buf(),
        None,
        Some(1), // 1 second timeout
        Some(r#"{"type": "object", "properties": {}}"#.to_string()),
        None,
    );

    let result = tool.execute(&serde_json::json!({}));
    match result {
        Ok(output) => {
            assert!(
                output.to_lowercase().contains("timeout") || output.to_lowercase().contains("timed out"),
                "Expected timeout error: {}",
                output
            );
        }
        Err(e) => {
            assert!(
                e.to_string().to_lowercase().contains("timeout") || e.to_string().to_lowercase().contains("timed out"),
                "Expected timeout error: {}",
                e
            );
        }
    }
}

#[test]
fn external_tool_needs_confirm_flag() {
    let dir = TempDir::new().unwrap();
    write_script(dir.path(), "run", "#!/bin/sh\necho ok");

    // needs_confirm = true
    let tool = ExternalTool::new(
        "confirm_tool".to_string(),
        "Needs confirmation".to_string(),
        dir.path().join("run").to_str().unwrap().to_string(),
        dir.path().to_path_buf(),
        Some(true),
        None,
        Some(r#"{"type": "object", "properties": {}}"#.to_string()),
        None,
    );
    assert!(tool.needs_confirm());

    // needs_confirm = false (default)
    let tool2 = ExternalTool::new(
        "safe_tool".to_string(),
        "No confirmation".to_string(),
        dir.path().join("run").to_str().unwrap().to_string(),
        dir.path().to_path_buf(),
        None,
        None,
        Some(r#"{"type": "object", "properties": {}}"#.to_string()),
        None,
    );
    assert!(!tool2.needs_confirm());
}

#[test]
fn external_tool_definition_has_correct_schema() {
    let dir = TempDir::new().unwrap();
    write_script(dir.path(), "run", "#!/bin/sh\necho ok");

    let schema = r#"{"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}"#;
    let tool = ExternalTool::new(
        "schema_tool".to_string(),
        "Has schema".to_string(),
        dir.path().join("run").to_str().unwrap().to_string(),
        dir.path().to_path_buf(),
        None,
        None,
        Some(schema.to_string()),
        None,
    );

    let def = tool.definition();
    assert_eq!(def.parameters["type"], "object");
    assert!(def.parameters["properties"]["query"].is_object());
    assert_eq!(def.parameters["required"][0], "query");
}

#[test]
fn external_tool_no_parameters_schema() {
    let dir = TempDir::new().unwrap();
    write_script(dir.path(), "run", "#!/bin/sh\necho ok");

    let tool = ExternalTool::new(
        "no_schema_tool".to_string(),
        "No schema".to_string(),
        dir.path().join("run").to_str().unwrap().to_string(),
        dir.path().to_path_buf(),
        None,
        None,
        None, // no parameters
        None,
    );

    let def = tool.definition();
    // Should have a default empty object schema
    assert_eq!(def.parameters["type"], "object");
}
