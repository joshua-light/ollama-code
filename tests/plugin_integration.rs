mod harness;

use std::collections::HashMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;

use harness::{run_agent_with_config, ConfirmStrategy, MockBackend, MockResponse};
use ollama_code::config::Config;
use tempfile::TempDir;

fn setup_plugin(dir: &std::path::Path, name: &str, tool_name: &str, script: &str) {
    let plugin_dir = dir.join(name);
    fs::create_dir_all(&plugin_dir).unwrap();

    let manifest = format!(
        r#"
name = "{name}"
version = "0.1.0"
description = "Test plugin {name}"

[[tools]]
name = "{tool_name}"
description = "A test tool from {name}"
command = "./run"
parameters = '{{"type": "object", "properties": {{"input": {{"type": "string"}}}}}}'
"#
    );
    fs::write(plugin_dir.join("PLUGIN.toml"), manifest).unwrap();

    let script_path = plugin_dir.join("run");
    fs::write(&script_path, script).unwrap();
    fs::set_permissions(&script_path, fs::Permissions::from_mode(0o755)).unwrap();
}

/// End-to-end: agent discovers an external plugin tool, the model calls it,
/// and the result flows back through the agent loop.
#[tokio::test]
async fn agent_with_external_plugin_end_to_end() {
    let plugin_dir = TempDir::new().unwrap();
    setup_plugin(
        plugin_dir.path(),
        "greeter",
        "greet",
        "#!/bin/sh\necho \"Hello from plugin!\"",
    );

    let config = Config {
        plugin_dirs: Some(vec![plugin_dir.path().to_string_lossy().to_string()]),
        ..Default::default()
    };

    // Model calls the external tool, then gives final response
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call(
            "greet",
            serde_json::json!({"input": "world"}),
        ),
        MockResponse::text("The plugin said hello!"),
    ]));

    let result = run_agent_with_config(
        backend.clone(),
        "Greet someone",
        ConfirmStrategy::ApproveAll,
        &config,
    )
    .await;

    assert!(result.is_done());
    assert_eq!(result.final_content(), "The plugin said hello!");

    // The tool should have been called and returned successfully
    let tool_results = result.tool_results();
    assert_eq!(tool_results.len(), 1);
    assert_eq!(tool_results[0].0, "greet");
    assert!(tool_results[0].2, "tool should succeed");
    assert!(
        tool_results[0].1.contains("Hello from plugin!"),
        "Expected plugin output, got: {}",
        tool_results[0].1
    );

    // Tool definition should have been sent to backend
    let calls = backend.calls();
    let tools = calls[0].tools.as_ref().unwrap();
    let tool_names: Vec<String> = tools
        .iter()
        .filter_map(|t| t["function"]["name"].as_str().map(|s| s.to_string()))
        .collect();
    assert!(tool_names.contains(&"greet".to_string()), "greet tool should be in definitions: {:?}", tool_names);
}

/// Disable a core tool and enable an external plugin in the same config.
#[tokio::test]
async fn disabled_core_plus_external_plugin() {
    let plugin_dir = TempDir::new().unwrap();
    setup_plugin(
        plugin_dir.path(),
        "helper",
        "helper_tool",
        "#!/bin/sh\necho \"helper output\"",
    );

    let mut plugins = HashMap::new();
    plugins.insert("bash".to_string(), toml::Value::Boolean(false));

    let config = Config {
        plugins: Some(plugins),
        plugin_dirs: Some(vec![plugin_dir.path().to_string_lossy().to_string()]),
        ..Default::default()
    };

    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text("ok"),
    ]));

    let result = run_agent_with_config(
        backend.clone(),
        "Hi",
        ConfirmStrategy::ApproveAll,
        &config,
    )
    .await;

    assert!(result.is_done());

    let calls = backend.calls();
    let tools = calls[0].tools.as_ref().unwrap();
    let tool_names: Vec<String> = tools
        .iter()
        .filter_map(|t| t["function"]["name"].as_str().map(|s| s.to_string()))
        .collect();

    // bash should be gone, helper_tool should be present
    assert!(!tool_names.contains(&"bash".to_string()), "bash should be disabled");
    assert!(tool_names.contains(&"helper_tool".to_string()), "helper_tool should be present: {:?}", tool_names);
    // core tools still present
    assert!(tool_names.contains(&"read".to_string()));
}

/// External plugin tool with needs_confirm should trigger confirmation.
#[tokio::test]
async fn external_plugin_tool_confirm() {
    let plugin_dir = TempDir::new().unwrap();
    let name = "danger";
    let pdir = plugin_dir.path().join(name);
    fs::create_dir_all(&pdir).unwrap();

    let manifest = format!(
        r#"
name = "{name}"
version = "0.1.0"
description = "Dangerous plugin"

[[tools]]
name = "dangerous_tool"
description = "Does dangerous things"
command = "./run"
needs_confirm = true
parameters = '{{"type": "object", "properties": {{}}}}'
"#
    );
    fs::write(pdir.join("PLUGIN.toml"), manifest).unwrap();
    let script_path = pdir.join("run");
    fs::write(&script_path, "#!/bin/sh\necho \"danger executed\"").unwrap();
    fs::set_permissions(&script_path, fs::Permissions::from_mode(0o755)).unwrap();

    let config = Config {
        plugin_dirs: Some(vec![plugin_dir.path().to_string_lossy().to_string()]),
        ..Default::default()
    };

    // Model calls the dangerous tool, user denies
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call("dangerous_tool", serde_json::json!({})),
        MockResponse::text("OK, I won't do that."),
    ]));

    let result = run_agent_with_config(
        backend.clone(),
        "Do something dangerous",
        ConfirmStrategy::DenyAll,
        &config,
    )
    .await;

    assert!(result.is_done());

    let tool_results = result.tool_results();
    assert_eq!(tool_results.len(), 1);
    assert!(!tool_results[0].2, "tool should have been denied");
    assert!(tool_results[0].1.contains("denied"));
}
