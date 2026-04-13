mod harness;

use std::collections::HashMap;
use std::sync::Arc;

use harness::{run_agent, run_agent_with_config, ConfirmStrategy, MockBackend, MockResponse};
use ollama_code::config::Config;

/// When no plugins config is set, all core tools should be registered.
#[tokio::test]
async fn core_tools_enabled_by_default() {
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text("Hello"),
    ]));

    let result = run_agent(backend.clone(), "Hi", ConfirmStrategy::ApproveAll).await;
    assert!(result.is_done());

    // Check tool definitions sent to the backend
    let calls = backend.calls();
    assert_eq!(calls.len(), 1);
    let tools = calls[0].tools.as_ref().expect("tools should be present");

    let tool_names: Vec<String> = tools
        .iter()
        .filter_map(|t| t["function"]["name"].as_str().map(|s| s.to_string()))
        .collect();

    // All core tools should be present
    assert!(tool_names.contains(&"bash".to_string()), "bash missing: {:?}", tool_names);
    assert!(tool_names.contains(&"read".to_string()), "read missing: {:?}", tool_names);
    assert!(tool_names.contains(&"edit".to_string()), "edit missing: {:?}", tool_names);
    assert!(tool_names.contains(&"write".to_string()), "write missing: {:?}", tool_names);
    assert!(tool_names.contains(&"glob".to_string()), "glob missing: {:?}", tool_names);
    assert!(tool_names.contains(&"grep".to_string()), "grep missing: {:?}", tool_names);
    assert!(tool_names.contains(&"subagent".to_string()), "subagent missing: {:?}", tool_names);
}

/// Setting `bash = false` in [plugins] should remove the bash tool.
#[tokio::test]
async fn core_tool_disabled_via_config() {
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text("Hello"),
    ]));

    let mut plugins = HashMap::new();
    plugins.insert("bash".to_string(), toml::Value::Boolean(false));

    let config = Config {
        plugins: Some(plugins),
        ..Default::default()
    };

    let result = run_agent_with_config(
        backend.clone(),
        "Hi",
        ConfirmStrategy::ApproveAll,
        &config,
    )
    .await;
    assert!(result.is_done());

    let calls = backend.calls();
    let tools = calls[0].tools.as_ref().expect("tools should be present");
    let tool_names: Vec<String> = tools
        .iter()
        .filter_map(|t| t["function"]["name"].as_str().map(|s| s.to_string()))
        .collect();

    // bash should be gone
    assert!(!tool_names.contains(&"bash".to_string()), "bash should be disabled: {:?}", tool_names);
    // other tools should still be present
    assert!(tool_names.contains(&"read".to_string()), "read missing: {:?}", tool_names);
    assert!(tool_names.contains(&"edit".to_string()), "edit missing: {:?}", tool_names);
}

/// Disabling multiple core tools should remove all of them.
#[tokio::test]
async fn multiple_core_tools_disabled() {
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text("Hello"),
    ]));

    let mut plugins = HashMap::new();
    plugins.insert("bash".to_string(), toml::Value::Boolean(false));
    plugins.insert("edit".to_string(), toml::Value::Boolean(false));
    plugins.insert("write".to_string(), toml::Value::Boolean(false));

    let config = Config {
        plugins: Some(plugins),
        ..Default::default()
    };

    let result = run_agent_with_config(
        backend.clone(),
        "Hi",
        ConfirmStrategy::ApproveAll,
        &config,
    )
    .await;
    assert!(result.is_done());

    let calls = backend.calls();
    let tools = calls[0].tools.as_ref().expect("tools should be present");
    let tool_names: Vec<String> = tools
        .iter()
        .filter_map(|t| t["function"]["name"].as_str().map(|s| s.to_string()))
        .collect();

    assert!(!tool_names.contains(&"bash".to_string()));
    assert!(!tool_names.contains(&"edit".to_string()));
    assert!(!tool_names.contains(&"write".to_string()));
    // remaining tools still present
    assert!(tool_names.contains(&"read".to_string()));
    assert!(tool_names.contains(&"glob".to_string()));
    assert!(tool_names.contains(&"grep".to_string()));
}

/// Unknown plugin names in config should not cause errors.
#[tokio::test]
async fn unknown_plugin_flag_ignored() {
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text("Hello"),
    ]));

    let mut plugins = HashMap::new();
    plugins.insert("nonexistent_tool".to_string(), toml::Value::Boolean(false));

    let config = Config {
        plugins: Some(plugins),
        ..Default::default()
    };

    let result = run_agent_with_config(
        backend.clone(),
        "Hi",
        ConfirmStrategy::ApproveAll,
        &config,
    )
    .await;
    assert!(result.is_done());

    // All core tools should still be present
    let calls = backend.calls();
    let tools = calls[0].tools.as_ref().expect("tools should be present");
    let tool_names: Vec<String> = tools
        .iter()
        .filter_map(|t| t["function"]["name"].as_str().map(|s| s.to_string()))
        .collect();

    assert!(tool_names.contains(&"bash".to_string()));
    assert!(tool_names.contains(&"read".to_string()));
}

/// `is_tool_enabled` should return true by default, false only for explicit `false`.
#[tokio::test]
async fn config_is_tool_enabled_logic() {
    // No plugins section
    let config = Config::default();
    assert!(config.is_tool_enabled("bash"));
    assert!(config.is_tool_enabled("anything"));

    // Empty plugins section
    let config = Config {
        plugins: Some(HashMap::new()),
        ..Default::default()
    };
    assert!(config.is_tool_enabled("bash"));

    // Explicitly true
    let mut plugins = HashMap::new();
    plugins.insert("bash".to_string(), toml::Value::Boolean(true));
    let config = Config {
        plugins: Some(plugins),
        ..Default::default()
    };
    assert!(config.is_tool_enabled("bash"));

    // Explicitly false
    let mut plugins = HashMap::new();
    plugins.insert("bash".to_string(), toml::Value::Boolean(false));
    let config = Config {
        plugins: Some(plugins),
        ..Default::default()
    };
    assert!(!config.is_tool_enabled("bash"));

    // Non-boolean value (table) should not disable
    let mut plugins = HashMap::new();
    let mut table = toml::map::Map::new();
    table.insert("key".to_string(), toml::Value::String("val".to_string()));
    plugins.insert("my_plugin".to_string(), toml::Value::Table(table));
    let config = Config {
        plugins: Some(plugins),
        ..Default::default()
    };
    assert!(config.is_tool_enabled("my_plugin"));
}
