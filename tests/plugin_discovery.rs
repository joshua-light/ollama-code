use std::fs;

use ollama_code::plugin::{discover_plugins, PluginManifest};
use tempfile::TempDir;

fn write_plugin(dir: &std::path::Path, name: &str, toml_content: &str) {
    let plugin_dir = dir.join(name);
    fs::create_dir_all(&plugin_dir).unwrap();
    fs::write(plugin_dir.join("PLUGIN.toml"), toml_content).unwrap();
}

fn minimal_manifest(name: &str) -> String {
    format!(
        r#"
name = "{name}"
version = "0.1.0"
description = "Test plugin {name}"

[[tools]]
name = "{name}_tool"
description = "A test tool"
command = "./run"
parameters = '{{"type": "object", "properties": {{}}}}'
"#
    )
}

#[test]
fn parse_valid_manifest() {
    let toml_str = r#"
name = "hello"
version = "0.1.0"
description = "A hello world plugin"

[[tools]]
name = "hello_world"
description = "Says hello"
command = "./run.sh"
needs_confirm = true
timeout = 60
parameters = '{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}'
"#;
    let manifest: PluginManifest = toml::from_str(toml_str).unwrap();
    assert_eq!(manifest.name, "hello");
    assert_eq!(manifest.version, "0.1.0");
    assert_eq!(manifest.description, "A hello world plugin");
    assert_eq!(manifest.tools.len(), 1);

    let tool = &manifest.tools[0];
    assert_eq!(tool.name, "hello_world");
    assert_eq!(tool.description, "Says hello");
    assert_eq!(tool.command, "./run.sh");
    assert_eq!(tool.needs_confirm, Some(true));
    assert_eq!(tool.timeout, Some(60));
    assert!(tool.parameters.is_some());
}

#[test]
fn parse_manifest_multiple_tools() {
    let toml_str = r#"
name = "multi"
version = "0.1.0"
description = "Multi-tool plugin"

[[tools]]
name = "tool_a"
description = "First tool"
command = "./a"
parameters = '{"type": "object", "properties": {}}'

[[tools]]
name = "tool_b"
description = "Second tool"
command = "./b"
parameters = '{"type": "object", "properties": {}}'
"#;
    let manifest: PluginManifest = toml::from_str(toml_str).unwrap();
    assert_eq!(manifest.tools.len(), 2);
    assert_eq!(manifest.tools[0].name, "tool_a");
    assert_eq!(manifest.tools[1].name, "tool_b");
}

#[test]
fn parse_manifest_no_tools() {
    let toml_str = r#"
name = "empty"
version = "0.1.0"
description = "No tools"
"#;
    let manifest: PluginManifest = toml::from_str(toml_str).unwrap();
    assert_eq!(manifest.tools.len(), 0);
}

#[test]
fn parse_manifest_minimal_tool() {
    let toml_str = r#"
name = "minimal"
version = "0.1.0"
description = "Minimal tool"

[[tools]]
name = "my_tool"
description = "A tool"
command = "./run"
parameters = '{"type": "object", "properties": {}}'
"#;
    let manifest: PluginManifest = toml::from_str(toml_str).unwrap();
    let tool = &manifest.tools[0];
    assert_eq!(tool.needs_confirm, None);
    assert_eq!(tool.timeout, None);
}

#[test]
fn discover_plugin_from_user_dir() {
    let user_dir = TempDir::new().unwrap();
    write_plugin(user_dir.path(), "hello", &minimal_manifest("hello"));

    let plugins = discover_plugins("/nonexistent", &[user_dir.path().to_path_buf()], None);
    assert_eq!(plugins.len(), 1);
    assert_eq!(plugins[0].manifest.name, "hello");
    assert_eq!(plugins[0].manifest.tools.len(), 1);
    assert_eq!(plugins[0].manifest.tools[0].name, "hello_tool");
}

#[test]
fn discover_plugin_from_project_dir() {
    let project_dir = TempDir::new().unwrap();
    let plugins_dir = project_dir.path().join(".agents").join("plugins");
    write_plugin(&plugins_dir, "proj_plugin", &minimal_manifest("proj_plugin"));

    let plugins = discover_plugins(
        project_dir.path().to_str().unwrap(),
        &[],
        None,
    );
    assert_eq!(plugins.len(), 1);
    assert_eq!(plugins[0].manifest.name, "proj_plugin");
}

#[test]
fn project_plugin_overrides_user() {
    let user_dir = TempDir::new().unwrap();
    let project_dir = TempDir::new().unwrap();

    // User plugin "shared"
    write_plugin(
        user_dir.path(),
        "shared",
        &format!(
            r#"
name = "shared"
version = "1.0.0"
description = "User version"

[[tools]]
name = "shared_tool"
description = "From user"
command = "./user_run"
parameters = '{{"type": "object", "properties": {{}}}}'
"#
        ),
    );

    // Project plugin "shared" (should win)
    let plugins_dir = project_dir.path().join(".agents").join("plugins");
    write_plugin(
        &plugins_dir,
        "shared",
        &format!(
            r#"
name = "shared"
version = "2.0.0"
description = "Project version"

[[tools]]
name = "shared_tool"
description = "From project"
command = "./project_run"
parameters = '{{"type": "object", "properties": {{}}}}'
"#
        ),
    );

    let plugins = discover_plugins(
        project_dir.path().to_str().unwrap(),
        &[user_dir.path().to_path_buf()],
        None,
    );

    // Should have exactly one "shared" plugin (project version wins)
    let shared: Vec<_> = plugins.iter().filter(|p| p.manifest.name == "shared").collect();
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].manifest.version, "2.0.0");
    assert_eq!(shared[0].manifest.description, "Project version");
}

#[test]
fn invalid_manifest_skipped() {
    let user_dir = TempDir::new().unwrap();

    // Valid plugin
    write_plugin(user_dir.path(), "good", &minimal_manifest("good"));

    // Invalid plugin (bad TOML)
    let bad_dir = user_dir.path().join("bad");
    fs::create_dir_all(&bad_dir).unwrap();
    fs::write(bad_dir.join("PLUGIN.toml"), "this is not valid toml {{{").unwrap();

    let plugins = discover_plugins("/nonexistent", &[user_dir.path().to_path_buf()], None);

    // Only the valid plugin should be discovered
    assert_eq!(plugins.len(), 1);
    assert_eq!(plugins[0].manifest.name, "good");
}

#[test]
fn plugin_disabled_via_config() {
    let user_dir = TempDir::new().unwrap();
    write_plugin(user_dir.path(), "hello", &minimal_manifest("hello"));
    write_plugin(user_dir.path(), "world", &minimal_manifest("world"));

    let mut plugin_flags = std::collections::HashMap::new();
    plugin_flags.insert("hello".to_string(), toml::Value::Boolean(false));
    let config = ollama_code::config::Config {
        plugins: Some(plugin_flags),
        ..Default::default()
    };

    let plugins = discover_plugins(
        "/nonexistent",
        &[user_dir.path().to_path_buf()],
        Some(&config),
    );

    // "hello" should be filtered out
    let names: Vec<_> = plugins.iter().map(|p| p.manifest.name.as_str()).collect();
    assert!(!names.contains(&"hello"), "hello should be disabled: {:?}", names);
    assert!(names.contains(&"world"), "world should be present: {:?}", names);
}

#[test]
fn discover_merges_user_and_project() {
    let user_dir = TempDir::new().unwrap();
    let project_dir = TempDir::new().unwrap();

    write_plugin(user_dir.path(), "user_only", &minimal_manifest("user_only"));

    let plugins_dir = project_dir.path().join(".agents").join("plugins");
    write_plugin(&plugins_dir, "proj_only", &minimal_manifest("proj_only"));

    let plugins = discover_plugins(
        project_dir.path().to_str().unwrap(),
        &[user_dir.path().to_path_buf()],
        None,
    );

    let names: Vec<_> = plugins.iter().map(|p| p.manifest.name.as_str()).collect();
    assert!(names.contains(&"user_only"));
    assert!(names.contains(&"proj_only"));
    assert_eq!(plugins.len(), 2);
}
