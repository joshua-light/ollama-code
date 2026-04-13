mod harness;

use std::collections::HashMap;
use std::sync::Arc;

use harness::{run_agent_with_config, ConfirmStrategy, MockBackend, MockResponse};
use ollama_code::config::Config;
use ollama_code::mcp::{McpServer, McpServerConfig};
use ollama_code::tools::Tool;

fn test_server_config() -> McpServerConfig {
    McpServerConfig {
        command: Some("python3".to_string()),
        args: vec![
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/mcp_test_server.py")
                .to_string(),
        ],
        env: HashMap::new(),
        url: None,
        headers: HashMap::new(),
        needs_confirm: false,
    }
}

// ---------------------------------------------------------------------------
// Unit-level: McpServer start / tool discovery
// ---------------------------------------------------------------------------

#[test]
fn mcp_server_starts_and_discovers_tools() {
    let server = McpServer::start("test", &test_server_config()).unwrap();
    assert_eq!(server.tools.len(), 2);

    let names: Vec<&str> = server.tools.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"echo"));
    assert!(names.contains(&"add"));
}

#[test]
fn mcp_server_creates_qualified_tool_names() {
    let server = McpServer::start("myserver", &test_server_config()).unwrap();
    let tools = server.create_tools();

    let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    assert!(names.contains(&"mcp__myserver__echo"));
    assert!(names.contains(&"mcp__myserver__add"));
}

#[test]
fn mcp_tool_execute_echo() {
    let server = McpServer::start("test", &test_server_config()).unwrap();
    let tools = server.create_tools();
    let echo_tool = tools.iter().find(|t| t.name().contains("echo")).unwrap();

    let result = echo_tool
        .execute(&serde_json::json!({"message": "hello world"}))
        .unwrap();
    assert_eq!(result, "hello world");
}

#[test]
fn mcp_tool_execute_add() {
    let server = McpServer::start("test", &test_server_config()).unwrap();
    let tools = server.create_tools();
    let add_tool = tools.iter().find(|t| t.name().contains("add")).unwrap();

    let result = add_tool
        .execute(&serde_json::json!({"a": 3, "b": 4}))
        .unwrap();
    assert_eq!(result, "7");
}

#[test]
fn mcp_tool_definition_has_schema() {
    let server = McpServer::start("test", &test_server_config()).unwrap();
    let tools = server.create_tools();
    let echo_tool = tools.iter().find(|t| t.name().contains("echo")).unwrap();
    let def = echo_tool.definition();

    assert_eq!(def.name, "mcp__test__echo");
    assert_eq!(def.description, "Echo back the input message");
    assert!(def.parameters["properties"]["message"].is_object());
}

#[test]
fn mcp_server_start_fails_for_bad_command() {
    let config = McpServerConfig {
        command: Some("/nonexistent/mcp-server-binary".to_string()),
        args: vec![],
        env: HashMap::new(),
        url: None,
        headers: HashMap::new(),
        needs_confirm: false,
    };
    let result = McpServer::start("bad", &config);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Agent integration: MCP tools in the agent loop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn agent_with_mcp_tools_end_to_end() {
    let mut mcp_servers = HashMap::new();
    mcp_servers.insert("testmcp".to_string(), test_server_config());

    let config = Config {
        mcp_servers: Some(mcp_servers),
        ..Default::default()
    };

    // Model calls the MCP echo tool, then gives final answer
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call(
            "mcp__testmcp__echo",
            serde_json::json!({"message": "ping"}),
        ),
        MockResponse::text("The server echoed: ping"),
    ]));

    let result = run_agent_with_config(
        backend.clone(),
        "Echo ping via MCP",
        ConfirmStrategy::ApproveAll,
        &config,
    )
    .await;

    assert!(result.is_done());
    assert_eq!(result.final_content(), "The server echoed: ping");

    let tool_results = result.tool_results();
    assert_eq!(tool_results.len(), 1);
    assert_eq!(tool_results[0].0, "mcp__testmcp__echo");
    assert!(tool_results[0].2, "tool should succeed");
    assert_eq!(tool_results[0].1, "ping");
}

#[tokio::test]
async fn agent_mcp_tools_appear_in_definitions() {
    let mut mcp_servers = HashMap::new();
    mcp_servers.insert("s".to_string(), test_server_config());

    let config = Config {
        mcp_servers: Some(mcp_servers),
        ..Default::default()
    };

    let backend = Arc::new(MockBackend::new(vec![MockResponse::text("ok")]));

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

    assert!(
        tool_names.contains(&"mcp__s__echo".to_string()),
        "MCP echo tool should be in definitions: {:?}",
        tool_names
    );
    assert!(
        tool_names.contains(&"mcp__s__add".to_string()),
        "MCP add tool should be in definitions: {:?}",
        tool_names
    );
}

#[tokio::test]
async fn agent_mcp_needs_confirm() {
    let mut confirm_config = test_server_config();
    confirm_config.needs_confirm = true;

    let mut mcp_servers = HashMap::new();
    mcp_servers.insert("guarded".to_string(), confirm_config);

    let config = Config {
        mcp_servers: Some(mcp_servers),
        ..Default::default()
    };

    // Model calls the MCP tool, user denies
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call(
            "mcp__guarded__echo",
            serde_json::json!({"message": "test"}),
        ),
        MockResponse::text("Denied."),
    ]));

    let result = run_agent_with_config(
        backend.clone(),
        "Try calling echo",
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

#[tokio::test]
async fn agent_mcp_server_disabled_via_config() {
    let mut mcp_servers = HashMap::new();
    mcp_servers.insert("disabled_server".to_string(), test_server_config());

    let mut plugins = HashMap::new();
    plugins.insert(
        "disabled_server".to_string(),
        toml::Value::Boolean(false),
    );

    let config = Config {
        mcp_servers: Some(mcp_servers),
        plugins: Some(plugins),
        ..Default::default()
    };

    let backend = Arc::new(MockBackend::new(vec![MockResponse::text("ok")]));

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

    // MCP tools should NOT be present since the server is disabled
    assert!(
        !tool_names.iter().any(|n| n.starts_with("mcp__disabled_server")),
        "Disabled server's tools should not appear: {:?}",
        tool_names
    );
}
