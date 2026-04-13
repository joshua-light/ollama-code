mod harness;

use std::sync::Arc;

use harness::{run_agent, run_agent_with_config, ConfirmStrategy, MockBackend, MockResponse};
use ollama_code::agent::AgentEvent;
use ollama_code::config::Config;

// ── Subagent basic execution ────────────────────────────────────────

#[tokio::test]
async fn subagent_executes_and_returns_result() {
    // Turn 1: parent calls subagent tool
    // Turn 2 (subagent): subagent reads a file
    // Turn 3 (subagent): subagent returns text
    // Turn 4: parent uses subagent result to respond
    let backend = Arc::new(MockBackend::new(vec![
        // Parent turn 1: call subagent
        MockResponse::tool_call(
            "subagent",
            serde_json::json!({"task": "Read the README and summarize it"}),
        ),
        // Subagent turn 1: it decides to just answer (text response)
        MockResponse::text("The README describes a CLI tool for local AI agents."),
        // Parent turn 2: uses subagent result
        MockResponse::text("Based on the sub-agent's findings: it's a CLI tool."),
    ]));

    let result = run_agent(
        backend.clone(),
        "What does the README say?",
        ConfirmStrategy::ApproveAll,
    )
    .await;

    assert!(result.is_done());

    // Check subagent lifecycle events
    let starts: Vec<_> = result.events.iter().filter(|e| matches!(e, AgentEvent::SubagentStart { .. })).collect();
    assert_eq!(starts.len(), 1);

    let ends: Vec<_> = result.events.iter().filter(|e| matches!(e, AgentEvent::SubagentEnd { .. })).collect();
    assert_eq!(ends.len(), 1);

    // Parent should have received the subagent's response as tool output
    let tool_results = result.tool_results();
    let subagent_result = tool_results.iter().find(|(name, _, _)| *name == "subagent");
    assert!(subagent_result.is_some());
    let (_, output, success) = subagent_result.unwrap();
    assert!(*success);
    assert!(output.contains("CLI tool"));
}

// ── Subagent tool calls forwarded ───────────────────────────────────

#[tokio::test]
async fn subagent_tool_calls_produce_events() {
    let backend = Arc::new(MockBackend::new(vec![
        // Parent: call subagent
        MockResponse::tool_call(
            "subagent",
            serde_json::json!({"task": "Search for config files"}),
        ),
        // Subagent turn 1: calls glob
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.toml"})),
        // Subagent turn 2: returns result
        MockResponse::text("Found: Cargo.toml"),
        // Parent: responds
        MockResponse::text("The subagent found Cargo.toml."),
    ]));

    let result = run_agent(
        backend.clone(),
        "find config files",
        ConfirmStrategy::ApproveAll,
    )
    .await;

    assert!(result.is_done());

    // Check for SubagentToolCall events
    let sub_tool_calls: Vec<_> = result
        .events
        .iter()
        .filter(|e| matches!(e, AgentEvent::SubagentToolCall { .. }))
        .collect();
    assert!(!sub_tool_calls.is_empty(), "expected SubagentToolCall events");

    // Check for SubagentToolResult events
    let sub_tool_results: Vec<_> = result
        .events
        .iter()
        .filter(|e| matches!(e, AgentEvent::SubagentToolResult { .. }))
        .collect();
    assert!(!sub_tool_results.is_empty(), "expected SubagentToolResult events");
}

// ── Subagent denied ─────────────────────────────────────────────────

#[tokio::test]
async fn subagent_denied_by_user() {
    let backend = Arc::new(MockBackend::new(vec![
        // Parent: call subagent
        MockResponse::tool_call(
            "subagent",
            serde_json::json!({"task": "do something risky"}),
        ),
        // Parent: responds after denial
        MockResponse::text("OK, I won't do that."),
    ]));

    let result = run_agent(
        backend.clone(),
        "do risky thing",
        ConfirmStrategy::DenyAll,
    )
    .await;

    assert!(result.is_done());

    // The subagent call should be denied
    let tool_results = result.tool_results();
    let subagent_result = tool_results.iter().find(|(name, _, _)| *name == "subagent");
    assert!(subagent_result.is_some());
    let (_, output, success) = subagent_result.unwrap();
    assert!(!success);
    assert!(output.contains("denied"));
}

// ── Subagent max turns enforced ─────────────────────────────────────

#[tokio::test]
async fn subagent_max_turns_enforced() {
    // Set max_turns very low
    let config = Config {
        subagent_max_turns: Some(2),
        ..Default::default()
    };

    let backend = Arc::new(MockBackend::new(vec![
        // Parent: call subagent
        MockResponse::tool_call(
            "subagent",
            serde_json::json!({"task": "do many things"}),
        ),
        // Subagent turn 1: tool call
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.rs"})),
        // Subagent turn 2: another tool call (will hit limit)
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.toml"})),
        // Subagent forced final: summarize findings
        MockResponse::text("Found rust and toml files."),
        // Parent: responds
        MockResponse::text("The subagent found some files."),
    ]));

    let result =
        run_agent_with_config(backend.clone(), "search", ConfirmStrategy::ApproveAll, &config)
            .await;

    assert!(result.is_done());
    assert!(result.final_content().contains("subagent found"));
}

// ── Subagent not available to subagent (recursion prevention) ───────

#[tokio::test]
async fn subagent_tool_not_in_subagent_definitions() {
    // The subagent should not have the "subagent" tool in its tool list.
    // We verify this indirectly: the parent agent's tool list includes subagent,
    // but the subagent's calls should not include it in the tools field.
    let backend = Arc::new(MockBackend::new(vec![
        // Parent: call subagent
        MockResponse::tool_call(
            "subagent",
            serde_json::json!({"task": "help me"}),
        ),
        // Subagent: just responds (we check its tools below)
        MockResponse::text("Here to help!"),
        // Parent: final
        MockResponse::text("Got help from subagent."),
    ]));

    let result = run_agent(
        backend.clone(),
        "get help",
        ConfirmStrategy::ApproveAll,
    )
    .await;

    assert!(result.is_done());

    let calls = backend.calls();
    // First call is parent (has subagent tool)
    // Second call is subagent (should NOT have subagent tool)
    assert!(calls.len() >= 2);

    let parent_tool_names = calls[0].tool_names();
    assert!(parent_tool_names.contains(&"subagent"), "parent should have subagent tool");

    let subagent_tool_names = calls[1].tool_names();
    assert!(!subagent_tool_names.contains(&"subagent"), "subagent should NOT have subagent tool");
}
