mod harness;

use std::sync::Arc;

use harness::{run_agent, run_agent_with_config, ConfirmStrategy, MockBackend, MockResponse};
use ollama_code::config::Config;
use ollama_code::message::Role;

// ── Empty response retries ──────────────────────────────────────────

#[tokio::test]
async fn empty_response_retried_then_succeeds() {
    // First two responses are empty, third is real
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text(""),
        MockResponse::text(""),
        MockResponse::text("Here is the answer."),
    ]));
    let result = run_agent(backend.clone(), "hello", ConfirmStrategy::ApproveAll).await;

    assert!(result.is_done());
    assert_eq!(result.final_content(), "Here is the answer.");
    // Backend should have been called 3 times
    assert_eq!(backend.calls().len(), 3);
}

#[tokio::test]
async fn empty_response_max_retries_exhausted() {
    // 11 empty responses — exceeds MAX_EMPTY_RETRIES (10)
    let responses: Vec<MockResponse> = (0..11).map(|_| MockResponse::text("")).collect();
    // The 11th empty should stop retrying and emit Done with empty content
    let backend = Arc::new(MockBackend::new(responses));
    let result = run_agent(backend.clone(), "hello", ConfirmStrategy::ApproveAll).await;

    assert!(result.is_done());
    // After max retries, the agent accepts the empty response
    let debug = result.debug_messages();
    let retry_msgs: Vec<_> = debug
        .iter()
        .filter(|m| m.contains("Empty response") || m.contains("retrying"))
        .collect();
    assert!(!retry_msgs.is_empty());
}

// ── Repetition detection ────────────────────────────────────────────

#[tokio::test]
async fn repetition_without_tool_calls_retries() {
    // First response: repetition detected with no tool calls → retry
    // Second response: normal
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text("garbage garbage garbage").with_repetition(),
        MockResponse::text("Clean response."),
    ]));
    let result = run_agent(backend.clone(), "hello", ConfirmStrategy::ApproveAll).await;

    assert!(result.is_done());
    assert_eq!(result.final_content(), "Clean response.");
    assert_eq!(backend.calls().len(), 2);
    // Should have a debug message about repetition
    let debug = result.debug_messages();
    assert!(debug.iter().any(|m| m.contains("Repetition detected")));
}

#[tokio::test]
async fn repetition_with_tool_calls_keeps_tools() {
    // Response has repetition but also a tool call — content cleared, tools kept
    let mut response = MockResponse::tool_call(
        "glob",
        serde_json::json!({"pattern": "**/*.rs"}),
    );
    response.content = "repeating repeating repeating".to_string();
    response.repetition_detected = true;

    let backend = Arc::new(MockBackend::new(vec![
        response,
        MockResponse::text("Found some files."),
    ]));
    let result = run_agent(backend.clone(), "find rust files", ConfirmStrategy::ApproveAll).await;

    assert!(result.is_done());
    // The glob tool should have been called
    let tool_calls = result.tool_calls();
    assert!(tool_calls.iter().any(|(name, _)| *name == "glob"));
    // Content should have been replaced (cleared)
    assert!(result.final_content().contains("Found some files"));
}

// ── Incomplete stream ───────────────────────────────────────────────

#[tokio::test]
async fn incomplete_empty_response_retries() {
    // Incomplete stream with no content → retry
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text("").with_incomplete(),
        MockResponse::text("Recovery response."),
    ]));
    let result = run_agent(backend.clone(), "hello", ConfirmStrategy::ApproveAll).await;

    assert!(result.is_done());
    assert_eq!(result.final_content(), "Recovery response.");
    assert_eq!(backend.calls().len(), 2);
}

#[tokio::test]
async fn incomplete_with_content_emits_error_and_done() {
    // Incomplete stream with content → error message + done
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text("Partial respon").with_incomplete(),
    ]));
    let result = run_agent(backend.clone(), "hello", ConfirmStrategy::ApproveAll).await;

    assert!(result.is_done());
    let errors = result.errors();
    assert!(errors.iter().any(|e| e.contains("stream ended unexpectedly")));
}

// ── Context exhaustion ──────────────────────────────────────────────

// ── Thinking budget ─────────────────────────────────────────────────

#[tokio::test]
async fn thinking_budget_flips_sticky_off_and_injects_follow_up() {
    // Turn 1: model trips the budget (thinking_budget_exceeded=true, no final answer).
    // Turn 2: backend should receive a synthetic "commit" user message,
    //         thinking_budget_tokens is None (sticky off), and the model
    //         answers normally.
    let mut first = MockResponse::text("Partial reasoning leaked into content")
        .with_thinking_budget_exceeded("long-winded reasoning that blew the budget");
    // Force the harness to stream this content so we can verify the tagged
    // assistant message gets persisted.
    first.content = "Partial reasoning leaked into content".to_string();

    let backend = Arc::new(MockBackend::new(vec![
        first,
        MockResponse::text("Here is the committed answer."),
    ]));

    let config = Config {
        thinking_budget_tokens: Some(128),
        ..Default::default()
    };
    let result =
        run_agent_with_config(backend.clone(), "solve this", ConfirmStrategy::ApproveAll, &config).await;

    assert!(result.is_done());

    // The final assistant message logged to the session is the committed
    // answer from turn 2. (`final_content` concatenates all streamed tokens
    // including the partial from the aborted turn, which isn't what we care
    // about here.)
    let logged = result.logged_messages();
    let final_assistant = logged
        .iter()
        .rev()
        .find(|m| matches!(m.role, Role::Assistant))
        .expect("expected a final assistant message");
    assert_eq!(final_assistant.content, "Here is the committed answer.");

    // Two backend calls — one that tripped the budget, one after the nudge.
    let calls = backend.calls();
    assert_eq!(calls.len(), 2);

    // First call: budget set to Some(128).
    assert_eq!(calls[0].thinking_budget_tokens, Some(128));
    // Second call: sticky-off kicks in.
    assert_eq!(calls[1].thinking_budget_tokens, None);

    // Second call's message history includes the partial assistant (tagged)
    // and the synthetic user nudge right before the fresh turn.
    let msgs = &calls[1].messages;
    assert!(
        msgs.iter().any(|m| matches!(m.role, Role::Assistant)
            && m.content.contains("[auto: thinking budget exceeded")),
        "expected tagged partial assistant message in history",
    );
    assert!(
        msgs.iter().any(|m| matches!(m.role, Role::User)
            && m.content.starts_with("[auto]")
            && m.content.contains("Commit to an implementation")),
        "expected synthetic 'commit' user message in history",
    );
}

#[tokio::test]
async fn thinking_budget_none_by_default_means_no_think_flag() {
    let backend = Arc::new(MockBackend::new(vec![MockResponse::text("Answer.")]));
    let result = run_agent(backend.clone(), "hello", ConfirmStrategy::ApproveAll).await;
    assert!(result.is_done());
    assert_eq!(backend.calls()[0].thinking_budget_tokens, None);
}

#[tokio::test]
async fn context_full_emits_error() {
    // prompt_eval_count + eval_count >= context_size → context full
    // Agent context_size is 8192 (set in harness)
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text("This response fills context")
            .with_prompt_eval_count(7000)
            .with_eval_count(2000),
    ]));
    let result = run_agent(backend.clone(), "hello", ConfirmStrategy::ApproveAll).await;

    assert!(result.is_done());
    let errors = result.errors();
    assert!(
        errors.iter().any(|e| e.contains("Context window is full")),
        "expected context full error, got: {:?}",
        errors
    );
}

// ── Context trimming ────────────────────────────────────────────────
// The agent auto-trims when prompt_eval_count exceeds threshold.
// Default threshold is 80% of context_size (8192 * 0.8 = 6553).

#[tokio::test]
async fn context_trimmed_when_threshold_exceeded() {
    // Need multiple turns to build up message history before trimming kicks in.
    // Turn 1: glob (low tokens) — builds up messages
    // Turn 2: read (low tokens) — more messages
    // Turn 3: glob again — now prompt_eval_count is high, should trigger trim
    // Turn 4: final response
    // Disable compaction so we test pure blind trimming.
    let config = Config {
        context_compaction: Some(false),
        ..Default::default()
    };
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.rs"}))
            .with_prompt_eval_count(2000),
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.toml"}))
            .with_prompt_eval_count(4000),
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.md"}))
            .with_prompt_eval_count(7000), // above 80% of 8192 → trim
        MockResponse::text("Done.").with_prompt_eval_count(4000),
    ]));
    let result = run_agent_with_config(backend.clone(), "search everything", ConfirmStrategy::ApproveAll, &config).await;

    assert!(result.is_done());
    let trimmed = result.events.iter().any(|e| {
        matches!(
            e,
            ollama_code::agent::AgentEvent::ContextTrimmed { .. }
        )
    });
    assert!(trimmed, "expected ContextTrimmed event");
}

#[tokio::test]
async fn context_not_trimmed_below_threshold() {
    // Multiple turns but prompt_eval_count stays well below threshold
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.rs"}))
            .with_prompt_eval_count(1000),
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.toml"}))
            .with_prompt_eval_count(2000),
        MockResponse::text("Done.").with_prompt_eval_count(2500),
    ]));
    let result = run_agent(backend.clone(), "search", ConfirmStrategy::ApproveAll).await;

    assert!(result.is_done());
    let trimmed = result.events.iter().any(|e| {
        matches!(
            e,
            ollama_code::agent::AgentEvent::ContextTrimmed { .. }
        )
    });
    assert!(!trimmed, "should not have trimmed context");
}

#[tokio::test]
async fn custom_trim_thresholds() {
    // Use a config with very low trim threshold (50%)
    // context_size = 8192, threshold = 50% = 4096
    let config = Config {
        trim_threshold: Some(50),
        trim_target: Some(30),
        context_compaction: Some(false),
        ..Default::default()
    };

    // Build up enough history, then exceed the 50% threshold
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.rs"}))
            .with_prompt_eval_count(2000),
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.toml"}))
            .with_prompt_eval_count(3000),
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.md"}))
            .with_prompt_eval_count(5000), // above 4096 threshold
        MockResponse::text("Done.").with_prompt_eval_count(2500),
    ]));
    let result =
        run_agent_with_config(backend.clone(), "search", ConfirmStrategy::ApproveAll, &config)
            .await;

    assert!(result.is_done());
    let trimmed = result.events.iter().any(|e| {
        matches!(
            e,
            ollama_code::agent::AgentEvent::ContextTrimmed { .. }
        )
    });
    assert!(trimmed, "expected trimming with low threshold");
}

// ── Context compaction ──────────────────────────────────────────────

#[tokio::test]
async fn context_compacted_when_enabled() {
    // With compaction enabled, exceeding the threshold triggers an LLM
    // summary call instead of blind trimming.
    // Response sequence:
    //   1-3: tool calls building up context
    //   4:   compaction summary (consumed by compact_context, tools=None)
    //   5:   final response (main loop resumes)
    let config = Config {
        context_compaction: Some(true),
        ..Default::default()
    };
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.rs"}))
            .with_prompt_eval_count(2000),
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.toml"}))
            .with_prompt_eval_count(4000),
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.md"}))
            .with_prompt_eval_count(7000), // above 80% of 8192 → compact
        // Compaction LLM call response:
        MockResponse::text("User searched for Rust, TOML, and Markdown files using glob. All searches completed successfully."),
        // Main loop continues:
        MockResponse::text("Done.").with_prompt_eval_count(4000),
    ]));
    let result =
        run_agent_with_config(backend.clone(), "search everything", ConfirmStrategy::ApproveAll, &config)
            .await;

    assert!(result.is_done());

    // Should have ContextCompacted, not ContextTrimmed
    let compacted = result.events.iter().any(|e| {
        matches!(e, ollama_code::agent::AgentEvent::ContextCompacted { .. })
    });
    assert!(compacted, "expected ContextCompacted event");

    let trimmed = result.events.iter().any(|e| {
        matches!(e, ollama_code::agent::AgentEvent::ContextTrimmed { .. })
    });
    assert!(!trimmed, "should not have ContextTrimmed when compaction succeeds");

    // Verify the compaction call was made without tools
    let calls = backend.calls();
    let compaction_call = calls.iter().find(|c| c.tools.is_none());
    assert!(compaction_call.is_some(), "expected a tools=None call for compaction");
}

#[tokio::test]
async fn context_compaction_fallback_on_empty_summary() {
    // When compaction returns an empty summary, fall back to blind trimming.
    let config = Config {
        context_compaction: Some(true),
        ..Default::default()
    };
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.rs"}))
            .with_prompt_eval_count(2000),
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.toml"}))
            .with_prompt_eval_count(4000),
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.md"}))
            .with_prompt_eval_count(7000), // triggers compaction
        // Compaction returns empty → fallback to blind trim
        MockResponse::text(""),
        MockResponse::text("Done.").with_prompt_eval_count(4000),
    ]));
    let result =
        run_agent_with_config(backend.clone(), "search everything", ConfirmStrategy::ApproveAll, &config)
            .await;

    assert!(result.is_done());

    // Should fall back to ContextTrimmed
    let trimmed = result.events.iter().any(|e| {
        matches!(e, ollama_code::agent::AgentEvent::ContextTrimmed { .. })
    });
    assert!(trimmed, "expected ContextTrimmed fallback when compaction returns empty");
}

#[tokio::test]
async fn context_compaction_disabled_uses_blind_trim() {
    // With compaction explicitly disabled, only blind trimming happens.
    let config = Config {
        context_compaction: Some(false),
        ..Default::default()
    };
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.rs"}))
            .with_prompt_eval_count(2000),
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.toml"}))
            .with_prompt_eval_count(4000),
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.md"}))
            .with_prompt_eval_count(7000),
        // No compaction response needed — goes straight to blind trim
        MockResponse::text("Done.").with_prompt_eval_count(4000),
    ]));
    let result =
        run_agent_with_config(backend.clone(), "search everything", ConfirmStrategy::ApproveAll, &config)
            .await;

    assert!(result.is_done());

    let trimmed = result.events.iter().any(|e| {
        matches!(e, ollama_code::agent::AgentEvent::ContextTrimmed { .. })
    });
    assert!(trimmed, "expected ContextTrimmed with compaction disabled");

    let compacted = result.events.iter().any(|e| {
        matches!(e, ollama_code::agent::AgentEvent::ContextCompacted { .. })
    });
    assert!(!compacted, "should not have ContextCompacted when disabled");
}

// ── Context nearly full with pending tool calls ─────────────────────

#[tokio::test]
async fn context_nearly_full_with_tool_calls() {
    // prompt_eval_count > 90% of context with tool calls → abort
    // 8192 * 90% = 7372
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.rs"}))
            .with_prompt_eval_count(7500)
            .with_eval_count(100),
    ]));
    let result = run_agent(backend.clone(), "search", ConfirmStrategy::ApproveAll).await;

    assert!(result.is_done());
    let errors = result.errors();
    assert!(
        errors.iter().any(|e| e.contains("Context window is full")),
        "expected context full error, got: {:?}",
        errors
    );
}

// ── Task re-injection ───────────────────────────────────────────────

#[tokio::test]
async fn task_reinjection_fires() {
    let config = Config {
        task_reinjection: Some(true),
        reinjection_interval: Some(2),
        ..Default::default()
    };

    // Need 3+ turns to trigger re-injection at turn 3
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.rs"})),
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.toml"})),
        MockResponse::text("All done."),
    ]));

    let result =
        run_agent_with_config(backend.clone(), "find all configs", ConfirmStrategy::ApproveAll, &config)
            .await;

    assert!(result.is_done());
    let debug = result.debug_messages();
    assert!(
        debug.iter().any(|m| m.contains("Task re-injection")),
        "expected task re-injection debug message, got: {:?}",
        debug
    );
}

// ── Tool scoping ────────────────────────────────────────────────────

#[tokio::test]
async fn tool_scoping_hides_mutation_tools_initially() {
    let config = Config {
        tool_scoping: Some(true),
        ..Default::default()
    };

    // First turn: model has no read/glob yet, so edit/write should be hidden
    // We'll verify by checking the tools passed to the backend
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text("I'll just respond."),
    ]));

    let result =
        run_agent_with_config(backend.clone(), "hello", ConfirmStrategy::ApproveAll, &config)
            .await;

    assert!(result.is_done());

    // Check the tools sent to the backend on the first call
    let calls = backend.calls();
    assert!(!calls.is_empty());
    let tool_names = calls[0].tool_names();

    // edit and write should be excluded
    assert!(!tool_names.contains(&"edit"), "edit should be hidden before exploration");
    assert!(!tool_names.contains(&"write"), "write should be hidden before exploration");
    // read, glob, grep, bash should still be present
    assert!(tool_names.contains(&"read"));
    assert!(tool_names.contains(&"glob"));
}

#[tokio::test]
async fn tool_scoping_shows_all_after_exploration() {
    let config = Config {
        tool_scoping: Some(true),
        ..Default::default()
    };

    // Turn 1: read a file (triggers exploration)
    // Turn 2: should now have edit/write available
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call("glob", serde_json::json!({"pattern": "*.rs"})),
        MockResponse::text("Found files, now I can edit."),
    ]));

    let result =
        run_agent_with_config(backend.clone(), "find and fix", ConfirmStrategy::ApproveAll, &config)
            .await;

    assert!(result.is_done());

    // Check the second call's tools — should include edit and write now
    let calls = backend.calls();
    assert!(calls.len() >= 2);
    let tool_names = calls[1].tool_names();

    assert!(tool_names.contains(&"edit"), "edit should be visible after exploration");
    assert!(tool_names.contains(&"write"), "write should be visible after exploration");
}
