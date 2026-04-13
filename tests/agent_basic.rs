mod harness;

use std::sync::Arc;

use harness::{run_agent, ConfirmStrategy, MockBackend, MockResponse};

#[tokio::test]
async fn simple_text_response() {
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text("Hello, world!"),
    ]));

    let result = run_agent(backend.clone(), "Hi", ConfirmStrategy::ApproveAll).await;

    assert!(result.is_done());
    assert_eq!(result.final_content(), "Hello, world!");
    assert!(result.tool_calls().is_empty());

    // Verify the backend received the right messages
    let calls = backend.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].model, "test-model");
    // Messages: system prompt + user message
    let user_msgs: Vec<_> = calls[0]
        .messages
        .iter()
        .filter(|m| matches!(m.role, ollama_code::message::Role::User))
        .collect();
    assert_eq!(user_msgs.len(), 1);
    assert_eq!(user_msgs[0].content, "Hi");
}

#[tokio::test]
async fn tool_call_read_file() {
    // Model asks to read a file, gets the result, then responds with text.
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call(
            "read",
            serde_json::json!({ "file_path": "/etc/hostname" }),
        ),
        MockResponse::text("Your hostname file contains the machine name."),
    ]));

    let result = run_agent(backend.clone(), "Read /etc/hostname", ConfirmStrategy::ApproveAll).await;

    assert!(result.is_done());
    assert_eq!(result.final_content(), "Your hostname file contains the machine name.");

    let tool_calls = result.tool_calls();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].0, "read");

    // Tool result should have been returned
    let tool_results = result.tool_results();
    assert_eq!(tool_results.len(), 1);
    assert_eq!(tool_results[0].0, "read");
    assert!(tool_results[0].2); // success

    // Backend should have been called twice: once for the tool call, once for the final response
    assert_eq!(backend.calls().len(), 2);
}

#[tokio::test]
async fn tool_call_denied() {
    // Model asks to run bash, user denies, model gives a text response.
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call(
            "bash",
            serde_json::json!({ "command": "rm -rf /" }),
        ),
        MockResponse::text("Understood, I won't run that command."),
    ]));

    let result = run_agent(backend.clone(), "Delete everything", ConfirmStrategy::DenyAll).await;

    assert!(result.is_done());

    let tool_results = result.tool_results();
    assert_eq!(tool_results.len(), 1);
    assert_eq!(tool_results[0].0, "bash");
    assert!(!tool_results[0].2); // denied = not success
    assert!(tool_results[0].1.contains("denied"));
}

#[tokio::test]
async fn multi_turn_tool_calls() {
    // Model makes two sequential tool calls, then gives final answer.
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call(
            "glob",
            serde_json::json!({ "pattern": "*.rs", "path": "/tmp/test" }),
        ),
        MockResponse::tool_call(
            "read",
            serde_json::json!({ "file_path": "/tmp/test/main.rs" }),
        ),
        MockResponse::text("I found and read the file."),
    ]));

    let result = run_agent(backend.clone(), "Find and read .rs files", ConfirmStrategy::ApproveAll).await;

    assert!(result.is_done());
    assert_eq!(result.final_content(), "I found and read the file.");

    let tool_calls = result.tool_calls();
    assert_eq!(tool_calls.len(), 2);
    assert_eq!(tool_calls[0].0, "glob");
    assert_eq!(tool_calls[1].0, "read");

    assert_eq!(backend.calls().len(), 3);
}

#[tokio::test]
async fn selective_confirm() {
    // Approve read but deny bash.
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call(
            "bash",
            serde_json::json!({ "command": "echo hi" }),
        ),
        // After bash is denied, model tries read instead
        MockResponse::tool_call(
            "read",
            serde_json::json!({ "file_path": "/etc/hostname" }),
        ),
        MockResponse::text("Done."),
    ]));

    let result = run_agent(
        backend.clone(),
        "Do something",
        ConfirmStrategy::Custom(Box::new(|name| name != "bash")),
    )
    .await;

    assert!(result.is_done());

    let tool_results = result.tool_results();
    assert_eq!(tool_results.len(), 2);
    // bash was denied
    assert_eq!(tool_results[0].0, "bash");
    assert!(!tool_results[0].2);
    // read succeeded
    assert_eq!(tool_results[1].0, "read");
    assert!(tool_results[1].2);
}

#[tokio::test]
async fn cancellation() {
    use std::sync::atomic::AtomicBool;

    let cancel = Arc::new(AtomicBool::new(true)); // pre-cancelled

    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text("This should not appear"),
    ]));

    let result = harness::run_agent_with_options(
        backend.clone(),
        "Hello",
        ConfirmStrategy::ApproveAll,
        Some(cancel),
    )
    .await;

    assert!(result.is_cancelled());
    // Backend should not have been called since we cancelled before the first turn
    assert_eq!(backend.calls().len(), 0);
}

#[tokio::test]
async fn bash_tool_execution() {
    // Model asks to run a simple bash command.
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::tool_call(
            "bash",
            serde_json::json!({ "command": "echo hello_from_test" }),
        ),
        MockResponse::text("The command printed hello_from_test."),
    ]));

    let result = run_agent(backend.clone(), "Run echo", ConfirmStrategy::ApproveAll).await;

    assert!(result.is_done());

    let tool_results = result.tool_results();
    assert_eq!(tool_results.len(), 1);
    assert_eq!(tool_results[0].0, "bash");
    assert!(tool_results[0].2); // success
    assert!(tool_results[0].1.contains("hello_from_test"));
}

#[tokio::test]
async fn streamed_tokens_arrive() {
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text("word1 word2 word3"),
    ]));

    let result = run_agent(backend, "Say words", ConfirmStrategy::ApproveAll).await;

    // Should have received individual token events
    let tokens: Vec<_> = result
        .events
        .iter()
        .filter_map(|e| match e {
            ollama_code::agent::AgentEvent::Token(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();

    // MockBackend splits on spaces, so we should get multiple tokens
    assert!(tokens.len() > 1, "Expected multiple token events, got {}", tokens.len());
    assert_eq!(result.final_content(), "word1 word2 word3");
}

#[tokio::test]
async fn messages_logged_correctly() {
    let backend = Arc::new(MockBackend::new(vec![
        MockResponse::text("Reply."),
    ]));

    let result = run_agent(backend, "Question?", ConfirmStrategy::ApproveAll).await;

    let logged = result.logged_messages();
    // Should have: system prompt, user message, assistant message
    let roles: Vec<_> = logged
        .iter()
        .map(|m| format!("{:?}", m.role))
        .collect();
    assert!(roles.contains(&"System".to_string()));
    assert!(roles.contains(&"User".to_string()));
    assert!(roles.contains(&"Assistant".to_string()));

    let user_msg = logged.iter().find(|m| matches!(m.role, ollama_code::message::Role::User)).unwrap();
    assert_eq!(user_msg.content, "Question?");

    let assistant_msg = logged.iter().find(|m| matches!(m.role, ollama_code::message::Role::Assistant)).unwrap();
    assert_eq!(assistant_msg.content, "Reply.");
}
