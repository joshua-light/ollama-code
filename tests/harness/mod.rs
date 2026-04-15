#![allow(dead_code)]

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use serde_json::Value;
use tokio::sync::mpsc;

use ollama_code::agent::{Agent, AgentEvent};
use ollama_code::backend::{ChatResponse, ModelBackend};
use ollama_code::config::Config;
use ollama_code::message::{FunctionCall, Message, ToolCall};

// ---------------------------------------------------------------------------
// MockResponse builder
// ---------------------------------------------------------------------------

/// A scripted response the mock backend will return for one `chat()` call.
pub struct MockResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    /// Override prompt_eval_count (default: 100).
    pub prompt_eval_count: Option<u64>,
    /// Override eval_count (default: 50).
    pub eval_count: Option<u64>,
    /// Simulate an incomplete stream.
    pub incomplete: bool,
    /// Simulate repetition detection.
    pub repetition_detected: bool,
}

impl MockResponse {
    /// A plain text response with no tool calls.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            tool_calls: Vec::new(),
            prompt_eval_count: None,
            eval_count: None,
            incomplete: false,
            repetition_detected: false,
        }
    }

    /// A response that calls a single tool.
    pub fn tool_call(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: None,
                call_type: Some("function".to_string()),
                function: FunctionCall {
                    name: name.into(),
                    arguments,
                },
            }],
            prompt_eval_count: None,
            eval_count: None,
            incomplete: false,
            repetition_detected: false,
        }
    }

    /// A response with both text content and tool calls.
    pub fn text_and_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<(String, Value)>,
    ) -> Self {
        Self {
            content: content.into(),
            tool_calls: tool_calls
                .into_iter()
                .map(|(name, arguments)| ToolCall {
                    id: None,
                    call_type: Some("function".to_string()),
                    function: FunctionCall { name, arguments },
                })
                .collect(),
            prompt_eval_count: None,
            eval_count: None,
            incomplete: false,
            repetition_detected: false,
        }
    }

    /// Override the prompt_eval_count for this response.
    pub fn with_prompt_eval_count(mut self, count: u64) -> Self {
        self.prompt_eval_count = Some(count);
        self
    }

    /// Override the eval_count for this response.
    pub fn with_eval_count(mut self, count: u64) -> Self {
        self.eval_count = Some(count);
        self
    }

    /// Mark this response as an incomplete stream.
    pub fn with_incomplete(mut self) -> Self {
        self.incomplete = true;
        self
    }

    /// Mark this response as having repetition detected.
    pub fn with_repetition(mut self) -> Self {
        self.repetition_detected = true;
        self
    }
}

// ---------------------------------------------------------------------------
// MockBackend
// ---------------------------------------------------------------------------

/// A captured `chat()` invocation for assertions.
#[derive(Debug, Clone)]
pub struct CapturedCall {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Option<Vec<Value>>,
    pub num_ctx: Option<u64>,
}

impl CapturedCall {
    /// Extract tool names from the tools JSON sent to the backend.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.as_ref().map_or(Vec::new(), |tools| {
            tools
                .iter()
                .filter_map(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                })
                .collect()
        })
    }
}

/// Mock implementation of `ModelBackend` that returns scripted responses.
pub struct MockBackend {
    responses: Mutex<VecDeque<MockResponse>>,
    calls: Mutex<Vec<CapturedCall>>,
}

impl MockBackend {
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Retrieve all captured `chat()` calls for assertions.
    pub fn calls(&self) -> Vec<CapturedCall> {
        self.calls.lock().unwrap().clone()
    }
}

impl ModelBackend for MockBackend {
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: &'a [Message],
        tools: Option<Vec<Value>>,
        num_ctx: Option<u64>,
        on_token: Box<dyn Fn(&str) + Send + 'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        // Capture the call
        self.calls.lock().unwrap().push(CapturedCall {
            model: model.to_string(),
            messages: messages.to_vec(),
            tools: tools.clone(),
            num_ctx,
        });

        // Pop the next scripted response
        let response = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("MockBackend: no more scripted responses (test has more chat() calls than expected)");

        Box::pin(async move {
            // Simulate streaming: emit content tokens word-by-word
            if !response.content.is_empty() {
                for word in response.content.split_inclusive(' ') {
                    on_token(word);
                }
            }

            Ok(ChatResponse {
                content: response.content,
                tool_calls: response.tool_calls,
                prompt_eval_count: response.prompt_eval_count.unwrap_or(100),
                prompt_eval_duration: 1_000_000,
                eval_count: response.eval_count.unwrap_or(50),
                eval_duration: 1_000_000,
                load_duration: 0,
                total_duration: 2_000_000,
                incomplete: response.incomplete,
                tool_calls_from_content: false,
                repetition_detected: response.repetition_detected,
            })
        })
    }
}

// ---------------------------------------------------------------------------
// TestHarness
// ---------------------------------------------------------------------------

/// Collects agent events and provides assertion helpers.
pub struct TestResult {
    pub events: Vec<AgentEvent>,
}

impl TestResult {
    /// All tokens concatenated.
    pub fn streamed_content(&self) -> String {
        let mut out = String::new();
        for event in &self.events {
            if let AgentEvent::Token(t) = event {
                out.push_str(t);
            }
        }
        out
    }

    /// The final text content (accounts for ContentReplaced).
    pub fn final_content(&self) -> String {
        let mut buf = String::new();
        for event in &self.events {
            match event {
                AgentEvent::Token(t) => buf.push_str(t),
                AgentEvent::ContentReplaced(new) => buf = new.clone(),
                _ => {}
            }
        }
        buf
    }

    /// All tool calls that were dispatched, as (name, args) pairs.
    pub fn tool_calls(&self) -> Vec<(&str, &str)> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolCall { name, args } => Some((name.as_str(), args.as_str())),
                _ => None,
            })
            .collect()
    }

    /// All tool results, as (name, output, success) triples.
    pub fn tool_results(&self) -> Vec<(&str, &str, bool)> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolResult {
                    name,
                    output,
                    success,
                } => Some((name.as_str(), output.as_str(), *success)),
                _ => None,
            })
            .collect()
    }

    /// Whether the agent finished normally (emitted Done).
    pub fn is_done(&self) -> bool {
        self.events.iter().any(|e| matches!(e, AgentEvent::Done { .. }))
    }

    /// Whether the agent was cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.events.iter().any(|e| matches!(e, AgentEvent::Cancelled))
    }

    /// All error messages.
    pub fn errors(&self) -> Vec<&str> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::Error(msg) => Some(msg.as_str()),
                _ => None,
            })
            .collect()
    }

    /// All debug messages.
    pub fn debug_messages(&self) -> Vec<&str> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::Debug(msg) => Some(msg.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Get logged messages (excluding system prompt).
    pub fn logged_messages(&self) -> Vec<&Message> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::MessageLogged(msg) => Some(msg),
                _ => None,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Running the agent
// ---------------------------------------------------------------------------

/// Confirmation strategy for tool calls during tests.
pub enum ConfirmStrategy {
    /// Approve all tool calls automatically.
    ApproveAll,
    /// Deny all tool calls.
    DenyAll,
    /// Approve or deny based on the tool name. Returns true to approve.
    Custom(Box<dyn Fn(&str) -> bool + Send + 'static>),
}

/// Run the agent with a user prompt and collect all events.
///
/// The `MockBackend` is wrapped in an `Arc` so assertions can be made on
/// captured calls after the run.
pub async fn run_agent(
    backend: Arc<MockBackend>,
    input: &str,
    confirm: ConfirmStrategy,
) -> TestResult {
    run_agent_with_options(backend, input, confirm, None).await
}

/// Run with a custom `Config` (for testing feature flags, plugin config, etc.).
pub async fn run_agent_with_config(
    backend: Arc<MockBackend>,
    input: &str,
    confirm: ConfirmStrategy,
    config: &Config,
) -> TestResult {
    run_agent_full(backend, input, confirm, None, Some(config)).await
}

/// Run with additional configuration (cancel flag, etc.).
pub async fn run_agent_with_options(
    backend: Arc<MockBackend>,
    input: &str,
    confirm: ConfirmStrategy,
    cancel: Option<Arc<AtomicBool>>,
) -> TestResult {
    run_agent_full(backend, input, confirm, cancel, None).await
}

/// Internal: full-featured agent runner.
async fn run_agent_full(
    backend: Arc<MockBackend>,
    input: &str,
    confirm: ConfirmStrategy,
    cancel: Option<Arc<AtomicBool>>,
    config: Option<&Config>,
) -> TestResult {
    let mut agent = match config {
        Some(cfg) => Agent::with_config(
            backend.clone(),
            "test-model".to_string(),
            8192,
            std::time::Duration::from_secs(30),
            4,
            cfg,
        ),
        None => Agent::new(
            backend.clone(),
            "test-model".to_string(),
            8192,
            std::time::Duration::from_secs(30),
            4,
        ),
    };

    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let (confirm_tx, mut confirm_rx) = mpsc::unbounded_channel::<bool>();
    let cancel = cancel.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));

    // Spawn the event collector + auto-confirmer
    let collector = tokio::spawn(async move {
        let mut events = Vec::new();
        while let Some(event) = event_rx.recv().await {
            // Handle confirmation requests
            if let AgentEvent::ToolConfirmRequest { ref name, .. } = event {
                let approved = match &confirm {
                    ConfirmStrategy::ApproveAll => true,
                    ConfirmStrategy::DenyAll => false,
                    ConfirmStrategy::Custom(f) => f(name),
                };
                let _ = confirm_tx.send(approved);
            }
            events.push(event);
        }
        events
    });

    let input = input.to_string();
    let (_steer_tx, mut steer_rx) = mpsc::unbounded_channel::<String>();
    // Run the agent — this blocks until done
    let _ = agent.run(&input, &event_tx, &mut confirm_rx, &mut steer_rx, cancel).await;
    // Drop sender to signal collector to finish
    drop(event_tx);

    let events = collector.await.expect("event collector panicked");
    TestResult { events }
}
