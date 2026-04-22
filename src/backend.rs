use std::future::Future;
use std::pin::Pin;

use anyhow::Result;
use serde::Deserialize;
use serde_json::Value;

use crate::message::{Message, ToolCall};

/// Response from a chat completion request.
pub struct ChatResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub prompt_eval_count: u64,
    /// Timing metrics (nanoseconds).
    pub prompt_eval_duration: u64,
    pub eval_count: u64,
    pub eval_duration: u64,
    pub load_duration: u64,
    pub total_duration: u64,
    /// True if the stream ended without receiving `done: true`.
    pub incomplete: bool,
    /// True if tool calls were extracted from text content rather than from
    /// the structured `tool_calls` field.
    pub tool_calls_from_content: bool,
    /// True if the response was cut short because repetitive/degenerate
    /// output was detected during streaming.
    pub repetition_detected: bool,
    /// Accumulated reasoning/thinking content from the model (separate channel
    /// from `content`). Empty when thinking wasn't requested or the model
    /// didn't emit any.
    pub thinking: String,
    /// True if the stream was aborted because accumulated thinking exceeded
    /// the configured budget.
    pub thinking_budget_exceeded: bool,
}

/// Model metadata returned by backend model listing.
#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    pub name: String,
}

/// Trait for model backends that can perform chat completions.
pub trait ModelBackend: Send + Sync {
    /// Send a chat completion request, streaming tokens via `on_token`.
    ///
    /// When `thinking_budget_tokens` is `Some(n)`, the backend requests a
    /// separate reasoning channel (`think: true` on Ollama) and aborts the
    /// stream mid-generation if accumulated thinking exceeds that budget. The
    /// returned `ChatResponse` carries `thinking_budget_exceeded: true` in
    /// that case. `None` disables thinking entirely.
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: &'a [Message],
        tools: Option<Vec<Value>>,
        num_ctx: Option<u64>,
        thinking_budget_tokens: Option<u64>,
        on_token: Box<dyn Fn(&str) + Send + 'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>>;
}
