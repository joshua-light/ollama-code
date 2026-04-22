use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::backend::ChatResponse;
use crate::message::{FunctionCall, ToolCall};

use super::parse::{extract_tool_calls_from_content, strip_special_tokens};

// --- Ollama native format types ---

#[derive(Debug, Deserialize)]
pub(super) struct ChatChunk {
    pub message: Option<ChunkMessage>,
    #[serde(default)]
    pub done: bool,
    pub prompt_eval_count: Option<u64>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<u64>,
    pub eval_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub total_duration: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ChunkMessage {
    #[allow(dead_code)]
    pub role: Option<String>,
    pub content: Option<String>,
    /// Ollama emits reasoning tokens here when `think: true` is set on the
    /// request. Kept separate from `content` so the budget can be counted
    /// without conflating it with the model's answer.
    pub thinking: Option<String>,
    pub tool_calls: Option<Vec<ToolCallResponse>>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ToolCallResponse {
    pub function: FunctionCallResponse,
}

#[derive(Debug, Deserialize)]
pub(super) struct FunctionCallResponse {
    pub name: String,
    pub arguments: Value,
}

// --- OpenAI SSE format types (used by llama-server's /api/chat) ---

#[derive(Debug, Deserialize)]
pub(super) struct OpenAIChunk {
    pub choices: Option<Vec<OpenAIChoice>>,
    pub usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
pub(super) struct OpenAIChoice {
    pub delta: Option<OpenAIDelta>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct OpenAIDelta {
    pub content: Option<String>,
    /// Reasoning channel in OpenAI-compatible streams (llama-server, Ollama
    /// GPT-OSS, etc.). Kept separate from `content` for the budget counter.
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
pub(super) struct OpenAIToolCallDelta {
    pub index: Option<usize>,
    pub function: Option<OpenAIFunctionDelta>,
}

#[derive(Debug, Deserialize)]
pub(super) struct OpenAIFunctionDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct OpenAIUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
}

/// State for accumulating streamed OpenAI tool calls (name and arguments
/// arrive in separate chunks).
pub(super) struct OpenAIToolCallAccum {
    pub name: String,
    pub arguments: String,
}

pub(super) struct TimingMetrics {
    pub prompt_eval_count: u64,
    pub prompt_eval_duration: u64,
    pub eval_count: u64,
    pub eval_duration: u64,
    pub load_duration: u64,
    pub total_duration: u64,
}

impl TimingMetrics {
    pub fn new() -> Self {
        Self {
            prompt_eval_count: 0,
            prompt_eval_duration: 0,
            eval_count: 0,
            eval_duration: 0,
            load_duration: 0,
            total_duration: 0,
        }
    }
}

pub(super) fn process_openai_chunk(
    parsed: &OpenAIChunk,
    content: &mut String,
    thinking: &mut String,
    tool_accum: &mut Vec<OpenAIToolCallAccum>,
    metrics: &mut TimingMetrics,
    on_token: &dyn Fn(&str),
) -> Option<String> {
    // finish_reason
    let mut finish = None;

    if let Some(choices) = &parsed.choices {
        for choice in choices {
            if let Some(delta) = &choice.delta {
                if let Some(text) = &delta.content {
                    if !text.is_empty() {
                        let cleaned = strip_special_tokens(text);
                        if !cleaned.is_empty() {
                            content.push_str(&cleaned);
                            on_token(&cleaned);
                        }
                    }
                }
                if let Some(reasoning) = &delta.reasoning_content {
                    if !reasoning.is_empty() {
                        thinking.push_str(reasoning);
                    }
                }
                if let Some(tool_calls) = &delta.tool_calls {
                    for (i, tc) in tool_calls.iter().enumerate() {
                        let idx = tc.index.unwrap_or(i);
                        // Grow accumulators as needed
                        while tool_accum.len() <= idx {
                            tool_accum.push(OpenAIToolCallAccum {
                                name: String::new(),
                                arguments: String::new(),
                            });
                        }
                        if let Some(func) = &tc.function {
                            if let Some(name) = &func.name {
                                tool_accum[idx].name = name.clone();
                            }
                            if let Some(args) = &func.arguments {
                                tool_accum[idx].arguments.push_str(args);
                            }
                        }
                    }
                }
            }
            if let Some(reason) = &choice.finish_reason {
                finish = Some(reason.clone());
            }
        }
    }

    if let Some(usage) = &parsed.usage {
        if let Some(n) = usage.prompt_tokens {
            metrics.prompt_eval_count = n;
        }
        if let Some(n) = usage.completion_tokens {
            metrics.eval_count = n;
        }
    }

    finish
}

pub(super) fn process_chunk(
    parsed: &ChatChunk,
    content: &mut String,
    thinking: &mut String,
    tool_calls: &mut Vec<ToolCall>,
    metrics: &mut TimingMetrics,
    on_token: &dyn Fn(&str),
) {
    if let Some(msg) = &parsed.message {
        if let Some(text) = &msg.content {
            if !text.is_empty() {
                let cleaned = strip_special_tokens(text);
                if !cleaned.is_empty() {
                    content.push_str(&cleaned);
                    on_token(&cleaned);
                }
            }
        }
        if let Some(reasoning) = &msg.thinking {
            if !reasoning.is_empty() {
                thinking.push_str(reasoning);
            }
        }
        if let Some(calls) = &msg.tool_calls {
            for call in calls {
                tool_calls.push(ToolCall {
                    id: None,
                    call_type: None,
                    function: FunctionCall {
                        name: call.function.name.clone(),
                        arguments: call.function.arguments.clone(),
                    },
                });
            }
        }
    }
    if parsed.done {
        if let Some(n) = parsed.prompt_eval_count {
            metrics.prompt_eval_count = n;
        }
        if let Some(n) = parsed.prompt_eval_duration {
            metrics.prompt_eval_duration = n;
        }
        if let Some(n) = parsed.eval_count {
            metrics.eval_count = n;
        }
        if let Some(n) = parsed.eval_duration {
            metrics.eval_duration = n;
        }
        if let Some(n) = parsed.load_duration {
            metrics.load_duration = n;
        }
        if let Some(n) = parsed.total_duration {
            metrics.total_duration = n;
        }
    }
}

/// Process any data remaining in the buffer after the streaming loop ends.
/// Returns `Ok(true)` if a done/finish signal was found in the remaining data.
#[allow(clippy::too_many_arguments)]
pub(super) fn process_remaining_buffer(
    buffer: &str,
    is_openai: Option<bool>,
    content: &mut String,
    thinking: &mut String,
    tool_calls: &mut Vec<ToolCall>,
    openai_tool_accum: &mut Vec<OpenAIToolCallAccum>,
    metrics: &mut TimingMetrics,
    on_token: &dyn Fn(&str),
) -> Result<bool> {
    let remaining = buffer.trim();
    if remaining.is_empty() || remaining.starts_with(':') {
        return Ok(false);
    }
    let mut saw_done = false;
    if is_openai == Some(true) {
        let data = remaining.strip_prefix("data:").unwrap_or(remaining).trim();
        if data != "[DONE]" && !data.is_empty() {
            let parsed: OpenAIChunk = serde_json::from_str(data)
                .with_context(|| {
                    format!("Failed to parse remaining OpenAI buffer: {}", data)
                })?;
            let finish = process_openai_chunk(
                &parsed,
                content,
                thinking,
                openai_tool_accum,
                metrics,
                on_token,
            );
            if finish.is_some() {
                saw_done = true;
            }
        } else {
            saw_done = true;
        }
    } else {
        let parsed: ChatChunk = serde_json::from_str(remaining)
            .with_context(|| {
                format!(
                    "Failed to parse remaining Ollama buffer ({} bytes): {}",
                    remaining.len(),
                    &remaining[..remaining.len().min(200)]
                )
            })?;
        process_chunk(
            &parsed,
            content,
            thinking,
            tool_calls,
            metrics,
            on_token,
        );
        if parsed.done {
            saw_done = true;
        }
    }
    Ok(saw_done)
}

/// Accumulated state from the streaming loop, passed to [`postprocess_response`].
pub(super) struct StreamResult {
    pub content: String,
    pub thinking: String,
    pub tool_calls: Vec<ToolCall>,
    pub openai_tool_accum: Vec<OpenAIToolCallAccum>,
    pub is_openai: bool,
    pub known_tool_names: Vec<String>,
    pub metrics: TimingMetrics,
    pub saw_done: bool,
    pub repetition_detected: bool,
    pub thinking_budget_exceeded: bool,
}

/// Convert accumulated state after streaming into a final [`ChatResponse`].
pub(super) fn postprocess_response(result: StreamResult) -> ChatResponse {
    let StreamResult {
        mut content,
        thinking,
        mut tool_calls,
        openai_tool_accum,
        is_openai,
        known_tool_names,
        metrics,
        saw_done,
        repetition_detected,
        thinking_budget_exceeded,
    } = result;
    // Convert accumulated OpenAI tool calls to our format
    if is_openai {
        for accum in &openai_tool_accum {
            if !accum.name.is_empty() {
                // Small models frequently emit argument fragments with trailing
                // commas, single quotes, or missing closing braces. Try strict
                // parsing first; on failure, run `repair_json` and retry before
                // giving up.
                let arguments: Value = super::parse::parse_json_lenient(&accum.arguments)
                    .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                tool_calls.push(ToolCall {
                    id: None,
                    call_type: None,
                    function: FunctionCall {
                        name: accum.name.clone(),
                        arguments,
                    },
                });
            }
        }
    }

    // Strip leaked special tokens (e.g. <|channel>, <|turn>) that some
    // models emit as plain text.
    let content_before = content.len();
    content = strip_special_tokens(&content);
    if content.len() != content_before {
        content = content.trim().to_string();
    }

    // Some models (e.g. Qwen) emit tool calls as plain-text JSON in the
    // content instead of using the structured tool_calls field.
    // When no structured calls were received, try to extract them from the
    // content as a fallback.
    let mut tool_calls_from_content = false;
    if tool_calls.is_empty() && !content.trim().is_empty() && !known_tool_names.is_empty() {
        let (extracted, remaining) =
            extract_tool_calls_from_content(&content, &known_tool_names);
        if !extracted.is_empty() {
            tool_calls = extracted;
            content = remaining;
            tool_calls_from_content = true;
        }
    }

    ChatResponse {
        content,
        tool_calls,
        prompt_eval_count: metrics.prompt_eval_count,
        prompt_eval_duration: metrics.prompt_eval_duration,
        eval_count: metrics.eval_count,
        eval_duration: metrics.eval_duration,
        load_duration: metrics.load_duration,
        total_duration: metrics.total_duration,
        incomplete: !saw_done,
        tool_calls_from_content,
        repetition_detected,
        thinking,
        thinking_budget_exceeded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Helper: no-op token callback
    fn noop_token(_: &str) {}

    // Helper: token-collecting callback
    fn collecting_token(collected: &std::cell::RefCell<Vec<String>>) -> impl Fn(&str) + '_ {
        move |t: &str| collected.borrow_mut().push(t.to_string())
    }

    // Test-local wrappers so existing test call sites don't need to thread a
    // discarded thinking buffer. The thinking channel itself has dedicated
    // tests further down.
    fn process_chunk(
        parsed: &ChatChunk,
        content: &mut String,
        tool_calls: &mut Vec<ToolCall>,
        metrics: &mut TimingMetrics,
        on_token: &dyn Fn(&str),
    ) {
        let mut thinking = String::new();
        super::process_chunk(parsed, content, &mut thinking, tool_calls, metrics, on_token);
    }

    fn process_openai_chunk(
        parsed: &OpenAIChunk,
        content: &mut String,
        tool_accum: &mut Vec<OpenAIToolCallAccum>,
        metrics: &mut TimingMetrics,
        on_token: &dyn Fn(&str),
    ) -> Option<String> {
        let mut thinking = String::new();
        super::process_openai_chunk(
            parsed, content, &mut thinking, tool_accum, metrics, on_token,
        )
    }

    fn process_remaining_buffer(
        buffer: &str,
        is_openai: Option<bool>,
        content: &mut String,
        tool_calls: &mut Vec<ToolCall>,
        openai_tool_accum: &mut Vec<OpenAIToolCallAccum>,
        metrics: &mut TimingMetrics,
        on_token: &dyn Fn(&str),
    ) -> Result<bool> {
        let mut thinking = String::new();
        super::process_remaining_buffer(
            buffer,
            is_openai,
            content,
            &mut thinking,
            tool_calls,
            openai_tool_accum,
            metrics,
            on_token,
        )
    }

    // ---------------------------------------------------------------
    // process_chunk (Ollama native format)
    // ---------------------------------------------------------------

    #[test]
    fn process_chunk_content_token() {
        let chunk: ChatChunk = serde_json::from_str(
            r#"{"message": {"role": "assistant", "content": "Hello"}, "done": false}"#,
        )
        .unwrap();

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_chunk(&chunk, &mut content, &mut tool_calls, &mut metrics, &noop_token);

        assert_eq!(content, "Hello");
        assert!(tool_calls.is_empty());
    }

    #[test]
    fn process_chunk_multiple_tokens() {
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = TimingMetrics::new();

        for text in &["Hello", ", ", "world", "!"] {
            let json = format!(
                r#"{{"message": {{"role": "assistant", "content": "{}"}}, "done": false}}"#,
                text
            );
            let chunk: ChatChunk = serde_json::from_str(&json).unwrap();
            process_chunk(&chunk, &mut content, &mut tool_calls, &mut metrics, &noop_token);
        }

        assert_eq!(content, "Hello, world!");
    }

    #[test]
    fn process_chunk_empty_content() {
        let chunk: ChatChunk = serde_json::from_str(
            r#"{"message": {"role": "assistant", "content": ""}, "done": false}"#,
        )
        .unwrap();

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_chunk(&chunk, &mut content, &mut tool_calls, &mut metrics, &noop_token);

        assert_eq!(content, "");
    }

    #[test]
    fn process_chunk_no_message() {
        let chunk: ChatChunk =
            serde_json::from_str(r#"{"done": false}"#).unwrap();

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_chunk(&chunk, &mut content, &mut tool_calls, &mut metrics, &noop_token);

        assert_eq!(content, "");
        assert!(tool_calls.is_empty());
    }

    #[test]
    fn process_chunk_tool_call() {
        let chunk: ChatChunk = serde_json::from_str(
            r#"{"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read", "arguments": {"file_path": "test.rs"}}}]}, "done": false}"#,
        ).unwrap();

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_chunk(&chunk, &mut content, &mut tool_calls, &mut metrics, &noop_token);

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "read");
        assert_eq!(
            tool_calls[0].function.arguments,
            json!({"file_path": "test.rs"})
        );
    }

    #[test]
    fn process_chunk_multiple_tool_calls() {
        let chunk: ChatChunk = serde_json::from_str(
            r#"{"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "read", "arguments": {"file_path": "a.rs"}}},
                {"function": {"name": "bash", "arguments": {"command": "ls"}}}
            ]}, "done": false}"#,
        ).unwrap();

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_chunk(&chunk, &mut content, &mut tool_calls, &mut metrics, &noop_token);

        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].function.name, "read");
        assert_eq!(tool_calls[1].function.name, "bash");
    }

    #[test]
    fn process_chunk_done_with_metrics() {
        let chunk: ChatChunk = serde_json::from_str(
            r#"{
                "message": {"role": "assistant", "content": ""},
                "done": true,
                "prompt_eval_count": 100,
                "prompt_eval_duration": 500000000,
                "eval_count": 50,
                "eval_duration": 1000000000,
                "load_duration": 200000000,
                "total_duration": 1700000000
            }"#,
        )
        .unwrap();

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_chunk(&chunk, &mut content, &mut tool_calls, &mut metrics, &noop_token);

        assert_eq!(metrics.prompt_eval_count, 100);
        assert_eq!(metrics.prompt_eval_duration, 500000000);
        assert_eq!(metrics.eval_count, 50);
        assert_eq!(metrics.eval_duration, 1000000000);
        assert_eq!(metrics.load_duration, 200000000);
        assert_eq!(metrics.total_duration, 1700000000);
    }

    #[test]
    fn process_chunk_not_done_ignores_metrics() {
        let chunk: ChatChunk = serde_json::from_str(
            r#"{
                "message": {"role": "assistant", "content": "hi"},
                "done": false,
                "eval_count": 999
            }"#,
        )
        .unwrap();

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_chunk(&chunk, &mut content, &mut tool_calls, &mut metrics, &noop_token);

        // Metrics should remain at default (0) since done=false
        assert_eq!(metrics.eval_count, 0);
    }

    #[test]
    fn process_chunk_strips_special_tokens() {
        let chunk: ChatChunk = serde_json::from_str(
            r#"{"message": {"role": "assistant", "content": "<|think|>Hello"}, "done": false}"#,
        )
        .unwrap();

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_chunk(&chunk, &mut content, &mut tool_calls, &mut metrics, &noop_token);

        assert_eq!(content, "Hello");
    }

    #[test]
    fn process_chunk_special_token_only_content() {
        // Content is only a special token -- should become empty, nothing appended
        let chunk: ChatChunk = serde_json::from_str(
            r#"{"message": {"role": "assistant", "content": "<|end|>"}, "done": false}"#,
        )
        .unwrap();

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_chunk(&chunk, &mut content, &mut tool_calls, &mut metrics, &noop_token);

        assert_eq!(content, "");
    }

    #[test]
    fn process_chunk_calls_on_token() {
        let chunk: ChatChunk = serde_json::from_str(
            r#"{"message": {"role": "assistant", "content": "Hi"}, "done": false}"#,
        )
        .unwrap();

        let collected = std::cell::RefCell::new(Vec::<String>::new());
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_chunk(
            &chunk,
            &mut content,
            &mut tool_calls,
            &mut metrics,
            &collecting_token(&collected),
        );

        assert_eq!(*collected.borrow(), vec!["Hi".to_string()]);
    }

    // ---------------------------------------------------------------
    // process_openai_chunk
    // ---------------------------------------------------------------

    #[test]
    fn process_openai_chunk_content_token() {
        let parsed: OpenAIChunk = serde_json::from_str(
            r#"{"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}"#,
        )
        .unwrap();

        let mut content = String::new();
        let mut tool_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        let finish =
            process_openai_chunk(&parsed, &mut content, &mut tool_accum, &mut metrics, &noop_token);

        assert_eq!(content, "Hello");
        assert!(finish.is_none());
    }

    #[test]
    fn process_openai_chunk_empty_content() {
        let parsed: OpenAIChunk = serde_json::from_str(
            r#"{"choices": [{"delta": {"content": ""}, "finish_reason": null}]}"#,
        )
        .unwrap();

        let mut content = String::new();
        let mut tool_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_openai_chunk(&parsed, &mut content, &mut tool_accum, &mut metrics, &noop_token);

        assert_eq!(content, "");
    }

    #[test]
    fn process_openai_chunk_finish_reason() {
        let parsed: OpenAIChunk = serde_json::from_str(
            r#"{"choices": [{"delta": {}, "finish_reason": "stop"}]}"#,
        )
        .unwrap();

        let mut content = String::new();
        let mut tool_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        let finish =
            process_openai_chunk(&parsed, &mut content, &mut tool_accum, &mut metrics, &noop_token);

        assert_eq!(finish, Some("stop".to_string()));
    }

    #[test]
    fn process_openai_chunk_tool_call_name_then_args() {
        // First chunk: tool call with name
        let chunk1: OpenAIChunk = serde_json::from_str(
            r#"{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "read", "arguments": ""}}]}, "finish_reason": null}]}"#,
        ).unwrap();

        // Second chunk: arguments fragment
        let chunk2: OpenAIChunk = serde_json::from_str(
            r#"{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"file_"}}]}, "finish_reason": null}]}"#,
        ).unwrap();

        // Third chunk: more arguments
        let chunk3: OpenAIChunk = serde_json::from_str(
            r#"{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "path\": \"test.rs\"}"}}]}, "finish_reason": null}]}"#,
        ).unwrap();

        let mut content = String::new();
        let mut tool_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_openai_chunk(&chunk1, &mut content, &mut tool_accum, &mut metrics, &noop_token);
        assert_eq!(tool_accum.len(), 1);
        assert_eq!(tool_accum[0].name, "read");
        assert_eq!(tool_accum[0].arguments, "");

        process_openai_chunk(&chunk2, &mut content, &mut tool_accum, &mut metrics, &noop_token);
        assert_eq!(tool_accum[0].arguments, r#"{"file_"#);

        process_openai_chunk(&chunk3, &mut content, &mut tool_accum, &mut metrics, &noop_token);
        assert_eq!(tool_accum[0].arguments, r#"{"file_path": "test.rs"}"#);
    }

    #[test]
    fn process_openai_chunk_multiple_tool_calls() {
        // Two tool calls arriving in the same chunk
        let chunk: OpenAIChunk = serde_json::from_str(
            r#"{"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"name": "read", "arguments": "{\"file_path\": \"a.rs\"}"}},
                {"index": 1, "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}
            ]}, "finish_reason": null}]}"#,
        ).unwrap();

        let mut content = String::new();
        let mut tool_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_openai_chunk(&chunk, &mut content, &mut tool_accum, &mut metrics, &noop_token);

        assert_eq!(tool_accum.len(), 2);
        assert_eq!(tool_accum[0].name, "read");
        assert_eq!(tool_accum[1].name, "bash");
        assert_eq!(tool_accum[1].arguments, r#"{"command": "ls"}"#);
    }

    #[test]
    fn process_openai_chunk_tool_calls_arrive_separately() {
        // First tool call in one chunk, second in next
        let chunk1: OpenAIChunk = serde_json::from_str(
            r#"{"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"name": "read", "arguments": "{\"file_path\": \"a.rs\"}"}}
            ]}, "finish_reason": null}]}"#,
        ).unwrap();

        let chunk2: OpenAIChunk = serde_json::from_str(
            r#"{"choices": [{"delta": {"tool_calls": [
                {"index": 1, "function": {"name": "bash", "arguments": "{\"command\": \"pwd\"}"}}
            ]}, "finish_reason": null}]}"#,
        ).unwrap();

        let mut content = String::new();
        let mut tool_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_openai_chunk(&chunk1, &mut content, &mut tool_accum, &mut metrics, &noop_token);
        assert_eq!(tool_accum.len(), 1);

        process_openai_chunk(&chunk2, &mut content, &mut tool_accum, &mut metrics, &noop_token);
        assert_eq!(tool_accum.len(), 2);
        assert_eq!(tool_accum[0].name, "read");
        assert_eq!(tool_accum[1].name, "bash");
    }

    #[test]
    fn process_openai_chunk_usage() {
        let parsed: OpenAIChunk = serde_json::from_str(
            r#"{"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 10}}"#,
        )
        .unwrap();

        let mut content = String::new();
        let mut tool_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_openai_chunk(&parsed, &mut content, &mut tool_accum, &mut metrics, &noop_token);

        assert_eq!(metrics.prompt_eval_count, 42);
        assert_eq!(metrics.eval_count, 10);
    }

    #[test]
    fn process_openai_chunk_no_choices() {
        let parsed: OpenAIChunk =
            serde_json::from_str(r#"{"choices": null}"#).unwrap();

        let mut content = String::new();
        let mut tool_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        let finish =
            process_openai_chunk(&parsed, &mut content, &mut tool_accum, &mut metrics, &noop_token);

        assert!(finish.is_none());
        assert_eq!(content, "");
    }

    #[test]
    fn process_openai_chunk_strips_special_tokens() {
        let parsed: OpenAIChunk = serde_json::from_str(
            r#"{"choices": [{"delta": {"content": "<|turn|>Hello"}, "finish_reason": null}]}"#,
        )
        .unwrap();

        let mut content = String::new();
        let mut tool_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_openai_chunk(&parsed, &mut content, &mut tool_accum, &mut metrics, &noop_token);

        assert_eq!(content, "Hello");
    }

    #[test]
    fn process_openai_chunk_calls_on_token() {
        let parsed: OpenAIChunk = serde_json::from_str(
            r#"{"choices": [{"delta": {"content": "world"}, "finish_reason": null}]}"#,
        )
        .unwrap();

        let collected = std::cell::RefCell::new(Vec::<String>::new());
        let mut content = String::new();
        let mut tool_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_openai_chunk(
            &parsed,
            &mut content,
            &mut tool_accum,
            &mut metrics,
            &collecting_token(&collected),
        );

        assert_eq!(*collected.borrow(), vec!["world".to_string()]);
    }

    #[test]
    fn process_openai_chunk_index_defaults_to_enumerate() {
        // When index is missing, should use enumeration position
        let chunk: OpenAIChunk = serde_json::from_str(
            r#"{"choices": [{"delta": {"tool_calls": [
                {"function": {"name": "read", "arguments": "{}"}}
            ]}, "finish_reason": null}]}"#,
        ).unwrap();

        let mut content = String::new();
        let mut tool_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        process_openai_chunk(&chunk, &mut content, &mut tool_accum, &mut metrics, &noop_token);

        assert_eq!(tool_accum.len(), 1);
        assert_eq!(tool_accum[0].name, "read");
    }

    // ---------------------------------------------------------------
    // process_remaining_buffer
    // ---------------------------------------------------------------

    #[test]
    fn process_remaining_buffer_empty() {
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut openai_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        let result = process_remaining_buffer(
            "",
            None,
            &mut content,
            &mut tool_calls,
            &mut openai_accum,
            &mut metrics,
            &noop_token,
        )
        .unwrap();

        assert!(!result);
    }

    #[test]
    fn process_remaining_buffer_whitespace_only() {
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut openai_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        let result = process_remaining_buffer(
            "   \n  \t  ",
            None,
            &mut content,
            &mut tool_calls,
            &mut openai_accum,
            &mut metrics,
            &noop_token,
        )
        .unwrap();

        assert!(!result);
    }

    #[test]
    fn process_remaining_buffer_sse_comment() {
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut openai_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        let result = process_remaining_buffer(
            ": keepalive",
            None,
            &mut content,
            &mut tool_calls,
            &mut openai_accum,
            &mut metrics,
            &noop_token,
        )
        .unwrap();

        assert!(!result);
    }

    #[test]
    fn process_remaining_buffer_ollama_done() {
        let buffer = r#"{"message": {"role": "assistant", "content": " end"}, "done": true, "eval_count": 25}"#;
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut openai_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        let result = process_remaining_buffer(
            buffer,
            Some(false),
            &mut content,
            &mut tool_calls,
            &mut openai_accum,
            &mut metrics,
            &noop_token,
        )
        .unwrap();

        assert!(result); // saw_done = true
        assert_eq!(content, " end");
        assert_eq!(metrics.eval_count, 25);
    }

    #[test]
    fn process_remaining_buffer_ollama_not_done() {
        let buffer = r#"{"message": {"role": "assistant", "content": "partial"}, "done": false}"#;
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut openai_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        let result = process_remaining_buffer(
            buffer,
            Some(false),
            &mut content,
            &mut tool_calls,
            &mut openai_accum,
            &mut metrics,
            &noop_token,
        )
        .unwrap();

        assert!(!result); // done was false
        assert_eq!(content, "partial");
    }

    #[test]
    fn process_remaining_buffer_openai_done_marker() {
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut openai_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        let result = process_remaining_buffer(
            "data: [DONE]",
            Some(true),
            &mut content,
            &mut tool_calls,
            &mut openai_accum,
            &mut metrics,
            &noop_token,
        )
        .unwrap();

        assert!(result);
    }

    #[test]
    fn process_remaining_buffer_openai_chunk() {
        let buffer = r#"data: {"choices": [{"delta": {"content": "tail"}, "finish_reason": "stop"}]}"#;
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut openai_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        let result = process_remaining_buffer(
            buffer,
            Some(true),
            &mut content,
            &mut tool_calls,
            &mut openai_accum,
            &mut metrics,
            &noop_token,
        )
        .unwrap();

        assert!(result); // finish_reason "stop" => saw_done
        assert_eq!(content, "tail");
    }

    #[test]
    fn process_remaining_buffer_openai_no_prefix() {
        // OpenAI chunk without "data:" prefix
        let buffer = r#"{"choices": [{"delta": {"content": "x"}, "finish_reason": null}]}"#;
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut openai_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        let result = process_remaining_buffer(
            buffer,
            Some(true),
            &mut content,
            &mut tool_calls,
            &mut openai_accum,
            &mut metrics,
            &noop_token,
        )
        .unwrap();

        assert!(!result);
        assert_eq!(content, "x");
    }

    #[test]
    fn process_remaining_buffer_auto_detect_ollama() {
        // is_openai = None, should auto-detect as Ollama (no "data:" prefix)
        let buffer = r#"{"message": {"role": "assistant", "content": "hi"}, "done": true}"#;
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut openai_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        let result = process_remaining_buffer(
            buffer,
            None,
            &mut content,
            &mut tool_calls,
            &mut openai_accum,
            &mut metrics,
            &noop_token,
        )
        .unwrap();

        assert!(result);
        assert_eq!(content, "hi");
    }

    #[test]
    fn process_remaining_buffer_invalid_json_errors() {
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut openai_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        let result = process_remaining_buffer(
            "not valid json",
            Some(false),
            &mut content,
            &mut tool_calls,
            &mut openai_accum,
            &mut metrics,
            &noop_token,
        );

        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // postprocess_response
    // ---------------------------------------------------------------

    fn make_stream_result() -> StreamResult {
        StreamResult {
            content: String::new(),
            thinking: String::new(),
            tool_calls: Vec::new(),
            openai_tool_accum: Vec::new(),
            is_openai: false,
            known_tool_names: Vec::new(),
            metrics: TimingMetrics::new(),
            saw_done: true,
            repetition_detected: false,
            thinking_budget_exceeded: false,
        }
    }

    #[test]
    fn postprocess_simple_content() {
        let mut r = make_stream_result();
        r.content = "Hello, world!".to_string();

        let resp = postprocess_response(r);
        assert_eq!(resp.content, "Hello, world!");
        assert!(resp.tool_calls.is_empty());
        assert!(!resp.incomplete);
        assert!(!resp.tool_calls_from_content);
        assert!(!resp.repetition_detected);
    }

    #[test]
    fn postprocess_incomplete_when_no_done() {
        let mut r = make_stream_result();
        r.content = "partial".to_string();
        r.saw_done = false;

        let resp = postprocess_response(r);
        assert!(resp.incomplete);
    }

    #[test]
    fn postprocess_repetition_flag() {
        let mut r = make_stream_result();
        r.content = "stuff".to_string();
        r.repetition_detected = true;

        let resp = postprocess_response(r);
        assert!(resp.repetition_detected);
    }

    #[test]
    fn postprocess_ollama_tool_calls_pass_through() {
        let mut r = make_stream_result();
        r.tool_calls.push(ToolCall {
            id: None,
            call_type: None,
            function: FunctionCall {
                name: "read".to_string(),
                arguments: json!({"file_path": "test.rs"}),
            },
        });

        let resp = postprocess_response(r);
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].function.name, "read");
        assert!(!resp.tool_calls_from_content);
    }

    #[test]
    fn postprocess_openai_tool_accum_conversion() {
        let mut r = make_stream_result();
        r.is_openai = true;
        r.openai_tool_accum.push(OpenAIToolCallAccum {
            name: "bash".to_string(),
            arguments: r#"{"command": "ls"}"#.to_string(),
        });

        let resp = postprocess_response(r);
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].function.name, "bash");
        assert_eq!(resp.tool_calls[0].function.arguments, json!({"command": "ls"}));
    }

    #[test]
    fn postprocess_openai_multiple_tool_accum() {
        let mut r = make_stream_result();
        r.is_openai = true;
        r.openai_tool_accum.push(OpenAIToolCallAccum {
            name: "read".to_string(),
            arguments: r#"{"file_path": "a.rs"}"#.to_string(),
        });
        r.openai_tool_accum.push(OpenAIToolCallAccum {
            name: "bash".to_string(),
            arguments: r#"{"command": "pwd"}"#.to_string(),
        });

        let resp = postprocess_response(r);
        assert_eq!(resp.tool_calls.len(), 2);
        assert_eq!(resp.tool_calls[0].function.name, "read");
        assert_eq!(resp.tool_calls[1].function.name, "bash");
    }

    #[test]
    fn postprocess_openai_empty_name_skipped() {
        let mut r = make_stream_result();
        r.is_openai = true;
        r.openai_tool_accum.push(OpenAIToolCallAccum {
            name: String::new(),
            arguments: r#"{"x": 1}"#.to_string(),
        });

        let resp = postprocess_response(r);
        assert!(resp.tool_calls.is_empty());
    }

    #[test]
    fn postprocess_openai_invalid_args_default_to_empty_object() {
        let mut r = make_stream_result();
        r.is_openai = true;
        r.openai_tool_accum.push(OpenAIToolCallAccum {
            name: "test".to_string(),
            arguments: "not valid json".to_string(),
        });

        let resp = postprocess_response(r);
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].function.arguments, json!({}));
    }

    #[test]
    fn postprocess_strips_special_tokens_from_content() {
        let mut r = make_stream_result();
        r.content = "<|think|>Hello world<|end|>".to_string();

        let resp = postprocess_response(r);
        assert_eq!(resp.content, "Hello world");
    }

    #[test]
    fn postprocess_strips_and_trims() {
        let mut r = make_stream_result();
        r.content = "  <|turn|>  Hello  ".to_string();

        let resp = postprocess_response(r);
        // After stripping <|turn|> we get "    Hello  ", then trimmed
        assert_eq!(resp.content, "Hello");
    }

    #[test]
    fn postprocess_no_strip_no_trim() {
        // When no special tokens are present, content should NOT be trimmed
        let mut r = make_stream_result();
        r.content = "  Hello  ".to_string();

        let resp = postprocess_response(r);
        assert_eq!(resp.content, "  Hello  ");
    }

    #[test]
    fn postprocess_fallback_tool_extraction_from_content() {
        let mut r = make_stream_result();
        r.content =
            r#"Sure, here you go. {"name": "read", "arguments": {"file_path": "main.rs"}}"#
                .to_string();
        r.known_tool_names = vec!["read".to_string()];

        let resp = postprocess_response(r);
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].function.name, "read");
        assert!(resp.tool_calls_from_content);
        // The JSON should have been removed from the content
        assert!(!resp.content.contains(r#"{"name""#));
        assert!(resp.content.contains("Sure"));
    }

    #[test]
    fn postprocess_no_fallback_when_structured_calls_present() {
        // When structured tool calls exist, should NOT try content extraction
        let mut r = make_stream_result();
        r.content =
            r#"{"name": "bash", "arguments": {"command": "ls"}}"#.to_string();
        r.tool_calls.push(ToolCall {
            id: None,
            call_type: None,
            function: FunctionCall {
                name: "read".to_string(),
                arguments: json!({"file_path": "test.rs"}),
            },
        });
        r.known_tool_names = vec!["read".to_string(), "bash".to_string()];

        let resp = postprocess_response(r);
        // Only the structured call, not extracted from content
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].function.name, "read");
        assert!(!resp.tool_calls_from_content);
    }

    #[test]
    fn postprocess_no_fallback_when_known_tools_empty() {
        let mut r = make_stream_result();
        r.content =
            r#"{"name": "read", "arguments": {"file_path": "main.rs"}}"#.to_string();
        // No known tool names

        let resp = postprocess_response(r);
        assert!(resp.tool_calls.is_empty());
        assert!(!resp.tool_calls_from_content);
    }

    #[test]
    fn postprocess_no_fallback_when_content_empty() {
        let mut r = make_stream_result();
        r.content = "   ".to_string();
        r.known_tool_names = vec!["read".to_string()];

        let resp = postprocess_response(r);
        assert!(resp.tool_calls.is_empty());
        assert!(!resp.tool_calls_from_content);
    }

    #[test]
    fn postprocess_metrics_pass_through() {
        let mut r = make_stream_result();
        r.metrics.prompt_eval_count = 100;
        r.metrics.prompt_eval_duration = 500;
        r.metrics.eval_count = 50;
        r.metrics.eval_duration = 1000;
        r.metrics.load_duration = 200;
        r.metrics.total_duration = 1700;

        let resp = postprocess_response(r);
        assert_eq!(resp.prompt_eval_count, 100);
        assert_eq!(resp.prompt_eval_duration, 500);
        assert_eq!(resp.eval_count, 50);
        assert_eq!(resp.eval_duration, 1000);
        assert_eq!(resp.load_duration, 200);
        assert_eq!(resp.total_duration, 1700);
    }

    #[test]
    fn postprocess_openai_with_content_and_tools() {
        // OpenAI response with both content and tool calls
        let mut r = make_stream_result();
        r.is_openai = true;
        r.content = "I'll read that file for you.".to_string();
        r.openai_tool_accum.push(OpenAIToolCallAccum {
            name: "read".to_string(),
            arguments: r#"{"file_path": "test.rs"}"#.to_string(),
        });

        let resp = postprocess_response(r);
        assert_eq!(resp.content, "I'll read that file for you.");
        assert_eq!(resp.tool_calls.len(), 1);
        // Since structured calls exist, no fallback extraction
        assert!(!resp.tool_calls_from_content);
    }

    // ---------------------------------------------------------------
    // End-to-end: simulating a full streaming flow
    // ---------------------------------------------------------------

    #[test]
    fn end_to_end_ollama_text_response() {
        // Simulate receiving 3 content chunks + done chunk via Ollama format
        let chunks = vec![
            r#"{"message": {"role": "assistant", "content": "Hello"}, "done": false}"#,
            r#"{"message": {"role": "assistant", "content": ", "}, "done": false}"#,
            r#"{"message": {"role": "assistant", "content": "world!"}, "done": false}"#,
            r#"{"message": {"role": "assistant", "content": ""}, "done": true, "eval_count": 3}"#,
        ];

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = TimingMetrics::new();
        let mut saw_done = false;

        for chunk_str in chunks {
            let chunk: ChatChunk = serde_json::from_str(chunk_str).unwrap();
            process_chunk(&chunk, &mut content, &mut tool_calls, &mut metrics, &noop_token);
            if chunk.done {
                saw_done = true;
            }
        }

        let resp = postprocess_response(StreamResult {
            content,
            thinking: String::new(),
            tool_calls,
            openai_tool_accum: Vec::new(),
            is_openai: false,
            known_tool_names: Vec::new(),
            metrics,
            saw_done,
            repetition_detected: false,
            thinking_budget_exceeded: false,
        });

        assert_eq!(resp.content, "Hello, world!");
        assert!(resp.tool_calls.is_empty());
        assert!(!resp.incomplete);
        assert_eq!(resp.eval_count, 3);
    }

    #[test]
    fn end_to_end_openai_tool_call_incremental() {
        // Simulate OpenAI incremental tool call accumulation
        let chunks = vec![
            r#"{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "read"}}]}, "finish_reason": null}]}"#,
            r#"{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"file"}}]}, "finish_reason": null}]}"#,
            r#"{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "_path\": "}}]}, "finish_reason": null}]}"#,
            r#"{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"main.rs\"}"}}]}, "finish_reason": null}]}"#,
            r#"{"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 50, "completion_tokens": 8}}"#,
        ];

        let mut content = String::new();
        let mut tool_accum = Vec::new();
        let mut metrics = TimingMetrics::new();
        let mut saw_done = false;

        for chunk_str in chunks {
            let parsed: OpenAIChunk = serde_json::from_str(chunk_str).unwrap();
            let finish =
                process_openai_chunk(&parsed, &mut content, &mut tool_accum, &mut metrics, &noop_token);
            if finish.is_some() {
                saw_done = true;
            }
        }

        let resp = postprocess_response(StreamResult {
            content,
            thinking: String::new(),
            tool_calls: Vec::new(),
            openai_tool_accum: tool_accum,
            is_openai: true,
            known_tool_names: Vec::new(),
            metrics,
            saw_done,
            repetition_detected: false,
            thinking_budget_exceeded: false,
        });

        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].function.name, "read");
        assert_eq!(
            resp.tool_calls[0].function.arguments,
            json!({"file_path": "main.rs"})
        );
        assert!(!resp.incomplete);
        assert_eq!(resp.prompt_eval_count, 50);
        assert_eq!(resp.eval_count, 8);
    }

    // ---------------------------------------------------------------
    // Thinking channel
    // ---------------------------------------------------------------

    #[test]
    fn process_chunk_accumulates_thinking_separately() {
        // Ollama emits `message.thinking` when `think: true` is requested.
        // It must land in the thinking buffer, not in content, and it must
        // NOT call on_token (we don't stream reasoning to the UI).
        let chunk: ChatChunk = serde_json::from_str(
            r#"{"message": {"role": "assistant", "thinking": "let me reason about this", "content": ""}, "done": false}"#,
        )
        .unwrap();

        let collected = std::cell::RefCell::new(Vec::<String>::new());
        let mut content = String::new();
        let mut thinking = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = TimingMetrics::new();

        super::process_chunk(
            &chunk,
            &mut content,
            &mut thinking,
            &mut tool_calls,
            &mut metrics,
            &collecting_token(&collected),
        );

        assert_eq!(content, "");
        assert_eq!(thinking, "let me reason about this");
        assert!(
            collected.borrow().is_empty(),
            "on_token must not fire for thinking tokens",
        );
    }

    #[test]
    fn process_chunk_thinking_accumulates_across_chunks() {
        let mut content = String::new();
        let mut thinking = String::new();
        let mut tool_calls = Vec::new();
        let mut metrics = TimingMetrics::new();

        for text in &["Step 1. ", "Step 2. ", "Step 3."] {
            let json = format!(
                r#"{{"message": {{"role": "assistant", "thinking": "{}", "content": ""}}, "done": false}}"#,
                text
            );
            let chunk: ChatChunk = serde_json::from_str(&json).unwrap();
            super::process_chunk(
                &chunk,
                &mut content,
                &mut thinking,
                &mut tool_calls,
                &mut metrics,
                &noop_token,
            );
        }

        assert_eq!(thinking, "Step 1. Step 2. Step 3.");
        assert_eq!(content, "");
    }

    #[test]
    fn process_openai_chunk_accumulates_reasoning_content() {
        // llama-server and Ollama's OpenAI-compat mode emit reasoning under
        // `delta.reasoning_content`.
        let parsed: OpenAIChunk = serde_json::from_str(
            r#"{"choices": [{"delta": {"reasoning_content": "thinking...", "content": "answer"}, "finish_reason": null}]}"#,
        )
        .unwrap();

        let mut content = String::new();
        let mut thinking = String::new();
        let mut tool_accum = Vec::new();
        let mut metrics = TimingMetrics::new();

        super::process_openai_chunk(
            &parsed,
            &mut content,
            &mut thinking,
            &mut tool_accum,
            &mut metrics,
            &noop_token,
        );

        assert_eq!(content, "answer");
        assert_eq!(thinking, "thinking...");
    }

    #[test]
    fn postprocess_passes_through_thinking_fields() {
        let mut r = make_stream_result();
        r.thinking = "I reasoned but got cut off".to_string();
        r.thinking_budget_exceeded = true;

        let resp = postprocess_response(r);
        assert_eq!(resp.thinking, "I reasoned but got cut off");
        assert!(resp.thinking_budget_exceeded);
    }
}
