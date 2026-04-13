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
pub(super) fn process_remaining_buffer(
    buffer: &str,
    is_openai: Option<bool>,
    content: &mut String,
    tool_calls: &mut Vec<ToolCall>,
    openai_tool_accum: &mut Vec<OpenAIToolCallAccum>,
    metrics: &mut TimingMetrics,
    on_token: &dyn Fn(&str),
) -> Result<bool> {
    let remaining = buffer.trim();
    if remaining.is_empty() {
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
    pub tool_calls: Vec<ToolCall>,
    pub openai_tool_accum: Vec<OpenAIToolCallAccum>,
    pub is_openai: bool,
    pub known_tool_names: Vec<String>,
    pub metrics: TimingMetrics,
    pub saw_done: bool,
    pub repetition_detected: bool,
}

/// Convert accumulated state after streaming into a final [`ChatResponse`].
pub(super) fn postprocess_response(result: StreamResult) -> ChatResponse {
    let StreamResult {
        mut content,
        mut tool_calls,
        openai_tool_accum,
        is_openai,
        known_tool_names,
        metrics,
        saw_done,
        repetition_detected,
    } = result;
    // Convert accumulated OpenAI tool calls to our format
    if is_openai {
        for accum in &openai_tool_accum {
            if !accum.name.is_empty() {
                let arguments: Value =
                    serde_json::from_str(&accum.arguments).unwrap_or(Value::Object(
                        serde_json::Map::new(),
                    ));
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
    }
}
