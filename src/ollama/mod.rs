mod parse;
pub mod startup;
mod streaming;

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
use serde::Serialize;
use serde_json::Value;

use crate::backend::{ChatResponse, ModelBackend, ModelInfo};
use crate::message::{self, Message, ToolCall};

/// If no bytes arrive within this window the stream is considered stalled.
const STREAM_INACTIVITY_TIMEOUT: Duration = Duration::from_secs(180);

use parse::detect_repetition;
use streaming::{
    ChatChunk, OpenAIChunk, OpenAIToolCallAccum, StreamResult, TimingMetrics,
    process_chunk, process_openai_chunk, process_remaining_buffer, postprocess_response,
};

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<ChatOptions>,
    /// Request token usage stats in OpenAI-compatible streaming responses.
    /// Ollama ignores this field; llama-server requires it to report token counts.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    /// Ask Ollama to emit a separate reasoning channel (`message.thinking`).
    /// Ignored by llama-server (it emits reasoning inline or under
    /// `reasoning_content` depending on the model's template).
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ChatOptions {
    num_ctx: u64,
    /// Penalize repeated token sequences to prevent degenerate repetition loops.
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_penalty: Option<f32>,
    /// How many recent tokens to consider for repeat penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_last_n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, serde::Deserialize)]
struct ModelsResponse {
    models: Vec<ModelInfo>,
}

/// Sampling parameters stored in the backend instance.
#[derive(Clone, Debug, Default)]
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
}

#[derive(Clone)]
pub struct OllamaBackend {
    client: Client,
    base_url: String,
    sampling: SamplingParams,
}

impl OllamaBackend {
    pub fn new(base_url: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.unwrap_or_else(|| crate::config::DEFAULT_OLLAMA_URL.to_string()),
            sampling: SamplingParams::default(),
        }
    }

    pub fn with_sampling(base_url: Option<String>, sampling: SamplingParams) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.unwrap_or_else(|| crate::config::DEFAULT_OLLAMA_URL.to_string()),
            sampling,
        }
    }

    /// Tell Ollama to unload a model from memory (frees VRAM).
    pub async fn unload_model(&self, model: &str) -> Result<()> {
        let body = serde_json::json!({
            "model": model,
            "keep_alive": 0
        });
        let _ = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&body)
            .send()
            .await;
        Ok(())
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let resp: ModelsResponse = self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .context("Failed to connect to Ollama. Is it running? Try: ollama serve")?
            .json()
            .await?;
        Ok(resp.models)
    }
}

impl ModelBackend for OllamaBackend {
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: &'a [Message],
        tools: Option<Vec<Value>>,
        num_ctx: Option<u64>,
        thinking_budget_tokens: Option<u64>,
        on_token: Box<dyn Fn(&str) + Send + 'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move {
            let options = num_ctx
                .filter(|&n| n > 0)
                .map(|n| ChatOptions {
                    num_ctx: n,
                    repeat_penalty: Some(1.1),
                    repeat_last_n: Some(64),
                    temperature: self.sampling.temperature,
                    top_p: self.sampling.top_p,
                    top_k: self.sampling.top_k,
                });

            // Extract known tool names before `tools` is moved into the request.
            // Used later to detect tool calls emitted as plain text.
            let known_tool_names: Vec<String> = tools
                .as_ref()
                .map(|t| {
                    t.iter()
                        .filter_map(|def| {
                            def.get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|n| n.as_str())
                                .map(|s| s.to_string())
                        })
                        .collect()
                })
                .unwrap_or_default();

            // Thinking-budget semantics:
            //   Some(0) → explicitly disable thinking (`think: false`, no
            //             stream-side budget enforcement). The harness flips
            //             to this after the budget tripped once, so models
            //             that think unconditionally stop emitting reasoning
            //             on retry.
            //   Some(N) → enable thinking with budget N tokens.
            //   None    → no explicit opt-in/out; still cap accumulated
            //             thinking at the default to avoid a single turn
            //             burning the whole stream on reasoning the harness
            //             can't see.
            const DEFAULT_THINKING_BUDGET_TOKENS: u64 = 2048;
            let (think, budget_chars): (Option<bool>, Option<usize>) = match thinking_budget_tokens {
                Some(0) => (Some(false), None),
                Some(n) => (
                    Some(true),
                    Some((n as usize) * (message::CHARS_PER_TOKEN as usize)),
                ),
                None => (
                    None,
                    Some(
                        (DEFAULT_THINKING_BUDGET_TOKENS as usize)
                            * (message::CHARS_PER_TOKEN as usize),
                    ),
                ),
            };

            let request = ChatRequest {
                model: model.to_string(),
                messages: messages.to_vec(),
                tools,
                stream: true,
                options,
                stream_options: Some(StreamOptions { include_usage: true }),
                think,
            };

            let resp = self
                .client
                .post(format!("{}/api/chat", self.base_url))
                .json(&request)
                .send()
                .await
                .context("Failed to connect to backend")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                if let Some(message) = parse::parse_api_error(&body) {
                    anyhow::bail!("{}", message);
                }
                anyhow::bail!("API error ({}): {}", status, body);
            }

            let mut content = String::new();
            let mut thinking = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut metrics = TimingMetrics::new();
            let mut saw_done = false;
            let mut repetition_detected = false;
            let mut thinking_budget_exceeded = false;
            let mut token_count: u32 = 0;

            let mut stream = resp.bytes_stream();
            // Use a byte buffer so that multi-byte UTF-8 characters split
            // across HTTP chunk boundaries are preserved intact instead of
            // being replaced with U+FFFD by String::from_utf8_lossy.
            let mut buffer: Vec<u8> = Vec::new();

            // Detect format from the first non-empty line.
            // OpenAI SSE lines start with "data: ", Ollama lines are raw JSON.
            let mut is_openai: Option<bool> = None;
            let mut openai_tool_accum: Vec<OpenAIToolCallAccum> = Vec::new();

            'stream: loop {
                let chunk = match tokio::time::timeout(STREAM_INACTIVITY_TIMEOUT, stream.next()).await {
                    Ok(Some(chunk)) => chunk?,
                    Ok(None) => break,               // stream ended normally
                    Err(_) => break 'stream,          // inactivity timeout
                };
                buffer.extend_from_slice(&chunk);

                while let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                    let line = match std::str::from_utf8(&buffer[..newline_pos]) {
                        Ok(s) => s.trim().to_string(),
                        Err(_) => String::from_utf8_lossy(&buffer[..newline_pos]).trim().to_string(),
                    };
                    buffer.drain(..=newline_pos);

                    if line.is_empty() {
                        continue;
                    }

                    // Skip SSE comment/heartbeat lines (e.g. ": keepalive").
                    if line.starts_with(':') {
                        continue;
                    }

                    // Auto-detect format on first non-empty line
                    if is_openai.is_none() {
                        is_openai = Some(line.starts_with("data:"));
                    }

                    if is_openai == Some(true) {
                        // OpenAI SSE format
                        let data = line.strip_prefix("data:").unwrap_or(&line).trim();
                        if data == "[DONE]" {
                            saw_done = true;
                            break;
                        }
                        if data.is_empty() {
                            continue;
                        }
                        let parsed: OpenAIChunk = serde_json::from_str(data)
                            .with_context(|| {
                                format!("Failed to parse OpenAI response: {}", data)
                            })?;
                        let finish = process_openai_chunk(
                            &parsed,
                            &mut content,
                            &mut thinking,
                            &mut openai_tool_accum,
                            &mut metrics,
                            &*on_token,
                        );
                        if finish.is_some() {
                            saw_done = true;
                        }
                    } else {
                        // Ollama format
                        let parsed: ChatChunk = serde_json::from_str(&line)
                            .with_context(|| {
                                format!("Failed to parse Ollama response: {}", line)
                            })?;
                        process_chunk(
                            &parsed,
                            &mut content,
                            &mut thinking,
                            &mut tool_calls,
                            &mut metrics,
                            &*on_token,
                        );
                        if parsed.done {
                            saw_done = true;
                            break;
                        }
                    }

                    // Thinking-budget cap: abort mid-stream once accumulated
                    // reasoning tokens blow past the configured budget. The
                    // reqwest body stream is dropped on scope exit, so we stop
                    // paying wall-clock immediately.
                    if let Some(limit) = budget_chars {
                        if thinking.len() > limit {
                            thinking_budget_exceeded = true;
                            break 'stream;
                        }
                    }

                    // Periodically check for degenerate repetition in the
                    // accumulated content.  Start checking after 50 tokens and
                    // re-check every 20 tokens to keep overhead low.
                    token_count += 1;
                    if token_count >= 50 && token_count.is_multiple_of(20) && detect_repetition(&content) {
                        repetition_detected = true;
                        break 'stream;
                    }
                }
            }

            let remaining = String::from_utf8_lossy(&buffer);
            if !saw_done && !thinking_budget_exceeded {
                saw_done = process_remaining_buffer(
                    &remaining,
                    is_openai,
                    &mut content,
                    &mut thinking,
                    &mut tool_calls,
                    &mut openai_tool_accum,
                    &mut metrics,
                    &*on_token,
                )?;
            }

            Ok(postprocess_response(StreamResult {
                content,
                thinking,
                tool_calls,
                openai_tool_accum,
                is_openai: is_openai == Some(true),
                known_tool_names,
                metrics,
                saw_done,
                repetition_detected,
                thinking_budget_exceeded,
            }))
        })
    }
}
