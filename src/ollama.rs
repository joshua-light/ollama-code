use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::{FunctionCall, Message, ToolCall};

/// Find balanced `{…}` JSON objects in a string by brace-matching.
/// Returns `(json_text, start_byte_offset, end_byte_offset)` for each match.
fn find_json_objects(text: &str) -> Vec<(String, usize, usize)> {
    let mut results = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'{' {
            let start = i;
            let mut depth: i32 = 0;
            let mut in_string = false;
            let mut escape = false;
            let mut j = i;

            while j < bytes.len() {
                let c = bytes[j];
                if escape {
                    escape = false;
                    j += 1;
                    continue;
                }
                if in_string {
                    if c == b'\\' {
                        escape = true;
                    } else if c == b'"' {
                        in_string = false;
                    }
                    j += 1;
                    continue;
                }
                // Outside a string
                match c {
                    b'"' => in_string = true,
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            let end = j + 1;
                            results.push((text[start..end].to_string(), start, end));
                            i = end;
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }

            // If we never closed the brace, skip past the opening {
            if depth != 0 {
                i = start + 1;
            }
        } else {
            i += 1;
        }
    }

    results
}

/// Try to extract tool calls from the text content when a model emits them as
/// plain-text JSON instead of using Ollama's structured `tool_calls` field.
/// Only matches JSON objects whose `"name"` value is in `known_tools`.
/// Returns extracted tool calls and the remaining (cleaned) content.
fn extract_tool_calls_from_content(
    content: &str,
    known_tools: &[String],
) -> (Vec<ToolCall>, String) {
    let mut calls = Vec::new();

    // Some models wrap tool calls in <tool_call> XML-style tags — strip them
    // so the JSON extractor can find the inner objects.
    let cleaned = content
        .replace("<tool_call>", "")
        .replace("</tool_call>", "");

    // Find all top-level JSON objects in the text.
    let objects = find_json_objects(&cleaned);

    // Byte ranges to remove (collected, then applied in reverse order so
    // earlier removals don't shift later offsets).
    let mut removals: Vec<(usize, usize)> = Vec::new();

    for (json_str, start, end) in &objects {
        if let Ok(val) = serde_json::from_str::<Value>(json_str) {
            if let (Some(name), Some(arguments)) = (
                val.get("name").and_then(|v| v.as_str()),
                val.get("arguments").filter(|a| a.is_object()),
            ) {
                if known_tools.iter().any(|t| t == name) {
                    calls.push(ToolCall {
                        function: FunctionCall {
                            name: name.to_string(),
                            arguments: arguments.clone(),
                        },
                    });
                    removals.push((*start, *end));
                }
            }
        }
    }

    if calls.is_empty() {
        return (calls, content.to_string());
    }

    // Remove matched JSON from the content in reverse order.
    let mut remaining = cleaned;
    for &(start, end) in removals.iter().rev() {
        remaining.replace_range(start..end, "");
    }

    (calls, remaining.trim().to_string())
}

#[derive(Clone)]
pub struct OllamaClient {
    client: Client,
    base_url: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<ChatOptions>,
}

#[derive(Debug, Serialize)]
struct ChatOptions {
    num_ctx: u64,
}

#[derive(Debug, Deserialize)]
struct ChatChunk {
    message: Option<ChunkMessage>,
    #[serde(default)]
    done: bool,
    prompt_eval_count: Option<u64>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<u64>,
    eval_duration: Option<u64>,
    load_duration: Option<u64>,
    total_duration: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ChunkMessage {
    #[allow(dead_code)]
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCallResponse>>,
}

#[derive(Debug, Deserialize)]
struct ToolCallResponse {
    function: FunctionCallResponse,
}

#[derive(Debug, Deserialize)]
struct FunctionCallResponse {
    name: String,
    arguments: Value,
}

#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    pub name: String,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    models: Vec<ModelInfo>,
}

pub struct ChatResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub prompt_eval_count: u64,
    /// Ollama timing metrics (nanoseconds).
    pub prompt_eval_duration: u64,
    pub eval_count: u64,
    pub eval_duration: u64,
    pub load_duration: u64,
    pub total_duration: u64,
    /// True if the stream ended without receiving `done: true` from Ollama.
    pub incomplete: bool,
}

struct TimingMetrics {
    prompt_eval_count: u64,
    prompt_eval_duration: u64,
    eval_count: u64,
    eval_duration: u64,
    load_duration: u64,
    total_duration: u64,
}

fn process_chunk<F: Fn(&str)>(
    parsed: &ChatChunk,
    content: &mut String,
    tool_calls: &mut Vec<ToolCall>,
    metrics: &mut TimingMetrics,
    on_token: &F,
) {
    if let Some(msg) = &parsed.message {
        if let Some(text) = &msg.content {
            if !text.is_empty() {
                content.push_str(text);
                on_token(text);
            }
        }
        if let Some(calls) = &msg.tool_calls {
            for call in calls {
                tool_calls.push(ToolCall {
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

impl OllamaClient {
    pub fn new(base_url: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
        }
    }

    /// Fetch model metadata from `/api/show` and extract the context window size.
    /// Scans `model_info` keys for one ending in `.context_length`.
    pub async fn context_length(&self, model: &str) -> Result<u64> {
        let body = serde_json::json!({ "name": model });
        let resp = self
            .client
            .post(format!("{}/api/show", self.base_url))
            .json(&body)
            .send()
            .await
            .context("Failed to connect to Ollama")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Ollama /api/show error ({}): {}", status, text);
        }

        let data: Value = resp.json().await?;

        if let Some(model_info) = data.get("model_info").and_then(|v| v.as_object()) {
            for (key, val) in model_info {
                if key.ends_with(".context_length") {
                    if let Some(n) = val.as_u64() {
                        return Ok(n);
                    }
                }
            }
        }

        // Fallback: not all models expose this; default to 0 (unknown)
        Ok(0)
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

    pub async fn chat<F>(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<Vec<Value>>,
        num_ctx: Option<u64>,
        on_token: F,
    ) -> Result<ChatResponse>
    where
        F: Fn(&str),
    {
        let options = num_ctx
            .filter(|&n| n > 0)
            .map(|n| ChatOptions { num_ctx: n });

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

        let request = ChatRequest {
            model: model.to_string(),
            messages: messages.to_vec(),
            tools,
            stream: true,
            options,
        };

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&request)
            .send()
            .await
            .context("Failed to connect to Ollama")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Ollama API error ({}): {}", status, body);
        }

        let mut content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut metrics = TimingMetrics {
            prompt_eval_count: 0,
            prompt_eval_duration: 0,
            eval_count: 0,
            eval_duration: 0,
            load_duration: 0,
            total_duration: 0,
        };
        let mut saw_done = false;

        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                let parsed: ChatChunk = serde_json::from_str(&line)
                    .with_context(|| format!("Failed to parse Ollama response: {}", line))?;

                process_chunk(&parsed, &mut content, &mut tool_calls, &mut metrics, &on_token);

                if parsed.done {
                    saw_done = true;
                    break;
                }
            }
        }

        // Process remaining buffer
        let remaining = buffer.trim();
        if !remaining.is_empty() {
            let parsed: ChatChunk = serde_json::from_str(remaining)
                .with_context(|| {
                    format!(
                        "Failed to parse remaining Ollama buffer ({} bytes): {}",
                        remaining.len(),
                        &remaining[..remaining.len().min(200)]
                    )
                })?;
            process_chunk(&parsed, &mut content, &mut tool_calls, &mut metrics, &on_token);
            if parsed.done {
                saw_done = true;
            }
        }

        // Some models (e.g. Qwen) emit tool calls as plain-text JSON in the
        // content instead of using Ollama's structured tool_calls field.
        // When no structured calls were received, try to extract them from the
        // content as a fallback.
        if tool_calls.is_empty() && !content.trim().is_empty() && !known_tool_names.is_empty() {
            let (extracted, remaining) =
                extract_tool_calls_from_content(&content, &known_tool_names);
            if !extracted.is_empty() {
                tool_calls = extracted;
                content = remaining;
            }
        }

        Ok(ChatResponse {
            content,
            tool_calls,
            prompt_eval_count: metrics.prompt_eval_count,
            prompt_eval_duration: metrics.prompt_eval_duration,
            eval_count: metrics.eval_count,
            eval_duration: metrics.eval_duration,
            load_duration: metrics.load_duration,
            total_duration: metrics.total_duration,
            incomplete: !saw_done,
        })
    }
}
