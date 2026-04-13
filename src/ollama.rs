use std::future::Future;
use std::pin::Pin;

use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::backend::{ChatResponse, ModelBackend, ModelInfo};
use crate::message::{FunctionCall, Message, ToolCall};

/// Strip leaked special tokens from model output.
///
/// Some models (e.g. Gemma 4) emit special/control tokens like `<|channel>`,
/// `<|turn>`, `<|think|>` as plain text instead of handling them internally.
/// These should never appear in user-facing content.
fn strip_special_tokens(text: &str) -> String {
    // Match patterns like <|something> or <|something|> or <something|>
    // These are control tokens that leaked through the tokenizer.
    let mut result = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();

    while let Some(&(i, c)) = chars.peek() {
        if c == '<' && text[i..].starts_with("<|") {
            // Look for closing > (with optional | before it)
            if let Some(end) = text[i..].find('>') {
                // Skip the entire special token
                let token_end = i + end + 1;
                // Advance the iterator past this token
                while let Some(&(j, _)) = chars.peek() {
                    if j >= token_end {
                        break;
                    }
                    chars.next();
                }
                continue;
            }
        }
        if c == '<' && text[i..].len() > 1 {
            // Also match <something|> pattern
            if let Some(end_rel) = text[i..].find("|>") {
                let candidate = &text[i..i + end_rel + 2];
                // Only strip if it looks like a token (no spaces, reasonable length)
                if candidate.len() <= 30 && !candidate[1..].contains(' ') {
                    let token_end = i + end_rel + 2;
                    while let Some(&(j, _)) = chars.peek() {
                        if j >= token_end {
                            break;
                        }
                        chars.next();
                    }
                    continue;
                }
            }
        }
        result.push(c);
        chars.next();
    }

    result
}

/// Detect degenerate repetition in streamed content.
///
/// Checks whether the tail of the text consists of a short substring pattern
/// (3–40 chars) repeated 8+ times consecutively.  This catches the common LLM
/// failure mode where the model gets stuck in a sampling loop, e.g.
/// "approach-approach approach-approach approach-approach …"
fn detect_repetition(text: &str) -> bool {
    // Need enough text to have a meaningful pattern + repetitions
    if text.len() < 100 {
        return false;
    }

    // Only inspect the tail — repetition is always at the end
    let start = text.len().saturating_sub(300);
    // Align to a char boundary
    let start = text.ceil_char_boundary(start);
    let tail = &text[start..];
    let tail_bytes = tail.as_bytes();

    for plen in 3..=40 {
        let min_repeats = 8;
        if tail.len() < plen * min_repeats {
            continue;
        }

        // Use the *last* `plen` bytes as the candidate pattern, then count
        // how many consecutive copies appear going backwards.
        let pattern = &tail_bytes[tail_bytes.len() - plen..];
        let mut count: usize = 0;
        let mut pos = tail_bytes.len();

        while pos >= plen {
            let candidate = &tail_bytes[pos - plen..pos];
            if candidate == pattern {
                count += 1;
                pos -= plen;
            } else {
                break;
            }
        }

        if count >= min_repeats {
            return true;
        }
    }

    false
}

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
                        id: None,
                        call_type: None,
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

    if !calls.is_empty() {
        // Remove matched JSON from the content in reverse order.
        let mut remaining = cleaned;
        for &(start, end) in removals.iter().rev() {
            remaining.replace_range(start..end, "");
        }
        return (calls, remaining.trim().to_string());
    }

    // Fallback: try to extract <function=NAME>...</function> XML-style tool calls.
    // Some models (e.g. Qwen3-coder) intermittently emit this format instead of
    // structured tool_calls or JSON objects.
    let (xml_calls, xml_remaining) = extract_function_tag_calls(&cleaned, known_tools);
    if !xml_calls.is_empty() {
        return (xml_calls, xml_remaining);
    }

    (calls, content.to_string())
}

/// Parse `<function=NAME>...<parameter=KEY>VALUE</parameter>...</function>` blocks
/// from text content. Returns extracted tool calls and cleaned content.
fn extract_function_tag_calls(
    content: &str,
    known_tools: &[String],
) -> (Vec<ToolCall>, String) {
    let mut calls = Vec::new();
    let mut removals: Vec<(usize, usize)> = Vec::new();

    let mut search_from = 0;
    while let Some(func_start_rel) = content[search_from..].find("<function=") {
        let abs_start = search_from + func_start_rel;
        let name_start = abs_start + "<function=".len();

        let Some(name_end_rel) = content[name_start..].find('>') else {
            search_from = name_start;
            continue;
        };
        let name = content[name_start..name_start + name_end_rel].trim();

        if !known_tools.iter().any(|t| t == name) {
            search_from = name_start + name_end_rel;
            continue;
        }

        let body_start = name_start + name_end_rel + 1;
        let Some(func_end_rel) = content[body_start..].find("</function>") else {
            search_from = body_start;
            continue;
        };
        let body = &content[body_start..body_start + func_end_rel];
        let block_end = body_start + func_end_rel + "</function>".len();

        let mut arguments = serde_json::Map::new();
        let mut param_search = 0;
        while let Some(param_start_rel) = body[param_search..].find("<parameter=") {
            let pname_start = param_search + param_start_rel + "<parameter=".len();
            let Some(pname_end_rel) = body[pname_start..].find('>') else {
                param_search = pname_start;
                continue;
            };
            let param_name = body[pname_start..pname_start + pname_end_rel].trim();
            let value_start = pname_start + pname_end_rel + 1;

            let Some(pclose_rel) = body[value_start..].find("</parameter>") else {
                param_search = value_start;
                continue;
            };
            let value = body[value_start..value_start + pclose_rel].trim();

            arguments.insert(
                param_name.to_string(),
                serde_json::Value::String(value.to_string()),
            );

            param_search = value_start + pclose_rel + "</parameter>".len();
        }

        calls.push(ToolCall {
            id: None,
            call_type: None,
            function: FunctionCall {
                name: name.to_string(),
                arguments: serde_json::Value::Object(arguments),
            },
        });
        removals.push((abs_start, block_end));
        search_from = block_end;
    }

    if calls.is_empty() {
        return (calls, content.to_string());
    }

    // Remove matched blocks from content in reverse order.
    let mut remaining = content.to_string();
    for &(start, end) in removals.iter().rev() {
        remaining.replace_range(start..end, "");
    }

    (calls, remaining.trim().to_string())
}

#[derive(Clone)]
pub struct OllamaBackend {
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
    /// Request token usage stats in OpenAI-compatible streaming responses.
    /// Ollama ignores this field; llama-server requires it to report token counts.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
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
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
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
struct ModelsResponse {
    models: Vec<ModelInfo>,
}

struct TimingMetrics {
    prompt_eval_count: u64,
    prompt_eval_duration: u64,
    eval_count: u64,
    eval_duration: u64,
    load_duration: u64,
    total_duration: u64,
}

// --- OpenAI SSE format types (used by llama-server's /api/chat) ---

#[derive(Debug, Deserialize)]
struct OpenAIChunk {
    choices: Option<Vec<OpenAIChoice>>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    delta: Option<OpenAIDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIToolCallDelta {
    index: Option<usize>,
    function: Option<OpenAIFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct OpenAIFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

/// State for accumulating streamed OpenAI tool calls (name and arguments
/// arrive in separate chunks).
struct OpenAIToolCallAccum {
    name: String,
    arguments: String,
}

fn process_openai_chunk(
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

fn process_chunk(
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

/// Parse API error JSON into a user-friendly message.
///
/// Handles two common shapes:
/// - `{"error": {"message": "...", "type": "...", ...}}` (llama-server / OpenAI)
/// - `{"error": "..."}` (Ollama native)
fn parse_api_error(body: &str) -> Option<String> {
    let json: Value = serde_json::from_str(body).ok()?;
    let error = json.get("error")?;

    // Nested error object (llama-server / OpenAI format)
    if let Some(obj) = error.as_object() {
        let error_type = obj.get("type").and_then(|t| t.as_str());
        if error_type == Some("exceed_context_size_error") {
            let n_prompt = obj.get("n_prompt_tokens").and_then(|n| n.as_u64());
            let n_ctx = obj.get("n_ctx").and_then(|n| n.as_u64());
            if let (Some(prompt), Some(ctx)) = (n_prompt, n_ctx) {
                return Some(format!(
                    "Context window exceeded ({} tokens requested, {} available). \
                     Use /clear to start fresh.",
                    prompt, ctx,
                ));
            }
        }
        // Fall back to the message field
        if let Some(msg) = obj.get("message").and_then(|m| m.as_str()) {
            return Some(msg.to_string());
        }
    }

    // Simple string: {"error": "..."}
    error.as_str().map(|s| s.to_string())
}

impl OllamaBackend {
    pub fn new(base_url: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
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
        on_token: Box<dyn Fn(&str) + Send + 'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move {
            let options = num_ctx
                .filter(|&n| n > 0)
                .map(|n| ChatOptions {
                    num_ctx: n,
                    repeat_penalty: Some(1.1),
                    repeat_last_n: Some(64),
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

            let request = ChatRequest {
                model: model.to_string(),
                messages: messages.to_vec(),
                tools,
                stream: true,
                options,
                stream_options: Some(StreamOptions { include_usage: true }),
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
                if let Some(message) = parse_api_error(&body) {
                    anyhow::bail!("{}", message);
                }
                anyhow::bail!("API error ({}): {}", status, body);
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
            let mut repetition_detected = false;
            let mut token_count: u32 = 0;

            let mut stream = resp.bytes_stream();
            let mut buffer = String::new();

            // Detect format from the first non-empty line.
            // OpenAI SSE lines start with "data: ", Ollama lines are raw JSON.
            let mut is_openai: Option<bool> = None;
            let mut openai_tool_accum: Vec<OpenAIToolCallAccum> = Vec::new();

            'stream: while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim().to_string();
                    buffer.drain(..=newline_pos);

                    if line.is_empty() {
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
                            &mut tool_calls,
                            &mut metrics,
                            &*on_token,
                        );
                        if parsed.done {
                            saw_done = true;
                            break;
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

            // Process remaining buffer
            let remaining = buffer.trim();
            if !remaining.is_empty() && !saw_done {
                if is_openai == Some(true) {
                    let data = remaining.strip_prefix("data:").unwrap_or(remaining).trim();
                    if data != "[DONE]" && !data.is_empty() {
                        let parsed: OpenAIChunk = serde_json::from_str(data)
                            .with_context(|| {
                                format!("Failed to parse remaining OpenAI buffer: {}", data)
                            })?;
                        let finish = process_openai_chunk(
                            &parsed,
                            &mut content,
                            &mut openai_tool_accum,
                            &mut metrics,
                            &*on_token,
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
                        &mut content,
                        &mut tool_calls,
                        &mut metrics,
                        &*on_token,
                    );
                    if parsed.done {
                        saw_done = true;
                    }
                }
            }

            // Convert accumulated OpenAI tool calls to our format
            if is_openai == Some(true) {
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
                tool_calls_from_content,
                repetition_detected,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_function_tag_single_param() {
        let content = "<function=read>\n<parameter=file_path>\nCLAUDE.md\n</parameter>\n</function>\n</tool_call>";
        let known = vec!["read".to_string(), "bash".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(
            calls[0].function.arguments.get("file_path").unwrap().as_str().unwrap(),
            "CLAUDE.md"
        );
        assert!(remaining.is_empty(), "remaining was: {:?}", remaining);
    }

    #[test]
    fn extract_function_tag_multiple_params() {
        let content = "<function=edit>\n<parameter=file_path>\nsrc/main.rs\n</parameter>\n<parameter=old>\nfn main()\n</parameter>\n<parameter=new>\nfn main() -> Result<()>\n</parameter>\n</function>";
        let known = vec!["edit".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "edit");
        assert_eq!(
            calls[0].function.arguments.get("file_path").unwrap().as_str().unwrap(),
            "src/main.rs"
        );
        assert_eq!(
            calls[0].function.arguments.get("old").unwrap().as_str().unwrap(),
            "fn main()"
        );
        assert_eq!(
            calls[0].function.arguments.get("new").unwrap().as_str().unwrap(),
            "fn main() -> Result<()>"
        );
        assert!(remaining.is_empty());
    }

    #[test]
    fn extract_function_tag_with_surrounding_text() {
        let content = "Let me read that file.\n<function=read>\n<parameter=file_path>\nCLAUDE.md\n</parameter>\n</function>\nDone.";
        let known = vec!["read".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(remaining, "Let me read that file.\n\nDone.");
    }

    #[test]
    fn extract_function_tag_unknown_tool_ignored() {
        let content = "<function=unknown_tool>\n<parameter=x>\n1\n</parameter>\n</function>";
        let known = vec!["read".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert!(calls.is_empty());
        assert_eq!(remaining, content);
    }

    #[test]
    fn extract_json_still_works() {
        let content = r#"{"name": "read", "arguments": {"file_path": "test.rs"}}"#;
        let known = vec!["read".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert!(remaining.is_empty());
    }

    #[test]
    fn extract_function_tag_multiple_calls() {
        let content = "<function=read>\n<parameter=file_path>\na.rs\n</parameter>\n</function>\n<function=read>\n<parameter=file_path>\nb.rs\n</parameter>\n</function>";
        let known = vec!["read".to_string()];
        let (calls, _remaining) = extract_tool_calls_from_content(content, &known);
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[0].function.arguments.get("file_path").unwrap().as_str().unwrap(),
            "a.rs"
        );
        assert_eq!(
            calls[1].function.arguments.get("file_path").unwrap().as_str().unwrap(),
            "b.rs"
        );
    }
}
