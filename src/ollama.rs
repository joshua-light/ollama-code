use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::{FunctionCall, Message, ToolCall};

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
}

#[derive(Debug, Deserialize)]
struct ChatChunk {
    message: Option<ChunkMessage>,
    #[serde(default)]
    done: bool,
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
}

impl OllamaClient {
    pub fn new(base_url: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
        }
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
        on_token: F,
    ) -> Result<ChatResponse>
    where
        F: Fn(&str),
    {
        let request = ChatRequest {
            model: model.to_string(),
            messages: messages.to_vec(),
            tools,
            stream: true,
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
                    break;
                }
            }
        }

        // Process remaining buffer
        let remaining = buffer.trim();
        if !remaining.is_empty() {
            if let Ok(parsed) = serde_json::from_str::<ChatChunk>(remaining) {
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
            }
        }

        Ok(ChatResponse {
            content,
            tool_calls,
        })
    }
}
