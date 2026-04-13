use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// OpenAI-compatible: links a tool result to the tool call that produced it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call (OpenAI-compatible).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Always "function" (OpenAI-compatible).
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    /// Generate a simple unique ID for this tool call.
    pub fn with_id(mut self) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        self.id = Some(format!("call_{}", n));
        self.call_type = Some("function".to_string());
        self
    }
}

/// Approximate chars-per-token ratio (biased toward code).
pub const CHARS_PER_TOKEN: u64 = 3;

/// Estimate tokens from a raw char length.
pub fn estimate_tokens(char_len: usize) -> u64 {
    (char_len as u64) / CHARS_PER_TOKEN
}

impl Message {
    /// Rough token estimate (chars / CHARS_PER_TOKEN heuristic + overhead).
    pub fn estimated_tokens(&self) -> u64 {
        let content_tokens = estimate_tokens(self.content.len());
        let tool_call_tokens = self.tool_calls.as_ref().map_or(0, |calls| {
            calls
                .iter()
                .map(|tc| {
                    estimate_tokens(tc.function.name.len() + tc.function.arguments.to_string().len())
                })
                .sum()
        });
        content_tokens + tool_call_tokens + 4 // 4 for role/formatting overhead
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn tool(content: impl Into<String>, tool_call_id: Option<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id,
        }
    }
}
