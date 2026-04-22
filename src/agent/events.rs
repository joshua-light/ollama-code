use anyhow::Result;
use tokio::sync::mpsc;

use crate::message::Message;

#[derive(Debug, Clone)]
pub enum AgentEvent {
    Token(String),
    /// Replace previously streamed content. Emitted when tool calls were
    /// extracted from the text content (some models emit tool calls as plain
    /// JSON instead of using structured tool_calls).
    ContentReplaced(String),
    ToolCall { name: String, args: String },
    /// Request user confirmation before executing a tool.
    /// The agent blocks waiting on the confirm channel.
    ToolConfirmRequest { name: String, args: String },
    ToolResult { name: String, output: String, success: bool },
    /// Partial output from a running tool (e.g. bash streaming stdout).
    ToolOutput { output: String },
    ContextUpdate { prompt_tokens: u64 },
    /// Context was auto-trimmed to stay within the window.
    ContextTrimmed { removed_messages: usize, estimated_tokens_freed: u64 },
    /// Context compaction is starting (LLM summarization of old messages).
    ContextCompacting,
    /// Context was compacted via LLM summarization.
    ContextCompacted {
        removed_messages: usize,
        summary_tokens: u64,
        estimated_tokens_freed: u64,
    },
    Done { prompt_tokens: u64, eval_count: u64 },
    /// Generation was cancelled by the user.
    Cancelled,
    Error(String),
    MessageLogged(Message),
    Debug(String),
    // Sub-agent lifecycle events
    SubagentStart { task: String },
    SubagentToolCall { name: String, args: String },
    SubagentToolResult { name: String, success: bool },
    SubagentEnd { result: String },
    /// Hot-reload completed — carries a summary and the new system prompt.
    ReloadComplete {
        summary: String,
        system_prompt: String,
    },
    /// Thinking budget was exceeded mid-stream. The agent has disabled
    /// thinking for the rest of the session and injected a follow-up asking
    /// the model to commit to an implementation.
    ThinkingBudgetExceeded {
        /// Approximate tokens of reasoning emitted before the abort.
        thinking_tokens: u64,
    },
    /// System prompt composition info (emitted once at first run).
    SystemPromptInfo {
        base_prompt_tokens: u64,
        project_docs: Vec<(String, u64)>,
        skills_tokens: u64,
        /// Per-category breakdown of tool definition tokens: (label, tokens).
        /// Labels are "Built-in", "MCP: <server>", or "Plugins".
        tool_defs_breakdown: Vec<(String, u64)>,
    },
}

/// Send an event through the channel, returning an error if the receiver is gone.
pub(super) fn send_event(
    events: &mpsc::UnboundedSender<AgentEvent>,
    event: AgentEvent,
) -> Result<()> {
    events
        .send(event)
        .map_err(|_| anyhow::anyhow!("Event channel closed"))
}
