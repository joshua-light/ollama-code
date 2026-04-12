use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;

use crate::backend::ModelBackend;
use crate::message::Message;
use crate::tools::{BashTool, EditTool, GlobTool, GrepTool, ReadTool, SubagentToolDef, WriteTool, ToolRegistry};

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
    ContextUpdate { prompt_tokens: u64 },
    /// Context was auto-trimmed to stay within the window.
    ContextTrimmed { removed_messages: usize, estimated_tokens_freed: u64 },
    Done { prompt_tokens: u64 },
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
}

/// Maximum number of lines to keep in tool output stored in context.
const MAX_TOOL_OUTPUT_LINES: usize = 300;

fn truncate_tool_output(output: &str) -> String {
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() <= MAX_TOOL_OUTPUT_LINES {
        return output.to_string();
    }

    let kept: String = lines[..MAX_TOOL_OUTPUT_LINES]
        .iter()
        .flat_map(|l| [*l, "\n"])
        .collect();
    format!(
        "{}... ({} more lines truncated. Refine your command to get more targeted output.)",
        kept,
        lines.len() - MAX_TOOL_OUTPUT_LINES,
    )
}

/// Poll a cancel flag, resolving when it becomes true.
async fn poll_cancel(flag: &AtomicBool) {
    loop {
        if flag.load(Ordering::Relaxed) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// Send an event through the channel, returning an error if the receiver is gone.
fn send_event(
    events: &mpsc::UnboundedSender<AgentEvent>,
    event: AgentEvent,
) -> Result<()> {
    events
        .send(event)
        .map_err(|_| anyhow::anyhow!("Event channel closed"))
}

pub struct Agent {
    backend: Arc<dyn ModelBackend>,
    tools: ToolRegistry,
    model: String,
    messages: Vec<Message>,
    system_prompt_logged: bool,
    context_size: u64,
    bash_timeout: std::time::Duration,
    /// If set, limits the number of agent-loop turns (used for sub-agents).
    max_turns: Option<u16>,
    /// Turn limit passed to sub-agents spawned by this agent.
    subagent_max_turns: u16,
}

impl Agent {
    /// Shared constructor. When `is_subagent` is true, the subagent tool is
    /// omitted (prevents recursion) and a turn limit is enforced.
    fn build(
        backend: Arc<dyn ModelBackend>,
        model: String,
        context_size: u64,
        bash_timeout: std::time::Duration,
        subagent_max_turns: u16,
        max_turns: Option<u16>,
    ) -> Self {
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(BashTool));
        tools.register(Box::new(ReadTool));
        tools.register(Box::new(EditTool));
        tools.register(Box::new(WriteTool));
        tools.register(Box::new(GlobTool));
        tools.register(Box::new(GrepTool));
        let is_subagent = max_turns.is_some();
        if !is_subagent {
            tools.register(Box::new(SubagentToolDef));
        }

        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let subagent_desc = if is_subagent {
            ""
        } else {
            "- subagent(task): Spawn a sub-agent with a fresh context to handle a focused task. \
             The sub-agent cannot see this conversation, so include all necessary context in the \
             task description. Use for: research across many files, complex multi-step operations, \
             or any task that would benefit from a clean context window"
        };

        let system_prompt = include_str!("../SYSTEM_PROMPT.md")
            .replace("{cwd}", &cwd)
            .replace("{subagent_tool}", subagent_desc);

        let messages = vec![Message::system(&system_prompt)];

        Self {
            backend,
            tools,
            model,
            messages,
            system_prompt_logged: false,
            context_size,
            bash_timeout,
            max_turns,
            subagent_max_turns,
        }
    }

    pub fn new(
        backend: Arc<dyn ModelBackend>,
        model: String,
        context_size: u64,
        bash_timeout: std::time::Duration,
        subagent_max_turns: u16,
    ) -> Self {
        Self::build(backend, model, context_size, bash_timeout, subagent_max_turns, None)
    }

    /// Create a sub-agent with fresh context and no subagent tool (prevents recursion).
    fn new_subagent(
        backend: Arc<dyn ModelBackend>,
        model: String,
        context_size: u64,
        bash_timeout: std::time::Duration,
        max_turns: u16,
    ) -> Self {
        Self::build(backend, model, context_size, bash_timeout, 0, Some(max_turns))
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }

    pub fn set_backend(&mut self, backend: Arc<dyn ModelBackend>) {
        self.backend = backend;
    }

    pub fn set_context_size(&mut self, size: u64) {
        self.context_size = size;
    }

    pub fn clear_history(&mut self) {
        self.messages.truncate(1); // keep system prompt
    }

    /// Return the last assistant message content, or a fallback string.
    fn last_assistant_message(&self) -> String {
        self.messages
            .iter()
            .rev()
            .find_map(|m| {
                if matches!(m.role, crate::message::Role::Assistant) && !m.content.trim().is_empty() {
                    Some(m.content.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "Sub-agent produced no response.".to_string())
    }

    /// Trim oldest non-system messages when context usage exceeds threshold.
    /// Removes complete exchanges (user + assistant + tool messages) as units.
    fn trim_context(&mut self, current_prompt_tokens: u64) -> Option<(usize, u64)> {
        if self.context_size == 0 {
            return None;
        }

        let threshold = self.context_size * 80 / 100;
        if current_prompt_tokens <= threshold {
            return None;
        }

        // Target: trim down to 60% of context
        let target = self.context_size * 60 / 100;
        let need_to_free = current_prompt_tokens.saturating_sub(target);

        let first_non_system = self
            .messages
            .iter()
            .position(|m| !matches!(m.role, crate::message::Role::System))
            .unwrap_or(self.messages.len());

        let mut freed: u64 = 0;
        let mut remove_until = first_non_system;

        // Walk forward, removing complete exchanges (up to the next User message)
        let mut i = first_non_system;
        while freed < need_to_free && i < self.messages.len().saturating_sub(1) {
            freed += self.messages[i].estimated_tokens();
            i += 1;
            // Keep going until we hit the next User message (start of a new exchange)
            while i < self.messages.len().saturating_sub(1)
                && !matches!(self.messages[i].role, crate::message::Role::User)
            {
                freed += self.messages[i].estimated_tokens();
                i += 1;
            }
            remove_until = i;
        }

        if remove_until > first_non_system {
            let removed = remove_until - first_non_system;
            self.messages.drain(first_non_system..remove_until);
            Some((removed, freed))
        } else {
            None
        }
    }

    pub async fn run(
        &mut self,
        user_input: &str,
        events: &mpsc::UnboundedSender<AgentEvent>,
        confirm_rx: &mut mpsc::UnboundedReceiver<bool>,
        cancel: Arc<AtomicBool>,
    ) -> Result<()> {
        if !self.system_prompt_logged {
            if let Some(sys_msg) = self.messages.first() {
                send_event(events, AgentEvent::MessageLogged(sys_msg.clone()))?;
            }
            self.system_prompt_logged = true;
        }

        let user_msg = Message::user(user_input);
        self.messages.push(user_msg.clone());
        send_event(events, AgentEvent::MessageLogged(user_msg))?;

        let mut empty_retries: u32 = 0;
        const MAX_EMPTY_RETRIES: u32 = 10;
        let mut turn: u32 = 0;
        let num_ctx = if self.context_size > 0 { Some(self.context_size) } else { None };

        loop {
            // Check cancellation before each turn
            if cancel.load(Ordering::Relaxed) {
                send_event(events, AgentEvent::Cancelled)?;
                return Ok(());
            }

            // Enforce turn limit (sub-agents)
            if let Some(max) = self.max_turns {
                if turn >= max as u32 {
                    // Force a final response without tools
                    self.messages.push(Message::user(
                        "You have reached your turn limit. Summarize your findings and \
                         provide your final answer now.",
                    ));
                    let events_clone = events.clone();
                    let response = self
                        .backend
                        .chat(
                            &self.model,
                            &self.messages,
                            None, // no tools — force text response
                            num_ctx,
                            Box::new(move |token| {
                                let _ = events_clone.send(AgentEvent::Token(token.to_string()));
                            }),
                        )
                        .await;
                    if let Ok(resp) = response {
                        if !resp.content.trim().is_empty() {
                            let msg = Message::assistant(&resp.content);
                            self.messages.push(msg.clone());
                            send_event(events, AgentEvent::MessageLogged(msg))?;
                        }
                    }
                    send_event(events, AgentEvent::Done { prompt_tokens: 0 })?;
                    return Ok(());
                }
            }
            turn += 1;

            let tool_defs = self.tools.definitions();
            let events_clone = events.clone();

            send_event(
                events,
                AgentEvent::Debug(format!(
                    "OLLAMA_REQUEST model={} messages={} tools={}",
                    self.model,
                    self.messages.len(),
                    tool_defs.len()
                )),
            )?;

            let mut response = tokio::select! {
                r = self.backend.chat(&self.model, &self.messages, Some(tool_defs), num_ctx, Box::new(move |token| {
                    let _ = events_clone.send(AgentEvent::Token(token.to_string()));
                })) => {
                    match r {
                        Ok(r) => r,
                        Err(e) => {
                            send_event(events, AgentEvent::Error(e.to_string()))?;
                            send_event(events, AgentEvent::Done { prompt_tokens: 0 })?;
                            return Ok(());
                        }
                    }
                }
                _ = poll_cancel(&cancel) => {
                    send_event(events, AgentEvent::Cancelled)?;
                    return Ok(());
                }
            };

            let ns_to_ms = |ns: u64| -> f64 { ns as f64 / 1_000_000.0 };
            let prompt_tps = if response.prompt_eval_duration > 0 {
                response.prompt_eval_count as f64 / (response.prompt_eval_duration as f64 / 1e9)
            } else {
                0.0
            };
            let gen_tps = if response.eval_duration > 0 {
                response.eval_count as f64 / (response.eval_duration as f64 / 1e9)
            } else {
                0.0
            };
            send_event(
                events,
                AgentEvent::Debug(format!(
                    "OLLAMA_RESPONSE content_len={} tool_calls={} prompt_eval={} incomplete={} \
                     | load={:.0}ms prompt={:.0}ms ({:.1} t/s) gen={:.0}ms/{} tokens ({:.1} t/s) total={:.0}ms",
                    response.content.len(),
                    response.tool_calls.len(),
                    response.prompt_eval_count,
                    response.incomplete,
                    ns_to_ms(response.load_duration),
                    ns_to_ms(response.prompt_eval_duration),
                    prompt_tps,
                    ns_to_ms(response.eval_duration),
                    response.eval_count,
                    gen_tps,
                    ns_to_ms(response.total_duration),
                )),
            )?;

            if response.prompt_eval_count > 0 {
                send_event(
                    events,
                    AgentEvent::ContextUpdate {
                        prompt_tokens: response.prompt_eval_count,
                    },
                )?;

                // Auto-trim context if approaching the limit
                if let Some((removed, freed)) = self.trim_context(response.prompt_eval_count) {
                    send_event(
                        events,
                        AgentEvent::ContextTrimmed {
                            removed_messages: removed,
                            estimated_tokens_freed: freed,
                        },
                    )?;
                }
            }

            if response.incomplete {
                // If the stream ended with no real content (e.g. the model
                // only emitted leaked special tokens that were stripped), treat
                // it as a retryable empty response instead of showing an error.
                if response.content.trim().is_empty()
                    && response.tool_calls.is_empty()
                    && empty_retries < MAX_EMPTY_RETRIES
                {
                    empty_retries += 1;
                    send_event(
                        events,
                        AgentEvent::Debug(format!(
                            "Incomplete stream with no content (likely special token leak), \
                             retrying ({}/{})",
                            empty_retries, MAX_EMPTY_RETRIES
                        )),
                    )?;
                    continue;
                }
                send_event(
                    events,
                    AgentEvent::Error(
                        "Ollama stream ended unexpectedly (connection may have dropped)".to_string(),
                    ),
                )?;
            }

            // Handle degenerate repetition detected during streaming.
            if response.repetition_detected {
                send_event(
                    events,
                    AgentEvent::Debug("Repetition detected in model output".to_string()),
                )?;

                if response.tool_calls.is_empty() {
                    // No tool calls — treat as a failed generation and retry.
                    if empty_retries < MAX_EMPTY_RETRIES {
                        empty_retries += 1;
                        send_event(
                            events,
                            AgentEvent::Debug(format!(
                                "Degenerate response with no tool calls, retrying ({}/{})",
                                empty_retries, MAX_EMPTY_RETRIES
                            )),
                        )?;
                        // Clear the streamed garbage from the TUI
                        send_event(events, AgentEvent::ContentReplaced(String::new()))?;
                        continue;
                    }
                } else {
                    // Tool calls present — discard the degenerate content but
                    // keep the tool calls.
                    response.content = String::new();
                    send_event(events, AgentEvent::ContentReplaced(String::new()))?;
                }
            }

            // Check if context window is exhausted
            if self.context_size > 0 && response.prompt_eval_count > 0 {
                let total = response.prompt_eval_count + response.eval_count;
                let context_full = total >= self.context_size;
                // If tool calls are pending, also check if prompt alone is >90%
                // of context — tool results will push us over on the next iteration.
                let context_nearly_full = !response.tool_calls.is_empty()
                    && response.prompt_eval_count > self.context_size * 90 / 100;

                if context_full || context_nearly_full {
                    // Save any text the model managed to generate
                    if !response.content.trim().is_empty() {
                        let assistant_msg = Message::assistant(&response.content);
                        self.messages.push(assistant_msg.clone());
                        send_event(events, AgentEvent::MessageLogged(assistant_msg))?;
                    }
                    send_event(
                        events,
                        AgentEvent::Error(
                            "Context window is full — the model can no longer process \
                             this conversation. Use /clear to start fresh."
                                .to_string(),
                        ),
                    )?;
                    send_event(
                        events,
                        AgentEvent::Done {
                            prompt_tokens: response.prompt_eval_count,
                        },
                    )?;
                    return Ok(());
                }
            }

            if response.tool_calls.is_empty() {
                // Retry if the model returned a completely empty response
                if response.content.trim().is_empty() && empty_retries < MAX_EMPTY_RETRIES {
                    empty_retries += 1;
                    send_event(
                        events,
                        AgentEvent::Debug(format!(
                            "Empty response from model, retrying ({}/{})",
                            empty_retries, MAX_EMPTY_RETRIES
                        )),
                    )?;
                    continue;
                }

                let assistant_msg = Message::assistant(&response.content);
                self.messages.push(assistant_msg.clone());
                send_event(events, AgentEvent::MessageLogged(assistant_msg))?;
                send_event(
                    events,
                    AgentEvent::Done {
                        prompt_tokens: response.prompt_eval_count,
                    },
                )?;
                return Ok(());
            }

            empty_retries = 0;

            // If tool calls were extracted from text content, tell the UI to
            // replace the already-streamed raw JSON with the cleaned content.
            if response.tool_calls_from_content {
                send_event(
                    events,
                    AgentEvent::ContentReplaced(response.content.clone()),
                )?;
            }

            // Ensure all tool calls have IDs (needed for OpenAI-compatible backends).
            let tool_calls_with_ids: Vec<_> = response
                .tool_calls
                .into_iter()
                .map(|tc| if tc.id.is_some() { tc } else { tc.with_id() })
                .collect();

            let mut assistant_msg = Message::assistant(&response.content);
            assistant_msg.tool_calls = Some(tool_calls_with_ids.clone());
            self.messages.push(assistant_msg.clone());
            send_event(events, AgentEvent::MessageLogged(assistant_msg))?;

            for tool_call in &tool_calls_with_ids {
                let name = &tool_call.function.name;
                let args = &tool_call.function.arguments;

                // Format args for display based on tool type
                let args_display = match name.as_str() {
                    "bash" => args
                        .get("command")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    "read" => {
                        let path = args
                            .get("file_path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("?");
                        let offset = args.get("offset").and_then(|v| v.as_u64());
                        let limit = args.get("limit").and_then(|v| v.as_u64());
                        match (offset, limit) {
                            (Some(o), Some(l)) => format!("{}, offset={}, limit={}", path, o, l),
                            (Some(o), None) => format!("{}, offset={}", path, o),
                            (None, Some(l)) => format!("{}, limit={}", path, l),
                            (None, None) => path.to_string(),
                        }
                    }
                    "edit" => args
                        .get("file_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .to_string(),
                    "subagent" => args
                        .get("task")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .to_string(),
                    "glob" | "grep" => {
                        let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("?");
                        let path = args.get("path").and_then(|v| v.as_str());
                        match path {
                            Some(p) => format!("{} in {}", pattern, p),
                            None => pattern.to_string(),
                        }
                    }
                    _ => args.to_string(),
                };

                send_event(
                    events,
                    AgentEvent::ToolCall {
                        name: name.clone(),
                        args: args_display.clone(),
                    },
                )?;

                // Request user confirmation for tools that modify state
                let needs_confirm = matches!(name.as_str(), "bash" | "edit" | "write" | "subagent");
                if needs_confirm {
                    send_event(
                        events,
                        AgentEvent::ToolConfirmRequest {
                            name: name.clone(),
                            args: args_display,
                        },
                    )?;

                    let approved = confirm_rx.recv().await.unwrap_or(false);
                    if !approved {
                        let denied = "Tool execution denied by user.".to_string();
                        send_event(
                            events,
                            AgentEvent::ToolResult {
                                name: name.clone(),
                                output: denied.clone(),
                                success: false,
                            },
                        )?;
                        let tool_msg = Message::tool(&denied, tool_call.id.clone());
                        self.messages.push(tool_msg.clone());
                        send_event(events, AgentEvent::MessageLogged(tool_msg))?;
                        continue;
                    }
                }

                let (result, success) = if name == "bash" {
                    BashTool.execute_async(args, self.bash_timeout).await
                } else if name == "subagent" {
                    let task = args
                        .get("task")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    send_event(events, AgentEvent::SubagentStart { task: task.clone() })?;

                    let sub_agent = Agent::new_subagent(
                        Arc::clone(&self.backend),
                        self.model.clone(),
                        self.context_size,
                        self.bash_timeout,
                        self.subagent_max_turns,
                    );

                    // Channels for the sub-agent
                    let (sub_tx, mut sub_rx) = mpsc::unbounded_channel::<AgentEvent>();
                    let (sub_confirm_tx, mut sub_confirm_rx) = mpsc::unbounded_channel::<bool>();

                    // Move the sub-agent and channel clones into an owned future.
                    // This avoids borrow issues: the future owns everything it needs.
                    let sub_tx_for_agent = sub_tx.clone();
                    let cancel_for_sub = cancel.clone();
                    let mut sub_future = Box::pin(async move {
                        let mut sa = sub_agent;
                        let result = sa.run(&task, &sub_tx_for_agent, &mut sub_confirm_rx, cancel_for_sub).await;
                        let last_msg = sa.last_assistant_message();
                        (result, last_msg)
                    });

                    // Drive the sub-agent and its event loop concurrently.
                    // We poll both the sub-agent future and its event channel so we
                    // can forward tool confirmations through the parent's confirm_rx
                    // instead of auto-approving.
                    let mut sub_finished: Option<(Result<()>, String)> = None;
                    let mut sub_tx_option = Some(sub_tx);

                    loop {
                        tokio::select! {
                            biased;
                            // Process sub-agent events first (higher priority)
                            event = sub_rx.recv() => {
                                match event {
                                    Some(AgentEvent::ToolConfirmRequest { name, args }) => {
                                        // Forward to parent TUI for user confirmation
                                        send_event(events, AgentEvent::ToolConfirmRequest {
                                            name,
                                            args,
                                        })?;
                                        let approved = confirm_rx.recv().await.unwrap_or(false);
                                        let _ = sub_confirm_tx.send(approved);
                                    }
                                    Some(AgentEvent::ToolCall { name, args }) => {
                                        let _ = events.send(AgentEvent::SubagentToolCall {
                                            name,
                                            args,
                                        });
                                    }
                                    Some(AgentEvent::ToolResult { name, success, .. }) => {
                                        let _ = events.send(AgentEvent::SubagentToolResult {
                                            name,
                                            success,
                                        });
                                    }
                                    Some(_) => {} // ignore other events
                                    None => break, // sub-agent channel closed
                                }
                            }
                            // Poll the sub-agent future
                            result = &mut sub_future, if sub_finished.is_none() => {
                                sub_finished = Some(result);
                                // Drop our sender clone to close channel (the future's
                                // clone was already dropped when the async block ended).
                                sub_tx_option.take();
                            }
                        }
                    }

                    match sub_finished {
                        Some((Ok(()), response)) => {
                            send_event(events, AgentEvent::SubagentEnd { result: response.clone() })?;
                            (response, true)
                        }
                        Some((Err(e), _)) => {
                            let err_msg = format!("Sub-agent error: {}", e);
                            send_event(events, AgentEvent::SubagentEnd { result: err_msg.clone() })?;
                            (err_msg, false)
                        }
                        None => {
                            let err_msg = "Sub-agent channel closed unexpectedly".to_string();
                            send_event(events, AgentEvent::SubagentEnd { result: err_msg.clone() })?;
                            (err_msg, false)
                        }
                    }
                } else {
                    match self.tools.execute(name, args) {
                        Ok(output) => (output, true),
                        Err(e) => (format!("Error: {}", e), false),
                    }
                };

                send_event(
                    events,
                    AgentEvent::ToolResult {
                        name: name.clone(),
                        output: result.clone(),
                        success,
                    },
                )?;

                // Truncate before storing in context to avoid blowing the window.
                // The full output was already sent to the UI above.
                let context_result = truncate_tool_output(&result);
                let tool_msg = Message::tool(&context_result, tool_call.id.clone());
                self.messages.push(tool_msg.clone());
                send_event(events, AgentEvent::MessageLogged(tool_msg))?;
            }

        }
    }
}
