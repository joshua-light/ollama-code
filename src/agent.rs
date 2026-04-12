use anyhow::Result;
use tokio::sync::mpsc;

use crate::message::Message;
use crate::ollama::OllamaClient;
use crate::tools::{BashTool, EditTool, ReadTool, WriteTool, ToolRegistry};

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
    Error(String),
    MessageLogged(Message),
    Debug(String),
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
    ollama: OllamaClient,
    tools: ToolRegistry,
    model: String,
    messages: Vec<Message>,
    system_prompt_logged: bool,
    context_size: u64,
    bash_timeout: std::time::Duration,
}

impl Agent {
    pub fn new(ollama: OllamaClient, model: String, context_size: u64, bash_timeout: std::time::Duration) -> Self {
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(BashTool));
        tools.register(Box::new(ReadTool));
        tools.register(Box::new(EditTool));
        tools.register(Box::new(WriteTool));

        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let system_prompt = include_str!("../SYSTEM_PROMPT.md").replace("{cwd}", &cwd);

        let messages = vec![Message::system(&system_prompt)];

        Self {
            ollama,
            tools,
            model,
            messages,
            system_prompt_logged: false,
            context_size,
            bash_timeout,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }

    pub fn set_client(&mut self, client: OllamaClient) {
        self.ollama = client;
    }

    pub fn set_context_size(&mut self, size: u64) {
        self.context_size = size;
    }

    pub fn clear_history(&mut self) {
        self.messages.truncate(1); // keep system prompt
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
    ) -> Result<()> {
        // Log system prompt on first run
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

        loop {
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

            let num_ctx = if self.context_size > 0 {
                Some(self.context_size)
            } else {
                None
            };

            let response = match self
                .ollama
                .chat(&self.model, &self.messages, Some(tool_defs), num_ctx, move |token| {
                    let _ = events_clone.send(AgentEvent::Token(token.to_string()));
                })
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    send_event(events, AgentEvent::Error(e.to_string()))?;
                    send_event(events, AgentEvent::Done { prompt_tokens: 0 })?;
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
                send_event(
                    events,
                    AgentEvent::Error(
                        "Ollama stream ended unexpectedly (connection may have dropped)".to_string(),
                    ),
                )?;
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

            // Got a non-empty response, reset retry counter
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
                let needs_confirm = matches!(name.as_str(), "bash" | "edit" | "write");
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
                    let command = args
                        .get("command")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let timeout_dur = self.bash_timeout;
                    match tokio::process::Command::new("bash")
                        .arg("-c")
                        .arg(&command)
                        .stdout(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::piped())
                        .kill_on_drop(true)
                        .spawn()
                    {
                        Ok(child) => {
                            match tokio::time::timeout(timeout_dur, child.wait_with_output()).await {
                                Ok(Ok(output)) => crate::tools::format_bash_output(&output),
                                Ok(Err(e)) => (format!("Error: {}", e), false),
                                Err(_) => (
                                    format!("Error: command timed out after {}s", timeout_dur.as_secs()),
                                    false,
                                ),
                            }
                        }
                        Err(e) => (format!("Error: {}", e), false),
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

                let tool_msg = Message::tool(&result, tool_call.id.clone());
                self.messages.push(tool_msg.clone());
                send_event(events, AgentEvent::MessageLogged(tool_msg))?;
            }

        }
    }
}
