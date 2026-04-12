use anyhow::Result;
use tokio::sync::mpsc;

use crate::message::Message;
use crate::ollama::OllamaClient;
use crate::tools::{BashTool, EditTool, ReadTool, WriteTool, ToolRegistry};

#[derive(Debug, Clone)]
pub enum AgentEvent {
    Token(String),
    ToolCall { name: String, args: String },
    ToolResult { name: String, output: String, success: bool },
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
}

impl Agent {
    pub fn new(ollama: OllamaClient, model: String, context_size: u64) -> Self {
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
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }

    pub fn set_context_size(&mut self, size: u64) {
        self.context_size = size;
    }

    pub fn clear_history(&mut self) {
        self.messages.truncate(1); // keep system prompt
    }

    pub async fn run(
        &mut self,
        user_input: &str,
        events: &mpsc::UnboundedSender<AgentEvent>,
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

            if response.incomplete {
                send_event(
                    events,
                    AgentEvent::Error(
                        "Ollama stream ended unexpectedly (connection may have dropped)".to_string(),
                    ),
                )?;
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

            let mut assistant_msg = Message::assistant(&response.content);
            assistant_msg.tool_calls = Some(response.tool_calls.clone());
            self.messages.push(assistant_msg.clone());
            send_event(events, AgentEvent::MessageLogged(assistant_msg))?;

            for tool_call in &response.tool_calls {
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
                        args: args_display,
                    },
                )?;

                let (result, success) = match self.tools.execute(name, args) {
                    Ok(output) => (output, true),
                    Err(e) => (format!("Error: {}", e), false),
                };

                send_event(
                    events,
                    AgentEvent::ToolResult {
                        name: name.clone(),
                        output: result.clone(),
                        success,
                    },
                )?;

                let tool_msg = Message::tool(&result);
                self.messages.push(tool_msg.clone());
                send_event(events, AgentEvent::MessageLogged(tool_msg))?;
            }

        }
    }
}
