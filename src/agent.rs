use anyhow::Result;
use tokio::sync::mpsc;

use crate::message::Message;
use crate::ollama::OllamaClient;
use crate::tools::{BashTool, ToolRegistry};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum AgentEvent {
    Token(String),
    ToolCall { name: String, args: String },
    ToolResult { name: String, output: String },
    Done,
    Error(String),
}

pub struct Agent {
    ollama: OllamaClient,
    tools: ToolRegistry,
    model: String,
    messages: Vec<Message>,
}

impl Agent {
    pub fn new(ollama: OllamaClient, model: String) -> Self {
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(BashTool));

        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let system_prompt = format!(
            "You are Ollama Code, a helpful terminal assistant. You help users by executing commands and providing clear, concise answers.\n\
             \n\
             Your working directory is: {cwd}\n\
             \n\
             You have access to a bash tool that lets you execute shell commands. Use it to:\n\
             - Read or write files\n\
             - Explore directories\n\
             - Run programs and scripts\n\
             - Check system information\n\
             - Perform any terminal operation\n\
             \n\
             Be concise and direct. When a task requires running commands, use the bash tool proactively. Show relevant output and summarize results clearly."
        );

        let messages = vec![Message::system(&system_prompt)];

        Self {
            ollama,
            tools,
            model,
            messages,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn run(
        &mut self,
        user_input: &str,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<()> {
        self.messages.push(Message::user(user_input));

        loop {
            let tool_defs = self.tools.definitions();
            let events_clone = events.clone();

            let response = match self
                .ollama
                .chat(&self.model, &self.messages, Some(tool_defs), move |token| {
                    let _ = events_clone.send(AgentEvent::Token(token.to_string()));
                })
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = events.send(AgentEvent::Error(e.to_string()));
                    let _ = events.send(AgentEvent::Done);
                    return Ok(());
                }
            };

            if response.tool_calls.is_empty() {
                self.messages.push(Message::assistant(&response.content));
                let _ = events.send(AgentEvent::Done);
                return Ok(());
            }

            // Add assistant message with tool calls to history
            let mut assistant_msg = Message::assistant(&response.content);
            assistant_msg.tool_calls = Some(response.tool_calls.clone());
            self.messages.push(assistant_msg);

            // Execute each tool call
            for tool_call in &response.tool_calls {
                let name = &tool_call.function.name;
                let args = &tool_call.function.arguments;

                // Format args for display
                let args_display =
                    if let Some(cmd) = args.get("command").and_then(|v| v.as_str()) {
                        cmd.to_string()
                    } else {
                        args.to_string()
                    };

                let _ = events.send(AgentEvent::ToolCall {
                    name: name.clone(),
                    args: args_display,
                });

                let result = match self.tools.execute(name, args) {
                    Ok(output) => output,
                    Err(e) => format!("Error: {}", e),
                };

                let _ = events.send(AgentEvent::ToolResult {
                    name: name.clone(),
                    output: result.clone(),
                });

                self.messages.push(Message::tool(result));
            }

            // Continue the loop — model will now respond to tool results
        }
    }
}
