use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

use crate::backend::ModelBackend;
use crate::config::Config;
use crate::format;
use crate::llama_server::ModelSource;
use crate::message::{Message, Role};
use crate::ollama::OllamaBackend;
use crate::skills::SkillMeta;

pub(crate) enum AgentInput {
    Message(String),
    ClearHistory,
    Rewind(usize),
    SetModel(String),
    SetContextSize(u64),
    SetBackend(Arc<dyn ModelBackend>),
    RestoreMessages(Vec<Message>),
}

#[derive(Clone)]
pub(crate) struct ToolResultData {
    pub(crate) output: String,
    pub(crate) success: bool,
}

#[derive(Clone)]
pub(crate) enum ChatMessage {
    User(String),
    Assistant(String),
    ToolCall {
        name: String,
        args: String,
        result: Option<ToolResultData>,
    },
    Error(String),
    Info(String),
    ContextInfo {
        context_used: u64,
        context_size: u64,
        user_messages: u32,
        assistant_messages: u32,
        tool_calls: u32,
        base_prompt_tokens: u64,
        project_docs_tokens: Vec<(String, u64)>,
        skills_tokens: u64,
        tool_defs_tokens: u64,
    },
    GenerationSummary { duration: std::time::Duration },
    SubagentToolCall {
        name: String,
        args: String,
        success: Option<bool>,
    },
}

pub(crate) struct PendingConfirm {
    pub(crate) name: String,
    pub(crate) args: String,
}

/// Deferred llama-server start — stored in App so the event loop can stop the
/// old server before spawning the new one (avoids VRAM contention).
pub(crate) struct PendingServerStart {
    pub(crate) server_path: String,
    pub(crate) model_source: ModelSource,
    pub(crate) ctx: u64,
    pub(crate) extra_args: Vec<String>,
    pub(crate) model_name: String,
    /// If set, unload this Ollama model before starting the server.
    pub(crate) unload: Option<(OllamaBackend, String)>,
}

/// State for the initial llama-server loading progress display.
pub(crate) struct ServerLoadingState {
    /// 0.0 to 1.0 — parsed from llama-server /health endpoint.
    pub(crate) progress: f32,
    /// When loading started, for elapsed time display.
    pub(crate) start: Instant,
    /// Model name being loaded, for display.
    pub(crate) model_name: String,
}

pub(crate) struct App {
    pub(crate) messages: Vec<ChatMessage>,
    pub(crate) current_response: String,
    pub(crate) input: String,
    pub(crate) cursor_pos: usize,
    pub(crate) is_processing: bool,
    pub(crate) model: String,
    pub(crate) should_quit: bool,
    pub(crate) scroll_offset: u16,
    pub(crate) max_scroll: u16,
    // Generation tracking
    pub(crate) generation_start: Option<Instant>,
    pub(crate) generation_tokens: usize,
    pub(crate) generation_verb: String,
    pub(crate) has_received_tokens: bool,
    // Status line
    pub(crate) dir_name: String,
    pub(crate) context_size: u64,
    pub(crate) context_used: u64,
    pub(crate) session_start: Instant,
    pub(crate) tool_call_count: usize,
    pub(crate) total_input_tokens: u64,
    pub(crate) total_output_tokens: u64,
    pub(crate) tools_expanded: bool,
    pub(crate) git_branch: Option<String>,
    pub(crate) git_dirty: bool,
    // Model selection
    pub(crate) ollama: OllamaBackend,
    pub(crate) model_choices: Option<Vec<String>>,
    pub(crate) config: Config,
    /// Set by model selection to signal the event loop to stop the llama-server
    pub(crate) stop_llama_server: bool,
    /// Pending tool confirmation awaiting user response
    pub(crate) pending_confirm: Option<PendingConfirm>,
    /// When true, all tool confirmations are auto-approved
    pub(crate) auto_approve: bool,
    /// Force a full terminal clear before the next draw (e.g. after expand/collapse)
    pub(crate) needs_clear: bool,
    /// Deferred llama-server start (event loop stops old server first, then spawns this)
    pub(crate) pending_server_start: Option<PendingServerStart>,
    /// Cancel flag shared with the agent task — set to true to request cancellation.
    pub(crate) cancel_flag: Arc<AtomicBool>,
    /// Estimated tokens for the base system prompt (SYSTEM_PROMPT.md).
    pub(crate) base_prompt_tokens: u64,
    /// Estimated tokens for each loaded project doc (filename, tokens).
    pub(crate) project_docs_tokens: Vec<(String, u64)>,
    /// Estimated tokens for skill summaries in the system prompt.
    pub(crate) skills_tokens: u64,
    /// Estimated tokens for tool definitions sent with each request.
    pub(crate) tool_defs_tokens: u64,
    /// Discovered skills available as slash commands.
    pub(crate) skills: Vec<SkillMeta>,
    /// When Some, the initial llama-server is still loading and we show a progress bar.
    pub(crate) server_loading: Option<ServerLoadingState>,
}

impl App {
    pub(crate) fn new(model: String, context_size: u64, ollama: OllamaBackend, config: Config, skills: Vec<SkillMeta>) -> Self {
        let (git_branch, git_dirty) = super::render::get_git_info_sync();
        let dir_name = std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .unwrap_or_else(|| ".".to_string());
        Self {
            messages: Vec::new(),
            current_response: String::new(),
            input: String::new(),
            cursor_pos: 0,
            is_processing: false,
            model,
            should_quit: false,
            scroll_offset: 0,
            max_scroll: 0,
            generation_start: None,
            generation_tokens: 0,
            generation_verb: super::render::pick_verb(),
            has_received_tokens: false,
            dir_name,
            context_size,
            context_used: 0,
            session_start: Instant::now(),
            tool_call_count: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            tools_expanded: false,
            git_branch,
            git_dirty,
            ollama,
            model_choices: None,
            config,
            stop_llama_server: false,
            pending_confirm: None,
            auto_approve: false,
            needs_clear: false,
            pending_server_start: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            base_prompt_tokens: 0,
            project_docs_tokens: Vec::new(),
            skills_tokens: 0,
            tool_defs_tokens: 0,
            skills,
            server_loading: None,
        }
    }

    pub(crate) fn submit(&mut self) -> Option<String> {
        if self.input.trim().is_empty() || self.is_processing {
            return None;
        }
        let msg = self.input.clone();
        self.input.clear();
        self.cursor_pos = 0;
        self.messages.push(ChatMessage::User(msg.clone()));
        self.is_processing = true;
        self.generation_start = Some(Instant::now());
        self.generation_tokens = 0;
        self.generation_verb = super::render::pick_verb();
        self.has_received_tokens = false;
        Some(msg)
    }

    pub(crate) fn flush_streaming(&mut self) {
        if !self.current_response.is_empty() {
            self.messages.push(ChatMessage::Assistant(std::mem::take(
                &mut self.current_response,
            )));
        }
    }

    pub(crate) fn scroll_up(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_add(lines).min(self.max_scroll);
    }

    pub(crate) fn scroll_down(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
    }

    pub(crate) fn is_at_bottom(&self) -> bool {
        self.scroll_offset == 0
    }

    pub(crate) fn backspace(&mut self) {
        if self.cursor_pos > 0 {
            let prev = self.input[..self.cursor_pos]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.input.remove(prev);
            self.cursor_pos = prev;
        }
    }

    pub(crate) fn dismiss_model_chooser(&mut self) {
        self.model_choices = None;
        self.input.clear();
        self.cursor_pos = 0;
    }

    /// Remove the last `n` user turns (user message + all subsequent non-user messages).
    /// Returns how many turns were actually removed.
    pub(crate) fn rewind_turns(&mut self, n: usize, input_tx: &mpsc::UnboundedSender<AgentInput>) -> usize {
        let user_indices: Vec<usize> = self
            .messages
            .iter()
            .enumerate()
            .filter_map(|(i, m)| matches!(m, ChatMessage::User(_)).then_some(i))
            .collect();

        if user_indices.is_empty() {
            return 0;
        }

        let actual_n = n.min(user_indices.len());
        let truncate_at = user_indices[user_indices.len() - actual_n];
        self.messages.truncate(truncate_at);

        let _ = input_tx.send(AgentInput::Rewind(actual_n));

        self.scroll_offset = 0;
        actual_n
    }

    pub(crate) fn reset_conversation(&mut self, input_tx: &mpsc::UnboundedSender<AgentInput>, msg: &str) {
        self.messages.clear();
        self.current_response.clear();
        self.context_used = 0;
        self.tool_call_count = 0;
        self.total_input_tokens = 0;
        self.total_output_tokens = 0;
        self.scroll_offset = 0;
        self.max_scroll = 0;
        let _ = input_tx.send(AgentInput::ClearHistory);
        self.messages.push(ChatMessage::Info(msg.into()));
    }
}

/// Convert a Vec<Message> (from a saved session) into Vec<ChatMessage> for TUI display.
pub(crate) fn messages_to_chat_messages(messages: &[Message]) -> Vec<ChatMessage> {
    let mut chat_msgs = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {} // not displayed
            Role::User => {
                chat_msgs.push(ChatMessage::User(msg.content.clone()));
            }
            Role::Assistant => {
                // If the assistant message has non-empty content, show it
                if !msg.content.trim().is_empty() {
                    chat_msgs.push(ChatMessage::Assistant(msg.content.clone()));
                }
                // If it has tool calls, add those
                if let Some(ref tool_calls) = msg.tool_calls {
                    for tc in tool_calls {
                        let args_display = format::format_tool_args_display(
                            &tc.function.name,
                            &tc.function.arguments,
                        );
                        chat_msgs.push(ChatMessage::ToolCall {
                            name: tc.function.name.clone(),
                            args: args_display,
                            result: None,
                        });
                    }
                }
            }
            Role::Tool => {
                // Fill in the result on the last pending ToolCall.
                // Orphaned tool results from saved sessions are expected; skip silently.
                for cm in chat_msgs.iter_mut().rev() {
                    if let ChatMessage::ToolCall { result, .. } = cm {
                        if result.is_none() {
                            *result = Some(ToolResultData {
                                output: msg.content.clone(),
                                success: true,
                            });
                            break;
                        }
                    }
                }
            }
        }
    }

    chat_msgs
}
