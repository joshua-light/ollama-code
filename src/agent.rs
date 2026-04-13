use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;

use crate::backend::ModelBackend;
use crate::config::Config;
use crate::hooks::HookRunner;
use crate::mcp;
use crate::message::{self, Message};
use crate::plugin::{self, ExternalTool};
use crate::skills::{self, SkillMeta};
use crate::tools::{BashTool, EditTool, GlobTool, GrepTool, ReadTool, SubagentToolDef, Tool, WriteTool, ToolRegistry};

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
    /// System prompt composition info (emitted once at first run).
    SystemPromptInfo {
        base_prompt_tokens: u64,
        project_docs: Vec<(String, u64)>,
        skills_tokens: u64,
        tool_defs_tokens: u64,
    },
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

/// Discover `CLAUDE.md` and `AGENTS.md` files by walking up from `cwd`.
/// Stops at the first directory that contains either file.
fn discover_project_docs(cwd: &str) -> Vec<(String, String)> {
    let mut dir = Path::new(cwd).to_path_buf();
    loop {
        let mut found = Vec::new();
        for name in &["CLAUDE.md", "AGENTS.md"] {
            let path = dir.join(name);
            if let Ok(content) = std::fs::read_to_string(&path) {
                found.push((name.to_string(), content));
            }
        }
        if !found.is_empty() {
            return found;
        }
        if !dir.pop() {
            break;
        }
    }
    Vec::new()
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
    /// Char count of the base system prompt (before project docs).
    system_prompt_base_len: usize,
    /// Loaded project doc files: (filename, char_count).
    project_docs_info: Vec<(String, usize)>,
    /// Char count of the skills summary appended to the system prompt.
    skills_summary_len: usize,
    /// Discovered skills (name + description only; body loaded on activation).
    skills: Vec<SkillMeta>,
    /// External plugin tool names that require user confirmation.
    plugin_confirm_tools: HashSet<String>,
    /// Running MCP server processes (kept alive so tool connections stay open).
    _mcp_servers: Vec<mcp::McpServer>,
    /// Hook runner for lifecycle hooks.
    hooks: HookRunner,
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
        config: Option<&Config>,
    ) -> Self {
        let default_config = Config::default();
        let cfg = config.unwrap_or(&default_config);

        let mut tools = ToolRegistry::new();
        let is_subagent = max_turns.is_some();

        macro_rules! register_if_enabled {
            ($name:expr, $tool:expr) => {
                if cfg.is_tool_enabled($name) {
                    tools.register(Box::new($tool));
                }
            };
        }
        register_if_enabled!("bash", BashTool);
        register_if_enabled!("read", ReadTool);
        register_if_enabled!("edit", EditTool);
        register_if_enabled!("write", WriteTool);
        register_if_enabled!("glob", GlobTool);
        register_if_enabled!("grep", GrepTool);
        if !is_subagent {
            register_if_enabled!("subagent", SubagentToolDef);
        }

        let mut plugin_confirm_tools = HashSet::new();
        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        if !is_subagent {
            let plugin_dirs: Vec<std::path::PathBuf> = cfg.plugin_dirs.as_ref()
                .map(|dirs| dirs.iter().map(std::path::PathBuf::from).collect())
                .unwrap_or_default();
            let discovered = plugin::discover_plugins(&cwd, &plugin_dirs, Some(cfg));
            for dp in &discovered {
                for tool_def in &dp.manifest.tools {
                    if !cfg.is_tool_enabled(&tool_def.name) {
                        continue;
                    }
                    let plugin_cfg = cfg.plugin_config(&dp.manifest.name).cloned();
                    let ext_tool = ExternalTool::from_plugin(dp, tool_def, plugin_cfg);
                    if ext_tool.needs_confirm() {
                        plugin_confirm_tools.insert(tool_def.name.clone());
                    }
                    tools.register(Box::new(ext_tool));
                }
            }
        }

        // Start MCP servers and register their tools.
        let mut mcp_servers = Vec::new();
        if !is_subagent {
            if let Some(ref servers) = cfg.mcp_servers {
                for server in mcp::start_servers(servers, |name| cfg.is_tool_enabled(name)) {
                    for mcp_tool in server.create_tools() {
                        let tool_name = mcp_tool.name().to_string();
                        if !cfg.is_tool_enabled(&tool_name) {
                            continue;
                        }
                        if server.needs_confirm {
                            plugin_confirm_tools.insert(tool_name);
                        }
                        tools.register(Box::new(mcp_tool));
                    }
                    mcp_servers.push(server);
                }
            }
        }

        let subagent_desc = if is_subagent {
            ""
        } else {
            "- subagent(task): Spawn a sub-agent with a fresh context to handle a focused task. \
             The sub-agent cannot see this conversation, so include all necessary context in the \
             task description. Use for: research across many files, complex multi-step operations, \
             or any task that would benefit from a clean context window"
        };

        let mut system_prompt = include_str!("../SYSTEM_PROMPT.md")
            .replace("{cwd}", &cwd)
            .replace("{subagent_tool}", subagent_desc);

        let system_prompt_base_len = system_prompt.len();

        // Load project docs (CLAUDE.md / AGENTS.md) so all agents follow conventions.
        let project_docs_info = {
            let docs = discover_project_docs(&cwd);
            let mut info = Vec::new();
            for (name, content) in &docs {
                system_prompt.push_str(&format!(
                    "\n\n---\n\n# Project Instructions ({})\n\n{}",
                    name, content
                ));
                info.push((name.clone(), content.len()));
            }
            info
        };

        // Discover skills (.agents/skills/*/SKILL.md) for top-level agents only.
        // Only name + description are injected into the system prompt (discovery layer).
        let (discovered_skills, skills_summary_len) = if !is_subagent {
            let found = skills::discover_skills(&cwd);
            let len = if !found.is_empty() {
                let summary = skills::format_skill_summaries(&found);
                let l = summary.len();
                system_prompt.push_str(&summary);
                l
            } else {
                0
            };
            (found, len)
        } else {
            (Vec::new(), 0)
        };

        // Discover hooks (top-level agents only).
        let hooks = if !is_subagent {
            HookRunner::discover(&cwd, config)
        } else {
            HookRunner::empty()
        };

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
            system_prompt_base_len,
            project_docs_info,
            skills_summary_len,
            skills: discovered_skills,
            plugin_confirm_tools,
            _mcp_servers: mcp_servers,
            hooks,
        }
    }

    pub fn new(
        backend: Arc<dyn ModelBackend>,
        model: String,
        context_size: u64,
        bash_timeout: std::time::Duration,
        subagent_max_turns: u16,
    ) -> Self {
        Self::build(backend, model, context_size, bash_timeout, subagent_max_turns, None, None)
    }

    /// Create an agent with custom config (controls which tools are enabled, etc.).
    pub fn with_config(
        backend: Arc<dyn ModelBackend>,
        model: String,
        context_size: u64,
        bash_timeout: std::time::Duration,
        subagent_max_turns: u16,
        config: &Config,
    ) -> Self {
        Self::build(backend, model, context_size, bash_timeout, subagent_max_turns, None, Some(config))
    }

    /// Create a sub-agent with fresh context and no subagent tool (prevents recursion).
    fn new_subagent(
        backend: Arc<dyn ModelBackend>,
        model: String,
        context_size: u64,
        bash_timeout: std::time::Duration,
        max_turns: u16,
    ) -> Self {
        Self::build(backend, model, context_size, bash_timeout, 0, Some(max_turns), None)
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn skills(&self) -> &[SkillMeta] {
        &self.skills
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

    /// Restore conversation from a previously saved session.
    /// Replaces the loaded system prompt with the agent's fresh one (current cwd/tools),
    /// and marks the system prompt as already logged to prevent re-logging.
    pub fn restore_messages(&mut self, mut messages: Vec<Message>) {
        // Drop the old system prompt from the saved messages
        if !messages.is_empty() && matches!(messages[0].role, crate::message::Role::System) {
            messages.remove(0);
        }
        // Prepend our fresh system prompt
        let fresh_system = self.messages[0].clone();
        messages.insert(0, fresh_system);
        self.messages = messages;
        self.system_prompt_logged = true;
    }

    /// Remove the last `n` user turns from the message history.
    /// The system prompt (index 0) is always preserved.
    pub fn rewind_turns(&mut self, n: usize) {
        let user_indices: Vec<usize> = self
            .messages
            .iter()
            .enumerate()
            .filter_map(|(i, m)| matches!(m.role, crate::message::Role::User).then_some(i))
            .collect();

        if user_indices.is_empty() {
            return;
        }

        let actual_n = n.min(user_indices.len());
        let truncate_at = user_indices[user_indices.len() - actual_n];
        self.messages.truncate(truncate_at);
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

    /// Force a final text response when the sub-agent turn limit is exceeded.
    /// Injects a summary request, calls the backend without tools, logs the
    /// assistant message, and emits `Done`.
    async fn force_final_response(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
        num_ctx: Option<u64>,
    ) -> Result<()> {
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
        send_event(events, AgentEvent::Done { prompt_tokens: 0, eval_count: 0 })?;
        Ok(())
    }

    /// Execute a sub-agent for the given `task`, forwarding its events and
    /// tool-confirmation requests to the parent's channels.
    /// Returns `(result_text, success)`.
    async fn execute_subagent(
        &self,
        task: &str,
        events: &mpsc::UnboundedSender<AgentEvent>,
        confirm_rx: &mut mpsc::UnboundedReceiver<bool>,
        cancel: &Arc<AtomicBool>,
    ) -> (String, bool) {
        send_event(events, AgentEvent::SubagentStart { task: task.to_string() })
            .unwrap_or(());

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
        let task_owned = task.to_string();
        let mut sub_future = Box::pin(async move {
            let mut sa = sub_agent;
            let result = sa
                .run(&task_owned, &sub_tx_for_agent, &mut sub_confirm_rx, cancel_for_sub)
                .await;
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
                            let _ = send_event(events, AgentEvent::ToolConfirmRequest { name, args });
                            let approved = confirm_rx.recv().await.unwrap_or(false);
                            let _ = sub_confirm_tx.send(approved);
                        }
                        Some(AgentEvent::ToolCall { name, args }) => {
                            let _ = events.send(AgentEvent::SubagentToolCall { name, args });
                        }
                        Some(AgentEvent::ToolResult { name, success, .. }) => {
                            let _ = events.send(AgentEvent::SubagentToolResult { name, success });
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
                let _ = send_event(events, AgentEvent::SubagentEnd { result: response.clone() });
                (response, true)
            }
            Some((Err(e), _)) => {
                let err_msg = format!("Sub-agent error: {}", e);
                let _ = send_event(events, AgentEvent::SubagentEnd { result: err_msg.clone() });
                (err_msg, false)
            }
            None => {
                let err_msg = "Sub-agent channel closed unexpectedly".to_string();
                let _ = send_event(events, AgentEvent::SubagentEnd { result: err_msg.clone() });
                (err_msg, false)
            }
        }
    }

    /// Dispatch a single tool call: run pre-hooks, handle confirmation,
    /// execute the tool (bash / subagent / registry), run post-hooks,
    /// emit events, and append the result message.
    /// Returns `Ok(true)` normally, `Ok(false)` when the user cancelled
    /// mid-execution (caller should return early).
    async fn dispatch_tool_call(
        &mut self,
        tool_call: &crate::message::ToolCall,
        events: &mpsc::UnboundedSender<AgentEvent>,
        confirm_rx: &mut mpsc::UnboundedReceiver<bool>,
        cancel: &Arc<AtomicBool>,
    ) -> Result<bool> {
        let name = &tool_call.function.name;
        let original_args = &tool_call.function.arguments;

        // --- pre_tool_execute hooks ---
        let args_override = match self.hooks.pre_tool_execute(name, original_args).await {
            Ok(pre) => match pre.action.as_deref() {
                Some("deny") => {
                    let msg = pre
                        .message
                        .unwrap_or_else(|| "Blocked by hook.".to_string());
                    send_event(
                        events,
                        AgentEvent::ToolCall {
                            name: name.clone(),
                            args: crate::format::format_tool_args_display(name, original_args),
                        },
                    )?;
                    send_event(
                        events,
                        AgentEvent::ToolResult {
                            name: name.clone(),
                            output: msg.clone(),
                            success: false,
                        },
                    )?;
                    let tool_msg = Message::tool(&msg, tool_call.id.clone());
                    self.messages.push(tool_msg.clone());
                    send_event(events, AgentEvent::MessageLogged(tool_msg))?;
                    return Ok(true);
                }
                Some("modify") => pre.arguments,
                _ => None, // "proceed" or absent
            },
            Err(e) => {
                send_event(
                    events,
                    AgentEvent::Debug(format!("pre_tool_execute hook error: {}", e)),
                )?;
                None
            }
        };
        let args = args_override.as_ref().unwrap_or(original_args);

        let args_display = crate::format::format_tool_args_display(name, args);

        send_event(
            events,
            AgentEvent::ToolCall {
                name: name.clone(),
                args: args_display.clone(),
            },
        )?;

        // Request user confirmation for tools that modify state
        let needs_confirm = matches!(name.as_str(), "bash" | "edit" | "write" | "subagent")
            || self.plugin_confirm_tools.contains(name.as_str());
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
                return Ok(true);
            }
        }

        // Check cancellation before each tool call
        if cancel.load(Ordering::Relaxed) {
            send_event(events, AgentEvent::Cancelled)?;
            return Ok(false);
        }

        let (mut result, mut success) = if name == "bash" {
            // Race tool execution against the cancel flag so ESC kills
            // long-running bash commands immediately.
            tokio::select! {
                output = BashTool.execute_async(args, self.bash_timeout) => output,
                _ = poll_cancel(cancel) => {
                    send_event(events, AgentEvent::Cancelled)?;
                    return Ok(false);
                }
            }
        } else if name == "subagent" {
            let task = args
                .get("task")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            self.execute_subagent(&task, events, confirm_rx, cancel).await
        } else {
            match self.tools.execute(name, args) {
                Ok(output) => (output, true),
                Err(e) => (format!("Error: {}", e), false),
            }
        };

        // --- post_tool_execute hooks ---
        let (post_result, post_success) = self
            .hooks
            .post_tool_execute(name, args, result, success)
            .await;
        result = post_result;
        success = post_success;

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

        Ok(true)
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
            // Estimate tool definition tokens from their JSON representation.
            let tool_defs_chars: usize = self.tools.definitions().iter()
                .map(|d| d.to_string().len())
                .sum();
            send_event(events, AgentEvent::SystemPromptInfo {
                base_prompt_tokens: message::estimate_tokens(self.system_prompt_base_len),
                project_docs: self.project_docs_info.iter()
                    .map(|(name, len)| (name.clone(), message::estimate_tokens(*len)))
                    .collect(),
                skills_tokens: message::estimate_tokens(self.skills_summary_len),
                tool_defs_tokens: message::estimate_tokens(tool_defs_chars),
            })?;
            self.system_prompt_logged = true;
        }

        // --- agent_start hooks ---
        if let Some(extra_context) = self.hooks.agent_start(user_input, &self.model).await {
            let ctx_msg = Message::system(format!(
                "[Hook-injected context]\n{}",
                extra_context
            ));
            self.messages.push(ctx_msg);
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
                    self.force_final_response(events, num_ctx).await?;
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
                            send_event(events, AgentEvent::Done { prompt_tokens: 0, eval_count: 0 })?;
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
                            eval_count: response.eval_count,
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

                // --- agent_done hooks ---
                let final_content = match self
                    .hooks
                    .agent_done(&response.content, turn, &self.model)
                    .await
                {
                    Some(rewritten) => {
                        send_event(
                            events,
                            AgentEvent::ContentReplaced(rewritten.clone()),
                        )?;
                        rewritten
                    }
                    None => response.content.clone(),
                };

                let assistant_msg = Message::assistant(&final_content);
                self.messages.push(assistant_msg.clone());
                send_event(events, AgentEvent::MessageLogged(assistant_msg))?;
                send_event(
                    events,
                    AgentEvent::Done {
                        prompt_tokens: response.prompt_eval_count,
                        eval_count: response.eval_count,
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
                let continue_loop = self
                    .dispatch_tool_call(tool_call, events, confirm_rx, &cancel)
                    .await?;
                if !continue_loop {
                    return Ok(());
                }
            }

        }
    }
}
