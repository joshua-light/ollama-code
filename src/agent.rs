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
use crate::tools::{BashTool, EditTool, GlobTool, GrepTool, ReadTool, SkillTool, SubagentToolDef, Tool, WriteTool, ToolRegistry};

/// How `rewind_turns` / `rewind_leaf` treats the anchor user message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewindMode {
    /// Remove the anchor user message itself (undo that turn).
    UndoTurn,
    /// Preserve the anchor; drop only the messages after it.
    RewindTo,
}

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

/// Format messages for the context compaction prompt.
fn format_messages_for_compaction(messages: &[Message]) -> String {
    use crate::format::truncate_args;
    use crate::message::Role;
    use std::fmt::Write;

    let mut out = String::new();
    for msg in messages {
        match msg.role {
            Role::User => {
                out.push_str("[User] ");
                out.push_str(&truncate_args(&msg.content, 500));
                out.push('\n');
            }
            Role::Assistant => {
                if !msg.content.is_empty() {
                    out.push_str("[Assistant] ");
                    out.push_str(&truncate_args(&msg.content, 500));
                    out.push('\n');
                }
                if let Some(ref calls) = msg.tool_calls {
                    for tc in calls {
                        let _ = writeln!(
                            out,
                            "[Tool call: {}({})]",
                            tc.function.name,
                            truncate_args(&tc.function.arguments.to_string(), 150),
                        );
                    }
                }
            }
            Role::Tool => {
                let status = match msg.success {
                    Some(true) => "success",
                    Some(false) => "failure",
                    None => "unknown",
                };
                let _ = write!(out, "[Tool result ({})]: ", status);
                out.push_str(&truncate_args(&msg.content, 200));
                out.push('\n');
            }
            Role::System => {
                out.push_str("[System] ");
                out.push_str(&truncate_args(&msg.content, 200));
                out.push('\n');
            }
        }
    }
    out
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

/// Exponential backoff delay: 500ms * 2^retries, capped at 2^5 = 32x.
fn retry_backoff_delay(retries: u32) -> std::time::Duration {
    const RETRY_BASE_MS: u64 = 500;
    std::time::Duration::from_millis(RETRY_BASE_MS * (1 << retries.min(5)))
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

/// Collapse 3+ consecutive newlines to 2 (one blank line max).
fn normalize_newlines(text: String) -> String {
    let mut result = String::with_capacity(text.len());
    let mut newline_count = 0u32;
    for ch in text.chars() {
        if ch == '\n' {
            newline_count += 1;
            if newline_count <= 2 {
                result.push(ch);
            }
        } else {
            newline_count = 0;
            result.push(ch);
        }
    }
    result
}

/// Build the tool registry, discover plugins, and start MCP servers.
/// When `include_extensions` is false (sub-agents), omits subagent tool, plugins, and MCP.
fn build_tools_and_servers(
    cfg: &Config,
    context_size: u64,
    cwd: &str,
    include_extensions: bool,
) -> (ToolRegistry, HashSet<String>, Vec<mcp::McpServer>) {
    let mut tools = ToolRegistry::new();
    macro_rules! register_if_enabled {
        ($name:expr, $tool:expr) => {
            if cfg.is_tool_enabled($name) {
                tools.register(Box::new($tool));
            }
        };
    }
    register_if_enabled!("bash", BashTool);
    register_if_enabled!("read", ReadTool::new());
    register_if_enabled!("edit", EditTool);
    register_if_enabled!("write", WriteTool);
    register_if_enabled!("glob", GlobTool);
    register_if_enabled!("grep", GrepTool::new(context_size));

    let mut plugin_confirm_tools = HashSet::new();
    let mut mcp_servers = Vec::new();

    if include_extensions {
        register_if_enabled!("subagent", SubagentToolDef);

        let plugin_dirs: Vec<std::path::PathBuf> = cfg
            .plugin_dirs
            .as_ref()
            .map(|dirs| dirs.iter().map(std::path::PathBuf::from).collect())
            .unwrap_or_default();
        let discovered = plugin::discover_plugins(cwd, &plugin_dirs, Some(cfg));
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

        if let Some(ref servers) = cfg.mcp {
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

    (tools, plugin_confirm_tools, mcp_servers)
}

/// Build the system prompt from template, skills, and project docs.
fn build_system_prompt(
    cwd: &str,
    include_extensions: bool,
    discovered_skills: &[SkillMeta],
) -> (String, usize, usize, Vec<(String, usize)>) {
    let subagent_desc = if include_extensions {
        "- subagent(task): Spawn a sub-agent with a fresh context to handle a focused task. \
         The sub-agent cannot see this conversation, so include all necessary context in the \
         task description. Use for: research across many files, complex multi-step operations, \
         or any task that would benefit from a clean context window"
    } else {
        ""
    };

    let skill_desc = if !discovered_skills.is_empty() {
        "- skill(name, args?): Activate a skill by name. Use when a task matches an available skill"
    } else {
        ""
    };

    let config_dir = crate::config::config_dir();

    let mut prompt = include_str!("../SYSTEM_PROMPT.md")
        .replace("{cwd}", cwd)
        .replace("{config_dir}", &config_dir.to_string_lossy())
        .replace("{subagent_tool}", subagent_desc)
        .replace("{skill_tool}", skill_desc);

    let base_len = prompt.len();

    let skills_summary_len = if !discovered_skills.is_empty() {
        let summary = skills::format_skill_summaries(discovered_skills);
        let l = summary.len();
        prompt.push_str(&summary);
        l
    } else {
        0
    };

    let project_docs_info = {
        let docs = discover_project_docs(cwd);
        let mut info = Vec::new();
        for (name, content) in &docs {
            prompt.push_str(&format!(
                "\n\n---\n\n# Project Instructions ({})\n\n{}",
                name, content
            ));
            info.push((name.clone(), content.len()));
        }
        info
    };

    let prompt = normalize_newlines(prompt);

    (prompt, base_len, skills_summary_len, project_docs_info)
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
    /// Stored config for propagation to sub-agents.
    config: Option<Config>,

    // --- Small-model improvement flags ---
    /// When true, dynamically scope available tools per turn.
    tool_scoping: bool,
    /// When true, re-inject the task objective every N turns.
    task_reinjection: bool,
    /// How often to re-inject (in agent turns).
    reinjection_interval: u16,
    /// Context trim threshold (percentage of context_size).
    trim_threshold_pct: u8,
    /// Context trim target (percentage of context_size).
    trim_target_pct: u8,
    /// Whether to use LLM-based context compaction instead of blind trimming.
    context_compaction: bool,
    /// Tracks whether a file-reading tool has been used (for tool scoping).
    has_explored: bool,
    /// The original user input for the current run (for task re-injection).
    current_task: String,
    /// Files read during this agent's lifetime: (path, start_line_0, end_line_0).
    /// Used to inject file context into subagent prompts.
    read_file_ranges: Vec<(String, usize, usize)>,
    /// Whether any edit/write tool succeeded during the current `run()`.
    had_edits_this_run: bool,
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
        let is_subagent = max_turns.is_some();
        let include_extensions = !is_subagent;

        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let (mut tools, plugin_confirm_tools, mcp_servers) =
            build_tools_and_servers(cfg, context_size, &cwd, include_extensions);

        let discovered_skills = if include_extensions {
            skills::discover_skills(&cwd)
        } else {
            Vec::new()
        };
        if !discovered_skills.is_empty() {
            tools.register(Box::new(SkillTool::new(discovered_skills.clone())));
        }

        let (system_prompt, system_prompt_base_len, skills_summary_len, project_docs_info) =
            build_system_prompt(&cwd, include_extensions, &discovered_skills);

        let hooks = if include_extensions {
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
            config: config.cloned(),
            tool_scoping: cfg.tool_scoping.unwrap_or(false),
            task_reinjection: cfg.task_reinjection.unwrap_or(false),
            reinjection_interval: cfg.effective_reinjection_interval(),
            trim_threshold_pct: cfg.effective_trim_threshold(),
            trim_target_pct: cfg.effective_trim_target(),
            context_compaction: cfg.effective_context_compaction(),
            has_explored: false,
            current_task: String::new(),
            read_file_ranges: Vec::new(),
            had_edits_this_run: false,
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
    /// Inherits the parent's config so tool enable/disable settings are respected.
    fn new_subagent(
        backend: Arc<dyn ModelBackend>,
        model: String,
        context_size: u64,
        bash_timeout: std::time::Duration,
        max_turns: u16,
        config: Option<&Config>,
    ) -> Self {
        Self::build(backend, model, context_size, bash_timeout, 0, Some(max_turns), config)
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn skills(&self) -> &[SkillMeta] {
        &self.skills
    }

    /// Return the full system prompt (first message content).
    pub fn system_prompt(&self) -> &str {
        self.messages
            .first()
            .map(|m| m.content.as_str())
            .unwrap_or("")
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

    /// Hot-reload config, skills, hooks, plugins, MCP servers, and system prompt.
    /// Preserves conversation history (messages[1..]).
    /// Returns (summary_string, new_system_prompt).
    pub fn reload(&mut self, config: Config) -> (String, String) {
        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let old_skill_names: HashSet<String> = self.skills.iter().map(|s| s.name.clone()).collect();
        let old_mcp_count = self._mcp_servers.len();

        // Drop old MCP servers before rebuilding (triggers cleanup).
        self._mcp_servers.clear();
        let (mut tools, plugin_confirm_tools, mcp_servers) =
            build_tools_and_servers(&config, self.context_size, &cwd, true);

        let discovered_skills = skills::discover_skills(&cwd);
        if !discovered_skills.is_empty() {
            tools.register(Box::new(SkillTool::new(discovered_skills.clone())));
        }

        let (system_prompt, system_prompt_base_len, skills_summary_len, project_docs_info) =
            build_system_prompt(&cwd, true, &discovered_skills);

        if !self.messages.is_empty() {
            self.messages[0] = Message::system(&system_prompt);
        }

        self.hooks = HookRunner::discover(&cwd, Some(&config));
        self.bash_timeout = config.bash_timeout_duration();
        self.subagent_max_turns = config.effective_subagent_max_turns();
        self.tool_scoping = config.tool_scoping.unwrap_or(false);
        self.task_reinjection = config.task_reinjection.unwrap_or(false);
        self.reinjection_interval = config.effective_reinjection_interval();
        self.trim_threshold_pct = config.effective_trim_threshold();
        self.trim_target_pct = config.effective_trim_target();
        self.context_compaction = config.effective_context_compaction();

        self.tools = tools;
        self.plugin_confirm_tools = plugin_confirm_tools;
        self._mcp_servers = mcp_servers;
        self.system_prompt_base_len = system_prompt_base_len;
        self.skills_summary_len = skills_summary_len;
        self.project_docs_info = project_docs_info;
        self.skills = discovered_skills;
        self.config = Some(config);

        let new_skill_names: HashSet<String> = self.skills.iter().map(|s| s.name.clone()).collect();
        let added: Vec<String> = new_skill_names.difference(&old_skill_names).cloned().collect();
        let removed: Vec<String> = old_skill_names.difference(&new_skill_names).cloned().collect();
        let mut parts = Vec::new();
        parts.push(format!("{} skills", self.skills.len()));
        if !added.is_empty() {
            parts.push(format!("+{}", added.join(", +")));
        }
        if !removed.is_empty() {
            parts.push(format!("-{}", removed.join(", -")));
        }
        parts.push(format!("{} MCP servers", self._mcp_servers.len()));
        if old_mcp_count != self._mcp_servers.len() {
            parts.push(format!("(was {})", old_mcp_count));
        }
        parts.push("hooks reloaded".to_string());
        parts.push("plugins reloaded".to_string());

        let summary = format!("Reloaded: {}", parts.join(", "));
        (summary, system_prompt)
    }

    pub fn clear_history(&mut self) {
        self.messages.truncate(1); // keep system prompt
        self.has_explored = false;
    }

    /// Return tool definitions, optionally scoped based on conversation state.
    /// When tool_scoping is enabled, edit/write are hidden until a read/glob/grep
    /// has been performed — reduces the "Chekhov's gun" problem for small models.
    fn scoped_tool_definitions(&self) -> Vec<serde_json::Value> {
        if !self.tool_scoping || self.has_explored {
            return self.tools.definitions();
        }
        // Haven't explored yet — hide mutation tools
        self.tools.definitions_excluding(&["edit", "write"])
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

    /// Rewind the last `n` user turns. `UndoTurn` removes the anchor user message
    /// itself; `RewindTo` preserves the anchor and drops only subsequent messages.
    /// The system prompt (index 0) is always preserved.
    pub fn rewind_turns(&mut self, n: usize, mode: RewindMode) {
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
        let anchor = user_indices[user_indices.len() - actual_n];
        let truncate_at = match mode {
            RewindMode::RewindTo => anchor + 1,
            RewindMode::UndoTurn => anchor,
        };
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

    /// Identify a range of old non-system messages to remove, walking forward
    /// through complete exchanges until `need_to_free` tokens are covered.
    /// Returns `(first_non_system, remove_until, estimated_freed)` or `None`.
    fn find_removal_range(&self, current_prompt_tokens: u64) -> Option<(usize, usize, u64)> {
        if self.context_size == 0 {
            return None;
        }

        let threshold = self.context_size * self.trim_threshold_pct as u64 / 100;
        if current_prompt_tokens <= threshold {
            return None;
        }

        let target = self.context_size * self.trim_target_pct as u64 / 100;
        let need_to_free = current_prompt_tokens.saturating_sub(target);

        let first_non_system = self
            .messages
            .iter()
            .position(|m| !matches!(m.role, crate::message::Role::System))
            .unwrap_or(self.messages.len());

        let mut freed: u64 = 0;
        let mut remove_until = first_non_system;

        let mut i = first_non_system;
        while freed < need_to_free && i < self.messages.len().saturating_sub(1) {
            freed += self.messages[i].estimated_tokens();
            i += 1;
            while i < self.messages.len().saturating_sub(1)
                && !matches!(self.messages[i].role, crate::message::Role::User)
            {
                freed += self.messages[i].estimated_tokens();
                i += 1;
            }
            remove_until = i;
        }

        if remove_until > first_non_system {
            Some((first_non_system, remove_until, freed))
        } else {
            None
        }
    }

    /// Trim oldest non-system messages when context usage exceeds threshold.
    /// Removes complete exchanges (user + assistant + tool messages) as units.
    fn trim_context(&mut self, current_prompt_tokens: u64) -> Option<(usize, u64)> {
        let (first_non_system, remove_until, freed) =
            self.find_removal_range(current_prompt_tokens)?;
        let removed = remove_until - first_non_system;
        self.messages.drain(first_non_system..remove_until);
        Some((removed, freed))
    }

    /// Attempt LLM-based context compaction. Summarizes old exchanges via
    /// the model before removing them, preserving critical context.
    /// Returns `Some((removed, freed, summary_tokens))` on success, or `None`
    /// if below threshold or compaction failed (caller should fall back to
    /// `trim_context`).
    async fn compact_context(
        &mut self,
        current_prompt_tokens: u64,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &AtomicBool,
        num_ctx: Option<u64>,
    ) -> Option<(usize, u64, u64)> {
        let (first_non_system, remove_until, freed) =
            self.find_removal_range(current_prompt_tokens)?;

        // Notify that compaction is starting
        let _ = send_event(events, AgentEvent::ContextCompacting);

        // Format the messages to be compacted
        let excerpt = format_messages_for_compaction(&self.messages[first_non_system..remove_until]);

        // Calculate a word budget (~25% of original size)
        let word_budget = (freed / 4).clamp(50, 500);

        let compaction_messages = vec![
            Message::system(format!(
                "You are a context compaction assistant. Summarize the following conversation \
                 excerpt concisely. This summary replaces the original messages to free context space.\n\
                 \n\
                 PRESERVE: file paths, function/type/variable names, error messages and resolutions, \
                 decisions and rationale, tool call outcomes, current task state.\n\
                 OMIT: full file contents (just note which files were read/modified), verbose tool output, \
                 redundant exchanges.\n\
                 \n\
                 Keep your summary under {} words. Output only the summary.",
                word_budget
            )),
            Message::user(excerpt),
        ];

        // Call the LLM for compaction (no tools, no streaming to user)
        let compaction_response = tokio::select! {
            r = self.backend.chat(
                &self.model,
                &compaction_messages,
                None,
                num_ctx,
                Box::new(|_| {}),
            ) => {
                match r {
                    Ok(r) => r,
                    Err(_) => return None,
                }
            }
            _ = poll_cancel(cancel) => {
                return None;
            }
        };

        let summary = compaction_response.content.trim().to_string();
        if summary.is_empty() {
            return None;
        }

        // Guard: if summary is too large relative to freed space, discard it
        let summary_tokens = message::estimate_tokens(summary.len());
        if summary_tokens > freed / 2 {
            return None;
        }

        // Remove old messages and insert the summary exchange
        let removed = remove_until - first_non_system;
        self.messages.drain(first_non_system..remove_until);

        let summary_user = Message::user(format!(
            "[Context compaction — summary of {} earlier messages]\n\n{}",
            removed, summary
        ));
        let summary_assistant = Message::assistant(
            "Understood. I have the context from the conversation summary and will continue from here.",
        );

        // Emit MessageLogged so the summary pair gets persisted to the session
        let _ = send_event(events, AgentEvent::MessageLogged(summary_user.clone()));
        let _ = send_event(events, AgentEvent::MessageLogged(summary_assistant.clone()));

        self.messages.insert(first_non_system, summary_user);
        self.messages.insert(first_non_system + 1, summary_assistant);

        Some((removed, freed, summary_tokens))
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
    /// Build a file-context preamble from the parent's read history.
    fn build_file_context_preamble(&self) -> String {
        if self.read_file_ranges.is_empty() {
            return String::new();
        }

        // Deduplicate: keep only unique paths with their widest range.
        let mut file_ranges: std::collections::HashMap<&str, (usize, usize)> =
            std::collections::HashMap::new();
        for (path, start, end) in &self.read_file_ranges {
            let entry = file_ranges.entry(path.as_str()).or_insert((*start, *end));
            entry.0 = entry.0.min(*start);
            entry.1 = entry.1.max(*end);
        }

        let mut preamble = String::from(
            "\n\n--- Parent agent file context ---\n\
             The parent agent has already read these files. You may reference this \
             information without re-reading:\n",
        );
        for (path, (start, end)) in &file_ranges {
            preamble.push_str(&format!("  - {} (lines {}-{})\n", path, start + 1, end));
        }
        preamble.push_str("---\n");
        preamble
    }

    /// Check if a task description contains action verbs that imply code changes.
    fn task_expects_edits(task: &str) -> bool {
        let lower = task.to_lowercase();
        // Look for action verbs that imply code changes
        ["implement", "edit", "modify", "update", "change", "fix", "add", "remove", "replace",
         "refactor", "rewrite", "write"]
            .iter()
            .any(|verb| lower.contains(verb))
    }

    async fn execute_subagent(
        &self,
        task: &str,
        events: &mpsc::UnboundedSender<AgentEvent>,
        confirm_rx: &mut mpsc::UnboundedReceiver<bool>,
        cancel: &Arc<AtomicBool>,
    ) -> (String, bool) {
        // Build enriched task with parent file context
        let preamble = self.build_file_context_preamble();
        let enriched_task = if preamble.is_empty() {
            task.to_string()
        } else {
            format!("{}{}", task, preamble)
        };

        let result = self
            .run_subagent_once(&enriched_task, events, confirm_rx, cancel)
            .await;

        // Task enforcement: if the task expected edits but the subagent made none, retry once
        if result.1 && Self::task_expects_edits(task) {
            // Check if the subagent's response suggests it only explored
            // We detect this by checking if the subagent made any edits
            // The subagent itself tracks had_edits_this_run, but we can't access it
            // from here. Instead, check if the result mentions code changes or
            // contains typical "exploration-only" patterns.
            let response = &result.0;
            let likely_no_edits = !response.is_empty()
                && !response.contains("edit")
                && !response.contains("modified")
                && !response.contains("updated")
                && !response.contains("changed")
                && !response.contains("wrote")
                && !response.contains("replaced");

            if likely_no_edits {
                let retry_task = format!(
                    "{}\n\n\
                     IMPORTANT: You were asked to make code changes but your previous attempt \
                     completed without editing any files. Please re-read the task above and \
                     make the requested edits using the edit or write tool. Do not just \
                     explore or summarize — actually make the changes.",
                    enriched_task
                );
                return self
                    .run_subagent_once(&retry_task, events, confirm_rx, cancel)
                    .await;
            }
        }

        result
    }

    /// Run a single subagent instance and return its result.
    async fn run_subagent_once(
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
            self.config.as_ref(),
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
            // Sub-agents don't receive steering messages
            let (_steer_tx, mut steer_rx) = mpsc::unbounded_channel::<String>();
            let result = sa
                .run(&task_owned, &sub_tx_for_agent, &mut sub_confirm_rx, &mut steer_rx, cancel_for_sub)
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
        let mut name = tool_call.function.name.clone();
        let original_args = &tool_call.function.arguments;

        // --- pre_tool_execute hooks ---
        let args_override = match self.hooks.pre_tool_execute(&name, original_args).await {
            Ok(pre) => match pre.action.as_deref() {
                Some("deny") => {
                    let msg = pre
                        .message
                        .unwrap_or_else(|| "Blocked by hook.".to_string());
                    send_event(
                        events,
                        AgentEvent::ToolCall {
                            name: name.clone(),
                            args: crate::format::format_tool_args_display(&name, original_args),
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
                    let tool_msg = Message::tool(&msg, tool_call.id.clone(), false);
                    self.messages.push(tool_msg.clone());
                    send_event(events, AgentEvent::MessageLogged(tool_msg))?;
                    return Ok(true);
                }
                Some("rewrite") | Some("modify") => {
                    if let Some(ref new_name) = pre.tool_name {
                        send_event(
                            events,
                            AgentEvent::Debug(format!(
                                "Hook rewrote tool call: {} -> {}",
                                name, new_name
                            )),
                        )?;
                        name = new_name.clone();
                    }
                    pre.arguments
                }
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

        let args_display = crate::format::format_tool_args_display(&name, args);

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
                let tool_msg = Message::tool(&denied, tool_call.id.clone(), false);
                self.messages.push(tool_msg.clone());
                send_event(events, AgentEvent::MessageLogged(tool_msg))?;
                return Ok(true);
            }
        }

        // Validate tool arguments against schema before execution.
        if let Some(Err(validation_err)) = self.tools.validate(&name, args) {
            let msg = format!(
                "Invalid arguments for '{}': {}",
                name, validation_err
            );
            send_event(
                events,
                AgentEvent::ToolResult {
                    name: name.clone(),
                    output: msg.clone(),
                    success: false,
                },
            )?;
            let tool_msg = Message::tool(&msg, tool_call.id.clone(), false);
            self.messages.push(tool_msg.clone());
            send_event(events, AgentEvent::MessageLogged(tool_msg))?;
            return Ok(true);
        }

        // Check cancellation before each tool call.
        // NOTE: callers rely on no tool result message being pushed when we
        // return Ok(false), so they can backfill "Cancelled" results for
        // this tool call and all remaining ones using tool_calls_with_ids[i..].
        if cancel.load(Ordering::Relaxed) {
            send_event(events, AgentEvent::Cancelled)?;
            return Ok(false);
        }

        let (mut result, mut success) = if name == "bash" {
            // Set up a channel for streaming partial output from the bash
            // process so the TUI can show live progress.
            let (output_tx, mut output_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
            let events_clone = events.clone();
            let forward_task = tokio::spawn(async move {
                while let Some(line) = output_rx.recv().await {
                    let _ = send_event(&events_clone, AgentEvent::ToolOutput { output: line });
                }
            });

            // Race tool execution against the cancel flag so ESC kills
            // long-running bash commands immediately.
            let output = tokio::select! {
                output = BashTool.execute_async(args, self.bash_timeout, Some(&output_tx)) => {
                    Some(output)
                },
                _ = poll_cancel(cancel) => None,
            };

            // Clean up forwarding on both paths.
            drop(output_tx);
            let _ = forward_task.await;

            match output {
                Some(o) => o,
                None => {
                    // NOTE: callers rely on no tool result message being pushed
                    // when we return Ok(false), so they can backfill "Cancelled"
                    // results starting from this tool call's index.
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
            match self.tools.execute(&name, args) {
                Ok(output) => (output, true),
                Err(e) => (format!("Error: {}", e), false),
            }
        };

        // Track exploration for dynamic tool scoping
        if success && matches!(name.as_str(), "read" | "glob" | "grep") {
            self.has_explored = true;
        }

        // Track file reads for subagent context injection
        if success && name == "read" {
            if let Some(path) = args.get("file_path").and_then(|v| v.as_str()) {
                let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
                let start = offset.max(1) - 1;
                self.read_file_ranges.push((path.to_string(), start, start + limit));
            }
        }

        // Track edits for post-edit compilation and subagent enforcement
        if success && matches!(name.as_str(), "edit" | "write") {
            self.had_edits_this_run = true;
        }

        // --- post_tool_execute hooks ---
        let (post_result, post_success) = self
            .hooks
            .post_tool_execute(&name, args, result, success)
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
        let tool_msg = Message::tool(&context_result, tool_call.id.clone(), success);
        self.messages.push(tool_msg.clone());
        send_event(events, AgentEvent::MessageLogged(tool_msg))?;

        Ok(true)
    }

    pub async fn run(
        &mut self,
        user_input: &str,
        events: &mpsc::UnboundedSender<AgentEvent>,
        confirm_rx: &mut mpsc::UnboundedReceiver<bool>,
        steer_rx: &mut mpsc::UnboundedReceiver<String>,
        cancel: Arc<AtomicBool>,
    ) -> Result<()> {
        if !self.system_prompt_logged {
            if let Some(sys_msg) = self.messages.first() {
                send_event(events, AgentEvent::MessageLogged(sys_msg.clone()))?;
            }
            // Estimate tool definition tokens, broken down by category.
            const BUILTIN_TOOLS: &[&str] = &[
                "bash", "read", "edit", "write", "glob", "grep", "subagent", "skill",
            ];
            let mut builtin_chars: usize = 0;
            let mut plugin_chars: usize = 0;
            let mut mcp_chars: Vec<(String, usize)> = Vec::new();

            for def in self.tools.definitions() {
                let chars = def.to_string().len();
                let name = def["function"]["name"].as_str().unwrap_or("");

                if name.starts_with("mcp__") {
                    let server = name
                        .strip_prefix("mcp__")
                        .and_then(|rest| rest.split("__").next())
                        .unwrap_or("unknown");
                    if let Some(entry) = mcp_chars.iter_mut().find(|(s, _)| s == server) {
                        entry.1 += chars;
                    } else {
                        mcp_chars.push((server.to_string(), chars));
                    }
                } else if BUILTIN_TOOLS.contains(&name) {
                    builtin_chars += chars;
                } else {
                    plugin_chars += chars;
                }
            }

            let mut tool_defs_breakdown: Vec<(String, u64)> = Vec::new();
            if builtin_chars > 0 {
                tool_defs_breakdown.push(("Built-in".to_string(), message::estimate_tokens(builtin_chars)));
            }
            for (server, chars) in &mcp_chars {
                tool_defs_breakdown.push((format!("MCP: {}", server), message::estimate_tokens(*chars)));
            }
            if plugin_chars > 0 {
                tool_defs_breakdown.push(("Plugins".to_string(), message::estimate_tokens(plugin_chars)));
            }

            send_event(events, AgentEvent::SystemPromptInfo {
                base_prompt_tokens: message::estimate_tokens(self.system_prompt_base_len),
                project_docs: self.project_docs_info.iter()
                    .map(|(name, len)| (name.clone(), message::estimate_tokens(*len)))
                    .collect(),
                skills_tokens: message::estimate_tokens(self.skills_summary_len),
                tool_defs_breakdown,
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

        // --- skill trigger auto-injection ---
        if let Some((skill_name, instructions)) =
            skills::check_triggers(&self.skills, user_input)
        {
            let ctx_msg = Message::system(format!(
                "[Auto-triggered skill: /{}]\n{}",
                skill_name, instructions
            ));
            self.messages.push(ctx_msg);
            send_event(
                events,
                AgentEvent::Debug(format!("Auto-triggered skill: /{}", skill_name)),
            )?;
        }

        // Store the current task for re-injection
        self.current_task = user_input.to_string();
        self.has_explored = false;
        self.had_edits_this_run = false;

        let user_msg = Message::user(user_input);
        self.messages.push(user_msg.clone());
        send_event(events, AgentEvent::MessageLogged(user_msg))?;

        let mut empty_retries: u32 = 0;
        const MAX_EMPTY_RETRIES: u32 = 10;
        let mut turn: u32 = 0;
        let mut tool_call_summary: Vec<String> = Vec::new();
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

            // Drain steering messages: inject user messages sent mid-run
            while let Ok(steer_msg) = steer_rx.try_recv() {
                let user_msg = Message::user(&steer_msg);
                self.messages.push(user_msg.clone());
                send_event(events, AgentEvent::MessageLogged(user_msg))?;
            }

            // Task re-injection: periodically remind the model what it's doing.
            if self.task_reinjection
                && turn > 1
                && self.reinjection_interval > 0
                && (turn - 1).is_multiple_of(self.reinjection_interval as u32)
                && !self.current_task.is_empty()
            {
                let summary = if tool_call_summary.is_empty() {
                    String::new()
                } else {
                    format!(" Completed so far: {}", tool_call_summary.join(", "))
                };
                let reminder = format!(
                    "[Task reminder] Your current task: {}.{}",
                    self.current_task, summary
                );
                let reminder_msg = Message::system(&reminder);
                self.messages.push(reminder_msg);
                send_event(
                    events,
                    AgentEvent::Debug(format!("Task re-injection at turn {}", turn)),
                )?;
            }

            let tool_defs = self.scoped_tool_definitions();
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

            // If cancellation was requested while the stream was completing
            // (race between backend.chat finishing and poll_cancel detecting
            // the flag), honour the cancellation immediately.
            if cancel.load(Ordering::Relaxed) {
                send_event(events, AgentEvent::Cancelled)?;
                return Ok(());
            }

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

                // Auto-trim/compact context if approaching the limit
                if self.context_compaction {
                    match self
                        .compact_context(
                            response.prompt_eval_count,
                            events,
                            &cancel,
                            num_ctx,
                        )
                        .await
                    {
                        Some((removed, freed, summary_tokens)) => {
                            send_event(
                                events,
                                AgentEvent::ContextCompacted {
                                    removed_messages: removed,
                                    summary_tokens,
                                    estimated_tokens_freed: freed,
                                },
                            )?;
                        }
                        None => {
                            // Below threshold or compaction failed — fall back to blind trim
                            if let Some((removed, freed)) =
                                self.trim_context(response.prompt_eval_count)
                            {
                                send_event(
                                    events,
                                    AgentEvent::ContextTrimmed {
                                        removed_messages: removed,
                                        estimated_tokens_freed: freed,
                                    },
                                )?;
                            }
                        }
                    }
                } else if let Some((removed, freed)) =
                    self.trim_context(response.prompt_eval_count)
                {
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
                    tokio::time::sleep(retry_backoff_delay(empty_retries)).await;
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
                        tokio::time::sleep(retry_backoff_delay(empty_retries)).await;
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
                    tokio::time::sleep(retry_backoff_delay(empty_retries)).await;
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

            for (i, tool_call) in tool_calls_with_ids.iter().enumerate() {
                // Track tool calls for re-injection summary
                if self.task_reinjection {
                    tool_call_summary.push(tool_call.function.name.clone());
                }

                let continue_loop = self
                    .dispatch_tool_call(tool_call, events, confirm_rx, &cancel)
                    .await?;
                if !continue_loop {
                    // If cancelled, add "Cancelled" tool results for the
                    // current (un-executed) tool call and all remaining ones
                    // so the message history stays consistent — otherwise the
                    // model sees its own tool-call request without a matching
                    // result and retries the same action.
                    if cancel.load(Ordering::Relaxed) {
                        // Starting from index `i` is correct because
                        // dispatch_tool_call does NOT push a tool result
                        // message when returning Ok(false) (cancel path),
                        // so the current tool call still needs a result.
                        for remaining in &tool_calls_with_ids[i..] {
                            let cancelled_msg = "Cancelled by user.";
                            send_event(events, AgentEvent::ToolResult {
                                name: remaining.function.name.clone(),
                                output: cancelled_msg.to_string(),
                                success: false,
                            })?;
                            let tool_msg = Message::tool(
                                cancelled_msg,
                                remaining.id.clone(),
                                false,
                            );
                            self.messages.push(tool_msg.clone());
                            send_event(events, AgentEvent::MessageLogged(tool_msg))?;
                        }
                    }
                    return Ok(());
                }
            }

            // Post-edit compilation check: if any edits were made this turn,
            // run `cargo check` (if Cargo.toml exists) and inject diagnostics.
            if self.had_edits_this_run && std::path::Path::new("Cargo.toml").exists() {
                self.had_edits_this_run = false;
                if let Ok(output) = tokio::process::Command::new("cargo")
                    .args(["check", "--message-format=short"])
                    .output()
                    .await
                {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    // Only inject if there are actual errors or warnings
                    let has_errors = stderr.contains("error[");
                    let has_warnings = stderr.contains("warning:");
                    if has_errors || has_warnings {
                        let label = if has_errors { "errors" } else { "warnings" };
                        let check_msg = format!(
                            "[Auto cargo check detected {}]\n{}",
                            label,
                            stderr.lines()
                                .filter(|l| l.contains("error") || l.contains("warning"))
                                .take(20)
                                .collect::<Vec<_>>()
                                .join("\n")
                        );
                        let system_hint = Message::system(&check_msg);
                        self.messages.push(system_hint);
                    }
                }
            }

        }
    }
}
