mod builder;
mod context;
mod events;
mod subagent;
mod tool_executor;
mod util;

use std::collections::{HashSet, VecDeque};
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
use crate::skills::{self, SkillMeta};
use crate::tools::{SkillTool, ToolRegistry};

use builder::{build_system_prompt, build_tools_and_servers};
use context::ContextManager;
pub use events::AgentEvent;
use events::send_event;
use util::{poll_cancel, retry_backoff_delay};

/// How `rewind_turns` / `rewind_leaf` treats the anchor user message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewindMode {
    /// Remove the anchor user message itself (undo that turn).
    UndoTurn,
    /// Preserve the anchor; drop only the messages after it.
    RewindTo,
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
    /// Context trim/compact policy.
    context: ContextManager,
    /// When true, rank discovered skills each turn and inject the top match's
    /// body as a transient system note (not persisted to self.messages).
    skill_inject: bool,
    /// Recent (tool_name, is_error) entries; capped at 4. Fed to the skill
    /// selector when `skill_inject` is enabled.
    recent_tool_results: VecDeque<(String, bool)>,
    /// Tracks whether a file-reading tool has been used (for tool scoping).
    has_explored: bool,
    /// The original user input for the current run (for task re-injection).
    current_task: String,
    /// Files read during this agent's lifetime: (path, start_line_0, end_line_0).
    /// Used to inject file context into subagent prompts.
    read_file_ranges: Vec<(String, usize, usize)>,
    /// Whether any edit/write tool succeeded during the current `run()`.
    had_edits_this_run: bool,
    /// Sticky flag: set true once the thinking budget has been exceeded in
    /// this session. When set, subsequent turns are sent with thinking off
    /// regardless of config. Cleared only by /settings changes or a new session.
    thinking_disabled: bool,
    /// Directory where per-session first-write-wins file snapshots are stored.
    /// Set via `set_session_dir`; `None` disables checkpointing.
    pub(super) checkpoint_dir: Option<std::path::PathBuf>,
    /// Canonical paths of files we've already snapshotted this session. Keeps
    /// the first snapshot intact across repeated edits.
    pub(super) checkpointed_files: HashSet<std::path::PathBuf>,
    /// Bash command prefixes that auto-approve without a confirmation prompt.
    /// Intended for cheap read-only commands (ls, cat, rg, git status, …).
    pub(super) bash_safe_prefixes: Vec<String>,
    /// Hard cap on the total number of agent-loop turns for the main agent.
    /// `None` = unlimited. Sub-agents use `max_turns` instead.
    main_turn_cap: Option<u32>,
    /// (tool_name, canonical_args) of the last dispatched tool call. Reset at
    /// the start of every `run()`. Used by `dispatch_tool_call` to detect
    /// immediate repeats and short-circuit them with a corrective synthetic
    /// result, so a small model that loops on the same call doesn't burn turns.
    pub(super) last_tool_signature: Option<(String, String)>,
    /// Number of consecutive immediate-repeat detections on the current
    /// signature. Each repeat escalates the corrective message; after a few
    /// the harness injects a system message and starts hard-refusing the
    /// call until the model picks a different tool or arguments. Reset when
    /// the model finally varies its call.
    pub(super) consecutive_repeat_count: u32,
    /// One entry per cargo-check observation in the current `run()`, holding
    /// the set of error signatures (code + primary span) seen at that point.
    /// Compared against new sets to detect oscillation/regression — the
    /// rendered diagnostics already tell the model *what* is broken; this
    /// tells it *whether its own edits are making things better or worse*.
    /// Reset at the start of each run.
    pub(super) compile_attempts: util::AttemptHistory,
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
            context: ContextManager {
                trim_threshold_pct: cfg.effective_trim_threshold(),
                trim_target_pct: cfg.effective_trim_target(),
                compaction_enabled: cfg.effective_context_compaction(),
            },
            skill_inject: cfg.skill_inject.unwrap_or(false),
            recent_tool_results: VecDeque::with_capacity(4),
            has_explored: false,
            current_task: String::new(),
            read_file_ranges: Vec::new(),
            had_edits_this_run: false,
            thinking_disabled: false,
            checkpoint_dir: None,
            checkpointed_files: HashSet::new(),
            bash_safe_prefixes: cfg.bash_safe_prefixes.clone().unwrap_or_default(),
            // Main-agent turn cap only applies when this isn't a sub-agent —
            // sub-agents use `max_turns` which is already set.
            main_turn_cap: if is_subagent { None } else { cfg.max_turns },
            last_tool_signature: None,
            consecutive_repeat_count: 0,
            compile_attempts: Vec::new(),
        }
    }

    /// Attach a session directory so file edits/writes get first-write-wins
    /// snapshots under `<session_dir>/checkpoints/`. Pass `None` to disable.
    pub fn set_session_dir(&mut self, session_dir: Option<&Path>) {
        self.checkpoint_dir = session_dir.map(|p| p.join("checkpoints"));
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
        self.context.trim_threshold_pct = config.effective_trim_threshold();
        self.context.trim_target_pct = config.effective_trim_target();
        self.context.compaction_enabled = config.effective_context_compaction();
        self.skill_inject = config.skill_inject.unwrap_or(false);
        self.bash_safe_prefixes = config.bash_safe_prefixes.clone().unwrap_or_default();
        // Sub-agents keep main_turn_cap at None (their cap is `max_turns`).
        if self.max_turns.is_none() {
            self.main_turn_cap = config.max_turns;
        }

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
                None, // thinking off — we want a final answer, not deliberation
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

    /// Emit `MessageLogged` for the system prompt and a `SystemPromptInfo`
    /// event with a per-category token breakdown. No-op after the first call.
    fn emit_system_prompt_info_once(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<()> {
        if self.system_prompt_logged {
            return Ok(());
        }
        if let Some(sys_msg) = self.messages.first() {
            send_event(events, AgentEvent::MessageLogged(sys_msg.clone()))?;
        }

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
            tool_defs_breakdown
                .push(("Built-in".to_string(), message::estimate_tokens(builtin_chars)));
        }
        for (server, chars) in &mcp_chars {
            tool_defs_breakdown
                .push((format!("MCP: {}", server), message::estimate_tokens(*chars)));
        }
        if plugin_chars > 0 {
            tool_defs_breakdown
                .push(("Plugins".to_string(), message::estimate_tokens(plugin_chars)));
        }

        send_event(
            events,
            AgentEvent::SystemPromptInfo {
                base_prompt_tokens: message::estimate_tokens(self.system_prompt_base_len),
                project_docs: self
                    .project_docs_info
                    .iter()
                    .map(|(name, len)| (name.clone(), message::estimate_tokens(*len)))
                    .collect(),
                skills_tokens: message::estimate_tokens(self.skills_summary_len),
                tool_defs_breakdown,
            },
        )?;
        self.system_prompt_logged = true;
        Ok(())
    }

    /// Emit a `Debug` event describing backend timings for the last response.
    fn emit_response_metrics(
        &self,
        response: &crate::backend::ChatResponse,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<()> {
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
        )
    }

    /// Apply context trim/compact policy and emit the corresponding event.
    async fn apply_context_management(
        &mut self,
        current_prompt_tokens: u64,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &AtomicBool,
        num_ctx: Option<u64>,
    ) -> Result<()> {
        if self.context.compaction_enabled {
            match self
                .context
                .compact(
                    &mut self.messages,
                    self.context_size,
                    current_prompt_tokens,
                    &*self.backend,
                    &self.model,
                    events,
                    cancel,
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
                    return Ok(());
                }
                None => {
                    // Below threshold or compaction failed — fall through to blind trim.
                }
            }
        }
        if let Some((removed, freed)) = self.context.trim(
            &mut self.messages,
            self.context_size,
            current_prompt_tokens,
        ) {
            send_event(
                events,
                AgentEvent::ContextTrimmed {
                    removed_messages: removed,
                    estimated_tokens_freed: freed,
                },
            )?;
        }
        Ok(())
    }

    /// If the context window is exhausted, log any partial content, emit
    /// Error + Done events, and return true so the caller stops the loop.
    fn handle_context_exhausted(
        &mut self,
        response: &crate::backend::ChatResponse,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<bool> {
        if self.context_size == 0 || response.prompt_eval_count == 0 {
            return Ok(false);
        }
        let total = response.prompt_eval_count + response.eval_count;
        let context_full = total >= self.context_size;
        // If tool calls are pending, also check if prompt alone is >90% of
        // context — tool results will push us over on the next iteration.
        let context_nearly_full = !response.tool_calls.is_empty()
            && response.prompt_eval_count > self.context_size * 90 / 100;

        if !(context_full || context_nearly_full) {
            return Ok(false);
        }

        if !response.content.trim().is_empty() {
            let assistant_msg = Message::assistant(&response.content);
            self.messages.push(assistant_msg.clone());
            send_event(events, AgentEvent::MessageLogged(assistant_msg))?;
        }
        send_event(
            events,
            AgentEvent::Error(
                "Context window is full — the model can no longer process this \
                 conversation. Use /clear to start fresh."
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
        Ok(true)
    }

    /// Append synthetic "Cancelled by user" tool results for each remaining
    /// tool call. Keeps the message history consistent so the model doesn't
    /// see orphan tool-call requests on the next run.
    fn backfill_cancelled_tool_results(
        &mut self,
        remaining: &[crate::message::ToolCall],
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<()> {
        const CANCELLED: &str = "Cancelled by user.";
        for tc in remaining {
            send_event(
                events,
                AgentEvent::ToolResult {
                    name: tc.function.name.clone(),
                    output: CANCELLED.to_string(),
                    success: false,
                },
            )?;
            let tool_msg = Message::tool(CANCELLED, tc.id.clone(), false);
            self.messages.push(tool_msg.clone());
            send_event(events, AgentEvent::MessageLogged(tool_msg))?;
        }
        Ok(())
    }

    /// Run `cargo check` after an edit/write tool succeeded this turn and
    /// inject the diagnostics as a system message if there are errors or
    /// warnings. No-op if no edits happened or there's no Cargo.toml.
    async fn run_post_edit_cargo_check(&mut self) {
        if !self.had_edits_this_run {
            return;
        }
        self.had_edits_this_run = false;
        if let Some(body) = self.check_and_record_cargo_diagnostics().await {
            self.messages.push(Message::system(&body));
        }
    }

    /// Run `cargo check` and record the resulting error signature set in
    /// `compile_attempts`. Returns the rendered diagnostics block (with any
    /// oscillation advisory appended) for the caller to inject. Skips the
    /// `compile_attempts` push when the new set is identical to the previous
    /// one — pushing duplicates inflates `analyze_compile_oscillation`'s
    /// `rposition` walk without changing what it can detect.
    pub(super) async fn check_and_record_cargo_diagnostics(&mut self) -> Option<String> {
        let diag = util::collect_cargo_diagnostics().await?;
        let mut body = diag.rendered_block;
        if let Some(advisory) =
            util::analyze_compile_oscillation(&self.compile_attempts, &diag.error_sigs)
        {
            body.push_str("\n\n");
            body.push_str(&advisory);
        }
        if self.compile_attempts.last() != Some(&diag.error_sigs) {
            self.compile_attempts.push(diag.error_sigs);
        }
        Some(body)
    }

    /// Build the message list for the next `backend.chat()` call. When
    /// `skill_inject` is on and the selector picks at least one skill, the
    /// guidance block is appended as a transient system message without
    /// touching `self.messages` — so the canonical history stays stable
    /// across turns and remains cache-friendly. Fast paths borrow the
    /// existing history; only an actual injection pays for a clone.
    fn build_turn_messages(&self) -> (std::borrow::Cow<'_, [Message]>, Option<String>) {
        use std::borrow::Cow;
        if !self.skill_inject || self.skills.is_empty() {
            return (Cow::Borrowed(self.messages.as_slice()), None);
        }
        let user_prompt = self
            .messages
            .iter()
            .rev()
            .find(|m| matches!(m.role, crate::message::Role::User))
            .map(|m| m.content.as_str());
        let tool_results: Vec<(String, bool)> = self.recent_tool_results.iter().cloned().collect();
        let body = match skills::select_skills_for_injection(
            &self.skills,
            &tool_results,
            user_prompt,
        ) {
            Some(b) => b,
            None => return (Cow::Borrowed(self.messages.as_slice()), None),
        };
        let mut msgs = self.messages.clone();
        msgs.push(Message::system(format!("[Tool Usage Guidance]\n{}", body)));
        (Cow::Owned(msgs), Some(body))
    }

    /// Record a tool call outcome for per-turn skill injection. Bounded at 4.
    pub(super) fn record_tool_result(&mut self, name: &str, success: bool) {
        if !self.skill_inject {
            return;
        }
        if self.recent_tool_results.len() == 4 {
            self.recent_tool_results.pop_front();
        }
        self.recent_tool_results
            .push_back((name.to_lowercase(), !success));
    }

    pub async fn run(
        &mut self,
        user_input: &str,
        events: &mpsc::UnboundedSender<AgentEvent>,
        confirm_rx: &mut mpsc::UnboundedReceiver<bool>,
        steer_rx: &mut mpsc::UnboundedReceiver<String>,
        cancel: Arc<AtomicBool>,
    ) -> Result<()> {
        self.emit_system_prompt_info_once(events)?;

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
        self.last_tool_signature = None;
        self.consecutive_repeat_count = 0;
        self.compile_attempts.clear();

        let user_msg = Message::user(user_input);
        self.messages.push(user_msg.clone());
        send_event(events, AgentEvent::MessageLogged(user_msg))?;

        let mut empty_retries: u32 = 0;
        const MAX_EMPTY_RETRIES: u32 = 10;
        let mut correction_count: u32 = 0;
        const MAX_CORRECTIONS: u32 = 2;
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
            // Main agent uses a hard stop instead of `force_final_response`
            // so the user keeps control: they can send a new message to continue.
            if let Some(max) = self.main_turn_cap {
                if turn >= max {
                    send_event(
                        events,
                        AgentEvent::Error(format!(
                            "Turn cap reached ({} turns). Stopping — start a new \
                             message to continue.",
                            max
                        )),
                    )?;
                    send_event(
                        events,
                        AgentEvent::Done {
                            prompt_tokens: 0,
                            eval_count: 0,
                        },
                    )?;
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

            let (turn_messages, injected) = self.build_turn_messages();
            if let Some(body) = &injected {
                let first_line = body.lines().next().unwrap_or("").trim_start_matches("### ");
                send_event(
                    events,
                    AgentEvent::Debug(format!(
                        "skill_inject: {} chars — {}",
                        body.len(),
                        first_line
                    )),
                )?;
            }

            send_event(
                events,
                AgentEvent::Debug(format!(
                    "OLLAMA_REQUEST model={} messages={} tools={}",
                    self.model,
                    turn_messages.len(),
                    tool_defs.len()
                )),
            )?;

            let thinking_budget = if self.thinking_disabled {
                Some(0)
            } else {
                self.config
                    .as_ref()
                    .and_then(|c| c.thinking_budget_tokens)
                    .filter(|&n| n > 0)
            };

            let mut response = tokio::select! {
                r = self.backend.chat(&self.model, &turn_messages, Some(tool_defs), num_ctx, thinking_budget, Box::new(move |token| {
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

            // Thinking budget tripped mid-stream. Bounded-thinking-with-retry:
            // preserve the partial reasoning trace, reinject it as assistant
            // context so the model sees its own conclusions in history, then
            // retry with thinking disabled. This stops the model from
            // re-deliberating the same problem from scratch on every turn —
            // its prior reasoning is now observable state it can act on.
            if response.thinking_budget_exceeded {
                self.thinking_disabled = true;
                let thinking_tokens = message::estimate_tokens(response.thinking.len());
                let trace = response.thinking.trim();
                let partial_content = response.content.trim();
                let mut combined = String::new();
                if !trace.is_empty() {
                    combined.push_str("[partial reasoning, truncated at budget]\n");
                    combined.push_str(trace);
                }
                if !partial_content.is_empty() {
                    if !combined.is_empty() {
                        combined.push_str("\n\n");
                    }
                    combined.push_str(partial_content);
                }
                if combined.is_empty() {
                    combined
                        .push_str("[auto: thinking budget exceeded — no partial output captured]");
                }
                send_event(events, AgentEvent::ContentReplaced(combined.clone()))?;
                let assistant_msg = Message::assistant(&combined);
                self.messages.push(assistant_msg.clone());
                send_event(events, AgentEvent::MessageLogged(assistant_msg))?;
                send_event(
                    events,
                    AgentEvent::ThinkingBudgetExceeded { thinking_tokens },
                )?;
                empty_retries = 0;
                continue;
            }

            self.emit_response_metrics(&response, events)?;

            if response.prompt_eval_count > 0 {
                send_event(
                    events,
                    AgentEvent::ContextUpdate {
                        prompt_tokens: response.prompt_eval_count,
                    },
                )?;
                self.apply_context_management(
                    response.prompt_eval_count,
                    events,
                    &cancel,
                    num_ctx,
                )
                .await?;
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

            if self.handle_context_exhausted(&response, events)? {
                return Ok(());
            }

            if response.tool_calls.is_empty() {
                // Empty response: push a corrective user message so the model
                // sees an explicit nudge next turn instead of silently retrying
                // into the same degenerate state.
                if response.content.trim().is_empty() {
                    if correction_count < MAX_CORRECTIONS {
                        correction_count += 1;
                        let nudge = Message::user(
                            "[auto] Your previous response was empty. You must either \
                             call a tool to make progress or give a final answer — do \
                             not return nothing.",
                        );
                        self.messages.push(nudge.clone());
                        send_event(events, AgentEvent::MessageLogged(nudge))?;
                        send_event(
                            events,
                            AgentEvent::Debug(format!(
                                "Empty response — injected corrective follow-up ({}/{})",
                                correction_count, MAX_CORRECTIONS
                            )),
                        )?;
                        continue;
                    }
                    send_event(
                        events,
                        AgentEvent::Error(format!(
                            "Model returned empty responses {} times in a row — giving up.",
                            correction_count + 1
                        )),
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
            // Tool calls = real progress; clear the empty-response correction streak.
            correction_count = 0;

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
                    // Backfill synthetic "Cancelled" results starting at `i`
                    // because dispatch_tool_call does NOT push a tool result
                    // message when returning Ok(false) — so the current tool
                    // call still needs a result to pair with its request.
                    if cancel.load(Ordering::Relaxed) {
                        self.backfill_cancelled_tool_results(
                            &tool_calls_with_ids[i..],
                            events,
                        )?;
                    }
                    return Ok(());
                }
            }

            self.run_post_edit_cargo_check().await;
        }
    }
}
