mod builder;
mod context;
mod events;
pub mod plan;
mod planner;
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
use crate::config::{Config, PlanConfig};
use crate::hooks::HookRunner;
use crate::mcp;
use crate::message::{self, Message};
use crate::skills::{self, SkillMeta};
use crate::tools::{SkillTool, ToolRegistry};

use builder::{build_system_prompt, build_tools_and_servers, AgentMode};
use context::ContextManager;
pub use events::AgentEvent;
use events::send_event;
use plan::{new_shared_todo_list, SharedTodoList};
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
    /// Shared TodoList written by the planner sub-agent (via plan_add_step)
    /// and updated by the main agent (via plan_mark_*). The agent loop reads
    /// it at termination time to decide whether to refuse `Done`.
    pub(super) plan: SharedTodoList,
    /// Resolved plan config (`enabled`, `append_steps`, `max_gate_retries`).
    pub(super) plan_config: PlanConfig,
    /// How many times the termination gate has fired in the current `run()`.
    /// Reset on each `run()`. Capped by `plan_config.max_gate_retries`.
    pub(super) plan_gate_retries: u32,
    /// Set true once behavioural verification (project test command) has
    /// passed in the current run. Stays false until tests pass — a failed
    /// run keeps it false so the next gate retry triggers another check.
    /// Reset on each `run()`.
    pub(super) behavioral_verify_passed: bool,
    /// How many times the behavioural-verify gate has run in the current
    /// `run()`. Reset on each `run()`. Cap prevents the model from spinning
    /// forever on a flaky or unfixable test suite.
    pub(super) behavioral_verify_retries: u32,
    /// Identifies a planner sub-agent so it doesn't recursively trigger its
    /// own planner. Set only via `Agent::new_planner_subagent`.
    pub(super) is_planner: bool,
}

impl Agent {
    /// Shared constructor. When `is_subagent` is true, the subagent tool is
    /// omitted (prevents recursion) and a turn limit is enforced.
    #[allow(clippy::too_many_arguments)]
    fn build(
        backend: Arc<dyn ModelBackend>,
        model: String,
        context_size: u64,
        bash_timeout: std::time::Duration,
        subagent_max_turns: u16,
        max_turns: Option<u16>,
        config: Option<&Config>,
        mode: AgentMode,
        plan: SharedTodoList,
    ) -> Self {
        let default_config = Config::default();
        let cfg = config.unwrap_or(&default_config);
        let is_subagent = max_turns.is_some();
        let include_extensions = matches!(mode, AgentMode::Main);

        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let (mut tools, plugin_confirm_tools, mcp_servers) =
            build_tools_and_servers(cfg, context_size, &cwd, mode, &plan);

        let discovered_skills = if include_extensions {
            skills::discover_skills(&cwd)
        } else {
            Vec::new()
        };
        if !discovered_skills.is_empty() {
            tools.register(Box::new(SkillTool::new(discovered_skills.clone())));
        }

        let (system_prompt, system_prompt_base_len, skills_summary_len, project_docs_info) =
            build_system_prompt(&cwd, mode, &discovered_skills);

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
            plan,
            plan_config: cfg.effective_plan_config(),
            plan_gate_retries: 0,
            behavioral_verify_passed: false,
            behavioral_verify_retries: 0,
            is_planner: matches!(mode, AgentMode::Planner),
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
        Self::build(
            backend,
            model,
            context_size,
            bash_timeout,
            subagent_max_turns,
            None,
            None,
            AgentMode::Main,
            new_shared_todo_list(),
        )
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
        Self::build(
            backend,
            model,
            context_size,
            bash_timeout,
            subagent_max_turns,
            None,
            Some(config),
            AgentMode::Main,
            new_shared_todo_list(),
        )
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
        // Sub-agents get their own (unused) plan list and the loop's planning
        // phase is gated off via `is_planner`/`is_subagent` checks.
        Self::build(
            backend,
            model,
            context_size,
            bash_timeout,
            0,
            Some(max_turns),
            config,
            AgentMode::Subagent,
            new_shared_todo_list(),
        )
    }

    /// Create a planner sub-agent that shares the parent's plan list. Has
    /// only read-only built-in tools (read/glob/grep) plus `plan_add_step`
    /// and `plan_list_steps`. No bash, edit, write, plugins, MCP, evidence,
    /// or subagent — strictly observe-and-plan.
    pub(super) fn new_planner_subagent(
        backend: Arc<dyn ModelBackend>,
        model: String,
        context_size: u64,
        bash_timeout: std::time::Duration,
        max_turns: u16,
        config: Option<&Config>,
        plan: SharedTodoList,
    ) -> Self {
        Self::build(
            backend,
            model,
            context_size,
            bash_timeout,
            0,
            Some(max_turns),
            config,
            AgentMode::Planner,
            plan,
        )
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
            build_tools_and_servers(&config, self.context_size, &cwd, AgentMode::Main, &self.plan);

        let discovered_skills = skills::discover_skills(&cwd);
        if !discovered_skills.is_empty() {
            tools.register(Box::new(SkillTool::new(discovered_skills.clone())));
        }

        let (system_prompt, system_prompt_base_len, skills_summary_len, project_docs_info) =
            build_system_prompt(&cwd, AgentMode::Main, &discovered_skills);

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
        self.plan_config = config.effective_plan_config();
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
            // User role: chat templates for several recent models reject
            // system messages that don't sit at the start of the conversation.
            self.messages.push(Message::user(&body));
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
        // User role: see post_edit_cargo_check note — mid-conversation system
        // messages are rejected by Qwen3/Qwen3.5-style chat templates.
        msgs.push(Message::user(format!("[Tool Usage Guidance]\n{}", body)));
        (Cow::Owned(msgs), Some(body))
    }

    /// Whether the plan-completion gate should run for the current agent.
    /// True only on the top-level main agent with planning enabled and a
    /// non-empty plan. Sub-agents and planners are gated off.
    fn should_gate_on_plan(&self) -> bool {
        if !self.plan_config.enabled || self.is_planner || self.max_turns.is_some() {
            return false;
        }
        // Lock is held briefly; poisoning means a prior panic in plan code,
        // and silently failing-open here would defeat the whole gate.
        !self.plan.lock().expect("plan lock poisoned").is_empty()
    }

    /// Descriptions of plan steps that are still pending or in progress.
    /// Empty if the plan is fully complete.
    fn unfinished_plan_steps(&self) -> Vec<String> {
        self.plan
            .lock()
            .expect("plan lock poisoned")
            .unfinished()
            .into_iter()
            .map(|(_, s)| s.description.clone())
            .collect()
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
            // User role (rather than a second system message): Qwen3/Qwen3.5
            // chat templates only allow one system message and require it at
            // index 0. Subsequent context injections must use a non-system
            // role to avoid `raise_exception('System message must be at the
            // beginning.')`.
            let ctx_msg = Message::user(format!(
                "[Hook-injected context]\n{}",
                extra_context
            ));
            self.messages.push(ctx_msg);
        }

        // --- skill trigger auto-injection ---
        if let Some((skill_name, instructions)) =
            skills::check_triggers(&self.skills, user_input)
        {
            // User role: see agent_start hook note above.
            let ctx_msg = Message::user(format!(
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
        self.plan_gate_retries = 0;
        self.behavioral_verify_passed = false;
        self.behavioral_verify_retries = 0;

        let user_msg = Message::user(user_input);
        self.messages.push(user_msg.clone());
        send_event(events, AgentEvent::MessageLogged(user_msg))?;

        // Enforced planning phase: only fires for the top-level main agent on
        // a fresh plan list. Sub-agents (planner included) skip this so they
        // don't recurse into another planner. We also skip planning for
        // trivial requests (short, no edit-action verbs) — running a full
        // sub-agent for "what is 2+2" would double latency for nothing.
        let plan_list_empty = self.plan.lock().expect("plan lock poisoned").is_empty();
        if self.plan_config.enabled
            && !self.is_planner
            && self.max_turns.is_none()
            && plan_list_empty
            && !is_trivial_request(user_input)
        {
            match self.produce_plan(user_input, events, confirm_rx, &cancel).await {
                Ok(true) => {
                    let summary = self
                        .plan
                        .lock()
                        .expect("plan lock poisoned")
                        .summary_for_prompt();
                    // User role: chat templates for Qwen3/Qwen3.5 (and other
                    // recent models) raise an exception when a system message
                    // appears after the start of the conversation. The plan
                    // reminder is informational, so a user-role message
                    // delivers the same instruction without breaking the
                    // template.
                    let reminder = Message::user(format!(
                        "[Enforced plan]\nA planner sub-agent produced this plan for the \
                         current task:\n\n{}\nWork through the steps in order using \
                         plan_mark_in_progress, plan_mark_done, and plan_skip_step. You \
                         cannot end your turn until every step is marked done or skipped.",
                        summary
                    ));
                    self.messages.push(reminder.clone());
                    send_event(events, AgentEvent::MessageLogged(reminder))?;
                }
                Ok(false) => {}
                Err(e) => {
                    send_event(
                        events,
                        AgentEvent::Error(format!("Planner error: {}", e)),
                    )?;
                }
            }
        }

        let mut empty_retries: u32 = 0;
        const MAX_EMPTY_RETRIES: u32 = 10;
        let mut correction_count: u32 = 0;
        const MAX_CORRECTIONS: u32 = 2;
        let mut per_turn_timeout_count: u32 = 0;
        const MAX_PER_TURN_TIMEOUTS: u32 = 3;
        let mut backend_retry_count: u32 = 0;
        const MAX_BACKEND_RETRIES: u32 = 3;
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
                // Inject as a user message rather than system: chat templates
                // for several recent models (Qwen3, Qwen3.5, GLM4) raise an
                // exception when a system message appears anywhere other than
                // the start of the conversation. A user-role reminder is
                // template-portable and the agent treats it the same way.
                let reminder = format!(
                    "[Task reminder] Your current task: {}.{}",
                    self.current_task, summary
                );
                let reminder_msg = Message::user(&reminder);
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

            // Per-turn wall-clock cap. The streaming layer already has a
            // chunk-level inactivity timeout, but that catches "no bytes for
            // 180s," not "trickle of bytes for 30+ minutes with no semantic
            // progress" — observed with qwen3.6:35b on a heavily-populated
            // context, where the model output rate slows to a crawl. After
            // this cap fires, we abort the whole chat() call and inject a
            // user nudge to retry, so a stalled turn doesn't wedge the
            // entire run indefinitely.
            const PER_TURN_TIMEOUT: std::time::Duration =
                std::time::Duration::from_secs(600);
            let chat_future = self.backend.chat(
                &self.model,
                &turn_messages,
                Some(tool_defs),
                num_ctx,
                thinking_budget,
                Box::new(move |token| {
                    let _ = events_clone.send(AgentEvent::Token(token.to_string()));
                }),
            );
            let mut response = tokio::select! {
                r = tokio::time::timeout(PER_TURN_TIMEOUT, chat_future) => {
                    match r {
                        Ok(Ok(resp)) => {
                            backend_retry_count = 0;
                            resp
                        }
                        Ok(Err(e)) => {
                            // Transient HTTP/transport errors (connection
                            // reset, keep-alive expiry, momentary unreachable
                            // socket) shouldn't kill the run. llama-server in
                            // particular sometimes drops idle keep-alive
                            // connections between turns — retry with a fresh
                            // connection before giving up.
                            let msg = e.to_string();
                            let transient = msg.contains("Failed to connect to backend")
                                || msg.contains("connection closed")
                                || msg.contains("connection reset")
                                || msg.contains("broken pipe")
                                || msg.contains("os error 104")
                                || msg.contains("error sending request");
                            if transient && backend_retry_count < MAX_BACKEND_RETRIES {
                                backend_retry_count += 1;
                                send_event(
                                    events,
                                    AgentEvent::Debug(format!(
                                        "[harness] Transient backend error ({}), retrying ({}/{}): {}",
                                        if msg.len() > 80 { &msg[..80] } else { &msg },
                                        backend_retry_count,
                                        MAX_BACKEND_RETRIES,
                                        msg
                                    )),
                                )?;
                                tokio::time::sleep(std::time::Duration::from_millis(
                                    500 * backend_retry_count as u64,
                                )).await;
                                continue;
                            }
                            send_event(events, AgentEvent::Error(msg))?;
                            send_event(events, AgentEvent::Done { prompt_tokens: 0, eval_count: 0 })?;
                            return Ok(());
                        }
                        Err(_) => {
                            per_turn_timeout_count += 1;
                            if per_turn_timeout_count >= MAX_PER_TURN_TIMEOUTS {
                                send_event(
                                    events,
                                    AgentEvent::Error(format!(
                                        "[harness] Per-turn timeout hit {} times in a row — giving up on this run.",
                                        per_turn_timeout_count
                                    )),
                                )?;
                                send_event(events, AgentEvent::Done { prompt_tokens: 0, eval_count: 0 })?;
                                return Ok(());
                            }
                            send_event(
                                events,
                                AgentEvent::Debug(format!(
                                    "[harness] Per-turn timeout fired after {}s — aborting this turn and retrying ({}/{})",
                                    PER_TURN_TIMEOUT.as_secs(),
                                    per_turn_timeout_count,
                                    MAX_PER_TURN_TIMEOUTS
                                )),
                            )?;
                            let nudge = Message::user(format!(
                                "[harness] Your previous turn timed out after {}s without producing a usable response. The chat request was aborted. Retry with a smaller, more focused next action — call one tool at a time and keep responses concise.",
                                PER_TURN_TIMEOUT.as_secs()
                            ));
                            self.messages.push(nudge.clone());
                            send_event(events, AgentEvent::MessageLogged(nudge))?;
                            continue;
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
            // preserve the partial reasoning trace, reinject it as a user
            // message so the model sees its own conclusions in history, then
            // retry with thinking disabled. This stops the model from
            // re-deliberating the same problem from scratch on every turn —
            // its prior reasoning is now observable state it can act on.
            //
            // We use a *user* role rather than *assistant* because Gemma 4 (and
            // similar templates that gate enable_thinking) treat a trailing
            // assistant message as a "prefill" and raise
            // `Assistant response prefill is incompatible with enable_thinking`
            // on the next chat call. Routing the trace through a user message
            // keeps the same recall benefit without confusing the template.
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
                let nudge = Message::user(format!(
                    "[harness] Your previous turn was cut off after the thinking \
                     budget was reached. Here is your captured reasoning so far:\n\n{}\n\n\
                     Continue from where you left off — call a tool or give a final \
                     answer. Thinking is now disabled for the rest of this run.",
                    combined
                ));
                self.messages.push(nudge.clone());
                send_event(events, AgentEvent::MessageLogged(nudge))?;
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
                if self.should_gate_on_plan() {
                    let unfinished = self.unfinished_plan_steps();
                    if !unfinished.is_empty()
                        && self.plan_gate_retries < self.plan_config.max_gate_retries
                    {
                        self.plan_gate_retries += 1;
                        send_event(
                            events,
                            AgentEvent::PlanGated {
                                remaining: unfinished.clone(),
                            },
                        )?;
                        if !response.content.trim().is_empty() {
                            let assistant_msg = Message::assistant(&response.content);
                            self.messages.push(assistant_msg.clone());
                            send_event(events, AgentEvent::MessageLogged(assistant_msg))?;
                        }
                        let nudge = Message::user(format!(
                            "[Plan gate] You have {} unfinished plan step{}: {}. \
                             Continue working — call plan_mark_in_progress on the next \
                             step, do the work, then plan_mark_done. Use plan_skip_step \
                             with a reason if a step truly does not apply. Do not end \
                             your turn until every step is marked done or skipped.",
                            unfinished.len(),
                            if unfinished.len() == 1 { "" } else { "s" },
                            unfinished.join("; ")
                        ));
                        self.messages.push(nudge.clone());
                        send_event(events, AgentEvent::MessageLogged(nudge))?;
                        continue;
                    }
                    let verify_cmd = if !self.behavioral_verify_passed
                        && self.behavioral_verify_retries < BEHAVIORAL_VERIFY_MAX_RETRIES
                    {
                        discover_test_command()
                    } else {
                        None
                    };
                    if let Some(cmd) = verify_cmd {
                        self.behavioral_verify_retries += 1;
                        send_event(
                            events,
                            AgentEvent::Debug(format!(
                                "[harness] Running behavioural verification: {} (attempt {}/{})",
                                cmd.join(" "),
                                self.behavioral_verify_retries,
                                BEHAVIORAL_VERIFY_MAX_RETRIES
                            )),
                        )?;
                        let (success, summary) = run_behavioral_verification(&cmd).await;
                        if success {
                            self.behavioral_verify_passed = true;
                            send_event(
                                events,
                                AgentEvent::Debug(format!(
                                    "[harness] Behavioural verification passed: {}",
                                    cmd.join(" ")
                                )),
                            )?;
                        } else {
                            let synthetic = format!(
                                "[harness-injected] Fix failing tests from `{}`. Tail of test \
                                 output:\n```\n{}\n```",
                                cmd.join(" "),
                                summary
                            );
                            // Best-effort: if the description exceeds the
                            // step-length cap (large test failure tails), we
                            // truncate so the gate still fires.
                            let synthetic = if synthetic.len() > crate::agent::plan::MAX_STEP_DESC_LEN {
                                let mut truncated = synthetic[..crate::agent::plan::MAX_STEP_DESC_LEN.saturating_sub(3)].to_string();
                                truncated.push_str("...");
                                truncated
                            } else {
                                synthetic
                            };
                            let _ = self.plan
                                .lock()
                                .expect("plan lock poisoned")
                                .add(&synthetic);
                            if !response.content.trim().is_empty() {
                                let assistant_msg = Message::assistant(&response.content);
                                self.messages.push(assistant_msg.clone());
                                send_event(
                                    events,
                                    AgentEvent::MessageLogged(assistant_msg),
                                )?;
                            }
                            let nudge = Message::user(format!(
                                "[Behavioural verification gate] Tests failed: `{}` exited \
                                 non-zero. Plan was reopened with a synthetic step containing \
                                 the failure tail. Address the failures before declaring done. \
                                 cargo check passing is not enough — the integration tests \
                                 exercise behaviour that structural checks cannot.",
                                cmd.join(" ")
                            ));
                            self.messages.push(nudge.clone());
                            send_event(events, AgentEvent::MessageLogged(nudge))?;
                            continue;
                        }
                    }
                }

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

/// Cap on how many times the behavioural-verification gate runs in a single
/// agent run. After hitting the cap the gate stops re-running and lets the
/// model exit (with the last failure visible in plan history). Prevents an
/// infinite loop on a flaky or unfixable test suite.
const BEHAVIORAL_VERIFY_MAX_RETRIES: u32 = 3;

/// Wall-clock cap on a single behavioural-verification command. A hung test
/// process must not be allowed to wedge the agent forever.
const BEHAVIORAL_VERIFY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// Heuristic: skip the planning phase for short, action-verb-free requests
/// (chit-chat, simple questions). Without this, every prompt — including
/// "what is 2+2" — triggers a full planner sub-agent run, doubling latency.
fn is_trivial_request(input: &str) -> bool {
    const ACTION_VERBS: &[&str] = &[
        "implement", "edit", "modify", "update", "change", "fix", "add", "remove",
        "replace", "refactor", "rewrite", "write", "create", "delete", "build",
        "run ", "test", "debug", "review", "audit", "investigate", "diagnose",
    ];
    if input.len() > 120 {
        return false;
    }
    let lower = input.to_lowercase();
    if ACTION_VERBS.iter().any(|v| lower.contains(v)) {
        return false;
    }
    // File-path-ish tokens (contain a slash + dot or a leading ./) imply
    // "operate on this file", which warrants planning.
    if input.split_whitespace().any(|tok| {
        tok.starts_with("./") || tok.starts_with("/") || (tok.contains('/') && tok.contains('.'))
    }) {
        return false;
    }
    true
}

/// Discover the project's behavioural-test command from the working
/// directory's layout. Preference order matches "most-specific to least":
/// xtask binary (`cargo xtask test`) → `cargo test` → `npm test`. None when
/// no recognised test infrastructure is present (the gate then becomes a
/// no-op for that project).
fn discover_test_command() -> Option<Vec<String>> {
    if std::path::Path::new("xtask/Cargo.toml").exists() {
        return Some(vec![
            "cargo".to_string(),
            "xtask".to_string(),
            "test".to_string(),
        ]);
    }
    if std::path::Path::new("Cargo.toml").exists() {
        return Some(vec!["cargo".to_string(), "test".to_string()]);
    }
    if std::path::Path::new("package.json").exists() {
        return Some(vec!["npm".to_string(), "test".to_string()]);
    }
    None
}

/// Run the discovered test command and return `(success, tail_summary)`.
/// The tail is the last 50 lines of merged stdout+stderr — enough to surface
/// failure messages without flooding context. Bounded by
/// `BEHAVIORAL_VERIFY_TIMEOUT` with `kill_on_drop` so a hung test process
/// can't wedge the agent.
async fn run_behavioral_verification(cmd: &[String]) -> (bool, String) {
    let mut command = tokio::process::Command::new(&cmd[0]);
    command.args(&cmd[1..]).kill_on_drop(true);
    let output = match tokio::time::timeout(BEHAVIORAL_VERIFY_TIMEOUT, command.output()).await {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            return (false, format!("[command failed to launch: {}]", e));
        }
        Err(_) => {
            return (
                false,
                format!(
                    "[command timed out after {}s; killed]",
                    BEHAVIORAL_VERIFY_TIMEOUT.as_secs()
                ),
            );
        }
    };
    let success = output.status.success();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}\n{}", stdout, stderr);
    let lines: Vec<&str> = combined.lines().collect();
    let tail_start = lines.len().saturating_sub(50);
    let summary = lines[tail_start..].join("\n");
    (success, summary)
}
