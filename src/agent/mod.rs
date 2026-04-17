mod builder;
mod context;
mod events;
mod subagent;
mod tool_executor;
mod util;

use std::collections::HashSet;
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
            context: ContextManager {
                trim_threshold_pct: cfg.effective_trim_threshold(),
                trim_target_pct: cfg.effective_trim_target(),
                compaction_enabled: cfg.effective_context_compaction(),
            },
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
        self.context.trim_threshold_pct = config.effective_trim_threshold();
        self.context.trim_target_pct = config.effective_trim_target();
        self.context.compaction_enabled = config.effective_context_compaction();

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
                if self.context.compaction_enabled {
                    match self
                        .context
                        .compact(
                            &mut self.messages,
                            self.context_size,
                            response.prompt_eval_count,
                            &*self.backend,
                            &self.model,
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
                            if let Some((removed, freed)) = self.context.trim(
                                &mut self.messages,
                                self.context_size,
                                response.prompt_eval_count,
                            ) {
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
                } else if let Some((removed, freed)) = self.context.trim(
                    &mut self.messages,
                    self.context_size,
                    response.prompt_eval_count,
                ) {
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
