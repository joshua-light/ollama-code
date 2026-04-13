use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Hook events
// ---------------------------------------------------------------------------

/// The lifecycle events where hooks can be triggered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookEvent {
    PreToolExecute,
    PostToolExecute,
    AgentStart,
    AgentDone,
}

impl HookEvent {
    fn as_str(&self) -> &'static str {
        match self {
            Self::PreToolExecute => "pre_tool_execute",
            Self::PostToolExecute => "post_tool_execute",
            Self::AgentStart => "agent_start",
            Self::AgentDone => "agent_done",
        }
    }
}

// ---------------------------------------------------------------------------
// Hook configuration (parsed from hooks.toml)
// ---------------------------------------------------------------------------

/// A single hook entry within a named hook section.
#[derive(Debug, Clone, Deserialize)]
pub struct HookEntry {
    pub event: HookEvent,
    pub command: String,
    /// Only run for specific tools (pre/post_tool_execute only).
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Timeout in seconds (default: 30).
    #[serde(default)]
    pub timeout: Option<u64>,
    /// If true, hook failure = deny/abort. Default: false (fail-open).
    #[serde(default)]
    pub fail_closed: Option<bool>,
    /// Priority for ordering (lower = earlier). Default: 50.
    #[serde(default)]
    pub priority: Option<i32>,
}

const DEFAULT_HOOK_TIMEOUT_SECS: u64 = 30;
const DEFAULT_HOOK_PRIORITY: i32 = 50;

impl HookEntry {
    fn timeout_duration(&self) -> Duration {
        Duration::from_secs(self.timeout.unwrap_or(DEFAULT_HOOK_TIMEOUT_SECS))
    }

    fn fail_closed(&self) -> bool {
        self.fail_closed.unwrap_or(false)
    }

    fn priority(&self) -> i32 {
        self.priority.unwrap_or(DEFAULT_HOOK_PRIORITY)
    }

    /// Check whether this hook should fire for the given tool name.
    fn matches_tool(&self, tool_name: &str) -> bool {
        match &self.tools {
            None => true,
            Some(tools) => tools.iter().any(|t| t == tool_name),
        }
    }
}

/// A resolved hook: entry + its name and base directory for relative commands.
#[derive(Debug, Clone)]
struct ResolvedHook {
    name: String,
    entry: HookEntry,
    /// Base directory for resolving relative command paths.
    base_dir: PathBuf,
    /// Hook-specific config from `[hooks.<name>]` in config.toml.
    config: Option<toml::map::Map<String, toml::Value>>,
}

// ---------------------------------------------------------------------------
// Hook input/output protocol (JSON on stdin/stdout)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct HookInput {
    hook: String,
    data: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<Value>,
}

/// Result of running a pre_tool_execute hook.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PreToolResult {
    #[serde(default)]
    pub action: Option<String>,
    /// Modified arguments (when action = "modify").
    #[serde(default)]
    pub arguments: Option<Value>,
    /// Denial reason (when action = "deny").
    #[serde(default)]
    pub message: Option<String>,
}

/// Result of running a post_tool_execute hook.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PostToolResult {
    #[serde(default)]
    pub action: Option<String>,
    /// Replacement output (when action = "modify").
    #[serde(default)]
    pub output: Option<String>,
    /// Override success flag (when action = "modify").
    #[serde(default)]
    pub success: Option<bool>,
}

/// Result of running an agent_start hook.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AgentStartResult {
    #[serde(default)]
    pub action: Option<String>,
    /// Extra context to inject into the system prompt.
    #[serde(default)]
    pub system_context: Option<String>,
}

/// Result of running an agent_done hook.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AgentDoneResult {
    #[serde(default)]
    pub action: Option<String>,
    /// Rewritten response (when action = "modify").
    #[serde(default)]
    pub response: Option<String>,
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Load hooks from a `hooks.toml` file. Each top-level key is a hook name,
/// and the value is a table matching `HookEntry`.
fn load_hooks_file(path: &Path) -> Vec<(String, HookEntry)> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let table: toml::map::Map<String, toml::Value> = match toml::from_str(&content) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    let mut hooks = Vec::new();
    for (name, value) in table {
        if let Ok(entry) = value.try_into::<HookEntry>() {
            hooks.push((name, entry));
        }
    }
    hooks
}

/// Default user hooks file: `~/.config/ollama-code/hooks.toml`.
fn user_hooks_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ollama-code")
        .join("hooks.toml")
}

/// Walk up from `cwd` looking for `.agents/hooks.toml`.
fn find_project_hooks(cwd: &str) -> Option<PathBuf> {
    let mut dir = Path::new(cwd).to_path_buf();
    loop {
        let candidate = dir.join(".agents").join("hooks.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

// ---------------------------------------------------------------------------
// HookRunner — the public API
// ---------------------------------------------------------------------------

/// Manages discovered hooks and executes them.
pub struct HookRunner {
    hooks: Vec<ResolvedHook>,
}

impl HookRunner {
    /// Discover and load hooks from user and project locations.
    ///
    /// - User hooks: `~/.config/ollama-code/hooks.toml`
    /// - Project hooks: `.agents/hooks.toml` (walk up from cwd)
    /// - Project hooks override user hooks by name.
    /// - Hooks disabled via `[hooks] name = false` in config are filtered out.
    pub fn discover(cwd: &str, config: Option<&crate::config::Config>) -> Self {
        let mut hooks_map: std::collections::HashMap<String, ResolvedHook> =
            std::collections::HashMap::new();

        // User-level hooks
        let user_path = user_hooks_path();
        if user_path.is_file() {
            let base = user_path.parent().unwrap_or(Path::new(".")).to_path_buf();
            for (name, entry) in load_hooks_file(&user_path) {
                let hook_config = config
                    .and_then(|c| c.hook_config(&name))
                    .cloned();
                hooks_map.insert(
                    name.clone(),
                    ResolvedHook {
                        name,
                        entry,
                        base_dir: base.clone(),
                        config: hook_config,
                    },
                );
            }
        }

        // Project-level hooks (override user by name)
        if let Some(project_path) = find_project_hooks(cwd) {
            let base = project_path
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf();
            for (name, entry) in load_hooks_file(&project_path) {
                let hook_config = config
                    .and_then(|c| c.hook_config(&name))
                    .cloned();
                hooks_map.insert(
                    name.clone(),
                    ResolvedHook {
                        name,
                        entry,
                        base_dir: base.clone(),
                        config: hook_config,
                    },
                );
            }
        }

        // Filter disabled hooks
        let hooks: Vec<ResolvedHook> = if let Some(cfg) = config {
            hooks_map
                .into_values()
                .filter(|h| cfg.is_hook_enabled(&h.name))
                .collect()
        } else {
            hooks_map.into_values().collect()
        };

        // Sort by priority then name
        let mut sorted = hooks;
        sorted.sort_by(|a, b| {
            a.entry
                .priority()
                .cmp(&b.entry.priority())
                .then_with(|| a.name.cmp(&b.name))
        });

        Self { hooks: sorted }
    }

    /// Create an empty hook runner (no hooks).
    pub fn empty() -> Self {
        Self { hooks: Vec::new() }
    }

    // -----------------------------------------------------------------------
    // Event runners
    // -----------------------------------------------------------------------

    /// Run `pre_tool_execute` hooks for the given tool call.
    ///
    /// Returns `Ok(PreToolResult)` with the cumulative result.
    /// A "deny" from any hook short-circuits immediately.
    pub async fn pre_tool_execute(
        &self,
        tool_name: &str,
        arguments: &Value,
    ) -> Result<PreToolResult> {
        let matching: Vec<&ResolvedHook> = self
            .hooks
            .iter()
            .filter(|h| h.entry.event == HookEvent::PreToolExecute)
            .filter(|h| h.entry.matches_tool(tool_name))
            .collect();

        let mut result = PreToolResult::default();
        let mut current_args = arguments.clone();

        for hook in matching {
            let data = serde_json::json!({
                "tool_name": tool_name,
                "arguments": current_args,
            });
            match execute_hook(hook, HookEvent::PreToolExecute, data).await {
                Ok(Some(output)) => {
                    if let Ok(r) = serde_json::from_value::<PreToolResult>(output) {
                        match r.action.as_deref() {
                            Some("deny") => return Ok(r),
                            Some("modify") => {
                                if let Some(args) = r.arguments {
                                    current_args = args;
                                }
                                result.action = Some("modify".to_string());
                                result.arguments = Some(current_args.clone());
                            }
                            _ => {} // "proceed" or absent — continue
                        }
                    }
                }
                Ok(None) => {} // no output = passthrough
                Err(e) => {
                    if hook.entry.fail_closed() {
                        return Ok(PreToolResult {
                            action: Some("deny".to_string()),
                            message: Some(format!("Hook '{}' failed: {}", hook.name, e)),
                            arguments: None,
                        });
                    }
                    eprintln!("[hooks] warning: '{}' failed (fail-open): {}", hook.name, e);
                }
            }
        }

        Ok(result)
    }

    /// Run `post_tool_execute` hooks for the given tool result.
    ///
    /// Returns the (possibly modified) output and success flag.
    pub async fn post_tool_execute(
        &self,
        tool_name: &str,
        arguments: &Value,
        output: String,
        success: bool,
    ) -> (String, bool) {
        let matching: Vec<&ResolvedHook> = self
            .hooks
            .iter()
            .filter(|h| h.entry.event == HookEvent::PostToolExecute)
            .filter(|h| h.entry.matches_tool(tool_name))
            .collect();

        if matching.is_empty() {
            return (output, success);
        }

        let mut current_output = output;
        let mut current_success = success;

        for hook in matching {
            let data = serde_json::json!({
                "tool_name": tool_name,
                "arguments": arguments,
                "output": current_output,
                "success": current_success,
            });
            match execute_hook(hook, HookEvent::PostToolExecute, data).await {
                Ok(Some(value)) => {
                    if let Ok(r) = serde_json::from_value::<PostToolResult>(value) {
                        if r.action.as_deref() == Some("modify") {
                            if let Some(new_output) = r.output {
                                current_output = new_output;
                            }
                            if let Some(new_success) = r.success {
                                current_success = new_success;
                            }
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    if hook.entry.fail_closed() {
                        eprintln!(
                            "[hooks] warning: post_tool_execute '{}' failed (fail-closed): {}",
                            hook.name, e
                        );
                    } else {
                        eprintln!("[hooks] warning: '{}' failed (fail-open): {}", hook.name, e);
                    }
                }
            }
        }

        (current_output, current_success)
    }

    /// Run `agent_start` hooks. Returns any extra system context to inject.
    pub async fn agent_start(
        &self,
        user_message: &str,
        model: &str,
    ) -> Option<String> {
        let matching: Vec<&ResolvedHook> = self
            .hooks
            .iter()
            .filter(|h| h.entry.event == HookEvent::AgentStart)
            .collect();

        let mut system_context = None;

        for hook in matching {
            let data = serde_json::json!({
                "user_message": user_message,
                "model": model,
            });
            match execute_hook(hook, HookEvent::AgentStart, data).await {
                Ok(Some(value)) => {
                    if let Ok(r) = serde_json::from_value::<AgentStartResult>(value) {
                        if let Some(ctx) = r.system_context {
                            // Accumulate context from multiple hooks
                            match &mut system_context {
                                Some(existing) => {
                                    let s: &mut String = existing;
                                    s.push('\n');
                                    s.push_str(&ctx);
                                }
                                None => {
                                    system_context = Some(ctx);
                                }
                            }
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("[hooks] warning: agent_start '{}' failed: {}", hook.name, e);
                }
            }
        }

        system_context
    }

    /// Run `agent_done` hooks. Returns a possibly rewritten response.
    pub async fn agent_done(
        &self,
        response: &str,
        tool_calls_count: u32,
        model: &str,
    ) -> Option<String> {
        let matching: Vec<&ResolvedHook> = self
            .hooks
            .iter()
            .filter(|h| h.entry.event == HookEvent::AgentDone)
            .collect();

        let mut current_response = response.to_string();
        let mut modified = false;

        for hook in matching {
            let data = serde_json::json!({
                "response": current_response,
                "tool_calls_count": tool_calls_count,
                "model": model,
            });
            match execute_hook(hook, HookEvent::AgentDone, data).await {
                Ok(Some(value)) => {
                    if let Ok(r) = serde_json::from_value::<AgentDoneResult>(value) {
                        if r.action.as_deref() == Some("modify") {
                            if let Some(new_resp) = r.response {
                                current_response = new_resp;
                                modified = true;
                            }
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("[hooks] warning: agent_done '{}' failed: {}", hook.name, e);
                }
            }
        }

        if modified {
            Some(current_response)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Subprocess execution
// ---------------------------------------------------------------------------

/// Execute a single hook as a subprocess. Returns parsed JSON output or None
/// if stdout was empty.
async fn execute_hook(
    hook: &ResolvedHook,
    event: HookEvent,
    data: Value,
) -> Result<Option<Value>> {
    let input = HookInput {
        hook: event.as_str().to_string(),
        data,
        config: hook
            .config
            .as_ref()
            .map(serde_json::to_value)
            .transpose()?,
    };

    let stdin_data = serde_json::to_string(&input)?;

    // Resolve command path (relative to base_dir)
    let command_str = &hook.entry.command;
    let (program, args) = parse_command(command_str);
    let program_path = if program.starts_with('/') || program.starts_with('.') {
        let p = PathBuf::from(&program);
        if p.is_relative() {
            hook.base_dir.join(p)
        } else {
            p
        }
    } else {
        PathBuf::from(&program)
    };

    let timeout = hook.entry.timeout_duration();

    // Spawn in a blocking task to avoid blocking the tokio runtime
    let program_path_clone = program_path.clone();
    let hook_name = hook.name.clone();
    let base_dir = hook.base_dir.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut child = Command::new(&program_path_clone)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(std::env::current_dir().unwrap_or(base_dir))
            .spawn()
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to spawn hook '{}' ({}): {}",
                    hook_name,
                    program_path_clone.display(),
                    e
                )
            })?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(stdin_data.as_bytes())?;
        }

        crate::process::wait_with_timeout(&mut child, timeout, &format!("Hook '{}'", hook_name))
    })
    .await??;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        anyhow::bail!(
            "Hook '{}' exited with {}: {}",
            hook.name,
            result.status,
            stderr.trim()
        );
    }

    let stdout = String::from_utf8_lossy(&result.stdout);
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let value: Value = serde_json::from_str(trimmed).map_err(|e| {
        anyhow::anyhow!(
            "Hook '{}' returned invalid JSON: {}: {}",
            hook.name,
            e,
            &trimmed[..trimmed.len().min(200)]
        )
    })?;

    Ok(Some(value))
}

/// Split a command string into program and arguments by whitespace.
/// Does not handle quoting — commands with quoted arguments should use
/// a wrapper script instead.
fn parse_command(cmd: &str) -> (String, Vec<String>) {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return (cmd.to_string(), Vec::new());
    }
    let program = parts[0].to_string();
    let args = parts[1..].iter().map(|s| s.to_string()).collect();
    (program, args)
}

