//! Configuration + protocol types for the hooks subsystem.

use std::path::PathBuf;
use std::time::Duration;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    pub(super) fn as_str(&self) -> &'static str {
        match self {
            Self::PreToolExecute => "pre_tool_execute",
            Self::PostToolExecute => "post_tool_execute",
            Self::AgentStart => "agent_start",
            Self::AgentDone => "agent_done",
        }
    }
}

/// A single hook entry within a named hook section.
#[derive(Debug, Clone, Deserialize)]
pub struct HookEntry {
    pub event: HookEvent,
    pub command: String,
    /// Only run for specific tools (pre/post_tool_execute only).
    /// Each entry is treated as an anchored regex pattern (`^pattern$`).
    /// Plain strings like `"bash"` still work as exact matches.
    /// Use regex features for flexible matching: `"file_.*"`, `"bash|write_file"`.
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Regex pattern matched against argument string values.
    /// If set, the hook only fires when at least one string value in the
    /// tool arguments matches this pattern (unanchored search).
    #[serde(default)]
    pub if_args: Option<String>,
    /// Timeout in seconds (default: 30).
    #[serde(default)]
    pub timeout: Option<u64>,
    /// If true, hook failure = deny/abort. Default: false (fail-open).
    #[serde(default)]
    pub fail_closed: Option<bool>,
    /// Priority for ordering (lower = earlier). Default: 50.
    #[serde(default)]
    pub priority: Option<i32>,

    // -- Compiled patterns (populated by `compile_patterns()` after load) --
    #[serde(skip)]
    compiled_tools: Vec<Regex>,
    #[serde(skip)]
    compiled_if_args: Option<Regex>,
}

const DEFAULT_HOOK_TIMEOUT_SECS: u64 = 30;
const DEFAULT_HOOK_PRIORITY: i32 = 50;

impl HookEntry {
    pub(super) fn timeout_duration(&self) -> Duration {
        Duration::from_secs(self.timeout.unwrap_or(DEFAULT_HOOK_TIMEOUT_SECS))
    }

    pub(super) fn fail_closed(&self) -> bool {
        self.fail_closed.unwrap_or(false)
    }

    pub(super) fn priority(&self) -> i32 {
        self.priority.unwrap_or(DEFAULT_HOOK_PRIORITY)
    }

    /// Compile `tools` and `if_args` patterns into cached regexes.
    /// Called once after deserialization so matching is allocation-free.
    pub(super) fn compile_patterns(&mut self) {
        if let Some(tools) = &self.tools {
            self.compiled_tools = tools
                .iter()
                .map(|pattern| {
                    let anchored = format!("^(?:{})$", pattern);
                    Regex::new(&anchored).unwrap_or_else(|_| {
                        // Bad regex — fall back to literal match
                        Regex::new(&format!("^{}$", regex::escape(pattern))).unwrap()
                    })
                })
                .collect();
        }
        if let Some(pattern) = &self.if_args {
            self.compiled_if_args = Regex::new(pattern).ok();
        }
    }

    /// Check whether this hook should fire for the given tool name.
    pub(super) fn matches_tool(&self, tool_name: &str) -> bool {
        match &self.tools {
            None => true,
            Some(_) => self.compiled_tools.iter().any(|re| re.is_match(tool_name)),
        }
    }

    /// Check whether the tool arguments match the `if_args` pattern.
    /// Returns `true` if `if_args` is not set.
    pub(super) fn matches_args(&self, arguments: &Value) -> bool {
        match &self.compiled_if_args {
            None => self.if_args.is_none(),
            Some(re) => super::exec::any_string_matches(arguments, re),
        }
    }
}

/// A resolved hook: entry + its name and base directory for relative commands.
#[derive(Debug, Clone)]
pub(super) struct ResolvedHook {
    pub name: String,
    pub entry: HookEntry,
    /// Base directory for resolving relative command paths.
    pub base_dir: PathBuf,
    /// Hook-specific config from `[hooks.<name>]` in config.toml.
    pub config: Option<toml::map::Map<String, toml::Value>>,
}

/// JSON payload written to a hook subprocess's stdin.
#[derive(Debug, Serialize)]
pub(super) struct HookInput {
    pub hook: String,
    pub data: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
}

/// Result of running a pre_tool_execute hook.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PreToolResult {
    #[serde(default)]
    pub action: Option<String>,
    /// Modified arguments (when action = "modify" or "rewrite").
    #[serde(default)]
    pub arguments: Option<Value>,
    /// Replacement tool name (when action = "rewrite").
    #[serde(default)]
    pub tool_name: Option<String>,
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
