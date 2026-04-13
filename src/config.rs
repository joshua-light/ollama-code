use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::mcp::McpServerConfig;

pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
pub const DEFAULT_CONTEXT_SIZE: u64 = 32768;
pub const DEFAULT_SUBAGENT_MAX_TURNS: u16 = 15;
pub const DEFAULT_BASH_TIMEOUT_SECS: u64 = 120;
pub const DEFAULT_TRIM_THRESHOLD_PCT: u8 = 80;
pub const DEFAULT_TRIM_TARGET_PCT: u8 = 60;
pub const DEFAULT_REINJECTION_INTERVAL: u16 = 3;

/// Root data directory for ollama-code (`$XDG_DATA_HOME/ollama-code` or `./ollama-code`).
pub fn data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ollama-code")
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub context_size: Option<u64>,

    /// Backend: "ollama" (default) or "llama-cpp".
    #[serde(default)]
    pub backend: Option<String>,

    /// Path to llama-server binary (for llama-cpp backend).
    #[serde(default)]
    pub llama_server_path: Option<String>,

    /// URL of a remote llama-server (e.g. "http://192.168.1.50:8080").
    /// When set, connects directly without spawning a local server.
    #[serde(default)]
    pub llama_server_url: Option<String>,

    /// Path to a GGUF model file (for llama-cpp backend).
    #[serde(default)]
    pub model_path: Option<String>,

    /// HuggingFace repo to download the model from (e.g. "google/gemma-3-27b-it-GGUF").
    /// Used by llama-server's -hf flag. Takes precedence over Ollama blob resolution.
    #[serde(default)]
    pub hf_repo: Option<String>,

    /// Extra arguments passed to llama-server (e.g. ["-ngl", "99"]).
    #[serde(default)]
    pub llama_server_args: Option<Vec<String>>,

    /// Timeout for bash tool commands in seconds (default: 120).
    #[serde(default)]
    pub bash_timeout: Option<u64>,

    /// Maximum number of agent-loop turns a sub-agent is allowed (default: 15).
    #[serde(default)]
    pub subagent_max_turns: Option<u16>,

    /// Show estimated Opus 4.6 cost on the status line (default: false).
    #[serde(default)]
    pub show_cost_estimate: Option<bool>,

    /// Ollama API base URL (default: "http://localhost:11434").
    #[serde(default)]
    pub ollama_url: Option<String>,

    /// Auto-approve all tool calls (skip confirmation prompts).
    #[serde(default)]
    pub no_confirm: Option<bool>,

    /// Enable verbose/debug output.
    #[serde(default)]
    pub verbose: Option<bool>,

    /// Enable bypass mode (auto-approve all tool calls) by default.
    #[serde(default)]
    pub bypass: Option<bool>,

    /// Plugin feature flags and configuration.
    ///
    /// Boolean values enable/disable tools by name:
    ///   `bash = false` disables the built-in bash tool.
    ///
    /// Table values provide plugin-specific configuration:
    ///   `[plugins.my-plugin]`
    ///   `key = "value"`
    #[serde(default)]
    pub plugins: Option<HashMap<String, toml::Value>>,

    /// Additional directories to search for plugins (for testing / advanced use).
    /// Each path is scanned for `*/PLUGIN.toml`.
    #[serde(default)]
    pub plugin_dirs: Option<Vec<String>>,

    /// MCP (Model Context Protocol) servers.
    ///
    /// Each entry spawns a server process at startup and registers its tools:
    ///   `[mcp_servers.filesystem]`
    ///   `command = "npx"`
    ///   `args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]`
    #[serde(default)]
    pub mcp_servers: Option<HashMap<String, McpServerConfig>>,

    /// Hook feature flags and per-hook configuration.
    ///
    /// Boolean values enable/disable hooks by name:
    ///   `my-hook = false` disables the hook.
    ///
    /// Table values provide hook-specific configuration passed on stdin:
    ///   `[hooks.my-hook]`
    ///   `key = "value"`
    #[serde(default)]
    pub hooks: Option<HashMap<String, toml::Value>>,

    /// Sampling temperature (0.0 = deterministic, higher = more random).
    /// Recommended: 0.1-0.3 for tool use, 0.5-0.8 for creative/planning.
    #[serde(default)]
    pub temperature: Option<f64>,

    /// Top-p (nucleus) sampling. Only tokens with cumulative probability <= top_p are considered.
    #[serde(default)]
    pub top_p: Option<f64>,

    /// Top-k sampling. Only the top k tokens are considered.
    #[serde(default)]
    pub top_k: Option<u32>,

    /// Enable dynamic tool scoping: only present relevant tools per turn.
    /// Reduces confusion in small models (the "Chekhov's gun" problem).
    #[serde(default)]
    pub tool_scoping: Option<bool>,

    /// Enable periodic task re-injection to fight coherence decay in small models.
    /// Re-states the current task objective every `reinjection_interval` turns.
    #[serde(default)]
    pub task_reinjection: Option<bool>,

    /// How often to re-inject the task objective (in agent turns). Default: 3.
    #[serde(default)]
    pub reinjection_interval: Option<u16>,

    /// Context trim threshold as percentage of context_size (default: 80).
    /// When prompt tokens exceed this, auto-trimming kicks in.
    #[serde(default)]
    pub trim_threshold: Option<u8>,

    /// Context trim target as percentage of context_size (default: 60).
    /// Trimming removes messages until usage drops to this level.
    #[serde(default)]
    pub trim_target: Option<u8>,

    /// Recently used HuggingFace model repos (most recent first, max 10).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recent_hf_models: Option<Vec<String>>,

    /// Path to the project config file that was loaded (if any). Not serialized.
    #[serde(skip)]
    pub project_config_path: Option<PathBuf>,
}

/// Walk up from `start_dir` looking for `.ollama-code.toml`.
/// Returns `None` if not found (hits filesystem root).
pub fn find_project_config(start_dir: &Path) -> Option<PathBuf> {
    let mut dir = start_dir.to_path_buf();
    loop {
        let candidate = dir.join(".ollama-code.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Check whether a named entry is enabled in an `Option<HashMap<String, toml::Value>>`.
///
/// Returns `false` only for an explicit `name = false`. Everything else
/// (missing key, `true`, table value, absent map) means enabled.
fn is_feature_enabled(map: &Option<HashMap<String, toml::Value>>, name: &str) -> bool {
    match map {
        Some(map) => !matches!(map.get(name), Some(toml::Value::Boolean(false))),
        None => true,
    }
}

/// Return the config table for a named entry, if it exists as a TOML table.
fn feature_config<'a>(
    map: &'a Option<HashMap<String, toml::Value>>,
    name: &str,
) -> Option<&'a toml::map::Map<String, toml::Value>> {
    map.as_ref()
        .and_then(|m| m.get(name))
        .and_then(|v| v.as_table())
}

impl Config {
    pub fn path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("ollama-code")
            .join("config.toml")
    }

    /// Load config from a specific file path. Returns default if file doesn't exist.
    fn load_file(path: &Path) -> Result<Self> {
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            Ok(toml::from_str(&content)?)
        } else {
            Ok(Self::default())
        }
    }

    /// Load layered config: project config (if found) merged over user config.
    /// CLI flags should be applied separately after this call.
    pub fn load() -> Result<Self> {
        let user = Self::load_file(&Self::path())?;

        let project_path = std::env::current_dir()
            .ok()
            .and_then(|dir| find_project_config(&dir));

        match project_path {
            Some(ref pp) => {
                let project = Self::load_file(pp)?;
                let mut merged = project.merge(&user);
                merged.project_config_path = Some(pp.clone());
                Ok(merged)
            }
            None => Ok(user),
        }
    }

    /// Merge `self` (higher priority) over `other` (lower priority).
    /// For each field, self's `Some` wins over other's `Some`.
    /// `recent_hf_models` always comes from `other` (user-scope only).
    pub fn merge(self, other: &Config) -> Config {
        fn merge_hashmap<V: Clone>(
            hi: Option<HashMap<String, V>>,
            lo: &Option<HashMap<String, V>>,
        ) -> Option<HashMap<String, V>> {
            match (hi, lo) {
                (Some(mut hi), Some(lo)) => {
                    for (k, v) in lo {
                        hi.entry(k.clone()).or_insert_with(|| v.clone());
                    }
                    Some(hi)
                }
                (hi @ Some(_), _) => hi,
                (None, lo) => lo.clone(),
            }
        }

        Config {
            model: self.model.or_else(|| other.model.clone()),
            context_size: self.context_size.or(other.context_size),
            backend: self.backend.or_else(|| other.backend.clone()),
            llama_server_path: self.llama_server_path.or_else(|| other.llama_server_path.clone()),
            llama_server_url: self.llama_server_url.or_else(|| other.llama_server_url.clone()),
            model_path: self.model_path.or_else(|| other.model_path.clone()),
            hf_repo: self.hf_repo.or_else(|| other.hf_repo.clone()),
            llama_server_args: self.llama_server_args.or_else(|| other.llama_server_args.clone()),
            bash_timeout: self.bash_timeout.or(other.bash_timeout),
            subagent_max_turns: self.subagent_max_turns.or(other.subagent_max_turns),
            show_cost_estimate: self.show_cost_estimate.or(other.show_cost_estimate),
            ollama_url: self.ollama_url.or_else(|| other.ollama_url.clone()),
            no_confirm: self.no_confirm.or(other.no_confirm),
            verbose: self.verbose.or(other.verbose),
            bypass: self.bypass.or(other.bypass),
            plugins: merge_hashmap(self.plugins, &other.plugins),
            plugin_dirs: self.plugin_dirs.or_else(|| other.plugin_dirs.clone()),
            mcp_servers: merge_hashmap(self.mcp_servers, &other.mcp_servers),
            hooks: merge_hashmap(self.hooks, &other.hooks),
            temperature: self.temperature.or(other.temperature),
            top_p: self.top_p.or(other.top_p),
            top_k: self.top_k.or(other.top_k),
            tool_scoping: self.tool_scoping.or(other.tool_scoping),
            task_reinjection: self.task_reinjection.or(other.task_reinjection),
            reinjection_interval: self.reinjection_interval.or(other.reinjection_interval),
            trim_threshold: self.trim_threshold.or(other.trim_threshold),
            trim_target: self.trim_target.or(other.trim_target),
            // User-scope only: always take from the lower-priority layer (user config).
            recent_hf_models: other.recent_hf_models.clone(),
            project_config_path: None,
        }
    }

    pub fn bash_timeout_duration(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.bash_timeout.unwrap_or(DEFAULT_BASH_TIMEOUT_SECS))
    }

    pub fn effective_subagent_max_turns(&self) -> u16 {
        self.subagent_max_turns.unwrap_or(DEFAULT_SUBAGENT_MAX_TURNS)
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, toml::to_string_pretty(self)?)?;
        Ok(())
    }

    /// Check whether a tool/plugin is enabled.
    pub fn is_tool_enabled(&self, name: &str) -> bool {
        is_feature_enabled(&self.plugins, name)
    }

    /// Return the plugin-specific config table for `name`, if any.
    pub fn plugin_config(&self, name: &str) -> Option<&toml::map::Map<String, toml::Value>> {
        feature_config(&self.plugins, name)
    }

    /// Check whether a hook is enabled.
    pub fn is_hook_enabled(&self, name: &str) -> bool {
        is_feature_enabled(&self.hooks, name)
    }

    /// Return the hook-specific config table for `name`, if any.
    pub fn hook_config(&self, name: &str) -> Option<&toml::map::Map<String, toml::Value>> {
        feature_config(&self.hooks, name)
    }

    pub fn effective_trim_threshold(&self) -> u8 {
        self.trim_threshold.unwrap_or(DEFAULT_TRIM_THRESHOLD_PCT)
    }

    pub fn effective_trim_target(&self) -> u8 {
        self.trim_target.unwrap_or(DEFAULT_TRIM_TARGET_PCT)
    }

    pub fn effective_reinjection_interval(&self) -> u16 {
        self.reinjection_interval.unwrap_or(DEFAULT_REINJECTION_INTERVAL)
    }

    /// Add a HuggingFace repo to the recent list (most recent first, max 10).
    pub fn add_recent_hf_model(&mut self, repo: &str) {
        let mut list = self.recent_hf_models.take().unwrap_or_default();
        list.retain(|m| m != repo);
        list.insert(0, repo.to_string());
        list.truncate(10);
        self.recent_hf_models = Some(list);
    }
}
