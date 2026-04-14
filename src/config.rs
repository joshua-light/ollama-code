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

    /// Show estimated cost on the status line (default: false).
    #[serde(default)]
    pub show_cost_estimate: Option<bool>,

    /// Cost rates as (input, output) dollars per million tokens.
    /// Default: (5.0, 25.0) matching Anthropic Opus 4.6 pricing.
    /// Set to (0.0, 0.0) for local Ollama models.
    #[serde(default)]
    pub cost_per_million_tokens: Option<(f64, f64)>,

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
    ///   `[mcp.filesystem]`
    ///   `command = "npx"`
    ///   `args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]`
    #[serde(default)]
    pub mcp: Option<HashMap<String, McpServerConfig>>,

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
            cost_per_million_tokens: self.cost_per_million_tokens.or(other.cost_per_million_tokens),
            ollama_url: self.ollama_url.or_else(|| other.ollama_url.clone()),
            no_confirm: self.no_confirm.or(other.no_confirm),
            verbose: self.verbose.or(other.verbose),
            bypass: self.bypass.or(other.bypass),
            plugins: merge_hashmap(self.plugins, &other.plugins),
            plugin_dirs: self.plugin_dirs.or_else(|| other.plugin_dirs.clone()),
            mcp: merge_hashmap(self.mcp, &other.mcp),
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

    /// If the user config file does not exist, write a commented template so
    /// the user can discover and edit available options.
    pub fn ensure_default_config() -> Result<()> {
        let path = Self::path();
        if path.exists() {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, Self::default_template())?;
        Ok(())
    }

    fn default_template() -> &'static str {
        r#"# ollama-code configuration
# Uncomment and edit the settings you want to change.

# Model to use (e.g. "qwen2.5-coder:7b", "gemma3:12b")
# model = "qwen2.5-coder:7b"

# Context window size in tokens (default: 32768)
# context_size = 32768

# Ollama API base URL (default: "http://localhost:11434")
# ollama_url = "http://localhost:11434"

# Backend: "ollama" (default) or "llama-cpp"
# backend = "ollama"

# Path to llama-server binary (for llama-cpp backend)
# llama_server_path = "/path/to/llama-server"

# URL of a remote llama-server (e.g. "http://192.168.1.50:8080")
# When set, connects directly without spawning a local server.
# llama_server_url = "http://192.168.1.50:8080"

# Path to a GGUF model file (for llama-cpp backend)
# model_path = "/path/to/model.gguf"

# HuggingFace repo to download the model from (e.g. "google/gemma-3-27b-it-GGUF")
# Used by llama-server's -hf flag. Takes precedence over Ollama blob resolution.
# hf_repo = "google/gemma-3-27b-it-GGUF"

# Extra arguments passed to llama-server (e.g. ["-ngl", "99"])
# llama_server_args = ["-ngl", "99"]

# Timeout for bash tool commands in seconds (default: 120)
# bash_timeout = 120

# Maximum number of agent-loop turns a sub-agent is allowed (default: 15)
# subagent_max_turns = 15

# Show estimated cost on the status line (default: false)
# show_cost_estimate = false

# Cost rates as [input, output] dollars per million tokens (default: [5.0, 25.0])
# Set to [0.0, 0.0] for local Ollama models.
# cost_per_million_tokens = [5.0, 25.0]

# Auto-approve all tool calls (default: false)
# no_confirm = false

# Enable bypass mode — auto-approve all tool calls by default (default: false)
# bypass = false

# Enable verbose/debug output (default: false)
# verbose = false

# Sampling temperature (0.0 = deterministic, higher = more random)
# Recommended: 0.1-0.3 for tool use, 0.5-0.8 for creative/planning
# temperature = 0.2

# Top-p (nucleus) sampling. Only tokens with cumulative probability <= top_p are considered.
# top_p = 0.9

# Top-k sampling. Only the top k tokens are considered.
# top_k = 40

# Enable dynamic tool scoping: only present relevant tools per turn (default: false)
# Reduces confusion in small models.
# tool_scoping = false

# Enable periodic task re-injection to fight coherence decay (default: false)
# task_reinjection = false

# How often to re-inject the task objective, in agent turns (default: 3)
# reinjection_interval = 3

# Context trim threshold as percentage of context_size (default: 80)
# When prompt tokens exceed this, auto-trimming kicks in.
# trim_threshold = 80

# Context trim target as percentage of context_size (default: 60)
# Trimming removes messages until usage drops to this level.
# trim_target = 60

# Plugin feature flags and configuration
# Boolean values enable/disable tools by name:
#   bash = false     # disables the built-in bash tool
# Table values provide plugin-specific configuration:
# [plugins.my-plugin]
# key = "value"

# MCP (Model Context Protocol) servers
# Each entry spawns a server process at startup and registers its tools.
# [mcp.filesystem]
# command = "npx"
# args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

# Hook feature flags and per-hook configuration
# Boolean values enable/disable hooks by name:
#   my-hook = false   # disables the hook
# [hooks.my-hook]
# key = "value"
"#
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
        self.trim_threshold.unwrap_or(DEFAULT_TRIM_THRESHOLD_PCT).min(100)
    }

    pub fn effective_trim_target(&self) -> u8 {
        self.trim_target.unwrap_or(DEFAULT_TRIM_TARGET_PCT).min(100)
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── merge ───────────────────────────────────────────────────────

    #[test]
    fn merge_hi_wins_over_lo() {
        let hi = Config {
            model: Some("hi-model".into()),
            context_size: Some(16384),
            bash_timeout: Some(60),
            ..Default::default()
        };
        let lo = Config {
            model: Some("lo-model".into()),
            context_size: Some(8192),
            bash_timeout: Some(30),
            ..Default::default()
        };
        let merged = hi.merge(&lo);
        assert_eq!(merged.model.as_deref(), Some("hi-model"));
        assert_eq!(merged.context_size, Some(16384));
        assert_eq!(merged.bash_timeout, Some(60));
    }

    #[test]
    fn merge_lo_fills_gaps() {
        let hi = Config {
            model: Some("hi-model".into()),
            ..Default::default()
        };
        let lo = Config {
            context_size: Some(8192),
            bash_timeout: Some(30),
            ..Default::default()
        };
        let merged = hi.merge(&lo);
        assert_eq!(merged.model.as_deref(), Some("hi-model"));
        assert_eq!(merged.context_size, Some(8192));
        assert_eq!(merged.bash_timeout, Some(30));
    }

    #[test]
    fn merge_both_none() {
        let hi = Config::default();
        let lo = Config::default();
        let merged = hi.merge(&lo);
        assert!(merged.model.is_none());
        assert!(merged.context_size.is_none());
    }

    #[test]
    fn merge_recent_hf_models_from_lo_only() {
        let hi = Config {
            recent_hf_models: Some(vec!["hi-repo".into()]),
            ..Default::default()
        };
        let lo = Config {
            recent_hf_models: Some(vec!["lo-repo".into()]),
            ..Default::default()
        };
        let merged = hi.merge(&lo);
        // User-scope only: always from lo
        assert_eq!(merged.recent_hf_models, Some(vec!["lo-repo".to_string()]));
    }

    #[test]
    fn merge_plugins_combined() {
        let mut hi_plugins = HashMap::new();
        hi_plugins.insert("bash".to_string(), toml::Value::Boolean(false));
        let mut lo_plugins = HashMap::new();
        lo_plugins.insert("read".to_string(), toml::Value::Boolean(false));
        lo_plugins.insert("bash".to_string(), toml::Value::Boolean(true)); // should be overridden

        let hi = Config { plugins: Some(hi_plugins), ..Default::default() };
        let lo = Config { plugins: Some(lo_plugins), ..Default::default() };

        let merged = hi.merge(&lo);
        let plugins = merged.plugins.unwrap();
        // hi's bash=false wins
        assert_eq!(plugins.get("bash"), Some(&toml::Value::Boolean(false)));
        // lo's read=false fills the gap
        assert_eq!(plugins.get("read"), Some(&toml::Value::Boolean(false)));
    }

    // ── is_tool_enabled / is_hook_enabled ───────────────────────────

    #[test]
    fn tool_enabled_by_default() {
        let cfg = Config::default();
        assert!(cfg.is_tool_enabled("bash"));
        assert!(cfg.is_tool_enabled("anything"));
    }

    #[test]
    fn tool_disabled_explicit_false() {
        let mut plugins = HashMap::new();
        plugins.insert("bash".to_string(), toml::Value::Boolean(false));
        let cfg = Config { plugins: Some(plugins), ..Default::default() };
        assert!(!cfg.is_tool_enabled("bash"));
        assert!(cfg.is_tool_enabled("read")); // other tools unaffected
    }

    #[test]
    fn tool_enabled_explicit_true() {
        let mut plugins = HashMap::new();
        plugins.insert("bash".to_string(), toml::Value::Boolean(true));
        let cfg = Config { plugins: Some(plugins), ..Default::default() };
        assert!(cfg.is_tool_enabled("bash"));
    }

    #[test]
    fn tool_enabled_table_value() {
        // A table value (plugin config) means enabled
        let mut plugins = HashMap::new();
        let mut table = toml::map::Map::new();
        table.insert("key".to_string(), toml::Value::String("value".into()));
        plugins.insert("my-plugin".to_string(), toml::Value::Table(table));
        let cfg = Config { plugins: Some(plugins), ..Default::default() };
        assert!(cfg.is_tool_enabled("my-plugin"));
    }

    #[test]
    fn hook_enabled_by_default() {
        let cfg = Config::default();
        assert!(cfg.is_hook_enabled("any-hook"));
    }

    #[test]
    fn hook_disabled_explicit_false() {
        let mut hooks = HashMap::new();
        hooks.insert("deny-rm".to_string(), toml::Value::Boolean(false));
        let cfg = Config { hooks: Some(hooks), ..Default::default() };
        assert!(!cfg.is_hook_enabled("deny-rm"));
    }

    // ── plugin_config / hook_config ─────────────────────────────────

    #[test]
    fn plugin_config_returns_table() {
        let mut plugins = HashMap::new();
        let mut table = toml::map::Map::new();
        table.insert("api_key".to_string(), toml::Value::String("secret".into()));
        plugins.insert("my-plugin".to_string(), toml::Value::Table(table));
        let cfg = Config { plugins: Some(plugins), ..Default::default() };

        let pc = cfg.plugin_config("my-plugin").unwrap();
        assert_eq!(pc.get("api_key").unwrap().as_str(), Some("secret"));
    }

    #[test]
    fn plugin_config_returns_none_for_bool() {
        let mut plugins = HashMap::new();
        plugins.insert("bash".to_string(), toml::Value::Boolean(true));
        let cfg = Config { plugins: Some(plugins), ..Default::default() };
        assert!(cfg.plugin_config("bash").is_none());
    }

    #[test]
    fn plugin_config_returns_none_for_missing() {
        let cfg = Config::default();
        assert!(cfg.plugin_config("nonexistent").is_none());
    }

    // ── effective_* helpers ──────────────────────────────────────────

    #[test]
    fn effective_defaults() {
        let cfg = Config::default();
        assert_eq!(cfg.effective_trim_threshold(), DEFAULT_TRIM_THRESHOLD_PCT);
        assert_eq!(cfg.effective_trim_target(), DEFAULT_TRIM_TARGET_PCT);
        assert_eq!(cfg.effective_subagent_max_turns(), DEFAULT_SUBAGENT_MAX_TURNS);
        assert_eq!(cfg.effective_reinjection_interval(), DEFAULT_REINJECTION_INTERVAL);
        assert_eq!(cfg.bash_timeout_duration(), std::time::Duration::from_secs(DEFAULT_BASH_TIMEOUT_SECS));
    }

    #[test]
    fn effective_custom_values() {
        let cfg = Config {
            trim_threshold: Some(90),
            trim_target: Some(50),
            subagent_max_turns: Some(5),
            reinjection_interval: Some(10),
            bash_timeout: Some(300),
            ..Default::default()
        };
        assert_eq!(cfg.effective_trim_threshold(), 90);
        assert_eq!(cfg.effective_trim_target(), 50);
        assert_eq!(cfg.effective_subagent_max_turns(), 5);
        assert_eq!(cfg.effective_reinjection_interval(), 10);
        assert_eq!(cfg.bash_timeout_duration(), std::time::Duration::from_secs(300));
    }

    // ── add_recent_hf_model ─────────────────────────────────────────

    #[test]
    fn add_recent_hf_model_fresh() {
        let mut cfg = Config::default();
        cfg.add_recent_hf_model("org/model-a");
        assert_eq!(cfg.recent_hf_models, Some(vec!["org/model-a".to_string()]));
    }

    #[test]
    fn add_recent_hf_model_dedupes_and_promotes() {
        let mut cfg = Config {
            recent_hf_models: Some(vec!["a".into(), "b".into(), "c".into()]),
            ..Default::default()
        };
        cfg.add_recent_hf_model("b");
        let list = cfg.recent_hf_models.unwrap();
        assert_eq!(list[0], "b"); // promoted to front
        assert_eq!(list.len(), 3); // no duplicates
    }

    #[test]
    fn add_recent_hf_model_truncates_at_10() {
        let mut cfg = Config {
            recent_hf_models: Some((0..10).map(|i| format!("model-{}", i)).collect()),
            ..Default::default()
        };
        cfg.add_recent_hf_model("new-model");
        let list = cfg.recent_hf_models.unwrap();
        assert_eq!(list.len(), 10);
        assert_eq!(list[0], "new-model");
    }

    // ── TOML parsing ────────────────────────────────────────────────

    #[test]
    fn parse_minimal_toml() {
        let toml_str = r#"model = "qwen2.5-coder:7b""#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.model.as_deref(), Some("qwen2.5-coder:7b"));
        assert!(cfg.context_size.is_none());
    }

    #[test]
    fn parse_full_toml() {
        let toml_str = r#"
model = "llama3"
context_size = 16384
backend = "ollama"
bash_timeout = 60
no_confirm = true
temperature = 0.2
top_p = 0.9
top_k = 40
tool_scoping = true
task_reinjection = true
reinjection_interval = 5
trim_threshold = 90
trim_target = 50

[plugins]
bash = false

[mcp.test]
command = "echo"
args = ["hello"]
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.model.as_deref(), Some("llama3"));
        assert_eq!(cfg.context_size, Some(16384));
        assert_eq!(cfg.bash_timeout, Some(60));
        assert_eq!(cfg.no_confirm, Some(true));
        assert_eq!(cfg.temperature, Some(0.2));
        assert_eq!(cfg.top_p, Some(0.9));
        assert_eq!(cfg.top_k, Some(40));
        assert_eq!(cfg.tool_scoping, Some(true));
        assert_eq!(cfg.task_reinjection, Some(true));
        assert_eq!(cfg.trim_threshold, Some(90));
        assert!(!cfg.is_tool_enabled("bash"));
    }

    #[test]
    fn parse_empty_toml() {
        let cfg: Config = toml::from_str("").unwrap();
        assert!(cfg.model.is_none());
        assert!(cfg.plugins.is_none());
    }

    // ── find_project_config ─────────────────────────────────────────

    #[test]
    fn find_project_config_found() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join(".ollama-code.toml");
        std::fs::write(&config_path, "model = \"test\"").unwrap();

        let child = dir.path().join("subdir");
        std::fs::create_dir(&child).unwrap();

        // Search from child should find it in parent
        let found = find_project_config(&child);
        assert_eq!(found, Some(config_path));
    }

    #[test]
    fn find_project_config_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let found = find_project_config(dir.path());
        assert!(found.is_none());
    }
}
