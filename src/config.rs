use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const DEFAULT_CONTEXT_SIZE: u64 = 32768;
pub const DEFAULT_SUBAGENT_MAX_TURNS: u16 = 15;
pub const DEFAULT_BASH_TIMEOUT_SECS: u64 = 120;

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

    /// Add a HuggingFace repo to the recent list (most recent first, max 10).
    pub fn add_recent_hf_model(&mut self, repo: &str) {
        let mut list = self.recent_hf_models.take().unwrap_or_default();
        list.retain(|m| m != repo);
        list.insert(0, repo.to_string());
        list.truncate(10);
        self.recent_hf_models = Some(list);
    }
}
