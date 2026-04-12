use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const DEFAULT_CONTEXT_SIZE: u64 = 32768;

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
}

impl Config {
    pub fn path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("ollama-code")
            .join("config.toml")
    }

    pub fn load() -> Result<Self> {
        let path = Self::path();
        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            Ok(toml::from_str(&content)?)
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, toml::to_string_pretty(self)?)?;
        Ok(())
    }
}
