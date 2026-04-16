//! CLI argument definitions and the merge from CLI → Config.

use std::io::{IsTerminal, Read};

use anyhow::Result;
use clap::Parser;

use crate::config::Config;

#[derive(Parser, Debug)]
#[command(name = "ollama-code", about = "A CLI agent built on Ollama")]
pub struct Cli {
    /// Run in pipe mode: send a prompt and get a response
    #[arg(short, long)]
    pub prompt: Option<String>,

    /// Model to use (overrides config)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Backend: "ollama" (default) or "llama-cpp"
    #[arg(long)]
    pub backend: Option<String>,

    /// Path to llama-server binary (for llama-cpp backend)
    #[arg(long)]
    pub llama_server_path: Option<String>,

    /// URL of a remote llama-server (e.g. "http://192.168.1.50:8080")
    #[arg(long)]
    pub llama_server_url: Option<String>,

    /// Path to GGUF model file (for llama-cpp backend)
    #[arg(long)]
    pub model_path: Option<String>,

    /// HuggingFace repo to download model from (for llama-cpp backend, e.g. "google/gemma-3-27b-it-GGUF")
    #[arg(long)]
    pub hf_repo: Option<String>,

    /// Context window size (overrides config)
    #[arg(long)]
    pub context_size: Option<u64>,

    /// Auto-approve all tool calls (skip confirmation prompts)
    #[arg(long)]
    pub no_confirm: bool,

    /// Enable verbose/debug output
    #[arg(long)]
    pub verbose: bool,

    /// Resume a previous session (most recent if no ID given, or by ID/prefix)
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    pub resume: Option<String>,
}

/// Parse CLI args, read a piped prompt from stdin if `-p` was not given,
/// and return (prompt, resume, merged config).
pub fn resolve() -> Result<(Option<String>, Option<String>, Config)> {
    let mut cli = Cli::parse();

    if let Some(ref p) = cli.prompt {
        if p.trim().is_empty() {
            anyhow::bail!("empty prompt: -p requires a non-empty string");
        }
    } else if !std::io::stdin().is_terminal() {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        if buf.trim().is_empty() {
            anyhow::bail!("no prompt: pass -p or pipe input on stdin");
        }
        cli.prompt = Some(buf);
    }

    let mut config = Config::load()?;

    // Ensure a default config file exists so the user can discover and edit it.
    if let Err(e) = Config::ensure_default_config() {
        eprintln!("Warning: could not create default config: {}", e);
    }

    // Apply CLI overrides (highest priority layer) — moves, not clones.
    if cli.model.is_some() {
        config.model = cli.model;
    }
    if cli.context_size.is_some() {
        config.context_size = cli.context_size;
    }
    if cli.backend.is_some() {
        config.backend = cli.backend;
    }
    if cli.llama_server_path.is_some() {
        config.llama_server_path = cli.llama_server_path;
    }
    if cli.llama_server_url.is_some() {
        config.llama_server_url = cli.llama_server_url;
    }
    if cli.model_path.is_some() {
        config.model_path = cli.model_path;
    }
    if cli.hf_repo.is_some() {
        config.hf_repo = cli.hf_repo;
    }
    if cli.no_confirm {
        config.no_confirm = Some(true);
    }
    if cli.verbose {
        config.verbose = Some(true);
    }

    if let Some(ref pp) = config.project_config_path {
        eprintln!("Using project config: {}", pp.display());
    }

    // Normalize llama_server_url
    if let Some(ref mut url) = config.llama_server_url {
        *url = url.trim_end_matches('/').to_string();
    }

    Ok((cli.prompt, cli.resume, config))
}
