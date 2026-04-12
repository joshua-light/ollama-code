mod agent;
mod commands;
mod config;
mod format;
mod llama_server;
mod message;
mod ollama;
mod session;
mod tools;
mod tui;

use anyhow::Result;
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentEvent};
use crate::config::{Config, DEFAULT_CONTEXT_SIZE};
use crate::llama_server::{LlamaServer, ModelSource};
use crate::ollama::OllamaClient;
use crate::session::Session;

#[derive(Parser)]
#[command(name = "ollama-code", about = "A CLI agent built on Ollama")]
struct Cli {
    /// Run in pipe mode: send a prompt and get a response
    #[arg(short, long)]
    prompt: Option<String>,

    /// Model to use (overrides config)
    #[arg(short, long)]
    model: Option<String>,

    /// Backend: "ollama" (default) or "llama-cpp"
    #[arg(long)]
    backend: Option<String>,

    /// Path to llama-server binary (for llama-cpp backend)
    #[arg(long)]
    llama_server_path: Option<String>,

    /// Path to GGUF model file (for llama-cpp backend)
    #[arg(long)]
    model_path: Option<String>,

    /// HuggingFace repo to download model from (for llama-cpp backend, e.g. "google/gemma-3-27b-it-GGUF")
    #[arg(long)]
    hf_repo: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::load()?;

    let context_size = config.context_size.unwrap_or(DEFAULT_CONTEXT_SIZE);

    let backend = cli
        .backend
        .as_deref()
        .or(config.backend.as_deref())
        .unwrap_or("ollama");

    match backend {
        "llama-cpp" => run_llama_cpp(cli, config, context_size).await,
        _ => run_ollama(cli, config, context_size).await,
    }
}

async fn run_ollama(cli: Cli, mut config: Config, context_size: u64) -> Result<()> {
    let ollama = OllamaClient::new(None);

    let model = if let Some(m) = cli.model {
        m
    } else if let Some(m) = config.model.clone() {
        m
    } else {
        let m = select_model(&ollama).await?;
        config.model = Some(m.clone());
        config.context_size = Some(context_size);
        if let Err(e) = config.save() {
            eprintln!("Warning: could not save config: {}", e);
        }
        m
    };

    let bash_timeout = std::time::Duration::from_secs(config.bash_timeout.unwrap_or(120));
    let agent = Agent::new(ollama.clone(), model, context_size, bash_timeout);
    let session = Session::new()?;

    if let Some(prompt) = cli.prompt {
        run_pipe(agent, &prompt, session).await
    } else {
        tui::run(agent, context_size, session, config, None).await
    }
}

async fn run_llama_cpp(cli: Cli, config: Config, context_size: u64) -> Result<()> {
    let server_path = cli
        .llama_server_path
        .as_deref()
        .or(config.llama_server_path.as_deref())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "llama-cpp backend requires --llama-server-path or llama_server_path in config"
            )
        })?;

    let server_binary = PathBuf::from(server_path);
    if !server_binary.exists() {
        anyhow::bail!("llama-server binary not found at: {}", server_binary.display());
    }

    // Determine model name (used as the alias for the API)
    let model_name = cli
        .model
        .as_deref()
        .or(config.model.as_deref())
        .unwrap_or("default");

    // Determine model source: --model-path > --hf-repo > config > Ollama blob resolution
    let model_source =
        if let Some(p) = cli.model_path.as_deref().or(config.model_path.as_deref()) {
            let path = PathBuf::from(p);
            if !path.exists() {
                anyhow::bail!("Model file not found: {}", path.display());
            }
            ModelSource::File(path)
        } else if let Some(hf) = cli.hf_repo.as_deref().or(config.hf_repo.as_deref()) {
            ModelSource::HuggingFace(hf.to_string())
        } else if model_name.contains('/') {
            // Looks like a HuggingFace repo (e.g. "unsloth/gemma-4-26B-A4B-it-GGUF")
            ModelSource::HuggingFace(model_name.to_string())
        } else {
            // Try to resolve from Ollama's model storage via the Ollama API
            eprintln!("Resolving model '{}' from Ollama storage...", model_name);
            ModelSource::File(llama_server::find_ollama_model_path(model_name).await?)
        };

    let extra_args = config.llama_server_args.clone().unwrap_or_default();

    let port = llama_server::find_free_port()?;
    eprintln!(
        "Starting llama-server on port {} with {}...",
        port,
        match &model_source {
            ModelSource::File(p) => format!("model {}", p.display()),
            ModelSource::HuggingFace(repo) => format!("HF repo {}", repo),
        }
    );

    let mut server =
        LlamaServer::start(&server_binary, &model_source, port, context_size, &extra_args)
            .await?;

    eprintln!("llama-server ready (log: {})", server.log_path.display());

    let ollama = OllamaClient::new(Some(server.base_url()));

    let bash_timeout = std::time::Duration::from_secs(config.bash_timeout.unwrap_or(120));
    let agent = Agent::new(ollama.clone(), model_name.to_string(), context_size, bash_timeout);
    let session = Session::new()?;

    if let Some(prompt) = cli.prompt {
        let result = run_pipe(agent, &prompt, session).await;
        server.stop().await;
        result
    } else {
        // TUI takes ownership of the server and handles its lifecycle
        tui::run(agent, context_size, session, config, Some(server)).await
    }
}

async fn select_model(ollama: &OllamaClient) -> Result<String> {
    let models = ollama.list_models().await?;

    if models.is_empty() {
        anyhow::bail!("No models found. Pull a model first:\n  ollama pull qwen2.5-coder:7b");
    }

    if models.len() == 1 {
        eprintln!("Using model: {}", models[0].name);
        return Ok(models[0].name.clone());
    }

    eprintln!("Available models:");
    for (i, model) in models.iter().enumerate() {
        eprintln!("  {}. {}", i + 1, model.name);
    }
    eprint!("Select model (1-{}): ", models.len());
    std::io::stderr().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let choice: usize = input
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid selection"))?;

    if choice < 1 || choice > models.len() {
        anyhow::bail!("Selection out of range");
    }

    Ok(models[choice - 1].name.clone())
}

async fn run_pipe(mut agent: Agent, prompt: &str, mut session: Session) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let (confirm_tx, mut confirm_rx) = mpsc::unbounded_channel::<bool>();

    eprintln!("Session: {}", session.path().display());

    let prompt = prompt.to_string();
    let handle = tokio::spawn(async move { agent.run(&prompt, &tx, &mut confirm_rx).await });

    while let Some(event) = rx.recv().await {
        session.log_agent_event(&event);
        if let AgentEvent::MessageLogged(ref msg) = event {
            session.log_message(msg);
        }
        match event {
            AgentEvent::Token(t) => {
                print!("{}", t);
                std::io::stdout().flush().ok();
            }
            AgentEvent::ToolCall { name, args } => {
                eprintln!(
                    "\n ● {}({})",
                    format::capitalize_first(&name),
                    format::truncate_args(&args, 77),
                );
            }
            AgentEvent::ToolConfirmRequest { .. } => {
                // Auto-approve in pipe mode
                let _ = confirm_tx.send(true);
            }
            AgentEvent::ToolResult { output, success, .. } => {
                if !success {
                    eprintln!("{}", format::format_tool_error(&output));
                } else {
                    for line in format::format_tool_output(&output) {
                        eprintln!("{}", line);
                    }
                }
            }
            AgentEvent::ContextTrimmed { removed_messages, estimated_tokens_freed } => {
                eprintln!("(context trimmed: {} messages, ~{} tokens freed)", removed_messages, estimated_tokens_freed);
            }
            AgentEvent::Done { .. } => {
                println!();
                break;
            }
            AgentEvent::Error(e) => {
                eprintln!("\nerror: {}", e);
                break;
            }
            AgentEvent::ContextUpdate { .. } | AgentEvent::ContentReplaced(_) | AgentEvent::MessageLogged(_) | AgentEvent::Debug(_) => {}
        }
    }

    handle.await??;
    Ok(())
}
