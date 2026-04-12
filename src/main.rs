mod agent;
mod backend;
mod commands;
mod config;
mod format;
mod llama_server;
mod message;
mod ollama;
mod session;
mod skills;
mod tools;
mod tui;

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentEvent};
use crate::config::{Config, DEFAULT_CONTEXT_SIZE, DEFAULT_SUBAGENT_MAX_TURNS};
use crate::llama_server::{LlamaCppBackend, LlamaServer, ModelSource};
use crate::ollama::OllamaBackend;
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

    /// URL of a remote llama-server (e.g. "http://192.168.1.50:8080")
    #[arg(long)]
    llama_server_url: Option<String>,

    /// Path to GGUF model file (for llama-cpp backend)
    #[arg(long)]
    model_path: Option<String>,

    /// HuggingFace repo to download model from (for llama-cpp backend, e.g. "google/gemma-3-27b-it-GGUF")
    #[arg(long)]
    hf_repo: Option<String>,

    /// Context window size (overrides config)
    #[arg(long)]
    context_size: Option<u64>,

    /// Auto-approve all tool calls (skip confirmation prompts)
    #[arg(long)]
    no_confirm: bool,

    /// Enable verbose/debug output
    #[arg(long)]
    verbose: bool,

    /// Resume a previous session (most recent if no ID given, or by ID/prefix)
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    resume: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut config = Config::load()?;

    // Apply CLI overrides (highest priority layer) — moves, not clones.
    if cli.model.is_some() { config.model = cli.model; }
    if cli.context_size.is_some() { config.context_size = cli.context_size; }
    if cli.backend.is_some() { config.backend = cli.backend; }
    if cli.llama_server_path.is_some() { config.llama_server_path = cli.llama_server_path; }
    if cli.llama_server_url.is_some() { config.llama_server_url = cli.llama_server_url; }
    if cli.model_path.is_some() { config.model_path = cli.model_path; }
    if cli.hf_repo.is_some() { config.hf_repo = cli.hf_repo; }
    if cli.no_confirm { config.no_confirm = Some(true); }
    if cli.verbose { config.verbose = Some(true); }

    if let Some(ref pp) = config.project_config_path {
        eprintln!("Using project config: {}", pp.display());
    }

    // Normalize llama_server_url
    if let Some(ref mut url) = config.llama_server_url {
        *url = url.trim_end_matches('/').to_string();
    }

    let context_size = config.context_size.unwrap_or(DEFAULT_CONTEXT_SIZE);

    let backend = config
        .backend
        .as_deref()
        .or(if config.llama_server_url.is_some() { Some("llama-cpp") } else { None })
        .unwrap_or("ollama");

    let resume = cli.resume;

    match (backend, config.llama_server_url.clone()) {
        ("llama-cpp", Some(url)) => run_remote_llama_cpp(cli.prompt, config, context_size, url, resume).await,
        ("llama-cpp", None) => run_llama_cpp(cli.prompt, config, context_size, resume).await,
        _ => run_ollama(cli.prompt, config, context_size, resume).await,
    }
}

/// Resolve a `--resume` argument into a (Session, Vec<Message>) pair.
/// An empty string means "resume the latest session".
fn resolve_resume(resume_arg: &str) -> Result<(Session, Vec<message::Message>)> {
    let session_id = if resume_arg.is_empty() {
        Session::latest()?
            .ok_or_else(|| anyhow::anyhow!("No previous sessions found"))?
    } else {
        Session::find_by_prefix(resume_arg)?
            .ok_or_else(|| anyhow::anyhow!("No session found matching '{}'", resume_arg))?
    };

    eprintln!("Resuming session: {}", session_id);
    let (session, messages) = Session::resume(&session_id)?;
    Ok((session, messages))
}

/// Create or resume a session based on the `--resume` flag.
fn resolve_session(resume: &Option<String>) -> Result<(Session, Option<Vec<message::Message>>)> {
    if let Some(ref resume_arg) = resume {
        let (session, msgs) = resolve_resume(resume_arg)?;
        Ok((session, Some(msgs)))
    } else {
        Ok((Session::new()?, None))
    }
}

async fn run_ollama(prompt: Option<String>, mut config: Config, context_size: u64, resume: Option<String>) -> Result<()> {
    let ollama = OllamaBackend::new(config.ollama_url.clone());

    let model = if let Some(m) = config.model.clone() {
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
    let subagent_max_turns = config.subagent_max_turns.unwrap_or(DEFAULT_SUBAGENT_MAX_TURNS);
    let backend = OllamaBackend::new(config.ollama_url.clone());
    let agent = Agent::new(Arc::new(backend), model, context_size, bash_timeout, subagent_max_turns);
    let verbose = config.verbose.unwrap_or(false);
    let (session, restored) = resolve_session(&resume)?;

    if let Some(prompt) = prompt {
        let mut agent = agent;
        if let Some(msgs) = restored {
            agent.restore_messages(msgs);
        }
        run_pipe(agent, &prompt, session, verbose).await
    } else {
        tui::run(agent, context_size, session, config, None, restored, None).await
    }
}

async fn run_llama_cpp(prompt: Option<String>, config: Config, context_size: u64, resume: Option<String>) -> Result<()> {
    let server_path = config
        .llama_server_path
        .as_deref()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "llama-cpp backend requires --llama-server-path or llama_server_path in config"
            )
        })?;

    let server_binary = PathBuf::from(server_path);
    if !server_binary.exists() {
        anyhow::bail!("llama-server binary not found at: {}", server_binary.display());
    }

    let model_name = config.model.as_deref().unwrap_or("default");

    // Determine model source: model_path > hf_repo > Ollama blob resolution
    let model_source =
        if let Some(p) = config.model_path.as_deref() {
            let path = PathBuf::from(p);
            if !path.exists() {
                anyhow::bail!("Model file not found: {}", path.display());
            }
            ModelSource::File(path)
        } else if let Some(hf) = config.hf_repo.as_deref() {
            ModelSource::HuggingFace(hf.to_string())
        } else if model_name.contains('/') {
            ModelSource::HuggingFace(model_name.to_string())
        } else {
            eprintln!("Resolving model '{}' from Ollama storage...", model_name);
            ModelSource::File(llama_server::find_ollama_model_path(model_name).await?)
        };

    let extra_args = config.llama_server_args.clone().unwrap_or_default();

    // Try to reuse an existing llama-server for the same model.
    // If reusing, we're ready immediately. If spawning fresh, defer readiness
    // wait to the TUI so a progress bar is shown.
    let (llama_server, initial_start) = if let Some(server) = LlamaServer::connect_existing(&model_source).await {
        eprintln!(
            "Connected to existing llama-server on port {}",
            server.port
        );
        (Some(server), None)
    } else {
        let port = llama_server::find_free_port()?;
        eprintln!(
            "Starting llama-server on port {} with {}...",
            port,
            match &model_source {
                ModelSource::File(p) => format!("model {}", p.display()),
                ModelSource::HuggingFace(repo) => format!("HF repo {}", repo),
            }
        );
        let server =
            LlamaServer::spawn(&server_binary, &model_source, port, context_size, &extra_args)
                .await?;
        (None, Some((server, model_source)))
    };

    let base_url = if let Some(ref s) = llama_server {
        s.base_url()
    } else if let Some((ref s, _)) = initial_start {
        s.base_url()
    } else {
        unreachable!()
    };
    let backend = LlamaCppBackend::new(base_url);

    let bash_timeout = std::time::Duration::from_secs(config.bash_timeout.unwrap_or(120));
    let subagent_max_turns = config.subagent_max_turns.unwrap_or(DEFAULT_SUBAGENT_MAX_TURNS);
    let agent = Agent::new(Arc::new(backend), model_name.to_string(), context_size, bash_timeout, subagent_max_turns);
    let verbose = config.verbose.unwrap_or(false);
    let (session, restored) = resolve_session(&resume)?;

    if let Some(prompt) = prompt {
        // Pipe mode: wait for server synchronously (no TUI to show progress)
        let mut agent = agent;
        if let Some(msgs) = &restored {
            agent.restore_messages(msgs.clone());
        }
        if let Some((mut server, model_source)) = initial_start {
            server.wait_until_ready(&model_source).await?;
            eprintln!("llama-server ready (log: {})", server.log_path.display());
            let result = run_pipe(agent, &prompt, session, verbose).await;
            server.stop().await;
            result
        } else {
            let mut server = llama_server.unwrap();
            let result = run_pipe(agent, &prompt, session, verbose).await;
            server.stop().await;
            result
        }
    } else {
        // TUI mode: defer server readiness to event loop (shows progress bar)
        let initial = initial_start.map(|(server, ms)| tui::InitialServerStart {
            server,
            model_source: ms,
        });
        tui::run(agent, context_size, session, config, llama_server, restored, initial).await
    }
}

async fn run_remote_llama_cpp(prompt: Option<String>, mut config: Config, context_size: u64, url: String, resume: Option<String>) -> Result<()> {
    // Health check the remote server
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let health_url = format!("{}/health", url);
    match client.get(&health_url).send().await {
        Ok(resp) if resp.status().is_success() => {}
        Ok(resp) => anyhow::bail!(
            "Remote llama-server at {} returned status {}. Is it healthy?",
            url, resp.status()
        ),
        Err(e) => anyhow::bail!(
            "Cannot reach llama-server at {} — is it running?\n  Error: {}",
            url, e
        ),
    }

    let model_name = config.model.as_deref().unwrap_or("default").to_string();

    eprintln!("Connected to remote llama-server at {}", url);

    // Ensure config reflects llama-cpp backend so TUI guards work correctly
    config.backend = Some("llama-cpp".to_string());
    config.llama_server_url = Some(url.clone());

    let backend = LlamaCppBackend::new(url);
    let bash_timeout = std::time::Duration::from_secs(config.bash_timeout.unwrap_or(120));
    let subagent_max_turns = config.subagent_max_turns.unwrap_or(DEFAULT_SUBAGENT_MAX_TURNS);
    let agent = Agent::new(Arc::new(backend), model_name, context_size, bash_timeout, subagent_max_turns);
    let verbose = config.verbose.unwrap_or(false);
    let (session, restored) = resolve_session(&resume)?;

    if let Some(prompt) = prompt {
        let mut agent = agent;
        if let Some(msgs) = restored {
            agent.restore_messages(msgs);
        }
        run_pipe(agent, &prompt, session, verbose).await
    } else {
        tui::run(agent, context_size, session, config, None, restored, None).await
    }
}

async fn select_model(ollama: &OllamaBackend) -> Result<String> {
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

async fn run_pipe(mut agent: Agent, prompt: &str, mut session: Session, verbose: bool) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let (confirm_tx, mut confirm_rx) = mpsc::unbounded_channel::<bool>();

    eprintln!("Session: {}", session.path().display());

    let prompt = prompt.to_string();
    let cancel = Arc::new(AtomicBool::new(false));
    let handle = tokio::spawn(async move { agent.run(&prompt, &tx, &mut confirm_rx, cancel).await });

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
            AgentEvent::SubagentStart { ref task } => {
                eprintln!("\n ◈ Subagent: {}", format::truncate_args(task, 77));
            }
            AgentEvent::SubagentToolCall { name, args } => {
                eprintln!(
                    "   ↳ {}({})",
                    format::capitalize_first(&name),
                    format::truncate_args(&args, 60),
                );
            }
            AgentEvent::SubagentToolResult { .. } => {}
            AgentEvent::SubagentEnd { .. } => {}
            AgentEvent::Cancelled => {
                eprintln!("\n(cancelled)");
                break;
            }
            AgentEvent::Debug(ref msg) if verbose => {
                eprintln!("[debug] {}", msg);
            }
            AgentEvent::ContextUpdate { .. } | AgentEvent::ContentReplaced(_) | AgentEvent::MessageLogged(_) | AgentEvent::Debug(_) | AgentEvent::SystemPromptInfo { .. } => {}
        }
    }

    handle.await??;
    Ok(())
}
