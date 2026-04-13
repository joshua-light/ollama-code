use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tokio::sync::mpsc;

use ollama_code::agent::{Agent, AgentEvent};
use ollama_code::config::{Config, DEFAULT_CONTEXT_SIZE};
use ollama_code::format;
use ollama_code::llama_server::{self, LlamaCppBackend, LlamaServer, ModelSource};
use ollama_code::message;
use ollama_code::ollama::{OllamaBackend, SamplingParams};
use ollama_code::session::Session;
use ollama_code::tui;

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

/// Shared dispatch: resolve the session, then run pipe or TUI.
///
/// `llama_server` and `initial_server_start` are only set when using a local
/// llama-cpp backend; they are `None` for Ollama and remote llama-cpp.
async fn run_with_backend(
    prompt: Option<String>,
    config: Config,
    context_size: u64,
    agent: Agent,
    llama_server: Option<LlamaServer>,
    initial_server_start: Option<tui::InitialServerStart>,
    resume: Option<String>,
) -> Result<()> {
    let verbose = config.verbose.unwrap_or(false);
    let (session, restored) = resolve_session(&resume)?;

    if let Some(prompt) = prompt {
        // Pipe mode: wait for server synchronously (no TUI to show progress)
        let mut agent = agent;
        if let Some(msgs) = restored {
            agent.restore_messages(msgs);
        }
        // Unpack the pending server start (if any) so we can wait for readiness
        // and clean up afterwards.
        let mut owned_server = if let Some(mut is) = initial_server_start {
            is.server.wait_until_ready(&is.model_source).await?;
            eprintln!("llama-server ready (log: {})", is.server.log_path.display());
            Some(is.server)
        } else {
            llama_server
        };
        let result = run_pipe(agent, &prompt, session, verbose).await;
        if let Some(ref mut s) = owned_server {
            s.stop().await;
        }
        result
    } else {
        // TUI mode: defer server readiness to event loop (shows progress bar)
        tui::run(agent, context_size, session, config, llama_server, restored, initial_server_start).await
    }
}

fn sampling_from_config(config: &Config) -> SamplingParams {
    SamplingParams {
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
    }
}

async fn run_ollama(prompt: Option<String>, mut config: Config, context_size: u64, resume: Option<String>) -> Result<()> {
    let ollama = OllamaBackend::with_sampling(config.ollama_url.clone(), sampling_from_config(&config));

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

    let agent = Agent::with_config(Arc::new(ollama), model, context_size, config.bash_timeout_duration(), config.effective_subagent_max_turns(), &config);

    run_with_backend(prompt, config, context_size, agent, None, None, resume).await
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
            ModelSource::File(llama_server::find_ollama_model_path(model_name, config.ollama_url.as_deref()).await?)
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
    let backend = LlamaCppBackend::with_sampling(base_url, sampling_from_config(&config));

    let agent = Agent::with_config(Arc::new(backend), model_name.to_string(), context_size, config.bash_timeout_duration(), config.effective_subagent_max_turns(), &config);

    let initial = initial_start.map(|(server, ms)| tui::InitialServerStart {
        server,
        model_source: ms,
    });
    run_with_backend(prompt, config, context_size, agent, llama_server, initial, resume).await
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

    let backend = LlamaCppBackend::with_sampling(url, sampling_from_config(&config));
    let agent = Agent::with_config(Arc::new(backend), model_name, context_size, config.bash_timeout_duration(), config.effective_subagent_max_turns(), &config);

    run_with_backend(prompt, config, context_size, agent, None, None, resume).await
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

    // Buffer streamed tokens so ContentReplaced can discard/replace them.
    // This handles models that emit tool calls as text (e.g. <function=...>
    // format) — without buffering, the raw text leaks to stdout before
    // extraction can clean it up.
    let mut token_buf = String::new();
    let mut agent_error: Option<String> = None;

    while let Some(event) = rx.recv().await {
        session.log_agent_event(&event);
        if let AgentEvent::MessageLogged(ref msg) = event {
            session.log_message(msg);
        }
        match event {
            AgentEvent::Token(t) => {
                token_buf.push_str(&t);
            }
            AgentEvent::ContentReplaced(new_content) => {
                // The agent extracted tool calls from the buffered text.
                // Replace the buffer with the cleaned content.
                token_buf = new_content;
            }
            AgentEvent::ToolCall { name, args } => {
                // Flush any buffered content before showing tool info.
                if !token_buf.is_empty() {
                    print!("{}", token_buf);
                    token_buf.clear();
                }
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
                session.record_trim(removed_messages);
                eprintln!("(context trimmed: {} messages, ~{} tokens freed)", removed_messages, estimated_tokens_freed);
            }
            AgentEvent::Done { .. } => {
                // Flush remaining buffer and finish.
                if !token_buf.is_empty() {
                    print!("{}", token_buf);
                    token_buf.clear();
                }
                println!();
                break;
            }
            AgentEvent::Error(e) => {
                eprintln!("\nerror: {}", e);
                agent_error = Some(e);
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
            AgentEvent::ContextUpdate { .. } | AgentEvent::MessageLogged(_) | AgentEvent::Debug(_) | AgentEvent::SystemPromptInfo { .. } => {}
        }
    }

    handle.await??;

    if let Some(e) = agent_error {
        anyhow::bail!("{}", e);
    }

    Ok(())
}
