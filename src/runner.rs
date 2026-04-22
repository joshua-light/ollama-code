//! Backend selection and run orchestration: given a resolved config, set up
//! the correct backend (Ollama / local llama-cpp / remote llama-cpp), build
//! the `Agent`, resolve the session, and dispatch to TUI or pipe mode.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;

use crate::agent::Agent;
use crate::config::{Config, DEFAULT_CONTEXT_SIZE, DEFAULT_OLLAMA_URL};
use crate::llama_server::{self, LlamaCppBackend, LlamaServer, ModelSource};
use crate::message;
use crate::ollama::{startup, OllamaBackend, SamplingParams};
use crate::pipe::run_pipe;
use crate::session::Session;
use crate::tui;

/// Top-level entry point: pick the backend, set it up, and run.
pub async fn dispatch(
    prompt: Option<String>,
    resume: Option<String>,
    config: Config,
) -> Result<()> {
    let context_size = config.context_size.unwrap_or(DEFAULT_CONTEXT_SIZE);

    let backend = config
        .backend
        .as_deref()
        .or(if config.llama_server_url.is_some() {
            Some("llama-cpp")
        } else {
            None
        })
        .unwrap_or("ollama");

    match (backend, config.llama_server_url.clone()) {
        ("llama-cpp", Some(url)) => {
            run_remote_llama_cpp(prompt, config, context_size, url, resume).await
        }
        ("llama-cpp", None) => run_llama_cpp(prompt, config, context_size, resume).await,
        _ => run_ollama(prompt, config, context_size, resume).await,
    }
}

fn sampling_from_config(config: &Config) -> SamplingParams {
    SamplingParams {
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
    }
}

/// Resolve a `--resume` argument into a (Session, Vec<Message>) pair.
/// An empty string means "resume the latest session".
fn resolve_resume(resume_arg: &str) -> Result<(Session, Vec<message::Message>)> {
    let session_id = if resume_arg.is_empty() {
        Session::latest()?.ok_or_else(|| anyhow::anyhow!("No previous sessions found"))?
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

    // Attach the session directory so file edits/writes get first-write-wins
    // snapshots under <session>/checkpoints/.
    let mut agent = agent;
    agent.set_session_dir(Some(session.path()));

    if let Some(prompt) = prompt {
        // Pipe mode: wait for server synchronously (no TUI to show progress)
        if let Some(msgs) = restored {
            agent.restore_messages(msgs);
        }
        let mut owned_server = if let Some(mut is) = initial_server_start {
            is.server.wait_until_ready(&is.model_source).await?;
            eprintln!(
                "llama-server ready (log: {})",
                is.server.log_path.display()
            );
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
        tui::run(
            agent,
            context_size,
            session,
            config,
            llama_server,
            restored,
            initial_server_start,
        )
        .await
    }
}

async fn run_ollama(
    prompt: Option<String>,
    mut config: Config,
    context_size: u64,
    resume: Option<String>,
) -> Result<()> {
    let ollama_url = config.ollama_url.as_deref().unwrap_or(DEFAULT_OLLAMA_URL);
    startup::ensure_ollama_running(ollama_url).await?;

    let ollama =
        OllamaBackend::with_sampling(config.ollama_url.clone(), sampling_from_config(&config));

    let model = if let Some(m) = config.model.clone() {
        m
    } else {
        let m = startup::select_model(&ollama).await?;
        config.model = Some(m.clone());
        config.context_size = Some(context_size);
        if let Err(e) = config.save() {
            eprintln!("Warning: could not save config: {}", e);
        }
        m
    };

    let agent = Agent::with_config(
        Arc::new(ollama),
        model,
        context_size,
        config.bash_timeout_duration(),
        config.effective_subagent_max_turns(),
        &config,
    );

    run_with_backend(prompt, config, context_size, agent, None, None, resume).await
}

async fn run_llama_cpp(
    prompt: Option<String>,
    config: Config,
    context_size: u64,
    resume: Option<String>,
) -> Result<()> {
    let server_path = config.llama_server_path.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "llama-cpp backend requires --llama-server-path or llama_server_path in config"
        )
    })?;

    let server_binary = PathBuf::from(server_path);
    if !server_binary.exists() {
        anyhow::bail!(
            "llama-server binary not found at: {}",
            server_binary.display()
        );
    }

    let model_name = config.model.as_deref().unwrap_or("default");

    // Determine model source: model_path > hf_repo > Ollama blob resolution
    let model_source = if let Some(p) = config.model_path.as_deref() {
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
        ModelSource::File(
            llama_server::find_ollama_model_path(model_name, config.ollama_url.as_deref()).await?,
        )
    };

    let extra_args = config.llama_server_args.clone().unwrap_or_default();

    // Try to reuse an existing llama-server for the same model.
    // If reusing, we're ready immediately. If spawning fresh, defer readiness
    // wait to the TUI so a progress bar is shown.
    let (llama_server, initial_start) =
        if let Some(server) = LlamaServer::connect_existing(&model_source).await {
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

    let agent = Agent::with_config(
        Arc::new(backend),
        model_name.to_string(),
        context_size,
        config.bash_timeout_duration(),
        config.effective_subagent_max_turns(),
        &config,
    );

    let initial = initial_start.map(|(server, ms)| tui::InitialServerStart {
        server,
        model_source: ms,
    });
    run_with_backend(
        prompt,
        config,
        context_size,
        agent,
        llama_server,
        initial,
        resume,
    )
    .await
}

async fn run_remote_llama_cpp(
    prompt: Option<String>,
    mut config: Config,
    context_size: u64,
    url: String,
    resume: Option<String>,
) -> Result<()> {
    // Health check the remote server
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let health_url = format!("{}/health", url);
    match client.get(&health_url).send().await {
        Ok(resp) if resp.status().is_success() => {}
        Ok(resp) => anyhow::bail!(
            "Remote llama-server at {} returned status {}. Is it healthy?",
            url,
            resp.status()
        ),
        Err(e) => anyhow::bail!(
            "Cannot reach llama-server at {} — is it running?\n  Error: {}",
            url,
            e
        ),
    }

    let model_name = config.model.as_deref().unwrap_or("default").to_string();

    eprintln!("Connected to remote llama-server at {}", url);

    // Ensure config reflects llama-cpp backend so TUI guards work correctly
    config.backend = Some("llama-cpp".to_string());
    config.llama_server_url = Some(url.clone());

    let backend = LlamaCppBackend::with_sampling(url, sampling_from_config(&config));
    let agent = Agent::with_config(
        Arc::new(backend),
        model_name,
        context_size,
        config.bash_timeout_duration(),
        config.effective_subagent_max_turns(),
        &config,
    );

    run_with_backend(prompt, config, context_size, agent, None, None, resume).await
}
