mod agent;
mod commands;
mod config;
mod message;
mod ollama;
mod session;
mod tools;
mod tui;

use anyhow::Result;
use clap::Parser;
use std::io::Write;
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentEvent};
use crate::config::Config;
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::load()?;

    let ollama = OllamaClient::new(None);

    let model = if let Some(m) = cli.model {
        m
    } else if let Some(m) = config.model.clone() {
        m
    } else {
        let m = select_model(&ollama).await?;
        let mut config = config;
        config.model = Some(m.clone());
        if let Err(e) = config.save() {
            eprintln!("Warning: could not save config: {}", e);
        }
        m
    };

    let context_size = ollama.context_length(&model).await.unwrap_or(0);

    let agent = Agent::new(ollama.clone(), model, context_size);
    let session = Session::new()?;

    if let Some(prompt) = cli.prompt {
        run_pipe(agent, &prompt, session).await
    } else {
        tui::run(agent, context_size, session, ollama).await
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

    eprintln!("Session: {}", session.path().display());

    let prompt = prompt.to_string();
    let handle = tokio::spawn(async move { agent.run(&prompt, &tx).await });

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
                eprintln!("\n⚙ {} $ {}", name, args);
            }
            AgentEvent::ToolResult { output, .. } => {
                let lines: Vec<&str> = output.lines().collect();
                if lines.len() > 10 {
                    for line in &lines[..10] {
                        eprintln!("  ┃ {}", line);
                    }
                    eprintln!("  ┃ ... ({} more lines)", lines.len() - 10);
                } else {
                    for line in &lines {
                        eprintln!("  ┃ {}", line);
                    }
                }
            }
            AgentEvent::Done { .. } => {
                println!();
                break;
            }
            AgentEvent::Error(e) => {
                eprintln!("\nerror: {}", e);
                break;
            }
            AgentEvent::MessageLogged(_) | AgentEvent::Debug(_) => {}
        }
    }

    handle.await??;
    Ok(())
}
