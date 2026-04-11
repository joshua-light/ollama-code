mod agent;
mod config;
mod message;
mod ollama;
mod tools;
mod tui;

use anyhow::Result;
use clap::Parser;
use std::io::Write;
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentEvent};
use crate::config::Config;
use crate::ollama::OllamaClient;

#[derive(Parser)]
#[command(name = "imp", about = "A CLI agent built on Ollama")]
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
        select_model(&ollama).await?
    };

    let agent = Agent::new(ollama, model);

    if let Some(prompt) = cli.prompt {
        run_pipe(agent, &prompt).await
    } else {
        tui::run(agent).await
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

async fn run_pipe(mut agent: Agent, prompt: &str) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel();

    let prompt = prompt.to_string();
    let handle = tokio::spawn(async move { agent.run(&prompt, &tx).await });

    while let Some(event) = rx.recv().await {
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
            AgentEvent::Done => {
                println!();
                break;
            }
            AgentEvent::Error(e) => {
                eprintln!("\nerror: {}", e);
                break;
            }
        }
    }

    handle.await??;
    Ok(())
}
