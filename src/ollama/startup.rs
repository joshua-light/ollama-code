//! Startup helpers specific to the Ollama backend: health-check the server,
//! auto-spawn it on localhost if needed, and interactively pick a model.

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::Result;

use super::OllamaBackend;

/// Check if Ollama is reachable; if not, try to start it automatically.
///
/// Only attempts auto-start for localhost URLs — remote servers require the
/// user to start Ollama themselves.
pub async fn ensure_ollama_running(url: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;

    // Quick check — is Ollama already running?
    if client.get(format!("{}/api/tags", url)).send().await.is_ok() {
        return Ok(());
    }

    // Only attempt auto-start for localhost URLs.
    let lower = url.to_lowercase();
    let is_local = lower.contains("://localhost")
        || lower.contains("://127.0.0.1")
        || lower.contains("://[::1]");
    if !is_local {
        anyhow::bail!(
            "Cannot connect to Ollama at {url}.\n\n\
             Make sure the Ollama server is running at that address."
        );
    }

    eprintln!("Ollama is not running. Starting it...");

    let child = std::process::Command::new("ollama")
        .arg("serve")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();

    match child {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => anyhow::bail!(
            "Ollama is not installed.\n\n\
             Install it from: https://ollama.com/download"
        ),
        Err(e) => anyhow::bail!(
            "Failed to start Ollama: {e}\n\n\
             Try starting it manually:\n  ollama serve"
        ),
        Ok(_) => {
            // Poll until ready (up to 10 seconds).
            let deadline = Instant::now() + Duration::from_secs(10);
            while Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(250)).await;
                if client.get(format!("{}/api/tags", url)).send().await.is_ok() {
                    eprintln!("Ollama started.");
                    return Ok(());
                }
            }
            anyhow::bail!(
                "Started Ollama but it did not become ready in time.\n\n\
                 Possible causes:\n  \
                 • Insufficient memory or VRAM\n  \
                 • Port already in use\n  \
                 • GPU driver issues\n\n\
                 Check the output of: ollama serve"
            )
        }
    }
}

/// Prompt the user to pick one of the installed Ollama models.
/// If only one is installed, return it without prompting.
pub async fn select_model(ollama: &OllamaBackend) -> Result<String> {
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
