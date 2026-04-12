use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
/// How to specify the model for llama-server.
pub enum ModelSource {
    /// Local GGUF file path.
    File(PathBuf),
    /// HuggingFace repo identifier (e.g. "google/gemma-3-27b-it-GGUF").
    HuggingFace(String),
}

/// Manages a llama-server child process.
pub struct LlamaServer {
    child: Child,
    port: u16,
    pub log_path: PathBuf,
}

impl LlamaServer {
    /// Spawn llama-server with the given model source and port.
    /// `extra_args` are passed directly to the process (e.g. ["-ngl", "99"]).
    /// Server stderr is written only to the log file, never to the terminal.
    pub async fn start(
        binary: &Path,
        model: &ModelSource,
        port: u16,
        context_size: u64,
        extra_args: &[String],
    ) -> Result<Self> {
        let mut cmd = Command::new(binary);

        match model {
            ModelSource::File(path) => {
                cmd.arg("-m").arg(path);
            }
            ModelSource::HuggingFace(repo) => {
                cmd.arg("-hf").arg(repo);
            }
        }

        cmd.arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("-c")
            .arg(context_size.to_string())
            .args(extra_args)
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        let child = cmd
            .spawn()
            .with_context(|| format!("Failed to start llama-server at {}", binary.display()))?;

        // Log file lives next to the session logs.
        let log_dir = crate::config::data_dir();
        std::fs::create_dir_all(&log_dir)?;
        let log_path = log_dir.join("llama-server.log");

        let mut server = Self {
            child,
            port,
            log_path,
        };
        server.wait_until_ready().await?;
        Ok(server)
    }

    /// Poll `/health` until the server responds OK.
    /// Stderr is written only to the log file.
    /// Detects early crashes by checking if the child process has exited.
    /// No fixed timeout — waits as long as the process is alive.
    async fn wait_until_ready(&mut self) -> Result<()> {
        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{}/health", self.port);

        // Take stderr and spawn a task that writes to the log file.
        // Keep only the last N lines in memory for crash diagnostics.
        const STDERR_TAIL_LINES: usize = 30;
        let stderr_lines = std::sync::Arc::new(std::sync::Mutex::new(VecDeque::<String>::new()));
        if let Some(stderr) = self.child.stderr.take() {
            let lines = stderr_lines.clone();
            let log_path = self.log_path.clone();

            tokio::spawn(async move {
                let log_file = std::fs::File::create(&log_path).ok();
                let reader = BufReader::new(stderr);
                let mut line_stream = reader.lines();
                while let Ok(Some(line)) = line_stream.next_line().await {
                    if let Some(ref f) = log_file {
                        use std::io::Write;
                        let mut f = f;
                        let _ = writeln!(f, "{}", line);
                    }

                    let mut buf = lines.lock().unwrap();
                    if buf.len() >= STDERR_TAIL_LINES {
                        buf.pop_front();
                    }
                    buf.push_back(line);
                }
            });
        }

        loop {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            // Check if the process crashed
            if let Some(status) = self.child.try_wait()? {
                let buf = stderr_lines.lock().unwrap();
                let tail: String = buf.iter().cloned().collect::<Vec<_>>().join("\n");
                anyhow::bail!(
                    "llama-server exited with {} before becoming ready\n{}",
                    status,
                    tail
                );
            }

            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                _ => continue,
            }
        }
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Kill the child process.
    pub async fn stop(&mut self) {
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        // Best-effort kill on drop (non-async).
        let _ = self.child.start_kill();
    }
}

/// Find a free port for llama-server to bind on.
pub fn find_free_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

/// Resolve a GGUF model blob path by querying the running Ollama instance.
/// Parses the `FROM /path/to/blob` line from the modelfile returned by `/api/show`.
pub async fn find_ollama_model_path(model_name: &str) -> Result<PathBuf> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({ "name": model_name });

    let resp = client
        .post("http://localhost:11434/api/show")
        .json(&body)
        .send()
        .await
        .context("Failed to connect to Ollama to resolve model path. Is Ollama running?")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!(
            "Ollama /api/show error for '{}' ({}): {}",
            model_name,
            status,
            text
        );
    }

    let data: serde_json::Value = resp.json().await?;

    let modelfile = data
        .get("modelfile")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Ollama /api/show response missing 'modelfile' field"))?;

    // Parse the "FROM /path/to/blob" line from the modelfile
    for line in modelfile.lines() {
        let trimmed = line.trim();
        if let Some(path_str) = trimmed.strip_prefix("FROM ") {
            let path_str = path_str.trim();
            if path_str.starts_with('/') {
                let path = PathBuf::from(path_str);
                if path.exists() {
                    return Ok(path);
                }
                anyhow::bail!(
                    "Model blob path from Ollama does not exist: {}",
                    path.display()
                );
            }
        }
    }

    anyhow::bail!(
        "Could not extract model file path from Ollama for '{}'",
        model_name
    )
}
