use std::future::Future;
use std::pin::Pin;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

use crate::backend::{ChatResponse, ModelBackend};
use crate::message::Message;
use crate::ollama::OllamaBackend;
/// How to specify the model for llama-server.
pub enum ModelSource {
    /// Local GGUF file path.
    File(PathBuf),
    /// HuggingFace repo identifier (e.g. "google/gemma-3-27b-it-GGUF").
    HuggingFace(String),
}

impl ModelSource {
    /// A string identifier for this model source, used to match running servers.
    pub fn model_id(&self) -> String {
        match self {
            ModelSource::File(path) => path.to_string_lossy().to_string(),
            ModelSource::HuggingFace(repo) => repo.clone(),
        }
    }
}

// --- Server info file for sharing a llama-server between instances ---

#[derive(Serialize, Deserialize)]
struct ServerInfo {
    pid: u32,
    port: u16,
    model_id: String,
}

fn server_info_path() -> PathBuf {
    crate::config::data_dir().join("llama-server.json")
}

fn write_server_info(pid: u32, port: u16, model_id: &str) -> Result<()> {
    let info = ServerInfo {
        pid,
        port,
        model_id: model_id.to_string(),
    };
    let path = server_info_path();
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(&path, serde_json::to_string(&info)?)?;
    Ok(())
}

fn read_server_info() -> Option<ServerInfo> {
    let data = std::fs::read_to_string(server_info_path()).ok()?;
    serde_json::from_str(&data).ok()
}

fn remove_server_info() {
    let _ = std::fs::remove_file(server_info_path());
}

/// Manages a llama-server child process, or connects to one started by another instance.
pub struct LlamaServer {
    /// `Some` if this instance spawned the server; `None` if reusing another instance's server.
    child: Option<Child>,
    pub port: u16,
    pub log_path: PathBuf,
}

impl LlamaServer {
    /// Try to connect to a llama-server already running for the given model.
    /// Returns `None` if no compatible server is running.
    pub async fn connect_existing(model: &ModelSource) -> Option<Self> {
        let info = read_server_info()?;
        if info.model_id != model.model_id() {
            return None;
        }
        // Probe the health endpoint to verify the server is actually alive.
        let url = format!("http://127.0.0.1:{}/health", info.port);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .ok()?;
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                let log_dir = crate::config::data_dir();
                Some(Self {
                    child: None,
                    port: info.port,
                    log_path: log_dir.join("llama-server.log"),
                })
            }
            _ => {
                // Stale info file — clean it up.
                remove_server_info();
                None
            }
        }
    }

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
            child: Some(child),
            port,
            log_path,
        };
        server.wait_until_ready(model).await?;
        Ok(server)
    }

    /// Poll `/health` until the server responds OK.
    /// Stderr is written only to the log file.
    /// Detects early crashes by checking if the child process has exited.
    /// No fixed timeout — waits as long as the process is alive.
    async fn wait_until_ready(&mut self, model: &ModelSource) -> Result<()> {
        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{}/health", self.port);

        let child = self.child.as_mut().expect("wait_until_ready called on non-owned server");

        // Take stderr and spawn a task that writes to the log file.
        // Keep only the last N lines in memory for crash diagnostics.
        const STDERR_TAIL_LINES: usize = 30;
        let stderr_lines = std::sync::Arc::new(std::sync::Mutex::new(VecDeque::<String>::new()));
        if let Some(stderr) = child.stderr.take() {
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

        // Record the PID for the server info file.
        let pid = child.id();

        loop {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            // Check if the process crashed
            let child = self.child.as_mut().unwrap();
            if let Some(status) = child.try_wait()? {
                let buf = stderr_lines.lock().unwrap();
                let tail: String = buf.iter().cloned().collect::<Vec<_>>().join("\n");
                anyhow::bail!(
                    "llama-server exited with {} before becoming ready\n{}",
                    status,
                    tail
                );
            }

            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    // Server is ready — write the info file so other instances can find it.
                    if let Some(pid) = pid {
                        let _ = write_server_info(pid, self.port, &model.model_id());
                    }
                    return Ok(());
                }
                _ => continue,
            }
        }
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Kill the child process (no-op if this instance doesn't own the server).
    pub async fn stop(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.kill().await;
            let _ = child.wait().await;
            remove_server_info();
        }
    }

    /// Whether this instance spawned and owns the server process.
    pub fn is_owned(&self) -> bool {
        self.child.is_some()
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        if let Some(ref mut child) = self.child {
            // Best-effort kill on drop (non-async).
            let _ = child.start_kill();
            remove_server_info();
        }
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

/// Backend that talks to a llama-server instance via its OpenAI-compatible API.
///
/// Delegates the actual HTTP/streaming work to an [`OllamaBackend`] pointed at
/// the llama-server URL (both speak the same chat protocol).
pub struct LlamaCppBackend {
    inner: OllamaBackend,
}

impl LlamaCppBackend {
    pub fn new(base_url: String) -> Self {
        Self {
            inner: OllamaBackend::new(Some(base_url)),
        }
    }
}

impl ModelBackend for LlamaCppBackend {
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: &'a [Message],
        tools: Option<Vec<Value>>,
        num_ctx: Option<u64>,
        on_token: Box<dyn Fn(&str) + Send + 'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        self.inner.chat(model, messages, tools, num_ctx, on_token)
    }
}
