//! Async helper for starting (or reusing) a `llama-server` child process and
//! handing the resulting backend back through a channel.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::llama_server::{self, LlamaCppBackend, LlamaServer, ModelSource};

use crate::tui::BackendReady;

/// Shared logic for starting a llama-server and sending the result over a channel.
pub(crate) async fn spawn_llama_server_inner(
    server_path: String,
    model_source: ModelSource,
    ctx: u64,
    extra_args: Vec<String>,
    model_name: String,
    sampling: crate::ollama::SamplingParams,
    tx: mpsc::UnboundedSender<BackendReady>,
) {
    // Try to reuse an existing llama-server for the same model.
    if let Some(server) = LlamaServer::connect_existing(&model_source).await {
        let backend = LlamaCppBackend::with_sampling(server.base_url(), sampling);
        let _ = tx.send(Ok((Arc::new(backend), model_name, server)));
        return;
    }

    let server_binary = std::path::PathBuf::from(&server_path);
    if !server_binary.exists() {
        let _ = tx.send(Err(anyhow::anyhow!(
            "llama-server binary not found at: {}",
            server_binary.display()
        )));
        return;
    }
    let port = match llama_server::find_free_port() {
        Ok(p) => p,
        Err(e) => {
            let _ = tx.send(Err(e));
            return;
        }
    };
    match LlamaServer::start(&server_binary, &model_source, port, ctx, &extra_args).await {
        Ok(server) => {
            let backend = LlamaCppBackend::with_sampling(server.base_url(), sampling);
            let _ = tx.send(Ok((Arc::new(backend), model_name, server)));
        }
        Err(e) => {
            let _ = tx.send(Err(e));
        }
    }
}
