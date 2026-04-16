//! JSON-RPC over HTTP POST (MCP Streamable HTTP transport, spec 2025-03-26).

use std::collections::HashMap;

use anyhow::Result;
use serde_json::Value;

use super::jsonrpc::{extract_matching_response, parse_sse_response};

pub(super) struct HttpConnection {
    client: reqwest::blocking::Client,
    endpoint: String,
    session_id: Option<String>,
    headers: HashMap<String, String>,
    next_id: u64,
}

impl HttpConnection {
    pub(super) fn new(endpoint: &str, headers: &HashMap<String, String>) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            client,
            endpoint: endpoint.to_string(),
            session_id: None,
            headers: headers.clone(),
            next_id: 1,
        }
    }

    pub(super) fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;

        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        match self.post_request(&body) {
            Ok(val) => Ok(val),
            Err(e) if is_session_expired(&e) => {
                self.session_id = None;
                self.reinitialize()?;
                // Reassign id after reinitialize consumed some ids.
                let id = self.next_id;
                self.next_id += 1;
                let body = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "method": method,
                    "params": params,
                });
                self.post_request(&body)
            }
            Err(e) => Err(e),
        }
    }

    fn post_request(&mut self, body: &Value) -> Result<Value> {
        let expected_id = body["id"].as_u64().unwrap_or(0);

        let mut req = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");

        if let Some(ref sid) = self.session_id {
            req = req.header("Mcp-Session-Id", sid.as_str());
        }
        for (k, v) in &self.headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let resp = req
            .json(body)
            .send()
            .map_err(|e| anyhow::anyhow!("MCP HTTP request failed: {e}"))?;

        let status = resp.status();
        if !status.is_success() && status.as_u16() != 202 {
            anyhow::bail!("MCP HTTP request failed with status {status}");
        }

        // Capture session id from response.
        if let Some(sid) = resp.headers().get("mcp-session-id") {
            if let Ok(s) = sid.to_str() {
                self.session_id = Some(s.to_string());
            }
        }

        let is_sse = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .contains("text/event-stream");

        if is_sse {
            parse_sse_response(resp, expected_id)
        } else {
            let text = resp
                .text()
                .map_err(|e| anyhow::anyhow!("Failed to read MCP HTTP response: {e}"))?;
            let resp_val: Value = serde_json::from_str(&text)
                .map_err(|e| anyhow::anyhow!("Invalid JSON from MCP HTTP server: {e}"))?;
            extract_matching_response(&resp_val, expected_id)?
                .ok_or_else(|| anyhow::anyhow!("MCP HTTP response id mismatch"))
        }
    }

    pub(super) fn notify(&mut self, method: &str) -> Result<()> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
        });

        let mut req = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");

        if let Some(ref sid) = self.session_id {
            req = req.header("Mcp-Session-Id", sid.as_str());
        }
        for (k, v) in &self.headers {
            req = req.header(k.as_str(), v.as_str());
        }

        // Notifications may get 202 Accepted (no body) or 200.
        let _resp = req
            .json(&body)
            .send()
            .map_err(|e| anyhow::anyhow!("MCP HTTP notify failed: {e}"))?;
        Ok(())
    }

    /// Re-run the initialize + initialized handshake after session expiry.
    fn reinitialize(&mut self) -> Result<()> {
        let id = self.next_id;
        self.next_id += 1;
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {
                    "name": "ollama-code",
                    "version": env!("CARGO_PKG_VERSION"),
                }
            },
        });
        self.post_request(&body)?;
        self.notify("notifications/initialized")?;
        Ok(())
    }

    /// Send DELETE to tear down the session (called from McpServer::drop).
    pub(super) fn teardown(&self) {
        if self.session_id.is_none() {
            return;
        }
        let mut req = self.client.delete(&self.endpoint);
        if let Some(ref sid) = self.session_id {
            req = req.header("Mcp-Session-Id", sid);
        }
        let _ = req.send();
    }
}

fn is_session_expired(err: &anyhow::Error) -> bool {
    let msg = err.to_string();
    msg.contains("404") || msg.contains("Not Found")
}
