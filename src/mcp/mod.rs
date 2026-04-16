//! Model Context Protocol client.
//!
//! Implements two MCP transports (stdio subprocess, Streamable HTTP) and
//! exposes discovered tools as [`Tool`](crate::tools::Tool) implementations.

mod connection;
mod http;
mod jsonrpc;
mod stdio;

use std::collections::HashMap;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tools::{Tool, ToolDefinition};

use self::connection::McpConnection;
use self::http::HttpConnection;
use self::jsonrpc::is_transport_error;
use self::stdio::StdioConnection;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a single MCP server in config.toml.
///
/// Exactly one of `command` (stdio transport) or `url` (Streamable HTTP) must
/// be set.
///
/// ```toml
/// # Stdio transport
/// [mcp.filesystem]
/// command = "npx"
/// args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
///
/// # Streamable HTTP transport
/// [mcp.remote]
/// url = "https://example.com/mcp"
/// headers = { Authorization = "Bearer tok_..." }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Command to spawn (stdio transport). Mutually exclusive with `url`.
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,

    /// HTTP endpoint URL (Streamable HTTP transport). Mutually exclusive with
    /// `command`.
    #[serde(default)]
    pub url: Option<String>,

    /// Extra HTTP headers (e.g. Authorization). Only used with `url`.
    #[serde(default)]
    pub headers: HashMap<String, String>,

    /// Whether tools from this server require user confirmation (default: true).
    #[serde(default = "default_true")]
    pub needs_confirm: bool,
}

fn default_true() -> bool {
    true
}

enum TransportKind {
    Stdio(String),
    Http(String),
}

impl McpServerConfig {
    fn transport_kind(&self) -> Result<TransportKind> {
        match (&self.command, &self.url) {
            (Some(cmd), None) => Ok(TransportKind::Stdio(cmd.clone())),
            (None, Some(url)) => Ok(TransportKind::Http(url.clone())),
            (Some(_), Some(_)) => {
                anyhow::bail!("MCP server config has both 'command' and 'url'; pick one")
            }
            (None, None) => {
                anyhow::bail!("MCP server config must have either 'command' or 'url'")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

/// Metadata about a single tool exposed by an MCP server.
#[derive(Debug, Clone)]
pub struct McpToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// A running MCP server (stdio child process or HTTP endpoint).
pub struct McpServer {
    pub name: String,
    child: Arc<Mutex<Option<Child>>>,
    connection: Arc<Mutex<McpConnection>>,
    pub tools: Vec<McpToolInfo>,
    pub needs_confirm: bool,
    /// Stored config for reconnection on child death.
    config: McpServerConfig,
}

/// Start all configured MCP servers, returning those that succeeded.
/// Failures are reported via eprintln and skipped.
pub fn start_servers(
    configs: &HashMap<String, McpServerConfig>,
    is_tool_enabled: impl Fn(&str) -> bool,
) -> Vec<McpServer> {
    let mut servers = Vec::new();
    for (name, cfg) in configs {
        if !is_tool_enabled(name) {
            continue;
        }
        match McpServer::start(name, cfg) {
            Ok(server) => servers.push(server),
            Err(e) => eprintln!("Warning: MCP server '{name}': {e}"),
        }
    }
    servers
}

impl McpServer {
    /// Connect to an MCP server, perform the initialize handshake, and
    /// discover tools. The transport (stdio or HTTP) is determined by the
    /// config.
    pub fn start(name: &str, config: &McpServerConfig) -> Result<Self> {
        match config.transport_kind()? {
            TransportKind::Stdio(cmd) => Self::start_stdio(name, config, &cmd),
            TransportKind::Http(url) => Self::start_http(name, config, &url),
        }
    }

    fn start_stdio(name: &str, config: &McpServerConfig, command: &str) -> Result<Self> {
        let (child, connection, tools) = Self::spawn_stdio(name, config, command)?;

        Ok(Self {
            name: name.to_string(),
            child: Arc::new(Mutex::new(Some(child))),
            connection: Arc::new(Mutex::new(connection)),
            tools,
            needs_confirm: config.needs_confirm,
            config: config.clone(),
        })
    }

    /// Spawn the stdio child process and perform the MCP handshake.
    fn spawn_stdio(
        name: &str,
        config: &McpServerConfig,
        command: &str,
    ) -> Result<(Child, McpConnection, Vec<McpToolInfo>)> {
        let mut cmd = Command::new(command);
        cmd.args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        for (k, v) in &config.env {
            cmd.env(k, v);
        }

        let mut child = cmd.spawn().map_err(|e| {
            anyhow::anyhow!("Failed to start MCP server '{name}' ({command}): {e}")
        })?;

        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");

        let conn = StdioConnection::new(stdin, stdout);

        let mut connection = McpConnection::Stdio(conn);
        let tools = connection.init_and_discover("2024-11-05")?;

        Ok((child, connection, tools))
    }

    fn start_http(name: &str, config: &McpServerConfig, url: &str) -> Result<Self> {
        let conn = HttpConnection::new(url, &config.headers);

        let mut connection = McpConnection::Http(conn);
        let tools = connection.init_and_discover("2025-03-26")?;

        Ok(Self {
            name: name.to_string(),
            child: Arc::new(Mutex::new(None)),
            connection: Arc::new(Mutex::new(connection)),
            tools,
            needs_confirm: config.needs_confirm,
            config: config.clone(),
        })
    }

    /// Attempt to reconnect a stdio MCP server after the child process dies.
    /// Can be called from McpTool via the shared Arc<Mutex> references.
    fn reconnect_stdio_shared(
        name: &str,
        config: &McpServerConfig,
        child: &Mutex<Option<Child>>,
        connection: &Mutex<McpConnection>,
    ) -> Result<()> {
        let command = config
            .command
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Cannot reconnect: not a stdio transport"))?;

        // Kill old child if it's still around
        if let Ok(mut child_opt) = child.lock() {
            if let Some(ref mut c) = *child_opt {
                let _ = c.kill();
                let _ = c.wait();
            }
            *child_opt = None;
        }

        let (new_child, new_conn, _tools) = Self::spawn_stdio(name, config, command)?;
        *child.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {e}"))? = Some(new_child);
        *connection.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {e}"))? = new_conn;
        Ok(())
    }

    /// Create `McpTool` instances (implementing `Tool`) for each tool this
    /// server exposes.
    pub fn create_tools(&self) -> Vec<McpTool> {
        self.tools
            .iter()
            .map(|info| McpTool {
                qualified_name: format!("mcp__{}__{}", self.name, info.name),
                tool_info: info.clone(),
                connection: Arc::clone(&self.connection),
                server_name: self.name.clone(),
                config: self.config.clone(),
                child: Arc::clone(&self.child),
            })
            .collect()
    }

    /// A short label for the transport type shown by `/mcp`.
    pub fn transport_label(&self) -> &'static str {
        if self.config.command.is_some() {
            "stdio"
        } else {
            "http"
        }
    }
}

impl Drop for McpServer {
    fn drop(&mut self) {
        // HTTP: attempt clean session teardown.
        if let Ok(conn) = self.connection.lock() {
            if let McpConnection::Http(ref http) = *conn {
                http.teardown();
            }
        }
        // Stdio: kill child process.
        if let Ok(mut child_opt) = self.child.lock() {
            if let Some(ref mut child) = *child_opt {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// McpTool -- Tool trait adapter
// ---------------------------------------------------------------------------

/// A single tool from an MCP server, wrapped as a `Tool` trait object.
pub struct McpTool {
    /// Qualified name: `mcp__<server>__<tool>`.
    qualified_name: String,
    tool_info: McpToolInfo,
    connection: Arc<Mutex<McpConnection>>,
    /// Stored for stdio reconnection on child death.
    server_name: String,
    config: McpServerConfig,
    child: Arc<Mutex<Option<Child>>>,
}

impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.qualified_name
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.qualified_name.clone(),
            description: self.tool_info.description.clone(),
            parameters: self.tool_info.input_schema.clone(),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let call_params = serde_json::json!({
            "name": self.tool_info.name,
            "arguments": arguments,
        });

        // First attempt
        let result = {
            let mut conn = self
                .connection
                .lock()
                .map_err(|e| anyhow::anyhow!("MCP connection lock poisoned: {e}"))?;
            conn.request("tools/call", call_params.clone())
        };

        match result {
            Ok(val) => format_tool_result(&val),
            Err(e) if self.config.command.is_some() && is_transport_error(&e) => {
                // Stdio transport error (EOF, broken pipe) — attempt reconnection
                eprintln!(
                    "MCP server '{}' transport error: {}. Attempting reconnection...",
                    self.server_name, e
                );
                McpServer::reconnect_stdio_shared(
                    &self.server_name,
                    &self.config,
                    &self.child,
                    &self.connection,
                )?;

                // Retry the call on the fresh connection
                let mut conn = self
                    .connection
                    .lock()
                    .map_err(|e| anyhow::anyhow!("MCP connection lock poisoned: {e}"))?;
                let result = conn.request("tools/call", call_params)?;
                format_tool_result(&result)
            }
            Err(e) => Err(e),
        }
    }
}

/// Extract text from an MCP tool result's `content` array.
fn format_tool_result(result: &Value) -> Result<String> {
    let text = result
        .get("content")
        .and_then(|c| c.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| match item.get("type").and_then(|t| t.as_str()) {
                    Some("text") => item.get("text").and_then(|t| t.as_str()).map(String::from),
                    Some(other) => Some(format!("[{other} content]")),
                    None => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_else(|| serde_json::to_string_pretty(result).unwrap_or_default());

    let is_error = result
        .get("isError")
        .and_then(|e| e.as_bool())
        .unwrap_or(false);
    if is_error {
        anyhow::bail!("{text}");
    }

    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- format_tool_result -------------------------------------------------

    #[test]
    fn format_text_result() {
        let result = serde_json::json!({
            "content": [{"type": "text", "text": "hello world"}]
        });
        assert_eq!(format_tool_result(&result).unwrap(), "hello world");
    }

    #[test]
    fn format_multi_content_result() {
        let result = serde_json::json!({
            "content": [
                {"type": "text", "text": "line 1"},
                {"type": "text", "text": "line 2"},
            ]
        });
        assert_eq!(format_tool_result(&result).unwrap(), "line 1\nline 2");
    }

    #[test]
    fn format_error_result() {
        let result = serde_json::json!({
            "content": [{"type": "text", "text": "something went wrong"}],
            "isError": true
        });
        assert!(format_tool_result(&result).is_err());
    }

    #[test]
    fn format_non_text_content() {
        let result = serde_json::json!({
            "content": [{"type": "image", "data": "..."}]
        });
        assert_eq!(format_tool_result(&result).unwrap(), "[image content]");
    }

    // -- config deserialization ---------------------------------------------

    #[test]
    fn config_deserialize_stdio_minimal() {
        let toml_str = r#"command = "echo""#;
        let cfg: McpServerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.command.as_deref(), Some("echo"));
        assert!(cfg.url.is_none());
        assert!(cfg.args.is_empty());
        assert!(cfg.needs_confirm);
    }

    #[test]
    fn config_deserialize_stdio_full() {
        let toml_str = r#"
            command = "npx"
            args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            needs_confirm = false
            [env]
            API_KEY = "secret"
        "#;
        let cfg: McpServerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.command.as_deref(), Some("npx"));
        assert_eq!(cfg.args.len(), 3);
        assert!(!cfg.needs_confirm);
        assert_eq!(cfg.env["API_KEY"], "secret");
    }

    #[test]
    fn config_deserialize_http_minimal() {
        let toml_str = r#"url = "https://example.com/mcp""#;
        let cfg: McpServerConfig = toml::from_str(toml_str).unwrap();
        assert!(cfg.command.is_none());
        assert_eq!(cfg.url.as_deref(), Some("https://example.com/mcp"));
        assert!(cfg.headers.is_empty());
        assert!(cfg.needs_confirm);
    }

    #[test]
    fn config_deserialize_http_with_headers() {
        let toml_str = r#"
            url = "https://example.com/mcp"
            needs_confirm = false
            [headers]
            Authorization = "Bearer tok_123"
        "#;
        let cfg: McpServerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.url.as_deref(), Some("https://example.com/mcp"));
        assert_eq!(cfg.headers["Authorization"], "Bearer tok_123");
        assert!(!cfg.needs_confirm);
    }

    // -- transport_kind validation ------------------------------------------

    #[test]
    fn transport_kind_stdio() {
        let cfg = McpServerConfig {
            command: Some("echo".into()),
            url: None,
            args: vec![],
            env: HashMap::new(),
            headers: HashMap::new(),
            needs_confirm: true,
        };
        assert!(matches!(cfg.transport_kind().unwrap(), TransportKind::Stdio(_)));
    }

    #[test]
    fn transport_kind_http() {
        let cfg = McpServerConfig {
            command: None,
            url: Some("https://example.com/mcp".into()),
            args: vec![],
            env: HashMap::new(),
            headers: HashMap::new(),
            needs_confirm: true,
        };
        assert!(matches!(cfg.transport_kind().unwrap(), TransportKind::Http(_)));
    }

    #[test]
    fn transport_kind_both_errors() {
        let cfg = McpServerConfig {
            command: Some("echo".into()),
            url: Some("https://example.com/mcp".into()),
            args: vec![],
            env: HashMap::new(),
            headers: HashMap::new(),
            needs_confirm: true,
        };
        assert!(cfg.transport_kind().is_err());
    }

    #[test]
    fn transport_kind_neither_errors() {
        let cfg = McpServerConfig {
            command: None,
            url: None,
            args: vec![],
            env: HashMap::new(),
            headers: HashMap::new(),
            needs_confirm: true,
        };
        assert!(cfg.transport_kind().is_err());
    }
}
