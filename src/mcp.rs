use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tools::{Tool, ToolDefinition};

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
/// [mcp_servers.filesystem]
/// command = "npx"
/// args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
///
/// # Streamable HTTP transport
/// [mcp_servers.remote]
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
// Transport abstraction
// ---------------------------------------------------------------------------

/// Unified connection to an MCP server (stdio or HTTP).
enum McpConnection {
    Stdio(StdioConnection),
    Http(HttpConnection),
}

impl McpConnection {
    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        match self {
            McpConnection::Stdio(c) => c.request(method, params),
            McpConnection::Http(c) => c.request(method, params),
        }
    }

    fn notify(&mut self, method: &str) -> Result<()> {
        match self {
            McpConnection::Stdio(c) => c.notify(method),
            McpConnection::Http(c) => c.notify(method),
        }
    }

    /// Perform the MCP initialize handshake and discover tools.
    fn init_and_discover(&mut self, protocol_version: &str) -> Result<Vec<McpToolInfo>> {
        self.request(
            "initialize",
            serde_json::json!({
                "protocolVersion": protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": "ollama-code",
                    "version": env!("CARGO_PKG_VERSION"),
                }
            }),
        )?;
        self.notify("notifications/initialized")?;

        let result = self.request("tools/list", serde_json::json!({}))?;
        Ok(parse_tools_list(&result))
    }
}

// ---------------------------------------------------------------------------
// Stdio transport
// ---------------------------------------------------------------------------

/// JSON-RPC over newline-delimited stdin/stdout.
struct StdioConnection {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl StdioConnection {
    /// Send a JSON-RPC request and wait for the matching response.
    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;

        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let line = serde_json::to_string(&msg)?;
        writeln!(self.stdin, "{}", line)?;
        self.stdin.flush()?;

        let mut buf = String::new();
        loop {
            buf.clear();
            let n = self.stdout.read_line(&mut buf)?;
            if n == 0 {
                anyhow::bail!("MCP server closed stdout (EOF)");
            }
            let trimmed = buf.trim();
            if trimmed.is_empty() {
                continue;
            }

            let resp: Value = serde_json::from_str(trimmed)
                .map_err(|e| anyhow::anyhow!("Invalid JSON from MCP server: {e}"))?;

            if let Some(result) = extract_matching_response(&resp, id)? {
                return Ok(result);
            }
        }
    }

    /// Send a JSON-RPC notification (no id, no response expected).
    fn notify(&mut self, method: &str) -> Result<()> {
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        let line = serde_json::to_string(&msg)?;
        writeln!(self.stdin, "{}", line)?;
        self.stdin.flush()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Streamable HTTP transport (MCP spec 2025-03-26)
// ---------------------------------------------------------------------------

/// JSON-RPC over HTTP POST with JSON or SSE responses.
struct HttpConnection {
    agent: ureq::Agent,
    endpoint: String,
    session_id: Option<String>,
    headers: HashMap<String, String>,
    next_id: u64,
}

impl HttpConnection {
    fn new(endpoint: &str, headers: &HashMap<String, String>) -> Self {
        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_global(Some(std::time::Duration::from_secs(120)))
                .build(),
        );
        Self {
            agent,
            endpoint: endpoint.to_string(),
            session_id: None,
            headers: headers.clone(),
            next_id: 1,
        }
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
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
            .agent
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
            .send_json(body)
            .map_err(|e| anyhow::anyhow!("MCP HTTP request failed: {e}"))?;

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
            parse_sse_response(resp.into_body().into_reader(), expected_id)
        } else {
            // Default: parse as JSON.
            let mut body = resp.into_body();
            let text = body
                .read_to_string()
                .map_err(|e| anyhow::anyhow!("Failed to read MCP HTTP response: {e}"))?;
            let resp_val: Value = serde_json::from_str(&text)
                .map_err(|e| anyhow::anyhow!("Invalid JSON from MCP HTTP server: {e}"))?;
            extract_matching_response(&resp_val, expected_id)?
                .ok_or_else(|| anyhow::anyhow!("MCP HTTP response id mismatch"))
        }
    }

    fn notify(&mut self, method: &str) -> Result<()> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
        });

        let mut req = self
            .agent
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
            .send_json(&body)
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
    fn teardown(&self) {
        if self.session_id.is_none() {
            return;
        }
        let mut req = self.agent.delete(&self.endpoint);
        if let Some(ref sid) = self.session_id {
            req = req.header("Mcp-Session-Id", sid);
        }
        let _ = req.call();
    }
}

fn is_session_expired(err: &anyhow::Error) -> bool {
    let msg = err.to_string();
    msg.contains("404") || msg.contains("Not Found")
}

// ---------------------------------------------------------------------------
// Shared JSON-RPC helpers
// ---------------------------------------------------------------------------

/// If `resp` is a JSON-RPC response matching `expected_id`, return its result.
/// Returns `Ok(None)` for notifications or non-matching ids (caller should
/// keep reading). Returns `Err` for JSON-RPC error responses.
fn extract_matching_response(resp: &Value, expected_id: u64) -> Result<Option<Value>> {
    // Skip notifications (no id field).
    if resp.get("id").is_none() || resp["id"].is_null() {
        return Ok(None);
    }
    // Check for matching id.
    if resp["id"].as_u64() != Some(expected_id) {
        return Ok(None);
    }
    // Check for JSON-RPC error.
    if let Some(err) = resp.get("error") {
        let code = err.get("code").and_then(|v| v.as_i64()).unwrap_or(0);
        let message = err
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        anyhow::bail!("MCP error {code}: {message}");
    }
    Ok(Some(resp.get("result").cloned().unwrap_or(Value::Null)))
}

/// Parse a Server-Sent Events stream to find the JSON-RPC response matching
/// `expected_id`. Notifications and non-matching messages are skipped.
fn parse_sse_response(reader: impl Read, expected_id: u64) -> Result<Value> {
    let buf = BufReader::new(reader);
    let mut data_buf = String::new();

    for line in buf.lines() {
        let line = line?;
        if let Some(data) = line.strip_prefix("data:") {
            let payload = data.trim();
            if data_buf.is_empty() {
                data_buf = payload.to_string();
            } else {
                // Multi-line data: concatenate with newline per SSE spec.
                data_buf.push('\n');
                data_buf.push_str(payload);
            }
        } else if line.is_empty() {
            // Event boundary.
            if !data_buf.is_empty() {
                if let Ok(resp) = serde_json::from_str::<Value>(&data_buf) {
                    if let Some(result) = extract_matching_response(&resp, expected_id)? {
                        return Ok(result);
                    }
                }
            }
            data_buf.clear();
        }
        // Lines starting with "event:", "id:", "retry:", or ":" are
        // acknowledged but don't affect our response extraction.
    }

    anyhow::bail!("SSE stream ended without response for request id {expected_id}")
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
    child: Option<Child>,
    connection: Arc<Mutex<McpConnection>>,
    pub tools: Vec<McpToolInfo>,
    pub needs_confirm: bool,
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

        let conn = StdioConnection {
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        };

        let mut connection = McpConnection::Stdio(conn);
        let tools = connection.init_and_discover("2024-11-05")?;

        Ok(Self {
            name: name.to_string(),
            child: Some(child),
            connection: Arc::new(Mutex::new(connection)),
            tools,
            needs_confirm: config.needs_confirm,
        })
    }

    fn start_http(name: &str, config: &McpServerConfig, url: &str) -> Result<Self> {
        let conn = HttpConnection::new(url, &config.headers);

        let mut connection = McpConnection::Http(conn);
        let tools = connection.init_and_discover("2025-03-26")?;

        Ok(Self {
            name: name.to_string(),
            child: None,
            connection: Arc::new(Mutex::new(connection)),
            tools,
            needs_confirm: config.needs_confirm,
        })
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
            })
            .collect()
    }

    /// A short label for the transport type shown by `/mcp`.
    pub fn transport_label(&self) -> &'static str {
        if self.child.is_some() {
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
        if let Some(ref mut child) = self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn parse_tools_list(result: &Value) -> Vec<McpToolInfo> {
    result
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    let name = t.get("name")?.as_str()?.to_string();
                    let description = t
                        .get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or("")
                        .to_string();
                    let input_schema = t.get("inputSchema").cloned().unwrap_or_else(|| {
                        serde_json::json!({"type": "object", "properties": {}})
                    });
                    Some(McpToolInfo {
                        name,
                        description,
                        input_schema,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
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
        let mut conn = self
            .connection
            .lock()
            .map_err(|e| anyhow::anyhow!("MCP connection lock poisoned: {e}"))?;

        let result = conn.request(
            "tools/call",
            serde_json::json!({
                "name": self.tool_info.name,
                "arguments": arguments,
            }),
        )?;

        format_tool_result(&result)
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

    // -- parse_tools_list ---------------------------------------------------

    #[test]
    fn parse_empty_tools_list() {
        let result = serde_json::json!({"tools": []});
        assert!(parse_tools_list(&result).is_empty());
    }

    #[test]
    fn parse_tools_list_basic() {
        let result = serde_json::json!({
            "tools": [{
                "name": "read_file",
                "description": "Read a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }
            }]
        });
        let tools = parse_tools_list(&result);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "read_file");
        assert_eq!(tools[0].description, "Read a file");
    }

    #[test]
    fn parse_tools_list_missing_schema() {
        let result = serde_json::json!({
            "tools": [{"name": "ping", "description": "Ping"}]
        });
        let tools = parse_tools_list(&result);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].input_schema["type"], "object");
    }

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

    // -- SSE parsing --------------------------------------------------------

    #[test]
    fn sse_single_json_response() {
        let sse = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"value\":42}}\n\n";
        let result = parse_sse_response(sse.as_bytes(), 1).unwrap();
        assert_eq!(result["value"], 42);
    }

    #[test]
    fn sse_skips_notifications() {
        let sse = concat!(
            "event: message\n",
            "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\"}\n",
            "\n",
            "event: message\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":5,\"result\":{\"ok\":true}}\n",
            "\n",
        );
        let result = parse_sse_response(sse.as_bytes(), 5).unwrap();
        assert_eq!(result["ok"], true);
    }

    #[test]
    fn sse_error_response() {
        let sse = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{\"code\":-32601,\"message\":\"Method not found\"}}\n\n";
        let err = parse_sse_response(sse.as_bytes(), 1).unwrap_err();
        assert!(err.to_string().contains("Method not found"));
    }

    #[test]
    fn sse_empty_data_skipped() {
        let sse = concat!(
            "id: init\n",
            "data: \n",
            "\n",
            "event: message\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n",
            "\n",
        );
        let result = parse_sse_response(sse.as_bytes(), 1).unwrap();
        assert_eq!(result, serde_json::json!({}));
    }

    #[test]
    fn sse_no_event_prefix_still_works() {
        // Some servers omit the "event: message" line.
        let sse = "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"v\":1}}\n\n";
        let result = parse_sse_response(sse.as_bytes(), 1).unwrap();
        assert_eq!(result["v"], 1);
    }

    #[test]
    fn sse_stream_ends_without_match() {
        let sse = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":99,\"result\":{}}\n\n";
        assert!(parse_sse_response(sse.as_bytes(), 1).is_err());
    }

    // -- extract_matching_response ------------------------------------------

    #[test]
    fn extract_skips_notification() {
        let resp = serde_json::json!({"jsonrpc": "2.0", "method": "ping"});
        assert!(extract_matching_response(&resp, 1).unwrap().is_none());
    }

    #[test]
    fn extract_skips_mismatched_id() {
        let resp = serde_json::json!({"jsonrpc": "2.0", "id": 99, "result": {}});
        assert!(extract_matching_response(&resp, 1).unwrap().is_none());
    }

    #[test]
    fn extract_returns_result() {
        let resp = serde_json::json!({"jsonrpc": "2.0", "id": 1, "result": {"x": 1}});
        let val = extract_matching_response(&resp, 1).unwrap().unwrap();
        assert_eq!(val["x"], 1);
    }

    #[test]
    fn extract_returns_error() {
        let resp = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -1, "message": "bad"}
        });
        assert!(extract_matching_response(&resp, 1).is_err());
    }
}
