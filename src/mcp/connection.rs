//! Transport-agnostic wrapper around an MCP connection.

use anyhow::Result;
use serde_json::Value;

use super::http::HttpConnection;
use super::stdio::StdioConnection;
use super::McpToolInfo;

/// Unified connection to an MCP server (stdio or HTTP).
pub(super) enum McpConnection {
    Stdio(StdioConnection),
    Http(HttpConnection),
}

impl McpConnection {
    pub(super) fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        match self {
            McpConnection::Stdio(c) => c.request(method, params),
            McpConnection::Http(c) => c.request(method, params),
        }
    }

    pub(super) fn notify(&mut self, method: &str) -> Result<()> {
        match self {
            McpConnection::Stdio(c) => c.notify(method),
            McpConnection::Http(c) => c.notify(method),
        }
    }

    /// Perform the MCP initialize handshake and discover tools.
    pub(super) fn init_and_discover(&mut self, protocol_version: &str) -> Result<Vec<McpToolInfo>> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
