//! JSON-RPC over newline-delimited stdin/stdout for a child process.

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout};

use anyhow::Result;
use serde_json::Value;

use super::jsonrpc::extract_matching_response;

pub(super) struct StdioConnection {
    pub(super) stdin: ChildStdin,
    pub(super) stdout: BufReader<ChildStdout>,
    pub(super) next_id: u64,
}

impl StdioConnection {
    pub(super) fn new(stdin: ChildStdin, stdout: ChildStdout) -> Self {
        Self {
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        }
    }

    /// Send a JSON-RPC request and wait for the matching response.
    pub(super) fn request(&mut self, method: &str, params: Value) -> Result<Value> {
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
    pub(super) fn notify(&mut self, method: &str) -> Result<()> {
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
