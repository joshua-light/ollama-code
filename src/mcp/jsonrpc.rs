//! Shared JSON-RPC decoding helpers for both MCP transports.

use std::io::{BufRead, BufReader, Read};

use anyhow::Result;
use serde_json::Value;

/// If `resp` is a JSON-RPC response matching `expected_id`, return its result.
/// Returns `Ok(None)` for notifications or non-matching ids (caller should
/// keep reading). Returns `Err` for JSON-RPC error responses.
pub(super) fn extract_matching_response(resp: &Value, expected_id: u64) -> Result<Option<Value>> {
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
pub(super) fn parse_sse_response(reader: impl Read, expected_id: u64) -> Result<Value> {
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

/// Check if an error indicates a dead stdio transport (EOF or broken pipe).
pub(super) fn is_transport_error(err: &anyhow::Error) -> bool {
    let msg = err.to_string();
    msg.contains("EOF")
        || msg.contains("Broken pipe")
        || msg.contains("broken pipe")
        || msg.contains("closed stdout")
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let sse = "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"v\":1}}\n\n";
        let result = parse_sse_response(sse.as_bytes(), 1).unwrap();
        assert_eq!(result["v"], 1);
    }

    #[test]
    fn sse_stream_ends_without_match() {
        let sse = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":99,\"result\":{}}\n\n";
        assert!(parse_sse_response(sse.as_bytes(), 1).is_err());
    }

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
