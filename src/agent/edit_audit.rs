//! Append-only audit log for failed `edit` tool calls.
//!
//! The model loses sub-agent tool-result bodies after a successful retry, so
//! when a small model produces three failed edits before the fourth succeeds,
//! the failed `old_string`s are gone — making post-mortems impossible.
//!
//! This module persists every failed edit to a JSONL file (default
//! `/tmp/ollama-code-edit-failures.jsonl`) so we can inspect what the model
//! actually emitted.
//!
//! All write paths are best-effort — IO errors are swallowed. Logging must
//! never crash the agent.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::SystemTime;

use serde_json::{json, Value};

use crate::format::truncate_args;
use crate::session::format_utc;

const DEFAULT_LOG_PATH: &str = "/tmp/ollama-code-edit-failures.jsonl";

/// Append a JSONL entry describing a failed edit. Silently skips on IO error.
///
/// `log_path` lets tests redirect output; production callers pass `None` to
/// use `DEFAULT_LOG_PATH`.
pub(super) fn log_edit_failure(args: &Value, error: &str, log_path: Option<&Path>) {
    let path: std::path::PathBuf = log_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| Path::new(DEFAULT_LOG_PATH).to_path_buf());

    let file_path = args.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
    let old_string = args.get("old_string").and_then(|v| v.as_str()).unwrap_or("");
    let new_string = args.get("new_string").and_then(|v| v.as_str()).unwrap_or("");
    let start_line = args.get("start_line").cloned().unwrap_or(Value::Null);
    let end_line = args.get("end_line").cloned().unwrap_or(Value::Null);

    let entry = json!({
        "ts": format_utc(SystemTime::now()),
        "tool": "edit",
        "args": {
            "file_path": file_path,
            "old_string": truncate_args(old_string, 800),
            "new_string": truncate_args(new_string, 800),
            "start_line": start_line,
            "end_line": end_line,
        },
        "error": truncate_args(error, 600),
    });

    let line = match serde_json::to_string(&entry) {
        Ok(s) => s,
        Err(_) => return,
    };

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path);
    let mut file = match file {
        Ok(f) => f,
        Err(_) => return,
    };
    let _ = writeln!(file, "{}", line);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_audit_writes_jsonl_entry() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("audit.jsonl");
        let args = json!({
            "file_path": "/some/path.rs",
            "old_string": "let x = 1;",
            "new_string": "let x = 2;",
            "start_line": 10,
            "end_line": 10,
        });
        log_edit_failure(&args, "old_string not found", Some(&log));

        let body = std::fs::read_to_string(&log).expect("log file should exist");
        assert!(body.ends_with('\n'), "expected trailing newline: {:?}", body);

        let mut lines = body.lines();
        let first = lines.next().expect("at least one line");
        assert!(lines.next().is_none(), "expected exactly one line");

        let parsed: serde_json::Value =
            serde_json::from_str(first).expect("must be valid JSON");
        assert_eq!(parsed["tool"], "edit");
        assert_eq!(parsed["args"]["file_path"], "/some/path.rs");
        assert_eq!(parsed["args"]["old_string"], "let x = 1;");
        assert_eq!(parsed["args"]["new_string"], "let x = 2;");
        assert_eq!(parsed["args"]["start_line"], 10);
        assert_eq!(parsed["args"]["end_line"], 10);
        assert_eq!(parsed["error"], "old_string not found");
        let ts = parsed["ts"].as_str().expect("ts should be a string");
        assert!(ts.ends_with('Z'), "ts should be RFC3339 UTC: {}", ts);
        assert!(ts.contains('T'));
    }

    #[test]
    fn test_audit_appends_on_repeat_call() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("audit.jsonl");
        let args = json!({ "file_path": "/x", "old_string": "a", "new_string": "b" });
        log_edit_failure(&args, "err1", Some(&log));
        log_edit_failure(&args, "err2", Some(&log));

        let body = std::fs::read_to_string(&log).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_audit_skips_silently_on_bad_path() {
        let bad = Path::new("/this/dir/definitely/does/not/exist/audit.jsonl");
        log_edit_failure(
            &json!({ "file_path": "x", "old_string": "y", "new_string": "z" }),
            "err",
            Some(bad),
        );
    }

    #[test]
    fn test_audit_truncates_long_strings() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("audit.jsonl");
        let big = "x".repeat(2000);
        let args = json!({ "file_path": "/p", "old_string": big.clone(), "new_string": big });
        log_edit_failure(&args, &big, Some(&log));

        let body = std::fs::read_to_string(&log).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(body.trim()).unwrap();
        let stored = parsed["args"]["old_string"].as_str().unwrap();
        assert!(stored.ends_with("..."));
        assert!(stored.len() <= 800 + 3);
        let stored_err = parsed["error"].as_str().unwrap();
        assert!(stored_err.ends_with("..."));
        assert!(stored_err.len() <= 600 + 3);
    }
}
