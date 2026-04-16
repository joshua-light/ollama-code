//! Subprocess execution of a single hook (stdin JSON → stdout JSON).

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::Result;
use regex::Regex;
use serde_json::Value;

use super::types::{HookEvent, HookInput, ResolvedHook};

/// Execute a single hook as a subprocess. Returns parsed JSON output or None
/// if stdout was empty.
pub(super) async fn execute_hook(
    hook: &ResolvedHook,
    event: HookEvent,
    data: Value,
) -> Result<Option<Value>> {
    let input = HookInput {
        hook: event.as_str().to_string(),
        data,
        config: hook
            .config
            .as_ref()
            .map(serde_json::to_value)
            .transpose()?,
    };

    let stdin_data = serde_json::to_string(&input)?;

    // Resolve command path (relative to base_dir)
    let command_str = &hook.entry.command;
    let (program, args) = parse_command(command_str);
    let program_path = if program.starts_with('/') || program.starts_with('.') {
        let p = PathBuf::from(&program);
        if p.is_relative() {
            hook.base_dir.join(p)
        } else {
            p
        }
    } else {
        PathBuf::from(&program)
    };

    let timeout = hook.entry.timeout_duration();

    // Spawn in a blocking task to avoid blocking the tokio runtime
    let program_path_clone = program_path.clone();
    let hook_name = hook.name.clone();
    let base_dir = hook.base_dir.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut child = Command::new(&program_path_clone)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(std::env::current_dir().unwrap_or(base_dir))
            .spawn()
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to spawn hook '{}' ({}): {}",
                    hook_name,
                    program_path_clone.display(),
                    e
                )
            })?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(stdin_data.as_bytes())?;
        }

        crate::process::wait_with_timeout(&mut child, timeout, &format!("Hook '{}'", hook_name))
    })
    .await??;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        anyhow::bail!(
            "Hook '{}' exited with {}: {}",
            hook.name,
            result.status,
            stderr.trim()
        );
    }

    let stdout = String::from_utf8_lossy(&result.stdout);
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let value: Value = serde_json::from_str(trimmed).map_err(|e| {
        anyhow::anyhow!(
            "Hook '{}' returned invalid JSON: {}: {}",
            hook.name,
            e,
            &trimmed[..trimmed.len().min(200)]
        )
    })?;

    Ok(Some(value))
}

/// Short-circuit search: returns true as soon as any string value matches `re`.
pub(super) fn any_string_matches(value: &Value, re: &Regex) -> bool {
    match value {
        Value::String(s) => re.is_match(s),
        Value::Array(arr) => arr.iter().any(|v| any_string_matches(v, re)),
        Value::Object(obj) => obj.values().any(|v| any_string_matches(v, re)),
        _ => false,
    }
}

/// Split a command string into program and arguments by whitespace.
/// Does not handle quoting — commands with quoted arguments should use
/// a wrapper script instead.
fn parse_command(cmd: &str) -> (String, Vec<String>) {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return (cmd.to_string(), Vec::new());
    }
    let program = parts[0].to_string();
    let args = parts[1..].iter().map(|s| s.to_string()).collect();
    (program, args)
}
