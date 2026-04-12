use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::agent::AgentEvent;
use crate::config;
use crate::message::Message;

pub struct Session {
    dir: PathBuf,
    messages_file: File,
    debug_file: File,
}

impl Session {
    pub fn new() -> anyhow::Result<Self> {
        let base = config::data_dir().join("sessions");

        let id = generate_session_id();
        let dir = base.join(&id);
        fs::create_dir_all(&dir)?;

        let messages_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("messages.jsonl"))?;

        let debug_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("debug.log"))?;

        let mut session = Self {
            dir,
            messages_file,
            debug_file,
        };

        session.log_debug("SESSION_START");
        Ok(session)
    }

    pub fn path(&self) -> &Path {
        &self.dir
    }

    pub fn log_message(&mut self, msg: &Message) {
        let ts = format_utc(SystemTime::now());
        // Wrap the message with a timestamp using a simple JSON merge
        if let Ok(mut value) = serde_json::to_value(msg) {
            if let Some(obj) = value.as_object_mut() {
                obj.insert("ts".to_string(), serde_json::Value::String(ts));
            }
            if let Ok(line) = serde_json::to_string(&value) {
                let _ = writeln!(self.messages_file, "{}", line);
                let _ = self.messages_file.flush();
            }
        }
    }

    pub fn log_debug(&mut self, event: &str) {
        let ts = format_utc(SystemTime::now());
        let _ = writeln!(self.debug_file, "[{}] {}", ts, event);
        let _ = self.debug_file.flush();
    }

    pub fn log_agent_event(&mut self, event: &AgentEvent) {
        let line = match event {
            AgentEvent::Token(t) => format!("TOKEN ({} chars)", t.len()),
            AgentEvent::ToolCall { name, args } => format!("TOOL_CALL {}({})", name, args),
            AgentEvent::ToolResult {
                name,
                output,
                success,
            } => {
                format!(
                    "TOOL_RESULT {} success={} ({} chars)",
                    name,
                    success,
                    output.len()
                )
            }
            AgentEvent::ContextUpdate { prompt_tokens } => {
                format!("CONTEXT_UPDATE prompt_tokens={}", prompt_tokens)
            }
            AgentEvent::Done { prompt_tokens } => {
                format!("DONE prompt_tokens={}", prompt_tokens)
            }
            AgentEvent::Error(e) => format!("ERROR {}", e),
            AgentEvent::ContentReplaced(_) => {
                "CONTENT_REPLACED (tool calls extracted from text)".to_string()
            }
            AgentEvent::ToolConfirmRequest { name, args } => {
                format!("TOOL_CONFIRM_REQUEST {}({})", name, args)
            }
            AgentEvent::ContextTrimmed {
                removed_messages,
                estimated_tokens_freed,
            } => {
                format!(
                    "CONTEXT_TRIMMED removed={} freed_est={}",
                    removed_messages, estimated_tokens_freed
                )
            }
            AgentEvent::SubagentStart { task } => {
                format!("SUBAGENT_START task={}", task)
            }
            AgentEvent::SubagentToolCall { name, args } => {
                format!("SUBAGENT_TOOL_CALL {}({})", name, args)
            }
            AgentEvent::SubagentToolResult { name, success } => {
                format!("SUBAGENT_TOOL_RESULT {} success={}", name, success)
            }
            AgentEvent::SubagentEnd { result } => {
                format!("SUBAGENT_END ({} chars)", result.len())
            }
            AgentEvent::MessageLogged(_) => return, // handled separately via log_message
            AgentEvent::Debug(s) => format!("DEBUG {}", s),
        };
        self.log_debug(&line);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.log_debug("SESSION_END");
        let _ = self.messages_file.flush();
        let _ = self.debug_file.flush();
    }
}

fn generate_session_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let nanos = now.subsec_nanos();
    let pid = std::process::id();

    // Simple hash for uniqueness suffix
    let hash = (pid as u64).wrapping_mul(2654435761) ^ (nanos as u64);
    let suffix = format!("{:06x}", hash & 0xFFFFFF);

    let (y, mon, d, h, m, s) = epoch_to_utc(secs);
    format!("{:04}-{:02}-{:02}_{:02}{:02}{:02}_{}", y, mon, d, h, m, s, suffix)
}

/// Format a SystemTime as an ISO 8601-ish UTC string.
fn format_utc(time: SystemTime) -> String {
    let dur = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = dur.as_secs();
    let millis = dur.subsec_millis();
    let (y, mon, d, h, m, s) = epoch_to_utc(secs);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        y, mon, d, h, m, s, millis
    )
}

/// Convert epoch seconds to (year, month, day, hour, minute, second) in UTC.
/// Uses Howard Hinnant's civil_from_days algorithm.
fn epoch_to_utc(epoch_secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (epoch_secs / 86400) as i64;
    let day_secs = (epoch_secs % 86400) as u32;
    let h = day_secs / 3600;
    let m = (day_secs % 3600) / 60;
    let s = day_secs % 60;

    // Civil date from days since epoch (Hinnant algorithm)
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mon = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mon <= 2 { y + 1 } else { y };

    (y, mon, d, h, m, s)
}
