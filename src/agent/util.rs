use std::sync::atomic::{AtomicBool, Ordering};

use crate::message::Message;

/// Maximum number of lines to keep in tool output stored in context.
pub(super) const MAX_TOOL_OUTPUT_LINES: usize = 300;

pub(super) fn truncate_tool_output(output: &str) -> String {
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() <= MAX_TOOL_OUTPUT_LINES {
        return output.to_string();
    }

    let kept: String = lines[..MAX_TOOL_OUTPUT_LINES]
        .iter()
        .flat_map(|l| [*l, "\n"])
        .collect();
    format!(
        "{}... ({} more lines truncated. Refine your command to get more targeted output.)",
        kept,
        lines.len() - MAX_TOOL_OUTPUT_LINES,
    )
}

/// Format messages for the context compaction prompt.
pub(super) fn format_messages_for_compaction(messages: &[Message]) -> String {
    use crate::format::truncate_args;
    use crate::message::Role;
    use std::fmt::Write;

    let mut out = String::new();
    for msg in messages {
        match msg.role {
            Role::User => {
                out.push_str("[User] ");
                out.push_str(&truncate_args(&msg.content, 500));
                out.push('\n');
            }
            Role::Assistant => {
                if !msg.content.is_empty() {
                    out.push_str("[Assistant] ");
                    out.push_str(&truncate_args(&msg.content, 500));
                    out.push('\n');
                }
                if let Some(ref calls) = msg.tool_calls {
                    for tc in calls {
                        let _ = writeln!(
                            out,
                            "[Tool call: {}({})]",
                            tc.function.name,
                            truncate_args(&tc.function.arguments.to_string(), 150),
                        );
                    }
                }
            }
            Role::Tool => {
                let status = match msg.success {
                    Some(true) => "success",
                    Some(false) => "failure",
                    None => "unknown",
                };
                let _ = write!(out, "[Tool result ({})]: ", status);
                out.push_str(&truncate_args(&msg.content, 200));
                out.push('\n');
            }
            Role::System => {
                out.push_str("[System] ");
                out.push_str(&truncate_args(&msg.content, 200));
                out.push('\n');
            }
        }
    }
    out
}

/// Poll a cancel flag, resolving when it becomes true.
pub(super) async fn poll_cancel(flag: &AtomicBool) {
    loop {
        if flag.load(Ordering::Relaxed) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// Exponential backoff delay: 500ms * 2^retries, capped at 2^5 = 32x.
pub(super) fn retry_backoff_delay(retries: u32) -> std::time::Duration {
    const RETRY_BASE_MS: u64 = 500;
    std::time::Duration::from_millis(RETRY_BASE_MS * (1 << retries.min(5)))
}
