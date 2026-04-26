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
                // Successful tool results are mostly file content / search hits
                // that the preceding assistant tool_call already implies. We
                // collapse them to a one-line outcome line, since the original
                // bodies are gone after compaction anyway. Failures, however,
                // carry the diagnostic — keep those verbatim.
                let status = match msg.success {
                    Some(true) => "success",
                    Some(false) => "failure",
                    None => "unknown",
                };
                if msg.success == Some(false) {
                    let _ = write!(out, "[Tool result ({})]: ", status);
                    out.push_str(&truncate_args(&msg.content, 200));
                    out.push('\n');
                } else {
                    let _ = writeln!(
                        out,
                        "[Tool result ({}, {} bytes)]",
                        status,
                        msg.content.len()
                    );
                }
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

/// Cap on the number of rendered diagnostic blocks (errors/warnings) we
/// surface to the model. Anything beyond is dropped — context-budget friendly,
/// and the first few diagnostics are typically enough to act on.
const MAX_RENDERED_ERRORS: usize = 6;
const MAX_RENDERED_WARNINGS: usize = 4;

/// For diagnostics that lack a stable error code (e.g. "could not compile"),
/// the signature falls back to a prefix of the message text. Char-bounded so
/// multibyte boundaries can't panic.
const UNCODED_ERROR_SIG_PREFIX_CHARS: usize = 60;

/// Sentinel prefix on the auto-injected cargo-check block. We match on this
/// in `tool_executor.rs` to avoid double-injecting when the model's own bash
/// output already contains a previous run's diagnostics.
pub(super) const AUTO_CARGO_CHECK_PREFIX: &str = "[Auto cargo check";

/// Stable identifier for one diagnostic: error code + primary span location.
/// We compare *sets* of these across cargo runs to detect oscillation
/// ("you've fixed nothing in the last N attempts") and regression ("your last
/// edit added new errors"). Using `BTreeSet` keeps comparisons deterministic
/// regardless of the order rustc happens to emit messages.
pub(super) type DiagnosticSig = (String, String, u32); // (code, file, line)

/// One observed cargo run's error signature set. The harness keeps a Vec of
/// these per `run()` to detect when the model is going in circles.
pub(super) type AttemptHistory = Vec<std::collections::BTreeSet<DiagnosticSig>>;

/// Output of one cargo check: the rendered text block (for the model) and
/// the structured signatures (for oscillation analysis).
pub(super) struct CargoDiagnostics {
    pub rendered_block: String,
    pub error_sigs: std::collections::BTreeSet<DiagnosticSig>,
}

/// Run `cargo check --message-format=json` and return both the rendered
/// diagnostics block (with rustc's `help:`/`note:` lines preserved) and the
/// structured per-error signature set. The signature set is what feeds the
/// oscillation detector; the rendered block is what the model reads.
/// Returns None if cargo failed to run, or there are no diagnostics.
pub(super) async fn collect_cargo_diagnostics() -> Option<CargoDiagnostics> {
    if !std::path::Path::new("Cargo.toml").exists() {
        return None;
    }
    let output = tokio::process::Command::new("cargo")
        .args(["check", "--message-format=json", "--quiet"])
        .output()
        .await
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut error_sigs: std::collections::BTreeSet<DiagnosticSig> =
        std::collections::BTreeSet::new();

    for line in stdout.lines() {
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
            continue;
        }
        let msg = match v.get("message") {
            Some(m) => m,
            None => continue,
        };
        let level = msg.get("level").and_then(|l| l.as_str()).unwrap_or("");
        let rendered = match msg.get("rendered").and_then(|r| r.as_str()) {
            Some(r) if !r.is_empty() => r.trim_end().to_string(),
            _ => continue,
        };

        // Extract a stable signature (code + primary span) for each error.
        // No code = uncoded error like "could not compile" — sig'd on the
        // first ~60 chars of the message instead, so we still catch repeats.
        if level == "error" || level == "error: internal compiler error" {
            let code = msg
                .get("code")
                .and_then(|c| c.get("code"))
                .and_then(|c| c.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    let m = msg
                        .get("message")
                        .and_then(|s| s.as_str())
                        .unwrap_or("");
                    let prefix: String = m.chars().take(UNCODED_ERROR_SIG_PREFIX_CHARS).collect();
                    format!("uncoded:{}", prefix)
                });
            let primary = msg
                .get("spans")
                .and_then(|s| s.as_array())
                .and_then(|spans| {
                    spans
                        .iter()
                        .find(|s| s.get("is_primary").and_then(|p| p.as_bool()).unwrap_or(false))
                });
            let (file, line) = primary
                .map(|s| {
                    let f = s
                        .get("file_name")
                        .and_then(|f| f.as_str())
                        .unwrap_or("?")
                        .to_string();
                    let l = s.get("line_start").and_then(|l| l.as_u64()).unwrap_or(0) as u32;
                    (f, l)
                })
                .unwrap_or_else(|| ("?".to_string(), 0));
            error_sigs.insert((code, file, line));
        }

        match level {
            "error" | "error: internal compiler error"
                if errors.len() < MAX_RENDERED_ERRORS =>
            {
                errors.push(rendered);
            }
            "warning" if warnings.len() < MAX_RENDERED_WARNINGS => {
                warnings.push(rendered);
            }
            _ => {}
        }
    }

    if errors.is_empty() && warnings.is_empty() {
        return None;
    }

    let mut out = String::new();
    out.push_str(AUTO_CARGO_CHECK_PREFIX);
    out.push_str(" — full rustc diagnostics with help/note suggestions]\n\n");
    if !errors.is_empty() {
        out.push_str(&format!("── {} error(s) ──\n", errors.len()));
        for e in &errors {
            out.push_str(e);
            out.push_str("\n\n");
        }
    }
    if !warnings.is_empty() {
        out.push_str(&format!("── {} warning(s) ──\n", warnings.len()));
        for w in &warnings {
            out.push_str(w);
            out.push_str("\n\n");
        }
    }
    out.push_str(
        "Note: each block above is rustc's verbatim output including any \
         `help:` and `note:` lines. Read those before retrying — they often \
         specify the exact fix.",
    );
    Some(CargoDiagnostics {
        rendered_block: out,
        error_sigs,
    })
}

/// Compare the latest cargo error signature set against history and return
/// an advisory string when the model is oscillating or regressing. The
/// advisory is meant to be appended to the rendered diagnostics block — it
/// gives the model what *the harness* observed about its own behaviour, which
/// no individual cargo invocation can convey. Returns None when progress
/// looks healthy (or there's not enough history to judge yet).
pub(super) fn analyze_compile_oscillation(
    history: &AttemptHistory,
    new_sigs: &std::collections::BTreeSet<DiagnosticSig>,
) -> Option<String> {
    if history.is_empty() || new_sigs.is_empty() {
        return None;
    }
    let last = history.last().unwrap();

    // Identical to the immediately previous attempt → the most recent edit
    // changed nothing the compiler can see.
    if last == new_sigs {
        return Some(format!(
            "[harness observation] The {} error(s) above are *identical* (same codes, files, \
             and lines) to those after your previous cargo run. The edit you made between \
             these two checks didn't affect the compiler's view. Try a different fix — read \
             the help/note lines literally, or revert and approach from a different angle.",
            new_sigs.len()
        ));
    }

    // Match further back: did we already see this exact set before?
    if let Some(idx) = history.iter().rposition(|s| s == new_sigs) {
        let ago = history.len() - idx;
        return Some(format!(
            "[harness observation] You're looping. The current {}-error set is identical to \
             one you already had {} cargo run(s) ago — the edits you made since then have \
             not made forward progress. Step back and try a fundamentally different approach \
             (e.g. restructure the data flow, extract a helper, or rebind a borrow before \
             reusing it).",
            new_sigs.len(),
            ago
        ));
    }

    // Strict regression vs the immediately previous attempt — new errors
    // appeared while none disappeared.
    if new_sigs.is_superset(last) && new_sigs.len() > last.len() {
        let added: Vec<&DiagnosticSig> = new_sigs.difference(last).collect();
        let preview: Vec<String> = added
            .iter()
            .take(3)
            .map(|(c, f, l)| format!("{} at {}:{}", c, f, l))
            .collect();
        return Some(format!(
            "[harness observation] Regression: your last edit *added* {} new error(s) \
             while fixing none. New: {}. Consider reverting that edit and approaching \
             differently.",
            added.len(),
            preview.join("; ")
        ));
    }

    // Specific spans we've been stuck on for a while: appear in 4+ consecutive
    // attempts. Worth surfacing because the model may not see they're the
    // load-bearing problem.
    if history.len() >= 3 {
        let last_three = &history[history.len().saturating_sub(3)..];
        let stuck: Vec<&DiagnosticSig> = new_sigs
            .iter()
            .filter(|sig| last_three.iter().all(|s| s.contains(sig)))
            .collect();
        if !stuck.is_empty() {
            let preview: Vec<String> = stuck
                .iter()
                .take(3)
                .map(|(c, f, l)| format!("{} at {}:{}", c, f, l))
                .collect();
            return Some(format!(
                "[harness observation] These {} error(s) have survived your last 4 cargo \
                 runs untouched: {}. Whatever you've been editing isn't reaching them — \
                 read those specific spans carefully before your next edit.",
                stuck.len(),
                preview.join("; ")
            ));
        }
    }

    None
}
