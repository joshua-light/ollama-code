use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use tokio::sync::mpsc;

use crate::message::{Message, Role};
use crate::tools::BashTool;

use super::events::{send_event, AgentEvent};
use super::util::{poll_cancel, truncate_tool_output, AUTO_CARGO_CHECK_PREFIX};
use super::Agent;

impl Agent {
    /// Emit the standard error-termination sequence for a tool call:
    /// `ToolResult{success: false}` event, tool message appended to history,
    /// `MessageLogged` event, and skill-inject bookkeeping. Used for the
    /// hook-deny / user-deny / validation-error branches so they stay in sync.
    fn emit_tool_error(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
        name: &str,
        tool_call_id: Option<String>,
        msg: String,
    ) -> Result<()> {
        send_event(
            events,
            AgentEvent::ToolResult {
                name: name.to_string(),
                output: msg.clone(),
                success: false,
            },
        )?;
        let tool_msg = Message::tool(&msg, tool_call_id, false);
        self.messages.push(tool_msg.clone());
        send_event(events, AgentEvent::MessageLogged(tool_msg))?;
        self.record_tool_result(name, false);
        Ok(())
    }

    /// Whether a bash tool call should auto-approve under the configured
    /// safe-prefix whitelist. Empty whitelist = everything prompts as before.
    fn is_bash_safe(&self, args: &serde_json::Value) -> bool {
        if self.bash_safe_prefixes.is_empty() {
            return false;
        }
        let cmd = args
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim_start();
        self.bash_safe_prefixes
            .iter()
            .any(|p| cmd.starts_with(p.as_str()))
    }

    /// If `name` is an edit or write tool and checkpointing is enabled,
    /// snapshot the target file's current contents to the session's
    /// checkpoint dir. First-write-wins: a subsequent edit of the same file
    /// leaves the original snapshot intact so rollback reaches pre-agent state.
    fn maybe_checkpoint(&mut self, name: &str, args: &serde_json::Value) {
        if !matches!(name, "edit" | "write") {
            return;
        }
        let Some(dir) = self.checkpoint_dir.clone() else {
            return;
        };
        let Some(raw_path) = args.get("file_path").and_then(|v| v.as_str()) else {
            return;
        };
        let expanded = crate::tools::expand_tilde(raw_path);
        let path = Path::new(expanded.as_ref());
        // `Write` refuses existing files, so canonicalize failure (non-existent
        // path) just means there's nothing to snapshot — skip silently.
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => return,
        };
        if self.checkpointed_files.contains(&canonical) {
            return;
        }
        let Ok(content) = std::fs::read(&canonical) else {
            return;
        };

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        canonical.hash(&mut h);
        let hash = format!("{:016x}", h.finish());

        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        let backup = dir.join(format!("{}.bak", hash));
        let meta = dir.join(format!("{}.meta.json", hash));
        if std::fs::write(&backup, &content).is_err() {
            return;
        }
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let meta_json = serde_json::json!({
            "original_path": canonical.display().to_string(),
            "timestamp": ts,
            "size": content.len(),
        });
        let _ = std::fs::write(&meta, meta_json.to_string());
        self.checkpointed_files.insert(canonical);
    }

    /// Dispatch a single tool call: run pre-hooks, handle confirmation,
    /// execute the tool (bash / subagent / registry), run post-hooks,
    /// emit events, and append the result message.
    /// Returns `Ok(true)` normally, `Ok(false)` when the user cancelled
    /// mid-execution (caller should return early).
    pub(super) async fn dispatch_tool_call(
        &mut self,
        tool_call: &crate::message::ToolCall,
        events: &mpsc::UnboundedSender<AgentEvent>,
        confirm_rx: &mut mpsc::UnboundedReceiver<bool>,
        cancel: &Arc<AtomicBool>,
    ) -> Result<bool> {
        let mut name = tool_call.function.name.clone();
        let original_args = &tool_call.function.arguments;

        // --- pre_tool_execute hooks ---
        let args_override = match self.hooks.pre_tool_execute(&name, original_args).await {
            Ok(pre) => match pre.action.as_deref() {
                Some("deny") => {
                    let msg = pre
                        .message
                        .unwrap_or_else(|| "Blocked by hook.".to_string());
                    send_event(
                        events,
                        AgentEvent::ToolCall {
                            name: name.clone(),
                            args: crate::format::format_tool_args_display(&name, original_args),
                        },
                    )?;
                    self.emit_tool_error(events, &name, tool_call.id.clone(), msg)?;
                    return Ok(true);
                }
                Some("rewrite") | Some("modify") => {
                    if let Some(ref new_name) = pre.tool_name {
                        send_event(
                            events,
                            AgentEvent::Debug(format!(
                                "Hook rewrote tool call: {} -> {}",
                                name, new_name
                            )),
                        )?;
                        name = new_name.clone();
                    }
                    pre.arguments
                }
                _ => None, // "proceed" or absent
            },
            Err(e) => {
                send_event(
                    events,
                    AgentEvent::Debug(format!("pre_tool_execute hook error: {}", e)),
                )?;
                None
            }
        };
        // Clone-and-coerce: small models routinely emit `"index": "1"` instead
        // of `1`, or `"~/foo"` instead of an expanded path. Both fail
        // validation for nothing — coerce stringly-typed integers/numbers/
        // booleans and expand `~` on path-like fields before validation.
        // Coercion is idempotent on already-correct args.
        let mut args_owned = args_override.unwrap_or_else(|| original_args.clone());
        self.tools.coerce(&name, &mut args_owned);
        let args = &args_owned;

        let args_display = crate::format::format_tool_args_display(&name, args);

        send_event(
            events,
            AgentEvent::ToolCall {
                name: name.clone(),
                args: args_display.clone(),
            },
        )?;

        // Request user confirmation for tools that modify state — but
        // auto-approve bash calls whose command matches a configured safe
        // prefix (read-only commands like `ls`, `rg`, `git status`).
        let mut needs_confirm = matches!(name.as_str(), "bash" | "edit" | "write" | "subagent")
            || self.plugin_confirm_tools.contains(name.as_str());
        if needs_confirm && name == "bash" && self.is_bash_safe(args) {
            needs_confirm = false;
            send_event(
                events,
                AgentEvent::Debug(
                    "bash auto-approved by safe-prefix whitelist".to_string(),
                ),
            )?;
        }
        if needs_confirm {
            send_event(
                events,
                AgentEvent::ToolConfirmRequest {
                    name: name.clone(),
                    args: args_display.clone(),
                },
            )?;

            let approved = confirm_rx.recv().await.unwrap_or(false);
            if !approved {
                let denied = "Tool execution denied by user.".to_string();
                self.emit_tool_error(events, &name, tool_call.id.clone(), denied)?;
                return Ok(true);
            }
        }

        // Validate tool arguments against schema before execution.
        if let Some(Err(validation_err)) = self.tools.validate(&name, args) {
            let msg = format!(
                "Invalid arguments for '{}': {}",
                name, validation_err
            );
            self.emit_tool_error(events, &name, tool_call.id.clone(), msg)?;
            return Ok(true);
        }

        // Destructive-action gate: once the agent has made edits in this run,
        // refuse bash commands that could roll its own work back. The previous
        // qwen run executed `git checkout src/sim/world/terrain_view.rs`
        // post-compaction and wiped 195 lines of in-progress diff. Recovery
        // from a compile error is cheap; recovery from a discarded diff is
        // impossible. Pattern is intentionally narrow — only blocks the
        // commands that have actually destroyed work in past runs. The model
        // can stage/commit/diff freely.
        if name == "bash" && self.any_edits_in_session {
            if let Some(reason) = destructive_command_reason(args) {
                let msg = format!(
                    "[harness] Destructive command refused: {}. You have already \
                     made edits in this run — running this would discard your \
                     own work. If you genuinely need to revert a single edit, \
                     use the `edit` tool to rewrite the file. If you need to \
                     compare against HEAD, use `git diff` (read-only). The \
                     harness blocks: git checkout/restore/reset --hard, \
                     git clean -f, and rm -rf on tracked paths after any edit.",
                    reason
                );
                self.emit_tool_error(events, &name, tool_call.id.clone(), msg)?;
                return Ok(true);
            }
        }

        // Short-circuit immediate repeats. A small model that gets stuck in a
        // loop will keep emitting the same tool call. Re-running it just burns
        // turns; injecting a corrective synthetic result breaks the cycle.
        // The corrective message escalates: a soft warning the first time, a
        // stronger system-level intervention the second, a hard refusal the
        // third+. Smaller models have been observed to re-issue the same
        // read/grep call several times despite a single warning.
        // `serde_json::Map` is `BTreeMap`-backed (no `preserve_order` feature),
        // so `to_string` produces a key-sorted, stable canonical form.
        let canonical_args = serde_json::to_string(args).unwrap_or_default();
        if let Some((last_name, last_args)) = self.last_tool_signature.as_ref() {
            if last_name == &name && last_args == &canonical_args {
                self.consecutive_repeat_count =
                    self.consecutive_repeat_count.saturating_add(1);
                let count = self.consecutive_repeat_count;
                let tool_msg = match count {
                    1 => format!(
                        "[auto] You just called {}({}) with identical arguments. \
                         The result was the same. Repeating won't help — try a \
                         different approach (different args, a different tool, or \
                         commit to a final answer).",
                        name, args_display
                    ),
                    2 => format!(
                        "[auto] {}({}) — that's the third identical call in a row. \
                         Stop. You either have enough context to act (call edit / \
                         write / bash, or finish), or you need different arguments. \
                         Repeating the same call will not change the result.",
                        name, args_display
                    ),
                    _ => format!(
                        "[auto] HARD REFUSAL: {}({}) — call #{} in a row with \
                         identical arguments. The harness will not execute this \
                         again until you change tool or arguments. Pick edit / \
                         write / bash, or emit a final answer.",
                        name,
                        args_display,
                        count + 1,
                    ),
                };
                self.emit_tool_error(events, &name, tool_call.id.clone(), tool_msg)?;
                // On the second escalation, also inject a system message —
                // tool errors live at Role::Tool and the model has demonstrably
                // ignored them. A Role::System message carries more weight.
                if count == 2 {
                    // User role: chat templates that gate system messages to
                    // index 0 (Qwen3, Qwen3.5, …) reject mid-conversation
                    // system inserts. The harness escalation still lands as a
                    // strong signal as a user-authored line.
                    let nudge = Message::user(
                        "[harness] You have been calling the same tool with the \
                         same arguments repeatedly despite warnings. You have \
                         enough context. Take a concrete action: call `edit`, \
                         `write`, or `bash` — or emit a final answer. Do not \
                         re-read or re-grep the same target.",
                    );
                    self.messages.push(nudge.clone());
                    send_event(events, AgentEvent::MessageLogged(nudge))?;
                }
                // Keep last_tool_signature unchanged so the next repeat is also caught.
                return Ok(true);
            }
        }
        self.last_tool_signature = Some((name.clone(), canonical_args));
        self.consecutive_repeat_count = 0;

        // Check cancellation before each tool call.
        // NOTE: callers rely on no tool result message being pushed when we
        // return Ok(false), so they can backfill "Cancelled" results for
        // this tool call and all remaining ones using tool_calls_with_ids[i..].
        if cancel.load(Ordering::Relaxed) {
            send_event(events, AgentEvent::Cancelled)?;
            return Ok(false);
        }

        // Snapshot the target file before any edit/write so we can roll back.
        self.maybe_checkpoint(&name, args);

        let (mut result, mut success) = if name == "bash" {
            // Set up a channel for streaming partial output from the bash
            // process so the TUI can show live progress.
            let (output_tx, mut output_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
            let events_clone = events.clone();
            let forward_task = tokio::spawn(async move {
                while let Some(line) = output_rx.recv().await {
                    let _ = send_event(&events_clone, AgentEvent::ToolOutput { output: line });
                }
            });

            // Race tool execution against the cancel flag so ESC kills
            // long-running bash commands immediately.
            let output = tokio::select! {
                output = BashTool.execute_async(args, self.bash_timeout, Some(&output_tx)) => {
                    Some(output)
                },
                _ = poll_cancel(cancel) => None,
            };

            // Clean up forwarding on both paths.
            drop(output_tx);
            let _ = forward_task.await;

            match output {
                Some(o) => o,
                None => {
                    // NOTE: callers rely on no tool result message being pushed
                    // when we return Ok(false), so they can backfill "Cancelled"
                    // results starting from this tool call's index.
                    send_event(events, AgentEvent::Cancelled)?;
                    return Ok(false);
                }
            }
        } else if name == "subagent" {
            let task = args
                .get("task")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            // Pre-read any files the parent passed in `files`. Local models
            // burn ~20 turns of warmup just doing read/glob before they edit.
            // If the parent already knows which file the sub-agent needs to
            // touch, we read it here and prepend the body — the sub-agent
            // starts with code in hand.
            let files: Vec<String> = args
                .get("files")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let preloaded = preload_files(&files);
            let task_with_files = if preloaded.is_empty() {
                task
            } else {
                format!("{}\n\n{}", task, preloaded)
            };
            self.execute_subagent(&task_with_files, events, confirm_rx, cancel).await
        } else {
            match self.tools.execute(&name, args) {
                Ok(output) => (output, true),
                Err(e) => {
                    let err_str = format!("Error: {}", e);
                    // Persist failed edits to the audit JSONL — see
                    // `edit_audit::log_edit_failure` for the schema.
                    if name == "edit" {
                        super::edit_audit::log_edit_failure(args, &err_str, None);
                    }
                    (err_str, false)
                }
            }
        };

        // Track exploration for dynamic tool scoping
        if success && matches!(name.as_str(), "read" | "glob" | "grep") {
            self.has_explored = true;
        }

        // Track file reads for subagent context injection.
        // Also rewrite prior overlapping read tool-results in history to a
        // short stub. The current full content stays at the most recent
        // position, so the model always sees fresh data "now" and prior
        // copies of the same range stop competing for attention.
        if success && name == "read" {
            if let Some(path) = args.get("file_path").and_then(|v| v.as_str()) {
                let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
                let start = offset.max(1) - 1;
                self.read_file_ranges.push((path.to_string(), start, start + limit));
                let expanded = crate::tools::expand_tilde(path).into_owned();
                self.rewrite_superseded_reads(&expanded, start, start + limit);
            }
        }

        if success && matches!(name.as_str(), "edit" | "write") {
            self.had_edits_this_run = true;
            self.any_edits_in_session = true;
        }

        // If the model ran a cargo command itself (typically piped through
        // `head`/`grep` that strips rustc's `help:`/`note:` lines) and the
        // surviving output mentions errors, append the full structured
        // diagnostics. Skip when no error markers are present or when the
        // output already looks like the rich format we'd produce.
        if name == "bash"
            && bash_command_runs_cargo(args)
            && bash_output_has_rustc_errors(&result)
            && !result.contains(AUTO_CARGO_CHECK_PREFIX)
        {
            if let Some(block) = self.check_and_record_cargo_diagnostics().await {
                result = format!("{}\n\n{}", result, block);
            }
        }

        // --- post_tool_execute hooks ---
        let (post_result, post_success) = self
            .hooks
            .post_tool_execute(&name, args, result, success)
            .await;
        result = post_result;
        success = post_success;

        send_event(
            events,
            AgentEvent::ToolResult {
                name: name.clone(),
                output: result.clone(),
                success,
            },
        )?;

        // Truncate before storing in context to avoid blowing the window.
        // The full output was already sent to the UI above.
        let context_result = truncate_tool_output(&result);
        let tool_msg = Message::tool(&context_result, tool_call.id.clone(), success);
        self.messages.push(tool_msg.clone());
        send_event(events, AgentEvent::MessageLogged(tool_msg))?;
        self.record_tool_result(&name, success);

        // After a successful edit/write, stub out prior FAILED edit/write
        // tool-results for the same file. The model on its next turn no
        // longer sees its own dead-end attempts — only the eventually
        // successful state.
        if success && matches!(name.as_str(), "edit" | "write") {
            if let Some(raw_path) = args.get("file_path").and_then(|v| v.as_str()) {
                let expanded = crate::tools::expand_tilde(raw_path).into_owned();
                self.rewrite_superseded_edits(&expanded);
            }
        }

        // Delegation nudge: when the model marks a "Delegate <change> in
        // <file> to a sub-agent ..." step in_progress, it routinely ignores
        // the planner's instruction and starts reading files inline. The
        // planner already mapped the file; the sub-agent will pre-load it.
        // Inject a synthetic user-role message so the next turn sees a
        // hard nudge ordering it to call `subagent` next. User-role rather
        // than system: Qwen3/3.5 chat templates reject mid-conversation
        // system messages (see comment further up about repeat-call escalations).
        if name == "plan_mark_in_progress" && success {
            if let Some(idx) = args.get("index").and_then(|v| v.as_u64()) {
                let idx = idx as usize;
                let desc = {
                    let list = self.plan.lock().expect("plan lock poisoned");
                    list.steps.get(idx).map(|s| s.description.clone())
                };
                if let Some(desc) = desc {
                    if desc.trim_start().to_lowercase().starts_with("delegate") {
                        let truncated: String = desc.chars().take(200).collect();
                        let extracted = extract_file_paths_from_step(&desc);
                        let files_hint = if extracted.is_empty() {
                            "extract the path(s) yourself from the step description above.".to_string()
                        } else {
                            format!("files=[{}]", extracted
                                .iter()
                                .map(|p| format!("\"{}\"", p))
                                .collect::<Vec<_>>()
                                .join(", "))
                        };
                        let nudge_text = format!(
                            "[harness] Step {} is a delegation step (\"{}\"). Your next \
                             tool call MUST be `subagent` with `files` populated from the \
                             path(s) named in the step description. Do NOT call `read`, \
                             `glob`, `grep`, or `edit` directly from here — the planner \
                             has already mapped the codebase, and the sub-agent will \
                             receive the file body pre-loaded. Extract the file path(s) \
                             from the step description (between \"in \" and \" to a \
                             sub-agent\" if present) and pass them in `files=[...]`. \
                             Suggested call: {}. The sub-agent will return when it's \
                             done; you then call `plan_mark_done({})`.",
                            idx, truncated, files_hint, idx
                        );
                        let nudge = Message::user(nudge_text);
                        self.messages.push(nudge.clone());
                        send_event(events, AgentEvent::MessageLogged(nudge))?;
                    }
                }
            }
        }

        Ok(true)
    }
}

/// Extract candidate file paths from a delegation-step description. The planner
/// emits steps like `Delegate <change> in <file> to a sub-agent (subagent tool,
/// files=[<path>]) — invariant: ... — risk: ...`, so we look in two places:
/// 1. A `files=[...]` array (most reliable when the planner included it).
/// 2. A bare `in <path> to a sub-agent` form, where `<path>` is whatever sits
///    between `in ` and ` to a sub-agent`.
/// Heuristic only — if neither pattern matches we return an empty vec and the
/// caller falls back to telling the model to extract the path itself.
fn extract_file_paths_from_step(description: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let lc = description.to_lowercase();

    // Pattern 1: files=[a, b, c]
    if let Some(start) = lc.find("files=[") {
        // Byte positions from `lc` align with `description` for ASCII content,
        // and planner-emitted file paths are ASCII.
        let after = &description[start + "files=[".len()..];
        if let Some(end) = after.find(']') {
            let inner = &after[..end];
            for tok in inner.split(',') {
                let cleaned = tok.trim().trim_matches('"').trim_matches('\'').trim();
                if !cleaned.is_empty() {
                    out.push(cleaned.to_string());
                }
            }
        }
    }

    // Pattern 2: `in <path> to a sub-agent`
    if out.is_empty() {
        if let Some(start) = lc.find(" in ") {
            let after = &description[start + " in ".len()..];
            let after_lc = &lc[start + " in ".len()..];
            if let Some(end) = after_lc.find(" to a sub-agent") {
                let path = after[..end].trim().trim_end_matches(',').trim();
                if !path.is_empty() {
                    out.push(path.to_string());
                }
            }
        }
    }

    out
}

/// Sentinel prefix on a stubbed-out `read` tool-result. Both written by
/// `rewrite_superseded_reads` and matched there to skip already-stubbed
/// messages on subsequent rewrites.
const SUPERSEDED_PREFIX: &str = "[superseded]";

/// Sentinel prefix on a stubbed-out failed-edit tool-result. Mirror of
/// SUPERSEDED_PREFIX, distinct so we can run both rewriters without churn.
const EDIT_SUPERSEDED_PREFIX: &str = "[edit superseded]";

impl Agent {
    /// Replace the bodies of prior `read` tool-result messages whose request
    /// range is fully covered by the just-completed read. The full content
    /// of the new read stays at its current position; older copies become a
    /// one-line stub. Token cost stays bounded — only one full copy of any
    /// given range lives in context at a time. We compare on the request
    /// args (path/offset/limit), not on returned content, because a prior
    /// read of `[1, 200)` is logically superseded by a later read of
    /// `[1, 500)` even if the file was only 50 lines.
    fn rewrite_superseded_reads(&mut self, new_path: &str, new_start: usize, new_end: usize) {
        // tool_call_id -> (expanded_path, start, end) for every `read` call
        // we can still see in history. Walk assistant messages once.
        let mut index: std::collections::HashMap<String, (String, usize, usize)> =
            std::collections::HashMap::new();
        for msg in &self.messages {
            if !matches!(msg.role, Role::Assistant) {
                continue;
            }
            let Some(ref calls) = msg.tool_calls else {
                continue;
            };
            for tc in calls {
                if tc.function.name != "read" {
                    continue;
                }
                let Some(id) = tc.id.clone() else { continue; };
                let args = &tc.function.arguments;
                let Some(raw_path) = args.get("file_path").and_then(|v| v.as_str()) else {
                    continue;
                };
                let path = crate::tools::expand_tilde(raw_path).into_owned();
                let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
                let start = offset.saturating_sub(1);
                let end = start.saturating_add(limit);
                index.insert(id, (path, start, end));
            }
        }

        for msg in self.messages.iter_mut() {
            if !matches!(msg.role, Role::Tool) {
                continue;
            }
            if msg.content.starts_with(SUPERSEDED_PREFIX) {
                continue;
            }
            let Some(id) = msg.tool_call_id.as_ref() else {
                continue;
            };
            let Some((path, start, end)) = index.get(id) else {
                continue;
            };
            if path != new_path {
                continue;
            }
            // Rewrite when the new read fully covers the prior request.
            if *start >= new_start && *end <= new_end {
                msg.content = format!(
                    "{} Lines {}-{} of '{}' — replaced by a later read of an overlapping range; refer to that result.",
                    SUPERSEDED_PREFIX,
                    start + 1,
                    end,
                    path,
                );
            }
        }
    }

    /// Replace the bodies of prior FAILED `edit`/`write` tool-result messages
    /// for the same `file_path`. Once a later edit succeeds, the model on its
    /// next turn shouldn't re-read its own failed attempts — they're noise
    /// that competes with the actual current state of the file.
    ///
    /// Defensive style: skips silently on any unexpected message shape, never
    /// panics, idempotent across repeated calls.
    pub(super) fn rewrite_superseded_edits(&mut self, file_path: &str) {
        // tool_call_id -> (raw_path) for every edit/write call we can still
        // see in history.
        let mut index: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for msg in &self.messages {
            if !matches!(msg.role, Role::Assistant) {
                continue;
            }
            let Some(ref calls) = msg.tool_calls else {
                continue;
            };
            for tc in calls {
                if !matches!(tc.function.name.as_str(), "edit" | "write") {
                    continue;
                }
                let Some(id) = tc.id.clone() else { continue; };
                let args = &tc.function.arguments;
                let Some(raw_path) = args.get("file_path").and_then(|v| v.as_str()) else {
                    continue;
                };
                let path = crate::tools::expand_tilde(raw_path).into_owned();
                index.insert(id, path);
            }
        }

        for msg in self.messages.iter_mut() {
            if !matches!(msg.role, Role::Tool) {
                continue;
            }
            if msg.success != Some(false) {
                continue;
            }
            if msg.content.starts_with(EDIT_SUPERSEDED_PREFIX) {
                continue;
            }
            let Some(id) = msg.tool_call_id.as_ref() else {
                continue;
            };
            let Some(path) = index.get(id) else {
                continue;
            };
            if path != file_path {
                continue;
            }
            msg.content = format!(
                "{} An earlier edit attempt on {} failed; a later edit on the same file succeeded. The file's current state reflects the successful edits — refer to those.",
                EDIT_SUPERSEDED_PREFIX, path,
            );
        }
    }
}

/// True if the bash command string starts with (or pipes through) a `cargo`
/// invocation that exercises rustc — so a follow-up `cargo check` would
/// surface diagnostics. Cheap string scan; false positives are harmless
/// (the diagnostic re-run is a no-op on a clean tree).
fn bash_command_runs_cargo(args: &serde_json::Value) -> bool {
    let cmd = args
        .get("command")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    // Strip leading `cd …; ` / `cd …&&` so common shell wrappers don't hide
    // the cargo call. We only need a coarse signal.
    let scan = cmd.replace("&&", " ").replace(';', " ");
    for token in ["cargo build", "cargo check", "cargo test", "cargo clippy", "cargo run"] {
        if scan.contains(token) {
            return true;
        }
    }
    false
}

fn bash_output_has_rustc_errors(output: &str) -> bool {
    output.contains("error[E") || output.contains("error: ") || output.contains("could not compile")
}

/// Read each file path and format as a single Markdown block ready to prepend
/// to a sub-agent task. Each file gets a fenced section with the language
/// inferred from extension and 1-based line numbers — matching what `read`
/// produces — so the sub-agent can reference line numbers in its edits.
///
/// Failures (missing file, IO error) are folded into the output as a
/// `[error: ...]` block so the sub-agent at least sees that the parent meant
/// to provide it but couldn't, rather than silently dropping the file.
///
/// Truncation: each file is capped at 2000 lines to bound the prepended
/// context. Anything longer is cut with a marker so the sub-agent knows to
/// `read` the rest. 2000 lines ≈ 25–35 KB of source ≈ ~10K tokens, which is
/// the budget where local-model degradation starts to bite.
fn preload_files(paths: &[String]) -> String {
    if paths.is_empty() {
        return String::new();
    }
    const MAX_LINES_PER_FILE: usize = 2000;
    let mut out = String::from("--- Pre-loaded file contents (parent agent provided these for you) ---\n\n");
    for raw in paths {
        let expanded = crate::tools::expand_tilde(raw);
        let path = std::path::Path::new(expanded.as_ref());
        out.push_str(&format!("### {}\n\n", raw));
        let lang = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| match e {
                "rs" => "rust",
                "ts" | "tsx" => "typescript",
                "js" | "jsx" => "javascript",
                "py" => "python",
                "toml" => "toml",
                "md" => "markdown",
                "json" => "json",
                _ => e,
            })
            .unwrap_or("");
        match std::fs::read_to_string(path) {
            Ok(content) => {
                out.push_str("```");
                out.push_str(lang);
                out.push('\n');
                let lines: Vec<&str> = content.lines().collect();
                let total = lines.len();
                let take = total.min(MAX_LINES_PER_FILE);
                for (i, line) in lines.iter().take(take).enumerate() {
                    out.push_str(&format!("{:>4}\t{}\n", i + 1, line));
                }
                if total > take {
                    out.push_str(&format!(
                        "... ({} more lines truncated; read {} with offset to see them)\n",
                        total - take,
                        raw
                    ));
                }
                out.push_str("```\n\n");
            }
            Err(e) => {
                out.push_str(&format!("[error: could not read {}: {}]\n\n", raw, e));
            }
        }
    }
    out.push_str("--- End of pre-loaded file contents ---\n");
    out
}

/// If the bash command would destroy uncommitted edits, return a short
/// human-readable reason. None when the command is safe.
///
/// Matches against tokenised words on the command string (split on whitespace,
/// `;`, `&&`, `|`) so embedded sub-commands like `cd foo && git checkout .`
/// are still caught. Conservative: only matches the patterns that have been
/// observed to wipe agent work. `git diff`, `git status`, `git log`, `git
/// stash` (which preserves the diff), `git add`, and `git commit` all pass.
fn destructive_command_reason(args: &serde_json::Value) -> Option<&'static str> {
    let cmd = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
    if cmd.is_empty() {
        return None;
    }
    // Split into shell-clause-ish tokens so `cd foo && git checkout .` still
    // matches `git checkout` in the second clause.
    let normalised = cmd
        .replace("&&", " && ")
        .replace("||", " || ")
        .replace(';', " ; ");
    let lc = normalised.to_lowercase();

    // git checkout <pathspec> — discards working-tree changes for those paths.
    // `git checkout -b <branch>` and `git checkout <branch>` (no pathspec)
    // are safe; we only block when a pathspec follows. Detect by looking for
    // `git checkout` followed by anything that isn't `-b`, `--`, a flag, or
    // a bare branch name. Cheap heuristic: if "git checkout" appears AND the
    // next non-flag token contains `/`, `.`, or `*`, treat as destructive.
    if lc.contains("git checkout") {
        // Find the position and look at what follows.
        if let Some(rest) = lc.split("git checkout").nth(1) {
            let next = rest.split_whitespace().find(|t| !t.starts_with('-'));
            match next {
                None => {} // bare `git checkout` — let it through
                Some(t) if t == "--" => {
                    // `git checkout -- <path>` — definitely destructive
                    return Some("git checkout -- discards working-tree changes");
                }
                Some(t) if t.contains('/') || t.contains('.') || t.contains('*') => {
                    return Some("git checkout <pathspec> discards working-tree changes for those paths");
                }
                _ => {}
            }
        }
    }
    if lc.contains("git restore") {
        return Some("git restore discards working-tree changes");
    }
    if lc.contains("git reset --hard") || lc.contains("git reset hard") {
        return Some("git reset --hard discards working-tree changes");
    }
    if lc.contains("git clean -f") || lc.contains("git clean -fd") || lc.contains("git clean --force") {
        return Some("git clean -f deletes untracked files");
    }
    // rm -rf on a tracked path. Allow common temp/build dirs (target/, dist/,
    // node_modules/, /tmp/) since they contain no agent edits.
    if lc.contains("rm -rf") || lc.contains("rm -fr") {
        let after = lc.split("rm -").nth(1).unwrap_or("");
        let safe_targets = ["target/", "target ", "dist/", "dist ", "node_modules", "/tmp/", "build/", "build "];
        let looks_safe = safe_targets.iter().any(|s| after.contains(s));
        if !looks_safe {
            return Some("rm -rf on a non-temp path can delete tracked files");
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn check(cmd: &str) -> Option<&'static str> {
        destructive_command_reason(&json!({ "command": cmd }))
    }

    #[test]
    fn safe_git_commands_pass() {
        assert!(check("git status").is_none());
        assert!(check("git diff src/foo.rs").is_none());
        assert!(check("git log --oneline -5").is_none());
        assert!(check("git add src/foo.rs && git commit -m msg").is_none());
        assert!(check("git stash").is_none());
        assert!(check("git checkout -b feature").is_none());
        assert!(check("git checkout main").is_none());
    }

    #[test]
    fn destructive_git_blocked() {
        assert!(check("git checkout src/foo.rs").is_some());
        assert!(check("git checkout -- src/foo.rs").is_some());
        assert!(check("git checkout .").is_some());
        assert!(check("git checkout *").is_some());
        assert!(check("git restore src/foo.rs").is_some());
        assert!(check("git reset --hard HEAD").is_some());
        assert!(check("git clean -fd").is_some());
        assert!(check("cd ash && git checkout src/foo.rs").is_some());
    }

    #[test]
    fn rm_rf_temp_paths_allowed() {
        assert!(check("rm -rf target/").is_none());
        assert!(check("rm -rf node_modules").is_none());
        assert!(check("rm -rf /tmp/foo").is_none());
    }

    #[test]
    fn rm_rf_real_paths_blocked() {
        assert!(check("rm -rf src/").is_some());
        assert!(check("rm -rf .").is_some());
    }

    #[test]
    fn extract_paths_from_typical_delegation_step() {
        let desc = "Delegate apply_writes change in src/sim/world/terrain_view.rs to a sub-agent (subagent tool, files=[src/sim/world/terrain_view.rs]) — invariant: idempotent — risk: low";
        let paths = extract_file_paths_from_step(desc);
        assert_eq!(paths, vec!["src/sim/world/terrain_view.rs".to_string()]);
    }

    #[test]
    fn extract_paths_from_files_array_form() {
        let desc = "Delegate refactor in foo.rs to a sub-agent (subagent tool, files=[src/foo.rs, src/bar.rs]) — invariant: x — risk: y";
        let paths = extract_file_paths_from_step(desc);
        assert_eq!(
            paths,
            vec!["src/foo.rs".to_string(), "src/bar.rs".to_string()]
        );
    }

    #[test]
    fn extract_paths_returns_empty_for_no_path() {
        let desc = "Read the project README and summarise the architecture";
        let paths = extract_file_paths_from_step(desc);
        assert!(paths.is_empty());
    }

    // ---------- rewrite_superseded_edits ----------

    use crate::backend::{ChatResponse, ModelBackend};
    use crate::message::{FunctionCall, ToolCall};
    use serde_json::Value;
    use std::future::Future;
    use std::pin::Pin;

    /// Stub backend — tested method is pure on `self.messages`, never calls
    /// the backend. Mirror of the stub in `agent::subagent::tests`.
    struct StubBackend;

    impl ModelBackend for StubBackend {
        fn chat<'a>(
            &'a self,
            _model: &'a str,
            _messages: &'a [Message],
            _tools: Option<Vec<Value>>,
            _num_ctx: Option<u64>,
            _thinking_budget_tokens: Option<u64>,
            _on_token: Box<dyn Fn(&str) + Send + 'a>,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<ChatResponse>> + Send + 'a>>
        {
            Box::pin(async { unreachable!("stub backend should not be called in this test") })
        }
    }

    fn make_agent() -> Agent {
        Agent::new(
            std::sync::Arc::new(StubBackend),
            "stub-model".to_string(),
            0,
            std::time::Duration::from_secs(1),
            0,
        )
    }

    /// Build an assistant message that requests an edit on `path` with the
    /// given tool-call id.
    fn assistant_edit_call(id: &str, path: &str) -> Message {
        let mut msg = Message::assistant("");
        msg.tool_calls = Some(vec![ToolCall {
            id: Some(id.to_string()),
            call_type: Some("function".to_string()),
            function: FunctionCall {
                name: "edit".to_string(),
                arguments: json!({
                    "file_path": path,
                    "old_string": "x",
                    "new_string": "y",
                }),
            },
        }]);
        msg
    }

    #[test]
    fn test_rewrite_superseded_edits_stubs_failed_messages() {
        let mut agent = make_agent();
        agent.messages.push(assistant_edit_call("call_a", "foo.rs"));
        agent.messages.push(Message::tool(
            "Error: old_string not found",
            Some("call_a".to_string()),
            false,
        ));
        agent.messages.push(assistant_edit_call("call_b", "foo.rs"));
        agent.messages.push(Message::tool(
            "diff: ok",
            Some("call_b".to_string()),
            true,
        ));

        agent.rewrite_superseded_edits("foo.rs");

        // The failed message body is stubbed.
        let failed = agent
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("call_a"))
            .expect("failed tool msg");
        assert!(
            failed.content.starts_with("[edit superseded]"),
            "expected stub, got: {}",
            failed.content
        );
        assert!(failed.content.contains("foo.rs"));

        // Successful message is untouched.
        let succeeded = agent
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("call_b"))
            .expect("success tool msg");
        assert_eq!(succeeded.content, "diff: ok");
    }

    #[test]
    fn test_rewrite_idempotent() {
        let mut agent = make_agent();
        agent.messages.push(assistant_edit_call("call_a", "foo.rs"));
        agent.messages.push(Message::tool(
            "Error: old_string not found",
            Some("call_a".to_string()),
            false,
        ));
        agent.rewrite_superseded_edits("foo.rs");
        let stubbed_once = agent
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("call_a"))
            .unwrap()
            .content
            .clone();
        agent.rewrite_superseded_edits("foo.rs");
        let stubbed_twice = agent
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("call_a"))
            .unwrap()
            .content
            .clone();
        assert_eq!(stubbed_once, stubbed_twice);
    }

    #[test]
    fn test_rewrite_skips_different_file() {
        let mut agent = make_agent();
        agent.messages.push(assistant_edit_call("call_a", "bar.rs"));
        agent.messages.push(Message::tool(
            "Error: old_string not found",
            Some("call_a".to_string()),
            false,
        ));
        agent.rewrite_superseded_edits("foo.rs");
        let unrelated = agent
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("call_a"))
            .unwrap();
        assert_eq!(unrelated.content, "Error: old_string not found");
    }
}
