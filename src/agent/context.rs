use std::sync::atomic::AtomicBool;

use tokio::sync::mpsc;

use crate::backend::ModelBackend;
use crate::message::{self, Message, Role};

use super::events::{send_event, AgentEvent};
use super::util::{format_messages_for_compaction, poll_cancel};

/// Fraction of `context_size` (in tokens) we keep verbatim at the tail of
/// the conversation, never compacting it. Compaction summarizes only what
/// lies *before* this window. Stops the model from losing recent reasoning
/// or in-progress plan state to a single coarse summary, which was the
/// observed cause of the post-compaction stutter (multi-step plan text in
/// content, no tool calls).
const TAIL_PRESERVE_FRACTION: u64 = 5; // ~20% of context kept fresh

/// Hard floor on the verbatim tail size — keeps at least this many recent
/// tokens uncompacted even on tiny contexts.
const TAIL_PRESERVE_MIN_TOKENS: u64 = 1500;

/// Locate the index in `messages` such that everything from there to the
/// end is kept verbatim. The boundary always lands on a `Role::User` start
/// (or end of list) so we never split an exchange. Returns `messages.len()`
/// when the tail is too small to bother preserving.
fn tail_boundary(messages: &[Message], context_size: u64) -> usize {
    let target = (context_size / TAIL_PRESERVE_FRACTION).max(TAIL_PRESERVE_MIN_TOKENS);
    let mut acc: u64 = 0;
    let mut i = messages.len();
    while i > 0 {
        i -= 1;
        acc += messages[i].estimated_tokens();
        if acc >= target {
            // Walk forward to the next user-message boundary so the kept
            // tail starts on a fresh exchange rather than mid-result.
            while i < messages.len() && !matches!(messages[i].role, Role::User) {
                i += 1;
            }
            return i;
        }
    }
    0
}

/// Scan `messages` from end to start and pull out the most recent block
/// of contiguous `- [ ]` / `- [x]` / `- [-]` checklist lines. The model
/// writes plans using these markers (per system-prompt guidance); the
/// extractor surfaces the latest plan state so it can be pinned across
/// compactions, since otherwise an old plan in compacted history would
/// be summarized away.
fn extract_latest_plan(messages: &[Message]) -> Option<String> {
    fn is_plan_line(line: &str) -> bool {
        let t = line.trim_start();
        t.starts_with("- [ ]")
            || t.starts_with("- [x]")
            || t.starts_with("- [X]")
            || t.starts_with("- [-]")
    }

    for msg in messages.iter().rev() {
        if !matches!(msg.role, Role::Assistant | Role::User) {
            continue;
        }
        let lines: Vec<&str> = msg.content.lines().collect();
        // Find a contiguous block of plan lines (allowing blank lines inside).
        let mut end: Option<usize> = None;
        for (idx, line) in lines.iter().enumerate().rev() {
            if is_plan_line(line) {
                end = Some(idx);
                break;
            }
        }
        let Some(end) = end else { continue };
        let mut start = end;
        while start > 0 {
            let prev = lines[start - 1];
            if is_plan_line(prev) || prev.trim().is_empty() {
                start -= 1;
            } else {
                break;
            }
        }
        // Trim leading blank lines inside the block. The trailing line at
        // `end` is itself a plan line by construction, so the resulting block
        // always contains at least one plan line.
        let block: Vec<&str> = lines[start..=end]
            .iter()
            .copied()
            .skip_while(|l| l.trim().is_empty())
            .collect();
        return Some(block.join("\n"));
    }
    None
}

/// Manages context-window trimming and LLM-based compaction.
pub(super) struct ContextManager {
    pub trim_threshold_pct: u8,
    pub trim_target_pct: u8,
    pub compaction_enabled: bool,
}

impl ContextManager {
    /// Identify a range of old non-system messages to remove, walking forward
    /// through complete exchanges until `need_to_free` tokens are covered.
    /// Returns `(first_non_system, remove_until, estimated_freed)` or `None`.
    fn find_removal_range(
        &self,
        messages: &[Message],
        context_size: u64,
        current_prompt_tokens: u64,
    ) -> Option<(usize, usize, u64)> {
        if context_size == 0 {
            return None;
        }

        let threshold = context_size * self.trim_threshold_pct as u64 / 100;
        if current_prompt_tokens <= threshold {
            return None;
        }

        let target = context_size * self.trim_target_pct as u64 / 100;
        let need_to_free = current_prompt_tokens.saturating_sub(target);

        let first_non_system = messages
            .iter()
            .position(|m| !matches!(m.role, Role::System))
            .unwrap_or(messages.len());

        // Verbatim-tail boundary: never compact past this index. Keeps the
        // model's recent reasoning/edits/tool results intact, which is what
        // drives "I have my bearings, here's the next tool call" continuation
        // instead of post-compaction plan stutter.
        let tail_start = tail_boundary(messages, context_size);
        let upper_bound = tail_start.min(messages.len().saturating_sub(1));

        let mut freed: u64 = 0;
        let mut remove_until = first_non_system;

        let mut i = first_non_system;
        while freed < need_to_free && i < upper_bound {
            freed += messages[i].estimated_tokens();
            i += 1;
            while i < upper_bound
                && !matches!(messages[i].role, Role::User)
            {
                freed += messages[i].estimated_tokens();
                i += 1;
            }
            remove_until = i;
        }

        if remove_until > first_non_system {
            Some((first_non_system, remove_until, freed))
        } else {
            None
        }
    }

    /// Trim oldest non-system messages when context usage exceeds threshold.
    /// Removes complete exchanges (user + assistant + tool messages) as units.
    pub fn trim(
        &self,
        messages: &mut Vec<Message>,
        context_size: u64,
        current_prompt_tokens: u64,
    ) -> Option<(usize, u64)> {
        let (first_non_system, remove_until, freed) =
            self.find_removal_range(messages, context_size, current_prompt_tokens)?;
        let removed = remove_until - first_non_system;
        messages.drain(first_non_system..remove_until);
        Some((removed, freed))
    }

    /// Attempt LLM-based context compaction. Summarizes old exchanges via
    /// the model before removing them, preserving critical context.
    /// Returns `Some((removed, freed, summary_tokens))` on success, or `None`
    /// if below threshold or compaction failed (caller should fall back to
    /// `trim`).
    #[allow(clippy::too_many_arguments)]
    pub async fn compact(
        &self,
        messages: &mut Vec<Message>,
        context_size: u64,
        current_prompt_tokens: u64,
        backend: &dyn ModelBackend,
        model: &str,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &AtomicBool,
        num_ctx: Option<u64>,
    ) -> Option<(usize, u64, u64)> {
        let (first_non_system, remove_until, freed) =
            self.find_removal_range(messages, context_size, current_prompt_tokens)?;

        // Pin the latest plan checklist so it survives compaction even if it
        // appears in the range being summarized. We re-emit it verbatim in
        // the synthetic summary message instead of trusting the summarizer
        // to preserve every checkbox.
        let pinned_plan = extract_latest_plan(messages);

        // Notify that compaction is starting
        let _ = send_event(events, AgentEvent::ContextCompacting);

        // Format the messages to be compacted (tool-result bodies for
        // *successful* calls collapse to one-line stubs in the formatter —
        // the body content is no longer reachable from the post-compaction
        // history anyway, so re-summarizing it is wasted compactor budget).
        let excerpt = format_messages_for_compaction(&messages[first_non_system..remove_until]);

        // Calculate a word budget (~25% of original size)
        let word_budget = (freed / 4).clamp(50, 500);

        let compaction_messages = vec![
            Message::system(format!(
                "You are a context compaction assistant. Summarize the conversation excerpt \
                 below into a structured digest that will replace those messages in the agent's \
                 context.\n\
                 \n\
                 Output exactly this Markdown structure. Omit any section that has no content. \
                 Do not write paragraphs or prose outside these sections.\n\
                 \n\
                 ## Files explored\n\
                 - path/to/foo.rs: 1-line note on what was learned\n\
                 \n\
                 ## Files modified\n\
                 - path/to/foo.rs: nature of edit (e.g. \"added X struct\", \"changed Y signature\")\n\
                 \n\
                 ## Decisions\n\
                 - decision and rationale, in one line each\n\
                 \n\
                 ## Errors hit\n\
                 - error code or message: how it was resolved (or \"unresolved\")\n\
                 \n\
                 ## Other context\n\
                 - any task-specific facts not covered above (one line each)\n\
                 \n\
                 Rules:\n\
                 - Preserve specific identifiers verbatim: file paths, type/function/variable names, \
                 error codes (E0xxx), line numbers.\n\
                 - Omit successful tool-result bodies (they appear only as \"[Tool result (success, N bytes)]\" \
                 stubs in the input — that is intentional).\n\
                 - Total length under {} words across all sections.\n\
                 - Output only the Markdown sections, nothing else.",
                word_budget
            )),
            Message::user(excerpt),
        ];

        // Call the LLM for compaction (no tools, no streaming to user, no
        // thinking). `Some(0)` actively disables reasoning on Ollama-side via
        // `think: false` — without this, models that think unconditionally
        // can spend the whole compactor turn reasoning, hit the budget cap,
        // and return near-empty content, which trips the `summary.is_empty`
        // guard and quietly falls back to blind trim.
        let compaction_response = tokio::select! {
            r = backend.chat(
                model,
                &compaction_messages,
                None,
                num_ctx,
                Some(0),
                Box::new(|_| {}),
            ) => {
                match r {
                    Ok(r) => r,
                    Err(_) => return None,
                }
            }
            _ = poll_cancel(cancel) => {
                return None;
            }
        };

        let summary = compaction_response.content.trim().to_string();
        if summary.is_empty() {
            return None;
        }

        // Guard: if summary is too large relative to freed space, discard it
        let summary_tokens = message::estimate_tokens(summary.len());
        if summary_tokens > freed / 2 {
            return None;
        }

        // Remove old messages and insert the summary exchange
        let removed = remove_until - first_non_system;
        messages.drain(first_non_system..remove_until);

        // Reinject the pinned plan verbatim if we found one — the summarizer
        // gets the structured digest, but the checklist is too important to
        // trust to a re-paraphrasing. Lives on a dedicated section so the
        // model can see at a glance which steps it has done vs has left.
        let summary_body = match pinned_plan {
            Some(plan) => format!("{}\n\n## Active plan\n{}", summary, plan),
            None => summary,
        };

        let summary_user = Message::user(format!(
            "[Context compaction — summary of {} earlier messages]\n\n{}",
            removed, summary_body
        ));
        let summary_assistant = Message::assistant(
            "Understood. Continuing from the summary above; my next action will use the active \
             plan if one is listed, otherwise I'll re-derive the next concrete step.",
        );

        // Emit MessageLogged so the summary pair gets persisted to the session
        let _ = send_event(events, AgentEvent::MessageLogged(summary_user.clone()));
        let _ = send_event(events, AgentEvent::MessageLogged(summary_assistant.clone()));

        messages.insert(first_non_system, summary_user);
        messages.insert(first_non_system + 1, summary_assistant);

        Some((removed, freed, summary_tokens))
    }
}
