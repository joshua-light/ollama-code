use std::sync::atomic::AtomicBool;

use tokio::sync::mpsc;

use crate::backend::ModelBackend;
use crate::message::{self, Message};

use super::events::{send_event, AgentEvent};
use super::util::{format_messages_for_compaction, poll_cancel};

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
            .position(|m| !matches!(m.role, crate::message::Role::System))
            .unwrap_or(messages.len());

        let mut freed: u64 = 0;
        let mut remove_until = first_non_system;

        let mut i = first_non_system;
        while freed < need_to_free && i < messages.len().saturating_sub(1) {
            freed += messages[i].estimated_tokens();
            i += 1;
            while i < messages.len().saturating_sub(1)
                && !matches!(messages[i].role, crate::message::Role::User)
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

        // Notify that compaction is starting
        let _ = send_event(events, AgentEvent::ContextCompacting);

        // Format the messages to be compacted
        let excerpt = format_messages_for_compaction(&messages[first_non_system..remove_until]);

        // Calculate a word budget (~25% of original size)
        let word_budget = (freed / 4).clamp(50, 500);

        let compaction_messages = vec![
            Message::system(format!(
                "You are a context compaction assistant. Summarize the following conversation \
                 excerpt concisely. This summary replaces the original messages to free context space.\n\
                 \n\
                 PRESERVE: file paths, function/type/variable names, error messages and resolutions, \
                 decisions and rationale, tool call outcomes, current task state.\n\
                 OMIT: full file contents (just note which files were read/modified), verbose tool output, \
                 redundant exchanges.\n\
                 \n\
                 Keep your summary under {} words. Output only the summary.",
                word_budget
            )),
            Message::user(excerpt),
        ];

        // Call the LLM for compaction (no tools, no streaming to user, no
        // thinking — we want a terse summary).
        let compaction_response = tokio::select! {
            r = backend.chat(
                model,
                &compaction_messages,
                None,
                num_ctx,
                None,
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

        let summary_user = Message::user(format!(
            "[Context compaction — summary of {} earlier messages]\n\n{}",
            removed, summary
        ));
        let summary_assistant = Message::assistant(
            "Understood. I have the context from the conversation summary and will continue from here.",
        );

        // Emit MessageLogged so the summary pair gets persisted to the session
        let _ = send_event(events, AgentEvent::MessageLogged(summary_user.clone()));
        let _ = send_event(events, AgentEvent::MessageLogged(summary_assistant.clone()));

        messages.insert(first_non_system, summary_user);
        messages.insert(first_non_system + 1, summary_assistant);

        Some((removed, freed, summary_tokens))
    }
}
