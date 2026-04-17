use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;
use tokio::sync::mpsc;

use super::events::{send_event, AgentEvent};
use super::Agent;

impl Agent {
    /// Return the last assistant message content, or a fallback string.
    pub(super) fn last_assistant_message(&self) -> String {
        self.messages
            .iter()
            .rev()
            .find_map(|m| {
                if matches!(m.role, crate::message::Role::Assistant)
                    && !m.content.trim().is_empty()
                {
                    Some(m.content.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "Sub-agent produced no response.".to_string())
    }

    /// Build a file-context preamble from the parent's read history.
    fn build_file_context_preamble(&self) -> String {
        if self.read_file_ranges.is_empty() {
            return String::new();
        }

        // Deduplicate: keep only unique paths with their widest range.
        let mut file_ranges: std::collections::HashMap<&str, (usize, usize)> =
            std::collections::HashMap::new();
        for (path, start, end) in &self.read_file_ranges {
            let entry = file_ranges.entry(path.as_str()).or_insert((*start, *end));
            entry.0 = entry.0.min(*start);
            entry.1 = entry.1.max(*end);
        }

        let mut preamble = String::from(
            "\n\n--- Parent agent file context ---\n\
             The parent agent has already read these files. You may reference this \
             information without re-reading:\n",
        );
        for (path, (start, end)) in &file_ranges {
            preamble.push_str(&format!("  - {} (lines {}-{})\n", path, start + 1, end));
        }
        preamble.push_str("---\n");
        preamble
    }

    /// Check if a task description contains action verbs that imply code changes.
    fn task_expects_edits(task: &str) -> bool {
        let lower = task.to_lowercase();
        ["implement", "edit", "modify", "update", "change", "fix", "add", "remove", "replace",
         "refactor", "rewrite", "write"]
            .iter()
            .any(|verb| lower.contains(verb))
    }

    /// Execute a sub-agent for the given `task`, forwarding its events and
    /// tool-confirmation requests to the parent's channels.
    /// Returns `(result_text, success)`.
    pub(super) async fn execute_subagent(
        &self,
        task: &str,
        events: &mpsc::UnboundedSender<AgentEvent>,
        confirm_rx: &mut mpsc::UnboundedReceiver<bool>,
        cancel: &Arc<AtomicBool>,
    ) -> (String, bool) {
        // Build enriched task with parent file context
        let preamble = self.build_file_context_preamble();
        let enriched_task = if preamble.is_empty() {
            task.to_string()
        } else {
            format!("{}{}", task, preamble)
        };

        let result = self
            .run_subagent_once(&enriched_task, events, confirm_rx, cancel)
            .await;

        // Task enforcement: if the task expected edits but the subagent made none, retry once.
        // The subagent tracks had_edits_this_run internally but we can't access it from
        // here, so sniff the response for edit-related keywords as a proxy.
        if result.1 && Self::task_expects_edits(task) {
            let response = &result.0;
            let likely_no_edits = !response.is_empty()
                && !response.contains("edit")
                && !response.contains("modified")
                && !response.contains("updated")
                && !response.contains("changed")
                && !response.contains("wrote")
                && !response.contains("replaced");

            if likely_no_edits {
                let retry_task = format!(
                    "{}\n\n\
                     IMPORTANT: You were asked to make code changes but your previous attempt \
                     completed without editing any files. Please re-read the task above and \
                     make the requested edits using the edit or write tool. Do not just \
                     explore or summarize — actually make the changes.",
                    enriched_task
                );
                return self
                    .run_subagent_once(&retry_task, events, confirm_rx, cancel)
                    .await;
            }
        }

        result
    }

    /// Run a single subagent instance and return its result.
    async fn run_subagent_once(
        &self,
        task: &str,
        events: &mpsc::UnboundedSender<AgentEvent>,
        confirm_rx: &mut mpsc::UnboundedReceiver<bool>,
        cancel: &Arc<AtomicBool>,
    ) -> (String, bool) {
        send_event(events, AgentEvent::SubagentStart { task: task.to_string() })
            .unwrap_or(());

        let sub_agent = Agent::new_subagent(
            Arc::clone(&self.backend),
            self.model.clone(),
            self.context_size,
            self.bash_timeout,
            self.subagent_max_turns,
            self.config.as_ref(),
        );

        // Channels for the sub-agent
        let (sub_tx, mut sub_rx) = mpsc::unbounded_channel::<AgentEvent>();
        let (sub_confirm_tx, mut sub_confirm_rx) = mpsc::unbounded_channel::<bool>();

        // Move the sub-agent and channel clones into an owned future.
        // This avoids borrow issues: the future owns everything it needs.
        let sub_tx_for_agent = sub_tx.clone();
        let cancel_for_sub = cancel.clone();
        let task_owned = task.to_string();
        let mut sub_future = Box::pin(async move {
            let mut sa = sub_agent;
            // Sub-agents don't receive steering messages
            let (_steer_tx, mut steer_rx) = mpsc::unbounded_channel::<String>();
            let result = sa
                .run(&task_owned, &sub_tx_for_agent, &mut sub_confirm_rx, &mut steer_rx, cancel_for_sub)
                .await;
            let last_msg = sa.last_assistant_message();
            (result, last_msg)
        });

        // Drive the sub-agent and its event loop concurrently.
        // We poll both the sub-agent future and its event channel so we
        // can forward tool confirmations through the parent's confirm_rx
        // instead of auto-approving.
        let mut sub_finished: Option<(Result<()>, String)> = None;
        let mut sub_tx_option = Some(sub_tx);

        loop {
            tokio::select! {
                biased;
                // Process sub-agent events first (higher priority)
                event = sub_rx.recv() => {
                    match event {
                        Some(AgentEvent::ToolConfirmRequest { name, args }) => {
                            // Forward to parent TUI for user confirmation
                            let _ = send_event(events, AgentEvent::ToolConfirmRequest { name, args });
                            let approved = confirm_rx.recv().await.unwrap_or(false);
                            let _ = sub_confirm_tx.send(approved);
                        }
                        Some(AgentEvent::ToolCall { name, args }) => {
                            let _ = events.send(AgentEvent::SubagentToolCall { name, args });
                        }
                        Some(AgentEvent::ToolResult { name, success, .. }) => {
                            let _ = events.send(AgentEvent::SubagentToolResult { name, success });
                        }
                        Some(_) => {} // ignore other events
                        None => break, // sub-agent channel closed
                    }
                }
                // Poll the sub-agent future
                result = &mut sub_future, if sub_finished.is_none() => {
                    sub_finished = Some(result);
                    // Drop our sender clone to close channel (the future's
                    // clone was already dropped when the async block ended).
                    sub_tx_option.take();
                }
            }
        }

        match sub_finished {
            Some((Ok(()), response)) => {
                let _ = send_event(events, AgentEvent::SubagentEnd { result: response.clone() });
                (response, true)
            }
            Some((Err(e), _)) => {
                let err_msg = format!("Sub-agent error: {}", e);
                let _ = send_event(events, AgentEvent::SubagentEnd { result: err_msg.clone() });
                (err_msg, false)
            }
            None => {
                let err_msg = "Sub-agent channel closed unexpectedly".to_string();
                let _ = send_event(events, AgentEvent::SubagentEnd { result: err_msg.clone() });
                (err_msg, false)
            }
        }
    }
}
