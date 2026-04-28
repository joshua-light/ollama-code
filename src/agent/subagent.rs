use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{anyhow, Result};
use tokio::sync::mpsc;

use super::events::{send_event, AgentEvent};
use super::Agent;

/// Drive an already-constructed sub-agent on `task`, pumping its events to
/// the parent's channels. Translates the sub-agent's `ToolCall`/`ToolResult`
/// into `SubagentToolCall`/`SubagentToolResult` and forwards
/// `ToolConfirmRequest` to the parent's `confirm_rx`. Returns the run result,
/// the sub-agent's last assistant message (empty if none), and the
/// `read_file_ranges` it accumulated so the caller can merge them into the
/// parent's read history (avoiding redundant re-reads).
///
/// Caller is responsible for emitting `SubagentStart`/`SubagentEnd` framing.
pub(super) async fn drive_subagent(
    sub_agent: Agent,
    task: String,
    events: &mpsc::UnboundedSender<AgentEvent>,
    confirm_rx: &mut mpsc::UnboundedReceiver<bool>,
    cancel: Arc<AtomicBool>,
) -> (Result<()>, String, Vec<(String, usize, usize)>) {
    let (sub_tx, mut sub_rx) = mpsc::unbounded_channel::<AgentEvent>();
    let (sub_confirm_tx, mut sub_confirm_rx) = mpsc::unbounded_channel::<bool>();

    let sub_tx_for_agent = sub_tx.clone();
    let mut sub_future = Box::pin(async move {
        let mut sa = sub_agent;
        let (_steer_tx, mut steer_rx) = mpsc::unbounded_channel::<String>();
        let result = sa
            .run(&task, &sub_tx_for_agent, &mut sub_confirm_rx, &mut steer_rx, cancel)
            .await;
        let last_msg = sa.last_assistant_message();
        let ranges = std::mem::take(&mut sa.read_file_ranges);
        (result, last_msg, ranges)
    });

    let mut sub_finished: Option<(Result<()>, String, Vec<(String, usize, usize)>)> = None;
    let mut sub_tx_option = Some(sub_tx);

    loop {
        tokio::select! {
            biased;
            event = sub_rx.recv() => {
                match event {
                    Some(AgentEvent::ToolConfirmRequest { name, args }) => {
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
                    Some(_) => {}
                    None => break,
                }
            }
            result = &mut sub_future, if sub_finished.is_none() => {
                sub_finished = Some(result);
                sub_tx_option.take();
            }
        }
    }

    sub_finished.unwrap_or_else(|| {
        (
            Err(anyhow!("Sub-agent channel closed unexpectedly")),
            String::new(),
            Vec::new(),
        )
    })
}

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

    /// Absorb a sub-agent's `read_file_ranges` into this agent's own.
    /// Subsequent calls to `build_file_context_preamble` will include the
    /// merged paths, so the next sub-agent's task is enriched with the union
    /// of parent + previous-sub-agent reads. The returned string is a
    /// deduplicated, line-prefixed summary ready for embedding in a chat
    /// message; empty if `incoming` was empty.
    pub(super) fn merge_subagent_reads(
        &mut self,
        incoming: Vec<(String, usize, usize)>,
    ) -> String {
        if incoming.is_empty() {
            return String::new();
        }

        // Track first-seen path order so the rendered list reads top-to-bottom
        // in the order the sub-agent actually visited the files.
        let mut order: Vec<String> = Vec::new();
        let mut by_path: std::collections::HashMap<String, Vec<(usize, usize)>> =
            std::collections::HashMap::new();
        for (path, start, end) in &incoming {
            if !by_path.contains_key(path) {
                order.push(path.clone());
            }
            by_path.entry(path.clone()).or_default().push((*start, *end));
        }

        // Append incoming ranges to the parent's history, dropping exact
        // duplicates so the vec doesn't grow unboundedly across runs in long
        // interactive sessions. `build_file_context_preamble` still folds
        // overlapping ranges per path; this dedupe just bounds memory.
        for range in incoming {
            if !self.read_file_ranges.contains(&range) {
                self.read_file_ranges.push(range);
            }
        }

        // Render a one-line-per-file summary with all ranges joined.
        let mut out = String::new();
        for path in &order {
            let ranges = by_path.get(path).expect("path was inserted above");
            let mut joined = String::new();
            for (i, (start, end)) in ranges.iter().enumerate() {
                if i > 0 {
                    joined.push_str(", ");
                }
                // Display as 1-indexed inclusive bounds for human readability.
                joined.push_str(&format!("{}-{}", start + 1, *end));
            }
            out.push_str(&format!("- {} (lines {})\n", path, joined));
        }
        out
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
    pub(super) fn task_expects_edits(task: &str) -> bool {
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
        &mut self,
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

    /// Run a single subagent instance and return its result. Merges the
    /// sub-agent's read history into `self.read_file_ranges` so subsequent
    /// preambles know what's already been seen.
    async fn run_subagent_once(
        &mut self,
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

        let (result, last_msg, sub_reads) =
            drive_subagent(sub_agent, task.to_string(), events, confirm_rx, cancel.clone()).await;

        // Merge regardless of success — partial reads still count.
        let _ = self.merge_subagent_reads(sub_reads);

        match result {
            Ok(()) => {
                let _ = send_event(events, AgentEvent::SubagentEnd { result: last_msg.clone() });
                (last_msg, true)
            }
            Err(e) => {
                let err_msg = format!("Sub-agent error: {}", e);
                let _ = send_event(events, AgentEvent::SubagentEnd { result: err_msg.clone() });
                (err_msg, false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{ChatResponse, ModelBackend};
    use crate::message::Message;
    use serde_json::Value;
    use std::future::Future;
    use std::pin::Pin;

    /// Minimal backend that never gets called — `merge_subagent_reads` is
    /// pure and doesn't touch the backend, so we just need a placeholder
    /// to satisfy `Agent::new`.
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
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<ChatResponse>> + Send + 'a>> {
            Box::pin(async { unreachable!("stub backend should not be called in this test") })
        }
    }

    fn make_agent() -> Agent {
        Agent::new(
            Arc::new(StubBackend),
            "stub-model".to_string(),
            0,
            std::time::Duration::from_secs(1),
            0,
        )
    }

    #[test]
    fn merge_subagent_reads_extends_parent_history() {
        let mut agent = make_agent();
        assert!(agent.read_file_ranges.is_empty());

        let summary = agent.merge_subagent_reads(vec![
            ("src/foo.rs".to_string(), 0, 200),
            ("src/foo.rs".to_string(), 380, 580),
            ("src/bar.rs".to_string(), 0, 100),
        ]);

        // All three ranges were appended.
        assert_eq!(agent.read_file_ranges.len(), 3);
        // Build_file_context_preamble now sees them.
        let preamble = agent.build_file_context_preamble();
        assert!(preamble.contains("src/foo.rs"));
        assert!(preamble.contains("src/bar.rs"));

        // The summary string deduplicates by path and joins ranges
        // visited-order top to bottom.
        assert!(summary.contains("- src/foo.rs (lines 1-200, 381-580)\n"));
        assert!(summary.contains("- src/bar.rs (lines 1-100)\n"));
        assert!(summary.find("src/foo.rs").unwrap() < summary.find("src/bar.rs").unwrap());
    }

    #[test]
    fn merge_subagent_reads_empty_returns_empty_string() {
        let mut agent = make_agent();
        let summary = agent.merge_subagent_reads(Vec::new());
        assert!(summary.is_empty());
        assert!(agent.read_file_ranges.is_empty());
    }
}
