use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;

use crate::message::Message;

use super::events::{send_event, AgentEvent};
use super::subagent::drive_subagent;
use super::Agent;

impl Agent {
    /// Run the planner sub-agent for the given user request, populating the
    /// shared TodoList. After the sub-agent finishes, append any auto-steps
    /// from `plan_config.append_steps` and emit `PlanReady`.
    ///
    /// On error or empty plan, returns `Ok(false)` so the caller can decide
    /// whether to fall back to non-planned execution.
    ///
    /// Mutates `self`: the planner's accumulated `read_file_ranges` are
    /// merged into the parent's so subsequent sub-agents see them in the
    /// "already-read" preamble, and a user-role "[Planner findings]" message
    /// is injected into `self.messages` listing those files for the
    /// orchestrator model.
    pub(super) async fn produce_plan(
        &mut self,
        user_request: &str,
        events: &mpsc::UnboundedSender<AgentEvent>,
        confirm_rx: &mut mpsc::UnboundedReceiver<bool>,
        cancel: &Arc<AtomicBool>,
    ) -> Result<bool> {
        send_event(events, AgentEvent::PlanningStarted)?;

        let planner = Agent::new_planner_subagent(
            Arc::clone(&self.backend),
            self.model.clone(),
            self.context_size,
            self.bash_timeout,
            self.subagent_max_turns,
            self.config.as_ref(),
            Arc::clone(&self.plan),
        );

        // The planner-mode system prompt already instructs it to break the
        // task into steps; just give it the user's request.
        let task = format!(
            "User request to plan:\n{}\n\nProduce a structured plan now using \
             plan_add_step. End your turn with no tool calls when the plan is \
             complete.",
            user_request
        );

        let _ = send_event(
            events,
            AgentEvent::SubagentStart {
                task: format!("[planner] {}", user_request.lines().next().unwrap_or("")),
            },
        );

        let (result, _last_msg, planner_reads) =
            drive_subagent(planner, task, events, confirm_rx, Arc::clone(cancel)).await;

        // Merge the planner's read history into the parent's so we don't
        // re-read the same files. We do this even on planner error — any
        // partial reads it managed are still useful context for the main
        // agent.
        let merge_summary = self.merge_subagent_reads(planner_reads);

        if let Err(e) = result {
            let _ = send_event(events, AgentEvent::SubagentEnd { result: String::new() });
            send_event(
                events,
                AgentEvent::Error(format!("Planner sub-agent failed: {}", e)),
            )?;
            return Ok(false);
        }

        // Inject a single user-role message advertising what the planner
        // already read. Skip if the planner didn't read anything.
        if !merge_summary.is_empty() {
            let body = format!(
                "[Planner findings]\nThe planner sub-agent already read these files \
                 while building the plan:\n{}\nDon't re-read them unless you need a \
                 section the planner skipped. Trust the plan; start with \
                 `plan_mark_in_progress(0)`.",
                merge_summary
            );
            let msg = Message::user(body);
            self.messages.push(msg.clone());
            let _ = send_event(events, AgentEvent::MessageLogged(msg));
        }

        // Append auto-steps from config and read out the descriptions in one
        // critical section.
        let descriptions = {
            let mut list = self.plan.lock().expect("plan lock poisoned");
            for step in &self.plan_config.append_steps {
                let _ = list.add(step);
            }
            list.descriptions()
        };

        let _ = send_event(
            events,
            AgentEvent::SubagentEnd {
                result: format!("planner produced {} step(s)", descriptions.len()),
            },
        );

        if descriptions.is_empty() {
            return Ok(false);
        }

        send_event(
            events,
            AgentEvent::PlanReady {
                steps: descriptions,
            },
        )?;
        Ok(true)
    }
}
