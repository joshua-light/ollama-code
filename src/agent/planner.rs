use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;

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
    pub(super) async fn produce_plan(
        &self,
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

        let (result, _last_msg) =
            drive_subagent(planner, task, events, confirm_rx, Arc::clone(cancel)).await;

        if let Err(e) = result {
            let _ = send_event(events, AgentEvent::SubagentEnd { result: String::new() });
            send_event(
                events,
                AgentEvent::Error(format!("Planner sub-agent failed: {}", e)),
            )?;
            return Ok(false);
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
