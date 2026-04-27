use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};

/// Cap on the number of plan steps. Bounds memory + the size of the
/// `summary_for_prompt` block that gets re-sent every turn.
pub const MAX_PLAN_STEPS: usize = 50;

/// Cap on the length of a single step description in characters.
pub const MAX_STEP_DESC_LEN: usize = 500;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    Pending,
    InProgress,
    Done,
    Skipped,
}

impl StepStatus {
    pub fn marker(self) -> &'static str {
        match self {
            StepStatus::Pending => "[ ]",
            StepStatus::InProgress => "[~]",
            StepStatus::Done => "[x]",
            StepStatus::Skipped => "[-]",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, StepStatus::Done | StepStatus::Skipped)
    }
}

#[derive(Debug, Clone)]
pub struct TodoStep {
    pub description: String,
    pub status: StepStatus,
    pub skip_reason: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct TodoList {
    pub steps: Vec<TodoStep>,
}

impl TodoList {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Append a step in `Pending` state. Returns the new step's index, or an
    /// error if the plan or description exceeds the safety caps.
    pub fn add(&mut self, description: &str) -> Result<usize> {
        if self.steps.len() >= MAX_PLAN_STEPS {
            return Err(anyhow!(
                "plan already has {} steps (max {}); refusing to add more",
                self.steps.len(),
                MAX_PLAN_STEPS
            ));
        }
        if description.len() > MAX_STEP_DESC_LEN {
            return Err(anyhow!(
                "step description is {} chars; max is {}",
                description.len(),
                MAX_STEP_DESC_LEN
            ));
        }
        let idx = self.steps.len();
        self.steps.push(TodoStep {
            description: description.to_string(),
            status: StepStatus::Pending,
            skip_reason: None,
        });
        Ok(idx)
    }

    /// Update the status of step `index`. Returns an error if out of range or
    /// if a transition is illegal (e.g. marking a Done step as Pending).
    pub fn set_status(
        &mut self,
        index: usize,
        status: StepStatus,
        reason: Option<String>,
    ) -> Result<()> {
        let total = self.steps.len();
        let step = self
            .steps
            .get_mut(index)
            .ok_or_else(|| anyhow!("step index {} is out of range (have {} steps)", index, total))?;
        if step.status.is_terminal() && !status.is_terminal() {
            return Err(anyhow!(
                "step {} is already {} — cannot move it back to {}",
                index,
                step.status.marker(),
                status.marker()
            ));
        }
        step.status = status;
        step.skip_reason = if matches!(status, StepStatus::Skipped) { reason } else { None };
        Ok(())
    }

    /// Indices and steps that are not yet Done or Skipped.
    pub fn unfinished(&self) -> Vec<(usize, &TodoStep)> {
        self.steps
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.status.is_terminal())
            .collect()
    }

    /// One-line-per-step formatted summary suitable for a system reminder.
    pub fn summary_for_prompt(&self) -> String {
        if self.steps.is_empty() {
            return "(no plan steps)".to_string();
        }
        let mut out = String::new();
        for (i, step) in self.steps.iter().enumerate() {
            out.push_str(&format!("  {}. {} {}", i, step.status.marker(), step.description));
            if let Some(reason) = &step.skip_reason {
                out.push_str(&format!(" — skipped: {}", reason));
            }
            out.push('\n');
        }
        out
    }

    /// Just the descriptions (in order), useful for `PlanReady` events.
    pub fn descriptions(&self) -> Vec<String> {
        self.steps.iter().map(|s| s.description.clone()).collect()
    }
}

pub type SharedTodoList = Arc<Mutex<TodoList>>;

pub fn new_shared_todo_list() -> SharedTodoList {
    Arc::new(Mutex::new(TodoList::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_appends_and_returns_index() {
        let mut list = TodoList::new();
        assert_eq!(list.add("first").unwrap(), 0);
        assert_eq!(list.add("second").unwrap(), 1);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn add_rejects_oversized_description() {
        let mut list = TodoList::new();
        let huge = "x".repeat(MAX_STEP_DESC_LEN + 1);
        assert!(list.add(&huge).is_err());
    }

    #[test]
    fn add_rejects_past_max_steps() {
        let mut list = TodoList::new();
        for _ in 0..MAX_PLAN_STEPS {
            list.add("step").unwrap();
        }
        assert!(list.add("one too many").is_err());
    }

    #[test]
    fn set_status_updates_step() {
        let mut list = TodoList::new();
        list.add("step").unwrap();
        list.set_status(0, StepStatus::Done, None).unwrap();
        assert_eq!(list.steps[0].status, StepStatus::Done);
    }

    #[test]
    fn set_status_out_of_range_errors() {
        let mut list = TodoList::new();
        let err = list.set_status(0, StepStatus::Done, None).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn cannot_unmark_terminal_step() {
        let mut list = TodoList::new();
        list.add("a").unwrap();
        list.set_status(0, StepStatus::Done, None).unwrap();
        let err = list.set_status(0, StepStatus::Pending, None).unwrap_err();
        assert!(err.to_string().contains("already"));
    }

    #[test]
    fn can_transition_between_terminal_states() {
        let mut list = TodoList::new();
        list.add("a").unwrap();
        list.set_status(0, StepStatus::Done, None).unwrap();
        list.set_status(0, StepStatus::Skipped, Some("changed mind".into()))
            .unwrap();
        assert_eq!(list.steps[0].status, StepStatus::Skipped);
        assert_eq!(list.steps[0].skip_reason.as_deref(), Some("changed mind"));
    }

    #[test]
    fn skip_reason_only_kept_for_skipped() {
        let mut list = TodoList::new();
        list.add("a").unwrap();
        list.set_status(0, StepStatus::InProgress, Some("ignored".into()))
            .unwrap();
        assert!(list.steps[0].skip_reason.is_none());
    }

    #[test]
    fn unfinished_excludes_done_and_skipped() {
        let mut list = TodoList::new();
        list.add("a").unwrap();
        list.add("b").unwrap();
        list.add("c").unwrap();
        list.set_status(0, StepStatus::Done, None).unwrap();
        list.set_status(2, StepStatus::Skipped, Some("n/a".into()))
            .unwrap();
        let pending = list.unfinished();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, 1);
        assert_eq!(pending[0].1.description, "b");
    }

    #[test]
    fn summary_format_sample() {
        let mut list = TodoList::new();
        list.add("explore").unwrap();
        list.add("edit").unwrap();
        list.set_status(0, StepStatus::Done, None).unwrap();
        let out = list.summary_for_prompt();
        assert!(out.contains("0. [x] explore"));
        assert!(out.contains("1. [ ] edit"));
    }

    #[test]
    fn summary_for_empty_list() {
        let list = TodoList::new();
        assert!(list.summary_for_prompt().contains("no plan steps"));
    }
}
