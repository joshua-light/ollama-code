use anyhow::{anyhow, Result};
use serde_json::Value;

use crate::agent::plan::{SharedTodoList, StepStatus};

use super::{required_str, Tool, ToolDefinition};

fn lock_list(list: &SharedTodoList) -> Result<std::sync::MutexGuard<'_, crate::agent::plan::TodoList>> {
    list.lock()
        .map_err(|e| anyhow!("plan list lock poisoned: {}", e))
}

fn required_index(args: &Value) -> Result<usize> {
    args.get("index")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .ok_or_else(|| anyhow!("Missing 'index' argument (integer >= 0)"))
}

pub struct PlanAddStepTool {
    list: SharedTodoList,
    /// When `true`, refuse to add a step if the plan is non-empty and every
    /// step is still `Pending` (i.e. the main agent hasn't engaged with the
    /// plan yet). Set to `false` for the planner sub-agent, which is the one
    /// populating the plan in the first place.
    gate_when_populated: bool,
}

impl PlanAddStepTool {
    /// Default constructor: gate is enabled. Use this for the main agent.
    pub fn new(list: SharedTodoList) -> Self {
        Self {
            list,
            gate_when_populated: true,
        }
    }

    /// Construct without the populated-plan gate. Use this for the planner
    /// sub-agent, which must be free to populate an empty plan.
    pub fn new_ungated(list: SharedTodoList) -> Self {
        Self {
            list,
            gate_when_populated: false,
        }
    }
}

impl Tool for PlanAddStepTool {
    fn name(&self) -> &str {
        "plan_add_step"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "plan_add_step".to_string(),
            description: "Append a step to the current plan. Use this during the planning \
                          phase to break the user's request into concrete, actionable steps. \
                          Each call adds one step. Steps appear in the order added."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Short imperative description of the step (e.g. 'Read src/foo.rs to understand the parser', 'Add a new field to Config')."
                    }
                },
                "required": ["description"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let description = required_str(arguments, "description")?.trim().to_string();
        if description.is_empty() {
            return Err(anyhow!("Step description cannot be empty"));
        }
        let mut list = lock_list(&self.list)?;
        if self.gate_when_populated
            && list.len() > 0
            && list.steps.iter().all(|s| matches!(s.status, StepStatus::Pending))
        {
            let n = list.len();
            return Err(anyhow!(
                "[harness] plan_add_step refused: a populated plan is already in place ({} steps, all still pending). \
                 Work the existing plan first — call plan_mark_in_progress(0) to start step 0. \
                 You may add new steps later, but only after at least one step has been marked done, in_progress, or skipped. \
                 If a step in the existing plan is wrong or missing context, mention that in your next turn's content; \
                 the harness will allow add after you've engaged with the plan.",
                n
            ));
        }
        let idx = list.add(&description)?;
        Ok(format!("Added step {} to plan: {}", idx, description))
    }
}

pub struct PlanMarkInProgressTool {
    list: SharedTodoList,
}

impl PlanMarkInProgressTool {
    pub fn new(list: SharedTodoList) -> Self {
        Self { list }
    }
}

impl Tool for PlanMarkInProgressTool {
    fn name(&self) -> &str {
        "plan_mark_in_progress"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "plan_mark_in_progress".to_string(),
            description: "Mark a plan step as currently in progress. Call this when you start \
                          working on a step. Only one step should be in progress at a time."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Zero-based index of the step (matches the index shown in the plan summary)."
                    }
                },
                "required": ["index"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let idx = required_index(arguments)?;
        let mut list = lock_list(&self.list)?;
        list.set_status(idx, StepStatus::InProgress, None)?;
        Ok(format!("Step {} marked in progress", idx))
    }
}

pub struct PlanMarkDoneTool {
    list: SharedTodoList,
}

impl PlanMarkDoneTool {
    pub fn new(list: SharedTodoList) -> Self {
        Self { list }
    }
}

impl Tool for PlanMarkDoneTool {
    fn name(&self) -> &str {
        "plan_mark_done"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "plan_mark_done".to_string(),
            description: "Mark a plan step as completed. Call this only after the work for the \
                          step has actually been finished — verify with a tool call (e.g. \
                          re-read the edited file, run a check) before marking done."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Zero-based index of the step."
                    }
                },
                "required": ["index"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let idx = required_index(arguments)?;
        let mut list = lock_list(&self.list)?;
        list.set_status(idx, StepStatus::Done, None)?;
        let remaining = list.unfinished().len();
        Ok(format!(
            "Step {} marked done ({} step{} remaining)",
            idx,
            remaining,
            if remaining == 1 { "" } else { "s" }
        ))
    }
}

pub struct PlanSkipStepTool {
    list: SharedTodoList,
}

impl PlanSkipStepTool {
    pub fn new(list: SharedTodoList) -> Self {
        Self { list }
    }
}

impl Tool for PlanSkipStepTool {
    fn name(&self) -> &str {
        "plan_skip_step"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "plan_skip_step".to_string(),
            description: "Skip a plan step that does not apply (e.g. 'Run tests' when there \
                          are no tests for this code). Requires a reason. Only use when a \
                          step is genuinely inapplicable, not as a way to dodge work."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Zero-based index of the step."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this step is being skipped."
                    }
                },
                "required": ["index", "reason"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let idx = required_index(arguments)?;
        let reason = required_str(arguments, "reason")?.trim().to_string();
        if reason.is_empty() {
            return Err(anyhow!("Skip reason cannot be empty"));
        }
        let mut list = lock_list(&self.list)?;
        list.set_status(idx, StepStatus::Skipped, Some(reason.clone()))?;
        Ok(format!("Step {} skipped: {}", idx, reason))
    }
}

pub struct PlanListStepsTool {
    list: SharedTodoList,
}

impl PlanListStepsTool {
    pub fn new(list: SharedTodoList) -> Self {
        Self { list }
    }
}

impl Tool for PlanListStepsTool {
    fn name(&self) -> &str {
        "plan_list_steps"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "plan_list_steps".to_string(),
            description: "Show the current plan with each step's index, status marker, and \
                          description. Useful when you need to recheck progress."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    fn execute(&self, _arguments: &Value) -> Result<String> {
        let list = lock_list(&self.list)?;
        Ok(list.summary_for_prompt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::plan::new_shared_todo_list;
    use serde_json::json;

    #[test]
    fn add_then_list() {
        // Use ungated tool so successive `add` calls aren't blocked by the
        // populated-plan gate (which only fires for the main agent's adder).
        let list = new_shared_todo_list();
        PlanAddStepTool::new_ungated(list.clone())
            .execute(&json!({"description": "first"}))
            .unwrap();
        PlanAddStepTool::new_ungated(list.clone())
            .execute(&json!({"description": "second"}))
            .unwrap();
        let out = PlanListStepsTool::new(list).execute(&json!({})).unwrap();
        assert!(out.contains("0. [ ] first"));
        assert!(out.contains("1. [ ] second"));
    }

    #[test]
    fn mark_done_decrements_remaining() {
        let list = new_shared_todo_list();
        let adder = PlanAddStepTool::new_ungated(list.clone());
        adder.execute(&json!({"description": "a"})).unwrap();
        adder.execute(&json!({"description": "b"})).unwrap();
        let done = PlanMarkDoneTool::new(list.clone())
            .execute(&json!({"index": 0}))
            .unwrap();
        assert!(done.contains("1 step remaining"));
    }

    #[test]
    fn skip_requires_reason() {
        let list = new_shared_todo_list();
        PlanAddStepTool::new(list.clone())
            .execute(&json!({"description": "a"}))
            .unwrap();
        let err = PlanSkipStepTool::new(list)
            .execute(&json!({"index": 0, "reason": "  "}))
            .unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn out_of_range_errors() {
        let list = new_shared_todo_list();
        let err = PlanMarkDoneTool::new(list)
            .execute(&json!({"index": 5}))
            .unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn empty_description_errors() {
        let list = new_shared_todo_list();
        let err = PlanAddStepTool::new(list)
            .execute(&json!({"description": "   "}))
            .unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn gate_blocks_add_when_plan_full_and_untouched() {
        let list = new_shared_todo_list();
        // Pre-populate as the planner would (ungated path).
        let planner_adder = PlanAddStepTool::new_ungated(list.clone());
        planner_adder
            .execute(&json!({"description": "step 0"}))
            .unwrap();
        planner_adder
            .execute(&json!({"description": "step 1"}))
            .unwrap();
        planner_adder
            .execute(&json!({"description": "step 2"}))
            .unwrap();

        // The main agent's gated tool should now refuse.
        let main_adder = PlanAddStepTool::new(list.clone());
        let err = main_adder
            .execute(&json!({"description": "duplicate of step 0"}))
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("plan_add_step refused"));
        assert!(msg.contains("3 steps"));
        assert!(msg.contains("plan_mark_in_progress"));
        // Plan length unchanged.
        assert_eq!(list.lock().unwrap().len(), 3);
    }

    #[test]
    fn gate_allows_add_after_step_marked_in_progress() {
        let list = new_shared_todo_list();
        let planner_adder = PlanAddStepTool::new_ungated(list.clone());
        planner_adder.execute(&json!({"description": "a"})).unwrap();
        planner_adder.execute(&json!({"description": "b"})).unwrap();

        PlanMarkInProgressTool::new(list.clone())
            .execute(&json!({"index": 0}))
            .unwrap();

        let main_adder = PlanAddStepTool::new(list.clone());
        let res = main_adder.execute(&json!({"description": "newly discovered step"}));
        assert!(res.is_ok(), "add should be allowed once a step is in_progress: {:?}", res);
        assert_eq!(list.lock().unwrap().len(), 3);
    }

    #[test]
    fn gate_allows_add_after_step_marked_done() {
        let list = new_shared_todo_list();
        let planner_adder = PlanAddStepTool::new_ungated(list.clone());
        planner_adder.execute(&json!({"description": "a"})).unwrap();
        planner_adder.execute(&json!({"description": "b"})).unwrap();

        PlanMarkDoneTool::new(list.clone())
            .execute(&json!({"index": 0}))
            .unwrap();

        let main_adder = PlanAddStepTool::new(list.clone());
        let res = main_adder.execute(&json!({"description": "follow-up step"}));
        assert!(res.is_ok(), "add should be allowed once a step is done: {:?}", res);
        assert_eq!(list.lock().unwrap().len(), 3);
    }

    #[test]
    fn ungated_tool_always_adds() {
        let list = new_shared_todo_list();
        let planner_adder = PlanAddStepTool::new_ungated(list.clone());
        planner_adder.execute(&json!({"description": "a"})).unwrap();
        planner_adder.execute(&json!({"description": "b"})).unwrap();
        planner_adder.execute(&json!({"description": "c"})).unwrap();

        // Even though plan is fully Pending, ungated tool keeps adding.
        let res = planner_adder.execute(&json!({"description": "d"}));
        assert!(res.is_ok(), "ungated add should always succeed: {:?}", res);
        assert_eq!(list.lock().unwrap().len(), 4);
    }
}
