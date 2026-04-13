use anyhow::Result;
use serde_json::Value;

use super::{Tool, ToolDefinition};

pub struct SubagentToolDef;

impl Tool for SubagentToolDef {
    fn name(&self) -> &str { "subagent" }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "subagent".to_string(),
            description: "Spawn a sub-agent with a fresh, clean context to handle a focused task. \
                          The sub-agent has its own conversation history (only the task you give it), \
                          making it ideal for research, exploration, and self-contained coding tasks \
                          that benefit from a clean context window. Returns the sub-agent's final \
                          response. The sub-agent cannot see this conversation, so include all \
                          necessary context in the task description."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "A self-contained task description. Include all necessary context — the sub-agent cannot see the current conversation."
                    }
                },
                "required": ["task"]
            }),
        }
    }

    fn execute(&self, _arguments: &Value) -> Result<String> {
        anyhow::bail!("subagent tool must be executed by the agent loop, not the tool registry")
    }
}
