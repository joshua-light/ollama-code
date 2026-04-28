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
                          necessary context in the task description.\n\
                          \n\
                          Strongly recommended for per-file work: pass `files` (a list of paths) \
                          and the harness will pre-read each file and prepend its full contents to \
                          the task — the sub-agent starts already aware of the code, with no warm-up \
                          turns. This is the preferred pathway for any single-file edit on a large \
                          codebase: the sub-agent's small context stays sharp on the actual edit \
                          instead of getting buried under exploration. Use one delegated sub-agent \
                          per file you need to change."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "A self-contained task description. Include all necessary context — the sub-agent cannot see the current conversation."
                    },
                    "files": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional list of file paths to pre-read for the sub-agent. \
                                        Each file's full contents are prepended to the task, so the \
                                        sub-agent doesn't have to spend turns reading them. Use this \
                                        when delegating a single-file edit: the sub-agent gets the \
                                        file ready-to-edit and stays focused on the change itself."
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
