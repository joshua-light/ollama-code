use anyhow::Result;
use serde_json::Value;

use super::{required_str, optional_str, Tool, ToolDefinition};
use crate::skills::SkillMeta;

/// Tool that lets the model activate a discovered skill by name.
/// Returns the skill's instructions (SKILL.md body) so the model can follow them.
pub struct SkillTool {
    skills: Vec<SkillMeta>,
}

impl SkillTool {
    pub fn new(skills: Vec<SkillMeta>) -> Self {
        Self { skills }
    }
}

impl Tool for SkillTool {
    fn name(&self) -> &str {
        "skill"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "skill".to_string(),
            description: "Activate a skill by name. Returns the skill's instructions for you \
                          to follow. Use this when a task matches one of the available skills."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name to activate"
                    },
                    "args": {
                        "type": "string",
                        "description": "Optional arguments or context to pass to the skill"
                    }
                },
                "required": ["name"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let name = required_str(arguments, "name")?;
        let args = optional_str(arguments, "args");

        let skill = self
            .skills
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| {
                let available: Vec<&str> = self.skills.iter().map(|s| s.name.as_str()).collect();
                anyhow::anyhow!(
                    "Unknown skill '{}'. Available skills: {}",
                    name,
                    available.join(", ")
                )
            })?;

        let instructions = skill.load_instructions()?;

        if let Some(user_args) = args {
            if !user_args.is_empty() {
                return Ok(format!("{}\n\nUser input: {}", instructions, user_args));
            }
        }

        Ok(instructions)
    }
}
