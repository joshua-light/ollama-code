#[derive(Debug)]
pub enum SlashCommand {
    Bypass,
    Clear,
    Context,
    Help,
    Mcp,
    Model,
    Resume,
    Rewind,
    Session,
    Skills,
    Stats,
    SystemPrompt,
    New,
    Unknown(String),
}

pub struct CommandInfo {
    pub name: &'static str,
    pub description: &'static str,
}

pub const COMMANDS: &[CommandInfo] = &[
    CommandInfo {
        name: "/bypass",
        description: "Toggle auto-approve for tool calls",
    },
    CommandInfo {
        name: "/clear",
        description: "Clear conversation history",
    },
    CommandInfo {
        name: "/context",
        description: "Show or set context window size",
    },
    CommandInfo {
        name: "/help",
        description: "Show available commands",
    },
    CommandInfo {
        name: "/mcp",
        description: "Show connected MCP servers and their tools",
    },
    CommandInfo {
        name: "/model",
        description: "Switch the active model",
    },
    CommandInfo {
        name: "/resume",
        description: "List recent sessions",
    },
    CommandInfo {
        name: "/rewind",
        description: "Undo the last turn (or /rewind N for N turns)",
    },
    CommandInfo {
        name: "/session",
        description: "Show session log directory",
    },
    CommandInfo {
        name: "/skills",
        description: "List available skills",
    },
    CommandInfo {
        name: "/stats",
        description: "Show session statistics",
    },
    CommandInfo {
        name: "/system-prompt",
        description: "Show the full system prompt sent to the model",
    },
    CommandInfo {
        name: "/new",
        description: "Start a new conversation (alias for /clear)",
    },
];

pub fn parse(input: &str) -> Option<SlashCommand> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return None;
    }
    match trimmed.split_whitespace().next()? {
        "/bypass" => Some(SlashCommand::Bypass),
        "/clear" => Some(SlashCommand::Clear),
        "/context" => Some(SlashCommand::Context),
        "/help" => Some(SlashCommand::Help),
        "/mcp" => Some(SlashCommand::Mcp),
        "/model" => Some(SlashCommand::Model),
        "/resume" => Some(SlashCommand::Resume),
        "/session" => Some(SlashCommand::Session),
        "/skills" => Some(SlashCommand::Skills),
        "/stats" => Some(SlashCommand::Stats),
        "/system-prompt" => Some(SlashCommand::SystemPrompt),
        "/new" => Some(SlashCommand::New),
        "/rewind" => Some(SlashCommand::Rewind),
        cmd => Some(SlashCommand::Unknown(cmd.to_string())),
    }
}

pub fn completions(prefix: &str) -> Vec<&'static CommandInfo> {
    let prefix = prefix.trim();
    if prefix.is_empty() || !prefix.starts_with('/') {
        return Vec::new();
    }
    COMMANDS
        .iter()
        .filter(|cmd| cmd.name.starts_with(prefix))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse ───────────────────────────────────────────────────────

    #[test]
    fn parse_all_known_commands() {
        let cases = vec![
            ("/bypass", "Bypass"),
            ("/clear", "Clear"),
            ("/context", "Context"),
            ("/help", "Help"),
            ("/mcp", "Mcp"),
            ("/model", "Model"),
            ("/resume", "Resume"),
            ("/rewind", "Rewind"),
            ("/session", "Session"),
            ("/skills", "Skills"),
            ("/stats", "Stats"),
            ("/system-prompt", "SystemPrompt"),
            ("/new", "New"),
        ];
        for (input, expected_name) in cases {
            let cmd = parse(input);
            assert!(cmd.is_some(), "failed to parse '{}'", input);
            let debug = format!("{:?}", cmd.unwrap());
            assert!(debug.contains(expected_name), "expected {} for '{}', got {}", expected_name, input, debug);
        }
    }

    #[test]
    fn parse_with_trailing_args() {
        // /context 4096 should still parse as Context
        let cmd = parse("/context 4096");
        assert!(matches!(cmd, Some(SlashCommand::Context)));
    }

    #[test]
    fn parse_with_whitespace() {
        let cmd = parse("  /help  ");
        assert!(matches!(cmd, Some(SlashCommand::Help)));
    }

    #[test]
    fn parse_unknown_command() {
        let cmd = parse("/foobar");
        assert!(matches!(cmd, Some(SlashCommand::Unknown(ref s)) if s == "/foobar"));
    }

    #[test]
    fn parse_not_a_command() {
        assert!(parse("hello").is_none());
        assert!(parse("").is_none());
        assert!(parse("no slash here").is_none());
    }

    // ── completions ─────────────────────────────────────────────────

    #[test]
    fn completions_full_prefix() {
        let results = completions("/help");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "/help");
    }

    #[test]
    fn completions_partial_prefix() {
        let results = completions("/cl");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "/clear");
    }

    #[test]
    fn completions_multiple_matches() {
        // /s matches /session, /skills, /stats, /system-prompt
        let results = completions("/s");
        assert!(results.len() >= 4);
        let names: Vec<&str> = results.iter().map(|c| c.name).collect();
        assert!(names.contains(&"/session"));
        assert!(names.contains(&"/skills"));
        assert!(names.contains(&"/stats"));
    }

    #[test]
    fn completions_no_match() {
        let results = completions("/zzz");
        assert!(results.is_empty());
    }

    #[test]
    fn completions_empty_input() {
        assert!(completions("").is_empty());
    }

    #[test]
    fn completions_not_a_command() {
        assert!(completions("help").is_empty());
    }

    #[test]
    fn completions_just_slash() {
        let results = completions("/");
        // Should return all commands
        assert_eq!(results.len(), COMMANDS.len());
    }
}
