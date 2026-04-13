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
