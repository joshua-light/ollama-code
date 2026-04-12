pub enum SlashCommand {
    Clear,
    Context,
    Model,
    Session,
    New,
    Unknown(String),
}

pub struct CommandInfo {
    pub name: &'static str,
    pub description: &'static str,
}

pub const COMMANDS: &[CommandInfo] = &[
    CommandInfo {
        name: "/clear",
        description: "Clear conversation history",
    },
    CommandInfo {
        name: "/context",
        description: "Show context window usage",
    },
    CommandInfo {
        name: "/model",
        description: "Switch the active model",
    },
    CommandInfo {
        name: "/session",
        description: "Show session log directory",
    },
    CommandInfo {
        name: "/new",
        description: "Start a new conversation",
    },
];

pub fn parse(input: &str) -> Option<SlashCommand> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return None;
    }
    match trimmed.split_whitespace().next()? {
        "/clear" => Some(SlashCommand::Clear),
        "/context" => Some(SlashCommand::Context),
        "/model" => Some(SlashCommand::Model),
        "/session" => Some(SlashCommand::Session),
        "/new" => Some(SlashCommand::New),
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
