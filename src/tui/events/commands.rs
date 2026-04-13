use anyhow::Result;
use tokio::sync::mpsc;

use crate::commands::SlashCommand;
use crate::llama_server::ModelSource;

use super::super::app::{AgentInput, App, ChatMessage, ContextInfoData, PendingServerStart, StatsInfoData};
use super::super::render::{format_number, pick_verb};

pub(super) fn handle_command(
    cmd: SlashCommand,
    raw_input: &str,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    model_tx: &mpsc::UnboundedSender<Result<Vec<String>>>,
    session: &crate::session::Session,
) {
    let was_at_bottom = app.is_at_bottom();

    match cmd {
        SlashCommand::Bypass => {
            app.auto_approve = !app.auto_approve;
            let status = if app.auto_approve { "on" } else { "off" };
            app.messages.push(ChatMessage::Info(format!(
                "Bypass permissions {}.", status
            )));
        }
        SlashCommand::Clear | SlashCommand::New => {
            app.reset_conversation(input_tx, "Conversation cleared.");
        }
        SlashCommand::Rewind => {
            let arg = raw_input.trim().strip_prefix("/rewind").unwrap_or("").trim();
            let n = if arg.is_empty() {
                1
            } else if let Ok(num) = arg.parse::<usize>() {
                if num == 0 {
                    app.messages.push(ChatMessage::Info(
                        "Usage: /rewind [N] — undo the last N turns (default: 1)".into(),
                    ));
                    if was_at_bottom { app.scroll_offset = 0; }
                    return;
                }
                num
            } else {
                1
            };

            let rewound = app.rewind_turns(n, input_tx);
            if rewound == 0 {
                app.messages.push(ChatMessage::Info("Nothing to rewind.".into()));
            } else if rewound == 1 {
                app.messages.push(ChatMessage::Info("Rewound last turn.".into()));
            } else {
                app.messages.push(ChatMessage::Info(
                    format!("Rewound last {} turns.", rewound),
                ));
            }
        }
        SlashCommand::Context => {
            // `/context <size>` — set context size for current model
            let arg = raw_input.trim().strip_prefix("/context").unwrap_or("").trim();
            if let Ok(new_size) = arg.parse::<u64>() {
                let is_llama_cpp = app.config.backend.as_deref() == Some("llama-cpp");

                app.context_size = new_size;
                let _ = input_tx.send(AgentInput::SetContextSize(new_size));

                // Save to config
                let mut config = app.config.clone();
                config.context_size = Some(new_size);
                if let Err(e) = config.save() {
                    app.messages.push(ChatMessage::Error(format!(
                        "Warning: could not save config: {}", e
                    )));
                }
                app.config = config;

                if is_llama_cpp && app.config.llama_server_url.is_some() {
                    // Remote server — can't restart it, just update the local parameter
                    app.messages.push(ChatMessage::Info(format!(
                        "Context size set to {} (local parameter). The remote server's actual context limit is controlled on the server side.",
                        format_number(new_size)
                    )));
                } else if is_llama_cpp {
                    // Restart llama-server with the new context size
                    let model_source = if let Some(ref hf) = app.config.hf_repo {
                        Some(ModelSource::HuggingFace(hf.clone()))
                    } else if let Some(ref p) = app.config.model_path {
                        let path = std::path::PathBuf::from(p);
                        if path.exists() {
                            Some(ModelSource::File(path))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let (Some(server_path), Some(model_source)) =
                        (app.config.llama_server_path.clone(), model_source)
                    {
                        app.messages.push(ChatMessage::Info(format!(
                            "Restarting llama-server with context {}...",
                            format_number(new_size)
                        )));
                        app.begin_processing("Restarting server".to_string());

                        let extra_args = app.config.llama_server_args.clone().unwrap_or_default();
                        app.server.pending_server_start = Some(PendingServerStart {
                            server_path,
                            model_source,
                            ctx: new_size,
                            extra_args,
                            model_name: app.model.clone(),
                            sampling: crate::ollama::SamplingParams {
                                temperature: app.config.temperature,
                                top_p: app.config.top_p,
                                top_k: app.config.top_k,
                            },
                            unload: None,
                        });
                        app.server.stop_llama_server = true;
                    } else {
                        app.messages.push(ChatMessage::Error(
                            "Cannot restart llama-server: missing server path or model source.".into(),
                        ));
                    }
                } else {
                    app.messages.push(ChatMessage::Info(format!(
                        "Context size set to {} for {}.",
                        format_number(new_size),
                        app.model
                    )));
                }
            } else {
                // No argument — show context info
                let mut user_messages = 0u32;
                let mut assistant_messages = 0u32;
                let mut tool_calls = 0u32;

                for msg in &app.messages {
                    match msg {
                        ChatMessage::User(_) => user_messages += 1,
                        ChatMessage::Assistant(_) => assistant_messages += 1,
                        ChatMessage::ToolCall { .. } => tool_calls += 1,
                        _ => {}
                    }
                }

                app.messages.push(ChatMessage::ContextInfo(ContextInfoData {
                    context_used: app.context_used,
                    context_size: app.context_size,
                    user_messages,
                    assistant_messages,
                    tool_calls,
                    base_prompt_tokens: app.stats.base_prompt_tokens,
                    project_docs_tokens: app.stats.project_docs_tokens.clone(),
                    skills_tokens: app.stats.skills_tokens,
                    tool_defs_tokens: app.stats.tool_defs_tokens,
                }));
            }
        }
        SlashCommand::Help => {
            let mut info = String::from("Commands:\n");
            for cmd in crate::commands::COMMANDS {
                info.push_str(&format!("  {:12} {}\n", cmd.name, cmd.description));
            }
            if !app.skills.is_empty() {
                info.push_str("\nSkills:\n");
                for skill in &app.skills {
                    info.push_str(&format!("  {:12} {}\n", format!("/{}", skill.name), skill.description));
                }
            }
            app.messages.push(ChatMessage::Info(info.trim_end().to_string()));
        }
        SlashCommand::Mcp => {
            match &app.config.mcp_servers {
                Some(servers) if !servers.is_empty() => {
                    let mut info = String::new();
                    for (name, cfg) in servers {
                        let enabled = app.config.is_tool_enabled(name);
                        let status = if enabled { "enabled" } else { "disabled" };
                        let transport = if let Some(ref url) = cfg.url {
                            format!("http: {}", url)
                        } else {
                            format!(
                                "stdio: {} {}",
                                cfg.command.as_deref().unwrap_or("?"),
                                cfg.args.join(" ")
                            )
                        };
                        info.push_str(&format!(
                            "  {} ({}, {})\n",
                            name, status, transport,
                        ));
                    }
                    app.messages.push(ChatMessage::Info(format!(
                        "MCP servers:\n{}",
                        info.trim_end()
                    )));
                }
                _ => {
                    app.messages.push(ChatMessage::Info(
                        "No MCP servers configured. Add [mcp_servers.<name>] to your config.".into(),
                    ));
                }
            }
        }
        SlashCommand::Model => {
            app.messages.push(ChatMessage::Info("Fetching models...".into()));
            let ollama = app.ollama.clone();
            let tx = model_tx.clone();
            tokio::spawn(async move {
                let result = match ollama.list_models().await {
                    Ok(models) => Ok(models.into_iter().map(|m| m.name).collect()),
                    Err(e) => Err(e),
                };
                let _ = tx.send(result);
            });
        }
        SlashCommand::Resume => {
            match crate::session::Session::list_recent(10) {
                Ok(sessions) if sessions.is_empty() => {
                    app.messages.push(ChatMessage::Info("No previous sessions found.".into()));
                }
                Ok(sessions) => {
                    let mut info = String::from("Recent sessions:\n");
                    for (i, id) in sessions.iter().enumerate() {
                        let current = if id == session.id() { " (current)" } else { "" };
                        info.push_str(&format!("  {}. {}{}\n", i + 1, id, current));
                    }
                    info.push_str("\nUse --resume <id> to resume a session.");
                    app.messages.push(ChatMessage::Info(info));
                }
                Err(e) => {
                    app.messages.push(ChatMessage::Error(format!(
                        "Failed to list sessions: {}", e
                    )));
                }
            }
        }
        SlashCommand::Session => {
            app.messages.push(ChatMessage::Info(format!(
                "Session: {}",
                session.path().display()
            )));
        }
        SlashCommand::Skills => {
            if app.skills.is_empty() {
                app.messages.push(ChatMessage::Info(
                    "No skills found. Add SKILL.md files to .agents/skills/<name>/ or ~/.config/ollama-code/skills/<name>/.".into(),
                ));
            } else {
                let mut info = String::from("Skills:\n");
                for skill in &app.skills {
                    info.push_str(&format!("  /{}: {}\n    {}\n", skill.name, skill.description, skill.dir.display()));
                }
                app.messages.push(ChatMessage::Info(info.trim_end().to_string()));
            }
        }
        SlashCommand::Stats => {
            let mut breakdown: Vec<(String, usize)> = app.stats.tool_call_breakdown
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            breakdown.sort_by(|a, b| b.1.cmp(&a.1));
            app.messages.push(ChatMessage::StatsInfo(StatsInfoData {
                session_duration: app.stats.session_start.elapsed(),
                agent_turns: app.stats.agent_turns,
                tool_call_count: app.stats.tool_call_count,
                failed_tool_call_count: app.stats.failed_tool_call_count,
                tool_call_breakdown: breakdown,
                input_tokens: app.stats.input_tokens,
                output_tokens: app.stats.output_tokens,
                context_trims: app.stats.context_trims,
                model: app.model.clone(),
            }));
        }
        SlashCommand::Unknown(name) => {
            let skill_name = name.trim_start_matches('/');
            if let Some(skill) = app.skills.iter().find(|s| s.name == skill_name) {
                match skill.load_instructions() {
                    Ok(instructions) => {
                        // Extract user args after the command name
                        let args = raw_input
                            .trim()
                            .strip_prefix(&name)
                            .unwrap_or("")
                            .trim();

                        // Show skill load indicator
                        app.messages.push(ChatMessage::SkillLoad {
                            name: skill_name.to_string(),
                        });

                        // Build the prompt: instructions + optional user args
                        let prompt = if args.is_empty() {
                            instructions
                        } else {
                            format!("{}\n\nUser input: {}", instructions, args)
                        };

                        app.begin_processing(pick_verb());
                        let _ = input_tx.send(AgentInput::Message(prompt));
                    }
                    Err(e) => {
                        app.messages.push(ChatMessage::Error(format!(
                            "Failed to load skill '{}': {}",
                            skill_name, e
                        )));
                    }
                }
            } else {
                app.messages
                    .push(ChatMessage::Info(format!("Unknown command: {}", name)));
            }
        }
    }

    app.input.clear();
    app.cursor_pos = 0;

    if was_at_bottom {
        app.scroll_offset = 0;
    }
}
