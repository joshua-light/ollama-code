//! Keyboard input handling when the app is in normal (chat input) mode.
//! Also handles prompt-template variable fill mode since it overlays input.

use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tokio::sync::mpsc;

use crate::session::Session;
use crate::tui::app::{AgentInput, App, ChatMessage};

use super::commands;

/// Find the byte offset of the previous word boundary (for Ctrl+W, Alt+Left).
pub(in crate::tui) fn prev_word_boundary(input: &str, pos: usize) -> usize {
    let before = &input[..pos];
    // Skip trailing whitespace, then skip word chars
    let trimmed = before.trim_end();
    if trimmed.is_empty() {
        return 0;
    }
    // Find last whitespace before the word
    trimmed
        .rfind(|c: char| c.is_whitespace())
        .map(|i| {
            // i is byte offset of the whitespace char; step past it
            i + trimmed[i..].chars().next().unwrap().len_utf8()
        })
        .unwrap_or(0)
}

/// Find the byte offset of the next word boundary (for Alt+Right).
fn next_word_boundary(input: &str, pos: usize) -> usize {
    let after = &input[pos..];
    // Skip leading whitespace, then skip word chars
    let skipped_ws = after.trim_start().len();
    let ws_bytes = after.len() - skipped_ws;
    let rest = &after[ws_bytes..];
    // Find first whitespace in the remaining word
    rest.find(|c: char| c.is_whitespace())
        .map(|i| pos + ws_bytes + i)
        .unwrap_or(input.len())
}

pub(super) fn handle_normal_input(
    key: KeyEvent,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    model_tx: &mpsc::UnboundedSender<Result<Vec<String>>>,
    session: &mut Session,
) {
    // ── Prompt template variable fill mode ───────────────────────────
    if app.pending_prompt.is_some() {
        match key.code {
            KeyCode::Esc => {
                app.pending_prompt = None;
                app.input.clear();
                app.cursor_pos = 0;
                app.messages
                    .push(ChatMessage::Info("Prompt cancelled.".into()));
                return;
            }
            KeyCode::Enter => {
                let fill = app.pending_prompt.as_mut().unwrap();
                let var = &fill.variables[fill.current];
                let value = if !app.input.trim().is_empty() {
                    app.input.trim().to_string()
                } else if let Some(ref default) = var.default {
                    default.clone()
                } else {
                    return;
                };
                let var_name = var.name.clone();
                fill.values.insert(var_name, value);
                fill.current += 1;
                app.input.clear();
                app.cursor_pos = 0;

                // Check if all variables are filled
                let fill = app.pending_prompt.as_ref().unwrap();
                if fill.current >= fill.variables.len() {
                    let fill = app.pending_prompt.take().unwrap();
                    let expanded = fill.template.expand(&fill.values);
                    let was_at_bottom = app.is_at_bottom();
                    app.messages.push(ChatMessage::SkillLoad {
                        name: format!("prompt:{}", fill.template.name),
                    });
                    app.begin_processing(crate::tui::render::pick_verb());
                    let _ = input_tx.send(AgentInput::Message(expanded));
                    if was_at_bottom {
                        app.scroll_offset = 0;
                    }
                }
                return;
            }
            // Allow normal text editing during fill mode — fall through
            _ => {}
        }
    }

    // Ctrl key combos
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('a') => {
                app.cursor_pos = 0;
                return;
            }
            KeyCode::Char('e') => {
                app.cursor_pos = app.input.len();
                return;
            }
            KeyCode::Char('w') => {
                app.delete_word_backward();
                return;
            }
            KeyCode::Char('k') => {
                app.input.truncate(app.cursor_pos);
                return;
            }
            KeyCode::Char('u') => {
                app.input.drain(..app.cursor_pos);
                app.cursor_pos = 0;
                return;
            }
            KeyCode::Char('d') => {
                if app.input.is_empty() {
                    app.should_quit = true;
                } else {
                    app.delete_forward();
                }
                return;
            }
            _ => {}
        }
    }

    // Alt key combos
    if key.modifiers.contains(KeyModifiers::ALT) {
        match key.code {
            // Alt+Enter — queue follow-up (sent after agent completes)
            KeyCode::Enter => {
                if app.is_processing {
                    app.submit_followup();
                } else {
                    // When idle, Alt+Enter behaves like Enter
                    if let Some(msg) = app.submit() {
                        let _ = input_tx.send(AgentInput::Message(msg));
                    }
                }
                return;
            }
            KeyCode::Left => {
                app.cursor_pos = prev_word_boundary(&app.input, app.cursor_pos);
                return;
            }
            KeyCode::Right => {
                app.cursor_pos = next_word_boundary(&app.input, app.cursor_pos);
                return;
            }
            KeyCode::Backspace => {
                app.delete_word_backward();
                return;
            }
            _ => {}
        }
    }

    match key.code {
        KeyCode::Enter => {
            if key.modifiers.contains(KeyModifiers::SHIFT) {
                app.input.insert(app.cursor_pos, '\n');
                app.cursor_pos += 1;
            } else if let Some(cmd) = crate::commands::parse(&app.input) {
                // Commands are only executed when idle
                if !app.is_processing {
                    let raw_input = app.input.clone();
                    commands::handle_command(cmd, &raw_input, app, input_tx, model_tx, session);
                }
            } else if let Some(msg) = app.submit() {
                // submit() handles both idle (returns Some) and processing
                // (sends steering message via steer_tx, returns None)
                let _ = input_tx.send(AgentInput::Message(msg));
            }
        }
        KeyCode::Tab => {
            let matches = crate::commands::completions(&app.input);
            if let Some(cmd) = matches.first() {
                app.input = cmd.name.to_string();
                app.cursor_pos = app.input.len();
            } else {
                // Check skill and prompt name completions
                let prefix = app.input.trim();
                if let Some(skill_prefix) = prefix.strip_prefix('/') {
                    if let Some(skill) = app.skills.iter().find(|s| s.name.starts_with(skill_prefix))
                    {
                        app.input = format!("/{}", skill.name);
                        app.cursor_pos = app.input.len();
                    } else if let Some(prompt) = app
                        .prompts
                        .iter()
                        .find(|p| p.name.starts_with(skill_prefix))
                    {
                        app.input = format!("/{}", prompt.name);
                        app.cursor_pos = app.input.len();
                    }
                }
            }
        }
        KeyCode::Up => {
            if app.input_history.is_empty() {
                app.scroll_up(1);
                return;
            }
            match app.history_index {
                None => {
                    // Save current input as draft, jump to last history entry
                    app.history_draft = app.input.clone();
                    let idx = app.input_history.len() - 1;
                    app.history_index = Some(idx);
                    app.input = app.input_history[idx].clone();
                    app.cursor_pos = app.input.len();
                }
                Some(idx) if idx > 0 => {
                    let new_idx = idx - 1;
                    app.history_index = Some(new_idx);
                    app.input = app.input_history[new_idx].clone();
                    app.cursor_pos = app.input.len();
                }
                _ => {} // already at oldest entry
            }
        }
        KeyCode::Down => match app.history_index {
            Some(idx) if idx + 1 < app.input_history.len() => {
                let new_idx = idx + 1;
                app.history_index = Some(new_idx);
                app.input = app.input_history[new_idx].clone();
                app.cursor_pos = app.input.len();
            }
            Some(_) => {
                // Past last entry — restore draft
                app.history_index = None;
                app.input = std::mem::take(&mut app.history_draft);
                app.cursor_pos = app.input.len();
            }
            None => {
                app.scroll_down(1);
            }
        },
        KeyCode::Char(c)
            if !key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT) =>
        {
            app.input.insert(app.cursor_pos, c);
            app.cursor_pos += c.len_utf8();
        }
        KeyCode::Backspace => {
            app.backspace();
        }
        KeyCode::Delete => {
            app.delete_forward();
        }
        KeyCode::Left => {
            if app.cursor_pos > 0 {
                app.cursor_pos = app.input[..app.cursor_pos]
                    .char_indices()
                    .next_back()
                    .map(|(i, _)| i)
                    .unwrap_or(0);
            }
        }
        KeyCode::Right => {
            if app.cursor_pos < app.input.len() {
                app.cursor_pos = app.input[app.cursor_pos..]
                    .char_indices()
                    .nth(1)
                    .map(|(i, _)| app.cursor_pos + i)
                    .unwrap_or(app.input.len());
            }
        }
        KeyCode::Home => {
            app.cursor_pos = 0;
        }
        KeyCode::End => {
            app.cursor_pos = app.input.len();
        }
        _ => {}
    }
}
