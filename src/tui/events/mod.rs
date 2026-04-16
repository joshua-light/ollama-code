mod agent_events;
mod commands;
mod input;
mod mouse;
mod picker;
mod server;

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use tokio::sync::mpsc;

use crate::session::Session;

use super::app::{App, AgentInput, ChatMessage};
use super::settings::SettingsResult;
use super::tree_browser::TreeBrowserResult;

pub(super) use agent_events::handle_agent_event;
pub(in crate::tui) use input::prev_word_boundary;
pub(crate) use server::spawn_llama_server_inner;

pub(super) fn handle_terminal_event(
    evt: Event,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    confirm_tx: &mpsc::UnboundedSender<bool>,
    model_tx: &mpsc::UnboundedSender<Result<Vec<String>>>,
    session: &mut Session,
) {
    match evt {
        Event::Paste(text) => {
            if app.pending_confirm.is_none() && app.picker.is_none() {
                app.input.insert_str(app.cursor_pos, &text);
                app.cursor_pos += text.len();
            }
        }
        Event::Mouse(mouse) => mouse::handle_mouse(mouse, app),
        Event::Key(key) if key.kind == KeyEventKind::Press => {
            handle_key(key, app, input_tx, confirm_tx, model_tx, session);
        }
        _ => {}
    }
}

fn handle_key(
    key: KeyEvent,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    confirm_tx: &mpsc::UnboundedSender<bool>,
    model_tx: &mpsc::UnboundedSender<Result<Vec<String>>>,
    session: &mut Session,
) {
    // Ctrl+C has the same meaning in every mode: back out, or quit if already idle.
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        handle_ctrl_c(app, confirm_tx);
        return;
    }

    // Tool confirmation overlay — y/n/Enter/Esc only.
    if app.pending_confirm.is_some() {
        handle_confirm_keys(key, app, confirm_tx);
        return;
    }

    // Ctrl+O toggles expanded tool output and always takes precedence.
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
        app.tools_expanded = !app.tools_expanded;
        app.needs_clear = true;
        return;
    }

    // Mode-specific overlays take all keys.
    if app.settings.is_some() {
        handle_settings_key(key, app);
        return;
    }
    if app.tree_browser.is_some() {
        handle_tree_browser_key(key, app, input_tx, session);
        return;
    }
    if app.picker.is_some() {
        picker::handle_picker_key(key, app, input_tx, session);
        return;
    }

    // PageUp/PageDown always scroll.
    let half_page = (app.max_scroll.max(10) / 2).max(5);
    match key.code {
        KeyCode::PageUp => {
            app.scroll_up(half_page);
            return;
        }
        KeyCode::PageDown => {
            app.scroll_down(half_page);
            return;
        }
        _ => {}
    }

    // Alt+Up/Down: always scroll.
    if key.modifiers.contains(KeyModifiers::ALT) {
        match key.code {
            KeyCode::Up => {
                app.scroll_up(1);
                return;
            }
            KeyCode::Down => {
                app.scroll_down(1);
                return;
            }
            _ => {}
        }
    }

    // Esc: cancel prompt fill, cancel generation, or quit.
    if key.code == KeyCode::Esc {
        if app.pending_prompt.is_some() {
            // Fall through — handle_normal_input cancels prompt-fill mode.
        } else if app.is_processing {
            if app.server.loading.is_some() {
                app.should_quit = true;
            } else {
                app.server
                    .cancel_flag
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
            return;
        } else {
            app.should_quit = true;
            return;
        }
    }

    input::handle_normal_input(key, app, input_tx, model_tx, session);
}

fn handle_ctrl_c(app: &mut App, confirm_tx: &mpsc::UnboundedSender<bool>) {
    if app.pending_confirm.is_some() {
        app.pending_confirm = None;
        let _ = confirm_tx.send(false);
        return;
    }
    if app.settings.is_some() {
        app.dismiss_settings();
        return;
    }
    if app.tree_browser.is_some() {
        app.dismiss_tree_browser();
        return;
    }
    if app.picker.is_some() {
        app.dismiss_picker();
        return;
    }
    if app.input.is_empty() {
        app.should_quit = true;
    } else {
        app.input.clear();
        app.cursor_pos = 0;
    }
}

fn handle_confirm_keys(key: KeyEvent, app: &mut App, confirm_tx: &mpsc::UnboundedSender<bool>) {
    match key.code {
        KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
            app.pending_confirm = None;
            let _ = confirm_tx.send(true);
        }
        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
            app.pending_confirm = None;
            let _ = confirm_tx.send(false);
        }
        _ => {} // ignore other keys
    }
}

fn handle_settings_key(key: KeyEvent, app: &mut App) {
    let result = app.settings.as_mut().unwrap().handle_key(key);
    match result {
        SettingsResult::Active => {}
        SettingsResult::Dismissed => {
            app.dismiss_settings();
        }
        SettingsResult::Modified => {
            let config = app.settings.as_ref().unwrap().config().clone();
            if let Err(e) = config.save() {
                app.messages.push(ChatMessage::Error(format!(
                    "Warning: could not save config: {}",
                    e
                )));
            }
            app.config = config;
        }
    }
}

fn handle_tree_browser_key(
    key: KeyEvent,
    app: &mut App,
    input_tx: &mpsc::UnboundedSender<AgentInput>,
    session: &mut Session,
) {
    let result = app.tree_browser.as_mut().unwrap().handle_key(key);
    match result {
        TreeBrowserResult::Cancelled => {
            app.dismiss_tree_browser();
        }
        TreeBrowserResult::Active => {}
        TreeBrowserResult::Selected(entry_id) => {
            app.dismiss_tree_browser();

            if let Err(e) = session.branch(&entry_id) {
                app.messages
                    .push(ChatMessage::Error(format!("Branch failed: {}", e)));
                return;
            }

            let msgs = session.get_branch_path();
            let short_id = if entry_id.len() > 8 {
                &entry_id[..8]
            } else {
                &entry_id
            };
            picker::restore_messages(
                app,
                input_tx,
                msgs,
                &format!("Switched to branch at {}.", short_id),
            );
        }
    }
}
