//! Mouse event handling: scroll, click-to-select, drag-to-extend, auto-scroll
//! at edges. Extracted so the keyboard dispatcher in `mod.rs` stays focused.

use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};

use crate::tui::app::{App, TextSelection};

/// Handle a single mouse event. Returns nothing — all state lives on `App`.
pub(super) fn handle_mouse(mouse: MouseEvent, app: &mut App) {
    let chat = app.chat_area;
    let in_chat = chat.height > 0
        && mouse.row >= chat.y
        && mouse.row < chat.y + chat.height
        && mouse.column >= chat.x
        && mouse.column < chat.x + chat.width;

    match mouse.kind {
        MouseEventKind::ScrollUp => {
            if app.picker.is_none() {
                app.scroll_up(5);
            }
        }
        MouseEventKind::ScrollDown => {
            if app.picker.is_none() {
                app.scroll_down(5);
            }
        }
        MouseEventKind::Down(MouseButton::Left) if in_chat => {
            let scroll_top = app.max_scroll.saturating_sub(app.scroll_offset) as usize;
            let content_line = scroll_top + (mouse.row - chat.y) as usize;
            let col = mouse.column.saturating_sub(chat.x);
            app.selection_line_cache.clear();
            app.selection = Some(TextSelection {
                anchor: (content_line, col),
                cursor: (content_line, col),
            });
        }
        MouseEventKind::Down(MouseButton::Left) => {
            // Click outside chat clears selection
            app.selection = None;
            app.selection_line_cache.clear();
        }
        MouseEventKind::Drag(MouseButton::Left) if app.selection.is_some() => {
            // Set continuous auto-scroll direction when near edges
            let edge = 2u16;
            if mouse.row <= chat.y.saturating_add(edge) {
                app.auto_scroll = -1;
                app.scroll_up(1);
            } else if mouse.row >= chat.y + chat.height.saturating_sub(edge) {
                app.auto_scroll = 1;
                app.scroll_down(1);
            } else {
                app.auto_scroll = 0;
            }

            let scroll_top = app.max_scroll.saturating_sub(app.scroll_offset) as usize;
            let row_in_chat =
                mouse.row.clamp(chat.y, chat.y + chat.height.saturating_sub(1)) - chat.y;
            let content_line = scroll_top + row_in_chat as usize;
            let col = mouse
                .column
                .saturating_sub(chat.x)
                .min(chat.width.saturating_sub(1));
            app.selection.as_mut().unwrap().cursor = (content_line, col);
        }
        MouseEventKind::Up(MouseButton::Left) if app.selection.is_some() => {
            app.auto_scroll = 0;
            let sel = app.selection.as_ref().unwrap();
            if sel.anchor == sel.cursor {
                // Zero-width click — clear
                app.selection = None;
                app.selection_line_cache.clear();
            } else {
                // Trigger clipboard copy on next render
                app.copy_selection = true;
            }
        }
        _ => {}
    }
}
