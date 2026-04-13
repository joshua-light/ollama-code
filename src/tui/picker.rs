use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

pub(crate) struct PickerItem {
    pub label: String,
    pub hint: String,
}

pub(crate) enum PickerKind {
    Model,
    Resume,
}

pub(crate) enum PickerResult {
    /// User selected an item (index into original `items`).
    Selected(usize),
    /// User pressed Enter with a non-empty filter but no matching items.
    FreeText(String),
    Cancelled,
    /// Picker is still active — no action needed.
    Active,
}

pub(crate) struct Picker {
    pub title: String,
    pub items: Vec<PickerItem>,
    /// Indices into `items` that match the current filter.
    filtered: Vec<usize>,
    /// Index into `filtered` for the highlighted row.
    selected: usize,
    pub filter: String,
    filter_cursor: usize,
    /// Scroll offset for the visible list window.
    scroll_offset: usize,
    pub kind: PickerKind,
}

impl Picker {
    pub fn new(title: impl Into<String>, items: Vec<PickerItem>, kind: PickerKind) -> Self {
        let filtered: Vec<usize> = (0..items.len()).collect();
        Self {
            title: title.into(),
            items,
            filtered,
            selected: 0,
            filter: String::new(),
            filter_cursor: 0,
            scroll_offset: 0,
            kind,
        }
    }

    /// Process a key event and return the result.
    pub fn handle_key(&mut self, key: KeyEvent) -> PickerResult {
        match key.code {
            KeyCode::Esc => return PickerResult::Cancelled,
            KeyCode::Enter => return self.confirm(),
            KeyCode::Up => self.move_up(),
            KeyCode::Down => self.move_down(),
            KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => self.move_up(),
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => self.move_up(),
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => self.move_down(),
            KeyCode::Char('n') if key.modifiers.contains(KeyModifiers::CONTROL) => self.move_down(),
            KeyCode::Char(c) => {
                self.filter.insert(self.filter_cursor, c);
                self.filter_cursor += c.len_utf8();
                self.refilter();
            }
            KeyCode::Backspace => {
                if self.filter_cursor > 0 {
                    let prev = self.filter[..self.filter_cursor]
                        .char_indices()
                        .next_back()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    self.filter.remove(prev);
                    self.filter_cursor = prev;
                    self.refilter();
                }
            }
            _ => {}
        }
        PickerResult::Active
    }

    /// Return the label of a selected item by its original index.
    pub fn item_label(&self, idx: usize) -> &str {
        &self.items[idx].label
    }

    fn confirm(&self) -> PickerResult {
        if let Some(&idx) = self.filtered.get(self.selected) {
            PickerResult::Selected(idx)
        } else if !self.filter.trim().is_empty() {
            PickerResult::FreeText(self.filter.trim().to_string())
        } else {
            PickerResult::Active
        }
    }

    fn move_up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    fn move_down(&mut self) {
        if !self.filtered.is_empty() && self.selected + 1 < self.filtered.len() {
            self.selected += 1;
        }
    }

    fn refilter(&mut self) {
        let query = self.filter.to_lowercase();
        self.filtered = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| {
                if query.is_empty() {
                    return true;
                }
                item.label.to_lowercase().contains(&query)
            })
            .map(|(i, _)| i)
            .collect();
        self.selected = 0;
        self.scroll_offset = 0;
    }

    /// Render the picker as a centered floating panel.
    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        let max_width = 60u16.min(area.width.saturating_sub(4)).max(20);
        // Height: border(1) + filter(1) + separator(1) + items + hint(1) + border(1)
        let list_height = self.filtered.len().max(1) as u16;
        let max_list_height = area.height.saturating_sub(9).max(3);
        let visible_list = list_height.min(max_list_height);
        let total_height = visible_list + 5; // top border + filter + sep + list + hint + bottom border

        // Center the panel
        let x = area.x + (area.width.saturating_sub(max_width)) / 2;
        let y = area.y + (area.height.saturating_sub(total_height)) / 2;
        let rect = Rect::new(x, y, max_width, total_height);

        // Clear background
        f.render_widget(Clear, rect);

        // Border block
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(
                format!(" {} ", self.title),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ));
        f.render_widget(block, rect);

        let inner = Rect::new(rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2);
        let content_width = inner.width as usize;

        // -- Filter line --
        let filter_display = if self.filter.is_empty() {
            Span::styled("type to filter...", Style::default().fg(Color::DarkGray))
        } else {
            Span::raw(self.filter.clone())
        };
        let filter_line = Line::from(vec![
            Span::styled("> ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            filter_display,
        ]);
        f.render_widget(
            Paragraph::new(filter_line),
            Rect::new(inner.x, inner.y, inner.width, 1),
        );

        // Cursor position in filter
        let cursor_chars = self.filter[..self.filter_cursor].chars().count();
        f.set_cursor_position((inner.x + 2 + cursor_chars as u16, inner.y));

        // -- Separator --
        let sep = "─".repeat(content_width);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                sep,
                Style::default().fg(Color::DarkGray),
            ))),
            Rect::new(inner.x, inner.y + 1, inner.width, 1),
        );

        // -- Item list --
        let list_area = Rect::new(inner.x, inner.y + 2, inner.width, visible_list);

        // Ensure selected item is visible and persist scroll state
        let vis = visible_list as usize;
        if self.selected >= self.scroll_offset + vis {
            self.scroll_offset = self.selected + 1 - vis;
        }
        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        }
        let scroll = self.scroll_offset;

        if self.filtered.is_empty() {
            let hint = if self.filter.trim().is_empty() {
                "No items"
            } else {
                "No matches — enter to submit"
            };
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    format!("  {}", hint),
                    Style::default().fg(Color::DarkGray),
                ))),
                Rect::new(list_area.x, list_area.y, list_area.width, 1),
            );
        } else {
            for (row, &item_idx) in self
                .filtered
                .iter()
                .skip(scroll)
                .take(visible_list as usize)
                .enumerate()
            {
                let item = &self.items[item_idx];
                let is_selected = scroll + row == self.selected;

                let marker = if is_selected { "> " } else { "  " };
                let label_style = if is_selected {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };

                let mut spans = vec![
                    Span::styled(
                        marker,
                        if is_selected {
                            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                        } else {
                            Style::default()
                        },
                    ),
                    Span::styled(item.label.clone(), label_style),
                ];

                if !item.hint.is_empty() {
                    // Right-align the hint
                    let used = marker.len() + item.label.len();
                    let hint_len = item.hint.len() + 1; // +1 for space before hint
                    if used + hint_len < content_width {
                        let pad = content_width - used - hint_len;
                        spans.push(Span::raw(" ".repeat(pad)));
                    } else {
                        spans.push(Span::raw(" "));
                    }
                    spans.push(Span::styled(
                        item.hint.clone(),
                        Style::default().fg(Color::DarkGray),
                    ));
                }

                f.render_widget(
                    Paragraph::new(Line::from(spans)),
                    Rect::new(list_area.x, list_area.y + row as u16, list_area.width, 1),
                );
            }
        }

        // -- Hint line --
        let hint_y = inner.y + 2 + visible_list;
        let hint = "↑↓ navigate · enter select · esc cancel";
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                hint,
                Style::default().fg(Color::DarkGray),
            ))),
            Rect::new(inner.x, hint_y, inner.width, 1),
        );
    }
}
