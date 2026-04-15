use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

use crate::config::{
    Config, DEFAULT_BASH_TIMEOUT_SECS, DEFAULT_CONTEXT_SIZE, DEFAULT_OLLAMA_URL,
    DEFAULT_REINJECTION_INTERVAL, DEFAULT_SUBAGENT_MAX_TURNS, DEFAULT_TRIM_TARGET_PCT,
    DEFAULT_TRIM_THRESHOLD_PCT,
};

// ── Result type ──────────────────────────────────────────────────────────

pub(crate) enum SettingsResult {
    /// Panel is still active.
    Active,
    /// Panel was dismissed (Esc from top level).
    Dismissed,
    /// A config value was modified; caller should save.
    Modified,
}

// ── View state ───────────────────────────────────────────────────────────

enum View {
    Categories,
    Fields(usize),
    /// (category, field_idx, edit_buffer, cursor_pos)
    Edit(usize, usize, String, usize),
}

// ── Field definitions ────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum FieldKind {
    Bool,
    Str,
    U8,
    U16,
    U32,
    U64,
    F64,
}

#[derive(Clone, Copy)]
struct FieldDef {
    name: &'static str,
    kind: FieldKind,
}

const CATEGORIES: &[&str] = &[
    "Model & Backend",
    "Sampling",
    "Server",
    "Behavior",
    "Context Management",
    "Cost",
];

const MODEL_FIELDS: &[FieldDef] = &[
    FieldDef { name: "model", kind: FieldKind::Str },
    FieldDef { name: "backend", kind: FieldKind::Str },
    FieldDef { name: "context_size", kind: FieldKind::U64 },
];

const SAMPLING_FIELDS: &[FieldDef] = &[
    FieldDef { name: "temperature", kind: FieldKind::F64 },
    FieldDef { name: "top_p", kind: FieldKind::F64 },
    FieldDef { name: "top_k", kind: FieldKind::U32 },
];

const SERVER_FIELDS: &[FieldDef] = &[
    FieldDef { name: "ollama_url", kind: FieldKind::Str },
    FieldDef { name: "llama_server_path", kind: FieldKind::Str },
    FieldDef { name: "llama_server_url", kind: FieldKind::Str },
];

const BEHAVIOR_FIELDS: &[FieldDef] = &[
    FieldDef { name: "no_confirm", kind: FieldKind::Bool },
    FieldDef { name: "bypass", kind: FieldKind::Bool },
    FieldDef { name: "verbose", kind: FieldKind::Bool },
    FieldDef { name: "bash_timeout", kind: FieldKind::U64 },
    FieldDef { name: "subagent_max_turns", kind: FieldKind::U16 },
    FieldDef { name: "tool_scoping", kind: FieldKind::Bool },
];

const CONTEXT_FIELDS: &[FieldDef] = &[
    FieldDef { name: "trim_threshold", kind: FieldKind::U8 },
    FieldDef { name: "trim_target", kind: FieldKind::U8 },
    FieldDef { name: "context_compaction", kind: FieldKind::Bool },
    FieldDef { name: "task_reinjection", kind: FieldKind::Bool },
    FieldDef { name: "reinjection_interval", kind: FieldKind::U16 },
];

const COST_FIELDS: &[FieldDef] = &[
    FieldDef { name: "show_cost_estimate", kind: FieldKind::Bool },
];

fn fields_for_category(cat: usize) -> &'static [FieldDef] {
    match cat {
        0 => MODEL_FIELDS,
        1 => SAMPLING_FIELDS,
        2 => SERVER_FIELDS,
        3 => BEHAVIOR_FIELDS,
        4 => CONTEXT_FIELDS,
        5 => COST_FIELDS,
        _ => &[],
    }
}

// ── SettingsPanel ────────────────────────────────────────────────────────

pub(crate) struct SettingsPanel {
    view: View,
    selected: usize,
    scroll_offset: usize,
    config: Config,
}

impl SettingsPanel {
    pub fn new(config: &Config) -> Self {
        Self {
            view: View::Categories,
            selected: 0,
            scroll_offset: 0,
            config: config.clone(),
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    // ── Value display ────────────────────────────────────────────────

    fn display_value(&self, field: &FieldDef) -> String {
        match field.name {
            "model" => opt_str(&self.config.model, "(not set)"),
            "backend" => opt_str(&self.config.backend, "ollama"),
            "context_size" => opt_num(self.config.context_size, DEFAULT_CONTEXT_SIZE),
            "temperature" => opt_f64(self.config.temperature),
            "top_p" => opt_f64(self.config.top_p),
            "top_k" => self.config.top_k.map_or_else(|| "(not set)".into(), |v| v.to_string()),
            "ollama_url" => opt_str(&self.config.ollama_url, DEFAULT_OLLAMA_URL),
            "llama_server_path" => opt_str(&self.config.llama_server_path, "(not set)"),
            "llama_server_url" => opt_str(&self.config.llama_server_url, "(not set)"),
            "no_confirm" => bool_display(self.config.no_confirm.unwrap_or(false)),
            "bypass" => bool_display(self.config.bypass.unwrap_or(false)),
            "verbose" => bool_display(self.config.verbose.unwrap_or(false)),
            "bash_timeout" => opt_num(self.config.bash_timeout, DEFAULT_BASH_TIMEOUT_SECS),
            "subagent_max_turns" => opt_num(self.config.subagent_max_turns, DEFAULT_SUBAGENT_MAX_TURNS),
            "tool_scoping" => bool_display(self.config.tool_scoping.unwrap_or(false)),
            "trim_threshold" => opt_num(self.config.trim_threshold, DEFAULT_TRIM_THRESHOLD_PCT),
            "trim_target" => opt_num(self.config.trim_target, DEFAULT_TRIM_TARGET_PCT),
            "context_compaction" => bool_display(self.config.context_compaction.unwrap_or(true)),
            "task_reinjection" => bool_display(self.config.task_reinjection.unwrap_or(false)),
            "reinjection_interval" => opt_num(self.config.reinjection_interval, DEFAULT_REINJECTION_INTERVAL),
            "show_cost_estimate" => bool_display(self.config.show_cost_estimate.unwrap_or(false)),
            _ => "(unknown)".into(),
        }
    }

    /// Raw value for the edit buffer (empty string = not set).
    fn raw_value(&self, key: &str) -> String {
        match key {
            "model" => self.config.model.clone().unwrap_or_default(),
            "backend" => self.config.backend.clone().unwrap_or_default(),
            "context_size" => self.config.context_size.map_or_else(String::new, |v| v.to_string()),
            "temperature" => self.config.temperature.map_or_else(String::new, |v| v.to_string()),
            "top_p" => self.config.top_p.map_or_else(String::new, |v| v.to_string()),
            "top_k" => self.config.top_k.map_or_else(String::new, |v| v.to_string()),
            "ollama_url" => self.config.ollama_url.clone().unwrap_or_default(),
            "llama_server_path" => self.config.llama_server_path.clone().unwrap_or_default(),
            "llama_server_url" => self.config.llama_server_url.clone().unwrap_or_default(),
            "bash_timeout" => self.config.bash_timeout.map_or_else(String::new, |v| v.to_string()),
            "subagent_max_turns" => self.config.subagent_max_turns.map_or_else(String::new, |v| v.to_string()),
            "trim_threshold" => self.config.trim_threshold.map_or_else(String::new, |v| v.to_string()),
            "trim_target" => self.config.trim_target.map_or_else(String::new, |v| v.to_string()),
            "reinjection_interval" => self.config.reinjection_interval.map_or_else(String::new, |v| v.to_string()),
            _ => String::new(),
        }
    }

    // ── Value mutation ───────────────────────────────────────────────

    fn toggle_bool(&mut self, key: &str) {
        match key {
            "no_confirm" => self.config.no_confirm = Some(!self.config.no_confirm.unwrap_or(false)),
            "bypass" => self.config.bypass = Some(!self.config.bypass.unwrap_or(false)),
            "verbose" => self.config.verbose = Some(!self.config.verbose.unwrap_or(false)),
            "tool_scoping" => self.config.tool_scoping = Some(!self.config.tool_scoping.unwrap_or(false)),
            "task_reinjection" => self.config.task_reinjection = Some(!self.config.task_reinjection.unwrap_or(false)),
            "context_compaction" => self.config.context_compaction = Some(!self.config.context_compaction.unwrap_or(true)),
            "show_cost_estimate" => self.config.show_cost_estimate = Some(!self.config.show_cost_estimate.unwrap_or(false)),
            _ => {}
        }
    }

    /// Apply a string value to the config field. Returns true on success.
    fn set_value(&mut self, key: &str, value: &str) -> bool {
        let value = value.trim();
        let none = value.is_empty();
        match key {
            "model" => { self.config.model = if none { None } else { Some(value.into()) }; }
            "backend" => { self.config.backend = if none { None } else { Some(value.into()) }; }
            "ollama_url" => { self.config.ollama_url = if none { None } else { Some(value.into()) }; }
            "llama_server_path" => { self.config.llama_server_path = if none { None } else { Some(value.into()) }; }
            "llama_server_url" => { self.config.llama_server_url = if none { None } else { Some(value.into()) }; }
            "context_size" => return parse_opt(&mut self.config.context_size, value),
            "bash_timeout" => return parse_opt(&mut self.config.bash_timeout, value),
            "temperature" => return parse_opt(&mut self.config.temperature, value),
            "top_p" => return parse_opt(&mut self.config.top_p, value),
            "top_k" => return parse_opt(&mut self.config.top_k, value),
            "subagent_max_turns" => return parse_opt(&mut self.config.subagent_max_turns, value),
            "trim_threshold" => return parse_opt(&mut self.config.trim_threshold, value),
            "trim_target" => return parse_opt(&mut self.config.trim_target, value),
            "reinjection_interval" => return parse_opt(&mut self.config.reinjection_interval, value),
            _ => return false,
        }
        true
    }

    // ── Key handling ─────────────────────────────────────────────────

    pub fn handle_key(&mut self, key: KeyEvent) -> SettingsResult {
        // Snapshot view discriminant to avoid borrow conflicts with &mut self methods.
        let (tag, category, field_idx) = match &self.view {
            View::Categories => (0u8, 0, 0),
            View::Fields(c) => (1, *c, 0),
            View::Edit(c, f, _, _) => (2, *c, *f),
        };

        match tag {
            0 => self.handle_categories_key(key),
            1 => self.handle_fields_key(key, category),
            _ => self.handle_edit_key(key, category, field_idx),
        }
    }

    fn handle_categories_key(&mut self, key: KeyEvent) -> SettingsResult {
        match key.code {
            KeyCode::Esc => SettingsResult::Dismissed,
            KeyCode::Up => { self.move_up(); SettingsResult::Active }
            KeyCode::Down => { self.move_down(CATEGORIES.len()); SettingsResult::Active }
            KeyCode::Char('k' | 'p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.move_up(); SettingsResult::Active
            }
            KeyCode::Char('j' | 'n') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.move_down(CATEGORIES.len()); SettingsResult::Active
            }
            KeyCode::Enter => {
                let cat = self.selected;
                self.view = View::Fields(cat);
                self.selected = 0;
                self.scroll_offset = 0;
                SettingsResult::Active
            }
            _ => SettingsResult::Active,
        }
    }

    fn handle_fields_key(&mut self, key: KeyEvent, category: usize) -> SettingsResult {
        let fields = fields_for_category(category);
        match key.code {
            KeyCode::Esc => {
                self.view = View::Categories;
                self.selected = category;
                self.scroll_offset = 0;
                SettingsResult::Active
            }
            KeyCode::Up => { self.move_up(); SettingsResult::Active }
            KeyCode::Down => { self.move_down(fields.len()); SettingsResult::Active }
            KeyCode::Char('k' | 'p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.move_up(); SettingsResult::Active
            }
            KeyCode::Char('j' | 'n') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.move_down(fields.len()); SettingsResult::Active
            }
            KeyCode::Enter => {
                if let Some(field) = fields.get(self.selected) {
                    if matches!(field.kind, FieldKind::Bool) {
                        self.toggle_bool(field.name);
                        SettingsResult::Modified
                    } else {
                        let raw = self.raw_value(field.name);
                        let cursor = raw.len();
                        self.view = View::Edit(category, self.selected, raw, cursor);
                        SettingsResult::Active
                    }
                } else {
                    SettingsResult::Active
                }
            }
            _ => SettingsResult::Active,
        }
    }

    fn handle_edit_key(&mut self, key: KeyEvent, category: usize, field_idx: usize) -> SettingsResult {
        match key.code {
            KeyCode::Esc => {
                self.view = View::Fields(category);
                self.selected = field_idx;
                SettingsResult::Active
            }
            KeyCode::Enter => {
                let value = match &self.view {
                    View::Edit(_, _, buf, _) => buf.clone(),
                    _ => return SettingsResult::Active,
                };
                let fields = fields_for_category(category);
                let ok = fields.get(field_idx).is_some_and(|f| self.set_value(f.name, &value));
                self.view = View::Fields(category);
                self.selected = field_idx;
                if ok { SettingsResult::Modified } else { SettingsResult::Active }
            }
            _ => {
                // Text editing within the buffer.
                if let View::Edit(_, _, ref mut buffer, ref mut cursor) = self.view {
                    Self::edit_buffer(key, buffer, cursor);
                }
                SettingsResult::Active
            }
        }
    }

    fn edit_buffer(key: KeyEvent, buffer: &mut String, cursor: &mut usize) {
        match key.code {
            KeyCode::Char(c) if !key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => {
                buffer.insert(*cursor, c);
                *cursor += c.len_utf8();
            }
            KeyCode::Backspace => {
                if *cursor > 0 {
                    let prev = buffer[..*cursor]
                        .char_indices()
                        .next_back()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    buffer.remove(prev);
                    *cursor = prev;
                }
            }
            KeyCode::Delete => {
                if *cursor < buffer.len() {
                    buffer.remove(*cursor);
                }
            }
            KeyCode::Left => {
                if *cursor > 0 {
                    *cursor = buffer[..*cursor]
                        .char_indices()
                        .next_back()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                }
            }
            KeyCode::Right => {
                if *cursor < buffer.len() {
                    *cursor = buffer[*cursor..]
                        .char_indices()
                        .nth(1)
                        .map(|(i, _)| *cursor + i)
                        .unwrap_or(buffer.len());
                }
            }
            KeyCode::Home => { *cursor = 0; }
            KeyCode::End => { *cursor = buffer.len(); }
            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => { *cursor = 0; }
            KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => { *cursor = buffer.len(); }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                buffer.drain(..*cursor);
                *cursor = 0;
            }
            KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                buffer.truncate(*cursor);
            }
            _ => {}
        }
    }

    fn move_up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    fn move_down(&mut self, count: usize) {
        if self.selected + 1 < count {
            self.selected += 1;
        }
    }

    // ── Rendering ────────────────────────────────────────────────────

    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        // Snapshot view parameters so we can call &mut self methods freely.
        let (tag, cat, field_idx, buf, cur) = match &self.view {
            View::Categories => (0u8, 0, 0, String::new(), 0),
            View::Fields(c) => (1, *c, 0, String::new(), 0),
            View::Edit(c, fi, b, cu) => (2, *c, *fi, b.clone(), *cu),
        };

        match tag {
            0 => {
                let items: Vec<(&str, String)> = CATEGORIES.iter()
                    .map(|&name| (name, String::new()))
                    .collect();
                self.render_panel(f, area, "Settings", &items,
                    "\u{2191}\u{2193} navigate \u{00b7} enter select \u{00b7} esc close", None);
            }
            1 => {
                let fields = fields_for_category(cat);
                let items: Vec<(&str, String)> = fields.iter()
                    .map(|fd| (fd.name, self.display_value(fd)))
                    .collect();
                let hint = if fields.get(self.selected).is_some_and(|f| matches!(f.kind, FieldKind::Bool)) {
                    "enter toggle \u{00b7} esc back"
                } else {
                    "enter edit \u{00b7} esc back"
                };
                self.render_panel(f, area, CATEGORIES[cat], &items, hint, None);
            }
            _ => {
                let fields = fields_for_category(cat);
                let items: Vec<(&str, String)> = fields.iter()
                    .map(|fd| (fd.name, self.display_value(fd)))
                    .collect();
                let field_name = fields.get(field_idx).map_or("?", |f| f.name);
                self.render_panel(f, area, CATEGORIES[cat], &items,
                    "enter save \u{00b7} esc cancel",
                    Some((field_idx, field_name, &buf, cur)));
            }
        }
    }

    /// Core render: draws a centered panel with an item list, optional edit line, and hint.
    fn render_panel(
        &mut self,
        f: &mut Frame,
        area: Rect,
        title: &str,
        items: &[(&str, String)],
        hint: &str,
        edit: Option<(usize, &str, &str, usize)>,
    ) {
        let max_width = 60u16.min(area.width.saturating_sub(4)).max(20);
        let item_count = items.len().max(1) as u16;
        let max_visible = area.height.saturating_sub(if edit.is_some() { 9 } else { 7 }).max(3);
        let visible = item_count.min(max_visible);
        let extra = if edit.is_some() { 2 } else { 0 }; // separator + edit line
        let total_height = visible + 3 + extra; // borders(2) + items + hint + extra

        let x = area.x + (area.width.saturating_sub(max_width)) / 2;
        let y = area.y + (area.height.saturating_sub(total_height)) / 2;
        let rect = Rect::new(x, y, max_width, total_height);

        f.render_widget(Clear, rect);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(
                format!(" {} ", title),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ));
        f.render_widget(block, rect);

        let inner = Rect::new(rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2);
        let content_width = inner.width as usize;

        // Scroll offset — keep selected (or edited) item visible.
        let focus = edit.map_or(self.selected, |(idx, _, _, _)| idx);
        let vis = visible as usize;
        if focus >= self.scroll_offset + vis {
            self.scroll_offset = focus + 1 - vis;
        }
        if focus < self.scroll_offset {
            self.scroll_offset = focus;
        }
        let scroll = self.scroll_offset;

        // Render items.
        for (row, (name, value)) in items.iter().skip(scroll).take(vis).enumerate() {
            let idx = scroll + row;
            let (is_highlight, highlight_color) = if let Some((edit_idx, _, _, _)) = edit {
                (idx == edit_idx, Color::Cyan)
            } else {
                (idx == self.selected, Color::Yellow)
            };

            let marker = if is_highlight { "> " } else { "  " };
            let marker_style = if is_highlight {
                Style::default().fg(highlight_color).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            let label_style = if is_highlight {
                Style::default().fg(highlight_color).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };

            let mut spans = vec![
                Span::styled(marker.to_string(), marker_style),
                Span::styled(name.to_string(), label_style),
            ];

            if !value.is_empty() {
                let used = marker.len() + name.chars().count();
                let val_len = value.chars().count() + 1;
                if used + val_len < content_width {
                    spans.push(Span::raw(" ".repeat(content_width - used - val_len)));
                } else {
                    spans.push(Span::raw(" "));
                }
                spans.push(Span::styled(value.to_string(), Style::default().fg(Color::DarkGray)));
            }

            f.render_widget(
                Paragraph::new(Line::from(spans)),
                Rect::new(inner.x, inner.y + row as u16, inner.width, 1),
            );
        }

        let mut next_y = inner.y + visible;

        // Edit mode: separator + edit input.
        if let Some((_edit_idx, field_name, buffer, cursor)) = edit {
            // Separator
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    "\u{2500}".repeat(content_width),
                    Style::default().fg(Color::DarkGray),
                ))),
                Rect::new(inner.x, next_y, inner.width, 1),
            );
            next_y += 1;

            // Edit line
            let prefix = format!("{}: ", field_name);
            let display = if buffer.is_empty() {
                Span::styled("(empty \u{2014} clears value)", Style::default().fg(Color::DarkGray))
            } else {
                Span::raw(buffer.to_string())
            };
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(
                        prefix.clone(),
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                    ),
                    display,
                ])),
                Rect::new(inner.x, next_y, inner.width, 1),
            );

            // Cursor
            let prefix_chars = prefix.chars().count() as u16;
            let cursor = cursor.min(buffer.len());
            let cursor_chars = buffer[..cursor].chars().count() as u16;
            f.set_cursor_position((inner.x + prefix_chars + cursor_chars, next_y));
            next_y += 1;
        }

        // Hint line.
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(hint, Style::default().fg(Color::DarkGray)))),
            Rect::new(inner.x, next_y, inner.width, 1),
        );
    }
}

// ── Private helpers ──────────────────────────────────────────────────────

fn opt_str(val: &Option<String>, default: &str) -> String {
    val.as_deref().unwrap_or(default).to_string()
}

fn opt_num<T: std::fmt::Display>(val: Option<T>, default: T) -> String {
    match val {
        Some(v) => v.to_string(),
        None => default.to_string(),
    }
}

fn opt_f64(val: Option<f64>) -> String {
    val.map_or_else(|| "(not set)".into(), |v| v.to_string())
}

fn bool_display(v: bool) -> String {
    (if v { "true" } else { "false" }).into()
}

/// Parse a string into an Option<T>, setting None for empty strings.
fn parse_opt<T: std::str::FromStr>(field: &mut Option<T>, value: &str) -> bool {
    if value.is_empty() {
        *field = None;
        return true;
    }
    match value.parse() {
        Ok(v) => { *field = Some(v); true }
        Err(_) => false,
    }
}
