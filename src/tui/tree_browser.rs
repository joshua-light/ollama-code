use std::collections::HashSet;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

use crate::message::Role;
use crate::session::TreeNode;

pub(crate) struct FlatTreeItem {
    pub id: String,
    pub depth: usize,
    /// Whether this node is the last sibling at its level.
    pub is_last: bool,
    /// For each ancestor level, whether that ancestor has more siblings below
    /// (determines whether to draw │ or blank in the gutter).
    pub ancestor_has_more: Vec<bool>,
    pub role: Role,
    pub summary: String,
}

pub(crate) enum TreeBrowserResult {
    /// User selected a node to branch to.
    Selected(String),
    Cancelled,
    /// Browser is still active.
    Active,
}

pub(crate) struct TreeBrowser {
    items: Vec<FlatTreeItem>,
    selected: usize,
    scroll_offset: usize,
    /// Entry IDs on the active branch path (root to current leaf).
    active_path: HashSet<String>,
    /// Current leaf ID.
    leaf_id: Option<String>,
}

impl TreeBrowser {
    pub fn new(tree: Vec<TreeNode>, leaf_id: Option<&str>, active_path: HashSet<String>) -> Self {
        let items = flatten_tree(&tree);

        // Start with the leaf selected, or the last item
        let initial = leaf_id
            .and_then(|lid| items.iter().position(|item| item.id == lid))
            .unwrap_or(items.len().saturating_sub(1));

        Self {
            items,
            selected: initial,
            scroll_offset: 0,
            active_path,
            leaf_id: leaf_id.map(String::from),
        }
    }

    fn move_up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    fn move_down(&mut self) {
        if self.selected + 1 < self.items.len() {
            self.selected += 1;
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> TreeBrowserResult {
        match key.code {
            KeyCode::Esc => return TreeBrowserResult::Cancelled,
            KeyCode::Enter => {
                if let Some(item) = self.items.get(self.selected) {
                    return TreeBrowserResult::Selected(item.id.clone());
                }
            }
            KeyCode::Up | KeyCode::Char('k') => self.move_up(),
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => self.move_up(),
            KeyCode::Down | KeyCode::Char('j') => self.move_down(),
            KeyCode::Char('n') if key.modifiers.contains(KeyModifiers::CONTROL) => self.move_down(),
            _ => {}
        }
        TreeBrowserResult::Active
    }

    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        let max_width = 80u16.min(area.width.saturating_sub(4)).max(30);
        let list_height = self.items.len().max(1) as u16;
        let max_list_height = area.height.saturating_sub(7).max(3);
        let visible_list = list_height.min(max_list_height);
        // border(1) + title_space(0) + list + separator(1) + hint(1) + border(1)
        let total_height = visible_list + 4;

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
                " Conversation Tree ",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ));
        f.render_widget(block, rect);

        let inner = Rect::new(rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2);
        let content_width = inner.width as usize;

        // Ensure selected item is visible
        let vis = visible_list as usize;
        if self.selected >= self.scroll_offset + vis {
            self.scroll_offset = self.selected + 1 - vis;
        }
        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        }

        // Render tree items
        let list_area = Rect::new(inner.x, inner.y, inner.width, visible_list);

        if self.items.is_empty() {
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    "  No conversation history.",
                    Style::default().fg(Color::DarkGray),
                ))),
                Rect::new(list_area.x, list_area.y, list_area.width, 1),
            );
        } else {
            for (row, item) in self
                .items
                .iter()
                .skip(self.scroll_offset)
                .take(vis)
                .enumerate()
            {
                let display_idx = self.scroll_offset + row;
                let is_selected = display_idx == self.selected;
                let is_on_path = self.active_path.contains(&item.id);
                let is_leaf = self.leaf_id.as_deref() == Some(&item.id);

                let mut spans = Vec::new();

                // Selection marker
                let marker = if is_selected { "> " } else { "  " };
                spans.push(Span::styled(
                    marker,
                    if is_selected {
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    },
                ));

                // Tree connectors
                let mut prefix = String::new();
                for &has_more in &item.ancestor_has_more {
                    if has_more {
                        prefix.push_str("\u{2502}  "); // │
                    } else {
                        prefix.push_str("   ");
                    }
                }
                if item.depth > 0 {
                    if item.is_last {
                        prefix.push_str("\u{2514}\u{2500} "); // └─
                    } else {
                        prefix.push_str("\u{251c}\u{2500} "); // ├─
                    }
                }
                spans.push(Span::styled(prefix, Style::default().fg(Color::DarkGray)));

                // Role tag
                let (role_tag, role_color) = match item.role {
                    Role::User => ("[user] ", Color::Cyan),
                    Role::Assistant => ("[asst] ", Color::Green),
                    Role::Tool => ("[tool] ", Color::DarkGray),
                    Role::System => ("[sys]  ", Color::DarkGray),
                };
                spans.push(Span::styled(role_tag, Style::default().fg(role_color)));

                // Summary text
                let summary_style = if is_selected {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else if is_on_path {
                    Style::default().fg(Color::White)
                } else {
                    Style::default().fg(Color::DarkGray)
                };

                // Truncate summary to fit
                let used: usize = spans.iter().map(|s| s.content.chars().count()).sum();
                let available = content_width.saturating_sub(used).saturating_sub(8); // leave room for markers
                let summary = crate::format::truncate_args(&item.summary, available);
                spans.push(Span::styled(summary, summary_style));

                // Leaf/active marker
                if is_leaf {
                    spans.push(Span::styled(
                        " \u{25c6}",  // ◆
                        Style::default().fg(Color::Yellow),
                    ));
                } else if is_on_path {
                    spans.push(Span::styled(
                        " \u{2190}",  // ←
                        Style::default().fg(Color::DarkGray),
                    ));
                }

                f.render_widget(
                    Paragraph::new(Line::from(spans)),
                    Rect::new(list_area.x, list_area.y + row as u16, list_area.width, 1),
                );
            }
        }

        // Separator
        let sep_y = inner.y + visible_list;
        let sep = "\u{2500}".repeat(content_width); // ─
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                sep,
                Style::default().fg(Color::DarkGray),
            ))),
            Rect::new(inner.x, sep_y, inner.width, 1),
        );

        // Hint line
        let hint_y = sep_y + 1;
        let hint = "\u{2191}\u{2193} navigate \u{00b7} enter select \u{00b7} esc cancel"; // ↑↓ navigate · enter select · esc cancel
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                hint,
                Style::default().fg(Color::DarkGray),
            ))),
            Rect::new(inner.x, hint_y, inner.width, 1),
        );
    }
}

/// Flatten a tree into DFS order for display.
fn flatten_tree(roots: &[TreeNode]) -> Vec<FlatTreeItem> {
    let mut items = Vec::new();

    fn flatten_node(
        node: &TreeNode,
        depth: usize,
        is_last: bool,
        ancestor_has_more: &[bool],
        items: &mut Vec<FlatTreeItem>,
    ) {
        items.push(FlatTreeItem {
            id: node.id.clone(),
            depth,
            is_last,
            ancestor_has_more: ancestor_has_more.to_vec(),
            role: node.role.clone(),
            summary: node.summary.clone(),
        });

        let child_count = node.children.len();
        for (i, child) in node.children.iter().enumerate() {
            let child_is_last = i == child_count - 1;
            let mut child_ancestors = ancestor_has_more.to_vec();
            if depth > 0 {
                child_ancestors.push(!is_last);
            }
            flatten_node(child, depth + 1, child_is_last, &child_ancestors, items);
        }
    }

    let root_count = roots.len();
    for (i, root) in roots.iter().enumerate() {
        let is_last = i == root_count - 1;
        flatten_node(root, 0, is_last, &[], &mut items);
    }
    items
}

