use ratatui::{
    style::{Color, Style},
    text::Span,
};
use std::sync::LazyLock;
use syntect::{
    easy::HighlightLines,
    highlighting::{self, ThemeSet},
    parsing::SyntaxSet,
};

static SYNTAX_SET: LazyLock<SyntaxSet> = LazyLock::new(SyntaxSet::load_defaults_newlines);
static THEME: LazyLock<highlighting::Theme> = LazyLock::new(|| {
    let ts = ThemeSet::load_defaults();
    ts.themes["base16-ocean.dark"].clone()
});

/// Map a file extension or language name to a syntect syntax reference.
fn find_syntax(lang: &str) -> Option<&'static syntect::parsing::SyntaxReference> {
    let ss = &*SYNTAX_SET;
    // Try by token directly (handles "rust", "python", "js", etc.)
    if let Some(syn) = ss.find_syntax_by_token(lang) {
        return Some(syn);
    }
    // Try by extension
    if let Some(syn) = ss.find_syntax_by_extension(lang) {
        return Some(syn);
    }
    // Common aliases
    let alias = match lang {
        "zsh" | "shell" | "sh" => "bash",
        "yml" => "yaml",
        "ts" | "tsx" => "typescript",
        "jsx" => "JavaScript",
        "dockerfile" => "Dockerfile",
        "makefile" => "Makefile",
        "cs" | "csharp" => "csharp",
        _ => return None,
    };
    ss.find_syntax_by_token(alias)
        .or_else(|| ss.find_syntax_by_extension(alias))
}

/// Convert a syntect foreground color to a ratatui Color.
fn to_ratatui_color(c: highlighting::Color) -> Color {
    Color::Rgb(c.r, c.g, c.b)
}

/// Highlight a single line of code and return styled spans.
/// Returns None if the language is unknown (caller should fall back to plain rendering).
pub(super) fn highlight_line(
    highlighter: &mut HighlightLines<'static>,
    line: &str,
) -> Vec<Span<'static>> {
    let ss = &*SYNTAX_SET;
    let regions = highlighter.highlight_line(line, ss).unwrap_or_default();
    regions
        .into_iter()
        .map(|(style, text)| {
            Span::styled(
                text.to_string(),
                Style::default().fg(to_ratatui_color(style.foreground)),
            )
        })
        .collect()
}

/// Create a highlighter for the given language tag (e.g. "rust", "py", "js").
/// Returns None if the language is not recognized.
pub(super) fn highlighter_for_lang(lang: &str) -> Option<HighlightLines<'static>> {
    let syntax = find_syntax(lang)?;
    Some(HighlightLines::new(syntax, &THEME))
}

/// Create a highlighter based on a file path (uses the extension).
/// Returns None if the extension is not recognized.
pub(super) fn highlighter_for_path(path: &str) -> Option<HighlightLines<'static>> {
    let ext = path.rsplit('.').next()?;
    let ss = &*SYNTAX_SET;
    let syntax = ss.find_syntax_by_extension(ext)?;
    Some(HighlightLines::new(syntax, &THEME))
}
