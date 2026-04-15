use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::Result;
use regex::Regex;

use crate::config::config_dir;

fn template_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\{\{([^}]+)\}\}").unwrap())
}

/// Parse a `{{name}}` or `{{name:default}}` placeholder interior.
fn parse_placeholder(inner: &str) -> (&str, Option<&str>) {
    if let Some((n, d)) = inner.split_once(':') {
        (n.trim(), Some(d.trim()))
    } else {
        (inner, None)
    }
}

/// Metadata and body for a prompt template.
#[derive(Clone, Debug)]
pub struct PromptTemplate {
    /// Template name (used as the slash command).
    pub name: String,
    /// One-line description shown in /help and completions.
    pub description: String,
    /// Template body with `{{variable}}` placeholders.
    pub body: String,
    /// Path to the template file.
    pub path: PathBuf,
}

/// A variable extracted from a template body.
#[derive(Clone, Debug)]
pub struct TemplateVar {
    /// Variable name (e.g. "focus_area").
    pub name: String,
    /// Optional default value (e.g. from `{{focus_area:security}}`).
    pub default: Option<String>,
}

impl PromptTemplate {
    /// Extract all `{{variable}}` and `{{variable:default}}` placeholders from the body.
    /// Returns them in order of first appearance, deduplicated.
    pub fn variables(&self) -> Vec<TemplateVar> {
        let re = template_re();
        let mut seen = HashSet::new();
        let mut vars = Vec::new();
        for cap in re.captures_iter(&self.body) {
            let (name, default) = parse_placeholder(cap[1].trim());
            let name = name.to_string();
            let default = default.map(|d| d.to_string());
            if seen.insert(name.clone()) {
                vars.push(TemplateVar { name, default });
            }
        }
        vars
    }

    /// Expand the template body by replacing `{{var}}` and `{{var:default}}`
    /// placeholders with the provided values.
    pub fn expand(&self, values: &HashMap<String, String>) -> String {
        let re = template_re();
        re.replace_all(&self.body, |caps: &regex::Captures| {
            let (name, default) = parse_placeholder(caps[1].trim());
            if let Some(val) = values.get(name) {
                val.clone()
            } else if let Some(d) = default {
                d.to_string()
            } else {
                // Leave unresolved placeholders as-is
                caps[0].to_string()
            }
        })
        .to_string()
    }
}

/// User-scoped prompts directory (`~/.config/ollama-code/prompts/`).
pub fn user_prompts_dir() -> PathBuf {
    config_dir().join("prompts")
}

/// Discover prompt templates from both project-scoped and user-scoped directories.
///
/// Project prompts: walks up from `cwd` looking for `.agents/prompts/*.md`.
/// User prompts: loaded from `~/.config/ollama-code/prompts/*.md`.
///
/// Project-scoped prompts override user-scoped prompts with the same name.
pub fn discover_prompts(cwd: &str) -> Vec<PromptTemplate> {
    ensure_default_prompts();

    // 1. Load user-scoped prompts (lower priority).
    let mut prompts = load_prompts_from_dir(&user_prompts_dir());

    // 2. Load project-scoped prompts (higher priority — override by name).
    let project_prompts = discover_project_prompts(cwd);
    for pp in project_prompts {
        if let Some(existing) = prompts.iter_mut().find(|p| p.name == pp.name) {
            *existing = pp;
        } else {
            prompts.push(pp);
        }
    }

    prompts.sort_by(|a, b| a.name.cmp(&b.name));
    prompts
}

/// Walk up from `cwd` looking for `.agents/prompts/*.md`.
fn discover_project_prompts(cwd: &str) -> Vec<PromptTemplate> {
    let mut dir = Path::new(cwd).to_path_buf();
    loop {
        let prompts_dir = dir.join(".agents").join("prompts");
        let found = load_prompts_from_dir(&prompts_dir);
        if !found.is_empty() {
            return found;
        }
        if !dir.pop() {
            break;
        }
    }
    Vec::new()
}

/// Load all valid prompt templates from a directory (`dir/*.md`).
fn load_prompts_from_dir(dir: &Path) -> Vec<PromptTemplate> {
    let mut prompts = Vec::new();
    if !dir.is_dir() {
        return prompts;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|e| e == "md") {
                if let Ok(template) = parse_template(&path) {
                    prompts.push(template);
                }
            }
        }
    }
    prompts
}

// ── Built-in default prompts ────────────────────────────────────────────

const DEFAULT_PROMPTS: &[(&str, &str)] = &[
    ("review.md", include_str!("../prompts/review.md")),
];

/// Ensure that built-in default prompts exist in the user prompts directory.
/// Only writes a prompt if the file does not already exist (so user edits are preserved).
fn ensure_default_prompts() {
    let base = user_prompts_dir();
    if let Err(e) = std::fs::create_dir_all(&base) {
        eprintln!(
            "Warning: could not create prompts dir {}: {}",
            base.display(),
            e
        );
        return;
    }
    for (filename, content) in DEFAULT_PROMPTS {
        let path = base.join(filename);
        if path.exists() {
            continue;
        }
        if let Err(e) = std::fs::write(&path, content) {
            eprintln!(
                "Warning: could not write default prompt {}: {}",
                path.display(),
                e
            );
        }
    }
}

// ── Frontmatter parsing ─────────────────────────────────────────────────

/// Parse a prompt template file with YAML-like frontmatter.
fn parse_template(path: &Path) -> Result<PromptTemplate> {
    use crate::format::{parse_frontmatter_value, split_frontmatter};

    let content = std::fs::read_to_string(path)?;
    let (frontmatter, body) = split_frontmatter(&content)?;

    let mut name = None;
    let mut description = None;

    for line in frontmatter.lines() {
        let line = line.trim();
        if let Some(v) = parse_frontmatter_value(line, "name:") {
            name = Some(v);
        } else if let Some(v) = parse_frontmatter_value(line, "description:") {
            description = Some(v);
        }
    }

    Ok(PromptTemplate {
        name: name.ok_or_else(|| anyhow::anyhow!("missing 'name'"))?,
        description: description.ok_or_else(|| anyhow::anyhow!("missing 'description'"))?,
        body: body.to_string(),
        path: path.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_variables_basic() {
        let template = PromptTemplate {
            name: "test".into(),
            description: "test".into(),
            body: "Review {{file_path}} focusing on {{focus_area}}.".into(),
            path: PathBuf::new(),
        };
        let vars = template.variables();
        assert_eq!(vars.len(), 2);
        assert_eq!(vars[0].name, "file_path");
        assert!(vars[0].default.is_none());
        assert_eq!(vars[1].name, "focus_area");
    }

    #[test]
    fn extract_variables_with_defaults() {
        let template = PromptTemplate {
            name: "test".into(),
            description: "test".into(),
            body: "Review {{file_path}} focusing on {{focus_area:security}}.".into(),
            path: PathBuf::new(),
        };
        let vars = template.variables();
        assert_eq!(vars.len(), 2);
        assert_eq!(vars[1].name, "focus_area");
        assert_eq!(vars[1].default.as_deref(), Some("security"));
    }

    #[test]
    fn extract_variables_dedup() {
        let template = PromptTemplate {
            name: "test".into(),
            description: "test".into(),
            body: "{{x}} and {{y}} and {{x}} again".into(),
            path: PathBuf::new(),
        };
        let vars = template.variables();
        assert_eq!(vars.len(), 2);
        assert_eq!(vars[0].name, "x");
        assert_eq!(vars[1].name, "y");
    }

    #[test]
    fn expand_basic() {
        let template = PromptTemplate {
            name: "test".into(),
            description: "test".into(),
            body: "Review {{file_path}} focusing on {{focus_area}}.".into(),
            path: PathBuf::new(),
        };
        let mut values = HashMap::new();
        values.insert("file_path".into(), "src/main.rs".into());
        values.insert("focus_area".into(), "error handling".into());
        let result = template.expand(&values);
        assert_eq!(result, "Review src/main.rs focusing on error handling.");
    }

    #[test]
    fn expand_uses_defaults() {
        let template = PromptTemplate {
            name: "test".into(),
            description: "test".into(),
            body: "Focus on {{focus_area:security}}.".into(),
            path: PathBuf::new(),
        };
        let values = HashMap::new();
        let result = template.expand(&values);
        assert_eq!(result, "Focus on security.");
    }

    #[test]
    fn expand_value_overrides_default() {
        let template = PromptTemplate {
            name: "test".into(),
            description: "test".into(),
            body: "Focus on {{focus_area:security}}.".into(),
            path: PathBuf::new(),
        };
        let mut values = HashMap::new();
        values.insert("focus_area".into(), "performance".into());
        let result = template.expand(&values);
        assert_eq!(result, "Focus on performance.");
    }

    #[test]
    fn no_variables() {
        let template = PromptTemplate {
            name: "test".into(),
            description: "test".into(),
            body: "Just a plain prompt with no variables.".into(),
            path: PathBuf::new(),
        };
        let vars = template.variables();
        assert!(vars.is_empty());
    }
}
