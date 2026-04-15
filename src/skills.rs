use std::path::{Path, PathBuf};

use anyhow::Result;
use regex::{Regex, RegexBuilder};

use crate::config::config_dir;

/// Metadata from SKILL.md frontmatter -- kept in memory for discovery.
#[derive(Clone, Debug)]
pub struct SkillMeta {
    /// Skill name (from frontmatter).
    pub name: String,
    /// One-line description (from frontmatter).
    pub description: String,
    /// Optional trigger pattern (raw string from frontmatter).
    pub trigger: Option<String>,
    /// Pre-compiled trigger regex (case-insensitive). Populated during discovery.
    pub compiled_trigger: Option<Regex>,
    /// Path to the skill directory containing SKILL.md.
    pub dir: PathBuf,
}

impl SkillMeta {
    /// Load the full instructions (SKILL.md body after frontmatter).
    /// Substitutes `{config_dir}` with the platform-specific config directory.
    pub fn load_instructions(&self) -> Result<String> {
        let content = std::fs::read_to_string(self.dir.join("SKILL.md"))?;
        let body = extract_body(&content);
        Ok(body.replace("{config_dir}", &config_dir().to_string_lossy()))
    }
}

/// User-scoped skills directory (`~/.config/ollama-code/skills/`).
pub fn user_skills_dir() -> PathBuf {
    config_dir().join("skills")
}

/// Discover skills from both project-scoped and user-scoped directories.
///
/// Project skills: walks up from `cwd` looking for `.agents/skills/*/SKILL.md`,
/// stops at the first directory that has valid skills.
///
/// User skills: loaded from `~/.config/ollama-code/skills/*/SKILL.md`.
///
/// Both sets are merged. Project-scoped skills override user-scoped skills
/// with the same name.
pub fn discover_skills(cwd: &str) -> Vec<SkillMeta> {
    // Ensure built-in default skills are installed.
    ensure_default_skills();

    // 1. Load user-scoped skills (lower priority).
    let mut skills = load_skills_from_dir(&user_skills_dir());

    // 2. Load project-scoped skills (higher priority — override by name).
    let project_skills = discover_project_skills(cwd);
    for ps in project_skills {
        if let Some(existing) = skills.iter_mut().find(|s| s.name == ps.name) {
            *existing = ps;
        } else {
            skills.push(ps);
        }
    }

    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

/// Walk up from `cwd` looking for `.agents/skills/*/SKILL.md`.
/// Stops at the first directory that contains valid skills.
fn discover_project_skills(cwd: &str) -> Vec<SkillMeta> {
    let mut dir = Path::new(cwd).to_path_buf();
    loop {
        let skills_dir = dir.join(".agents").join("skills");
        let found = load_skills_from_dir(&skills_dir);
        if !found.is_empty() {
            return found;
        }
        if !dir.pop() {
            break;
        }
    }
    Vec::new()
}

/// Load all valid skills from a single directory (`dir/*/SKILL.md`).
fn load_skills_from_dir(dir: &Path) -> Vec<SkillMeta> {
    let mut skills = Vec::new();
    if !dir.is_dir() {
        return skills;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let skill_dir = entry.path();
            if skill_dir.is_dir() {
                let skill_md = skill_dir.join("SKILL.md");
                if skill_md.is_file() {
                    if let Ok(meta) = parse_frontmatter(&skill_md, &skill_dir) {
                        skills.push(meta);
                    }
                }
            }
        }
    }
    skills
}

// ── Built-in default skills ──────────────────────────────────────────────

/// Built-in skills that are installed into the user-scoped skills directory
/// on first run. Each entry is (directory_name, SKILL.md content).
const DEFAULT_SKILLS: &[(&str, &str)] = &[
    ("self-modify", include_str!("../skills/self-modify/SKILL.md")),
];

/// Ensure that built-in default skills exist in the user skills directory.
/// Only writes a skill if its directory does not already exist (so user edits
/// are preserved).
fn ensure_default_skills() {
    let base = user_skills_dir();
    for (name, content) in DEFAULT_SKILLS {
        let skill_dir = base.join(name);
        let skill_md = skill_dir.join("SKILL.md");
        if skill_md.exists() {
            continue;
        }
        if let Err(e) = std::fs::create_dir_all(&skill_dir) {
            eprintln!("Warning: could not create skill dir {}: {}", skill_dir.display(), e);
            continue;
        }
        if let Err(e) = std::fs::write(&skill_md, content) {
            eprintln!("Warning: could not write default skill {}: {}", skill_md.display(), e);
        }
    }
}

// ── Frontmatter parsing ──────────────────────────────────────────────────

/// Parse SKILL.md YAML frontmatter to extract name and description.
fn parse_frontmatter(path: &Path, dir: &Path) -> Result<SkillMeta> {
    use crate::format::{parse_frontmatter_value, split_frontmatter};

    let content = std::fs::read_to_string(path)?;
    let (frontmatter, _body) = split_frontmatter(&content)?;

    let mut name = None;
    let mut description = None;
    let mut trigger = None;

    for line in frontmatter.lines() {
        let line = line.trim();
        if let Some(v) = parse_frontmatter_value(line, "name:") {
            name = Some(v);
        } else if let Some(v) = parse_frontmatter_value(line, "description:") {
            description = Some(v);
        } else if let Some(v) = parse_frontmatter_value(line, "trigger:") {
            if !v.is_empty() {
                trigger = Some(v);
            }
        }
    }

    let compiled_trigger = trigger.as_ref().and_then(|p| {
        RegexBuilder::new(p).case_insensitive(true).build().ok()
    });

    Ok(SkillMeta {
        name: name.ok_or_else(|| anyhow::anyhow!("missing 'name'"))?,
        description: description.ok_or_else(|| anyhow::anyhow!("missing 'description'"))?,
        trigger,
        compiled_trigger,
        dir: dir.to_path_buf(),
    })
}

/// Extract the body of a SKILL.md (everything after the closing `---` of frontmatter).
fn extract_body(content: &str) -> String {
    use crate::format::split_frontmatter;
    match split_frontmatter(content) {
        Ok((_fm, body)) => body.to_string(),
        Err(_) => content.to_string(),
    }
}

/// Check if a user message triggers any skill. Returns the first matching
/// skill's name and loaded instructions. Uses pre-compiled regexes.
pub fn check_triggers(skills: &[SkillMeta], user_input: &str) -> Option<(String, String)> {
    for skill in skills {
        if let Some(re) = &skill.compiled_trigger {
            if re.is_match(user_input) {
                if let Ok(instructions) = skill.load_instructions() {
                    return Some((skill.name.clone(), instructions));
                }
            }
        }
    }
    None
}

/// Format skill summaries for inclusion in the system prompt (discovery layer).
pub fn format_skill_summaries(skills: &[SkillMeta]) -> String {
    let mut s = String::from("\n\n## Available Skills\n\n");
    s.push_str(
        "The following skills are available. The user can activate them with slash commands, \
         and you can activate them with the skill(name, args?) tool when a task matches.\n\n",
    );
    for skill in skills {
        s.push_str(&format!("- {} \u{2014} {}", skill.name, skill.description));
        if let Some(trigger) = &skill.trigger {
            s.push_str(&format!("\n  Trigger: {}", trigger));
        }
        s.push('\n');
    }
    s
}
