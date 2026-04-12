use std::path::{Path, PathBuf};

use anyhow::Result;

/// Metadata from SKILL.md frontmatter -- kept in memory for discovery.
#[derive(Clone, Debug)]
pub struct SkillMeta {
    /// Skill name (from frontmatter).
    pub name: String,
    /// One-line description (from frontmatter).
    pub description: String,
    /// Path to the skill directory containing SKILL.md.
    pub dir: PathBuf,
}

impl SkillMeta {
    /// Load the full instructions (SKILL.md body after frontmatter).
    pub fn load_instructions(&self) -> Result<String> {
        let content = std::fs::read_to_string(self.dir.join("SKILL.md"))?;
        Ok(extract_body(&content))
    }
}

/// User-scoped skills directory (`~/.config/ollama-code/skills/`).
pub fn user_skills_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ollama-code")
        .join("skills")
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
    ("config", include_str!("../skills/config/SKILL.md")),
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
    let content = std::fs::read_to_string(path)?;
    let trimmed = content.trim_start();

    if !trimmed.starts_with("---") {
        anyhow::bail!("missing frontmatter");
    }

    let after_opening = &trimmed[3..];
    let end = after_opening
        .find("\n---")
        .ok_or_else(|| anyhow::anyhow!("malformed frontmatter"))?;

    let frontmatter = &after_opening[..end];

    let mut name = None;
    let mut description = None;

    for line in frontmatter.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("name:") {
            name = Some(
                value
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string(),
            );
        } else if let Some(value) = line.strip_prefix("description:") {
            description = Some(
                value
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string(),
            );
        }
    }

    Ok(SkillMeta {
        name: name.ok_or_else(|| anyhow::anyhow!("missing 'name'"))?,
        description: description.ok_or_else(|| anyhow::anyhow!("missing 'description'"))?,
        dir: dir.to_path_buf(),
    })
}

/// Extract the body of a SKILL.md (everything after the closing `---` of frontmatter).
fn extract_body(content: &str) -> String {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return content.to_string();
    }
    let after_opening = &trimmed[3..];
    if let Some(end) = after_opening.find("\n---") {
        after_opening[end + 4..].trim_start_matches('\n').to_string()
    } else {
        content.to_string()
    }
}

/// Format skill summaries for inclusion in the system prompt (discovery layer).
pub fn format_skill_summaries(skills: &[SkillMeta]) -> String {
    let mut s = String::from("\n\n## Available Skills\n\n");
    s.push_str("The following skills can be activated by the user with slash commands:\n\n");
    for skill in skills {
        s.push_str(&format!("- /{} \u{2014} {}\n", skill.name, skill.description));
    }
    s
}
