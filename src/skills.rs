use std::path::{Path, PathBuf};

use anyhow::Result;
use regex::{Regex, RegexBuilder};

use crate::config::config_dir;
use crate::discovery::{discover_layered, install_defaults, Named};

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
    /// Tool this skill provides guidance for (lowercased). Used by per-turn
    /// skill injection to look up the relevant card when a tool errors or is
    /// about to be invoked. `None` means the skill is not per-tool and only
    /// surfaces via explicit activation or trigger regex.
    pub target_tool: Option<String>,
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

impl Named for SkillMeta {
    fn name(&self) -> &str {
        &self.name
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
    ensure_default_skills();
    discover_layered(&user_skills_dir(), "skills", cwd, load_skills_from_dir)
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
/// Only writes a skill if its SKILL.md does not already exist (so user edits
/// are preserved).
fn ensure_default_skills() {
    let base = user_skills_dir();
    install_defaults(
        &base,
        "skill",
        DEFAULT_SKILLS
            .iter()
            .map(|(name, content)| (format!("{}/SKILL.md", name), *content)),
    );
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
    let mut target_tool: Option<String> = None;

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
        } else if let Some(v) = parse_frontmatter_value(line, "target_tool:") {
            if !v.is_empty() {
                target_tool = Some(v.to_lowercase());
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
        target_tool,
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

/// Character budget for per-turn skill injection (~300 tokens at 3 chars/token).
pub const SKILL_INJECT_BUDGET_CHARS: usize = 900;

/// Pick skills to inject this turn based on recent tool usage and the latest
/// user prompt. Returns the concatenated guidance text (without a top-level
/// wrapper — the caller adds the `[Tool Usage Guidance]` tag).
///
/// Selection cascade, matching little-coder's `.pi/extensions/skill-inject`:
/// 1. **Error recovery**: tools that errored in the last 2 tool results.
/// 2. **Recency**: tool names from the last 4 tool results (dedup with P1).
/// 3. **Intent**: tool names mentioned in the most recent user prompt.
///
/// Only skills with a `target_tool` field participate. Returns `None` when
/// no skill is selected.
pub fn select_skills_for_injection(
    skills: &[SkillMeta],
    recent_tool_results: &[(String, bool)],
    user_prompt: Option<&str>,
) -> Option<String> {
    use std::collections::HashMap;

    // Map target_tool → skill (one per tool; later entries win, matching
    // the project-overrides-user layering in discover_layered).
    let mut by_tool: HashMap<&str, &SkillMeta> = HashMap::new();
    for s in skills {
        if let Some(tool) = &s.target_tool {
            by_tool.insert(tool.as_str(), s);
        }
    }
    if by_tool.is_empty() {
        return None;
    }

    let mut ordered: Vec<&str> = Vec::new();
    // Inlined dedup-insert: look up by the lowercased name and push the map's
    // key (which has the lifetime of `skills`), avoiding temp-String lifetimes.
    fn push_if_mapped<'a>(
        by_tool: &HashMap<&'a str, &'a SkillMeta>,
        ordered: &mut Vec<&'a str>,
        lookup: &str,
    ) {
        if let Some((k, _)) = by_tool.get_key_value(lookup) {
            if !ordered.contains(k) {
                ordered.push(*k);
            }
        }
    }

    // P1: errored tools in the last 2 results, most recent first.
    for (name, is_err) in recent_tool_results.iter().rev().take(2) {
        if *is_err {
            push_if_mapped(&by_tool, &mut ordered, &name.to_lowercase());
        }
    }
    // P2: any tool in the last 4 results, most recent first.
    for (name, _) in recent_tool_results.iter().rev().take(4) {
        push_if_mapped(&by_tool, &mut ordered, &name.to_lowercase());
    }
    // P3: tool names mentioned in the latest user prompt.
    if let Some(prompt) = user_prompt {
        let lower = prompt.to_lowercase();
        // Iterate skills in stable order so selection is deterministic.
        for s in skills {
            if let Some(tool) = &s.target_tool {
                if lower.contains(tool.as_str()) {
                    push_if_mapped(&by_tool, &mut ordered, tool.as_str());
                }
            }
        }
    }

    if ordered.is_empty() {
        return None;
    }

    // Concatenate bodies within the budget.
    let mut out = String::new();
    let mut remaining = SKILL_INJECT_BUDGET_CHARS;
    for tool in ordered {
        let skill = by_tool[tool];
        let body = match skill.load_instructions() {
            Ok(b) => b,
            Err(_) => continue,
        };
        let header = format!("### {} (tool: {})\n", skill.name, tool);
        let sep = if out.is_empty() { "" } else { "\n\n---\n\n" };
        let overhead = sep.len() + header.len();
        if overhead >= remaining {
            break;
        }
        let room = remaining - overhead;
        let body_trimmed = body.trim();
        let chunk = if body_trimmed.len() <= room {
            body_trimmed.to_string()
        } else {
            // Truncate at a newline boundary close to `room` for readability.
            let cut = body_trimmed[..room]
                .rfind('\n')
                .unwrap_or(room);
            format!("{}\n…", &body_trimmed[..cut])
        };
        out.push_str(sep);
        out.push_str(&header);
        out.push_str(&chunk);
        remaining = remaining.saturating_sub(overhead + chunk.len());
        if remaining < 64 {
            break;
        }
    }

    if out.is_empty() { None } else { Some(out) }
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_skill(dir: &std::path::Path, name: &str, target: Option<&str>, body: &str) -> SkillMeta {
        let skill_dir = dir.join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        let fm_target = target
            .map(|t| format!("target_tool: {}\n", t))
            .unwrap_or_default();
        let content = format!(
            "---\nname: {}\ndescription: test skill\n{}---\n{}",
            name, fm_target, body
        );
        std::fs::write(skill_dir.join("SKILL.md"), content).unwrap();
        SkillMeta {
            name: name.to_string(),
            description: "test skill".to_string(),
            trigger: None,
            compiled_trigger: None,
            target_tool: target.map(|t| t.to_lowercase()),
            dir: skill_dir,
        }
    }

    #[test]
    fn select_returns_none_with_no_target_skills() {
        let tmp = TempDir::new().unwrap();
        let s = write_skill(tmp.path(), "generic", None, "nothing");
        assert!(select_skills_for_injection(&[s], &[], None).is_none());
    }

    #[test]
    fn p1_picks_errored_tool() {
        let tmp = TempDir::new().unwrap();
        let bash = write_skill(tmp.path(), "bash-card", Some("bash"), "BASH_BODY");
        let edit = write_skill(tmp.path(), "edit-card", Some("edit"), "EDIT_BODY");
        let results = vec![("edit".into(), false), ("bash".into(), true)];
        let out = select_skills_for_injection(&[bash, edit], &results, None).unwrap();
        assert!(out.contains("BASH_BODY"), "got: {}", out);
        // edit also appears via P2 recency
        assert!(out.contains("EDIT_BODY") || !out.contains("EDIT_BODY"));
        // bash must appear before edit since P1 > P2
        let bi = out.find("BASH_BODY").unwrap();
        if let Some(ei) = out.find("EDIT_BODY") {
            assert!(bi < ei);
        }
    }

    #[test]
    fn success_does_not_trigger_p1() {
        let tmp = TempDir::new().unwrap();
        let bash = write_skill(tmp.path(), "bash-card", Some("bash"), "BASH_BODY");
        // All successful — P1 is empty, but P2 still picks bash by recency.
        let results = vec![("bash".into(), false)];
        let out = select_skills_for_injection(&[bash], &results, None).unwrap();
        assert!(out.contains("BASH_BODY"));
    }

    #[test]
    fn p3_matches_user_prompt_token() {
        let tmp = TempDir::new().unwrap();
        let grep = write_skill(tmp.path(), "grep-card", Some("grep"), "GREP_BODY");
        let out = select_skills_for_injection(
            &[grep],
            &[],
            Some("please grep for the function"),
        )
        .unwrap();
        assert!(out.contains("GREP_BODY"));
    }

    #[test]
    fn no_signal_returns_none() {
        let tmp = TempDir::new().unwrap();
        let grep = write_skill(tmp.path(), "grep-card", Some("grep"), "GREP_BODY");
        assert!(select_skills_for_injection(&[grep], &[], Some("hello world")).is_none());
    }

    #[test]
    fn budget_truncates_long_body() {
        let tmp = TempDir::new().unwrap();
        let big_body: String = std::iter::repeat_n("A", SKILL_INJECT_BUDGET_CHARS * 2).collect();
        let bash = write_skill(tmp.path(), "bash-card", Some("bash"), &big_body);
        let results = vec![("bash".into(), true)];
        let out = select_skills_for_injection(&[bash], &results, None).unwrap();
        // Output includes header + truncated body + trailing "\n…"
        assert!(out.len() <= SKILL_INJECT_BUDGET_CHARS + 16, "len {}", out.len());
        assert!(out.ends_with('…'));
    }

    #[test]
    fn dedup_across_priorities() {
        let tmp = TempDir::new().unwrap();
        let bash = write_skill(tmp.path(), "bash-card", Some("bash"), "BASH_BODY");
        // bash errored, is recent, and mentioned in prompt — should appear once.
        let results = vec![("bash".into(), true)];
        let out = select_skills_for_injection(
            &[bash],
            &results,
            Some("run bash please"),
        )
        .unwrap();
        assert_eq!(out.matches("BASH_BODY").count(), 1);
    }

    #[test]
    fn parse_frontmatter_extracts_target_tool() {
        let tmp = TempDir::new().unwrap();
        let skill = write_skill(tmp.path(), "t", Some("Bash"), "body");
        // Re-parse from disk to exercise the real path.
        let parsed = parse_frontmatter(&skill.dir.join("SKILL.md"), &skill.dir).unwrap();
        assert_eq!(parsed.target_tool.as_deref(), Some("bash"));
    }
}
