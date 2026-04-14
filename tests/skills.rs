use std::fs;

use ollama_code::skills;

// ── Frontmatter parsing (via discover_skills with tempdir) ──────────

fn create_skill(dir: &std::path::Path, name: &str, description: &str, body: &str) {
    let skill_dir = dir.join(name);
    fs::create_dir_all(&skill_dir).unwrap();
    fs::write(
        skill_dir.join("SKILL.md"),
        format!(
            "---\nname: {}\ndescription: {}\n---\n{}",
            name, description, body
        ),
    )
    .unwrap();
}

#[test]
fn load_skills_from_dir() {
    let dir = tempfile::tempdir().unwrap();
    create_skill(dir.path(), "my-skill", "A test skill", "Do the thing.");

    // discover_project_skills walks up from cwd; we'll use load_skills_from_dir indirectly
    // by structuring as .agents/skills/
    let project = tempfile::tempdir().unwrap();
    let skills_dir = project.path().join(".agents").join("skills");
    fs::create_dir_all(&skills_dir).unwrap();
    create_skill(&skills_dir, "proj-skill", "Project skill", "Project body");

    // discover_skills from the project dir should find proj-skill
    let found = skills::discover_skills(project.path().to_str().unwrap());
    let names: Vec<&str> = found.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"proj-skill"));
}

#[test]
fn skill_load_instructions() {
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join("test-skill");
    fs::create_dir_all(&skill_dir).unwrap();
    fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: test-skill\ndescription: A test\n---\nThis is the body.\nLine 2.",
    )
    .unwrap();

    let meta = skills::SkillMeta {
        name: "test-skill".to_string(),
        description: "A test".to_string(),
        trigger: None,
        compiled_trigger: None,
        dir: skill_dir,
    };
    let body = meta.load_instructions().unwrap();
    assert_eq!(body, "This is the body.\nLine 2.");
}

#[test]
fn skill_load_instructions_no_body() {
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join("empty-skill");
    fs::create_dir_all(&skill_dir).unwrap();
    fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: empty-skill\ndescription: Empty\n---\n",
    )
    .unwrap();

    let meta = skills::SkillMeta {
        name: "empty-skill".to_string(),
        description: "Empty".to_string(),
        trigger: None,
        compiled_trigger: None,
        dir: skill_dir,
    };
    let body = meta.load_instructions().unwrap();
    assert!(body.is_empty());
}

#[test]
fn project_skills_override_user() {
    // We can't override the real user dir, but we can test the merge logic
    // by checking that project-scoped skills show up in discover_skills
    let project = tempfile::tempdir().unwrap();
    let skills_dir = project.path().join(".agents").join("skills");

    create_skill(&skills_dir, "alpha", "Alpha skill", "Alpha body");
    create_skill(&skills_dir, "beta", "Beta skill", "Beta body");

    let found = skills::discover_skills(project.path().to_str().unwrap());
    let names: Vec<&str> = found.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"alpha"));
    assert!(names.contains(&"beta"));
}

#[test]
fn skills_sorted_by_name() {
    let project = tempfile::tempdir().unwrap();
    let skills_dir = project.path().join(".agents").join("skills");

    create_skill(&skills_dir, "zebra", "Z skill", "");
    create_skill(&skills_dir, "alpha", "A skill", "");
    create_skill(&skills_dir, "middle", "M skill", "");

    let found = skills::discover_skills(project.path().to_str().unwrap());
    let project_names: Vec<&str> = found
        .iter()
        .filter(|s| s.name == "zebra" || s.name == "alpha" || s.name == "middle")
        .map(|s| s.name.as_str())
        .collect();
    assert_eq!(project_names, vec!["alpha", "middle", "zebra"]);
}

#[test]
fn invalid_skill_skipped() {
    let project = tempfile::tempdir().unwrap();
    let skills_dir = project.path().join(".agents").join("skills");

    // Valid skill
    create_skill(&skills_dir, "good", "Good skill", "body");

    // Invalid skill (no frontmatter)
    let bad_dir = skills_dir.join("bad");
    fs::create_dir_all(&bad_dir).unwrap();
    fs::write(bad_dir.join("SKILL.md"), "No frontmatter here").unwrap();

    // Skill with missing name
    let partial_dir = skills_dir.join("partial");
    fs::create_dir_all(&partial_dir).unwrap();
    fs::write(
        partial_dir.join("SKILL.md"),
        "---\ndescription: Missing name\n---\nbody",
    )
    .unwrap();

    let found = skills::discover_skills(project.path().to_str().unwrap());
    let names: Vec<&str> = found.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"good"));
    assert!(!names.contains(&"bad"));
    assert!(!names.contains(&"partial"));
}

#[test]
fn format_skill_summaries_output() {
    let skills = vec![
        skills::SkillMeta {
            name: "deploy".to_string(),
            description: "Deploy to production".to_string(),
            trigger: None,
            compiled_trigger: None,
            dir: std::path::PathBuf::from("/tmp"),
        },
        skills::SkillMeta {
            name: "test".to_string(),
            description: "Run tests".to_string(),
            trigger: None,
            compiled_trigger: None,
            dir: std::path::PathBuf::from("/tmp"),
        },
    ];
    let summary = skills::format_skill_summaries(&skills);
    assert!(summary.contains("## Available Skills"));
    assert!(summary.contains("- deploy \u{2014} Deploy to production"));
    assert!(summary.contains("- test \u{2014} Run tests"));
}

#[test]
fn format_skill_summaries_empty() {
    let summary = skills::format_skill_summaries(&[]);
    assert!(summary.contains("## Available Skills"));
    // No individual entries
    assert!(!summary.contains("\u{2014}"));
}

#[test]
fn frontmatter_with_quoted_values() {
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".agents").join("skills").join("quoted");
    fs::create_dir_all(&skill_dir).unwrap();
    fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: \"quoted-name\"\ndescription: 'single quoted'\n---\nbody",
    )
    .unwrap();

    let found = skills::discover_skills(dir.path().to_str().unwrap());
    let skill = found.iter().find(|s| s.name == "quoted-name");
    assert!(skill.is_some());
    assert_eq!(skill.unwrap().description, "single quoted");
}

#[test]
fn no_skills_dir_returns_empty() {
    let dir = tempfile::tempdir().unwrap();
    // No .agents/skills/ directory
    let found = skills::discover_skills(dir.path().to_str().unwrap());
    // May include user-scoped skills (self-modify), but no project skills
    let project_skills: Vec<_> = found
        .iter()
        .filter(|s| s.dir.starts_with(dir.path()))
        .collect();
    assert!(project_skills.is_empty());
}
