use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use ollama_code::hooks::HookRunner;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create the `.agents/` directory and write `hooks.toml` inside it.
fn setup_hooks(root: &Path, hooks_toml: &str) {
    let agents_dir = root.join(".agents");
    fs::create_dir_all(&agents_dir).unwrap();
    fs::write(agents_dir.join("hooks.toml"), hooks_toml).unwrap();
}

/// Write an executable script inside `.agents/`.
fn write_hook_script(root: &Path, name: &str, content: &str) {
    let path = root.join(".agents").join(name);
    fs::write(&path, content).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
}

/// Hook script that always denies (for pre_tool_execute).
/// Reads stdin to avoid broken-pipe race with the parent process.
const DENY_SCRIPT: &str =
    "#!/bin/sh\ncat > /dev/null\necho '{\"action\":\"deny\",\"message\":\"hook fired\"}'";

/// Hook script that modifies output (for post_tool_execute).
/// Reads stdin to avoid broken-pipe race with the parent process.
const MODIFY_OUTPUT_SCRIPT: &str =
    "#!/bin/sh\ncat > /dev/null\necho '{\"action\":\"modify\",\"output\":\"modified by hook\",\"success\":true}'";

/// Build a HookRunner from hooks.toml in the temp dir's `.agents/` folder.
fn runner(dir: &TempDir) -> HookRunner {
    HookRunner::from_file(&dir.path().join(".agents/hooks.toml"), None)
}

// ===========================================================================
// Regex tool matching — RED tests (will fail until regex is implemented)
// ===========================================================================

#[tokio::test]
async fn regex_tool_pattern_matches() {
    let dir = TempDir::new().unwrap();
    setup_hooks(
        dir.path(),
        r#"
[deny-file-tools]
event = "pre_tool_execute"
command = "./deny.sh"
tools = ["file_.*"]
"#,
    );
    write_hook_script(dir.path(), "deny.sh", DENY_SCRIPT);

    let runner = runner(&dir);
    let result = runner
        .pre_tool_execute("file_read", &serde_json::json!({}))
        .await
        .unwrap();

    assert_eq!(
        result.action.as_deref(),
        Some("deny"),
        "regex 'file_.*' should match 'file_read'"
    );
}

#[tokio::test]
async fn regex_tool_alternation_in_single_pattern() {
    let dir = TempDir::new().unwrap();
    setup_hooks(
        dir.path(),
        r#"
[deny-multi]
event = "pre_tool_execute"
command = "./deny.sh"
tools = ["bash|write_file"]
"#,
    );
    write_hook_script(dir.path(), "deny.sh", DENY_SCRIPT);

    let runner = runner(&dir);

    let r1 = runner
        .pre_tool_execute("bash", &serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(
        r1.action.as_deref(),
        Some("deny"),
        "'bash|write_file' should match 'bash'"
    );

    let r2 = runner
        .pre_tool_execute("write_file", &serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(
        r2.action.as_deref(),
        Some("deny"),
        "'bash|write_file' should match 'write_file'"
    );

    let r3 = runner
        .pre_tool_execute("read_file", &serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(
        r3.action, None,
        "'bash|write_file' should NOT match 'read_file'"
    );
}

#[tokio::test]
async fn regex_tool_character_class() {
    let dir = TempDir::new().unwrap();
    setup_hooks(
        dir.path(),
        r#"
[deny-class]
event = "pre_tool_execute"
command = "./deny.sh"
tools = ["[a-z]+_file"]
"#,
    );
    write_hook_script(dir.path(), "deny.sh", DENY_SCRIPT);

    let runner = runner(&dir);

    let r1 = runner
        .pre_tool_execute("read_file", &serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(
        r1.action.as_deref(),
        Some("deny"),
        "'[a-z]+_file' should match 'read_file'"
    );

    let r2 = runner
        .pre_tool_execute("bash", &serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(
        r2.action, None,
        "'[a-z]+_file' should NOT match 'bash'"
    );
}

#[tokio::test]
async fn regex_in_post_tool_execute() {
    let dir = TempDir::new().unwrap();
    setup_hooks(
        dir.path(),
        r#"
[modify-file-tools]
event = "post_tool_execute"
command = "./modify.sh"
tools = ["file_.*"]
"#,
    );
    write_hook_script(dir.path(), "modify.sh", MODIFY_OUTPUT_SCRIPT);

    let runner = runner(&dir);

    // Should match: file_read matches file_.*
    let (output, success) = runner
        .post_tool_execute("file_read", &serde_json::json!({}), "original".to_string(), true)
        .await;
    assert_eq!(output, "modified by hook", "regex should match file_read in post_tool_execute");
    assert!(success);

    // Should NOT match: bash doesn't match file_.*
    let (output2, _) = runner
        .post_tool_execute("bash", &serde_json::json!({}), "original".to_string(), true)
        .await;
    assert_eq!(output2, "original", "regex should NOT match bash in post_tool_execute");
}

// ===========================================================================
// if_args filtering — RED tests (will fail until if_args is implemented)
// ===========================================================================

#[tokio::test]
async fn if_args_skips_non_matching_arguments() {
    let dir = TempDir::new().unwrap();
    setup_hooks(
        dir.path(),
        r#"
[deny-git-only]
event = "pre_tool_execute"
command = "./deny.sh"
tools = ["bash"]
if_args = "^git "
"#,
    );
    write_hook_script(dir.path(), "deny.sh", DENY_SCRIPT);

    let runner = runner(&dir);
    let result = runner
        .pre_tool_execute("bash", &serde_json::json!({"command": "npm install"}))
        .await
        .unwrap();

    assert_eq!(
        result.action, None,
        "if_args '^git ' should NOT match args containing 'npm install'"
    );
}

#[tokio::test]
async fn if_args_combined_with_tools_no_arg_match() {
    let dir = TempDir::new().unwrap();
    setup_hooks(
        dir.path(),
        r#"
[deny-rs-files]
event = "pre_tool_execute"
command = "./deny.sh"
tools = ["write_file|edit_file"]
if_args = "\\.rs$"
"#,
    );
    write_hook_script(dir.path(), "deny.sh", DENY_SCRIPT);

    let runner = runner(&dir);

    // Tool matches (write_file) but args don't contain a .rs string value
    let result = runner
        .pre_tool_execute(
            "write_file",
            &serde_json::json!({"file_path": "main.py", "content": "print('hi')"}),
        )
        .await
        .unwrap();
    assert_eq!(
        result.action, None,
        "if_args '\\.rs$' should NOT match args with 'main.py'"
    );
}

// ===========================================================================
// Backward compatibility — GREEN tests (should already pass)
// ===========================================================================

#[tokio::test]
async fn exact_tool_match_still_works() {
    let dir = TempDir::new().unwrap();
    setup_hooks(
        dir.path(),
        r#"
[deny-bash]
event = "pre_tool_execute"
command = "./deny.sh"
tools = ["bash"]
"#,
    );
    write_hook_script(dir.path(), "deny.sh", DENY_SCRIPT);

    let runner = runner(&dir);
    let result = runner
        .pre_tool_execute("bash", &serde_json::json!({}))
        .await
        .unwrap();

    assert_eq!(
        result.action.as_deref(),
        Some("deny"),
        "exact match 'bash' should match 'bash'"
    );
}

#[tokio::test]
async fn exact_match_anchored_no_partial() {
    let dir = TempDir::new().unwrap();
    setup_hooks(
        dir.path(),
        r#"
[deny-bash]
event = "pre_tool_execute"
command = "./deny.sh"
tools = ["bash"]
"#,
    );
    write_hook_script(dir.path(), "deny.sh", DENY_SCRIPT);

    let runner = runner(&dir);

    let r1 = runner
        .pre_tool_execute("bash_extended", &serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(
        r1.action, None,
        "'bash' should NOT match 'bash_extended' (must be anchored)"
    );

    let r2 = runner
        .pre_tool_execute("my_bash", &serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(
        r2.action, None,
        "'bash' should NOT match 'my_bash' (must be anchored)"
    );
}

#[tokio::test]
async fn no_tools_field_matches_all() {
    let dir = TempDir::new().unwrap();
    setup_hooks(
        dir.path(),
        r#"
[deny-everything]
event = "pre_tool_execute"
command = "./deny.sh"
"#,
    );
    write_hook_script(dir.path(), "deny.sh", DENY_SCRIPT);

    let runner = runner(&dir);

    let r1 = runner
        .pre_tool_execute("bash", &serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(
        r1.action.as_deref(),
        Some("deny"),
        "no tools field should match 'bash'"
    );

    let r2 = runner
        .pre_tool_execute("anything_at_all", &serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(
        r2.action.as_deref(),
        Some("deny"),
        "no tools field should match 'anything_at_all'"
    );
}

#[tokio::test]
async fn if_args_fires_when_args_match() {
    let dir = TempDir::new().unwrap();
    setup_hooks(
        dir.path(),
        r#"
[deny-git-only]
event = "pre_tool_execute"
command = "./deny.sh"
tools = ["bash"]
if_args = "^git "
"#,
    );
    write_hook_script(dir.path(), "deny.sh", DENY_SCRIPT);

    let runner = runner(&dir);
    let result = runner
        .pre_tool_execute("bash", &serde_json::json!({"command": "git push origin main"}))
        .await
        .unwrap();

    assert_eq!(
        result.action.as_deref(),
        Some("deny"),
        "if_args '^git ' should match args containing 'git push origin main'"
    );
}
