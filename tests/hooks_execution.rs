use std::fs;

use ollama_code::hooks::HookRunner;

/// Create a temp hooks.toml and hook scripts for testing.
fn make_hook_script(dir: &std::path::Path, name: &str, script: &str) -> std::path::PathBuf {
    let script_path = dir.join(name);
    fs::write(&script_path, script).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&script_path, fs::Permissions::from_mode(0o755)).unwrap();
    }
    script_path
}

fn write_hooks_toml(dir: &std::path::Path, content: &str) -> std::path::PathBuf {
    let path = dir.join("hooks.toml");
    fs::write(&path, content).unwrap();
    path
}

// ── pre_tool_execute: deny ──────────────────────────────────────────

#[tokio::test]
async fn pre_tool_deny() {
    let dir = tempfile::tempdir().unwrap();
    make_hook_script(
        dir.path(),
        "deny.sh",
        r#"#!/bin/sh
echo '{"action":"deny","message":"Blocked by test hook"}'
"#,
    );
    let hooks_path = write_hooks_toml(
        dir.path(),
        &format!(
            r#"
[deny-all]
event = "pre_tool_execute"
command = "{}/deny.sh"
"#,
            dir.path().display()
        ),
    );

    let runner = HookRunner::from_file(&hooks_path, None);
    let result = runner
        .pre_tool_execute("bash", &serde_json::json!({"command": "rm -rf /"}))
        .await
        .unwrap();

    assert_eq!(result.action.as_deref(), Some("deny"));
    assert_eq!(result.message.as_deref(), Some("Blocked by test hook"));
}

// ── pre_tool_execute: modify ────────────────────────────────────────

#[tokio::test]
async fn pre_tool_modify_arguments() {
    let dir = tempfile::tempdir().unwrap();
    make_hook_script(
        dir.path(),
        "modify.sh",
        r#"#!/bin/sh
echo '{"action":"modify","arguments":{"command":"echo safe"}}'
"#,
    );
    let hooks_path = write_hooks_toml(
        dir.path(),
        &format!(
            r#"
[sanitize]
event = "pre_tool_execute"
command = "{}/modify.sh"
tools = ["bash"]
"#,
            dir.path().display()
        ),
    );

    let runner = HookRunner::from_file(&hooks_path, None);
    let result = runner
        .pre_tool_execute("bash", &serde_json::json!({"command": "rm -rf /"}))
        .await
        .unwrap();

    assert_eq!(result.action.as_deref(), Some("modify"));
    let args = result.arguments.unwrap();
    assert_eq!(args["command"].as_str(), Some("echo safe"));
}

// ── pre_tool_execute: proceed (empty stdout) ────────────────────────

#[tokio::test]
async fn pre_tool_proceed_on_empty_output() {
    let dir = tempfile::tempdir().unwrap();
    make_hook_script(
        dir.path(),
        "pass.sh",
        "#!/bin/sh\n# no output = proceed\n",
    );
    let hooks_path = write_hooks_toml(
        dir.path(),
        &format!(
            r#"
[passthrough]
event = "pre_tool_execute"
command = "{}/pass.sh"
"#,
            dir.path().display()
        ),
    );

    let runner = HookRunner::from_file(&hooks_path, None);
    let result = runner
        .pre_tool_execute("read", &serde_json::json!({"file_path": "/tmp/foo"}))
        .await
        .unwrap();

    // No action = proceed
    assert!(
        result.action.is_none()
            || result.action.as_deref() == Some("proceed"),
    );
}

// ── pre_tool_execute: tool filter ───────────────────────────────────

#[tokio::test]
async fn pre_tool_filter_by_tool_name() {
    let dir = tempfile::tempdir().unwrap();
    make_hook_script(
        dir.path(),
        "deny.sh",
        r#"#!/bin/sh
echo '{"action":"deny","message":"denied"}'
"#,
    );
    let hooks_path = write_hooks_toml(
        dir.path(),
        &format!(
            r#"
[bash-only]
event = "pre_tool_execute"
command = "{}/deny.sh"
tools = ["bash"]
"#,
            dir.path().display()
        ),
    );

    let runner = HookRunner::from_file(&hooks_path, None);

    // Should fire for bash
    let result = runner
        .pre_tool_execute("bash", &serde_json::json!({"command": "ls"}))
        .await
        .unwrap();
    assert_eq!(result.action.as_deref(), Some("deny"));

    // Should NOT fire for read
    let result = runner
        .pre_tool_execute("read", &serde_json::json!({"file_path": "/tmp"}))
        .await
        .unwrap();
    assert!(result.action.is_none());
}

// ── post_tool_execute: modify output ────────────────────────────────

#[tokio::test]
async fn post_tool_modify_output() {
    let dir = tempfile::tempdir().unwrap();
    make_hook_script(
        dir.path(),
        "redact.sh",
        r#"#!/bin/sh
echo '{"action":"modify","output":"[REDACTED]","success":true}'
"#,
    );
    let hooks_path = write_hooks_toml(
        dir.path(),
        &format!(
            r#"
[redact]
event = "post_tool_execute"
command = "{}/redact.sh"
"#,
            dir.path().display()
        ),
    );

    let runner = HookRunner::from_file(&hooks_path, None);
    let (output, success) = runner
        .post_tool_execute(
            "bash",
            &serde_json::json!({"command": "cat /etc/passwd"}),
            "root:x:0:0:...".to_string(),
            true,
        )
        .await;

    assert_eq!(output, "[REDACTED]");
    assert!(success);
}

// ── post_tool_execute: passthrough ──────────────────────────────────

#[tokio::test]
async fn post_tool_passthrough() {
    let dir = tempfile::tempdir().unwrap();
    make_hook_script(
        dir.path(),
        "noop.sh",
        "#!/bin/sh\n# nothing\n",
    );
    let hooks_path = write_hooks_toml(
        dir.path(),
        &format!(
            r#"
[noop]
event = "post_tool_execute"
command = "{}/noop.sh"
"#,
            dir.path().display()
        ),
    );

    let runner = HookRunner::from_file(&hooks_path, None);
    let (output, success) = runner
        .post_tool_execute(
            "bash",
            &serde_json::json!({"command": "ls"}),
            "original output".to_string(),
            true,
        )
        .await;

    assert_eq!(output, "original output");
    assert!(success);
}

// ── agent_start: inject system context ──────────────────────────────

#[tokio::test]
async fn agent_start_injects_context() {
    let dir = tempfile::tempdir().unwrap();
    make_hook_script(
        dir.path(),
        "inject.sh",
        r#"#!/bin/sh
echo '{"system_context":"Always be polite."}'
"#,
    );
    let hooks_path = write_hooks_toml(
        dir.path(),
        &format!(
            r#"
[inject-ctx]
event = "agent_start"
command = "{}/inject.sh"
"#,
            dir.path().display()
        ),
    );

    let runner = HookRunner::from_file(&hooks_path, None);
    let ctx = runner.agent_start("hello", "test-model").await;

    assert_eq!(ctx.as_deref(), Some("Always be polite."));
}

// ── agent_start: no context returned ────────────────────────────────

#[tokio::test]
async fn agent_start_no_context() {
    let dir = tempfile::tempdir().unwrap();
    make_hook_script(dir.path(), "empty.sh", "#!/bin/sh\n");
    let hooks_path = write_hooks_toml(
        dir.path(),
        &format!(
            r#"
[empty]
event = "agent_start"
command = "{}/empty.sh"
"#,
            dir.path().display()
        ),
    );

    let runner = HookRunner::from_file(&hooks_path, None);
    let ctx = runner.agent_start("hello", "test-model").await;
    assert!(ctx.is_none());
}

// ── agent_done: rewrite response ────────────────────────────────────

#[tokio::test]
async fn agent_done_rewrites_response() {
    let dir = tempfile::tempdir().unwrap();
    make_hook_script(
        dir.path(),
        "rewrite.sh",
        r#"#!/bin/sh
echo '{"action":"modify","response":"Rewritten response."}'
"#,
    );
    let hooks_path = write_hooks_toml(
        dir.path(),
        &format!(
            r#"
[rewrite]
event = "agent_done"
command = "{}/rewrite.sh"
"#,
            dir.path().display()
        ),
    );

    let runner = HookRunner::from_file(&hooks_path, None);
    let result = runner.agent_done("Original.", 5, "test-model").await;

    assert_eq!(result.as_deref(), Some("Rewritten response."));
}

// ── agent_done: no modification ─────────────────────────────────────

#[tokio::test]
async fn agent_done_no_modification() {
    let dir = tempfile::tempdir().unwrap();
    make_hook_script(dir.path(), "noop.sh", "#!/bin/sh\n");
    let hooks_path = write_hooks_toml(
        dir.path(),
        &format!(
            r#"
[noop]
event = "agent_done"
command = "{}/noop.sh"
"#,
            dir.path().display()
        ),
    );

    let runner = HookRunner::from_file(&hooks_path, None);
    let result = runner.agent_done("Original.", 0, "test-model").await;
    assert!(result.is_none());
}

// ── fail_closed: hook failure denies ────────────────────────────────

#[tokio::test]
async fn pre_tool_fail_closed_denies() {
    let dir = tempfile::tempdir().unwrap();
    make_hook_script(
        dir.path(),
        "fail.sh",
        "#!/bin/sh\nexit 1\n",
    );
    let hooks_path = write_hooks_toml(
        dir.path(),
        &format!(
            r#"
[strict]
event = "pre_tool_execute"
command = "{}/fail.sh"
fail_closed = true
"#,
            dir.path().display()
        ),
    );

    let runner = HookRunner::from_file(&hooks_path, None);
    let result = runner
        .pre_tool_execute("bash", &serde_json::json!({"command": "ls"}))
        .await
        .unwrap();

    assert_eq!(result.action.as_deref(), Some("deny"));
    assert!(result.message.as_deref().unwrap().contains("failed"));
}

// ── fail_open: hook failure proceeds ────────────────────────────────

#[tokio::test]
async fn pre_tool_fail_open_proceeds() {
    let dir = tempfile::tempdir().unwrap();
    make_hook_script(
        dir.path(),
        "fail.sh",
        "#!/bin/sh\nexit 1\n",
    );
    let hooks_path = write_hooks_toml(
        dir.path(),
        &format!(
            r#"
[lenient]
event = "pre_tool_execute"
command = "{}/fail.sh"
fail_closed = false
"#,
            dir.path().display()
        ),
    );

    let runner = HookRunner::from_file(&hooks_path, None);
    let result = runner
        .pre_tool_execute("bash", &serde_json::json!({"command": "ls"}))
        .await
        .unwrap();

    // Should proceed (no deny)
    assert!(result.action.is_none() || result.action.as_deref() == Some("proceed"));
}

// ── priority ordering ───────────────────────────────────────────────

#[tokio::test]
async fn hooks_execute_in_priority_order() {
    let dir = tempfile::tempdir().unwrap();

    // Low priority hook: modify args to "low"
    make_hook_script(
        dir.path(),
        "low.sh",
        r#"#!/bin/sh
echo '{"action":"modify","arguments":{"command":"low"}}'
"#,
    );
    // High priority hook: modify args to "high"
    make_hook_script(
        dir.path(),
        "high.sh",
        r#"#!/bin/sh
echo '{"action":"modify","arguments":{"command":"high"}}'
"#,
    );

    let hooks_path = write_hooks_toml(
        dir.path(),
        &format!(
            r#"
[low-priority]
event = "pre_tool_execute"
command = "{dir}/low.sh"
priority = 100

[high-priority]
event = "pre_tool_execute"
command = "{dir}/high.sh"
priority = 10
"#,
            dir = dir.path().display()
        ),
    );

    let runner = HookRunner::from_file(&hooks_path, None);
    let result = runner
        .pre_tool_execute("bash", &serde_json::json!({"command": "original"}))
        .await
        .unwrap();

    // high-priority (10) runs first → sets "high"
    // low-priority (100) runs second → overwrites with "low"
    assert_eq!(result.action.as_deref(), Some("modify"));
    let args = result.arguments.unwrap();
    assert_eq!(args["command"].as_str(), Some("low"));
}

// ── hook receives correct stdin data ────────────────────────────────

#[tokio::test]
async fn hook_receives_stdin_data() {
    let dir = tempfile::tempdir().unwrap();
    let output_path = dir.path().join("stdin_capture.json");

    // Script that captures stdin to a file
    make_hook_script(
        dir.path(),
        "capture.sh",
        &format!(
            "#!/bin/sh\ncat > {}\n",
            output_path.display()
        ),
    );
    let hooks_path = write_hooks_toml(
        dir.path(),
        &format!(
            r#"
[capture]
event = "pre_tool_execute"
command = "{}/capture.sh"
"#,
            dir.path().display()
        ),
    );

    let runner = HookRunner::from_file(&hooks_path, None);
    let _ = runner
        .pre_tool_execute("bash", &serde_json::json!({"command": "echo hello"}))
        .await;

    // Read captured stdin
    let captured = fs::read_to_string(&output_path).unwrap();
    let data: serde_json::Value = serde_json::from_str(&captured).unwrap();

    assert_eq!(data["hook"].as_str(), Some("pre_tool_execute"));
    assert_eq!(data["data"]["tool_name"].as_str(), Some("bash"));
    assert_eq!(data["data"]["arguments"]["command"].as_str(), Some("echo hello"));
}

// ── empty hook runner ───────────────────────────────────────────────

#[tokio::test]
async fn empty_runner_passes_through() {
    let runner = HookRunner::empty();

    let pre = runner
        .pre_tool_execute("bash", &serde_json::json!({}))
        .await
        .unwrap();
    assert!(pre.action.is_none());

    let (output, success) = runner
        .post_tool_execute("bash", &serde_json::json!({}), "out".to_string(), true)
        .await;
    assert_eq!(output, "out");
    assert!(success);

    assert!(runner.agent_start("hello", "model").await.is_none());
    assert!(runner.agent_done("resp", 0, "model").await.is_none());
}

// ── hook disabled via config ────────────────────────────────────────

#[tokio::test]
async fn hook_disabled_via_config() {
    let dir = tempfile::tempdir().unwrap();
    // Create .agents/hooks.toml structure for discovery
    let agents_dir = dir.path().join(".agents");
    fs::create_dir_all(&agents_dir).unwrap();

    make_hook_script(
        dir.path(),
        "deny.sh",
        r#"#!/bin/sh
echo '{"action":"deny","message":"should not fire"}'
"#,
    );
    write_hooks_toml(
        &agents_dir,
        &format!(
            r#"
[blocker]
event = "pre_tool_execute"
command = "{}/deny.sh"
"#,
            dir.path().display()
        ),
    );

    // Disable the hook via config
    let mut config = ollama_code::config::Config::default();
    let mut hooks_map = std::collections::HashMap::new();
    hooks_map.insert("blocker".to_string(), toml::Value::Boolean(false));
    config.hooks = Some(hooks_map);

    // Use discover (which filters disabled hooks) instead of from_file
    let runner = HookRunner::discover(dir.path().to_str().unwrap(), Some(&config));
    let result = runner
        .pre_tool_execute("bash", &serde_json::json!({"command": "ls"}))
        .await
        .unwrap();

    // Should not fire — hook disabled
    assert!(result.action.is_none());
}
