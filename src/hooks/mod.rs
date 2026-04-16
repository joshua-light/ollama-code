//! Subprocess hooks fired at agent lifecycle events.
//!
//! Hooks are defined in `hooks.toml` files (user- and project-scoped) and run
//! as short-lived subprocesses that exchange JSON on stdin/stdout. See the
//! README for the wire protocol.

mod discover;
mod exec;
mod types;

pub use types::{
    AgentDoneResult, AgentStartResult, HookEntry, HookEvent, PostToolResult, PreToolResult,
};

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use self::discover::{find_project_hooks, load_hooks_file, user_hooks_path};
use self::exec::execute_hook;
use self::types::ResolvedHook;

/// Manages discovered hooks and executes them.
pub struct HookRunner {
    hooks: Vec<ResolvedHook>,
}

/// Sort hooks by priority (lower first), then by name.
fn sort_hooks(hooks: &mut [ResolvedHook]) {
    hooks.sort_by(|a, b| {
        a.entry
            .priority()
            .cmp(&b.entry.priority())
            .then_with(|| a.name.cmp(&b.name))
    });
}

impl HookRunner {
    /// Discover and load hooks from user and project locations.
    ///
    /// - User hooks: `~/.config/ollama-code/hooks.toml`
    /// - Project hooks: `.agents/hooks.toml` (walk up from cwd)
    /// - Project hooks override user hooks by name.
    /// - Hooks disabled via `[hooks] name = false` in config are filtered out.
    pub fn discover(cwd: &str, config: Option<&crate::config::Config>) -> Self {
        let mut hooks_map: std::collections::HashMap<String, ResolvedHook> =
            std::collections::HashMap::new();

        // User-level hooks
        let user_path = user_hooks_path();
        if user_path.is_file() {
            let base = user_path.parent().unwrap_or(Path::new(".")).to_path_buf();
            for (name, entry) in load_hooks_file(&user_path) {
                let hook_config = config.and_then(|c| c.hook_config(&name)).cloned();
                hooks_map.insert(
                    name.clone(),
                    ResolvedHook {
                        name,
                        entry,
                        base_dir: base.clone(),
                        config: hook_config,
                    },
                );
            }
        }

        // Project-level hooks (override user by name)
        if let Some(project_path) = find_project_hooks(cwd) {
            let base = project_path
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf();
            for (name, entry) in load_hooks_file(&project_path) {
                let hook_config = config.and_then(|c| c.hook_config(&name)).cloned();
                hooks_map.insert(
                    name.clone(),
                    ResolvedHook {
                        name,
                        entry,
                        base_dir: base.clone(),
                        config: hook_config,
                    },
                );
            }
        }

        // Filter disabled hooks
        let hooks: Vec<ResolvedHook> = if let Some(cfg) = config {
            hooks_map
                .into_values()
                .filter(|h| cfg.is_hook_enabled(&h.name))
                .collect()
        } else {
            hooks_map.into_values().collect()
        };

        let mut sorted = hooks;
        sort_hooks(&mut sorted);
        Self { hooks: sorted }
    }

    /// Create a hook runner from a specific hooks file, bypassing the
    /// standard user/project discovery. Useful for testing and for loading
    /// hooks from non-standard locations.
    pub fn from_file(path: &Path, config: Option<&crate::config::Config>) -> Self {
        let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let entries = load_hooks_file(path);
        let mut hooks: Vec<ResolvedHook> = entries
            .into_iter()
            .map(|(name, entry)| {
                let hook_config = config.and_then(|c| c.hook_config(&name)).cloned();
                ResolvedHook {
                    name,
                    entry,
                    base_dir: base.clone(),
                    config: hook_config,
                }
            })
            .collect();
        sort_hooks(&mut hooks);
        Self { hooks }
    }

    /// Create an empty hook runner (no hooks).
    pub fn empty() -> Self {
        Self { hooks: Vec::new() }
    }

    fn hooks_for(&self, event: HookEvent) -> impl Iterator<Item = &ResolvedHook> {
        self.hooks.iter().filter(move |h| h.entry.event == event)
    }

    /// Run `pre_tool_execute` hooks for the given tool call.
    ///
    /// Returns `Ok(PreToolResult)` with the cumulative result.
    /// A "deny" from any hook short-circuits immediately.
    pub async fn pre_tool_execute(
        &self,
        tool_name: &str,
        arguments: &Value,
    ) -> Result<PreToolResult> {
        // Evaluate matching dynamically so a "rewrite" that changes the tool
        // name is visible to subsequent hooks in the chain.
        let mut result = PreToolResult::default();
        let mut current_tool_name = tool_name.to_string();
        let mut current_args = arguments.clone();

        for hook in self.hooks_for(HookEvent::PreToolExecute) {
            if !hook.entry.matches_tool(&current_tool_name) {
                continue;
            }
            if !hook.entry.matches_args(&current_args) {
                continue;
            }

            let data = serde_json::json!({
                "tool_name": current_tool_name,
                "arguments": current_args,
            });
            match execute_hook(hook, HookEvent::PreToolExecute, data).await {
                Ok(Some(output)) => {
                    if let Ok(r) = serde_json::from_value::<PreToolResult>(output) {
                        match r.action.as_deref() {
                            Some("deny") => return Ok(r),
                            Some("rewrite") => {
                                if let Some(new_name) = r.tool_name {
                                    current_tool_name = new_name;
                                    result.action = Some("rewrite".to_string());
                                    result.tool_name = Some(current_tool_name.clone());
                                } else if result.action.as_deref() != Some("rewrite") {
                                    result.action = Some("modify".to_string());
                                }
                                if let Some(args) = r.arguments {
                                    current_args = args;
                                }
                                result.arguments = Some(current_args.clone());
                            }
                            Some("modify") => {
                                if let Some(args) = r.arguments {
                                    current_args = args;
                                }
                                // Keep "rewrite" if a prior hook already rewrote the name
                                if result.action.as_deref() != Some("rewrite") {
                                    result.action = Some("modify".to_string());
                                }
                                result.arguments = Some(current_args.clone());
                            }
                            _ => {} // "proceed" or absent — continue
                        }
                    }
                }
                Ok(None) => {} // no output = passthrough
                Err(e) => {
                    if hook.entry.fail_closed() {
                        return Ok(PreToolResult {
                            action: Some("deny".to_string()),
                            message: Some(format!("Hook '{}' failed: {}", hook.name, e)),
                            arguments: None,
                            tool_name: None,
                        });
                    }
                    eprintln!("[hooks] warning: '{}' failed (fail-open): {}", hook.name, e);
                }
            }
        }

        Ok(result)
    }

    /// Run `post_tool_execute` hooks for the given tool result.
    ///
    /// Returns the (possibly modified) output and success flag.
    pub async fn post_tool_execute(
        &self,
        tool_name: &str,
        arguments: &Value,
        output: String,
        success: bool,
    ) -> (String, bool) {
        let mut current_output = output;
        let mut current_success = success;

        for hook in self.hooks_for(HookEvent::PostToolExecute) {
            if !hook.entry.matches_tool(tool_name) {
                continue;
            }
            if !hook.entry.matches_args(arguments) {
                continue;
            }

            let data = serde_json::json!({
                "tool_name": tool_name,
                "arguments": arguments,
                "output": current_output,
                "success": current_success,
            });
            match execute_hook(hook, HookEvent::PostToolExecute, data).await {
                Ok(Some(value)) => {
                    if let Ok(r) = serde_json::from_value::<PostToolResult>(value) {
                        if r.action.as_deref() == Some("modify") {
                            if let Some(new_output) = r.output {
                                current_output = new_output;
                            }
                            if let Some(new_success) = r.success {
                                current_success = new_success;
                            }
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    let tag = if hook.entry.fail_closed() { "fail-closed" } else { "fail-open" };
                    eprintln!(
                        "[hooks] warning: post_tool_execute '{}' failed ({}): {}",
                        hook.name, tag, e
                    );
                }
            }
        }

        (current_output, current_success)
    }

    /// Run `agent_start` hooks. Returns any extra system context to inject.
    pub async fn agent_start(&self, user_message: &str, model: &str) -> Option<String> {
        let mut system_context: Option<String> = None;

        for hook in self.hooks_for(HookEvent::AgentStart) {
            let data = serde_json::json!({
                "user_message": user_message,
                "model": model,
            });
            match execute_hook(hook, HookEvent::AgentStart, data).await {
                Ok(Some(value)) => {
                    if let Ok(r) = serde_json::from_value::<AgentStartResult>(value) {
                        if let Some(ctx) = r.system_context {
                            match &mut system_context {
                                Some(existing) => {
                                    existing.push('\n');
                                    existing.push_str(&ctx);
                                }
                                None => system_context = Some(ctx),
                            }
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("[hooks] warning: agent_start '{}' failed: {}", hook.name, e);
                }
            }
        }

        system_context
    }

    /// Run `agent_done` hooks. Returns a possibly rewritten response.
    pub async fn agent_done(
        &self,
        response: &str,
        tool_calls_count: u32,
        model: &str,
    ) -> Option<String> {
        let mut current_response = response.to_string();
        let mut modified = false;

        for hook in self.hooks_for(HookEvent::AgentDone) {
            let data = serde_json::json!({
                "response": current_response,
                "tool_calls_count": tool_calls_count,
                "model": model,
            });
            match execute_hook(hook, HookEvent::AgentDone, data).await {
                Ok(Some(value)) => {
                    if let Ok(r) = serde_json::from_value::<AgentDoneResult>(value) {
                        if r.action.as_deref() == Some("modify") {
                            if let Some(new_resp) = r.response {
                                current_response = new_resp;
                                modified = true;
                            }
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("[hooks] warning: agent_done '{}' failed: {}", hook.name, e);
                }
            }
        }

        if modified {
            Some(current_response)
        } else {
            None
        }
    }
}
