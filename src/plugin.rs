use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::Result;
use serde::Deserialize;
use serde_json::Value;

use crate::config::Config;
use crate::tools::{Tool, ToolDefinition};

/// Parsed `PLUGIN.toml` manifest.
#[derive(Debug, Clone, Deserialize)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    pub description: String,
    #[serde(default)]
    pub tools: Vec<PluginToolDef>,
}

/// A tool definition within a plugin manifest.
#[derive(Debug, Clone, Deserialize)]
pub struct PluginToolDef {
    pub name: String,
    pub description: String,
    pub command: String,
    pub needs_confirm: Option<bool>,
    pub timeout: Option<u64>,
    /// JSON Schema string for tool parameters.
    pub parameters: Option<String>,
}

/// A discovered plugin: manifest + location on disk.
#[derive(Debug, Clone)]
pub struct DiscoveredPlugin {
    pub manifest: PluginManifest,
    pub dir: PathBuf,
}

/// Discover plugins from project and user directories.
///
/// - Project plugins: `.agents/plugins/*/PLUGIN.toml` (walk up from `cwd`).
/// - User plugins: `user_plugins_dir/*/PLUGIN.toml`.
/// - Project plugins override user plugins by name.
/// - Plugins disabled via config (`name = false`) are filtered out.
///
/// `user_plugins_dir` and `config` can be overridden for testing.
pub fn discover_plugins(
    cwd: &str,
    plugin_dirs: &[PathBuf],
    config: Option<&Config>,
) -> Vec<DiscoveredPlugin> {
    let dirs: Vec<PathBuf> = if plugin_dirs.is_empty() {
        vec![default_user_plugins_dir()]
    } else {
        plugin_dirs.to_vec()
    };

    let mut plugins = Vec::new();
    for dir in &dirs {
        for p in load_plugins_from_dir(dir) {
            if let Some(existing) = plugins.iter_mut().find(|x: &&mut DiscoveredPlugin| x.manifest.name == p.manifest.name) {
                *existing = p;
            } else {
                plugins.push(p);
            }
        }
    }

    // Project-scoped plugins override user/extra plugins by name.
    let project_plugins = discover_project_plugins(cwd);
    for pp in project_plugins {
        if let Some(existing) = plugins.iter_mut().find(|p| p.manifest.name == pp.manifest.name) {
            *existing = pp;
        } else {
            plugins.push(pp);
        }
    }

    if let Some(cfg) = config {
        plugins.retain(|p| cfg.is_tool_enabled(&p.manifest.name));
    }

    plugins.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
    plugins
}

/// Default user plugins directory: `~/.config/ollama-code/plugins/`.
fn default_user_plugins_dir() -> PathBuf {
    crate::config::config_dir().join("plugins")
}

/// Walk up from `cwd` looking for `.agents/plugins/*/PLUGIN.toml`.
/// Stops at the first directory that contains valid plugins.
fn discover_project_plugins(cwd: &str) -> Vec<DiscoveredPlugin> {
    let mut dir = Path::new(cwd).to_path_buf();
    loop {
        let plugins_dir = dir.join(".agents").join("plugins");
        let found = load_plugins_from_dir(&plugins_dir);
        if !found.is_empty() {
            return found;
        }
        if !dir.pop() {
            break;
        }
    }
    Vec::new()
}

/// Load all valid plugins from a directory (`dir/*/PLUGIN.toml`).
fn load_plugins_from_dir(dir: &Path) -> Vec<DiscoveredPlugin> {
    let mut plugins = Vec::new();
    if !dir.is_dir() {
        return plugins;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return plugins,
    };
    for entry in entries.flatten() {
        let plugin_dir = entry.path();
        if !plugin_dir.is_dir() {
            continue;
        }
        let manifest_path = plugin_dir.join("PLUGIN.toml");
        if !manifest_path.is_file() {
            continue;
        }
        match load_manifest(&manifest_path) {
            Ok(manifest) => plugins.push(DiscoveredPlugin {
                manifest,
                dir: plugin_dir,
            }),
            Err(_) => {
                // Skip invalid manifests silently.
            }
        }
    }
    plugins
}

/// Parse a `PLUGIN.toml` file.
fn load_manifest(path: &Path) -> anyhow::Result<PluginManifest> {
    let content = std::fs::read_to_string(path)?;
    let manifest: PluginManifest = toml::from_str(&content)?;
    Ok(manifest)
}

// ---------------------------------------------------------------------------
// ExternalTool — wraps a plugin tool as a Tool trait object
// ---------------------------------------------------------------------------

const DEFAULT_PLUGIN_TIMEOUT_SECS: u64 = 120;

/// A tool backed by an external process (from a plugin).
pub struct ExternalTool {
    tool_name: String,
    description: String,
    command_path: PathBuf,
    needs_confirm_flag: bool,
    timeout: Duration,
    parameters: Value,
    plugin_config: Option<toml::map::Map<String, toml::Value>>,
}

impl ExternalTool {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tool_name: String,
        description: String,
        command: String,
        plugin_dir: PathBuf,
        needs_confirm: Option<bool>,
        timeout: Option<u64>,
        parameters: Option<String>,
        plugin_config: Option<toml::map::Map<String, toml::Value>>,
    ) -> Self {
        let params = parameters
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));

        // Resolve relative commands to absolute paths once at construction time.
        let command_path = if command.starts_with('/') {
            PathBuf::from(command)
        } else {
            plugin_dir.join(command)
        };

        Self {
            tool_name,
            description,
            command_path,
            needs_confirm_flag: needs_confirm.unwrap_or(false),
            timeout: Duration::from_secs(timeout.unwrap_or(DEFAULT_PLUGIN_TIMEOUT_SECS)),
            parameters: params,
            plugin_config,
        }
    }

    /// Whether this tool requires user confirmation before execution.
    pub fn needs_confirm(&self) -> bool {
        self.needs_confirm_flag
    }

    /// Build an ExternalTool from a discovered plugin and one of its tool defs.
    pub fn from_plugin(
        plugin: &DiscoveredPlugin,
        tool_def: &PluginToolDef,
        plugin_config: Option<toml::map::Map<String, toml::Value>>,
    ) -> Self {
        Self::new(
            tool_def.name.clone(),
            tool_def.description.clone(),
            tool_def.command.clone(),
            plugin.dir.clone(),
            tool_def.needs_confirm,
            tool_def.timeout,
            tool_def.parameters.clone(),
            plugin_config,
        )
    }
}

impl Tool for ExternalTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.tool_name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let mut payload = serde_json::json!({
            "arguments": arguments,
        });
        if let Some(cfg) = &self.plugin_config {
            payload["config"] = serde_json::to_value(cfg)?;
        }

        let stdin_data = serde_json::to_string(&payload)?;

        let mut child = Command::new(&self.command_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn plugin command '{}': {}", self.command_path.display(), e))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(stdin_data.as_bytes())?;
        }

        let output = crate::process::wait_with_timeout(&mut child, self.timeout, "Plugin command")?;
        let (result, _) = crate::tools::format_bash_output(&output);
        Ok(result)
    }
}
