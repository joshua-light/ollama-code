use std::collections::HashSet;
use std::path::Path;

use crate::config::Config;
use crate::mcp;
use crate::plugin::{self, ExternalTool};
use crate::skills::{self, SkillMeta};
use crate::tools::{
    new_evidence_store, BashTool, EditTool, EvidenceAddTool, EvidenceGetTool, EvidenceListTool,
    GlobTool, GrepTool, ReadTool, SubagentToolDef, Tool, ToolRegistry, WriteTool,
};

/// Discover `CLAUDE.md` and `AGENTS.md` files by walking up from `cwd`.
/// Stops at the first directory that contains either file.
fn discover_project_docs(cwd: &str) -> Vec<(String, String)> {
    let mut dir = Path::new(cwd).to_path_buf();
    loop {
        let mut found = Vec::new();
        for name in &["CLAUDE.md", "AGENTS.md"] {
            let path = dir.join(name);
            if let Ok(content) = std::fs::read_to_string(&path) {
                found.push((name.to_string(), content));
            }
        }
        if !found.is_empty() {
            return found;
        }
        if !dir.pop() {
            break;
        }
    }
    Vec::new()
}

/// Collapse 3+ consecutive newlines to 2 (one blank line max).
fn normalize_newlines(text: String) -> String {
    let mut result = String::with_capacity(text.len());
    let mut newline_count = 0u32;
    for ch in text.chars() {
        if ch == '\n' {
            newline_count += 1;
            if newline_count <= 2 {
                result.push(ch);
            }
        } else {
            newline_count = 0;
            result.push(ch);
        }
    }
    result
}

/// Build the tool registry, discover plugins, and start MCP servers.
/// When `include_extensions` is false (sub-agents), omits subagent tool, plugins, and MCP.
pub(super) fn build_tools_and_servers(
    cfg: &Config,
    context_size: u64,
    cwd: &str,
    include_extensions: bool,
) -> (ToolRegistry, HashSet<String>, Vec<mcp::McpServer>) {
    let mut tools = ToolRegistry::new();
    macro_rules! register_if_enabled {
        ($name:expr, $tool:expr) => {
            if cfg.is_tool_enabled($name) {
                tools.register(Box::new($tool));
            }
        };
    }
    register_if_enabled!("bash", BashTool);
    register_if_enabled!("read", ReadTool::new());
    register_if_enabled!("edit", EditTool);
    register_if_enabled!("write", WriteTool);
    register_if_enabled!("glob", GlobTool);
    register_if_enabled!("grep", GrepTool::new(context_size));

    let mut plugin_confirm_tools = HashSet::new();
    let mut mcp_servers = Vec::new();

    if include_extensions {
        register_if_enabled!("subagent", SubagentToolDef);

        // Evidence store: three tools share a single Arc<Mutex<HashMap>> so
        // they read and write the same in-memory snippet collection. Lives
        // on the Agent so it survives context compaction.
        let evidence_store = new_evidence_store();
        register_if_enabled!("evidence_add", EvidenceAddTool::new(evidence_store.clone()));
        register_if_enabled!("evidence_get", EvidenceGetTool::new(evidence_store.clone()));
        register_if_enabled!("evidence_list", EvidenceListTool::new(evidence_store));

        let plugin_dirs: Vec<std::path::PathBuf> = cfg
            .plugin_dirs
            .as_ref()
            .map(|dirs| dirs.iter().map(std::path::PathBuf::from).collect())
            .unwrap_or_default();
        let discovered = plugin::discover_plugins(cwd, &plugin_dirs, Some(cfg));
        for dp in &discovered {
            for tool_def in &dp.manifest.tools {
                if !cfg.is_tool_enabled(&tool_def.name) {
                    continue;
                }
                let plugin_cfg = cfg.plugin_config(&dp.manifest.name).cloned();
                let ext_tool = ExternalTool::from_plugin(dp, tool_def, plugin_cfg);
                if ext_tool.needs_confirm() {
                    plugin_confirm_tools.insert(tool_def.name.clone());
                }
                tools.register(Box::new(ext_tool));
            }
        }

        if let Some(ref servers) = cfg.mcp {
            for server in mcp::start_servers(servers, |name| cfg.is_tool_enabled(name)) {
                for mcp_tool in server.create_tools() {
                    let tool_name = mcp_tool.name().to_string();
                    if !cfg.is_tool_enabled(&tool_name) {
                        continue;
                    }
                    if server.needs_confirm {
                        plugin_confirm_tools.insert(tool_name);
                    }
                    tools.register(Box::new(mcp_tool));
                }
                mcp_servers.push(server);
            }
        }
    }

    (tools, plugin_confirm_tools, mcp_servers)
}

/// Build the system prompt from template, skills, and project docs.
pub(super) fn build_system_prompt(
    cwd: &str,
    include_extensions: bool,
    discovered_skills: &[SkillMeta],
) -> (String, usize, usize, Vec<(String, usize)>) {
    let subagent_desc = if include_extensions {
        "- subagent(task): Spawn a sub-agent with a fresh context to handle a focused task. \
         The sub-agent cannot see this conversation, so include all necessary context in the \
         task description. Use for: research across many files, complex multi-step operations, \
         or any task that would benefit from a clean context window"
    } else {
        ""
    };

    let skill_desc = if !discovered_skills.is_empty() {
        "- skill(name, args?): Activate a skill by name. Use when a task matches an available skill"
    } else {
        ""
    };

    let config_dir = crate::config::config_dir();

    let mut prompt = include_str!("../../SYSTEM_PROMPT.md")
        .replace("{cwd}", cwd)
        .replace("{config_dir}", &config_dir.to_string_lossy())
        .replace("{subagent_tool}", subagent_desc)
        .replace("{skill_tool}", skill_desc);

    let base_len = prompt.len();

    let skills_summary_len = if !discovered_skills.is_empty() {
        let summary = skills::format_skill_summaries(discovered_skills);
        let l = summary.len();
        prompt.push_str(&summary);
        l
    } else {
        0
    };

    let project_docs_info = {
        let docs = discover_project_docs(cwd);
        let mut info = Vec::new();
        for (name, content) in &docs {
            prompt.push_str(&format!(
                "\n\n---\n\n# Project Instructions ({})\n\n{}",
                name, content
            ));
            info.push((name.clone(), content.len()));
        }
        info
    };

    let prompt = normalize_newlines(prompt);

    (prompt, base_len, skills_summary_len, project_docs_info)
}
