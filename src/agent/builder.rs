use std::collections::HashSet;
use std::path::Path;

use crate::agent::plan::SharedTodoList;
use crate::config::Config;
use crate::mcp;
use crate::plugin::{self, ExternalTool};
use crate::skills::{self, SkillMeta};
use crate::tools::{
    new_evidence_store, BashTool, EditTool, EvidenceAddTool, EvidenceGetTool, EvidenceListTool,
    GlobTool, GrepTool, PlanAddStepTool, PlanListStepsTool, PlanMarkDoneTool,
    PlanMarkInProgressTool, PlanSkipStepTool, ReadTool, SubagentToolDef, Tool, ToolRegistry,
    WriteTool,
};

/// Which kind of agent we're building tools for. Controls which tools are
/// registered (read-only-only for the planner, all five plan tools for the
/// main agent, none for plain sub-agents).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AgentMode {
    /// Top-level user-facing agent.
    Main,
    /// Generic sub-agent spawned via the `subagent` tool.
    Subagent,
    /// Planner sub-agent: read-only tools + `plan_add_step` only.
    Planner,
}

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
///
/// `mode` controls which tool subset is registered:
/// - `Main`: every enabled tool, plus all five plan tools sharing `plan_list`.
/// - `Subagent`: built-ins only (no plugins/MCP/subagent recursion); no plan tools.
/// - `Planner`: read-only built-ins (read/glob/grep) + `plan_add_step` only.
pub(super) fn build_tools_and_servers(
    cfg: &Config,
    context_size: u64,
    cwd: &str,
    mode: AgentMode,
    plan_list: &SharedTodoList,
) -> (ToolRegistry, HashSet<String>, Vec<mcp::McpServer>) {
    let mut tools = ToolRegistry::new();
    let include_extensions = matches!(mode, AgentMode::Main);
    macro_rules! register_if_enabled {
        ($name:expr, $tool:expr) => {
            if cfg.is_tool_enabled($name) {
                tools.register(Box::new($tool));
            }
        };
    }
    if matches!(mode, AgentMode::Main | AgentMode::Subagent) {
        register_if_enabled!("bash", BashTool);
    }
    register_if_enabled!("read", ReadTool::new());
    if matches!(mode, AgentMode::Main | AgentMode::Subagent) {
        register_if_enabled!("edit", EditTool);
        register_if_enabled!("write", WriteTool);
    }
    register_if_enabled!("glob", GlobTool);
    register_if_enabled!("grep", GrepTool::new(context_size));

    let mut plugin_confirm_tools = HashSet::new();
    let mut mcp_servers = Vec::new();

    // Plan tools share a single Arc<Mutex<TodoList>> so the planner sub-agent
    // and the main agent operate on the same plan state.
    match mode {
        AgentMode::Main => {
            register_if_enabled!("plan_add_step", PlanAddStepTool::new(plan_list.clone()));
            register_if_enabled!(
                "plan_mark_in_progress",
                PlanMarkInProgressTool::new(plan_list.clone())
            );
            register_if_enabled!("plan_mark_done", PlanMarkDoneTool::new(plan_list.clone()));
            register_if_enabled!("plan_skip_step", PlanSkipStepTool::new(plan_list.clone()));
            register_if_enabled!("plan_list_steps", PlanListStepsTool::new(plan_list.clone()));
        }
        AgentMode::Planner => {
            register_if_enabled!(
                "plan_add_step",
                PlanAddStepTool::new_ungated(plan_list.clone())
            );
            register_if_enabled!("plan_list_steps", PlanListStepsTool::new(plan_list.clone()));
        }
        AgentMode::Subagent => {}
    }

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
    mode: AgentMode,
    discovered_skills: &[SkillMeta],
) -> (String, usize, usize, Vec<(String, usize)>) {
    let include_extensions = matches!(mode, AgentMode::Main);
    let subagent_desc = if include_extensions {
        "- subagent(task, files?): Spawn a sub-agent with a fresh context to handle a focused \
         task. The sub-agent cannot see this conversation, so include all necessary context in \
         the task description. STRONGLY PREFER passing `files` (an array of paths) for any \
         single-file edit — the harness pre-reads each file and prepends it to the task, so the \
         sub-agent starts with code in hand and skips warm-up turns. **For multi-file refactors, \
         delegate one sub-agent per file** rather than editing them all from this top-level \
         conversation. The smaller the sub-agent's context, the sharper the local model is on \
         the actual edit. Your top-level role is to plan, dispatch one sub-agent per file, and \
         verify with a final cargo check"
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

    // Plan tooling guidance varies by mode. Main agents work through a
    // pre-populated TodoList; planner sub-agents only populate it.
    let plan_section = match mode {
        AgentMode::Main => {
            "\n\n## Plan tracking\n\
             \n\
             A planner sub-agent has already produced a structured TodoList for \
             this task. The plan is visible to you via `plan_list_steps`.\n\
             \n\
             Work through the plan in order:\n\
             1. Call `plan_mark_in_progress` with the next step's index before you \
                begin acting on it.\n\
             2. Do the work (read, edit, run commands, etc.).\n\
             3. Call `plan_mark_done` only after you have *verified* the step is \
                complete (re-read the edited file, ran a check, etc.).\n\
             4. If a step does not apply (e.g. \"Run tests\" but the project has \
                none), call `plan_skip_step` with a reason — do not silently skip.\n\
             5. You may call `plan_add_step` if you discover a missing step.\n\
             \n\
             You cannot end your turn while pending or in-progress steps remain. \
             The harness will reject your final response and re-prompt you to \
             continue. Mark every step done or skipped before giving your final \
             answer.\n"
        }
        AgentMode::Planner => {
            "\n\n## Planning task\n\
             \n\
             You are the planning sub-agent. Your only job is to break the user's \
             request into concrete actionable steps for a separate execution \
             agent. You cannot edit files, run commands, or write code — only \
             explore (read/glob/grep) and call `plan_add_step` once per step.\n\
             \n\
             ### Pre-plan research (mandatory before any plan_add_step)\n\
             \n\
             You MUST complete all of the following before emitting any plan \
             step. Skipping any of these will produce a plan that misses \
             correctness pitfalls — the most common failure mode of small models \
             on multi-file refactors.\n\
             \n\
             1. **Read the request literally.** Identify every file/module/symbol \
                it names. Each named entity is in scope unless you can prove \
                otherwise.\n\
             2. **Map the affected code.** Use `glob`/`grep` to find all call \
                sites of the symbols being changed, not just their definitions. \
                A refactor that changes a struct's invariants but misses one \
                caller silently breaks the simulation.\n\
             3. **Locate existing behavioural tests.** Use `glob` to find test \
                files in the area being changed (`tests/*.test`, `tests/*.rs`, \
                `**/*_test.rs`, etc.), then `read` the most relevant ones. \
                Without knowing what behaviour is tested, you cannot plan steps \
                that preserve it.\n\
             4. **Read at least one consumer per affected interface.** If a \
                step changes a struct used by 5 systems, read at least one of \
                them so your plan reflects how the interface is actually used.\n\
             \n\
             ### Step format\n\
             \n\
             Each `plan_add_step` description must include three parts, separated \
             by `—`:\n\
             \n\
             `<concrete change> — invariant: <what must still hold> — risk: <what could go wrong>`\n\
             \n\
             Examples:\n\
             - `Add CombinedTerrainMap as a Bevy Resource via app.init_resource — invariant: physics tick must read from CombinedTerrainMap, not assemble per-tick — risk: forgetting init_resource panics at runtime on first world.resource() call`\n\
             - `Update apply_writes in terrain_view.rs:519 to also write each tile to CombinedTerrainMap — invariant: combined map and per-chunk maps must agree byte-for-byte after any tick — risk: writing only to chunks (current behaviour) leaves combined map with stale data, silently breaking spread/decay across tick boundaries`\n\
             - `Index combined map with stride = size_x * CHUNK_TILES, not CHUNK_TILES alone — invariant: combined.tile(x, y) returns same value as ChunkManager.get_terrain(x, y) for all valid (x, y) — risk: hardcoding stride to 64 corrupts every read past the first row of a multi-chunk grid`\n\
             \n\
             If a step has no risk worth naming, write `risk: none beyond the obvious`. But for any change touching coordinate math, indexing, sync between two stores, or resource registration, name a specific risk.\n\
             \n\
             ### Required final step\n\
             \n\
             The last plan step MUST be a behavioural verification step — not \
             just `cargo check`. It must run the existing integration tests in \
             the area being changed (e.g. `cargo xtask test`, or specific tests \
             like `cargo test --test fire-spread`). Compile-clean is necessary \
             but insufficient: a refactor that breaks coordinate math or sync \
             semantics still compiles. Tests catch what the type system cannot.\n\
             \n\
             ### Other rules\n\
             \n\
             - Aim for 5–12 steps. Too few hides work; too many fragments it.\n\
             - Order steps so each one's prerequisites are satisfied by earlier \
                steps. Adding a Resource must come before any step that reads it.\n\
             - When the same file is touched in multiple steps, group them \
                contiguously (one open of the file → multiple edits → one close).\n\
             - **Prefer per-file delegation steps** over bulk edits done from \
                the orchestrator. Phrase implementation steps as: \
                `Delegate <change> in <file> to a sub-agent (subagent tool, \
                files=[<path>]) — invariant: ... — risk: ...`. The execution \
                agent will then call `subagent(task=..., files=[<path>])` for \
                that step, keeping the orchestrator's context small. Reserve \
                non-delegated edits for trivial changes (one line or two).\n\
             - End your turn with no tool calls when the plan is complete.\n\
             \n\
             Do not write code. Do not produce a final summary — your output is \
             the TodoList itself.\n"
        }
        AgentMode::Subagent => "",
    };
    prompt.push_str(plan_section);
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
