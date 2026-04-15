use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::agent::AgentEvent;
use crate::config;
use crate::message::{Message, Role};

#[derive(Clone)]
struct TreeEntry {
    id: String,
    parent_id: Option<String>,
    message: Message,
}

pub struct TreeNode {
    pub id: String,
    pub role: Role,
    pub summary: String,
    pub children: Vec<TreeNode>,
}

pub struct Session {
    dir: PathBuf,
    messages_file: File,
    debug_file: File,
    trim_watermark: usize,
    entries: Vec<TreeEntry>,
    entry_index: HashMap<String, usize>,
    /// Tip of the active branch.
    leaf: Option<String>,
}

impl Session {
    pub fn new() -> anyhow::Result<Self> {
        let base = config::data_dir().join("sessions");

        let id = generate_session_id();
        let dir = base.join(&id);
        fs::create_dir_all(&dir)?;

        let messages_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("messages.jsonl"))?;

        let debug_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("debug.log"))?;

        let mut session = Self {
            dir,
            messages_file,
            debug_file,
            trim_watermark: 0,
            entries: Vec::new(),
            entry_index: HashMap::new(),
            leaf: None,
        };

        session.log_debug("SESSION_START");
        Ok(session)
    }

    /// Resume an existing session by ID, returning the session handle and
    /// the messages on the current branch path (ready for agent restore).
    pub fn resume(id: &str) -> anyhow::Result<(Self, Vec<Message>)> {
        let base = config::data_dir().join("sessions");
        let dir = base.join(id);

        if !dir.exists() {
            anyhow::bail!("Session not found: {}", id);
        }

        let loaded = load_tree_entries(&dir)?;

        let mut messages = build_branch_path(&loaded.entries, &loaded.entry_index, loaded.leaf.as_deref());
        apply_trim_watermark(&mut messages, loaded.trim_watermark);

        let messages_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("messages.jsonl"))?;

        let debug_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("debug.log"))?;

        let mut session = Self {
            dir,
            messages_file,
            debug_file,
            trim_watermark: loaded.trim_watermark,
            entries: loaded.entries,
            entry_index: loaded.entry_index,
            leaf: loaded.leaf,
        };

        session.log_debug("SESSION_RESUME");
        Ok((session, messages))
    }

    /// Load messages from a session by ID without opening file handles.
    /// Returns messages on the current branch path.
    pub fn load_messages(id: &str) -> anyhow::Result<Vec<Message>> {
        let base = config::data_dir().join("sessions");
        let dir = base.join(id);

        if !dir.exists() {
            anyhow::bail!("Session not found: {}", id);
        }

        let loaded = load_tree_entries(&dir)?;
        let mut messages = build_branch_path(&loaded.entries, &loaded.entry_index, loaded.leaf.as_deref());
        apply_trim_watermark(&mut messages, loaded.trim_watermark);
        Ok(messages)
    }

    /// List recent session IDs, sorted most-recent first.
    pub fn list_recent(limit: usize) -> anyhow::Result<Vec<String>> {
        let base = config::data_dir().join("sessions");
        if !base.exists() {
            return Ok(Vec::new());
        }

        let mut entries: Vec<String> = fs::read_dir(&base)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                if entry.file_type().ok()?.is_dir() {
                    Some(entry.file_name().to_string_lossy().to_string())
                } else {
                    None
                }
            })
            .collect();

        entries.sort_unstable();
        entries.reverse();
        entries.truncate(limit);
        Ok(entries)
    }

    /// Return the most recent session ID, if any.
    pub fn latest() -> anyhow::Result<Option<String>> {
        Ok(Self::list_recent(1)?.into_iter().next())
    }

    /// Find a session ID by prefix match.
    pub fn find_by_prefix(prefix: &str) -> anyhow::Result<Option<String>> {
        let base = config::data_dir().join("sessions");
        if !base.exists() {
            return Ok(None);
        }

        let mut matches: Vec<String> = fs::read_dir(&base)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                if entry.file_type().ok()?.is_dir() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with(prefix) {
                        return Some(name);
                    }
                }
                None
            })
            .collect();

        matches.sort_unstable();
        // Return the most recent match (last alphabetically)
        Ok(matches.pop())
    }

    pub fn id(&self) -> &str {
        self.dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
    }

    pub fn path(&self) -> &Path {
        &self.dir
    }

    /// Return the current leaf entry ID.
    pub fn leaf_id(&self) -> Option<&str> {
        self.leaf.as_deref()
    }

    /// Rewind the tree leaf by `n` user turns on the current branch path.
    /// Walks backward from the leaf, counting User messages, and sets the
    /// leaf to the entry just before the nth User message.
    pub fn rewind_leaf(&mut self, n: usize) {
        let path = self.branch_path_indices();
        // Find positions of user messages in the path
        let user_positions: Vec<usize> = path
            .iter()
            .enumerate()
            .filter(|(_, &idx)| matches!(self.entries[idx].message.role, Role::User))
            .map(|(i, _)| i)
            .collect();

        if user_positions.is_empty() || n == 0 {
            return;
        }

        let actual_n = n.min(user_positions.len());
        let truncate_at = user_positions[user_positions.len() - actual_n];

        if truncate_at > 0 {
            self.leaf = Some(self.entries[path[truncate_at - 1]].id.clone());
        } else {
            self.leaf = None;
        }
        self.persist_meta();
    }

    /// Get indices of entries on the current branch path (root to leaf order).
    fn branch_path_indices(&self) -> Vec<usize> {
        walk_path_indices(&self.entries, &self.entry_index, self.leaf.as_deref())
    }

    /// Move the leaf pointer to an earlier entry, creating a branch point.
    /// The next `log_message` call will create a child of this entry.
    pub fn branch(&mut self, entry_id: &str) -> anyhow::Result<()> {
        if !self.entry_index.contains_key(entry_id) {
            anyhow::bail!("Entry not found: {}", entry_id);
        }
        self.leaf = Some(entry_id.to_string());
        self.persist_meta();
        Ok(())
    }

    /// Get messages on the current branch path (leaf to root, reversed).
    pub fn get_branch_path(&self) -> Vec<Message> {
        build_branch_path(&self.entries, &self.entry_index, self.leaf.as_deref())
    }

    /// Return the set of entry IDs on the current branch path.
    pub fn active_path_ids(&self) -> std::collections::HashSet<String> {
        self.branch_path_indices()
            .iter()
            .map(|&idx| self.entries[idx].id.clone())
            .collect()
    }

    /// Build a tree structure for the /tree browser.
    /// Returns root nodes (usually one, but could be multiple for edge cases).
    pub fn get_tree(&self) -> Vec<TreeNode> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        // Group children by parent_id
        let mut children_map: HashMap<Option<&str>, Vec<usize>> = HashMap::new();
        for (i, entry) in self.entries.iter().enumerate() {
            children_map
                .entry(entry.parent_id.as_deref())
                .or_default()
                .push(i);
        }

        // Build tree recursively from roots (entries with no parent)
        fn build_node(
            entries: &[TreeEntry],
            children_map: &HashMap<Option<&str>, Vec<usize>>,
            idx: usize,
        ) -> TreeNode {
            let entry = &entries[idx];
            let summary = make_summary(&entry.message);
            let children = children_map
                .get(&Some(entry.id.as_str()))
                .map(|child_indices| {
                    child_indices
                        .iter()
                        .map(|&ci| build_node(entries, children_map, ci))
                        .collect()
                })
                .unwrap_or_default();

            TreeNode {
                id: entry.id.clone(),
                role: entry.message.role.clone(),
                summary,
                children,
            }
        }

        let root_indices = children_map.get(&None).cloned().unwrap_or_default();
        root_indices
            .iter()
            .map(|&i| build_node(&self.entries, &children_map, i))
            .collect()
    }

    /// Record that a context trim removed `removed` messages.
    /// Persists the cumulative watermark so resumed sessions skip trimmed messages.
    pub fn record_trim(&mut self, removed: usize) {
        self.trim_watermark += removed;
        self.persist_meta();
    }

    pub fn log_message(&mut self, msg: &Message) {
        let id = generate_entry_id();
        let parent_id = self.leaf.clone();
        let ts = format_utc(SystemTime::now());

        if let Ok(mut value) = serde_json::to_value(msg) {
            if let Some(obj) = value.as_object_mut() {
                obj.insert("id".to_string(), serde_json::Value::String(id.clone()));
                match &parent_id {
                    Some(pid) => {
                        obj.insert("parent_id".to_string(), serde_json::Value::String(pid.clone()));
                    }
                    None => {
                        obj.insert("parent_id".to_string(), serde_json::Value::Null);
                    }
                }
                obj.insert("ts".to_string(), serde_json::Value::String(ts));
            }
            if let Ok(line) = serde_json::to_string(&value) {
                let _ = writeln!(self.messages_file, "{}", line);
                let _ = self.messages_file.flush();
            }
        }

        let entry = TreeEntry {
            id: id.clone(),
            parent_id,
            message: msg.clone(),
        };
        self.entry_index.insert(id.clone(), self.entries.len());
        self.entries.push(entry);
        self.leaf = Some(id);
    }

    pub fn log_debug(&mut self, event: &str) {
        let ts = format_utc(SystemTime::now());
        let _ = writeln!(self.debug_file, "[{}] {}", ts, event);
        let _ = self.debug_file.flush();
    }

    pub fn log_agent_event(&mut self, event: &AgentEvent) {
        let line = match event {
            AgentEvent::Token(t) => format!("TOKEN ({} chars)", t.len()),
            AgentEvent::ToolCall { name, args } => format!("TOOL_CALL {}({})", name, args),
            AgentEvent::ToolResult {
                name,
                output,
                success,
            } => {
                format!(
                    "TOOL_RESULT {} success={} ({} chars)",
                    name,
                    success,
                    output.len()
                )
            }
            AgentEvent::ContextUpdate { prompt_tokens } => {
                format!("CONTEXT_UPDATE prompt_tokens={}", prompt_tokens)
            }
            AgentEvent::Done { prompt_tokens, .. } => {
                format!("DONE prompt_tokens={}", prompt_tokens)
            }
            AgentEvent::Error(e) => format!("ERROR {}", e),
            AgentEvent::ContentReplaced(_) => {
                "CONTENT_REPLACED (tool calls extracted from text)".to_string()
            }
            AgentEvent::ToolConfirmRequest { name, args } => {
                format!("TOOL_CONFIRM_REQUEST {}({})", name, args)
            }
            AgentEvent::ContextTrimmed {
                removed_messages,
                estimated_tokens_freed,
            } => {
                format!(
                    "CONTEXT_TRIMMED removed={} freed_est={}",
                    removed_messages, estimated_tokens_freed
                )
            }
            AgentEvent::ContextCompacting => "CONTEXT_COMPACTING".to_string(),
            AgentEvent::ContextCompacted {
                removed_messages,
                summary_tokens,
                estimated_tokens_freed,
            } => {
                format!(
                    "CONTEXT_COMPACTED removed={} freed_est={} summary={}",
                    removed_messages, estimated_tokens_freed, summary_tokens
                )
            }
            AgentEvent::SubagentStart { task } => {
                format!("SUBAGENT_START task={}", task)
            }
            AgentEvent::SubagentToolCall { name, args } => {
                format!("SUBAGENT_TOOL_CALL {}({})", name, args)
            }
            AgentEvent::SubagentToolResult { name, success } => {
                format!("SUBAGENT_TOOL_RESULT {} success={}", name, success)
            }
            AgentEvent::SubagentEnd { result } => {
                format!("SUBAGENT_END ({} chars)", result.len())
            }
            AgentEvent::Cancelled => "CANCELLED".to_string(),
            AgentEvent::MessageLogged(_) => return, // handled separately via log_message
            AgentEvent::Debug(s) => format!("DEBUG {}", s),
            AgentEvent::SystemPromptInfo { base_prompt_tokens, ref project_docs, skills_tokens, ref tool_defs_breakdown } => {
                format!("SYSTEM_PROMPT_INFO base={} docs={:?} skills={} tools={:?}", base_prompt_tokens, project_docs, skills_tokens, tool_defs_breakdown)
            }
            AgentEvent::ReloadComplete { ref summary, .. } => {
                format!("RELOAD_COMPLETE {}", summary)
            }
        };
        self.log_debug(&line);
    }

    fn persist_meta(&self) {
        let mut meta = serde_json::json!({"trim_watermark": self.trim_watermark});
        if let Some(ref leaf) = self.leaf {
            meta["leaf"] = serde_json::Value::String(leaf.clone());
        }
        let _ = fs::write(self.dir.join("meta.json"), meta.to_string());
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.persist_meta();
        self.log_debug("SESSION_END");
        let _ = self.messages_file.flush();
        let _ = self.debug_file.flush();
    }
}

fn generate_session_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let nanos = now.subsec_nanos();
    let pid = std::process::id();

    // Simple hash for uniqueness suffix
    let hash = (pid as u64).wrapping_mul(2654435761) ^ (nanos as u64);
    let suffix = format!("{:06x}", hash & 0xFFFFFF);

    let (y, mon, d, h, m, s) = epoch_to_utc(secs);
    format!("{:04}-{:02}-{:02}_{:02}{:02}{:02}_{}", y, mon, d, h, m, s, suffix)
}

/// Generate a unique 8-char hex ID for a tree entry.
fn generate_entry_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let nanos = now.subsec_nanos();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let hash = (std::process::id() as u64)
        .wrapping_mul(2654435761)
        ^ (nanos as u64)
        ^ seq.wrapping_mul(6364136223846793005);
    format!("{:08x}", hash & 0xFFFFFFFF)
}

/// Format a SystemTime as an ISO 8601-ish UTC string.
fn format_utc(time: SystemTime) -> String {
    let dur = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = dur.as_secs();
    let millis = dur.subsec_millis();
    let (y, mon, d, h, m, s) = epoch_to_utc(secs);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        y, mon, d, h, m, s, millis
    )
}

/// Convert epoch seconds to (year, month, day, hour, minute, second) in UTC.
/// Uses Howard Hinnant's civil_from_days algorithm.
fn epoch_to_utc(epoch_secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (epoch_secs / 86400) as i64;
    let day_secs = (epoch_secs % 86400) as u32;
    let h = day_secs / 3600;
    let m = (day_secs % 3600) / 60;
    let s = day_secs % 60;

    // Civil date from days since epoch (Hinnant algorithm)
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mon = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mon <= 2 { y + 1 } else { y };

    (y, mon, d, h, m, s)
}

/// Create a short summary from a message for the tree browser.
fn make_summary(msg: &Message) -> String {
    let content = msg.content.trim();
    if content.is_empty() {
        if let Some(ref calls) = msg.tool_calls {
            if let Some(first) = calls.first() {
                return format!("{}()", first.function.name);
            }
        }
        return "(empty)".to_string();
    }
    let first_line = content.lines().next().unwrap_or(content);
    crate::format::truncate_args(first_line, 60)
}

/// Apply trim watermark to a message list: skip N non-system messages from the front.
fn apply_trim_watermark(messages: &mut Vec<Message>, trim_watermark: usize) {
    if trim_watermark > 0 && messages.len() > 1 {
        // Find the first non-system message
        let first_non_system = messages
            .iter()
            .position(|m| !matches!(m.role, Role::System))
            .unwrap_or(messages.len());
        let skip = trim_watermark.min(messages.len() - first_non_system);
        if skip > 0 {
            messages.drain(first_non_system..first_non_system + skip);
        }
    }
}

/// Walk from a leaf entry to the root, returning entry indices in root-to-leaf order.
fn walk_path_indices(
    entries: &[TreeEntry],
    entry_index: &HashMap<String, usize>,
    leaf: Option<&str>,
) -> Vec<usize> {
    let leaf_id = match leaf {
        Some(id) => id,
        None => {
            // No leaf set — return all entries linearly (empty or pre-branch session)
            return (0..entries.len()).collect();
        }
    };

    let mut path = Vec::new();
    let mut current: Option<&str> = Some(leaf_id);

    while let Some(id) = current {
        if let Some(&idx) = entry_index.get(id) {
            path.push(idx);
            current = entries[idx].parent_id.as_deref();
        } else {
            break;
        }
    }

    path.reverse();
    path
}

/// Walk from a leaf entry to the root, collecting messages in root-to-leaf order.
fn build_branch_path(
    entries: &[TreeEntry],
    entry_index: &HashMap<String, usize>,
    leaf: Option<&str>,
) -> Vec<Message> {
    walk_path_indices(entries, entry_index, leaf)
        .iter()
        .map(|&idx| entries[idx].message.clone())
        .collect()
}

/// Result of loading tree entries from a session directory.
struct LoadedTree {
    entries: Vec<TreeEntry>,
    entry_index: HashMap<String, usize>,
    leaf: Option<String>,
    trim_watermark: usize,
}

/// Load tree entries from a session directory.
/// Handles backward compatibility: old entries without id/parent_id get synthetic IDs.
fn load_tree_entries(dir: &Path) -> anyhow::Result<LoadedTree> {
    let messages_path = dir.join("messages.jsonl");
    let mut entries = Vec::new();
    let mut entry_index = HashMap::new();
    let mut prev_id: Option<String> = None;
    if messages_path.exists() {
        let file = File::open(&messages_path)?;
        let reader = std::io::BufReader::new(file);
        for (i, line) in reader.lines().enumerate() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    eprintln!(
                        "Warning: could not read line {} of messages.jsonl: {}",
                        i + 1,
                        e
                    );
                    continue;
                }
            };
            if line.trim().is_empty() {
                continue;
            }

            let value: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "Warning: could not parse line {} of messages.jsonl: {}",
                        i + 1,
                        e
                    );
                    continue;
                }
            };

            // Extract tree fields before consuming value for Message
            let explicit_id = value
                .get("id")
                .and_then(|v| v.as_str())
                .map(String::from);
            let id = explicit_id.unwrap_or_else(|| format!("{:08x}", i));
            let parent_id = value
                .get("parent_id")
                .and_then(|v| {
                    if v.is_null() {
                        None
                    } else {
                        v.as_str().map(String::from)
                    }
                })
                .or_else(|| prev_id.clone());

            let msg: Message = match serde_json::from_value(value) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!(
                        "Warning: could not deserialize message on line {}: {}",
                        i + 1,
                        e
                    );
                    continue;
                }
            };

            entry_index.insert(id.clone(), entries.len());
            entries.push(TreeEntry {
                id: id.clone(),
                parent_id,
                message: msg,
            });
            prev_id = Some(id);
        }
    }

    // Read leaf and trim watermark from meta.json
    let meta = read_meta(dir);
    let leaf = meta.leaf
        .or_else(|| entries.last().map(|e| e.id.clone()));

    Ok(LoadedTree { entries, entry_index, leaf, trim_watermark: meta.trim_watermark })
}

struct SessionMeta {
    leaf: Option<String>,
    trim_watermark: usize,
}

/// Read leaf pointer and trim watermark from a session's meta.json.
/// Returns defaults if the file is missing or malformed (backward-compatible).
fn read_meta(session_dir: &Path) -> SessionMeta {
    let meta_path = session_dir.join("meta.json");
    let value = fs::read_to_string(&meta_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());

    let leaf = value
        .as_ref()
        .and_then(|v| v.get("leaf")?.as_str().map(String::from));
    let trim_watermark = value
        .as_ref()
        .and_then(|v| v.get("trim_watermark")?.as_u64())
        .unwrap_or(0) as usize;

    SessionMeta { leaf, trim_watermark }
}
