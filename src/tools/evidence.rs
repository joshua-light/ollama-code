use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::{optional_str, required_str, Tool, ToolDefinition};

/// Shared, per-session evidence store: short text snippets keyed by short
/// identifiers. Lives on the `Agent` so it survives context compaction — only
/// the raw in-memory map is preserved, not messages referencing it.
pub type EvidenceStore = Arc<Mutex<HashMap<String, String>>>;

pub fn new_evidence_store() -> EvidenceStore {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Cap on a single stored snippet; anything longer is truncated with an
/// explicit suffix so the model doesn't silently lose content.
const MAX_SNIPPET_LEN: usize = 1024;

pub struct EvidenceAddTool {
    store: EvidenceStore,
}

pub struct EvidenceGetTool {
    store: EvidenceStore,
}

pub struct EvidenceListTool {
    store: EvidenceStore,
}

impl EvidenceAddTool {
    pub fn new(store: EvidenceStore) -> Self {
        Self { store }
    }
}

impl EvidenceGetTool {
    pub fn new(store: EvidenceStore) -> Self {
        Self { store }
    }
}

impl EvidenceListTool {
    pub fn new(store: EvidenceStore) -> Self {
        Self { store }
    }
}

impl Tool for EvidenceAddTool {
    fn name(&self) -> &str {
        "evidence_add"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "evidence_add".to_string(),
            description: "Stash a short text snippet under a key you can recall later with \
                          evidence_get. Use this for findings worth remembering across many \
                          turns (file locations, flag values, API shapes) — it survives context \
                          compaction. Overwrites any existing entry with the same key. Snippets \
                          longer than 1 KB are truncated."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short identifier for the snippet (e.g. 'auth-flow', 'bug-repro')."
                    },
                    "content": {
                        "type": "string",
                        "description": "The snippet to store. Keep under 1 KB."
                    }
                },
                "required": ["key", "content"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let key = required_str(arguments, "key")?.to_string();
        let content_raw = required_str(arguments, "content")?;

        let content = if content_raw.len() > MAX_SNIPPET_LEN {
            let mut truncated = content_raw
                .char_indices()
                .take_while(|(i, _)| *i < MAX_SNIPPET_LEN)
                .map(|(_, c)| c)
                .collect::<String>();
            truncated.push_str("…[truncated]");
            truncated
        } else {
            content_raw.to_string()
        };

        let mut map = self
            .store
            .lock()
            .map_err(|e| anyhow::anyhow!("evidence store lock poisoned: {}", e))?;
        let existed = map.insert(key.clone(), content).is_some();
        let verb = if existed { "Updated" } else { "Stored" };
        Ok(format!(
            "{} evidence under '{}' ({} entries total)",
            verb,
            key,
            map.len()
        ))
    }
}

impl Tool for EvidenceGetTool {
    fn name(&self) -> &str {
        "evidence_get"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "evidence_get".to_string(),
            description: "Retrieve a previously-stored evidence snippet by key. Returns the \
                          full snippet content, or an error if no entry exists under that key."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key the snippet was stored under."
                    }
                },
                "required": ["key"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let key = required_str(arguments, "key")?;
        let map = self
            .store
            .lock()
            .map_err(|e| anyhow::anyhow!("evidence store lock poisoned: {}", e))?;
        match map.get(key) {
            Some(content) => Ok(content.clone()),
            None => anyhow::bail!("No evidence found under key '{}'", key),
        }
    }
}

impl Tool for EvidenceListTool {
    fn name(&self) -> &str {
        "evidence_list"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "evidence_list".to_string(),
            description: "List all evidence keys currently in the store, each with a short \
                          preview of its content. Optionally filter by prefix."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "prefix": {
                        "type": "string",
                        "description": "If set, only keys starting with this prefix are listed."
                    }
                }
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let prefix = optional_str(arguments, "prefix").unwrap_or("");
        let map = self
            .store
            .lock()
            .map_err(|e| anyhow::anyhow!("evidence store lock poisoned: {}", e))?;
        let mut entries: Vec<(&String, &String)> = map
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        if entries.is_empty() {
            return Ok(if prefix.is_empty() {
                "Evidence store is empty.".to_string()
            } else {
                format!("No evidence keys start with '{}'.", prefix)
            });
        }

        let mut out = format!("{} entries:\n", entries.len());
        for (k, v) in entries {
            let preview: String = v.chars().take(80).collect();
            let suffix = if v.chars().count() > 80 { "…" } else { "" };
            out.push_str(&format!("  {} — {}{}\n", k, preview, suffix));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn add_and_get_roundtrip() {
        let store = new_evidence_store();
        let adder = EvidenceAddTool::new(store.clone());
        let getter = EvidenceGetTool::new(store.clone());

        let out = adder
            .execute(&json!({"key": "foo", "content": "bar baz"}))
            .unwrap();
        assert!(out.contains("Stored"));

        let fetched = getter.execute(&json!({"key": "foo"})).unwrap();
        assert_eq!(fetched, "bar baz");
    }

    #[test]
    fn add_overwrites() {
        let store = new_evidence_store();
        let adder = EvidenceAddTool::new(store.clone());

        adder
            .execute(&json!({"key": "k", "content": "first"}))
            .unwrap();
        let out = adder
            .execute(&json!({"key": "k", "content": "second"}))
            .unwrap();
        assert!(out.contains("Updated"));
    }

    #[test]
    fn get_missing_key_errors() {
        let store = new_evidence_store();
        let getter = EvidenceGetTool::new(store);
        let err = getter.execute(&json!({"key": "nope"})).unwrap_err();
        assert!(err.to_string().contains("No evidence"));
    }

    #[test]
    fn list_empty_store() {
        let store = new_evidence_store();
        let lister = EvidenceListTool::new(store);
        let out = lister.execute(&json!({})).unwrap();
        assert!(out.contains("empty"));
    }

    #[test]
    fn list_with_prefix_filter() {
        let store = new_evidence_store();
        let adder = EvidenceAddTool::new(store.clone());
        adder.execute(&json!({"key": "api-v1", "content": "a"})).unwrap();
        adder.execute(&json!({"key": "api-v2", "content": "b"})).unwrap();
        adder.execute(&json!({"key": "db-pool", "content": "c"})).unwrap();

        let lister = EvidenceListTool::new(store);
        let out = lister.execute(&json!({"prefix": "api-"})).unwrap();
        assert!(out.contains("api-v1"));
        assert!(out.contains("api-v2"));
        assert!(!out.contains("db-pool"));
    }

    #[test]
    fn long_content_truncated() {
        let store = new_evidence_store();
        let adder = EvidenceAddTool::new(store.clone());
        let getter = EvidenceGetTool::new(store);

        let big = "x".repeat(2048);
        adder.execute(&json!({"key": "big", "content": big})).unwrap();
        let fetched = getter.execute(&json!({"key": "big"})).unwrap();
        assert!(fetched.ends_with("…[truncated]"));
        assert!(fetched.len() < 2048);
    }
}
