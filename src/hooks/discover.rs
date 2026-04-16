//! Loading `hooks.toml` files from user and project-scoped locations.

use std::path::{Path, PathBuf};

use super::types::HookEntry;

/// Load hooks from a `hooks.toml` file. Each top-level key is a hook name,
/// and the value is a table matching `HookEntry`.
pub(super) fn load_hooks_file(path: &Path) -> Vec<(String, HookEntry)> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let table: toml::map::Map<String, toml::Value> = match toml::from_str(&content) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    let mut hooks = Vec::new();
    for (name, value) in table {
        if let Ok(mut entry) = value.try_into::<HookEntry>() {
            entry.compile_patterns();
            hooks.push((name, entry));
        }
    }
    hooks
}

/// Default user hooks file: `~/.config/ollama-code/hooks.toml`.
pub(super) fn user_hooks_path() -> PathBuf {
    crate::config::config_dir().join("hooks.toml")
}

/// Walk up from `cwd` looking for `.agents/hooks.toml`.
pub(super) fn find_project_hooks(cwd: &str) -> Option<PathBuf> {
    let mut dir = Path::new(cwd).to_path_buf();
    loop {
        let candidate = dir.join(".agents").join("hooks.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}
