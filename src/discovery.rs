//! Layered discovery for user-scoped and project-scoped assets.
//!
//! Skills and prompt templates share the same discovery pattern:
//! load from the user config dir, walk up from `cwd` looking for a
//! project-scoped `.agents/<subdir>`, then merge the two — project entries
//! override user entries with the same name.

use std::path::Path;

/// Any asset that can be looked up by name. Needed for the override-by-name
/// merge step. Both `SkillMeta` and `PromptTemplate` implement this.
pub trait Named {
    fn name(&self) -> &str;
}

/// Load user-scoped + project-scoped assets, merging by name.
///
/// * `user_dir` — absolute path to the user-scoped directory (e.g.
///   `~/.config/ollama-code/skills`).
/// * `project_subdir` — the subdirectory name under `.agents/` (e.g. `"skills"`).
/// * `cwd` — the working directory to start the project-walk from.
/// * `load` — per-directory loader. Called with a directory path; should return
///   the valid assets found there (empty `Vec` if the directory is missing or
///   contains nothing valid).
///
/// Returns a sorted `Vec<T>`; project entries override user entries by name.
pub fn discover_layered<T, L>(
    user_dir: &Path,
    project_subdir: &str,
    cwd: &str,
    load: L,
) -> Vec<T>
where
    T: Named,
    L: Fn(&Path) -> Vec<T>,
{
    let mut items = load(user_dir);

    let project_items = walk_up_for_project(cwd, project_subdir, &load);
    for p in project_items {
        if let Some(existing) = items.iter_mut().find(|i| i.name() == p.name()) {
            *existing = p;
        } else {
            items.push(p);
        }
    }

    items.sort_by(|a, b| a.name().cmp(b.name()));
    items
}

/// Walk up from `cwd` looking for `.agents/<subdir>`. Returns the first
/// non-empty loader result, or an empty `Vec` if none is found.
fn walk_up_for_project<T, L>(cwd: &str, subdir: &str, load: &L) -> Vec<T>
where
    L: Fn(&Path) -> Vec<T>,
{
    let mut dir = Path::new(cwd).to_path_buf();
    loop {
        let agents_dir = dir.join(".agents").join(subdir);
        let found = load(&agents_dir);
        if !found.is_empty() {
            return found;
        }
        if !dir.pop() {
            break;
        }
    }
    Vec::new()
}

/// Ensure each `(relative_path, contents)` entry exists under `base`.
/// `relative_path` may contain directory separators (e.g. `"self-modify/SKILL.md"`).
/// Existing files are never overwritten — this is an install-on-first-run helper.
/// Writes a `Warning:` line to stderr on I/O failure and keeps going.
pub fn install_defaults<P, I>(base: &Path, label: &str, entries: I)
where
    P: AsRef<Path>,
    I: IntoIterator<Item = (P, &'static str)>,
{
    if let Err(e) = std::fs::create_dir_all(base) {
        eprintln!("Warning: could not create {} dir {}: {}", label, base.display(), e);
        return;
    }
    for (relative, content) in entries {
        let path = base.join(relative.as_ref());
        if path.exists() {
            continue;
        }
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!(
                    "Warning: could not create {} dir {}: {}",
                    label,
                    parent.display(),
                    e
                );
                continue;
            }
        }
        if let Err(e) = std::fs::write(&path, content) {
            eprintln!(
                "Warning: could not write default {} {}: {}",
                label,
                path.display(),
                e
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[derive(Debug, PartialEq, Clone)]
    struct Item {
        name: String,
        origin: &'static str,
    }

    impl Named for Item {
        fn name(&self) -> &str {
            &self.name
        }
    }

    fn mkdir(p: &Path) {
        fs::create_dir_all(p).unwrap();
    }

    #[test]
    fn project_overrides_user() {
        let tmp = tempfile::tempdir().unwrap();
        let user = tmp.path().join("user");
        let cwd = tmp.path().join("proj");
        let proj = cwd.join(".agents").join("widgets");
        mkdir(&user);
        mkdir(&proj);
        fs::write(user.join("a.marker"), "a").unwrap();
        fs::write(user.join("b.marker"), "b").unwrap();
        fs::write(proj.join("a.marker"), "a").unwrap();

        let load = |d: &Path| -> Vec<Item> {
            if !d.is_dir() {
                return vec![];
            }
            fs::read_dir(d)
                .unwrap()
                .flatten()
                .map(|e| Item {
                    name: e.path().file_stem().unwrap().to_string_lossy().into_owned(),
                    origin: if e.path().starts_with(&user) { "user" } else { "proj" },
                })
                .collect()
        };

        let result = discover_layered(&user, "widgets", cwd.to_str().unwrap(), load);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "a");
        assert_eq!(result[0].origin, "proj");
        assert_eq!(result[1].name, "b");
        assert_eq!(result[1].origin, "user");
    }

    #[test]
    fn walks_up_for_project_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = tmp.path().join("proj");
        let nested = proj.join("a").join("b").join("c");
        let agents = proj.join(".agents").join("widgets");
        mkdir(&nested);
        mkdir(&agents);
        fs::write(agents.join("x.marker"), "x").unwrap();

        let load = |d: &Path| -> Vec<Item> {
            if !d.is_dir() {
                return vec![];
            }
            fs::read_dir(d)
                .unwrap()
                .flatten()
                .map(|e| Item {
                    name: e.path().file_stem().unwrap().to_string_lossy().into_owned(),
                    origin: "proj",
                })
                .collect()
        };

        let user = tmp.path().join("empty-user");
        mkdir(&user);
        let result = discover_layered(&user, "widgets", nested.to_str().unwrap(), load);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "x");
    }
}
