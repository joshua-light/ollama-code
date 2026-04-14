use anyhow::Result;
use serde_json::Value;

use super::{format_bash_output, optional_str, Tool, ToolDefinition};

pub struct BashTool;

/// Resolve the path to a bash executable.
///
/// On Unix this simply returns `"bash"` and relies on `$PATH`.
///
/// On Windows, bare `bash` is unlikely to exist. We try, in order:
///   1. `bash.exe` on `$PATH` (works when Git Bash / WSL is already in PATH).
///   2. `$GIT_INSTALL_ROOT\bin\bash.exe` (set by some Git for Windows installers).
///   3. `$ProgramFiles\Git\bin\bash.exe` (default 64-bit install location).
///   4. `$ProgramFiles(x86)\Git\bin\bash.exe` (32-bit install on 64-bit Windows).
///   5. `$LOCALAPPDATA\Programs\Git\bin\bash.exe` (user-level install).
///
/// Returns the first path that exists, or falls back to `"bash"` so the
/// caller gets a clear "program not found" error.
fn resolve_bash_path() -> String {
    #[cfg(not(windows))]
    {
        "bash".to_string()
    }

    #[cfg(windows)]
    {
        use std::path::PathBuf;

        // Helper: check if a candidate path exists and return it.
        let try_path = |p: PathBuf| -> Option<String> {
            if p.is_file() {
                Some(p.to_string_lossy().into_owned())
            } else {
                None
            }
        };

        // 1. Check if `bash.exe` is already reachable via PATH.
        if let Ok(path_var) = std::env::var("PATH") {
            for dir in path_var.split(';') {
                let candidate = PathBuf::from(dir).join("bash.exe");
                if candidate.is_file() {
                    return candidate.to_string_lossy().into_owned();
                }
            }
        }

        // 2. $GIT_INSTALL_ROOT\bin\bash.exe
        if let Ok(root) = std::env::var("GIT_INSTALL_ROOT") {
            if let Some(p) = try_path(PathBuf::from(&root).join("bin").join("bash.exe")) {
                return p;
            }
        }

        // 3. $ProgramFiles\Git\bin\bash.exe  (typically C:\Program Files\Git)
        if let Ok(pf) = std::env::var("ProgramFiles") {
            if let Some(p) = try_path(PathBuf::from(&pf).join("Git").join("bin").join("bash.exe")) {
                return p;
            }
        }

        // 4. $ProgramFiles(x86)\Git\bin\bash.exe
        if let Ok(pf86) = std::env::var("ProgramFiles(x86)") {
            if let Some(p) =
                try_path(PathBuf::from(&pf86).join("Git").join("bin").join("bash.exe"))
            {
                return p;
            }
        }

        // 5. $LOCALAPPDATA\Programs\Git\bin\bash.exe
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            if let Some(p) = try_path(
                PathBuf::from(&local)
                    .join("Programs")
                    .join("Git")
                    .join("bin")
                    .join("bash.exe"),
            ) {
                return p;
            }
        }

        // Fallback — let the OS produce "program not found".
        "bash".to_string()
    }
}

impl BashTool {
    /// Async execution with timeout and kill-on-drop. This is the primary
    /// execution path — the sync `Tool::execute()` should never be called.
    pub async fn execute_async(
        &self,
        arguments: &Value,
        timeout: std::time::Duration,
    ) -> (String, bool) {
        let command = optional_str(arguments, "command").unwrap_or("");
        let bash = resolve_bash_path();
        match tokio::process::Command::new(&bash)
            .arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
        {
            Ok(child) => {
                match tokio::time::timeout(timeout, child.wait_with_output()).await {
                    Ok(Ok(output)) => format_bash_output(&output),
                    Ok(Err(e)) => (format!("Error: {}", e), false),
                    Err(_) => (
                        format!("Error: command timed out after {}s", timeout.as_secs()),
                        false,
                    ),
                }
            }
            Err(e) => (format!("Error spawning bash ({}): {}", bash, e), false),
        }
    }
}

impl Tool for BashTool {
    fn name(&self) -> &str { "bash" }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "bash".to_string(),
            description: "Execute a bash command and return its output. Use this for running \
                          shell commands, installing packages, running programs, git operations, \
                          and other terminal tasks. Do NOT use this to read or edit files — use \
                          the read and edit tools instead."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    }
                },
                "required": ["command"]
            }),
        }
    }

    fn execute(&self, _arguments: &Value) -> Result<String> {
        anyhow::bail!("BashTool must be executed via execute_async()")
    }
}
