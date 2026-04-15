use std::ffi::OsStr;
use std::sync::OnceLock;

use anyhow::Result;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};

use super::{optional_str, Tool, ToolDefinition};

pub struct BashTool;

/// Resolve the path to a bash executable, cached for the lifetime of the process.
///
/// On Unix this simply returns `"bash"` and relies on `$PATH`.
///
/// On Windows, `Command::new("bash")` already searches `$PATH`, so we use
/// that as the default. If well-known Git Bash installation directories
/// contain `bash.exe`, we return the first match instead — this covers the
/// common case where Git for Windows is installed but its `bin` directory
/// is not on `$PATH`.
///
/// Probed locations (Windows only):
///   1. `$GIT_INSTALL_ROOT\bin\bash.exe`
///   2. `$ProgramFiles\Git\bin\bash.exe`
///   3. `$ProgramFiles(x86)\Git\bin\bash.exe`
///   4. `$LOCALAPPDATA\Programs\Git\bin\bash.exe`
///
/// Returns a reference to a cached `OsStr` — either one of the probed
/// paths or the bare string `"bash"` so the caller gets a clear
/// "program not found" error from the OS.
fn resolve_bash_path() -> &'static OsStr {
    static BASH_PATH: OnceLock<std::ffi::OsString> = OnceLock::new();
    BASH_PATH
        .get_or_init(|| {
            #[cfg(not(windows))]
            {
                std::ffi::OsString::from("bash")
            }

            #[cfg(windows)]
            {
                use std::path::PathBuf;

                let try_path = |p: PathBuf| -> Option<std::ffi::OsString> {
                    if p.is_file() {
                        Some(p.into_os_string())
                    } else {
                        None
                    }
                };

                // 1. $GIT_INSTALL_ROOT\bin\bash.exe
                if let Ok(root) = std::env::var("GIT_INSTALL_ROOT") {
                    if let Some(p) = try_path(PathBuf::from(&root).join("bin").join("bash.exe")) {
                        return p;
                    }
                }

                // 2. $ProgramFiles\Git\bin\bash.exe  (typically C:\Program Files\Git)
                if let Ok(pf) = std::env::var("ProgramFiles") {
                    if let Some(p) =
                        try_path(PathBuf::from(&pf).join("Git").join("bin").join("bash.exe"))
                    {
                        return p;
                    }
                }

                // 3. $ProgramFiles(x86)\Git\bin\bash.exe
                if let Ok(pf86) = std::env::var("ProgramFiles(x86)") {
                    if let Some(p) =
                        try_path(PathBuf::from(&pf86).join("Git").join("bin").join("bash.exe"))
                    {
                        return p;
                    }
                }

                // 4. $LOCALAPPDATA\Programs\Git\bin\bash.exe
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

                // Fallback — let the OS search PATH and produce "program not found"
                // if bash is not available at all.
                std::ffi::OsString::from("bash")
            }
        })
        .as_os_str()
}

/// Read all lines from an async reader, accumulating into a String.
/// If `tx` is provided, each line is also forwarded for live display.
async fn drain_lines<R: tokio::io::AsyncRead + Unpin>(
    reader: Option<R>,
    tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
) -> String {
    let mut output = String::new();
    let Some(r) = reader else { return output };
    let mut reader = BufReader::new(r);
    let mut buf = String::new();
    loop {
        buf.clear();
        match reader.read_line(&mut buf).await {
            Ok(0) => break,
            Ok(_) => {
                if let Some(ref tx) = tx {
                    let _ = tx.send(buf.clone());
                }
                output.push_str(&buf);
            }
            Err(_) => break,
        }
    }
    output
}

impl BashTool {
    /// Async execution with timeout and kill-on-drop. This is the primary
    /// execution path — the sync `Tool::execute()` should never be called.
    ///
    /// If `output_tx` is provided, partial output lines are sent as they arrive
    /// so the TUI can show live progress while the command runs.
    pub async fn execute_async(
        &self,
        arguments: &Value,
        timeout: std::time::Duration,
        output_tx: Option<&tokio::sync::mpsc::UnboundedSender<String>>,
    ) -> (String, bool) {
        let command = optional_str(arguments, "command").unwrap_or("");
        let bash = resolve_bash_path();
        match tokio::process::Command::new(bash)
            .arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
        {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();

                let read_lines = async {
                    let tx = output_tx.cloned();
                    let stdout_task = drain_lines(stdout, tx);
                    let stderr_task = drain_lines(stderr, None);

                    let (stdout_output, stderr_output) = tokio::join!(stdout_task, stderr_task);
                    let status = child.wait().await;

                    match status {
                        Ok(exit) => {
                            let mut result = String::new();
                            if !stdout_output.is_empty() {
                                result.push_str(&stdout_output);
                            }
                            if !stderr_output.is_empty() {
                                if !result.is_empty() {
                                    result.push('\n');
                                }
                                result.push_str(&stderr_output);
                            }
                            if result.is_empty() {
                                result.push_str("(no output)");
                            }
                            let success = exit.success();
                            if !success {
                                let code = exit.code().unwrap_or(-1);
                                result.push_str(&format!(
                                    "\n(exit code: {})",
                                    code
                                ));
                            }
                            (result, success)
                        }
                        Err(e) => (format!("Error: {}", e), false),
                    }
                };

                match tokio::time::timeout(timeout, read_lines).await {
                    Ok(output) => output,
                    Err(_) => (
                        format!("Error: command timed out after {}s", timeout.as_secs()),
                        false,
                    ),
                }
            }
            Err(e) => (
                format!("Error spawning bash ({}): {}", bash.to_string_lossy(), e),
                false,
            ),
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
