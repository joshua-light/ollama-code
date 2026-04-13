use std::time::Duration;

use anyhow::Result;

/// Wait for a child process with a timeout. Kills the process if it exceeds the deadline.
///
/// Stdout and stderr pipes are drained concurrently in separate threads to
/// prevent deadlocks when the child produces more output than the OS pipe
/// buffer (~64 KB on Linux). On timeout, SIGTERM is sent first with a 2-second
/// grace period before falling back to SIGKILL.
pub fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout: Duration,
    context: &str,
) -> Result<std::process::Output> {
    use std::time::Instant;

    let deadline = Instant::now() + timeout;
    let poll_interval = Duration::from_millis(50);

    let stdout_thread = spawn_drain(child.stdout.take());
    let stderr_thread = spawn_drain(child.stderr.take());

    // Poll for exit while reader threads drain the pipes.
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    graceful_kill(child, poll_interval);
                    anyhow::bail!("{} timed out after {}s", context, timeout.as_secs());
                }
                std::thread::sleep(poll_interval);
            }
            Err(e) => {
                anyhow::bail!("Error waiting for {}: {}", context, e);
            }
        }
    };

    let stdout = stdout_thread.join().unwrap_or_default();
    let stderr = stderr_thread.join().unwrap_or_default();

    Ok(std::process::Output {
        status,
        stdout,
        stderr,
    })
}

fn spawn_drain(pipe: Option<impl std::io::Read + Send + 'static>) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut r) = pipe {
            let _ = std::io::Read::read_to_end(&mut r, &mut buf);
        }
        buf
    })
}

/// Send SIGTERM, wait up to 2 seconds for exit, then SIGKILL.
fn graceful_kill(child: &mut std::process::Child, poll_interval: Duration) {
    use std::time::Instant;

    #[cfg(unix)]
    {
        // SAFETY: child.id() is a valid PID for a process we spawned and
        // haven't yet reaped (try_wait returned None).
        unsafe {
            libc::kill(child.id() as libc::pid_t, libc::SIGTERM);
        }
        let grace_deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Ok(Some(_)) = child.try_wait() {
                return;
            }
            if Instant::now() >= grace_deadline {
                break;
            }
            std::thread::sleep(poll_interval);
        }
    }

    let _ = child.kill();
    let _ = child.wait();
}
