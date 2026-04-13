use std::time::Duration;

use anyhow::Result;

/// Wait for a child process with a timeout. Kills the process if it exceeds the deadline.
pub fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout: Duration,
    context: &str,
) -> Result<std::process::Output> {
    use std::time::Instant;

    let deadline = Instant::now() + timeout;
    let poll_interval = Duration::from_millis(50);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = Vec::new();
                let mut stderr = Vec::new();
                if let Some(mut out) = child.stdout.take() {
                    std::io::Read::read_to_end(&mut out, &mut stdout).ok();
                }
                if let Some(mut err) = child.stderr.take() {
                    std::io::Read::read_to_end(&mut err, &mut stderr).ok();
                }
                return Ok(std::process::Output {
                    status,
                    stdout,
                    stderr,
                });
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    anyhow::bail!("{} timed out after {}s", context, timeout.as_secs());
                }
                std::thread::sleep(poll_interval);
            }
            Err(e) => {
                anyhow::bail!("Error waiting for {}: {}", context, e);
            }
        }
    }
}
