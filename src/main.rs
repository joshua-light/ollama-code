use anyhow::Result;

use ollama_code::{cli, runner};

#[tokio::main]
async fn main() -> Result<()> {
    let (prompt, resume, config) = cli::resolve()?;
    runner::dispatch(prompt, resume, config).await
}
