//! Non-interactive pipe-mode event loop: subscribe to agent events, print
//! tokens/tool activity, and exit on `Done`/`Error`/`Cancelled`.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentEvent};
use crate::format;
use crate::session::Session;

pub async fn run_pipe(
    mut agent: Agent,
    prompt: &str,
    mut session: Session,
    verbose: bool,
) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let (confirm_tx, mut confirm_rx) = mpsc::unbounded_channel::<bool>();

    eprintln!("Session: {}", session.path().display());

    let prompt = prompt.to_string();
    let cancel = Arc::new(AtomicBool::new(false));
    let handle = tokio::spawn(async move {
        let (_steer_tx, mut steer_rx) = mpsc::unbounded_channel::<String>();
        agent
            .run(&prompt, &tx, &mut confirm_rx, &mut steer_rx, cancel)
            .await
    });

    // Buffer streamed tokens so ContentReplaced can discard/replace them.
    // This handles models that emit tool calls as text (e.g. <function=...>
    // format) — without buffering, the raw text leaks to stdout before
    // extraction can clean it up.
    let mut token_buf = String::new();
    let mut agent_error: Option<String> = None;

    while let Some(event) = rx.recv().await {
        session.log_agent_event(&event);
        if let AgentEvent::MessageLogged(ref msg) = event {
            session.log_message(msg);
        }
        match event {
            AgentEvent::Token(t) => {
                token_buf.push_str(&t);
            }
            AgentEvent::ContentReplaced(new_content) => {
                // The agent extracted tool calls from the buffered text.
                // Replace the buffer with the cleaned content.
                token_buf = new_content;
            }
            AgentEvent::ToolCall { name, args } => {
                // Flush any buffered content before showing tool info.
                if !token_buf.is_empty() {
                    print!("{}", token_buf);
                    token_buf.clear();
                }
                eprintln!(
                    "\n ● {}({})",
                    format::format_tool_name(&name),
                    format::truncate_args(&args, 77),
                );
            }
            AgentEvent::ToolConfirmRequest { .. } => {
                // Auto-approve in pipe mode
                let _ = confirm_tx.send(true);
            }
            AgentEvent::ToolResult { output, success, .. } => {
                if !success {
                    eprintln!("{}", format::format_tool_error(&output));
                } else {
                    for line in format::format_tool_output(&output) {
                        eprintln!("{}", line);
                    }
                }
            }
            AgentEvent::ContextTrimmed {
                removed_messages,
                estimated_tokens_freed,
            } => {
                session.record_trim(removed_messages);
                eprintln!(
                    "(context trimmed: {} messages, ~{} tokens freed)",
                    removed_messages, estimated_tokens_freed
                );
            }
            AgentEvent::ContextCompacting => {
                eprintln!("(compacting context...)");
            }
            AgentEvent::ContextCompacted {
                removed_messages,
                summary_tokens,
                estimated_tokens_freed,
            } => {
                session.record_trim(removed_messages);
                eprintln!(
                    "(context compacted: {} messages summarized, ~{} tokens freed, ~{} token summary)",
                    removed_messages, estimated_tokens_freed, summary_tokens
                );
            }
            AgentEvent::Done { .. } => {
                // Flush remaining buffer and finish.
                if !token_buf.is_empty() {
                    print!("{}", token_buf);
                    token_buf.clear();
                }
                println!();
                break;
            }
            AgentEvent::Error(e) => {
                agent_error = Some(e);
                break;
            }
            AgentEvent::SubagentStart { ref task } => {
                eprintln!("\n ◈ Subagent: {}", format::truncate_args(task, 77));
            }
            AgentEvent::SubagentToolCall { name, args } => {
                eprintln!(
                    "   ↳ {}({})",
                    format::format_tool_name(&name),
                    format::truncate_args(&args, 60),
                );
            }
            AgentEvent::SubagentToolResult { .. } => {}
            AgentEvent::SubagentEnd { .. } => {}
            AgentEvent::Cancelled => {
                eprintln!("\n(cancelled)");
                break;
            }
            AgentEvent::Debug(ref msg) if verbose => {
                eprintln!("[debug] {}", msg);
            }
            AgentEvent::ThinkingBudgetExceeded { thinking_tokens } => {
                eprintln!(
                    "[thinking budget exceeded ({} tokens) — thinking disabled, forcing commit]",
                    thinking_tokens
                );
            }
            AgentEvent::PlanningStarted => {
                eprintln!("\n ◧ Planning phase...");
            }
            AgentEvent::PlanReady { steps } => {
                eprintln!(" ◧ Plan ({} steps):", steps.len());
                for (i, step) in steps.iter().enumerate() {
                    eprintln!("   {}. {}", i, step);
                }
            }
            AgentEvent::PlanGated { remaining } => {
                eprintln!(
                    "(plan gate: {} step{} still pending — looping)",
                    remaining.len(),
                    if remaining.len() == 1 { "" } else { "s" }
                );
            }
            AgentEvent::ReloadComplete { .. }
            | AgentEvent::ContextUpdate { .. }
            | AgentEvent::MessageLogged(_)
            | AgentEvent::Debug(_)
            | AgentEvent::SystemPromptInfo { .. }
            | AgentEvent::ToolOutput { .. } => {}
        }
    }

    handle.await??;

    if let Some(e) = agent_error {
        anyhow::bail!("{}", e);
    }

    Ok(())
}
