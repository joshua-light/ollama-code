use crate::agent::AgentEvent;

use super::super::app::{App, ChatMessage, PendingConfirm, ToolResultData};
use super::super::render::{get_git_info_sync, pick_verb};

pub(in crate::tui) fn handle_agent_event(event: AgentEvent, app: &mut App) {
    let was_at_bottom = app.is_at_bottom();

    match event {
        AgentEvent::Token(t) => {
            app.current_response.push_str(&t);
            app.generation.tokens += 1;
            app.generation.has_received_tokens = true;
        }
        AgentEvent::ContentReplaced(content) => {
            // Tool calls were extracted from streamed text — replace the
            // raw JSON the user saw with the cleaned content.
            app.current_response = content;
        }
        AgentEvent::ToolCall { name, args } => {
            app.flush_streaming();
            app.stats.tool_call_count += 1;
            *app.stats.tool_call_breakdown.entry(name.clone()).or_insert(0) += 1;
            app.generation.has_received_tokens = false;
            app.generation.verb = pick_verb();
            app.messages
                .push(ChatMessage::ToolCall { name, args, result: None });
        }
        AgentEvent::ToolResult {
            name, output, success,
        } => {
            if !success {
                app.stats.failed_tool_call_count += 1;
            }
            // Merge result into the last pending ToolCall
            let mut found = false;
            for msg in app.messages.iter_mut().rev() {
                if let ChatMessage::ToolCall { result, .. } = msg {
                    if result.is_none() {
                        *result = Some(ToolResultData { output, success });
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                app.messages.push(ChatMessage::Error(
                    format!("Orphaned tool result for '{}' (no pending tool call found)", name),
                ));
            }
        }
        AgentEvent::ContextUpdate { prompt_tokens } => {
            app.context_used = prompt_tokens;
        }
        AgentEvent::Done { prompt_tokens, eval_count } => {
            app.stats.agent_turns += 1;
            app.flush_streaming();
            if prompt_tokens > 0 {
                app.context_used = prompt_tokens;
            }
            app.stats.input_tokens += prompt_tokens;
            app.stats.output_tokens += eval_count;
            if let Some(start) = app.generation.start.take() {
                let duration = start.elapsed();
                if duration.as_secs() >= 1 {
                    app.messages
                        .push(ChatMessage::GenerationSummary { duration });
                }
            }
            app.finish_processing();
            // Refresh git status after agent finishes (bash tool may have changed things)
            let (branch, dirty) = get_git_info_sync();
            app.git_branch = branch;
            app.git_dirty = dirty;
        }
        AgentEvent::Error(e) => {
            app.flush_streaming();
            app.messages.push(ChatMessage::Error(e));
            app.finish_processing();
        }
        AgentEvent::ToolConfirmRequest { name, args } => {
            app.pending_confirm = Some(PendingConfirm { name, args });
        }
        AgentEvent::ContextTrimmed {
            removed_messages,
            estimated_tokens_freed,
        } => {
            app.stats.context_trims += 1;
            app.messages.push(ChatMessage::Info(format!(
                "Context trimmed: removed {} oldest messages (~{} tokens freed)",
                removed_messages, estimated_tokens_freed
            )));
        }
        AgentEvent::SubagentStart { .. } => {
            // The ToolCall event already shows "Subagent(task...)" in the UI.
        }
        AgentEvent::SubagentToolCall { name, args } => {
            app.messages.push(ChatMessage::SubagentToolCall {
                name,
                args,
                success: None,
            });
        }
        AgentEvent::SubagentToolResult { name, success } => {
            // Merge success into the last SubagentToolCall with matching name
            for msg in app.messages.iter_mut().rev() {
                if let ChatMessage::SubagentToolCall {
                    name: ref n,
                    success: ref mut s,
                    ..
                } = msg
                {
                    if *n == name && s.is_none() {
                        *s = Some(success);
                        break;
                    }
                }
            }
        }
        AgentEvent::SubagentEnd { .. } => {
            // The ToolResult event merges the final response into the ToolCall display.
        }
        AgentEvent::Cancelled => {
            app.flush_streaming();
            app.messages.push(ChatMessage::Info("Generation cancelled.".into()));
            app.finish_processing();
        }
        AgentEvent::SystemPromptInfo { base_prompt_tokens, project_docs, skills_tokens, tool_defs_breakdown } => {
            app.stats.base_prompt_tokens = base_prompt_tokens;
            app.stats.project_docs_tokens = project_docs;
            app.stats.skills_tokens = skills_tokens;
            app.stats.tool_defs_breakdown = tool_defs_breakdown;
        }
        // MessageLogged and Debug are handled by the session logger in the event loop,
        // not by the app state.
        AgentEvent::MessageLogged(_) | AgentEvent::Debug(_) => {}
    }

    // Auto-scroll to bottom if user hadn't scrolled up
    if was_at_bottom {
        app.scroll_offset = 0;
    }
}
