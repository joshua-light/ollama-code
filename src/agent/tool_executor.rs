use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use tokio::sync::mpsc;

use crate::message::Message;
use crate::tools::BashTool;

use super::events::{send_event, AgentEvent};
use super::util::{poll_cancel, truncate_tool_output};
use super::Agent;

impl Agent {
    /// Dispatch a single tool call: run pre-hooks, handle confirmation,
    /// execute the tool (bash / subagent / registry), run post-hooks,
    /// emit events, and append the result message.
    /// Returns `Ok(true)` normally, `Ok(false)` when the user cancelled
    /// mid-execution (caller should return early).
    pub(super) async fn dispatch_tool_call(
        &mut self,
        tool_call: &crate::message::ToolCall,
        events: &mpsc::UnboundedSender<AgentEvent>,
        confirm_rx: &mut mpsc::UnboundedReceiver<bool>,
        cancel: &Arc<AtomicBool>,
    ) -> Result<bool> {
        let mut name = tool_call.function.name.clone();
        let original_args = &tool_call.function.arguments;

        // --- pre_tool_execute hooks ---
        let args_override = match self.hooks.pre_tool_execute(&name, original_args).await {
            Ok(pre) => match pre.action.as_deref() {
                Some("deny") => {
                    let msg = pre
                        .message
                        .unwrap_or_else(|| "Blocked by hook.".to_string());
                    send_event(
                        events,
                        AgentEvent::ToolCall {
                            name: name.clone(),
                            args: crate::format::format_tool_args_display(&name, original_args),
                        },
                    )?;
                    send_event(
                        events,
                        AgentEvent::ToolResult {
                            name: name.clone(),
                            output: msg.clone(),
                            success: false,
                        },
                    )?;
                    let tool_msg = Message::tool(&msg, tool_call.id.clone(), false);
                    self.messages.push(tool_msg.clone());
                    send_event(events, AgentEvent::MessageLogged(tool_msg))?;
                    return Ok(true);
                }
                Some("rewrite") | Some("modify") => {
                    if let Some(ref new_name) = pre.tool_name {
                        send_event(
                            events,
                            AgentEvent::Debug(format!(
                                "Hook rewrote tool call: {} -> {}",
                                name, new_name
                            )),
                        )?;
                        name = new_name.clone();
                    }
                    pre.arguments
                }
                _ => None, // "proceed" or absent
            },
            Err(e) => {
                send_event(
                    events,
                    AgentEvent::Debug(format!("pre_tool_execute hook error: {}", e)),
                )?;
                None
            }
        };
        let args = args_override.as_ref().unwrap_or(original_args);

        let args_display = crate::format::format_tool_args_display(&name, args);

        send_event(
            events,
            AgentEvent::ToolCall {
                name: name.clone(),
                args: args_display.clone(),
            },
        )?;

        // Request user confirmation for tools that modify state
        let needs_confirm = matches!(name.as_str(), "bash" | "edit" | "write" | "subagent")
            || self.plugin_confirm_tools.contains(name.as_str());
        if needs_confirm {
            send_event(
                events,
                AgentEvent::ToolConfirmRequest {
                    name: name.clone(),
                    args: args_display,
                },
            )?;

            let approved = confirm_rx.recv().await.unwrap_or(false);
            if !approved {
                let denied = "Tool execution denied by user.".to_string();
                send_event(
                    events,
                    AgentEvent::ToolResult {
                        name: name.clone(),
                        output: denied.clone(),
                        success: false,
                    },
                )?;
                let tool_msg = Message::tool(&denied, tool_call.id.clone(), false);
                self.messages.push(tool_msg.clone());
                send_event(events, AgentEvent::MessageLogged(tool_msg))?;
                return Ok(true);
            }
        }

        // Validate tool arguments against schema before execution.
        if let Some(Err(validation_err)) = self.tools.validate(&name, args) {
            let msg = format!(
                "Invalid arguments for '{}': {}",
                name, validation_err
            );
            send_event(
                events,
                AgentEvent::ToolResult {
                    name: name.clone(),
                    output: msg.clone(),
                    success: false,
                },
            )?;
            let tool_msg = Message::tool(&msg, tool_call.id.clone(), false);
            self.messages.push(tool_msg.clone());
            send_event(events, AgentEvent::MessageLogged(tool_msg))?;
            return Ok(true);
        }

        // Check cancellation before each tool call.
        // NOTE: callers rely on no tool result message being pushed when we
        // return Ok(false), so they can backfill "Cancelled" results for
        // this tool call and all remaining ones using tool_calls_with_ids[i..].
        if cancel.load(Ordering::Relaxed) {
            send_event(events, AgentEvent::Cancelled)?;
            return Ok(false);
        }

        let (mut result, mut success) = if name == "bash" {
            // Set up a channel for streaming partial output from the bash
            // process so the TUI can show live progress.
            let (output_tx, mut output_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
            let events_clone = events.clone();
            let forward_task = tokio::spawn(async move {
                while let Some(line) = output_rx.recv().await {
                    let _ = send_event(&events_clone, AgentEvent::ToolOutput { output: line });
                }
            });

            // Race tool execution against the cancel flag so ESC kills
            // long-running bash commands immediately.
            let output = tokio::select! {
                output = BashTool.execute_async(args, self.bash_timeout, Some(&output_tx)) => {
                    Some(output)
                },
                _ = poll_cancel(cancel) => None,
            };

            // Clean up forwarding on both paths.
            drop(output_tx);
            let _ = forward_task.await;

            match output {
                Some(o) => o,
                None => {
                    // NOTE: callers rely on no tool result message being pushed
                    // when we return Ok(false), so they can backfill "Cancelled"
                    // results starting from this tool call's index.
                    send_event(events, AgentEvent::Cancelled)?;
                    return Ok(false);
                }
            }
        } else if name == "subagent" {
            let task = args
                .get("task")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            self.execute_subagent(&task, events, confirm_rx, cancel).await
        } else {
            match self.tools.execute(&name, args) {
                Ok(output) => (output, true),
                Err(e) => (format!("Error: {}", e), false),
            }
        };

        // Track exploration for dynamic tool scoping
        if success && matches!(name.as_str(), "read" | "glob" | "grep") {
            self.has_explored = true;
        }

        // Track file reads for subagent context injection
        if success && name == "read" {
            if let Some(path) = args.get("file_path").and_then(|v| v.as_str()) {
                let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
                let start = offset.max(1) - 1;
                self.read_file_ranges.push((path.to_string(), start, start + limit));
            }
        }

        // Track edits for post-edit compilation and subagent enforcement
        if success && matches!(name.as_str(), "edit" | "write") {
            self.had_edits_this_run = true;
        }

        // --- post_tool_execute hooks ---
        let (post_result, post_success) = self
            .hooks
            .post_tool_execute(&name, args, result, success)
            .await;
        result = post_result;
        success = post_success;

        send_event(
            events,
            AgentEvent::ToolResult {
                name: name.clone(),
                output: result.clone(),
                success,
            },
        )?;

        // Truncate before storing in context to avoid blowing the window.
        // The full output was already sent to the UI above.
        let context_result = truncate_tool_output(&result);
        let tool_msg = Message::tool(&context_result, tool_call.id.clone(), success);
        self.messages.push(tool_msg.clone());
        send_event(events, AgentEvent::MessageLogged(tool_msg))?;

        Ok(true)
    }
}
