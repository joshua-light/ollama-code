use serde_json::Value;

use crate::message::{FunctionCall, ToolCall};

/// Strip leaked special tokens from model output.
///
/// Some models (e.g. Gemma 4) emit special/control tokens like `<|channel>`,
/// `<|turn>`, `<|think|>` as plain text instead of handling them internally.
/// These should never appear in user-facing content.
pub(super) fn strip_special_tokens(text: &str) -> String {
    // Match patterns like <|something> or <|something|> or <something|>
    // These are control tokens that leaked through the tokenizer.
    let mut result = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();

    while let Some(&(i, c)) = chars.peek() {
        if c == '<' && text[i..].starts_with("<|") {
            // Look for closing > (with optional | before it)
            if let Some(end) = text[i..].find('>') {
                // Skip the entire special token
                let token_end = i + end + 1;
                // Advance the iterator past this token
                while let Some(&(j, _)) = chars.peek() {
                    if j >= token_end {
                        break;
                    }
                    chars.next();
                }
                continue;
            }
        }
        if c == '<' && text[i..].len() > 1 {
            // Also match <something|> pattern
            if let Some(end_rel) = text[i..].find("|>") {
                let candidate = &text[i..i + end_rel + 2];
                // Only strip if it looks like a token (no spaces, reasonable length)
                if candidate.len() <= 30 && !candidate[1..].contains(' ') {
                    let token_end = i + end_rel + 2;
                    while let Some(&(j, _)) = chars.peek() {
                        if j >= token_end {
                            break;
                        }
                        chars.next();
                    }
                    continue;
                }
            }
        }
        result.push(c);
        chars.next();
    }

    result
}

/// Detect degenerate repetition in streamed content.
///
/// Checks whether the tail of the text consists of a short substring pattern
/// (3–40 chars) repeated 8+ times consecutively.  This catches the common LLM
/// failure mode where the model gets stuck in a sampling loop, e.g.
/// "approach-approach approach-approach approach-approach …"
pub(super) fn detect_repetition(text: &str) -> bool {
    // Need enough text to have a meaningful pattern + repetitions
    if text.len() < 100 {
        return false;
    }

    // Only inspect the tail — repetition is always at the end
    let start = text.len().saturating_sub(300);
    // Align to a char boundary
    let start = text.ceil_char_boundary(start);
    let tail = &text[start..];
    let tail_bytes = tail.as_bytes();

    for plen in 3..=40 {
        let min_repeats = 8;
        if tail.len() < plen * min_repeats {
            continue;
        }

        // Use the *last* `plen` bytes as the candidate pattern, then count
        // how many consecutive copies appear going backwards.
        let pattern = &tail_bytes[tail_bytes.len() - plen..];
        let mut count: usize = 0;
        let mut pos = tail_bytes.len();

        while pos >= plen {
            let candidate = &tail_bytes[pos - plen..pos];
            if candidate == pattern {
                count += 1;
                pos -= plen;
            } else {
                break;
            }
        }

        if count >= min_repeats {
            return true;
        }
    }

    false
}

/// Find balanced `{…}` JSON objects in a string by brace-matching.
/// Returns `(json_text, start_byte_offset, end_byte_offset)` for each match.
fn find_json_objects(text: &str) -> Vec<(String, usize, usize)> {
    let mut results = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'{' {
            let start = i;
            let mut depth: i32 = 0;
            let mut in_string = false;
            let mut escape = false;
            let mut j = i;

            while j < bytes.len() {
                let c = bytes[j];
                if escape {
                    escape = false;
                    j += 1;
                    continue;
                }
                if in_string {
                    if c == b'\\' {
                        escape = true;
                    } else if c == b'"' {
                        in_string = false;
                    }
                    j += 1;
                    continue;
                }
                // Outside a string
                match c {
                    b'"' => in_string = true,
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            let end = j + 1;
                            results.push((text[start..end].to_string(), start, end));
                            i = end;
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }

            // If we never closed the brace, skip past the opening {
            if depth != 0 {
                i = start + 1;
            }
        } else {
            i += 1;
        }
    }

    results
}

/// Try to extract tool calls from the text content when a model emits them as
/// plain-text JSON instead of using Ollama's structured `tool_calls` field.
/// Only matches JSON objects whose `"name"` value is in `known_tools`.
/// Returns extracted tool calls and the remaining (cleaned) content.
pub(super) fn extract_tool_calls_from_content(
    content: &str,
    known_tools: &[String],
) -> (Vec<ToolCall>, String) {
    let mut calls = Vec::new();

    // Some models wrap tool calls in <tool_call> XML-style tags — strip them
    // so the JSON extractor can find the inner objects.
    let cleaned = content
        .replace("<tool_call>", "")
        .replace("</tool_call>", "");

    // Find all top-level JSON objects in the text.
    let objects = find_json_objects(&cleaned);

    // Byte ranges to remove (collected, then applied in reverse order so
    // earlier removals don't shift later offsets).
    let mut removals: Vec<(usize, usize)> = Vec::new();

    for (json_str, start, end) in &objects {
        if let Ok(val) = serde_json::from_str::<Value>(json_str) {
            if let (Some(name), Some(arguments)) = (
                val.get("name").and_then(|v| v.as_str()),
                val.get("arguments").filter(|a| a.is_object()),
            ) {
                if known_tools.iter().any(|t| t == name) {
                    calls.push(ToolCall {
                        id: None,
                        call_type: None,
                        function: FunctionCall {
                            name: name.to_string(),
                            arguments: arguments.clone(),
                        },
                    });
                    removals.push((*start, *end));
                }
            }
        }
    }

    if !calls.is_empty() {
        // Remove matched JSON from the content in reverse order.
        let mut remaining = cleaned;
        for &(start, end) in removals.iter().rev() {
            remaining.replace_range(start..end, "");
        }
        return (calls, remaining.trim().to_string());
    }

    // Fallback: try to extract <function=NAME>...</function> XML-style tool calls.
    let (xml_calls, xml_remaining) = extract_function_tag_calls(&cleaned, known_tools);
    if !xml_calls.is_empty() {
        return (xml_calls, xml_remaining);
    }

    (calls, content.to_string())
}

/// Parse API error JSON into a user-friendly message.
///
/// Handles two common shapes:
/// - `{"error": {"message": "...", "type": "...", ...}}` (llama-server / OpenAI)
/// - `{"error": "..."}` (Ollama native)
pub(super) fn parse_api_error(body: &str) -> Option<String> {
    let json: Value = serde_json::from_str(body).ok()?;
    let error = json.get("error")?;

    // Nested error object (llama-server / OpenAI format)
    if let Some(obj) = error.as_object() {
        let error_type = obj.get("type").and_then(|t| t.as_str());
        if error_type == Some("exceed_context_size_error") {
            let n_prompt = obj.get("n_prompt_tokens").and_then(|n| n.as_u64());
            let n_ctx = obj.get("n_ctx").and_then(|n| n.as_u64());
            if let (Some(prompt), Some(ctx)) = (n_prompt, n_ctx) {
                return Some(format!(
                    "Context window exceeded ({} tokens requested, {} available). \
                     Use /clear to start fresh.",
                    prompt, ctx,
                ));
            }
        }
        // Fall back to the message field
        if let Some(msg) = obj.get("message").and_then(|m| m.as_str()) {
            return Some(msg.to_string());
        }
    }

    // Simple string: {"error": "..."}
    error.as_str().map(|s| s.to_string())
}

/// Parse `<function=NAME>...<parameter=KEY>VALUE</parameter>...</function>` blocks
/// from text content. Returns extracted tool calls and cleaned content.
fn extract_function_tag_calls(
    content: &str,
    known_tools: &[String],
) -> (Vec<ToolCall>, String) {
    let mut calls = Vec::new();
    let mut removals: Vec<(usize, usize)> = Vec::new();

    let mut search_from = 0;
    while let Some(func_start_rel) = content[search_from..].find("<function=") {
        let abs_start = search_from + func_start_rel;
        let name_start = abs_start + "<function=".len();

        let Some(name_end_rel) = content[name_start..].find('>') else {
            search_from = name_start;
            continue;
        };
        let name = content[name_start..name_start + name_end_rel].trim();

        if !known_tools.iter().any(|t| t == name) {
            search_from = name_start + name_end_rel;
            continue;
        }

        let body_start = name_start + name_end_rel + 1;
        let Some(func_end_rel) = content[body_start..].find("</function>") else {
            search_from = body_start;
            continue;
        };
        let body = &content[body_start..body_start + func_end_rel];
        let block_end = body_start + func_end_rel + "</function>".len();

        let mut arguments = serde_json::Map::new();
        let mut param_search = 0;
        while let Some(param_start_rel) = body[param_search..].find("<parameter=") {
            let pname_start = param_search + param_start_rel + "<parameter=".len();
            let Some(pname_end_rel) = body[pname_start..].find('>') else {
                param_search = pname_start;
                continue;
            };
            let param_name = body[pname_start..pname_start + pname_end_rel].trim();
            let value_start = pname_start + pname_end_rel + 1;

            let Some(pclose_rel) = body[value_start..].find("</parameter>") else {
                param_search = value_start;
                continue;
            };
            let value = body[value_start..value_start + pclose_rel].trim();

            arguments.insert(
                param_name.to_string(),
                serde_json::Value::String(value.to_string()),
            );

            param_search = value_start + pclose_rel + "</parameter>".len();
        }

        calls.push(ToolCall {
            id: None,
            call_type: None,
            function: FunctionCall {
                name: name.to_string(),
                arguments: serde_json::Value::Object(arguments),
            },
        });
        removals.push((abs_start, block_end));
        search_from = block_end;
    }

    if calls.is_empty() {
        return (calls, content.to_string());
    }

    // Remove matched blocks from content in reverse order.
    let mut remaining = content.to_string();
    for &(start, end) in removals.iter().rev() {
        remaining.replace_range(start..end, "");
    }
    (calls, remaining.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_function_tag_single_param() {
        let content = "<function=read>\n<parameter=file_path>\nCLAUDE.md\n</parameter>\n</function>\n</tool_call>";
        let known = vec!["read".to_string(), "bash".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(
            calls[0].function.arguments.get("file_path").unwrap().as_str().unwrap(),
            "CLAUDE.md"
        );
        assert!(remaining.is_empty(), "remaining was: {:?}", remaining);
    }

    #[test]
    fn extract_function_tag_multiple_params() {
        let content = "<function=edit>\n<parameter=file_path>\nsrc/main.rs\n</parameter>\n<parameter=old>\nfn main()\n</parameter>\n<parameter=new>\nfn main() -> Result<()>\n</parameter>\n</function>";
        let known = vec!["edit".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "edit");
        assert_eq!(
            calls[0].function.arguments.get("file_path").unwrap().as_str().unwrap(),
            "src/main.rs"
        );
        assert_eq!(
            calls[0].function.arguments.get("old").unwrap().as_str().unwrap(),
            "fn main()"
        );
        assert_eq!(
            calls[0].function.arguments.get("new").unwrap().as_str().unwrap(),
            "fn main() -> Result<()>"
        );
        assert!(remaining.is_empty());
    }

    #[test]
    fn extract_function_tag_with_surrounding_text() {
        let content = "Let me read that file.\n<function=read>\n<parameter=file_path>\nCLAUDE.md\n</parameter>\n</function>\nDone.";
        let known = vec!["read".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(remaining, "Let me read that file.\n\nDone.");
    }

    #[test]
    fn extract_function_tag_unknown_tool_ignored() {
        let content = "<function=unknown_tool>\n<parameter=x>\n1\n</parameter>\n</function>";
        let known = vec!["read".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert!(calls.is_empty());
        assert_eq!(remaining, content);
    }

    #[test]
    fn extract_json_still_works() {
        let content = r#"{"name": "read", "arguments": {"file_path": "test.rs"}}"#;
        let known = vec!["read".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert!(remaining.is_empty());
    }

    #[test]
    fn extract_function_tag_multiple_calls() {
        let content = "<function=read>\n<parameter=file_path>\na.rs\n</parameter>\n</function>\n<function=read>\n<parameter=file_path>\nb.rs\n</parameter>\n</function>";
        let known = vec!["read".to_string()];
        let (calls, _remaining) = extract_tool_calls_from_content(content, &known);
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[0].function.arguments.get("file_path").unwrap().as_str().unwrap(),
            "a.rs"
        );
        assert_eq!(
            calls[1].function.arguments.get("file_path").unwrap().as_str().unwrap(),
            "b.rs"
        );
    }
}
