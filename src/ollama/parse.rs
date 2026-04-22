use std::borrow::Cow;

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

/// Best-effort repair of near-JSON fragments small models sometimes emit as
/// tool-call arguments.
///
/// Handles: single-quoted strings, trailing commas before `}`/`]`, unescaped
/// control chars inside strings (`\n`/`\r`/`\t`), and unclosed string/array/
/// object tokens (closed with `"`, `]`, `}` at the end). Does not attempt to
/// fix unquoted keys or Python-ish literals — those are beyond the budget of
/// a single-pass repair.
pub(crate) fn repair_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    let mut chars = s.chars().peekable();
    let mut in_string = false;
    let mut escape = false;
    let mut open_braces: i32 = 0;
    let mut open_brackets: i32 = 0;

    while let Some(c) = chars.next() {
        if escape {
            out.push(c);
            escape = false;
            continue;
        }
        if in_string {
            match c {
                '\\' => {
                    escape = true;
                    out.push(c);
                }
                '"' => {
                    in_string = false;
                    out.push(c);
                }
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                _ => out.push(c),
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push(c);
            }
            '\'' => {
                // Rewrite a single-quoted string as a double-quoted one.
                out.push('"');
                while let Some(d) = chars.next() {
                    if d == '\\' {
                        out.push('\\');
                        if let Some(e) = chars.next() {
                            out.push(e);
                        }
                        continue;
                    }
                    if d == '\'' {
                        out.push('"');
                        break;
                    }
                    match d {
                        '"' => {
                            out.push('\\');
                            out.push('"');
                        }
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        _ => out.push(d),
                    }
                }
            }
            '{' => {
                open_braces += 1;
                out.push(c);
            }
            '}' => {
                strip_trailing_comma(&mut out);
                open_braces -= 1;
                out.push(c);
            }
            '[' => {
                open_brackets += 1;
                out.push(c);
            }
            ']' => {
                strip_trailing_comma(&mut out);
                open_brackets -= 1;
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    if in_string {
        out.push('"');
    }
    for _ in 0..open_brackets.max(0) {
        out.push(']');
    }
    for _ in 0..open_braces.max(0) {
        out.push('}');
    }
    out
}

fn strip_trailing_comma(out: &mut String) {
    let trimmed_len = out.trim_end().len();
    if out[..trimmed_len].ends_with(',') {
        out.truncate(trimmed_len - 1);
    }
}

/// Parse `s` as JSON, falling back to a repaired version if strict parsing
/// fails. Returns `None` if both attempts fail.
pub(crate) fn parse_json_lenient(s: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str(s) {
        return Some(v);
    }
    let repaired = repair_json(s);
    if repaired == s {
        return None;
    }
    serde_json::from_str(&repaired).ok()
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

/// Remove empty code-fence pairs left behind after tool-call JSON extraction.
///
/// After we strip a fenced JSON tool-call out of the content, we can be left
/// with a bare ```` ```json ```` / ```` ``` ```` pair surrounding nothing but
/// whitespace. Collapse any such pair (with optional language tag).
fn strip_empty_fences(text: &str) -> Cow<'_, str> {
    if !text.contains("```") {
        return Cow::Borrowed(text);
    }
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut idx = 0;
    let mut copy_start = 0;
    while idx < bytes.len() {
        // All markers scanned here are ASCII, so fence starts always land on a
        // UTF-8 char boundary — non-matching positions can safely advance by 1
        // byte because continuation bytes (>= 0x80) never equal b'`'.
        if bytes[idx] == b'`' && text[idx..].starts_with("```") {
            let mut j = idx + 3;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
            let mut k = j;
            while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                k += 1;
            }
            if k + 3 <= bytes.len() && &bytes[k..k + 3] == b"```" {
                out.push_str(&text[copy_start..idx]);
                idx = k + 3;
                copy_start = idx;
                continue;
            }
        }
        idx += 1;
    }
    out.push_str(&text[copy_start..]);
    Cow::Owned(out)
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
        if let Some(val) = parse_json_lenient(json_str) {
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
        return (calls, strip_empty_fences(&remaining).trim().to_string());
    }

    // Fallback: try to extract <function=NAME>...</function> XML-style tool calls.
    let (xml_calls, xml_remaining) = extract_function_tag_calls(&cleaned, known_tools);
    if !xml_calls.is_empty() {
        let xml_remaining = strip_empty_fences(&xml_remaining).trim().to_string();
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

    // ---------------------------------------------------------------
    // strip_special_tokens
    // ---------------------------------------------------------------

    #[test]
    fn strip_special_tokens_no_tokens() {
        assert_eq!(strip_special_tokens("Hello, world!"), "Hello, world!");
    }

    #[test]
    fn strip_special_tokens_empty() {
        assert_eq!(strip_special_tokens(""), "");
    }

    #[test]
    fn strip_pipe_open_close_pattern() {
        // <|something|> pattern
        assert_eq!(strip_special_tokens("<|think|>hello"), "hello");
        assert_eq!(strip_special_tokens("hello<|end|>"), "hello");
        assert_eq!(strip_special_tokens("a<|turn|>b"), "ab");
    }

    #[test]
    fn strip_pipe_open_pattern() {
        // <|something> pattern (no trailing pipe)
        assert_eq!(strip_special_tokens("<|channel>hello"), "hello");
        assert_eq!(strip_special_tokens("hello<|eos>"), "hello");
    }

    #[test]
    fn strip_close_pipe_pattern() {
        // <something|> pattern
        assert_eq!(strip_special_tokens("<endoftext|>hello"), "hello");
        assert_eq!(strip_special_tokens("hello<pad|>"), "hello");
    }

    #[test]
    fn strip_multiple_tokens() {
        assert_eq!(
            strip_special_tokens("<|think|>Hello<|turn|> world<|end|>"),
            "Hello world"
        );
    }

    #[test]
    fn strip_preserves_normal_angle_brackets() {
        // Normal HTML-like content should pass through
        assert_eq!(strip_special_tokens("<div>hello</div>"), "<div>hello</div>");
        assert_eq!(strip_special_tokens("a < b > c"), "a < b > c");
    }

    #[test]
    fn strip_long_token_not_stripped() {
        // <something|> pattern but longer than 30 chars should NOT be stripped
        let long = "<abcdefghijklmnopqrstuvwxyz1234|>";
        assert_eq!(strip_special_tokens(long), long);
    }

    #[test]
    fn strip_token_with_space_not_stripped() {
        // <something with space|> should not be stripped
        assert_eq!(strip_special_tokens("<hello world|>"), "<hello world|>");
    }

    // ---------------------------------------------------------------
    // detect_repetition
    // ---------------------------------------------------------------

    #[test]
    fn detect_repetition_short_text() {
        // Text under 100 chars should never trigger
        assert!(!detect_repetition("abc"));
        assert!(!detect_repetition(""));
        assert!(!detect_repetition(&"x".repeat(99)));
    }

    #[test]
    fn detect_repetition_no_repetition() {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(5);
        // This is repeated but the pattern is too long (>40 chars) to detect
        // with the 3..=40 window. Also only 5 repetitions < 8.
        assert!(!detect_repetition(&text));
    }

    #[test]
    fn detect_repetition_obvious_loop() {
        // "abc" repeated many times at the end of a sufficiently long text
        let prefix = "x".repeat(80);
        let text = prefix + &"abc".repeat(20);
        assert!(detect_repetition(&text));
    }

    #[test]
    fn detect_repetition_approach_loop() {
        // Real-world failure mode: model repeating "approach-"
        let text =
            "Let me think about this problem. ".to_string() + &"approach-".repeat(15);
        assert!(detect_repetition(&text));
    }

    #[test]
    fn detect_repetition_barely_enough() {
        // Exactly 8 repetitions of a 5-char pattern at the end
        let prefix = "x".repeat(100);
        let text = prefix + &"hello".repeat(8);
        assert!(detect_repetition(&text));
    }

    #[test]
    fn detect_repetition_just_below_threshold() {
        // 7 repetitions should NOT trigger (min is 8)
        let prefix = "x".repeat(100);
        let text = prefix + &"hello".repeat(7);
        assert!(!detect_repetition(&text));
    }

    // ---------------------------------------------------------------
    // repair_json / parse_json_lenient
    // ---------------------------------------------------------------

    #[test]
    fn repair_json_trailing_comma_object() {
        let input = r#"{"a": 1, "b": 2,}"#;
        let v: Value = serde_json::from_str(&repair_json(input)).unwrap();
        assert_eq!(v["a"], 1);
        assert_eq!(v["b"], 2);
    }

    #[test]
    fn repair_json_trailing_comma_array() {
        let input = r#"[1, 2, 3,]"#;
        let v: Value = serde_json::from_str(&repair_json(input)).unwrap();
        assert_eq!(v.as_array().unwrap().len(), 3);
    }

    #[test]
    fn repair_json_single_quotes() {
        let input = r#"{'command': 'ls -la'}"#;
        let v: Value = serde_json::from_str(&repair_json(input)).unwrap();
        assert_eq!(v["command"], "ls -la");
    }

    #[test]
    fn repair_json_single_quotes_with_embedded_double() {
        let input = r#"{'msg': 'say "hi"'}"#;
        let v: Value = serde_json::from_str(&repair_json(input)).unwrap();
        assert_eq!(v["msg"], r#"say "hi""#);
    }

    #[test]
    fn repair_json_unclosed_object() {
        let input = r#"{"command": "ls""#;
        let v: Value = serde_json::from_str(&repair_json(input)).unwrap();
        assert_eq!(v["command"], "ls");
    }

    #[test]
    fn repair_json_unclosed_string_and_object() {
        let input = r#"{"command": "ls"#;
        let v: Value = serde_json::from_str(&repair_json(input)).unwrap();
        assert_eq!(v["command"], "ls");
    }

    #[test]
    fn repair_json_raw_newlines_in_string() {
        let input = "{\"content\": \"line1\nline2\"}";
        let v: Value = serde_json::from_str(&repair_json(input)).unwrap();
        assert_eq!(v["content"], "line1\nline2");
    }

    #[test]
    fn repair_json_wellformed_unchanged() {
        let input = r#"{"a": 1}"#;
        assert_eq!(repair_json(input), input);
    }

    #[test]
    fn parse_lenient_falls_back_to_repair() {
        let v = parse_json_lenient(r#"{"a": 1,}"#).unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn parse_lenient_returns_none_for_garbage() {
        assert!(parse_json_lenient("not json at all !!!").is_none());
    }

    #[test]
    fn parse_lenient_strict_first() {
        // Well-formed JSON should not be run through repair (round-trips fine
        // either way, but verifies the first branch is taken).
        let v = parse_json_lenient(r#"{"b": [1, 2, 3]}"#).unwrap();
        assert_eq!(v["b"][1], 2);
    }

    // ---------------------------------------------------------------
    // find_json_objects
    // ---------------------------------------------------------------

    #[test]
    fn find_json_objects_empty_string() {
        assert!(find_json_objects("").is_empty());
    }

    #[test]
    fn find_json_objects_no_json() {
        assert!(find_json_objects("hello world").is_empty());
    }

    #[test]
    fn find_json_objects_simple_object() {
        let results = find_json_objects(r#"{"key": "value"}"#);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, r#"{"key": "value"}"#);
        assert_eq!(results[0].1, 0);
        assert_eq!(results[0].2, 16);
    }

    #[test]
    fn find_json_objects_nested() {
        let text = r#"{"outer": {"inner": 42}}"#;
        let results = find_json_objects(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, text);
    }

    #[test]
    fn find_json_objects_multiple() {
        let text = r#"before {"a": 1} middle {"b": 2} after"#;
        let results = find_json_objects(text);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, r#"{"a": 1}"#);
        assert_eq!(results[1].0, r#"{"b": 2}"#);
    }

    #[test]
    fn find_json_objects_braces_in_strings() {
        // Braces inside JSON strings should not confuse the parser
        let text = r#"{"key": "value with { and } inside"}"#;
        let results = find_json_objects(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, text);
    }

    #[test]
    fn find_json_objects_escaped_quotes() {
        let text = r#"{"key": "value with \" escaped"}"#;
        let results = find_json_objects(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, text);
    }

    #[test]
    fn find_json_objects_unclosed_brace() {
        // Unclosed brace should be skipped
        let text = r#"{"key": "value"#;
        let results = find_json_objects(text);
        assert!(results.is_empty());
    }

    #[test]
    fn find_json_objects_offsets_correct() {
        let text = r#"abc{"x":1}def"#;
        let results = find_json_objects(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 3); // start
        assert_eq!(results[0].2, 10); // end
    }

    // ---------------------------------------------------------------
    // parse_api_error
    // ---------------------------------------------------------------

    #[test]
    fn parse_api_error_ollama_string() {
        let body = r#"{"error": "model not found"}"#;
        assert_eq!(parse_api_error(body), Some("model not found".to_string()));
    }

    #[test]
    fn parse_api_error_openai_nested() {
        let body = r#"{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}"#;
        assert_eq!(
            parse_api_error(body),
            Some("Rate limit exceeded".to_string())
        );
    }

    #[test]
    fn parse_api_error_context_exceeded() {
        let body = r#"{"error": {"type": "exceed_context_size_error", "n_prompt_tokens": 5000, "n_ctx": 4096}}"#;
        let msg = parse_api_error(body).unwrap();
        assert!(msg.contains("5000"));
        assert!(msg.contains("4096"));
        assert!(msg.contains("Context window exceeded"));
    }

    #[test]
    fn parse_api_error_context_exceeded_missing_fields() {
        // If n_prompt/n_ctx are missing, fall through to message or None
        let body = r#"{"error": {"type": "exceed_context_size_error"}}"#;
        // No message field and no string tokens => None
        assert_eq!(parse_api_error(body), None);
    }

    #[test]
    fn parse_api_error_invalid_json() {
        assert_eq!(parse_api_error("not json"), None);
    }

    #[test]
    fn parse_api_error_no_error_field() {
        assert_eq!(parse_api_error(r#"{"status": "ok"}"#), None);
    }

    #[test]
    fn parse_api_error_empty() {
        assert_eq!(parse_api_error(""), None);
    }

    // ---------------------------------------------------------------
    // extract_tool_calls_from_content
    // ---------------------------------------------------------------

    #[test]
    fn extract_no_tools_empty_content() {
        let (calls, remaining) = extract_tool_calls_from_content("", &[]);
        assert!(calls.is_empty());
        assert_eq!(remaining, "");
    }

    #[test]
    fn extract_no_known_tools() {
        let content = r#"{"name": "read", "arguments": {"file_path": "test.rs"}}"#;
        let (calls, remaining) = extract_tool_calls_from_content(content, &[]);
        assert!(calls.is_empty());
        // With no known_tool_names, the caller passes empty slice,
        // but the function is only called when known_tool_names is non-empty
        // in postprocess_response. Still, the function itself should return unchanged.
        assert_eq!(remaining, content);
    }

    #[test]
    fn extract_json_with_tool_call_tags() {
        // Some models wrap JSON in <tool_call> tags
        let content = r#"<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>"#;
        let known = vec!["bash".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "bash");
        assert_eq!(
            calls[0].function.arguments.get("command").unwrap().as_str().unwrap(),
            "ls"
        );
        assert!(remaining.is_empty());
    }

    #[test]
    fn extract_json_multiple_tool_calls() {
        let content = r#"{"name": "read", "arguments": {"file_path": "a.rs"}} {"name": "read", "arguments": {"file_path": "b.rs"}}"#;
        let known = vec!["read".to_string()];
        let (calls, _) = extract_tool_calls_from_content(content, &known);
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

    #[test]
    fn extract_json_wrong_tool_name_ignored() {
        let content = r#"{"name": "delete_everything", "arguments": {"path": "/"}}"#;
        let known = vec!["read".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert!(calls.is_empty());
        assert_eq!(remaining, content);
    }

    #[test]
    fn extract_json_missing_arguments_field() {
        // JSON has name but no arguments object
        let content = r#"{"name": "read", "other": "field"}"#;
        let known = vec!["read".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert!(calls.is_empty());
        assert_eq!(remaining, content);
    }

    #[test]
    fn extract_json_fenced_strips_empty_fences() {
        // Models sometimes emit tool calls wrapped in ```json ... ``` fences.
        // After the JSON is pulled out, the surrounding fence pair should be
        // collapsed so the assistant's remaining content isn't just "```json\n\n```".
        let content = "```json\n{\"name\":\"write\",\"arguments\":{\"file_path\":\"x\"}}\n```";
        let known = vec!["write".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "write");
        assert_eq!(
            calls[0]
                .function
                .arguments
                .get("file_path")
                .unwrap()
                .as_str()
                .unwrap(),
            "x"
        );
        assert!(remaining.is_empty(), "remaining was: {:?}", remaining);
    }

    #[test]
    fn extract_json_arguments_not_object() {
        // arguments is a string, not object
        let content = r#"{"name": "read", "arguments": "not an object"}"#;
        let known = vec!["read".to_string()];
        let (calls, remaining) = extract_tool_calls_from_content(content, &known);
        assert!(calls.is_empty());
        assert_eq!(remaining, content);
    }

    // ---------------------------------------------------------------
    // Existing extract_function_tag tests
    // ---------------------------------------------------------------

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
