use anyhow::Result;
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::Path;

use super::{expand_tilde, optional_str, required_str, Tool, ToolDefinition};
use crate::format::truncate_args;

enum FindResult {
    None,
    One(usize),
    Multiple(usize),
}

/// Exact substring search — current behavior.
fn find_exact(content: &str, needle: &str) -> FindResult {
    let count = content.matches(needle).count();
    match count {
        0 => FindResult::None,
        1 => FindResult::One(content.find(needle).unwrap()),
        n => FindResult::Multiple(n),
    }
}

/// Strip trailing whitespace from each line.
fn normalize_trailing(s: &str) -> String {
    s.lines()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Search for `needle` in `content` after stripping trailing whitespace from
/// every line in both. Returns the byte offset in the *original* content where
/// the matching region starts.
fn find_normalized(content: &str, needle: &str) -> FindResult {
    let norm_content = normalize_trailing(content);
    let norm_needle = normalize_trailing(needle);

    let count = norm_content.matches(&norm_needle).count();
    match count {
        0 => FindResult::None,
        1 => {
            // Map the char offset in the normalized string back to a byte
            // offset in the original content by finding which original line
            // the match starts on.
            let norm_pos = norm_content.find(&norm_needle).unwrap();
            let norm_line_idx = norm_content[..norm_pos].matches('\n').count();
            // The match starts at the beginning of this line in the original.
            let orig_pos = content
                .match_indices('\n')
                .nth(norm_line_idx.wrapping_sub(1))
                .map(|(i, _)| i + 1)
                .unwrap_or(0);
            // If norm_pos doesn't fall on a line boundary, add the column offset.
            let norm_line_start = if norm_line_idx == 0 {
                0
            } else {
                norm_content
                    .match_indices('\n')
                    .nth(norm_line_idx - 1)
                    .map(|(i, _)| i + 1)
                    .unwrap_or(0)
            };
            let col = norm_pos - norm_line_start;
            FindResult::One(orig_pos + col)
        }
        n => FindResult::Multiple(n),
    }
}

/// Replace the region in the original content that matches `needle` (after
/// trailing-whitespace normalization) at position `pos` with `replacement`.
fn replace_normalized(content: &str, needle: &str, replacement: &str, _pos: usize) -> String {
    let norm_needle = normalize_trailing(needle);
    let needle_line_count = norm_needle.matches('\n').count() + 1;

    // Find which original lines correspond to the normalized match.
    let norm_content = normalize_trailing(content);
    let norm_pos = norm_content.find(&norm_needle).unwrap();
    let start_line = norm_content[..norm_pos].matches('\n').count();
    let end_line = start_line + needle_line_count;

    // Rebuild: lines before + replacement + lines after
    let orig_lines: Vec<&str> = content.lines().collect();
    let mut result = String::new();
    for line in &orig_lines[..start_line] {
        result.push_str(line);
        result.push('\n');
    }
    result.push_str(replacement);
    if !replacement.ends_with('\n') && end_line < orig_lines.len() {
        result.push('\n');
    }
    for (i, line) in orig_lines[end_line..].iter().enumerate() {
        result.push_str(line);
        if end_line + i + 1 < orig_lines.len() || content.ends_with('\n') {
            result.push('\n');
        }
    }
    result
}

/// Find the closest matching region in `content` for `needle` by looking for
/// lines that contain the first non-empty line of `needle`. Returns
/// `Some((start_line_1indexed, end_line_1indexed, snippet))` or `None`.
fn find_closest_match(content: &str, needle: &str) -> Option<(usize, usize, String)> {
    let needle_lines: Vec<&str> = needle.lines().collect();
    let file_lines: Vec<&str> = content.lines().collect();

    if needle_lines.is_empty() || file_lines.is_empty() {
        return None;
    }

    // Find the first non-empty line from needle to use as anchor
    let anchor = needle_lines
        .iter()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim())?;

    // Find all file lines that contain the anchor (trimmed)
    let mut best_idx: Option<usize> = None;
    let mut best_score: usize = 0;

    for (i, file_line) in file_lines.iter().enumerate() {
        if file_line.trim().contains(anchor) {
            // Score: count how many subsequent needle lines match file lines
            let span = needle_lines.len().min(file_lines.len() - i);
            let mut score = 0;
            for j in 0..span {
                if file_lines[i + j].trim() == needle_lines[j].trim() {
                    score += 1;
                }
            }
            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }
    }

    // If exact anchor wasn't found, try substring match on trimmed first line
    if best_idx.is_none() {
        // Try each word token from the anchor (at least 4 chars) as fallback
        let tokens: Vec<&str> = anchor.split_whitespace().filter(|t| t.len() >= 4).collect();
        for (i, file_line) in file_lines.iter().enumerate() {
            let trimmed = file_line.trim();
            let matching_tokens = tokens.iter().filter(|t| trimmed.contains(**t)).count();
            if matching_tokens > best_score {
                best_score = matching_tokens;
                best_idx = Some(i);
            }
        }
        // Only use token match if we matched at least 2 tokens or 1 long token
        if best_score < 1 {
            best_idx = None;
        }
    }

    let idx = best_idx?;
    let span = needle_lines.len();
    let end = (idx + span).min(file_lines.len());
    let snippet: String = file_lines[idx..end]
        .iter()
        .enumerate()
        .map(|(j, line)| format!("{:>4} | {}", idx + j + 1, line))
        .collect::<Vec<_>>()
        .join("\n");

    Some((idx + 1, end, snippet))
}

/// Find candidate window starts by anchor-line matching, returning all
/// candidate starting indices (0-based). Used by indent-tolerance fallback to
/// scan candidate regions without re-implementing the anchor search.
fn find_candidate_window_starts(content: &str, needle: &str) -> Vec<usize> {
    let needle_lines: Vec<&str> = needle.lines().collect();
    let file_lines: Vec<&str> = content.lines().collect();
    if needle_lines.is_empty() || file_lines.is_empty() {
        return Vec::new();
    }
    let Some(anchor) = needle_lines
        .iter()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim())
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (i, file_line) in file_lines.iter().enumerate() {
        if file_line.trim() == anchor || file_line.trim().contains(anchor) {
            out.push(i);
        }
    }
    out
}

/// If most non-empty lines in `s` start with a "{:>4}\t" or "{:>4} |"
/// prefix (the format `read` produces), strip that prefix from every line.
/// Returns `Some(stripped)` when the heuristic fires, `None` otherwise.
fn strip_line_number_prefix(s: &str) -> Option<String> {
    let lines: Vec<&str> = s.lines().collect();
    if lines.is_empty() {
        return None;
    }
    let mut total_nonempty = 0usize;
    let mut matching = 0usize;
    for line in &lines {
        if line.trim().is_empty() {
            continue;
        }
        total_nonempty += 1;
        if line_has_number_prefix(line) {
            matching += 1;
        }
    }
    if total_nonempty == 0 {
        return None;
    }
    // 80% threshold
    if matching * 5 < total_nonempty * 4 {
        return None;
    }
    let stripped: Vec<String> = lines
        .iter()
        .map(|line| {
            if line_has_number_prefix(line) {
                strip_one_number_prefix(line).to_string()
            } else {
                (*line).to_string()
            }
        })
        .collect();
    Some(stripped.join("\n"))
}

/// True when the line starts with optional whitespace, then digits, then
/// either `\t` or ` |` (the two prefix forms `read` produces).
fn line_has_number_prefix(line: &str) -> bool {
    let trimmed = line.trim_start();
    let digits_end = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits_end == 0 {
        return false;
    }
    let after_digits = &trimmed[digits_end..];
    after_digits.starts_with('\t') || after_digits.starts_with(" |")
}

/// Strip a single number-prefix from a line. Caller must have already
/// confirmed `line_has_number_prefix(line)` is true.
fn strip_one_number_prefix(line: &str) -> &str {
    let leading_ws = line.len() - line.trim_start().len();
    let after_ws = &line[leading_ws..];
    let digits_end = after_ws.chars().take_while(|c| c.is_ascii_digit()).count();
    let after_digits = &after_ws[digits_end..];
    if let Some(rest) = after_digits.strip_prefix('\t') {
        rest
    } else if let Some(rest) = after_digits.strip_prefix(" |") {
        // Drop one optional space following the pipe (e.g. " 42 | foo").
        rest.strip_prefix(' ').unwrap_or(rest)
    } else {
        line
    }
}

/// Compute the minimum count of leading-whitespace chars across non-empty
/// lines. Returns 0 when all lines are empty or none is indented.
fn min_indent(s: &str) -> usize {
    let mut min: Option<usize> = None;
    for line in s.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        min = Some(match min {
            Some(m) => m.min(indent),
            None => indent,
        });
    }
    min.unwrap_or(0)
}

/// Strip exactly `n` leading-whitespace chars from each non-empty line. Lines
/// with fewer than `n` leading whitespace chars are left as-is (defensive —
/// caller normally derived `n` from `min_indent`).
fn strip_leading_indent(s: &str, n: usize) -> String {
    if n == 0 {
        return s.to_string();
    }
    s.lines()
        .map(|line| {
            if line.trim().is_empty() {
                line.to_string()
            } else {
                let mut stripped = 0;
                let mut idx = 0;
                for c in line.chars() {
                    if stripped >= n {
                        break;
                    }
                    if c == ' ' || c == '\t' {
                        stripped += 1;
                        idx += c.len_utf8();
                    } else {
                        break;
                    }
                }
                line[idx..].to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Re-indent `new_string` so its first non-empty line matches `target_indent`
/// of leading whitespace, preserving relative indentation of subsequent lines.
fn reindent_replacement(new_string: &str, target_indent: &str) -> String {
    let new_min = min_indent(new_string);
    new_string
        .lines()
        .map(|line| {
            if line.trim().is_empty() {
                line.to_string()
            } else if line.len() >= new_min {
                // Strip the model's own min_indent then prepend target.
                let trimmed = &line[new_min.min(line.len() - line.trim_start().len())..];
                // Defensive: if `new_min` exceeds this line's actual leading
                // whitespace count, fall back to trim_start to avoid eating
                // non-whitespace characters.
                let actual_lead = line.len() - line.trim_start().len();
                if actual_lead < new_min {
                    format!("{}{}", target_indent, line.trim_start())
                } else {
                    format!("{}{}", target_indent, trimmed)
                }
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Try the indent-tolerance fallback: strip `old_string`'s common indent,
/// scan candidate windows in `content` whose own de-indented body matches the
/// de-indented needle exactly (after also normalizing trailing whitespace).
/// Returns `Some((window_start_line_0, window_end_line_0_exclusive, target_indent))`
/// when exactly one candidate matches, else `None`.
fn try_indent_tolerance(
    content: &str,
    old_string: &str,
) -> Option<(usize, usize, String)> {
    let indent = min_indent(old_string);
    if indent == 0 {
        return None;
    }
    let de_needle = normalize_trailing(&strip_leading_indent(old_string, indent));
    let needle_lines: Vec<&str> = old_string.lines().collect();
    if needle_lines.is_empty() {
        return None;
    }
    let file_lines: Vec<&str> = content.lines().collect();
    let candidate_starts = find_candidate_window_starts(content, old_string);

    let mut hits: Vec<(usize, usize, String)> = Vec::new();
    for start in candidate_starts {
        let end = (start + needle_lines.len()).min(file_lines.len());
        if end - start != needle_lines.len() {
            continue;
        }
        let window: String = file_lines[start..end].join("\n");
        let window_indent = min_indent(&window);
        let de_window = normalize_trailing(&strip_leading_indent(&window, window_indent));
        if de_window == de_needle {
            // Capture the actual leading-whitespace string of the matched
            // window's first non-empty line — that's what we'll re-indent
            // `new_string` to.
            let target_indent = file_lines[start..end]
                .iter()
                .find(|l| !l.trim().is_empty())
                .map(|l| {
                    let n = l.len() - l.trim_start().len();
                    l[..n].to_string()
                })
                .unwrap_or_default();
            hits.push((start, end, target_indent));
        }
    }
    if hits.len() == 1 {
        Some(hits.into_iter().next().unwrap())
    } else {
        None
    }
}

/// Apply a line-range replacement: replace lines `start..=end` (1-based,
/// inclusive) of `content` with `new_string`. Returns the new full content.
/// Caller is responsible for bounds-checking; this helper assumes
/// `1 <= start <= end <= total_lines`.
pub(crate) fn apply_line_range_replacement(
    content: &str,
    start: usize,
    end: usize,
    new_string: &str,
) -> String {
    let file_lines: Vec<&str> = content.lines().collect();
    let start_idx = start - 1;
    let end_idx = end; // exclusive

    let mut new_content = String::new();
    for line in &file_lines[..start_idx] {
        new_content.push_str(line);
        new_content.push('\n');
    }
    new_content.push_str(new_string);
    if !new_string.ends_with('\n') && end_idx < file_lines.len() {
        new_content.push('\n');
    }
    for (i, line) in file_lines[end_idx..].iter().enumerate() {
        new_content.push_str(line);
        if end_idx + i + 1 < file_lines.len() || content.ends_with('\n') {
            new_content.push('\n');
        }
    }
    new_content
}

/// Mismatch classification for error and fuzzy-success messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MismatchType {
    Indentation,
    LineNumberPrefix,
    Whitespace,
    TokenSubstitution,
}

impl MismatchType {
    fn as_str(self) -> &'static str {
        match self {
            MismatchType::Indentation => "indentation",
            MismatchType::LineNumberPrefix => "line_number_prefix",
            MismatchType::Whitespace => "whitespace",
            MismatchType::TokenSubstitution => "token_substitution",
        }
    }
}

/// Classify how `old_string` differs from the matched `window` text.
fn classify_mismatch(old_string: &str, window: &str) -> MismatchType {
    // 1. Line-number prefix wins if it would convert the needle into the
    // window after stripping.
    if strip_line_number_prefix(old_string).is_some() {
        return MismatchType::LineNumberPrefix;
    }
    // 2. Indentation: same content after stripping each side's min_indent.
    let needle_indent = min_indent(old_string);
    let window_indent = min_indent(window);
    if needle_indent != window_indent {
        let de_needle = strip_leading_indent(old_string, needle_indent);
        let de_window = strip_leading_indent(window, window_indent);
        if normalize_trailing(&de_needle) == normalize_trailing(&de_window) {
            return MismatchType::Indentation;
        }
    }
    // 3. Trailing-whitespace-only difference.
    if normalize_trailing(old_string) == normalize_trailing(window) {
        return MismatchType::Whitespace;
    }
    MismatchType::TokenSubstitution
}

/// Result of attempting to compute a new file body from `old_string`.
struct EditOutcome {
    new_content: String,
    /// Prefix prepended to the rendered diff. Empty for exact/normalized
    /// matches; populated by the fuzzy auto-apply path with a `[harness]`
    /// note describing the match score and mismatch type.
    diff_prefix: String,
    /// 0-based line span replaced (inclusive start, exclusive end). Used by
    /// the diff renderer to know which lines changed.
    replaced_start_line: usize,
    replaced_end_line: usize,
}

/// Try every fallback strategy on `old_string`. Returns the outcome on
/// success, or `Err` carrying the formatted error message for the model.
fn resolve_string_edit(
    file_path: &str,
    content: &str,
    old_string: &str,
    new_string: &str,
) -> Result<EditOutcome> {
    // 0. Exact match (existing behavior).
    if let FindResult::One(pos) = find_exact(content, old_string) {
        let new_content = content.replacen(old_string, new_string, 1);
        let start_line = content[..pos].matches('\n').count();
        let end_line = start_line + old_string.lines().count();
        return Ok(EditOutcome {
            new_content,
            diff_prefix: String::new(),
            replaced_start_line: start_line,
            replaced_end_line: end_line,
        });
    }
    if let FindResult::Multiple(count) = find_exact(content, old_string) {
        anyhow::bail!(
            "old_string found {} times in '{}'. Provide more surrounding context to make the match unique.",
            count,
            file_path
        );
    }

    // 1. Trailing-whitespace-normalized match (existing behavior).
    // `find_normalized` returns a byte offset in the original content where
    // the match starts at a line boundary plus a column delta — newlines
    // never appear in the column delta, so counting newlines in
    // `content[..pos]` recovers the line index without re-normalizing.
    if let FindResult::One(pos) = find_normalized(content, old_string) {
        let new_content = replace_normalized(content, old_string, new_string, pos);
        let start_line = content[..pos].matches('\n').count();
        let end_line = start_line + old_string.lines().count();
        return Ok(EditOutcome {
            new_content,
            diff_prefix: String::new(),
            replaced_start_line: start_line,
            replaced_end_line: end_line,
        });
    }
    if let FindResult::Multiple(count) = find_normalized(content, old_string) {
        anyhow::bail!(
            "old_string found {} times in '{}' (after normalizing trailing whitespace). Provide more surrounding context to make the match unique.",
            count,
            file_path
        );
    }

    // 2. Fallback A: strip line-number prefix from old_string and retry.
    if let Some(stripped) = strip_line_number_prefix(old_string) {
        if let FindResult::One(pos) = find_exact(content, &stripped) {
            let new_content = content.replacen(&stripped, new_string, 1);
            let start_line = content[..pos].matches('\n').count();
            let end_line = start_line + stripped.lines().count();
            return Ok(EditOutcome {
                new_content,
                diff_prefix: String::new(),
                replaced_start_line: start_line,
                replaced_end_line: end_line,
            });
        }
        if let FindResult::One(pos) = find_normalized(content, &stripped) {
            let new_content = replace_normalized(content, &stripped, new_string, pos);
            let start_line = content[..pos].matches('\n').count();
            let end_line = start_line + stripped.lines().count();
            return Ok(EditOutcome {
                new_content,
                diff_prefix: String::new(),
                replaced_start_line: start_line,
                replaced_end_line: end_line,
            });
        }
    }

    // 3. Fallback B: indentation tolerance.
    if let Some((start_idx, end_idx, target_indent)) = try_indent_tolerance(content, old_string) {
        let reindented = reindent_replacement(new_string, &target_indent);
        let new_content =
            apply_line_range_replacement(content, start_idx + 1, end_idx, &reindented);
        return Ok(EditOutcome {
            new_content,
            diff_prefix: String::new(),
            replaced_start_line: start_idx,
            replaced_end_line: end_idx,
        });
    }

    // 4. Fallback C: auto-apply find_closest_match when confidence is high.
    if let Some((start_1, end_1, snippet)) = find_closest_match(content, old_string) {
        let needle_lines: Vec<&str> = old_string.lines().collect();
        let file_lines: Vec<&str> = content.lines().collect();
        let needle_count = needle_lines.len();
        if needle_count >= 3 {
            // Compute scores across all candidate windows. Auto-apply when the
            // top window's score >= 0.8 * needle_count and is unique.
            let candidate_starts = find_candidate_window_starts(content, old_string);
            let mut scored: Vec<(usize, usize)> = Vec::new(); // (score, start_idx)
            for start in candidate_starts {
                let end = (start + needle_count).min(file_lines.len());
                if end - start != needle_count {
                    continue;
                }
                let mut score = 0usize;
                for j in 0..needle_count {
                    if file_lines[start + j].trim() == needle_lines[j].trim() {
                        score += 1;
                    }
                }
                scored.push((score, start));
            }
            scored.sort_by(|a, b| b.0.cmp(&a.0));
            if !scored.is_empty() {
                let top = scored[0];
                let unique = scored.iter().filter(|s| s.0 == top.0).count() == 1;
                let threshold = (needle_count * 4 + 4) / 5; // ceil(0.8 * n)
                if unique && top.0 >= threshold {
                    let start_idx = top.1;
                    let end_idx = (start_idx + needle_count).min(file_lines.len());
                    let window: String = file_lines[start_idx..end_idx].join("\n");
                    let mtype = classify_mismatch(old_string, &window);
                    // Re-indent new_string so it matches the matched window's
                    // leading indent (mirrors the indent-tolerance path).
                    let target_indent = file_lines[start_idx..end_idx]
                        .iter()
                        .find(|l| !l.trim().is_empty())
                        .map(|l| {
                            let n = l.len() - l.trim_start().len();
                            l[..n].to_string()
                        })
                        .unwrap_or_default();
                    let reindented = reindent_replacement(new_string, &target_indent);
                    let new_content = apply_line_range_replacement(
                        content,
                        start_idx + 1,
                        end_idx,
                        &reindented,
                    );
                    let msg = format!(
                        "[harness] Edit applied via fuzzy match (lines {}-{}, score {}/{}, mismatch type: {}). Verify the diff:",
                        start_idx + 1,
                        end_idx,
                        top.0,
                        needle_count,
                        mtype.as_str(),
                    );
                    return Ok(EditOutcome {
                        new_content,
                        diff_prefix: format!("{}\n", msg),
                        replaced_start_line: start_idx,
                        replaced_end_line: end_idx,
                    });
                }
            }
        }
        // Fall through to error path with the closest-match hint.
        let window: String = file_lines
            [start_1.saturating_sub(1)..end_1.min(file_lines.len())]
            .join("\n");
        let mtype = classify_mismatch(old_string, &window);
        let echoed = truncate_args(old_string, 200);
        anyhow::bail!(
            "old_string not found in '{}' (mismatch type: {}). Closest match at lines {}-{}:\n{}\n\n\
             Failed old_string (truncated to 200 chars): {}\n\n\
             Easier path: call edit again with file_path={}, start_line={}, end_line={}, new_string=<your intended replacement>. \
             Line-range mode replaces lines deterministically without needing exact old_string match.",
            file_path,
            mtype.as_str(),
            start_1,
            end_1,
            snippet,
            echoed,
            file_path,
            start_1,
            end_1,
        );
    }

    let echoed = truncate_args(old_string, 200);
    anyhow::bail!(
        "old_string not found in '{}'. No close match could be located.\n\n\
         Failed old_string (truncated to 200 chars): {}\n\n\
         Make sure old_string matches exactly, including whitespace and indentation. \
         Easier path: re-read the file with the `read` tool to obtain the current line numbers, \
         then call edit again in line-range mode with start_line/end_line/new_string.",
        file_path,
        echoed,
    );
}

/// Render a contextual diff for a line-range replacement.
fn render_diff(
    file_lines: &[&str],
    new_file_lines: &[&str],
    start_idx: usize,
    old_end: usize,
    new_end: usize,
) -> String {
    let ctx = 3;
    let ctx_start = start_idx.saturating_sub(ctx);
    let ctx_after_end = (new_end + ctx).min(new_file_lines.len());
    let max_line_num = ctx_after_end.max(old_end);
    let num_width = format!("{}", max_line_num).len().max(3);

    let mut diff = String::new();

    // Context before
    for i in ctx_start..start_idx {
        if i >= file_lines.len() {
            break;
        }
        diff.push_str(&format!(
            " {:>width$}  {}\n",
            i + 1,
            file_lines[i],
            width = num_width
        ));
    }
    // Removed lines (from old file)
    for i in start_idx..old_end {
        if i >= file_lines.len() {
            break;
        }
        diff.push_str(&format!(
            "-{:>width$}  {}\n",
            i + 1,
            file_lines[i],
            width = num_width
        ));
    }
    // Added lines (from new file)
    for i in start_idx..new_end {
        if i >= new_file_lines.len() {
            break;
        }
        diff.push_str(&format!(
            "+{:>width$}  {}\n",
            i + 1,
            new_file_lines[i],
            width = num_width
        ));
    }
    // Context after (from new file)
    for i in new_end..ctx_after_end {
        if i >= new_file_lines.len() {
            break;
        }
        diff.push_str(&format!(
            " {:>width$}  {}\n",
            i + 1,
            new_file_lines[i],
            width = num_width
        ));
    }
    diff
}

/// Write `data` to `path` atomically via a temp file + rename.
/// This prevents data loss if the process crashes mid-write.
fn atomic_write(path: &str, data: &[u8]) -> Result<()> {
    let target = Path::new(path);
    let dir = target
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine parent directory for '{}'", path))?;

    let mut tmp = tempfile::NamedTempFile::new_in(dir)
        .map_err(|e| anyhow::anyhow!("Failed to create temp file in '{}': {}", dir.display(), e))?;
    tmp.write_all(data)
        .map_err(|e| anyhow::anyhow!("Failed to write temp file for '{}': {}", path, e))?;
    tmp.persist(target)
        .map_err(|e| anyhow::anyhow!("Failed to rename temp file to '{}': {}", path, e))?;
    Ok(())
}

pub struct EditTool;

impl Tool for EditTool {
    fn name(&self) -> &str { "edit" }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit".to_string(),
            description: "Edit a file. Two modes: (1) String mode — provide old_string and \
                          new_string to replace an exact unique match. (2) Line-range mode — \
                          provide start_line, end_line, and new_string to replace an inclusive \
                          range of lines. Returns a diff of the change."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace (must be unique in the file). Required for string mode."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "The first line to replace (1-based, inclusive). Used with end_line for line-range mode."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "The last line to replace (1-based, inclusive). Used with start_line for line-range mode."
                    }
                },
                "required": ["file_path", "new_string"]
            }),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<String> {
        let raw_path = required_str(arguments, "file_path")?;
        let file_path = expand_tilde(raw_path);
        let file_path = file_path.as_ref();
        let new_string = required_str(arguments, "new_string")?;

        let start_line_arg = arguments.get("start_line").and_then(|v| v.as_u64());
        let end_line_arg = arguments.get("end_line").and_then(|v| v.as_u64());
        let old_string = optional_str(arguments, "old_string");

        // Decide which mode to use
        let use_line_range =
            start_line_arg.is_some() && end_line_arg.is_some() && old_string.is_none();

        if use_line_range {
            // --- Line-range mode ---
            let start = start_line_arg.unwrap() as usize;
            let end = end_line_arg.unwrap() as usize;

            if start == 0 {
                anyhow::bail!("start_line must be >= 1 (1-based)");
            }
            if end < start {
                anyhow::bail!(
                    "end_line ({}) must be >= start_line ({})",
                    end,
                    start
                );
            }

            let content = fs::read_to_string(file_path)
                .map_err(|e| anyhow::anyhow!("Failed to read '{}': {}", file_path, e))?;

            let file_lines: Vec<&str> = content.lines().collect();
            let total_lines = file_lines.len();

            if start > total_lines {
                anyhow::bail!(
                    "start_line {} is beyond end of file ({} lines)",
                    start,
                    total_lines
                );
            }
            if end > total_lines {
                anyhow::bail!(
                    "end_line {} is beyond end of file ({} lines)",
                    end,
                    total_lines
                );
            }

            let new_content = apply_line_range_replacement(&content, start, end, new_string);
            atomic_write(file_path, new_content.as_bytes())?;

            let new_file_lines: Vec<&str> = new_content.lines().collect();
            let start_idx = start - 1;
            let end_idx = end;
            let new_lines_count = new_string.lines().count();
            let new_end = start_idx + new_lines_count;

            Ok(render_diff(
                &file_lines,
                &new_file_lines,
                start_idx,
                end_idx,
                new_end,
            ))
        } else {
            // --- String mode (with fallbacks) ---
            let old_string = old_string.ok_or_else(|| {
                anyhow::anyhow!(
                    "old_string is required when start_line/end_line are not both provided"
                )
            })?;

            if old_string == new_string {
                anyhow::bail!("old_string and new_string are identical — nothing to change");
            }

            let content = fs::read_to_string(file_path)
                .map_err(|e| anyhow::anyhow!("Failed to read '{}': {}", file_path, e))?;

            let outcome = resolve_string_edit(file_path, &content, old_string, new_string)?;

            atomic_write(file_path, outcome.new_content.as_bytes())?;

            let file_lines: Vec<&str> = content.lines().collect();
            let new_file_lines: Vec<&str> = outcome.new_content.lines().collect();
            let new_lines_count = new_string.lines().count();
            let new_end = outcome.replaced_start_line + new_lines_count;
            let diff = render_diff(
                &file_lines,
                &new_file_lines,
                outcome.replaced_start_line,
                outcome.replaced_end_line,
                new_end,
            );
            Ok(format!("{}{}", outcome.diff_prefix, diff))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn test_exact_match() {
        let content = "fn foo() {\n    let x = 1;\n}\n";
        match find_exact(content, "    let x = 1;") {
            FindResult::One(_) => {}
            _ => panic!("expected one match"),
        }
    }

    #[test]
    fn test_normalized_trailing_whitespace() {
        // File has empty line (just \n), model sends line with a space
        let content = "line1\n\nline3\n";
        let needle = "line1\n \nline3";
        assert!(matches!(find_exact(content, needle), FindResult::None));
        assert!(matches!(find_normalized(content, needle), FindResult::One(_)));
    }

    #[test]
    fn test_replace_normalized_empty_line() {
        let content = "before\n/// doc comment\n\n    let x = 1;\nafter\n";
        let needle = "/// doc comment\n \n    let x = 1;";
        let replacement = "/// doc comment\nfn foo() {\n    let x = 1;";

        let result = replace_normalized(content, needle, replacement, 0);
        assert_eq!(result, "before\n/// doc comment\nfn foo() {\n    let x = 1;\nafter\n");
    }

    #[test]
    fn test_normalize_trailing() {
        assert_eq!(normalize_trailing("a \nb\n  c  "), "a\nb\n  c");
    }

    // -------- (1) smart-edit fallback tests --------

    /// Run `EditTool::execute` against a tempdir-backed file and return the
    /// final on-disk content along with the function's `Ok` value (or `Err`).
    fn run_edit(content_in: &str, args: serde_json::Value) -> (Result<String>, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("foo.txt");
        std::fs::write(&path, content_in).unwrap();

        let mut args = args;
        args.as_object_mut()
            .unwrap()
            .insert("file_path".to_string(), serde_json::Value::String(path.display().to_string()));

        let result = EditTool.execute(&args);

        let mut out = String::new();
        std::fs::File::open(&path).unwrap().read_to_string(&mut out).unwrap();
        (result, out)
    }

    #[test]
    fn test_strip_line_number_prefix_match() {
        let content = "fn foo() {\n    let x = 1;\n    let y = 2;\n}\n";
        let needle = "  41\t    let x = 1;\n  42\t    let y = 2;";
        let (res, after) = run_edit(
            content,
            serde_json::json!({
                "old_string": needle,
                "new_string": "    let x = 10;\n    let y = 20;",
            }),
        );
        assert!(res.is_ok(), "edit should succeed: {:?}", res.err());
        assert!(after.contains("let x = 10;"), "after = {}", after);
        assert!(after.contains("let y = 20;"));
    }

    #[test]
    fn test_strip_line_number_prefix_pipe_form() {
        let needle = "   1 | fn foo() {\n   2 |     let x = 1;";
        let stripped = strip_line_number_prefix(needle).expect("should strip pipe form");
        assert_eq!(stripped, "fn foo() {\n    let x = 1;");
    }

    #[test]
    fn test_strip_line_number_prefix_no_match() {
        // Plain code without prefixes: must not be mangled.
        let s = "fn foo() {\n    let x = 1;\n}";
        assert!(strip_line_number_prefix(s).is_none());
    }

    #[test]
    fn test_indent_tolerance_match() {
        // File uses 4-space indent, needle uses 2-space.
        let content = "fn outer() {\n    fn inner() {\n        let x = 1;\n    }\n}\n";
        let needle = "  fn inner() {\n      let x = 1;\n  }";
        let new_string = "  fn inner() {\n      let x = 99;\n  }";
        let (res, after) = run_edit(
            content,
            serde_json::json!({
                "old_string": needle,
                "new_string": new_string,
            }),
        );
        assert!(res.is_ok(), "indent-tolerant match should succeed: {:?}", res.err());
        // The new content must use the file's original 4-space indent.
        assert!(after.contains("    fn inner() {"), "expected 4-space indent, got: {}", after);
        assert!(after.contains("        let x = 99;"), "expected 8-space body indent, got: {}", after);
        assert!(!after.contains("let x = 1;"), "old line should be gone, got: {}", after);
    }

    #[test]
    fn test_fuzzy_auto_apply_above_threshold() {
        // 5-line content, 5-line needle differing in one line (one token).
        // 4/5 lines match -> 80% threshold is met; auto-apply fires.
        let content = "fn first() {}\nfn second() {}\nfn third() {}\nfn fourth() {}\nfn fifth() {}\n";
        let needle = "fn first() {}\nfn second() {}\nfn THIRD() {}\nfn fourth() {}\nfn fifth() {}";
        let new_string =
            "fn first() {}\nfn second() {}\nfn third_renamed() {}\nfn fourth() {}\nfn fifth() {}";
        let (res, after) = run_edit(
            content,
            serde_json::json!({
                "old_string": needle,
                "new_string": new_string,
            }),
        );
        let ok = res.expect("fuzzy auto-apply should succeed");
        assert!(ok.contains("via fuzzy match"), "message: {}", ok);
        assert!(after.contains("fn third_renamed()"), "after = {}", after);
        assert!(!after.contains("fn third()"));
    }

    #[test]
    fn test_fuzzy_no_auto_apply_below_threshold() {
        // 5-line needle with 4 differences -> only 1/5 line score, below threshold.
        let content = "alpha\nbeta\ngamma\ndelta\nepsilon\n";
        let needle = "alpha\nBETA\nGAMMA\nDELTA\nEPSILON";
        let new_string = "alpha\nb1\nb2\nb3\nb4";
        let (res, _after) = run_edit(
            content,
            serde_json::json!({
                "old_string": needle,
                "new_string": new_string,
            }),
        );
        assert!(res.is_err(), "should not auto-apply below threshold");
    }

    #[test]
    fn test_fuzzy_no_auto_apply_short_needle() {
        // 2-line needle exactly matching: still must NOT auto-apply (too risky).
        // To force the fuzzy path we corrupt one of the two lines slightly.
        let content = "alpha\nbeta\ngamma\n";
        let needle = "alpha\nBETA";
        let new_string = "alpha\nbeta_renamed";
        let (res, _after) = run_edit(
            content,
            serde_json::json!({
                "old_string": needle,
                "new_string": new_string,
            }),
        );
        assert!(res.is_err(), "should not auto-apply on a 2-line needle");
    }

    #[test]
    fn test_error_message_suggests_line_range() {
        let content = "fn foo() {\n    let x = 1;\n}\n";
        // Anchor matches but body is wrong, so we land in the error branch.
        let needle = "fn foo() {\n    let totally_different = 999;\n}";
        let new_string = "fn foo() { /* nothing */ }";
        let (res, _after) = run_edit(
            content,
            serde_json::json!({
                "old_string": needle,
                "new_string": new_string,
            }),
        );
        let err = res.err().expect("should fail");
        let s = format!("{}", err);
        assert!(s.contains("Easier path: call edit again with"), "msg: {}", s);
        assert!(s.contains("start_line="), "msg: {}", s);
        assert!(s.contains("end_line="), "msg: {}", s);
        // Mismatch type is named.
        assert!(s.contains("mismatch type:"), "msg: {}", s);
        // Echoes the failed needle (truncated) so the model can see its input.
        assert!(s.contains("Failed old_string"), "msg: {}", s);
    }

    #[test]
    fn test_classify_mismatch_indentation() {
        let needle = "  fn foo() {\n    let x = 1;\n  }";
        let window = "    fn foo() {\n      let x = 1;\n    }";
        assert_eq!(classify_mismatch(needle, window), MismatchType::Indentation);
    }

    #[test]
    fn test_classify_mismatch_line_number_prefix() {
        let needle = "  41\tfn foo() {\n  42\t    let x = 1;";
        let window = "fn foo() {\n    let x = 1;";
        assert_eq!(classify_mismatch(needle, window), MismatchType::LineNumberPrefix);
    }

    #[test]
    fn test_apply_line_range_replacement_helper() {
        let content = "a\nb\nc\nd\ne\n";
        let result = apply_line_range_replacement(content, 2, 3, "B\nC");
        assert_eq!(result, "a\nB\nC\nd\ne\n");
    }
}
