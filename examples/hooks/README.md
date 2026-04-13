# Hook: Rewrite bash to native tools

## Problem

Many models — especially smaller ones — default to familiar shell commands instead of using ollama-code's purpose-built tools. For example, a model will call `bash("ls -R src/")` instead of `glob("src/**/*")`, or `bash("grep -r pattern src/")` instead of `grep("src/", "pattern")`.

This matters because the native tools:
- Return structured, consistent output that's easier for the model to parse
- Don't waste tokens on noisy shell output (permission bits, dates, colors)
- Don't require confirmation (bash always prompts, glob/grep don't)
- Integrate with features like dynamic tool scoping and exploration tracking

## Solution

This hook uses the `"rewrite"` action (added in v0.9.0) to transparently intercept bash tool calls and redirect them to native tools. The model doesn't know the rewrite happened — it thinks it called bash and got the result back.

### Rewrites

| Model calls | Hook rewrites to |
|---|---|
| `bash("ls src/")` | `glob(pattern="src/*")` |
| `bash("ls -R src/")` | `glob(pattern="src/**/*")` |
| `bash("ls -la")` | `glob(pattern="*")` |
| `bash("find . -name '*.py'")` | `glob(pattern="**/*.py")` |
| `bash("grep -r pattern src/")` | `grep(pattern="pattern", path="src/")` |
| `bash("rg TODO")` | `grep(pattern="TODO")` |
| `bash("python3 test.py")` | *(passes through unchanged)* |

## Setup

### 1. Copy files to your config directory

```bash
mkdir -p ~/.config/ollama-code/hooks
cp rewrite-bash-to-native.sh ~/.config/ollama-code/hooks/
cp hooks.toml ~/.config/ollama-code/hooks.toml
```

### 2. Make the script executable

```bash
chmod +x ~/.config/ollama-code/hooks/rewrite-bash-to-native.sh
```

### 3. Verify

```bash
# Should output a rewrite to glob:
echo '{"hook":"pre_tool_execute","data":{"tool_name":"bash","arguments":{"command":"ls -R src/"}}}' \
  | ~/.config/ollama-code/hooks/rewrite-bash-to-native.sh
```

Expected output:
```json
{"action":"rewrite","tool_name":"glob","arguments":{"pattern":"src/**/*"}}
```

## Requirements

- `jq` must be installed (`sudo pacman -S jq` / `brew install jq` / `apt install jq`)

## How it works

The hook is configured in `hooks.toml` as a `pre_tool_execute` hook that fires only for `bash` tool calls where the command argument matches `ls`, `find`, `grep`, or `rg`. The `if_args` regex filter ensures other bash commands (running scripts, installing packages, git, etc.) pass through untouched.

The script receives the tool call as JSON on stdin, parses the bash command, and outputs a JSON response with `action: "rewrite"`, the target `tool_name`, and translated `arguments`. If the command doesn't match any rewrite rule, the script produces no output (which means "proceed as normal").

### Hook protocol

Input (on stdin):
```json
{
  "hook": "pre_tool_execute",
  "data": {
    "tool_name": "bash",
    "arguments": {"command": "ls -R src/"}
  }
}
```

Output (on stdout):
```json
{
  "action": "rewrite",
  "tool_name": "glob",
  "arguments": {"pattern": "src/**/*"}
}
```

The `"rewrite"` action changes both the tool name and arguments before execution. This is different from `"modify"` (which only changes arguments) and `"deny"` (which blocks execution entirely).

## Limitations

- The bash command parsing is heuristic — complex pipelines or commands with unusual quoting won't be detected. These fall through to regular bash execution, which is the safe default.
- `find` rewriting only handles the `-name` flag. More complex find expressions (e.g., `-type f -mtime -1`) pass through to bash.
- The hook adds a small latency per bash call (~5ms for jq parsing). This is negligible compared to model inference time.
