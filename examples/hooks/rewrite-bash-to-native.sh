#!/bin/bash
# Hook: rewrite bash(ls/find/grep/rg) calls to native glob/grep tools.
#
# Many models default to familiar shell commands (ls, find, grep) instead of
# using the purpose-built glob and grep tools. This hook transparently rewrites
# those bash tool calls into native tool calls, giving the model better output
# formatting and avoiding the need for shell parsing.
#
# Requires: jq
#
# Rewrites:
#   bash("ls src/")           → glob(pattern="src/*")
#   bash("ls -R src/")        → glob(pattern="src/**/*")
#   bash("find . -name *.py") → glob(pattern="**/*.py")
#   bash("grep -r pattern")   → grep(pattern="pattern")
#   bash("rg pattern src/")   → grep(pattern="pattern", path="src/")
#
# Non-matching bash commands pass through unchanged.

set -euo pipefail

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.data.arguments.command // ""')

# Skip if command is empty
[ -z "$COMMAND" ] && exit 0

# Strip leading whitespace
COMMAND=$(echo "$COMMAND" | sed 's/^[[:space:]]*//')

# --- ls / ls -la / ls -R → glob ---
if echo "$COMMAND" | grep -qE '^ls([[:space:]]|$)'; then
    # Extract the path argument (skip flags)
    PATH_ARG=""
    for word in $COMMAND; do
        case "$word" in
            ls) continue ;;
            -*) continue ;;  # skip flags like -la, -R
            *) PATH_ARG="$word"; break ;;
        esac
    done

    # Build glob pattern
    if echo "$COMMAND" | grep -qE -- '-R|-r|--recursive'; then
        # Recursive ls → recursive glob
        if [ -n "$PATH_ARG" ]; then
            PATTERN="${PATH_ARG%/}/**/*"
        else
            PATTERN="**/*"
        fi
    else
        # Non-recursive ls
        if [ -n "$PATH_ARG" ]; then
            PATTERN="${PATH_ARG%/}/*"
        else
            PATTERN="*"
        fi
    fi

    jq -n --arg pattern "$PATTERN" '{action:"rewrite",tool_name:"glob",arguments:{pattern:$pattern}}'
    exit 0
fi

# --- find . -name "pattern" → glob ---
if echo "$COMMAND" | grep -qE '^find[[:space:]]'; then
    # Extract -name argument
    NAME_PAT=$(echo "$COMMAND" | grep -oP "(?<=-name\s['\"])[^'\"]+(?=['\"])" || echo "$COMMAND" | grep -oP '(?<=-name\s)\S+' || true)
    # Extract the search directory
    FIND_DIR=$(echo "$COMMAND" | awk '{print $2}')
    [ "$FIND_DIR" = "." ] && FIND_DIR=""

    if [ -n "$NAME_PAT" ]; then
        if [ -n "$FIND_DIR" ]; then
            PATTERN="${FIND_DIR%/}/**/${NAME_PAT}"
        else
            PATTERN="**/${NAME_PAT}"
        fi
        jq -n --arg pattern "$PATTERN" '{action:"rewrite",tool_name:"glob",arguments:{pattern:$pattern}}'
        exit 0
    fi
fi

# --- grep/rg → grep tool ---
if echo "$COMMAND" | grep -qE '^(grep|rg)[[:space:]]'; then
    # Parse grep/rg command for pattern and path
    TOOL_CMD=$(echo "$COMMAND" | awk '{print $1}')

    # Remove the command name and collect args
    ARGS=$(echo "$COMMAND" | sed "s/^${TOOL_CMD}[[:space:]]*//")

    SEARCH_PATTERN=""
    SEARCH_PATH=""
    SKIP_NEXT=false
    POSITIONALS=()

    for word in $ARGS; do
        if $SKIP_NEXT; then
            SKIP_NEXT=false
            continue
        fi
        case "$word" in
            -r|-R|--recursive|-i|--ignore-case|-n|--line-number|-l|--files-with-matches|-c|--count|-w|--word-regexp|-v|--invert-match|-q|--quiet|-s|--no-messages|-H|--with-filename|-h|--no-filename)
                continue ;;
            -e|-f|-m|--max-count|--include|--exclude|--exclude-dir|-A|-B|-C|--context|-t|--type|-g|--glob)
                SKIP_NEXT=true
                continue ;;
            -*)
                continue ;;  # skip other flags
            *)
                POSITIONALS+=("$word") ;;
        esac
    done

    # First positional is pattern, second (if any) is path
    if [ ${#POSITIONALS[@]} -ge 1 ]; then
        SEARCH_PATTERN="${POSITIONALS[0]}"
        # Remove surrounding quotes if present
        SEARCH_PATTERN=$(echo "$SEARCH_PATTERN" | sed "s/^['\"]//;s/['\"]$//")
    fi
    if [ ${#POSITIONALS[@]} -ge 2 ]; then
        SEARCH_PATH="${POSITIONALS[1]}"
    fi

    if [ -n "$SEARCH_PATTERN" ]; then
        if [ -n "$SEARCH_PATH" ]; then
            jq -n --arg pat "$SEARCH_PATTERN" --arg path "$SEARCH_PATH" \
                '{action:"rewrite",tool_name:"grep",arguments:{pattern:$pat,path:$path}}'
        else
            jq -n --arg pat "$SEARCH_PATTERN" \
                '{action:"rewrite",tool_name:"grep",arguments:{pattern:$pat}}'
        fi
        exit 0
    fi
fi

# No match — passthrough (empty output = proceed)
