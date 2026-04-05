# Hooks and Event-Driven Automation

**Previous**: [04. Permission Modes and Security](./04_Permission_Modes.md) | **Next**: [06. Skills and Slash Commands](./06_Skills_and_Slash_Commands.md)

---

Hooks are shell commands that Claude Code executes automatically in response to specific events. When Claude edits a file, you can have a hook auto-format it with Prettier or Black. When Claude finishes a task, you can trigger a notification. When Claude is about to run a dangerous command, a hook can intercept and block it. Hooks transform Claude Code from an interactive assistant into an event-driven automation pipeline — and they do so deterministically, without relying on the model to remember instructions.

**Difficulty**: ⭐⭐

**Prerequisites**:
- [03. CLAUDE.md and Project Setup](./03_CLAUDE_md_and_Project_Setup.md)
- [04. Permission Modes and Security](./04_Permission_Modes.md)
- Shell scripting basics (see **Shell_Script** topic)
- Familiarity with JSON configuration files

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand the four hook types and when each fires
2. Write hook configurations in JSON with matchers and commands
3. Use environment variables to access hook context data
4. Build practical hooks for formatting, linting, testing, and notifications
5. Debug hooks when they fail or produce unexpected results
6. Distinguish between hooks (deterministic) and CLAUDE.md instructions (suggestive)

---

## Table of Contents

1. [What Are Hooks?](#1-what-are-hooks)
2. [Hook Types](#2-hook-types)
3. [Configuration Format](#3-configuration-format)
4. [Hook Matchers](#4-hook-matchers)
5. [Environment Variables](#5-environment-variables)
6. [Practical Examples](#6-practical-examples)
7. [Hook Execution Flow](#7-hook-execution-flow)
8. [Error Handling and Debugging](#8-error-handling-and-debugging)
9. [Hooks vs CLAUDE.md Instructions](#9-hooks-vs-claudemd-instructions)
10. [Advanced Patterns](#10-advanced-patterns)
11. [Exercises](#11-exercises)
12. [Next Steps](#12-next-steps)

---

## 1. What Are Hooks?

A **hook** is a shell command that Claude Code runs when a specific event occurs. Hooks are:

- **Deterministic**: They always run when the event fires — unlike CLAUDE.md instructions, which the model may or may not follow
- **Configurable**: Defined in settings JSON files, not in natural language
- **Scoped**: Can target specific tools, file paths, or command patterns
- **Non-blocking or blocking**: Some hooks can prevent Claude's action from proceeding

Think of hooks as Git hooks or CI/CD pipeline triggers, but for Claude Code's tool use.

```
Without hooks:
  Claude edits file → Done (file may not be formatted)

With auto-format hook:
  Claude edits file → Hook: prettier --write file → Done (file is formatted)
```

---

## 2. Hook Types

Claude Code supports four hook types, each triggered at a different point in the tool execution lifecycle.

### Hook Type Overview

| Hook Type | When It Fires | Can Block? | Common Uses |
|-----------|--------------|------------|-------------|
| **PreToolUse** | Before a tool executes | Yes | Validation, blocking dangerous commands |
| **PostToolUse** | After a tool executes | No | Formatting, linting, notifications |
| **Notification** | When Claude sends a notification | No | Custom notification routing |
| **Stop** | When Claude finishes its turn | No | Final checks, summaries, test runs |

### Lifecycle Diagram

```
                    User sends a message
                           │
                           ▼
                    Claude decides to use a tool
                           │
                    ┌──────┴──────┐
                    │ PreToolUse  │ ← Can block the tool
                    │   hooks     │
                    └──────┬──────┘
                           │
                    Tool is blocked? ──Yes──▶ Claude adjusts
                           │
                          No
                           │
                    ┌──────┴──────┐
                    │  Tool       │
                    │  executes   │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │ PostToolUse │ ← Format, lint, notify
                    │   hooks     │
                    └──────┬──────┘
                           │
                    Claude continues or finishes
                           │
                    ┌──────┴──────┐
                    │   Stop      │ ← Final checks
                    │   hooks     │
                    └─────────────┘
```

---

## 3. Configuration Format

Hooks are configured in settings JSON files (`.claude/settings.json`, `~/.claude/settings.json`, or `.claude/settings.local.json`).

### Basic Structure

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "<pattern>",
        "command": "<shell command>"
      }
    ],
    "PostToolUse": [
      {
        "matcher": "<pattern>",
        "command": "<shell command>"
      }
    ],
    "Notification": [
      {
        "command": "<shell command>"
      }
    ],
    "Stop": [
      {
        "command": "<shell command>"
      }
    ]
  }
}
```

### Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `matcher` | No* | Pattern to match tool name or file path. If omitted, hook fires for all events of that type. |
| `command` | Yes | Shell command to execute. Receives context via environment variables. |

*Matcher is not applicable for `Notification` and `Stop` hooks.

### Minimal Example

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "echo 'A file was edited: $CLAUDE_FILE_PATH'"
      }
    ]
  }
}
```

---

## 4. Hook Matchers

Matchers determine which tool invocations trigger the hook. They support tool names and file path patterns.

### Tool Name Matchers

| Matcher | Matches |
|---------|---------|
| `"Edit"` | Any file edit |
| `"Write"` | Any file write (new file creation) |
| `"Bash"` | Any bash command execution |
| `"Read"` | Any file read |
| `"Glob"` | Any file search |
| `"Grep"` | Any content search |
| `"NotebookEdit"` | Any Jupyter notebook edit |

### Combined Matchers

You can define multiple hooks for the same event type:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "npx prettier --write $CLAUDE_FILE_PATH"
      },
      {
        "matcher": "Write",
        "command": "npx prettier --write $CLAUDE_FILE_PATH"
      },
      {
        "matcher": "Bash",
        "command": "echo 'Command executed: $CLAUDE_BASH_COMMAND'"
      }
    ]
  }
}
```

### File Path Matchers

Some hooks can match on file paths using glob patterns:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit:*.py",
        "command": "black $CLAUDE_FILE_PATH"
      },
      {
        "matcher": "Edit:*.ts",
        "command": "npx prettier --write $CLAUDE_FILE_PATH"
      },
      {
        "matcher": "Edit:*.go",
        "command": "gofmt -w $CLAUDE_FILE_PATH"
      }
    ]
  }
}
```

---

## 5. Environment Variables

When a hook runs, Claude Code sets environment variables that provide context about the event.

### Available Environment Variables

| Variable | Available In | Description |
|----------|-------------|-------------|
| `CLAUDE_FILE_PATH` | PostToolUse (Edit, Write) | Absolute path of the edited/written file |
| `CLAUDE_BASH_COMMAND` | PreToolUse, PostToolUse (Bash) | The command that will be or was executed |
| `CLAUDE_TOOL_NAME` | All hooks | Name of the tool (Edit, Bash, etc.) |
| `CLAUDE_EXIT_CODE` | PostToolUse (Bash) | Exit code of the bash command |
| `CLAUDE_NOTIFICATION` | Notification | The notification message text |
| `CLAUDE_PROJECT_DIR` | All hooks | Absolute path to the project root |

### Using Environment Variables in Commands

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "echo \"File edited: $CLAUDE_FILE_PATH in project $CLAUDE_PROJECT_DIR\""
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "echo \"About to run: $CLAUDE_BASH_COMMAND\""
      }
    ]
  }
}
```

---

## 6. Practical Examples

### Example 1: Auto-Format After Edits

The most common hook: automatically format files after Claude edits them.

**For JavaScript/TypeScript projects (Prettier)**:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "npx prettier --write \"$CLAUDE_FILE_PATH\" 2>/dev/null || true"
      },
      {
        "matcher": "Write",
        "command": "npx prettier --write \"$CLAUDE_FILE_PATH\" 2>/dev/null || true"
      }
    ]
  }
}
```

**For Python projects (Black + isort)**:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit:*.py",
        "command": "black \"$CLAUDE_FILE_PATH\" && isort \"$CLAUDE_FILE_PATH\""
      }
    ]
  }
}
```

**For Go projects (gofmt)**:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit:*.go",
        "command": "gofmt -w \"$CLAUDE_FILE_PATH\""
      }
    ]
  }
}
```

### Example 2: Run Linter After Edits

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit:*.py",
        "command": "ruff check \"$CLAUDE_FILE_PATH\" --fix --quiet"
      },
      {
        "matcher": "Edit:*.ts",
        "command": "npx eslint \"$CLAUDE_FILE_PATH\" --fix --quiet"
      }
    ]
  }
}
```

### Example 3: Run Tests After Code Changes

Run the test suite whenever a source file is edited:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit:src/**",
        "command": "npm test --silent 2>&1 | tail -5"
      }
    ]
  }
}
```

> **Note**: Be careful with test hooks on every edit — they can slow down the workflow significantly. Consider running tests only on the Stop event instead.

### Example 4: Custom Notifications

Send desktop notifications when Claude finishes a task:

**macOS**:

```json
{
  "hooks": {
    "Stop": [
      {
        "command": "osascript -e 'display notification \"Claude Code has finished the task\" with title \"Claude Code\"'"
      }
    ],
    "Notification": [
      {
        "command": "osascript -e \"display notification \\\"$CLAUDE_NOTIFICATION\\\" with title \\\"Claude Code\\\"\""
      }
    ]
  }
}
```

**Linux (notify-send)**:

```json
{
  "hooks": {
    "Stop": [
      {
        "command": "notify-send 'Claude Code' 'Task completed'"
      }
    ]
  }
}
```

### Example 5: Block Dangerous Commands

Use PreToolUse hooks to intercept and block dangerous commands:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "case \"$CLAUDE_BASH_COMMAND\" in *'rm -rf /'*|*'dd if='*|*'mkfs'*|*': >'*) echo 'BLOCKED: Dangerous command detected' >&2; exit 1;; esac"
      }
    ]
  }
}
```

When a PreToolUse hook exits with a non-zero status, the tool call is blocked and Claude receives the hook's stderr output as an error message.

### Example 6: Git Pre-Commit Checks

Run checks before Claude makes a commit:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "if echo \"$CLAUDE_BASH_COMMAND\" | grep -q 'git commit'; then npm test --silent || (echo 'Tests must pass before committing' >&2; exit 1); fi"
      }
    ]
  }
}
```

### Example 7: Log All Actions

Create an audit trail of everything Claude does:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "command": "echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) | $CLAUDE_TOOL_NAME | $CLAUDE_FILE_PATH $CLAUDE_BASH_COMMAND\" >> /tmp/claude-audit.log"
      }
    ]
  }
}
```

---

## 7. Hook Execution Flow

Understanding the precise execution flow helps you write correct hooks and debug issues.

### Execution Order

When multiple hooks match the same event, they execute in the order they are defined in the configuration:

```json
{
  "hooks": {
    "PostToolUse": [
      { "matcher": "Edit:*.py", "command": "isort $CLAUDE_FILE_PATH" },
      { "matcher": "Edit:*.py", "command": "black $CLAUDE_FILE_PATH" },
      { "matcher": "Edit:*.py", "command": "ruff check $CLAUDE_FILE_PATH" }
    ]
  }
}
```

Execution order for a `.py` file edit:
1. isort (sorts imports)
2. black (formats code)
3. ruff (checks for errors)

### Timing

```
Hook timing:

PreToolUse:   Runs BEFORE the tool. If it exits non-zero, the tool is blocked.
              Claude sees the hook's stderr as an error message.

PostToolUse:  Runs AFTER the tool completes. Cannot undo the tool's action.
              Output is available to Claude as context.

Notification: Runs when Claude produces a notification message.
              Does not affect Claude's behavior.

Stop:         Runs when Claude finishes its response turn.
              Output is shown to Claude on the next turn.
```

### Settings Merging for Hooks

Hooks from different settings files are merged (not overridden):

```
Global hooks (~/.claude/settings.json):
  PostToolUse: [format_hook]

Project hooks (.claude/settings.json):
  PostToolUse: [lint_hook]

Local hooks (.claude/settings.local.json):
  PostToolUse: [notify_hook]

Effective hooks:
  PostToolUse: [format_hook, lint_hook, notify_hook]
```

---

## 8. Error Handling and Debugging

### Hook Failures

When a hook command fails (exits with non-zero status):

| Hook Type | Behavior on Failure |
|-----------|-------------------|
| **PreToolUse** | Tool call is **blocked**; Claude receives error message |
| **PostToolUse** | Error is logged; Claude is informed but the edit stands |
| **Notification** | Error is logged silently |
| **Stop** | Error is logged; output shown to Claude on next turn |

### Common Errors

**Command not found**:

```json
// Problem: prettier is not installed globally
{ "command": "prettier --write $CLAUDE_FILE_PATH" }

// Fix: use npx or full path
{ "command": "npx prettier --write $CLAUDE_FILE_PATH" }
// or
{ "command": "./node_modules/.bin/prettier --write $CLAUDE_FILE_PATH" }
```

**Missing quotes around file paths**:

```json
// Problem: breaks on paths with spaces
{ "command": "black $CLAUDE_FILE_PATH" }

// Fix: always quote the variable
{ "command": "black \"$CLAUDE_FILE_PATH\"" }
```

**Hook runs on wrong file types**:

```json
// Problem: prettier fails on binary files
{ "matcher": "Edit", "command": "npx prettier --write $CLAUDE_FILE_PATH" }

// Fix: match only the file types you want
{ "matcher": "Edit:*.{ts,tsx,js,jsx,json,css,md}", "command": "npx prettier --write $CLAUDE_FILE_PATH" }
```

### Debugging Techniques

**1. Add echo statements**:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "echo \"DEBUG: Editing $CLAUDE_FILE_PATH with tool $CLAUDE_TOOL_NAME\" && npx prettier --write \"$CLAUDE_FILE_PATH\""
      }
    ]
  }
}
```

**2. Log to a file**:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "command": "echo \"$(date): $CLAUDE_TOOL_NAME $CLAUDE_FILE_PATH\" >> /tmp/claude-hooks.log"
      }
    ]
  }
}
```

**3. Suppress errors gracefully**:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "npx prettier --write \"$CLAUDE_FILE_PATH\" 2>/dev/null || true"
      }
    ]
  }
}
```

The `|| true` ensures the hook does not report a failure even if prettier encounters an error (e.g., unsupported file type).

---

## 9. Hooks vs CLAUDE.md Instructions

This distinction is critical for effective Claude Code configuration. Hooks and CLAUDE.md serve different roles and should be used for different purposes.

### Comparison Table

| Aspect | Hooks | CLAUDE.md |
|--------|-------|-----------|
| **Nature** | Deterministic automation | Natural language suggestions |
| **Enforcement** | Always executes when triggered | Model may or may not follow |
| **Scope** | Specific tool events | General project context |
| **Language** | Shell commands (JSON config) | Markdown (natural language) |
| **Flexibility** | Rigid, exact behavior | Adaptable to context |
| **Examples** | Format code, run linters | Coding style preferences |

### When to Use Hooks

```
Use hooks for things that MUST happen:
  ✓ Code formatting (prettier, black, gofmt)
  ✓ Import sorting (isort, organize-imports)
  ✓ Blocking dangerous commands
  ✓ Audit logging
  ✓ Desktop notifications
  ✓ Pre-commit validation
```

### When to Use CLAUDE.md

```
Use CLAUDE.md for things that SHOULD happen:
  ✓ Coding style preferences
  ✓ Architecture guidelines
  ✓ Naming conventions
  ✓ Test writing patterns
  ✓ API design conventions
  ✓ Documentation standards
```

### Combined Strategy

The best approach uses both:

```markdown
# CLAUDE.md
## Code Style
- Use 4-space indentation
- Function names in snake_case
- Always add type hints to function signatures
```

```json
// .claude/settings.json — hooks
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit:*.py",
        "command": "black --line-length 100 \"$CLAUDE_FILE_PATH\" && isort \"$CLAUDE_FILE_PATH\""
      }
    ]
  }
}
```

The CLAUDE.md tells Claude **what** style to write in. The hook **guarantees** the output conforms, even if Claude's initial output is slightly off.

---

## 10. Advanced Patterns

### Conditional Hooks

Execute different commands based on the file or project state:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "case \"$CLAUDE_FILE_PATH\" in *.py) black \"$CLAUDE_FILE_PATH\";; *.ts|*.tsx) npx prettier --write \"$CLAUDE_FILE_PATH\";; *.go) gofmt -w \"$CLAUDE_FILE_PATH\";; esac"
      }
    ]
  }
}
```

### Hook Scripts

For complex hooks, use external scripts instead of inline commands:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": ".claude/hooks/post-edit.sh"
      }
    ]
  }
}
```

`.claude/hooks/post-edit.sh`:

```bash
#!/bin/bash
set -euo pipefail

FILE="$CLAUDE_FILE_PATH"
EXT="${FILE##*.}"

case "$EXT" in
    py)
        echo "Formatting Python: $FILE"
        black "$FILE" 2>/dev/null
        isort "$FILE" 2>/dev/null
        ruff check "$FILE" --fix --quiet 2>/dev/null || true
        ;;
    ts|tsx|js|jsx)
        echo "Formatting TypeScript/JavaScript: $FILE"
        npx prettier --write "$FILE" 2>/dev/null
        npx eslint --fix "$FILE" 2>/dev/null || true
        ;;
    go)
        echo "Formatting Go: $FILE"
        gofmt -w "$FILE"
        ;;
    rs)
        echo "Formatting Rust: $FILE"
        rustfmt "$FILE" 2>/dev/null || true
        ;;
    *)
        echo "No formatter configured for .$EXT files"
        ;;
esac
```

Make it executable:

```bash
chmod +x .claude/hooks/post-edit.sh
```

### Stop Hook for Final Validation

Run a comprehensive check after Claude finishes:

```json
{
  "hooks": {
    "Stop": [
      {
        "command": ".claude/hooks/final-check.sh"
      }
    ]
  }
}
```

`.claude/hooks/final-check.sh`:

```bash
#!/bin/bash

echo "=== Final Validation ==="

# Check for uncommitted changes
if ! git diff --quiet; then
    echo "WARNING: Uncommitted changes detected"
    git diff --stat
fi

# Run quick linter check
if command -v npx &>/dev/null && [ -f "package.json" ]; then
    echo "Running lint check..."
    npx eslint src/ --quiet 2>/dev/null && echo "Lint: PASS" || echo "Lint: ISSUES FOUND"
fi

# Check test status
if [ -f "package.json" ] && grep -q '"test"' package.json; then
    echo "Running tests..."
    npm test --silent 2>&1 | tail -3
fi
```

---

## 11. Exercises

### Exercise 1: Basic Hook Configuration

Create a `.claude/settings.json` with hooks that:

1. Auto-format Python files with `black` after editing
2. Auto-format JavaScript files with `prettier` after editing
3. Log all bash commands Claude runs to `/tmp/claude-commands.log`
4. Send a macOS notification when Claude finishes a task

### Exercise 2: Pre-Commit Validation Hook

Write a PreToolUse hook that:

1. Detects when Claude is about to run `git commit`
2. Runs `npm test` before allowing the commit
3. Blocks the commit if tests fail
4. Allows the commit if tests pass

### Exercise 3: Multi-Language Format Script

Create a `.claude/hooks/format.sh` script that:

1. Detects the file extension from `$CLAUDE_FILE_PATH`
2. Runs the appropriate formatter (black, prettier, gofmt, rustfmt)
3. Handles missing formatters gracefully (no error if formatter not installed)
4. Logs what it did to `/tmp/claude-format.log`

Configure it as a PostToolUse hook for both Edit and Write events.

### Exercise 4: Debugging a Broken Hook

The following hook configuration has three bugs. Identify and fix them:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "prettier --write $CLAUDE_FILE_PATH"
      },
      {
        "matcher": "Bash",
        "command": "echo 'Ran: $CLAUDE_BASH_COMMAND' >> ~/claude.log"
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Edit",
        "command": "if [ ! -f $CLAUDE_FILE_PATH ]; then echo File not found; exit 1; fi"
      }
    ]
  }
}
```

---

## 12. Next Steps

Hooks give you deterministic automation for Claude Code's tool use. But automation is only part of the picture — you also need reusable **instructions** that Claude follows for specific tasks. The next lesson covers **Skills and Slash Commands** — a system for packaging custom instructions, workflows, and conventions into reusable units that you or your team can invoke with a single command.

**Next**: [06. Skills and Slash Commands](./06_Skills_and_Slash_Commands.md)
