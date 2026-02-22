# Troubleshooting and Debugging

**Previous**: [21. Best Practices and Patterns](./21_Best_Practices.md)

---

Even with proper setup and best practices, issues will arise when working with Claude Code. This lesson provides a systematic approach to diagnosing and resolving the most common problems -- from permission errors and hook failures to context window limits, MCP connection issues, and API errors. Use this as a reference guide when something goes wrong.

**Difficulty**: ⭐⭐

**Prerequisites**:
- Working Claude Code installation ([Lesson 2](./02_Claude_Code_Getting_Started.md))
- Understanding of permission modes ([Lesson 4](./04_Permission_Modes.md))
- Familiarity with hooks ([Lesson 5](./05_Hooks.md))
- Basic knowledge of MCP ([Lesson 12](./12_Model_Context_Protocol.md))
- Understanding of API fundamentals ([Lesson 15](./15_Claude_API_Fundamentals.md))

**Learning Objectives**:
- Diagnose and resolve permission errors systematically
- Debug hook configuration and execution failures
- Manage context window limitations effectively
- Troubleshoot MCP server connection problems
- Handle API errors with appropriate retry strategies
- Identify and resolve performance issues
- Use the `/doctor` command for built-in diagnostics
- Know where to find help when self-diagnosis is not sufficient

---

## Table of Contents

1. [Permission Errors](#1-permission-errors)
2. [Hook Failures](#2-hook-failures)
3. [Context Window Issues](#3-context-window-issues)
4. [MCP Connection Problems](#4-mcp-connection-problems)
5. [API Errors](#5-api-errors)
6. [Performance Issues](#6-performance-issues)
7. [Tool Execution Problems](#7-tool-execution-problems)
8. [The /doctor Command](#8-the-doctor-command)
9. [Where to Get Help](#9-where-to-get-help)
10. [Troubleshooting Decision Tree](#10-troubleshooting-decision-tree)
11. [Exercises](#11-exercises)

---

## 1. Permission Errors

Permission errors are the most common issue when starting with Claude Code. They occur when Claude tries to use a tool that the current permission mode does not allow.

### 1.1 "Permission denied" for File Operations

**Symptom**: Claude reports it cannot read, edit, or create a file.

```
Error: Permission denied — cannot write to /path/to/file.py
```

**Diagnosis checklist:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Permission Denied — Diagnosis Steps                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Check the permission mode                                       │
│     $ claude config get permission_mode                             │
│     → If "plan-only", Claude cannot edit files                     │
│                                                                     │
│  2. Check allow/deny rules                                          │
│     $ cat .claude/settings.json                                     │
│     → Look for deny rules that match the file path                 │
│                                                                     │
│  3. Check file system permissions                                   │
│     $ ls -la /path/to/file.py                                      │
│     → The file might be read-only at the OS level                  │
│                                                                     │
│  4. Check if the file is in a restricted directory                  │
│     → Some directories are blocked by default (e.g., .git/)       │
│                                                                     │
│  5. Check the settings hierarchy                                    │
│     ~/.claude/settings.json      (global — enterprise)             │
│     .claude/settings.json        (project — checked into git)      │
│     .claude/local_settings.json  (local — not in git)              │
│     → A higher-level deny overrides lower-level allow              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Solutions:**

```json
// .claude/settings.json — Adding allow rules
{
  "permissions": {
    "allow": [
      "Edit:*",           // Allow editing all files
      "Write:src/**",     // Allow writing only in src/
      "Bash:npm test",    // Allow running npm test
      "Bash:npm run *"    // Allow npm run commands
    ],
    "deny": [
      "Edit:.env*",       // Never edit env files
      "Write:.git/**",    // Never write to .git/
      "Bash:rm -rf *"     // Never allow recursive delete
    ]
  }
}
```

```bash
# Fix OS-level file permissions
chmod 644 /path/to/file.py

# Fix directory permissions
chmod 755 /path/to/directory/
```

### 1.2 "Tool not allowed in current permission mode"

**Symptom**: Claude says a tool is not available in the current mode.

```
The Bash tool is not allowed in plan-only mode.
```

**Root cause**: The permission mode restricts which tools are available.

```
┌───────────────────────────────────────────────────────────┐
│  Permission Mode → Available Tools                         │
├────────────────┬──────────────────────────────────────────┤
│ plan-only      │ Read, Glob, Grep only (no modifications)│
│ default        │ All tools (with approval prompt)         │
│ auto-accept    │ All tools matching allow rules (no prompt│
│                │ for allowed, prompt for others)           │
│ bypass         │ All tools (no prompts at all)            │
└────────────────┴──────────────────────────────────────────┘
```

**Solution**: Switch to an appropriate permission mode:

```bash
# Check current mode
claude config get permission_mode

# Switch mode for current session
# (Use the settings menu within Claude Code)
# Or set it in settings:
```

```json
// .claude/local_settings.json
{
  "permission_mode": "default"
}
```

### 1.3 Settings Hierarchy Conflicts

When allow/deny rules seem to be ignored, check for conflicts across the settings hierarchy:

```bash
# Check all settings files that might affect permissions
cat ~/.claude/settings.json 2>/dev/null        # Enterprise/global
cat .claude/settings.json 2>/dev/null           # Project (shared)
cat .claude/local_settings.json 2>/dev/null     # Local (personal)
```

**Resolution priority**: Enterprise > Project > Local. If the enterprise settings deny a tool, project and local settings cannot override that denial.

---

## 2. Hook Failures

Hooks run external commands at specific points in Claude Code's lifecycle. When they fail, they can block Claude from completing tasks.

### 2.1 Hook Command Not Found

**Symptom:**

```
Hook error: command not found: my-lint-script
```

**Diagnosis:**

```bash
# Check if the command exists
which my-lint-script

# Check if it's a script in the project
ls -la .claude/scripts/my-lint-script

# Check if it has execute permissions
file .claude/scripts/my-lint-script
```

**Common causes and fixes:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Hook "Command Not Found" — Common Causes                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Script not in PATH                                              │
│     Fix: Use absolute path or ./relative/path                      │
│     "command": "/usr/local/bin/eslint"                              │
│     "command": "./.claude/scripts/lint.sh"                          │
│                                                                     │
│  2. Missing shebang line                                            │
│     Fix: Add #!/bin/bash or #!/usr/bin/env python3                 │
│                                                                     │
│  3. No execute permission                                           │
│     Fix: chmod +x .claude/scripts/lint.sh                          │
│                                                                     │
│  4. Wrong shell (bash vs zsh)                                       │
│     Fix: Be explicit about the shell in the shebang                │
│     #!/bin/bash (not #!/bin/sh on macOS where sh != bash)          │
│                                                                     │
│  5. Node/Python not in PATH in hook context                        │
│     Fix: Use full path: /usr/local/bin/node                        │
│     Or: /usr/bin/env node                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Hook Timeout

**Symptom**: Claude Code hangs, then reports a hook timeout.

```
Hook timed out after 10000ms: pre-commit-check
```

**Diagnosis and fixes:**

```bash
# Test the hook command manually and time it
time ./.claude/scripts/pre-commit-check.sh

# If it's slow, identify the bottleneck
# Common culprits:
# - Running full test suite in a hook (should be a subset)
# - Network calls (API checks, package downloads)
# - Large file processing
```

**Solutions:**

```json
// Increase timeout in hook configuration
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "./.claude/scripts/check.sh",
        "timeout": 30000  // 30 seconds (default is 10 seconds)
      }
    ]
  }
}
```

```bash
# Make the hook faster
# Instead of running ALL tests:
pytest tests/  # Slow (30 seconds)

# Run only fast, relevant tests:
pytest tests/unit/ -x --timeout=5  # Fast (2 seconds)
```

### 2.3 Hook Returning Errors

**Symptom**: Hook runs but exits with a non-zero status, blocking Claude.

```
Hook failed with exit code 1: lint-check
Output: src/utils.py:15:1: E302 expected 2 blank lines, found 1
```

**Diagnosis:**

```bash
# Run the hook command manually to see the full output
./.claude/scripts/lint-check.sh src/utils.py
echo $?  # Check exit code
```

**Key insight**: Hooks that exit with non-zero status will be reported to Claude, which can then fix the issue and retry. This is actually the intended behavior for many hooks (like linting). However, if the hook is reporting false positives, you need to fix the hook itself.

```bash
# Example: lint hook that's too strict
# Before (fails on minor style issues):
#!/bin/bash
flake8 "$1" --max-line-length=79

# After (more reasonable for AI-generated code):
#!/bin/bash
flake8 "$1" --max-line-length=100 --ignore=E501,W503
```

### 2.4 Debugging Hooks Step by Step

```bash
# Step 1: Check hook configuration
cat .claude/settings.json | python3 -m json.tool

# Step 2: Identify which hook is failing
# Add debug output to your hook script:
#!/bin/bash
echo "DEBUG: Hook started" >&2
echo "DEBUG: Args = $@" >&2
echo "DEBUG: Working directory = $(pwd)" >&2
echo "DEBUG: PATH = $PATH" >&2

# Run your actual command
eslint "$1"
EXIT_CODE=$?

echo "DEBUG: Exit code = $EXIT_CODE" >&2
exit $EXIT_CODE

# Step 3: Check environment variables available to hooks
# Hooks receive context via environment variables:
# $CLAUDE_FILE_PATH — the file being operated on
# $CLAUDE_TOOL_NAME — the tool being used
# $CLAUDE_SESSION_ID — current session identifier
```

---

## 3. Context Window Issues

### 3.1 "Context too long" Errors

**Symptom**: Claude reports that the conversation exceeds the context window limit.

```
Error: Total token count exceeds model context window (200K tokens).
```

**Immediate solutions:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Context Window Overflow — Quick Fixes                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. /compact                                                        │
│     Compresses conversation history while preserving key context   │
│     Best when you need to continue the current task                │
│                                                                     │
│  2. Start a new session                                             │
│     Fresh 200K context window                                       │
│     Best when switching to a different task                        │
│                                                                     │
│  3. Use subagents                                                   │
│     Each subagent gets its own context window                      │
│     Best for independent research or parallel tasks                │
│                                                                     │
│  4. Reduce CLAUDE.md size                                           │
│     Remove non-essential information                               │
│     Link to files instead of including content inline              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Symptoms of Context Degradation

Before hitting a hard limit, context quality degrades gradually:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Signs of Context Degradation (in order of appearance)             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Early warning signs:                                               │
│  ⚠ Claude re-reads a file it already read earlier in the session  │
│  ⚠ Claude asks a question you already answered                    │
│  ⚠ Responses reference outdated information from early in session │
│                                                                     │
│  Moderate degradation:                                              │
│  ⚠ Claude forgets a decision made 20+ messages ago                │
│  ⚠ Generated code contradicts earlier patterns                    │
│  ⚠ Claude loses track of the overall task structure               │
│                                                                     │
│  Severe degradation:                                                │
│  ⚠ Claude cannot follow multi-step instructions                   │
│  ⚠ Responses become generic rather than project-specific          │
│  ⚠ /compact no longer helps — session is too far gone             │
│                                                                     │
│  Action: Use /compact at early warning signs.                      │
│  Start a new session at moderate degradation.                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Prevention Strategies

```python
# Strategy 1: Keep prompts focused
# Instead of including your entire codebase as context,
# reference specific files:

# BAD: "Here's my entire project: [50,000 tokens of code]"
# GOOD: "The relevant files are src/auth/service.py and src/auth/middleware.py"

# Strategy 2: Use CLAUDE.md wisely
# CLAUDE.md should be < 200 lines
# Include pointers, not content

# Strategy 3: Decompose large tasks
# Instead of one 200-message session:
# Session 1: Plan the feature (10 messages)
# Session 2: Implement the backend (30 messages)
# Session 3: Implement the frontend (30 messages)
# Session 4: Write tests and documentation (20 messages)

# Strategy 4: Use subagents for research
# "Use a subagent to investigate the authentication module
#  and report back the key interfaces I need to know about."
# The subagent's work doesn't consume your main context window.
```

---

## 4. MCP Connection Problems

### 4.1 Server Not Starting

**Symptom**: MCP server fails to start when Claude Code initializes.

```
MCP error: Failed to start server "my-mcp-server"
```

**Diagnosis:**

```bash
# Step 1: Check the MCP configuration
cat .claude/mcp_settings.json | python3 -m json.tool

# Step 2: Try starting the server manually
# For stdio transport:
node /path/to/mcp-server/index.js

# For Python servers:
python /path/to/mcp-server/main.py

# Step 3: Check if dependencies are installed
cd /path/to/mcp-server
npm install  # or pip install -r requirements.txt

# Step 4: Check for port conflicts (HTTP transport)
lsof -i :3001  # Check if the port is already in use
```

**Common MCP configuration issues:**

```json
// .claude/mcp_settings.json

{
  "mcpServers": {
    "my-database": {
      // WRONG: Relative path (may not resolve correctly)
      "command": "./mcp-servers/database/index.js",

      // CORRECT: Absolute path
      "command": "/home/user/project/mcp-servers/database/index.js",

      // CORRECT: Using npx for npm packages
      "command": "npx",
      "args": ["-y", "@mcp/database-server"],

      // Environment variables the server needs
      "env": {
        "DATABASE_URL": "postgresql://localhost:5432/mydb"
      }
    }
  }
}
```

### 4.2 Transport Errors

**Symptom**: Server starts but communication fails.

```
MCP transport error: unexpected end of stream
MCP transport error: invalid JSON-RPC message
```

**Diagnosis by transport type:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  MCP Transport Debugging                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  stdio transport (most common):                                     │
│  ├── Server MUST NOT print to stdout except MCP messages           │
│  ├── Debug logging MUST go to stderr                               │
│  ├── Check: does the server print a banner on startup?             │
│  │   → Redirect it to stderr: console.error() not console.log()   │
│  └── Check: is the server reading from stdin correctly?            │
│                                                                     │
│  HTTP/SSE transport:                                                │
│  ├── Check: is the server running and listening?                   │
│  │   → curl http://localhost:3001/health                           │
│  ├── Check: CORS configuration if browser-based                    │
│  ├── Check: SSL/TLS certificate for HTTPS endpoints               │
│  └── Check: proxy or firewall blocking connections                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Fix for the most common stdio issue:**

```javascript
// WRONG: Prints to stdout (interferes with MCP protocol)
console.log("Server starting...");

// CORRECT: Prints to stderr (does not interfere)
console.error("Server starting...");

// CORRECT: Using a proper logger
const logger = {
  info: (msg) => process.stderr.write(`[INFO] ${msg}\n`),
  error: (msg) => process.stderr.write(`[ERROR] ${msg}\n`),
};
logger.info("Server starting...");
```

### 4.3 Authentication Failures

**Symptom**: MCP server rejects connections due to authentication.

```
MCP error: Authentication failed for server "my-api"
```

**Solutions:**

```json
// Pass authentication credentials via environment variables
{
  "mcpServers": {
    "my-api": {
      "command": "npx",
      "args": ["-y", "@mcp/api-server"],
      "env": {
        "API_KEY": "${env:MY_API_KEY}",
        "API_SECRET": "${env:MY_API_SECRET}"
      }
    }
  }
}
```

```bash
# Make sure environment variables are set before starting Claude Code
export MY_API_KEY="your-api-key-here"
export MY_API_SECRET="your-api-secret-here"
claude  # Start Claude Code with env vars available
```

### 4.4 Debugging with MCP Inspector

The MCP Inspector is a diagnostic tool for testing MCP servers:

```bash
# Install MCP Inspector
npx @modelcontextprotocol/inspector

# Test your MCP server interactively
# The Inspector provides a UI to:
# - Connect to your MCP server
# - List available tools, resources, and prompts
# - Execute individual tool calls
# - Inspect request/response messages
# - View error details

# Test a specific server
npx @modelcontextprotocol/inspector node /path/to/your/server.js
```

### 4.5 Common MCP Configuration Mistakes

```json
// MISTAKE 1: Missing "args" for commands that need arguments
{
  "mcpServers": {
    "my-server": {
      "command": "python main.py --port 3001"  // WRONG: args in command string
    }
  }
}
// FIX:
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["main.py", "--port", "3001"]  // CORRECT: separate args
    }
  }
}

// MISTAKE 2: Forgetting to install dependencies
// FIX: Run npm install or pip install before configuring

// MISTAKE 3: Wrong working directory
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["server.js"],
      "cwd": "/correct/working/directory"  // Specify working directory
    }
  }
}
```

---

## 5. API Errors

### 5.1 401 Unauthorized

**Symptom:**

```
Error: 401 Unauthorized — Invalid API key
```

**Diagnosis:**

```bash
# Check if the API key is set
echo $ANTHROPIC_API_KEY | head -c 10  # Show first 10 chars only

# Verify the key format (should start with sk-ant-)
# If using a different provider, check their key format

# Common issues:
# - Key copied with trailing whitespace
# - Key from wrong environment (test vs production)
# - Key has been revoked or expired
# - Key set in wrong shell profile (.bashrc vs .zshrc)
```

**Solutions:**

```bash
# Set the API key correctly
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Verify it works
curl -s https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "content-type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "Hi"}]
  }' | python3 -m json.tool

# If using a configuration file
# ~/.claude/config.json
{
  "api_key_source": "environment"  // or "keychain"
}
```

### 5.2 429 Rate Limited

**Symptom:**

```
Error: 429 Too Many Requests — Rate limit exceeded
```

**Understanding rate limits:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Rate Limit Types                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Requests per minute (RPM)                                       │
│     → Too many API calls in a short period                         │
│     → Solution: Space out requests, use batch API                  │
│                                                                     │
│  2. Tokens per minute (TPM)                                         │
│     → Sending too much text too quickly                            │
│     → Solution: Reduce prompt sizes, use caching                   │
│                                                                     │
│  3. Tokens per day (TPD)                                            │
│     → Daily quota exhausted                                        │
│     → Solution: Optimize token usage, upgrade plan                 │
│                                                                     │
│  Rate limit headers in response:                                    │
│  x-ratelimit-limit-requests: 100                                   │
│  x-ratelimit-remaining-requests: 23                                │
│  x-ratelimit-reset-requests: 2024-01-01T00:01:00Z                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementing backoff in custom code:**

```python
import time
import anthropic
from anthropic import RateLimitError

def call_with_backoff(client, max_retries=5, **kwargs):
    """Call the API with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise  # Give up after max retries

            # Exponential backoff: 1s, 2s, 4s, 8s, 16s
            wait_time = 2 ** attempt
            print(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}...")
            time.sleep(wait_time)

    raise RuntimeError("Max retries exceeded")

# Usage
client = anthropic.Anthropic()
response = call_with_backoff(
    client,
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

**For Claude Code users**: Claude Code handles rate limiting automatically with built-in retry logic. If you see persistent rate limit errors, it usually means you are running too many parallel agents. Reduce the number of concurrent subagents.

### 5.3 529 Overloaded

**Symptom:**

```
Error: 529 — API is temporarily overloaded
```

This is different from rate limiting -- it means the API itself is under heavy load across all users.

**Solutions:**

```python
import time
import random
import anthropic
from anthropic import APIStatusError

def call_with_jitter(client, max_retries=3, **kwargs):
    """Retry with jitter for overloaded errors."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except APIStatusError as e:
            if e.status_code == 529 and attempt < max_retries - 1:
                # Jittered backoff to avoid thundering herd
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"API overloaded. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise
```

**For Claude Code users**: Wait a few minutes and try again. If the issue persists, check https://status.anthropic.com for service status.

### 5.4 Timeout Errors

**Symptom:**

```
Error: Request timed out after 60000ms
```

**Common causes:**
- Very large input (close to context window limit)
- Complex reasoning tasks that take longer to process
- Network connectivity issues

**Solutions:**

```python
# Increase timeout for large/complex requests
client = anthropic.Anthropic(
    timeout=120.0  # 2 minutes instead of default 60 seconds
)

# Or per-request timeout
response = client.messages.create(
    model="claude-opus-4-20250514",
    max_tokens=8192,
    messages=[{"role": "user", "content": very_long_prompt}],
    timeout=180.0  # 3 minutes for this specific request
)
```

---

## 6. Performance Issues

### 6.1 Slow Response Times

**Diagnosis:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Slow Response — Root Cause Analysis                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Is the slowness in Claude's response or tool execution?           │
│                                                                     │
│  Claude is slow to respond:                                        │
│  ├── Large input context → Reduce context, use caching            │
│  ├── Complex task → Expected; Opus is slower than Sonnet          │
│  ├── Extended thinking enabled → Budget more time                  │
│  └── Service degradation → Check status.anthropic.com             │
│                                                                     │
│  Tool execution is slow:                                            │
│  ├── Bash commands taking long → Check what's running             │
│  ├── File reads on large files → Use targeted reads               │
│  ├── Grep on large codebase → Use more specific patterns          │
│  └── MCP server responding slowly → Debug the MCP server          │
│                                                                     │
│  Session startup is slow:                                           │
│  ├── Large CLAUDE.md → Trim to essentials                         │
│  ├── Many MCP servers → Only enable what you need                 │
│  └── Slow network → Check internet connection                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 High Token Usage

If your token usage (and costs) are higher than expected:

```
# Causes of high token usage:
1. CLAUDE.md is too large (sent with every message)
2. Including entire files when only a section is relevant
3. Verbose prompts with unnecessary words
4. Not using prompt caching for repeated context
5. Using Opus when Sonnet would suffice
6. Claude re-reading files it already has in context

# Diagnostic steps:
# Check usage in the API response
response = client.messages.create(...)
print(f"Input tokens:  {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
# If input tokens are consistently high, check your context size
```

### 6.3 Session Startup Delays

```bash
# Check CLAUDE.md size (should be < 200 lines)
wc -l CLAUDE.md

# Check number of MCP servers
cat .claude/mcp_settings.json | python3 -c "
import json, sys
config = json.load(sys.stdin)
servers = config.get('mcpServers', {})
print(f'MCP servers configured: {len(servers)}')
for name in servers:
    print(f'  - {name}')
"

# If startup is slow, try disabling MCP servers temporarily
# to isolate the issue
```

---

## 7. Tool Execution Problems

### 7.1 Bash Commands Failing

**Symptom**: A Bash command that works in your terminal fails when Claude runs it.

**Common causes:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Bash Command Failures — Common Causes                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Working directory mismatch                                      │
│     Claude's working directory resets between Bash calls            │
│     Fix: Use absolute paths, not relative paths                    │
│                                                                     │
│  2. Environment variable differences                                │
│     Claude's shell may not have your full environment              │
│     Fix: Set variables explicitly or use .env files                │
│                                                                     │
│  3. Shell differences (bash vs zsh)                                 │
│     Some syntax works in zsh but not bash (or vice versa)          │
│     Fix: Use POSIX-compatible syntax when possible                 │
│     Note: declare -A (bash) is not available in zsh               │
│                                                                     │
│  4. Interactive commands                                            │
│     Commands requiring user input will hang or fail                │
│     Fix: Use non-interactive flags (e.g., -y, --yes, --batch)    │
│                                                                     │
│  5. Timeout                                                         │
│     Commands taking longer than 2 minutes will timeout             │
│     Fix: Use --timeout flag for long-running commands              │
│                                                                     │
│  6. Permission denied                                               │
│     The command is not in the allow list                           │
│     Fix: Add to allow rules in settings                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 File Edit Conflicts

**Symptom**: Claude's file edit fails because the content does not match what it expected.

```
Error: old_string not found in file (content may have changed)
```

**Causes:**
- You edited the file in your IDE while Claude was working
- Another tool (formatter, linter) modified the file
- A hook modified the file after a previous edit

**Solutions:**

```
# Tell Claude to re-read the file
> "The file has been modified externally. Re-read src/app.py and try
>  the edit again."

# If using auto-formatters, configure them to not run on save
# while Claude is actively editing, or configure Claude's hook
# to run the formatter AFTER edits are complete

# Prevent conflicts by telling your IDE to detect external changes:
# VS Code: "files.watcherExclude" or disable auto-save temporarily
```

### 7.3 Git Operation Errors

**Symptom**: Git commands fail unexpectedly.

```
error: Your local changes to the following files would be overwritten by merge
```

**Common scenarios and fixes:**

```bash
# Scenario 1: Uncommitted changes blocking checkout
# Claude tries to switch branches but has uncommitted work
git stash  # Stash changes
git checkout target-branch
git stash pop  # Restore changes

# Scenario 2: Merge conflicts
# Claude should NOT resolve merge conflicts automatically
# unless explicitly asked
# Tell Claude: "There are merge conflicts in these files. Help me resolve them."

# Scenario 3: Detached HEAD state
git checkout main  # Return to a branch
# Or create a branch from the current state:
git checkout -b recovery-branch
```

---

## 8. The /doctor Command

Claude Code includes a built-in diagnostic command that checks common configuration issues:

```
> /doctor

Claude Code Diagnostic Report
==============================

✓ Claude Code version: 1.x.x (up to date)
✓ API key: configured (sk-ant-...xxxx)
✓ API connectivity: OK (response time: 145ms)
✓ Permission mode: default
✓ CLAUDE.md: found (142 lines)
✓ Settings: .claude/settings.json valid
⚠ MCP servers: 2/3 healthy
  ✓ filesystem: connected
  ✓ database: connected
  ✗ slack: failed to start (command not found: npx)
✓ Hooks: 2 configured, all valid
✓ Git: repository detected, clean working tree
✓ Shell: /bin/zsh
✓ Node.js: v20.11.0
✓ Python: 3.12.1
```

**Use /doctor when:**
- Starting with a new project
- After changing configuration
- When experiencing unexplained issues
- After updating Claude Code
- When onboarding a new team member

---

## 9. Where to Get Help

### 9.1 Documentation

```
┌─────────────────────────────────────────────────────────────────────┐
│  Documentation Resources                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Official Documentation                                             │
│  └── https://docs.anthropic.com                                    │
│      ├── Claude Code guide                                          │
│      ├── API reference                                              │
│      ├── MCP specification                                          │
│      └── Prompt engineering guide                                   │
│                                                                     │
│  GitHub                                                             │
│  └── https://github.com/anthropics/claude-code                     │
│      ├── Issues: report bugs, search existing issues               │
│      ├── Discussions: ask questions, share patterns                │
│      └── README: installation and quick start                      │
│                                                                     │
│  MCP Specification                                                  │
│  └── https://modelcontextprotocol.io                               │
│      ├── Protocol specification                                    │
│      ├── Server registry                                           │
│      └── SDK documentation                                         │
│                                                                     │
│  API Status                                                         │
│  └── https://status.anthropic.com                                  │
│      ├── Current service status                                    │
│      ├── Incident history                                          │
│      └── Subscribe for notifications                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 Filing Effective Bug Reports

When filing an issue on GitHub, include:

```markdown
## Bug Report Template

**Environment:**
- Claude Code version: (claude --version)
- OS: macOS 15.3 / Ubuntu 24.04 / Windows 11
- Shell: zsh / bash
- Node.js version: (node --version)
- Python version: (python3 --version)

**Description:**
What happened? What did you expect to happen?

**Steps to reproduce:**
1. Start Claude Code in a project with this structure: ...
2. Type this prompt: ...
3. Claude executes this tool: ...
4. Error occurs: ...

**Configuration:**
- Permission mode: default
- CLAUDE.md: [attach or summarize]
- Settings: [attach .claude/settings.json]
- MCP servers: [list configured servers]

**Error output:**
```
[paste the full error message here]
```

**Screenshots (if applicable):**
[attach screenshots showing the issue]
```

### 9.3 Community Resources

```
- GitHub Discussions: Ask questions, share tips
  https://github.com/anthropics/claude-code/discussions

- Anthropic Discord: Real-time community help
  https://discord.gg/anthropic

- Stack Overflow: Tag questions with [claude-code] or [anthropic]
```

---

## 10. Troubleshooting Decision Tree

When something goes wrong, use this decision tree to quickly identify the category of issue:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Troubleshooting Decision Tree                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  What kind of error are you seeing?                                │
│  │                                                                  │
│  ├── "Permission denied" or "not allowed"                          │
│  │   └── Go to Section 1: Permission Errors                       │
│  │                                                                  │
│  ├── "Hook failed" or "hook timeout"                               │
│  │   └── Go to Section 2: Hook Failures                           │
│  │                                                                  │
│  ├── Claude forgets context or "context too long"                  │
│  │   └── Go to Section 3: Context Window Issues                   │
│  │                                                                  │
│  ├── "MCP error" or "server not starting"                          │
│  │   └── Go to Section 4: MCP Connection Problems                 │
│  │                                                                  │
│  ├── HTTP error (401, 429, 500, 529)                               │
│  │   └── Go to Section 5: API Errors                              │
│  │                                                                  │
│  ├── Slow responses or high costs                                  │
│  │   └── Go to Section 6: Performance Issues                      │
│  │                                                                  │
│  ├── Bash/Edit/Git command failures                                │
│  │   └── Go to Section 7: Tool Execution Problems                 │
│  │                                                                  │
│  ├── Not sure / multiple issues                                    │
│  │   └── Run /doctor (Section 8)                                  │
│  │                                                                  │
│  └── None of the above                                             │
│      └── Go to Section 9: Where to Get Help                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 11. Exercises

### Exercise 1: Permission Configuration (Beginner)

Create a `.claude/settings.json` that:
1. Allows Claude to edit files only in `src/` and `tests/`
2. Allows running `npm test`, `npm run lint`, and `npm run build`
3. Denies editing any `.env` file
4. Denies running `rm`, `docker`, or `sudo` commands

Test your configuration by asking Claude to perform both allowed and denied operations.

### Exercise 2: Hook Debugging (Intermediate)

Create a hook script that:
1. Runs ESLint (or flake8) on any file Claude edits
2. Includes debug logging to stderr
3. Handles the case where the linter is not installed
4. Has a 15-second timeout

Deliberately introduce a bug in the hook and practice debugging it using the steps in Section 2.4.

### Exercise 3: Context Management (Intermediate)

Start a Claude Code session and intentionally push it toward context limits:
1. Read 5 large files (500+ lines each)
2. Have a 30-message conversation about the code
3. Use `/compact` and observe what gets preserved
4. Continue the conversation and note when context degradation begins
5. Document: How many messages before degradation? Did `/compact` help?

### Exercise 4: API Error Handling (Advanced)

Write a Python script that:
1. Makes 50 rapid API calls to trigger rate limiting
2. Implements exponential backoff with jitter
3. Logs each retry attempt with timing information
4. Reports success rate and average response time

```python
# Starter code
import anthropic
import time

client = anthropic.Anthropic()
results = {"success": 0, "rate_limited": 0, "errors": 0}

# Your implementation here...
# Track: attempts, retries, total time, success rate
```

### Exercise 5: Full Diagnostic (Beginner)

Run `/doctor` on your Claude Code installation and:
1. Document any warnings or errors
2. Fix each issue found
3. Run `/doctor` again to verify the fixes
4. Create a checklist of items to verify when setting up Claude Code on a new machine

---

## References

- Claude Code Documentation - https://docs.anthropic.com/en/docs/claude-code
- Anthropic API Reference - https://docs.anthropic.com/en/api
- MCP Specification - https://modelcontextprotocol.io
- Anthropic Status Page - https://status.anthropic.com
- GitHub Issues - https://github.com/anthropics/claude-code/issues

---

## Conclusion

This lesson completes the Claude Ecosystem topic. You now have a comprehensive understanding of Claude Code from installation through advanced workflows, and a troubleshooting reference for when things go wrong. The most effective Claude Code users combine good prompting habits (Lesson 21) with systematic debugging skills (this lesson) to maintain productive, efficient development sessions.

For continued learning, revisit earlier lessons as you encounter new scenarios, and stay current with the Anthropic documentation as Claude Code continues to evolve.
