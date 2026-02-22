# Claude Code: Getting Started

**Previous**: [01. Introduction to Claude](./01_Introduction_to_Claude.md) | **Next**: [03. CLAUDE.md and Project Setup](./03_CLAUDE_md_and_Project_Setup.md)

---

Claude Code is a command-line AI coding assistant that runs directly in your terminal. Unlike web-based AI tools, Claude Code operates in the context of your actual project — it can read your files, edit code, run commands, execute tests, and make git commits. This lesson walks you through installation, your first session, and the fundamental workflow that makes Claude Code an effective development partner.

**Difficulty**: ⭐

**Prerequisites**:
- [01. Introduction to Claude](./01_Introduction_to_Claude.md)
- Terminal/command-line basics (see **Shell_Script** topic)
- A code editor and a project to work with
- Node.js 18+ installed (for npm installation)

**Learning Objectives**:
- Install Claude Code and authenticate
- Start and navigate an interactive session
- Understand the core tools Claude Code uses (Read, Write, Edit, Bash, Glob, Grep)
- Follow the read-edit-test-commit workflow
- Use basic slash commands for session management
- Complete a practical debugging walkthrough from start to finish

---

## Table of Contents

1. [What Is Claude Code?](#1-what-is-claude-code)
2. [Installation](#2-installation)
3. [Authentication](#3-authentication)
4. [Your First Session](#4-your-first-session)
5. [The Tool System](#5-the-tool-system)
6. [The Core Workflow](#6-the-core-workflow)
7. [Session Management](#7-session-management)
8. [Essential Slash Commands](#8-essential-slash-commands)
9. [Practical Walkthrough: Fixing a Bug](#9-practical-walkthrough-fixing-a-bug)
10. [Working Directory and Project Scope](#10-working-directory-and-project-scope)
11. [Tips for Effective Use](#11-tips-for-effective-use)
12. [Exercises](#12-exercises)
13. [Next Steps](#13-next-steps)

---

## 1. What Is Claude Code?

Claude Code is an **agentic coding tool** — it does not just answer questions about code, it takes actions. When you describe a task, Claude Code:

1. **Reads** your project files to understand context
2. **Plans** an approach based on your codebase
3. **Edits** files to implement changes
4. **Runs** commands to test, build, or verify
5. **Iterates** based on results (fixing errors, adjusting approach)

This agentic loop continues until the task is complete. You remain in control — Claude Code asks for permission before making changes (unless configured otherwise).

```
┌─────────────────────────────────────────────────────────┐
│                   Claude Code Agent Loop                 │
│                                                         │
│     You describe a task                                 │
│           │                                             │
│           ▼                                             │
│     ┌───────────┐                                       │
│     │   Read    │ ← Understand codebase                 │
│     └─────┬─────┘                                       │
│           ▼                                             │
│     ┌───────────┐                                       │
│     │   Plan    │ ← Decide approach                     │
│     └─────┬─────┘                                       │
│           ▼                                             │
│     ┌───────────┐                                       │
│     │   Edit    │ ← Make changes                        │
│     └─────┬─────┘                                       │
│           ▼                                             │
│     ┌───────────┐      ┌──────────┐                     │
│     │   Test    │─────▶│  Errors? │──── Yes ──┐         │
│     └───────────┘      └──────────┘           │         │
│                              │                 │         │
│                             No                 │         │
│                              │                 ▼         │
│                              ▼          ┌───────────┐   │
│                         ┌────────┐      │   Fix     │   │
│                         │  Done  │      └─────┬─────┘   │
│                         └────────┘            │         │
│                                               └──▶ Read │
└─────────────────────────────────────────────────────────┘
```

### What Claude Code Is NOT

- **Not an IDE plugin** (though IDE integrations exist — see Lesson 09)
- **Not a code completion engine** (it handles whole tasks, not line-by-line suggestions)
- **Not a chatbot** (it takes real actions on your filesystem and terminal)

---

## 2. Installation

### Method 1: npm (Recommended)

```bash
# Install globally via npm
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version
```

### Method 2: Homebrew (macOS)

```bash
# Install via Homebrew
brew install claude-code

# Verify installation
claude --version
```

### System Requirements

| Requirement | Minimum |
|------------|---------|
| **OS** | macOS 12+, Ubuntu 20.04+, Windows (via WSL2) |
| **Node.js** | 18.0 or later |
| **RAM** | 4 GB (8 GB recommended) |
| **Terminal** | Any modern terminal emulator |
| **Shell** | bash, zsh, or fish |

### Updating

```bash
# Update to the latest version
npm update -g @anthropic-ai/claude-code

# Or with Homebrew
brew upgrade claude-code
```

---

## 3. Authentication

Claude Code needs an API key or an active Anthropic account to function. There are two authentication methods.

### Method 1: Interactive Login

```bash
# Start the login flow
claude login

# This opens your browser for authentication
# After logging in, the CLI stores your credentials locally
```

### Method 2: API Key

```bash
# Set your API key as an environment variable
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Or add it to your shell profile (~/.zshrc, ~/.bashrc)
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-..."' >> ~/.zshrc
source ~/.zshrc
```

### Verifying Authentication

```bash
# Start Claude Code — if authentication is correct, you'll see the prompt
claude

# If there's an auth issue, you'll see a clear error message
# with instructions on how to fix it
```

### Authentication for Teams

If you are using Claude Code through an organization (Claude for Work or Enterprise):

```bash
# Login with your organization account
claude login

# The CLI will detect your organization membership automatically
# Organization policies (model access, rate limits) apply
```

---

## 4. Your First Session

### Starting a Session

Navigate to your project directory and start Claude Code:

```bash
# Navigate to your project
cd ~/projects/my-app

# Start an interactive session
claude
```

You will see the Claude Code prompt:

```
╭────────────────────────────────────────────╮
│ Claude Code                                │
│                                            │
│ /help for commands, /exit to quit          │
│                                            │
│ cwd: /Users/you/projects/my-app            │
╰────────────────────────────────────────────╯

>
```

### Natural Language Interaction

Type your request in plain English (or any supported language). Claude Code understands natural language and translates it into actions.

```
> What does this project do?

Claude will:
1. Read your README.md, package.json, or equivalent files
2. Scan the directory structure
3. Provide a summary of the project's purpose, stack, and structure
```

```
> Find all TODO comments in the codebase

Claude will:
1. Use the Grep tool to search for "TODO" across all files
2. Present the results organized by file
3. Optionally suggest fixes for the TODOs
```

```
> Add input validation to the user registration endpoint

Claude will:
1. Find the registration endpoint code
2. Read the current implementation
3. Propose validation logic
4. Ask for permission to edit the file
5. Edit the file with validation code
6. Suggest running tests to verify
```

### Permission Prompts

By default, Claude Code asks for permission before taking actions. You will see prompts like:

```
Claude wants to edit src/routes/auth.ts

  + import { z } from 'zod';
  +
  + const registrationSchema = z.object({
  +   email: z.string().email(),
  +   password: z.string().min(8),
  +   name: z.string().min(1).max(100),
  + });

Allow? (y/n/always)
```

- **y**: Allow this specific action
- **n**: Deny this action
- **always**: Allow all similar actions for this session

---

## 5. The Tool System

Claude Code operates through a set of built-in **tools** — each designed for a specific type of action. Understanding these tools helps you predict and guide Claude Code's behavior.

### Tool Reference Table

| Tool | Purpose | Example |
|------|---------|---------|
| **Read** | Read file contents | Reading `src/app.py` to understand structure |
| **Write** | Create or overwrite a file | Creating a new `test_auth.py` file |
| **Edit** | Make targeted edits to existing files | Changing a function signature in `utils.py` |
| **Bash** | Execute shell commands | Running `pytest`, `npm test`, `git status` |
| **Glob** | Find files by name pattern | Finding all `*.test.ts` files |
| **Grep** | Search file contents | Finding all uses of `deprecated_function` |
| **WebFetch** | Retrieve web content | Fetching API documentation |
| **WebSearch** | Search the web | Looking up a library's latest version |
| **NotebookEdit** | Edit Jupyter notebooks | Modifying a cell in `analysis.ipynb` |

### How Tools Work Together

A typical task involves multiple tools in sequence:

```
Task: "Fix the failing test in test_auth.py"

Step 1: Bash       → Run `pytest test_auth.py` to see the failure
Step 2: Read       → Read test_auth.py to understand the test
Step 3: Grep       → Search for the function being tested
Step 4: Read       → Read the source file with the bug
Step 5: Edit       → Fix the bug in the source file
Step 6: Bash       → Run `pytest test_auth.py` to verify the fix
Step 7: (Done)     → Report success to you
```

### Tool Permissions

Each tool can be individually controlled through permission settings (covered in detail in Lesson 04). By default:

- **Read, Glob, Grep**: Allowed without prompting (read-only operations)
- **Edit, Write**: Require approval (they modify your files)
- **Bash**: Requires approval (it executes commands)

---

## 6. The Core Workflow

The fundamental Claude Code workflow follows four steps: **Read → Edit → Test → Commit**. This mirrors how experienced developers work — understand the code, make changes, verify they work, and save progress.

### Step 1: Read (Understand)

Before making any changes, Claude Code reads relevant files to build context.

```
> Explain the authentication flow in this project

Claude:
  Tool: Glob("**/auth*", "**/*login*", "**/*session*")
  Tool: Read("src/middleware/auth.ts")
  Tool: Read("src/routes/login.ts")
  Tool: Read("src/models/user.ts")

  "The authentication flow works as follows:
   1. User submits credentials to POST /api/login
   2. The login route validates against the User model
   3. On success, a JWT token is created and returned
   4. The auth middleware verifies the JWT on protected routes
   ..."
```

### Step 2: Edit (Change)

Claude Code makes targeted changes using the Edit tool for existing files or Write for new files.

```
> Add rate limiting to the login endpoint

Claude:
  Tool: Edit("src/routes/login.ts")
    + import rateLimit from 'express-rate-limit';
    +
    + const loginLimiter = rateLimit({
    +   windowMs: 15 * 60 * 1000,  // 15 minutes
    +   max: 5,                      // 5 attempts per window
    +   message: 'Too many login attempts, please try again later'
    + });
    +
    - router.post('/login', loginHandler);
    + router.post('/login', loginLimiter, loginHandler);
```

### Step 3: Test (Verify)

After editing, Claude Code runs your project's test suite to verify nothing is broken.

```
> Now run the tests to make sure everything works

Claude:
  Tool: Bash("npm test")

  "All 47 tests pass. The rate limiting is working correctly.
   I also see the existing test for the login route passes."
```

### Step 4: Commit (Save)

Once tests pass, Claude Code can create a well-formatted git commit.

```
> Commit this change

Claude:
  Tool: Bash("git add src/routes/login.ts")
  Tool: Bash("git commit -m 'Add rate limiting to login endpoint

  Limit login attempts to 5 per 15-minute window to prevent
  brute force attacks. Uses express-rate-limit middleware.'")

  "Changes committed. The commit includes the rate limiting
   addition to the login route."
```

---

## 7. Session Management

### Session Lifetime

A Claude Code session persists as long as the terminal is open. The conversation history builds up context over time, allowing Claude to reference earlier discussions.

```bash
# Start a session (runs until you exit)
claude

# Inside the session, you can have many exchanges
> Fix the login bug
> Now add tests for it
> Update the documentation
> Commit everything
> /exit
```

### Resuming Sessions

```bash
# Resume the most recent session
claude --resume

# Continue a specific session
claude --resume <session-id>
```

### One-Shot Mode

For quick tasks that do not require an interactive session:

```bash
# Execute a single prompt and exit
claude -p "How many TODO comments are in this project?"

# Pipe input to Claude
cat error.log | claude -p "Explain this error and suggest a fix"

# Use a specific model
claude -p "Explain this code" --model claude-opus-4-20250514
```

### Context Management

As conversations grow, they consume more tokens. Use these strategies to manage context:

```
> /compact

# Summarizes the conversation so far, freeing up context space
# Useful during long sessions when you notice Claude forgetting earlier details
```

```
> /clear

# Clears the conversation entirely and starts fresh
# Use when switching to a completely different task
```

---

## 8. Essential Slash Commands

Claude Code provides built-in slash commands for session management and common operations.

### Command Reference

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands and usage tips |
| `/exit` | End the current session |
| `/clear` | Clear conversation history |
| `/compact` | Summarize conversation to free context space |
| `/cost` | Show token usage and cost for the current session |
| `/doctor` | Diagnose configuration and connection issues |
| `/init` | Initialize a CLAUDE.md file for the current project |
| `/model` | Display or change the current model |
| `/config` | View or edit configuration |

### Using /cost

The `/cost` command is valuable for tracking spending during development sessions:

```
> /cost

Session cost:
  Input tokens:  45,230  ($0.14)
  Output tokens: 12,847  ($0.19)
  Total cost:    $0.33

  Model: claude-sonnet-4-20250514
  Session duration: 23 minutes
```

### Using /doctor

When things are not working correctly, `/doctor` runs diagnostics:

```
> /doctor

Checking configuration...
  ✓ Authentication: Valid API key
  ✓ Model access: claude-sonnet-4-20250514
  ✓ CLAUDE.md: Found at /Users/you/projects/my-app/CLAUDE.md
  ✓ Settings: Valid JSON
  ✓ Hooks: 2 hooks configured, all valid
  ✗ MCP: Server "github" not responding

  1 issue found. Run `/doctor --fix` for suggestions.
```

---

## 9. Practical Walkthrough: Fixing a Bug

Let us walk through a complete, realistic debugging session. This demonstrates the full Claude Code workflow from problem identification to committed fix.

### The Scenario

You have a Python Flask application with a failing test. The test report shows:

```
FAILED tests/test_api.py::test_create_user - AssertionError:
  Expected status code 201, got 400
```

### Step-by-Step Session

```bash
# Navigate to project and start Claude Code
cd ~/projects/user-service
claude
```

**Exchange 1: Understand the failure**

```
> The test test_create_user in tests/test_api.py is failing.
  It expects a 201 but gets 400. Can you investigate and fix it?
```

Claude Code begins its investigation:

```
Tool: Bash("python -m pytest tests/test_api.py::test_create_user -v")

FAILED tests/test_api.py::test_create_user - AssertionError:
  assert 400 == 201

Tool: Read("tests/test_api.py")

# Claude reads the test and sees:
def test_create_user(client):
    response = client.post("/api/users", json={
        "name": "Alice",
        "email": "alice@example.com",
        "age": 25
    })
    assert response.status_code == 201

Tool: Grep("create_user", type="py")

# Finds the route handler
Tool: Read("src/routes/users.py")
```

**Exchange 2: Identify the root cause**

Claude explains:

```
I found the issue. In `src/routes/users.py`, the `create_user`
endpoint was recently updated to require a `role` field in the
request body:

    schema = {
        "name": {"type": "string", "required": True},
        "email": {"type": "string", "required": True},
        "age": {"type": "integer", "required": True},
        "role": {"type": "string", "required": True},  # ← Added recently
    }

But the test doesn't include the `role` field. There are two
possible fixes:

1. Make `role` optional with a default value (backward compatible)
2. Update the test to include a `role` field

Which approach do you prefer?
```

**Exchange 3: Apply the fix**

```
> Make role optional with a default of "user"
```

```
Tool: Edit("src/routes/users.py")

  - "role": {"type": "string", "required": True},
  + "role": {"type": "string", "required": False, "default": "user"},

Tool: Edit("src/routes/users.py")

  # In the handler function, add default value logic:
  + role = data.get("role", "user")
```

**Exchange 4: Verify**

```
Tool: Bash("python -m pytest tests/test_api.py -v")

tests/test_api.py::test_create_user PASSED
tests/test_api.py::test_list_users PASSED
tests/test_api.py::test_delete_user PASSED

3 passed in 0.42s
```

**Exchange 5: Commit**

```
> Commit this fix

Tool: Bash("git add src/routes/users.py")
Tool: Bash('git commit -m "Make role field optional with default value

The create_user endpoint required a role field after a recent
update, breaking backward compatibility. Made the field optional
with a default of \"user\" to maintain API compatibility."')
```

---

## 10. Working Directory and Project Scope

Claude Code operates within the context of your **working directory** — the directory where you launched the `claude` command. This determines what files Claude can see and what project context it loads.

### Project Root Detection

Claude Code automatically detects your project root by looking for:

1. A `CLAUDE.md` file
2. A `.git` directory
3. A `package.json`, `pyproject.toml`, `Cargo.toml`, or similar project file
4. A `.claude/` directory

### Scope Rules

```
/Users/you/projects/my-app/     ← Project root (has .git)
├── CLAUDE.md                    ← Read at session start
├── src/
│   ├── app.py                   ← Within scope ✓
│   └── utils.py                 ← Within scope ✓
├── tests/
│   └── test_app.py              ← Within scope ✓
└── node_modules/                ← Within scope (but usually ignored)

/Users/you/other-project/        ← Outside scope ✗
```

### Multiple Projects

If you need to work across multiple projects, you can:

```bash
# Option 1: Start Claude Code from a parent directory
cd ~/projects
claude
# Claude can now access all subdirectories

# Option 2: Use separate sessions
# Terminal 1
cd ~/projects/frontend && claude

# Terminal 2
cd ~/projects/backend && claude
```

---

## 11. Tips for Effective Use

### Be Specific

```
# Less effective
> Fix the bug

# More effective
> The /api/users endpoint returns 500 when the email field
  contains unicode characters. Fix the validation logic.
```

### Provide Context

```
# Less effective
> Add caching

# More effective
> Add Redis caching to the get_user_by_id function in
  src/services/user.py. Cache entries should expire after
  5 minutes. We're using the redis-py library.
```

### Let Claude Verify

```
# Good pattern: ask Claude to run tests after changes
> Fix the sorting bug in utils.py and run the tests to confirm
```

### Use Iterative Refinement

```
> Add pagination to the list endpoint
# Claude implements basic pagination

> The offset-based pagination is fine but add a total_count
  field in the response so the frontend knows how many pages
  there are
# Claude refines the implementation

> Now add tests for the pagination edge cases — empty results,
  last page, invalid page numbers
# Claude adds comprehensive tests
```

---

## 12. Exercises

### Exercise 1: Installation Verification

Install Claude Code and verify it works:

1. Install via npm or Homebrew
2. Authenticate with `claude login` or an API key
3. Run `claude --version` and note the version
4. Start a session with `claude` and ask "What directory am I in?"
5. Run `/doctor` to check your configuration
6. Run `/cost` to see the session cost
7. Exit with `/exit`

### Exercise 2: Codebase Exploration

Navigate to any project you are working on and start a Claude Code session:

1. Ask Claude to describe the project structure
2. Ask it to find all files that import a specific module
3. Ask it to explain a function you find confusing
4. Ask it to identify potential bugs or code smells

### Exercise 3: Bug Fix Practice

Create a deliberate bug in one of your projects (or use a test project):

1. Introduce a bug (e.g., an off-by-one error, a typo in a variable name)
2. Start Claude Code and describe the symptom without revealing the cause
3. Let Claude investigate and propose a fix
4. Review the fix, approve it, and run tests
5. Have Claude create a commit

---

## 13. Next Steps

You now have Claude Code installed and understand its core workflow. The next lesson covers **CLAUDE.md** — the project configuration file that teaches Claude Code about your project's conventions, coding standards, and testing procedures. A well-crafted CLAUDE.md dramatically improves Claude Code's effectiveness by giving it the context it needs to follow your team's practices.

**Next**: [03. CLAUDE.md and Project Setup](./03_CLAUDE_md_and_Project_Setup.md)
