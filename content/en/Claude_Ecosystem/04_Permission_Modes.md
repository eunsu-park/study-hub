# Permission Modes and Security

**Previous**: [03. CLAUDE.md and Project Setup](./03_CLAUDE_md_and_Project_Setup.md) | **Next**: [05. Hooks and Event-Driven Automation](./05_Hooks.md)

---

Claude Code can read files, write code, and execute arbitrary shell commands on your machine. This power demands a robust permission system. Claude Code provides five distinct permission modes, each designed for a different trust level and workflow. This lesson covers every mode in detail, explains how to configure allow/deny rules, and establishes security best practices for individual and team use.

**Difficulty**: ⭐⭐

**Prerequisites**:
- [02. Claude Code: Getting Started](./02_Claude_Code_Getting_Started.md)
- [03. CLAUDE.md and Project Setup](./03_CLAUDE_md_and_Project_Setup.md)
- Understanding of file system permissions and shell commands

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand why permission controls exist and the threat model they address
2. Configure each of the five permission modes
3. Write allow/deny rules with glob patterns for fine-grained control
4. Set up file access rules to restrict read and edit operations
5. Choose the appropriate mode for different workflows
6. Apply security best practices including the principle of least privilege

---

## Table of Contents

1. [Why Permissions Matter](#1-why-permissions-matter)
2. [Permission Modes Overview](#2-permission-modes-overview)
3. [Default Mode](#3-default-mode)
4. [Auto-Accept Mode](#4-auto-accept-mode)
5. [Plan Mode](#5-plan-mode)
6. [Don't Ask Mode](#6-dont-ask-mode)
7. [Bypass Mode](#7-bypass-mode)
8. [Allow and Deny Rules](#8-allow-and-deny-rules)
9. [File Access Rules](#9-file-access-rules)
10. [Choosing the Right Mode](#10-choosing-the-right-mode)
11. [Security Best Practices](#11-security-best-practices)
12. [Exercises](#12-exercises)
13. [Next Steps](#13-next-steps)

---

## 1. Why Permissions Matter

Claude Code is not a sandboxed chatbot — it operates directly on your filesystem and terminal. A single misstep can:

- **Delete files**: `rm -rf` in the wrong directory
- **Overwrite code**: An incorrect edit to a production configuration
- **Expose secrets**: Running a command that prints environment variables to a log
- **Break builds**: Installing incompatible dependencies or modifying build files
- **Push code**: Git operations that affect remote repositories

The permission system is your safety net. It ensures Claude Code only takes actions you explicitly authorize — or, when you trust the environment, allows you to relax controls for faster iteration.

### Threat Model

```
┌─────────────────────────────────────────────────────┐
│                  Trust Levels                        │
│                                                     │
│  High Trust                         Low Trust       │
│  ◀─────────────────────────────────────────────▶    │
│                                                     │
│  Disposable VM   Personal project   Production      │
│  CI container    Side project       Client code     │
│  Playground      Known codebase     Shared machine  │
│                                                     │
│  ← Bypass mode   Sonnet mode →     Plan mode →      │
│  ← Auto-accept   Default mode →    Don't Ask →      │
└─────────────────────────────────────────────────────┘
```

---

## 2. Permission Modes Overview

| Mode | Prompts? | Edits? | Commands? | Best For |
|------|----------|--------|-----------|----------|
| **Default** | Yes, for each action | With approval | With approval | Learning, sensitive code |
| **Auto-Accept** | No | Auto-approved | Auto-approved | Trusted projects, fast iteration |
| **Plan** | N/A | No | No (read-only) | Analysis, architecture review |
| **Don't Ask** | Only for unknown actions | Rule-based | Rule-based | Configured workflows |
| **Bypass** | No | Yes, no sandbox | Yes, no sandbox | Containers, VMs, CI/CD |

### How to Set the Mode

```bash
# Start in a specific mode
claude --mode default
claude --mode auto-accept
claude --mode plan
claude --mode bypass

# Inside a session, switch modes (if your settings allow it)
> /mode plan
> /mode auto-accept
```

---

## 3. Default Mode

Default mode is the most cautious option. Claude Code asks for your explicit approval before every file edit and every command execution. Read operations (Read, Glob, Grep) are allowed without prompts because they do not modify anything.

### When Prompts Appear

```
Allowed without prompt:
  ✓ Read file contents
  ✓ Search for files (Glob)
  ✓ Search file contents (Grep)
  ✓ Web search and fetch

Requires approval:
  ⚠ Edit existing files
  ⚠ Write new files
  ⚠ Execute bash commands
  ⚠ Edit Jupyter notebooks
```

### What the Prompt Looks Like

When Claude wants to edit a file:

```
Claude wants to edit src/auth/login.ts

  @@ -15,7 +15,10 @@
   export async function login(req: Request, res: Response) {
  -  const { email, password } = req.body;
  +  const { email, password } = req.body;
  +  if (!email || !password) {
  +    return res.status(400).json({ error: 'Email and password required' });
  +  }

Allow this edit? [y/n/always]
```

When Claude wants to run a command:

```
Claude wants to run: npm test

Allow? [y/n/always]
```

### Response Options

| Option | Effect |
|--------|--------|
| `y` | Allow this specific action |
| `n` | Deny this specific action |
| `always` | Allow all similar actions for the rest of the session |

### When to Use Default Mode

- You are new to Claude Code and learning how it works
- You are working on critical or production code
- You want to review every change before it happens
- You are demonstrating Claude Code to others

---

## 4. Auto-Accept Mode

Auto-Accept mode pre-approves all tool calls. Claude Code will read files, make edits, and run commands without asking. This is the fastest mode for iterative development but requires trust in both the model and the codebase.

### Starting Auto-Accept Mode

```bash
# From the command line
claude --mode auto-accept

# Or inside a session
> /mode auto-accept
```

### What Happens in Auto-Accept

```
> Fix the authentication bug and run the tests

Claude will automatically:
  1. Read relevant files (no prompt)
  2. Edit source files (no prompt)
  3. Run `npm test` (no prompt)
  4. If tests fail, read error output and fix (no prompt)
  5. Run tests again (no prompt)
  6. Report results to you
```

### Safety Considerations

Even in Auto-Accept mode, some safety measures remain:

1. **Deny rules still apply**: If you have deny rules in your settings, they are enforced regardless of mode
2. **Sandbox**: File operations are sandboxed to the project directory by default
3. **Network restrictions**: Outbound network calls from Bash are restricted by default

### When to Use Auto-Accept Mode

- You are working on a personal project with version control
- You want fast iteration (fix → test → fix → test)
- The project has good test coverage as a safety net
- You are doing a large refactoring and want to review changes at the end via `git diff`

### Risk Mitigation

```bash
# Before using auto-accept, ensure you have a clean git state
git status          # Should show no uncommitted changes
git stash           # Stash any work in progress

# Start auto-accept session
claude --mode auto-accept

# After the session, review all changes
git diff            # Review every change Claude made
git diff --stat     # Summary of files changed

# If something went wrong
git checkout .      # Revert all changes (destructive)
git stash pop       # Restore your stashed work
```

---

## 5. Plan Mode

Plan mode is **read-only**. Claude Code can read files, search the codebase, and analyze code, but it cannot make any edits or run any commands. This is ideal for analysis, architecture review, and planning sessions.

### Starting Plan Mode

```bash
claude --mode plan
```

### What Claude Can Do in Plan Mode

```
✓ Read files and understand code
✓ Search with Glob and Grep
✓ Analyze architecture and patterns
✓ Suggest changes (but not implement them)
✓ Create plans and action items
✓ Web search for reference information

✗ Edit or write files
✗ Run bash commands
✗ Execute tests or builds
✗ Make git commits
```

### Use Cases

**Architecture Review**:
```
> Analyze the dependency graph of this project. Identify any
  circular dependencies and suggest how to break them.
```

**Code Audit**:
```
> Review the authentication flow for security vulnerabilities.
  Check for SQL injection, XSS, CSRF, and insecure token handling.
```

**Planning**:
```
> I need to add multi-tenancy support. Analyze the current
  data model and propose a migration plan. List all files
  that would need to change.
```

**Onboarding**:
```
> Explain this project's architecture. How does a request
  flow from the API gateway to the database and back?
```

---

## 6. Don't Ask Mode

Don't Ask mode uses pre-configured rules to automatically approve or deny actions. Actions that match an allow rule are executed silently. Actions that match a deny rule are blocked. Actions that match neither are denied by default (fail-closed).

### Configuration

Configure Don't Ask mode through settings files:

```json
// .claude/settings.json
{
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Edit",
      "Write",
      "Bash(npm test)",
      "Bash(npm run lint)",
      "Bash(npm run build)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git add *)",
      "Bash(git commit *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Bash(git reset --hard *)",
      "Bash(curl *)",
      "Bash(wget *)",
      "Bash(sudo *)"
    ]
  }
}
```

### How Rule Matching Works

Rules are evaluated in this order:
1. Check deny list — if matched, **block** the action
2. Check allow list — if matched, **allow** the action
3. If no rule matches — **deny** the action (fail-closed)

```
Action: Bash("npm test")
  → Deny list: no match
  → Allow list: matches "Bash(npm test)"
  → Result: ALLOWED

Action: Bash("rm -rf /tmp/cache")
  → Deny list: matches "Bash(rm -rf *)"
  → Result: BLOCKED

Action: Bash("python script.py")
  → Deny list: no match
  → Allow list: no match
  → Result: DENIED (no matching rule)
```

### When to Use Don't Ask Mode

- You have a well-defined workflow with known commands
- You want automation without blanket approval
- Team environments where specific commands should always be allowed
- CI/CD pipelines with controlled command sets

---

## 7. Bypass Mode

Bypass mode disables all permission checks and sandboxing. Claude Code runs with the full privileges of your user account. This mode is designed **exclusively for isolated environments** — containers, virtual machines, and CI/CD runners.

### Starting Bypass Mode

```bash
# Only use in containers or VMs
claude --mode bypass
```

### What Bypass Disables

```
Disabled in Bypass Mode:
  ✗ Permission prompts (all actions auto-approved)
  ✗ File system sandboxing (can access any path)
  ✗ Network restrictions (can make outbound calls)
  ✗ Deny rules (even deny rules are ignored)
```

### Appropriate Environments

```
Safe for Bypass Mode:
  ✓ Docker containers (disposable)
  ✓ GitHub Actions runners
  ✓ CI/CD build agents
  ✓ Development VMs
  ✓ Cloud-based dev environments (Codespaces, Gitpod)

NEVER use Bypass Mode on:
  ✗ Your personal machine
  ✗ Production servers
  ✗ Shared workstations
  ✗ Any machine with sensitive data outside the project
```

### Docker Example

```dockerfile
FROM node:20-slim

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Set API key (use secrets in production)
ENV ANTHROPIC_API_KEY=sk-ant-api03-...

WORKDIR /app
COPY . .

# Run Claude Code in bypass mode inside the container
CMD ["claude", "--mode", "bypass", "-p", "Run all tests and fix any failures"]
```

---

## 8. Allow and Deny Rules

Rules provide fine-grained control over what Claude Code can do. They use a pattern-matching syntax that supports wildcards and specific tool targeting.

### Rule Syntax

```
ToolName(pattern)
```

Where:
- `ToolName` is the tool name: `Bash`, `Edit`, `Write`, `Read`, `Glob`, `Grep`
- `pattern` is a glob-style match for the tool's argument
- `*` matches any characters

### Bash Rules

```json
{
  "permissions": {
    "allow": [
      "Bash(npm test)",
      "Bash(npm test *)",
      "Bash(npm run lint)",
      "Bash(npx prettier *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(python -m pytest *)",
      "Bash(make *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push --force *)",
      "Bash(git reset --hard *)",
      "Bash(sudo *)",
      "Bash(curl *)",
      "Bash(wget *)",
      "Bash(pip install *)",
      "Bash(npm install *)"
    ]
  }
}
```

### Edit and Write Rules

```json
{
  "permissions": {
    "allow": [
      "Edit(src/**)",
      "Edit(tests/**)",
      "Write(tests/**)"
    ],
    "deny": [
      "Edit(.env*)",
      "Edit(*.pem)",
      "Edit(*.key)",
      "Write(.env*)",
      "Edit(package-lock.json)",
      "Edit(pnpm-lock.yaml)"
    ]
  }
}
```

### Read Rules

Even read operations can be restricted for sensitive files:

```json
{
  "permissions": {
    "deny": [
      "Read(.env*)",
      "Read(**/*.pem)",
      "Read(**/*.key)",
      "Read(**/credentials*)"
    ]
  }
}
```

### Complete Settings Example

Here is a comprehensive settings file for a Node.js project:

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Edit(src/**)",
      "Edit(tests/**)",
      "Edit(docs/**)",
      "Write(src/**)",
      "Write(tests/**)",
      "Bash(npm test)",
      "Bash(npm test *)",
      "Bash(npm run lint)",
      "Bash(npm run lint:fix)",
      "Bash(npm run build)",
      "Bash(npx prettier --write *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(git branch *)",
      "Bash(git checkout *)",
      "Bash(node -e *)",
      "Bash(npx tsc --noEmit)"
    ],
    "deny": [
      "Read(.env*)",
      "Read(**/*.pem)",
      "Edit(.env*)",
      "Edit(package-lock.json)",
      "Edit(node_modules/**)",
      "Write(.env*)",
      "Write(node_modules/**)",
      "Bash(rm -rf *)",
      "Bash(sudo *)",
      "Bash(git push *)",
      "Bash(git reset --hard *)",
      "Bash(npm publish *)",
      "Bash(curl *)",
      "Bash(wget *)"
    ]
  }
}
```

---

## 9. File Access Rules

Beyond tool-level permissions, Claude Code supports path-based rules that restrict which files Claude can read or edit, regardless of which tool it uses.

### Path Patterns

File access rules use glob patterns:

| Pattern | Matches |
|---------|---------|
| `src/**` | All files under `src/` recursively |
| `*.ts` | All TypeScript files in the current directory |
| `**/*.test.ts` | All test files anywhere in the project |
| `config/*.json` | JSON files directly in `config/` |
| `!.env*` | Exclude all `.env` files (deny pattern) |

### Practical Configurations

**Restrict Claude to specific directories**:

```json
{
  "permissions": {
    "allow": [
      "Edit(src/**)",
      "Edit(tests/**)"
    ],
    "deny": [
      "Edit(infrastructure/**)",
      "Edit(deployment/**)",
      "Edit(.github/**)"
    ]
  }
}
```

**Protect configuration files**:

```json
{
  "permissions": {
    "deny": [
      "Edit(*.config.js)",
      "Edit(*.config.ts)",
      "Edit(tsconfig.json)",
      "Edit(package.json)",
      "Edit(.eslintrc*)",
      "Edit(Dockerfile*)",
      "Edit(docker-compose*)"
    ]
  }
}
```

**Read-only access to certain files**:

```json
{
  "permissions": {
    "allow": [
      "Read(infrastructure/**)"
    ],
    "deny": [
      "Edit(infrastructure/**)",
      "Write(infrastructure/**)"
    ]
  }
}
```

---

## 10. Choosing the Right Mode

### Decision Flowchart

```
Is the environment disposable (container, VM)?
├── Yes → Bypass mode
│
└── No → Do you want Claude to make changes?
    ├── No → Plan mode (read-only)
    │
    └── Yes → Do you have well-defined rules?
        ├── Yes → Don't Ask mode (rule-based)
        │
        └── No → Do you trust the codebase has good tests?
            ├── Yes → Auto-Accept mode
            │
            └── No → Default mode (manual approval)
```

### Mode Comparison by Scenario

| Scenario | Recommended Mode | Rationale |
|----------|-----------------|-----------|
| First time with Claude Code | Default | Learn what Claude does before trusting it |
| Personal project, good tests | Auto-Accept | Fast iteration with test safety net |
| Team project, shared rules | Don't Ask | Consistent permissions across developers |
| Production hotfix | Default | Maximum control for critical changes |
| Architecture review | Plan | Read-only analysis, no accidental changes |
| CI/CD pipeline | Bypass | Isolated container, no human present |
| Codebase exploration | Plan | Understand code without changing it |
| Large refactoring | Auto-Accept | Too many changes to approve individually |
| Client code review | Plan | Read-only for security and liability |

### Switching Modes Mid-Session

You can change modes during a session:

```
> /mode plan
# Analyze the codebase in read-only mode

> /mode auto-accept
# Switch to auto-accept for implementation

> /mode default
# Switch back to default for sensitive changes
```

---

## 11. Security Best Practices

### Principle of Least Privilege

Grant Claude Code only the permissions it needs for the current task. Start restrictive and expand as needed.

```json
// Good: Specific permissions
{
  "permissions": {
    "allow": [
      "Bash(npm test)",
      "Bash(npm run lint)",
      "Edit(src/**)",
      "Edit(tests/**)"
    ]
  }
}

// Bad: Overly permissive
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Edit(*)",
      "Write(*)"
    ]
  }
}
```

### Protect Secrets

Always deny access to files containing secrets:

```json
{
  "permissions": {
    "deny": [
      "Read(.env*)",
      "Read(**/*.pem)",
      "Read(**/*.key)",
      "Read(**/credentials*)",
      "Read(**/secrets*)",
      "Edit(.env*)",
      "Write(.env*)"
    ]
  }
}
```

### Restrict Dangerous Commands

```json
{
  "permissions": {
    "deny": [
      "Bash(rm -rf *)",
      "Bash(sudo *)",
      "Bash(chmod 777 *)",
      "Bash(git push --force *)",
      "Bash(git reset --hard *)",
      "Bash(git clean -f *)",
      "Bash(> *)",
      "Bash(curl * | sh)",
      "Bash(curl * | bash)",
      "Bash(eval *)"
    ]
  }
}
```

### Use Version Control as a Safety Net

```bash
# Before starting a Claude Code session:
# 1. Ensure all changes are committed
git status  # Should be clean

# 2. Create a working branch
git checkout -b feature/claude-changes

# 3. After the session, review all changes
git diff main...HEAD

# 4. If something went wrong, you can always go back
git checkout main
git branch -D feature/claude-changes
```

### Team Security Checklist

For teams using Claude Code, establish shared security policies:

```markdown
## Claude Code Security Policy (example for your team)

1. **Committed settings**: All team members use `.claude/settings.json`
2. **Deny list minimum**: rm -rf, sudo, git push --force, curl|sh
3. **Secret protection**: .env, .pem, .key files denied in all operations
4. **Mode restrictions**: Bypass mode only in CI/CD containers
5. **Code review**: All Claude-generated changes go through PR review
6. **Audit**: Run `/cost` at end of each session; report anomalies
```

---

## 12. Exercises

### Exercise 1: Mode Selection

For each scenario, choose the appropriate permission mode and explain your reasoning:

1. You are exploring a new open-source project you just cloned
2. You are fixing a bug in your company's payment processing service
3. You are running Claude Code in a GitHub Actions workflow
4. You are doing a large refactoring of your personal blog codebase
5. A junior developer wants to use Claude Code for the first time

### Exercise 2: Write Permission Rules

Create a `.claude/settings.json` for a Python Django project with these requirements:

1. Allow running `python manage.py test` and `python manage.py migrate`
2. Allow editing files in `apps/` and `tests/` directories
3. Deny editing `settings.py`, `urls.py`, and any migration files
4. Deny all `pip install` commands (dependencies should be managed manually)
5. Allow git status, diff, and log but deny push and force operations

### Exercise 3: Security Audit

Review the following settings file and identify all security issues:

```json
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Edit(*)",
      "Write(*)",
      "Read(*)"
    ],
    "deny": [
      "Bash(rm -rf /)"
    ]
  }
}
```

List at least 5 problems and provide a corrected version.

### Exercise 4: Gradual Trust Escalation

Design a permission progression for a new team member using Claude Code:

- Week 1: What mode and rules?
- Week 2-3: How do you expand permissions?
- Month 2+: What is the steady-state configuration?

Document your reasoning for each phase.

---

## 13. Next Steps

You now understand how to control Claude Code's access to your system. The next lesson introduces **Hooks** — event-driven automation that runs shell commands in response to Claude Code actions. Hooks let you automate formatting, linting, testing, and notifications without any manual intervention, turning Claude Code into a fully automated development workflow.

**Next**: [05. Hooks and Event-Driven Automation](./05_Hooks.md)
