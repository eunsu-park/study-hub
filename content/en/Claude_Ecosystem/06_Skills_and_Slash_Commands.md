# Skills and Slash Commands

**Previous**: [05. Hooks and Event-Driven Automation](./05_Hooks.md) | **Next**: [07. Subagents and Task Delegation](./07_Subagents.md)

---

Skills are reusable instruction sets that extend Claude Code's capabilities for specific tasks. While CLAUDE.md provides general project context and hooks provide deterministic automation, skills fill the gap between the two — they are structured, named bundles of instructions that Claude follows when invoked. A `/commit` skill ensures every commit follows your team's format. A `/review` skill runs a consistent code review checklist. A `/deploy` skill executes your deployment workflow step by step. This lesson covers how to create, configure, and use skills effectively.

**Difficulty**: ⭐⭐

**Prerequisites**:
- [03. CLAUDE.md and Project Setup](./03_CLAUDE_md_and_Project_Setup.md)
- [05. Hooks and Event-Driven Automation](./05_Hooks.md)
- Familiarity with Markdown and YAML frontmatter

**Learning Objectives**:
- Understand the difference between skills, hooks, and CLAUDE.md instructions
- Create custom skills using the SKILL.md format
- Configure skill storage locations (global and project-level)
- Use built-in slash commands for session management
- Design skills for common development workflows
- Make informed decisions about when to use skills vs hooks vs CLAUDE.md

---

## Table of Contents

1. [What Are Skills?](#1-what-are-skills)
2. [Skills vs Hooks vs CLAUDE.md](#2-skills-vs-hooks-vs-claudemd)
3. [SKILL.md File Structure](#3-skillmd-file-structure)
4. [Skill Storage Locations](#4-skill-storage-locations)
5. [Creating Your First Skill](#5-creating-your-first-skill)
6. [Built-in Slash Commands](#6-built-in-slash-commands)
7. [Practical Skill Examples](#7-practical-skill-examples)
8. [Auto-Invoke Triggers](#8-auto-invoke-triggers)
9. [Invoking Skills](#9-invoking-skills)
10. [Skill Best Practices](#10-skill-best-practices)
11. [Exercises](#11-exercises)
12. [Next Steps](#12-next-steps)

---

## 1. What Are Skills?

A **skill** is a Markdown file with YAML frontmatter that contains structured instructions for Claude Code. When a skill is invoked — either manually by the user or automatically by a trigger — its contents are loaded into the conversation context, and Claude follows the instructions as part of its response.

Skills are useful because they:

- **Standardize workflows**: Every team member runs the same review process
- **Encode expertise**: Capture complex procedures that are hard to remember
- **Reduce repetition**: No need to type the same instructions every session
- **Are version-controlled**: Stored as files, committed to git, reviewed in PRs

```
Without skills:
  User: "Review this code. Check for security issues, performance
         problems, error handling, test coverage, and make sure it
         follows our style guide. Also check the SQL queries for
         injection vulnerabilities and..."

With skills:
  User: /review
  (Claude loads the complete review checklist automatically)
```

---

## 2. Skills vs Hooks vs CLAUDE.md

These three configuration mechanisms serve different purposes. Understanding when to use each is key to effective Claude Code configuration.

### Comparison Table

| Aspect | CLAUDE.md | Hooks | Skills |
|--------|-----------|-------|--------|
| **Format** | Markdown (free-form) | JSON (structured) | Markdown + YAML frontmatter |
| **When active** | Always (every session) | On tool events | When invoked |
| **Nature** | Context/suggestions | Deterministic commands | Structured instructions |
| **Enforcement** | Model discretion | Always executes | Model follows when loaded |
| **Invocation** | Automatic | Event-triggered | Manual or auto-trigger |
| **Scope** | Project-wide context | Specific tool actions | Specific tasks/workflows |

### Decision Guide

```
"Claude should always know this about the project"
  → CLAUDE.md

"This must happen every time a file is edited"
  → Hook (PostToolUse)

"When I ask for a code review, follow this checklist"
  → Skill

"Never allow rm -rf"
  → Settings (deny rule)

"Format code after every edit"
  → Hook (PostToolUse)

"When deploying, follow these 10 steps in order"
  → Skill
```

### Concrete Examples

| Requirement | Implementation |
|-------------|---------------|
| "We use 4-space indentation" | CLAUDE.md |
| "Run Prettier after every edit" | Hook |
| "Code reviews should check 8 specific areas" | Skill |
| "Always use conventional commits" | Skill (auto-invoke on commit) |
| "Block all sudo commands" | Settings (deny rule) |
| "Our API follows REST conventions" | CLAUDE.md |
| "Deployment requires 5 steps in sequence" | Skill |

---

## 3. SKILL.md File Structure

A skill file is a Markdown file with YAML frontmatter. The frontmatter defines metadata and trigger conditions. The body contains the instructions Claude follows when the skill is activated.

### Complete Structure

```markdown
---
name: skill-name
description: Brief description of what this skill does
auto_invoke:
  - when: "pattern or condition"
---

# Skill Title

Instructions for Claude to follow when this skill is invoked.

## Step 1: ...

Detailed instructions...

## Step 2: ...

More instructions...
```

### Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier for the skill (used in `/skill-name` invocation) |
| `description` | Yes | Brief description shown in skill listings |
| `auto_invoke` | No | Conditions under which the skill activates automatically |

### Example: Complete SKILL.md

```markdown
---
name: commit
description: Create a well-formatted conventional commit
---

# Commit Skill

When the user asks to commit changes, follow this process:

## 1. Review Changes

Run `git status` and `git diff --staged` to understand what is being committed. If nothing is staged, run `git diff` to see unstaged changes and ask the user what to stage.

## 2. Analyze the Changes

Categorize the changes:
- **feat**: New feature
- **fix**: Bug fix
- **refactor**: Code restructuring without behavior change
- **docs**: Documentation only
- **test**: Adding or modifying tests
- **chore**: Build, CI, dependency updates
- **style**: Formatting, whitespace, semicolons
- **perf**: Performance improvement

## 3. Write the Commit Message

Format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Rules:
- Subject line: imperative mood, lowercase, no period, under 50 characters
- Body: explain what and why (not how), wrap at 72 characters
- Footer: reference issues (Fixes #123, Closes #456)

## 4. Stage and Commit

- Stage only the relevant files (not `git add .` unless all changes are related)
- Create the commit with the formatted message
- Show the commit hash and summary

## 5. Verify

Run `git log -1 --oneline` to confirm the commit was created correctly.
```

---

## 4. Skill Storage Locations

Skills can be stored at two levels, similar to settings files.

### Global Skills (~/.claude/skills/)

Skills stored here are available in all projects on your machine.

```
~/.claude/
├── settings.json
└── skills/
    ├── commit.md       # Available everywhere
    ├── review.md       # Available everywhere
    └── daily.md        # Available everywhere
```

### Project Skills (.claude/skills/)

Skills stored here are specific to the current project and can be committed to version control.

```
my-project/
├── .claude/
│   ├── settings.json
│   └── skills/
│       ├── deploy.md       # Project-specific deployment
│       ├── test.md         # Project-specific test workflow
│       └── release.md      # Project-specific release process
├── CLAUDE.md
└── src/
```

### Precedence

If both a global and project skill have the same name, the project skill takes precedence:

```
~/.claude/skills/commit.md          ← Global default
my-project/.claude/skills/commit.md ← Project override (wins)
```

### Listing Available Skills

```
> /help

Available skills:
  /commit   - Create a well-formatted conventional commit
  /review   - Run code review checklist
  /deploy   - Deploy to staging or production
  /test     - Run test suite with coverage
```

---

## 5. Creating Your First Skill

Let us walk through creating a skill from scratch.

### Step 1: Decide What the Skill Does

We will create a `/test` skill that runs the project's test suite, analyzes failures, and helps fix them.

### Step 2: Create the Directory

```bash
# Create project skills directory
mkdir -p .claude/skills
```

### Step 3: Write the SKILL.md

Create `.claude/skills/test.md`:

```markdown
---
name: test
description: Run tests, analyze failures, and suggest fixes
---

# Test Skill

Run the project's test suite and help resolve any failures.

## Process

### 1. Run the Full Test Suite

Execute the project's test command:
- If `package.json` exists: `npm test`
- If `pyproject.toml` or `setup.py` exists: `python -m pytest -v`
- If `Cargo.toml` exists: `cargo test`
- If `go.mod` exists: `go test ./...`
- If a `Makefile` has a `test` target: `make test`

### 2. Report Results

Provide a clear summary:
- Total tests: passed / failed / skipped
- Execution time
- If all pass, say so and stop

### 3. Analyze Failures

For each failing test:
1. Read the test file to understand what it expects
2. Read the source code being tested
3. Identify the root cause of the failure
4. Propose a fix with a clear explanation

### 4. Fix (with Approval)

Ask the user: "Should I fix these failures?"
- If yes: apply the fixes and re-run tests
- If no: provide the analysis as a report

### 5. Coverage (Optional)

If the user asks for coverage:
- Run with coverage: `pytest --cov=src` or `npm test -- --coverage`
- Identify files with low coverage
- Suggest test cases for uncovered code paths
```

### Step 4: Test the Skill

```bash
claude
```

```
> /test

Claude loads the skill instructions and follows the process:
1. Detects the project type
2. Runs the appropriate test command
3. Reports results
4. Analyzes any failures
```

---

## 6. Built-in Slash Commands

Claude Code comes with built-in slash commands that are always available, regardless of your skill configuration. These are not skills — they are native commands that control the Claude Code session.

### Command Reference

| Command | Description | Details |
|---------|-------------|---------|
| `/help` | Show help information | Lists all commands, skills, and usage tips |
| `/exit` | Exit the session | Ends the current Claude Code session |
| `/clear` | Clear conversation | Resets conversation history completely |
| `/compact` | Compact conversation | Summarizes conversation to free context space |
| `/cost` | Show session costs | Displays token usage and estimated cost |
| `/doctor` | Run diagnostics | Checks configuration, auth, model access, hooks |
| `/init` | Initialize project | Creates a CLAUDE.md based on project analysis |
| `/model` | View/change model | Shows current model or switches to another |
| `/config` | View/edit configuration | Shows merged settings from all levels |
| `/mode` | View/change permission mode | Switches between default, auto-accept, plan, etc. |

### /compact: Managing Context

As your session grows, the context window fills up. `/compact` summarizes the conversation, preserving key information while freeing tokens.

```
> /compact

Compacting conversation...
  Before: 87,432 tokens
  After:  12,847 tokens
  Freed:  74,585 tokens (85%)

Summary preserved:
  - Working on user authentication module
  - Fixed login rate limiting bug
  - Added input validation to registration
  - Tests are passing (47/47)
```

**When to use /compact**:
- You notice Claude forgetting earlier context
- The `/cost` command shows high token usage
- You are switching to a different task within the same session
- The session has been running for over 30 minutes

### /init: Bootstrapping a Project

```
> /init

Claude will analyze your project and generate CLAUDE.md:

Detected:
  - Language: TypeScript
  - Framework: Next.js 14
  - Package manager: pnpm
  - Test framework: Jest + Playwright
  - Linter: ESLint + Prettier
  - Database: PostgreSQL (Prisma ORM)

Generated CLAUDE.md with:
  - Project structure overview
  - Tech stack details
  - Development commands
  - Coding standards (inferred from config files)
  - Testing instructions

Review the generated file? (y/n)
```

### /doctor: Troubleshooting

```
> /doctor

Running diagnostics...

Authentication:
  ✓ API key: Valid (sk-ant-...7x2m)
  ✓ Account: Active, Claude for Work plan

Model Access:
  ✓ claude-opus-4-20250514: Available
  ✓ claude-sonnet-4-20250514: Available
  ✓ claude-haiku-3-20241022: Available

Configuration:
  ✓ Global settings: ~/.claude/settings.json (valid)
  ✓ Project settings: .claude/settings.json (valid)
  ✓ CLAUDE.md: Found (1,247 tokens)

Hooks:
  ✓ PostToolUse: 2 hooks configured
  ✗ Hook "prettier" test: npx prettier not found
    Suggestion: Run `npm install --save-dev prettier`

Skills:
  ✓ 3 project skills: commit, review, test
  ✓ 1 global skill: daily

Network:
  ✓ API connection: 142ms latency

1 issue found.
```

---

## 7. Practical Skill Examples

### Skill: Code Review (/review)

```markdown
---
name: review
description: Comprehensive code review following team standards
---

# Code Review Skill

Perform a thorough code review of the specified files or recent changes.

## Scope

If the user specifies files, review those files. Otherwise:
1. Run `git diff --name-only HEAD~1` to find recently changed files
2. Run `git diff HEAD~1` to see the actual changes
3. Review only the changed code (not the entire file)

## Review Checklist

### 1. Correctness
- Does the code do what it claims to do?
- Are there off-by-one errors, null pointer risks, or race conditions?
- Are error cases handled?
- Are edge cases considered?

### 2. Security
- Any SQL injection, XSS, or CSRF vulnerabilities?
- Are inputs validated and sanitized?
- Are secrets hardcoded?
- Are authentication/authorization checks present where needed?

### 3. Performance
- Any N+1 query problems?
- Are there unnecessary loops or computations?
- Is caching used where appropriate?
- Are large datasets paginated?

### 4. Readability
- Are variable and function names descriptive?
- Is the code self-documenting?
- Are complex sections commented?
- Is the code DRY (Don't Repeat Yourself)?

### 5. Testing
- Do tests exist for the new/changed code?
- Are edge cases tested?
- Are test names descriptive?

### 6. Architecture
- Does the code follow the project's architectural patterns?
- Are dependencies flowing in the right direction?
- Is the code in the right module/layer?

## Output Format

For each finding, provide:
- **Severity**: Critical / Warning / Suggestion / Nitpick
- **Location**: File and line reference
- **Issue**: Clear description of the problem
- **Suggestion**: Specific recommendation with code example

End with a summary: total findings by severity and overall assessment.
```

### Skill: Deployment (/deploy)

```markdown
---
name: deploy
description: Deploy application to staging or production
---

# Deployment Skill

Guide the deployment process step by step.

## Pre-Deployment Checks

Before deploying, verify:

1. **Branch**: Must be on `main` (production) or `develop` (staging)
   - Run `git branch --show-current`
   - If on the wrong branch, warn and stop

2. **Clean state**: No uncommitted changes
   - Run `git status`
   - If dirty, warn and stop

3. **Tests**: All tests must pass
   - Run `npm test`
   - If any fail, warn and stop

4. **Build**: Project must build successfully
   - Run `npm run build`
   - If build fails, warn and stop

## Deployment Steps

### Staging (develop branch)

```bash
# 1. Pull latest
git pull origin develop

# 2. Build
npm run build

# 3. Deploy to staging
npm run deploy:staging

# 4. Run smoke tests
npm run test:smoke -- --env=staging

# 5. Report
echo "Staging deployment complete"
```

### Production (main branch)

```bash
# 1. Create release tag
VERSION=$(node -p "require('./package.json').version")
git tag -a "v$VERSION" -m "Release v$VERSION"

# 2. Build
NODE_ENV=production npm run build

# 3. Deploy
npm run deploy:production

# 4. Run smoke tests
npm run test:smoke -- --env=production

# 5. Push tags
git push origin "v$VERSION"

# 6. Report
echo "Production deployment v$VERSION complete"
```

## Post-Deployment

1. Verify the deployment URL is responding
2. Check error monitoring (if configured)
3. Report success or failure to the user
```

### Skill: Documentation (/docs)

```markdown
---
name: docs
description: Generate or update documentation for code changes
---

# Documentation Skill

Generate or update documentation based on recent code changes.

## Process

### 1. Identify What Changed

Run `git diff --name-only HEAD~1` to see changed files.

### 2. Update JSDoc/Docstrings

For each changed function:
- Ensure it has a docstring/JSDoc comment
- Update parameter descriptions if signatures changed
- Update return type descriptions
- Add @throws/@raises for new error conditions
- Add usage examples for public API functions

### 3. Update README

If the changes affect:
- Installation steps → update README installation section
- API endpoints → update API documentation
- Configuration → update configuration section
- Dependencies → update prerequisites

### 4. Update CHANGELOG

Add an entry under "Unreleased" section:
```markdown
## [Unreleased]

### Added
- Description of new features

### Changed
- Description of changes

### Fixed
- Description of bug fixes
```

### 5. Verify Links

Check that all internal links in documentation are valid:
- Relative file links exist
- Anchor links match headings
- External URLs are accessible
```

---

## 8. Auto-Invoke Triggers

Skills can be configured to activate automatically when certain conditions are met, without the user explicitly typing the slash command.

### Auto-Invoke Configuration

```markdown
---
name: commit
description: Standardized commit messages
auto_invoke:
  - when: "user asks to commit"
  - when: "user mentions git commit"
---
```

The `when` field contains a natural language description of the trigger condition. Claude evaluates whether the user's message matches the condition and loads the skill if it does.

### Examples of Auto-Invoke Triggers

**Commit skill**:

```yaml
auto_invoke:
  - when: "user asks to commit changes"
  - when: "user wants to create a git commit"
```

**Review skill**:

```yaml
auto_invoke:
  - when: "user asks for a code review"
  - when: "user wants to review changes or a PR"
```

**Test skill**:

```yaml
auto_invoke:
  - when: "user asks to run tests"
  - when: "user mentions test failures"
```

### When to Use Auto-Invoke

Auto-invoke is best for skills that correspond to common tasks the user frequently requests in natural language:

```
Good candidates for auto-invoke:
  ✓ Commit conventions (triggered by "commit this", "save changes")
  ✓ Code review (triggered by "review this", "check the code")
  ✓ Test runner (triggered by "run tests", "check if tests pass")

Poor candidates for auto-invoke:
  ✗ Deployment (should be explicit and intentional)
  ✗ Database migration (too risky for accidental triggering)
  ✗ Release process (requires deliberate initiation)
```

---

## 9. Invoking Skills

### Manual Invocation

Type the skill name as a slash command:

```
> /commit

Claude loads the commit skill and follows its instructions.
```

```
> /review src/auth/login.ts

Claude loads the review skill and applies it to the specified file.
```

```
> /deploy staging

Claude loads the deploy skill with "staging" as an argument.
```

### Skill Arguments

When you type text after the slash command, it is passed to Claude as additional context. The skill instructions plus your arguments together guide Claude's behavior.

```
> /review --focus security

Claude loads the review skill and emphasizes security checks.
```

```
> /test tests/unit/auth/

Claude loads the test skill and focuses on the specified test directory.
```

### Listing Skills

```
> /help

Built-in commands:
  /help     - Show this help message
  /exit     - End session
  /clear    - Clear conversation
  /compact  - Compact conversation
  /cost     - Show costs
  /doctor   - Run diagnostics
  /init     - Initialize CLAUDE.md
  /model    - View/change model
  /config   - View/edit config
  /mode     - View/change mode

Project skills:
  /commit   - Create a well-formatted conventional commit
  /review   - Comprehensive code review following team standards
  /deploy   - Deploy application to staging or production
  /test     - Run tests, analyze failures, and suggest fixes

Global skills:
  /daily    - Morning standup summary
```

---

## 10. Skill Best Practices

### Keep Skills Focused

Each skill should do one thing well. If a skill is trying to do too much, split it into multiple skills.

```
# Bad: one giant "dev" skill that does everything
/dev → runs tests, reviews code, deploys, writes docs

# Good: focused skills
/test    → runs tests and analyzes failures
/review  → reviews code changes
/deploy  → deploys to environments
/docs    → generates documentation
```

### Use Clear Step-by-Step Instructions

Claude follows skills more reliably when instructions are structured as numbered steps with clear actions.

```markdown
# Good: Clear, actionable steps
## Process

1. Run `git status` to check the working tree
2. If there are unstaged changes, ask the user what to stage
3. Run `git diff --staged` to review staged changes
4. Generate a commit message following the format below
5. Create the commit
6. Verify with `git log -1`

# Bad: Vague instructions
## Process

Make a good commit. Follow best practices. Make sure it looks right.
```

### Include Examples in Skills

When a skill involves generating output in a specific format, include examples:

```markdown
## Commit Message Format

```
feat(auth): add rate limiting to login endpoint

Limit login attempts to 5 per 15-minute window to prevent
brute force attacks. Uses express-rate-limit middleware.

Closes #142
```

```
fix(api): handle null email in user registration

The registration endpoint crashed when email was null.
Added explicit null check with a 400 response.

Fixes #256
```
```

### Version Control Your Skills

```bash
# Add skills to version control
git add .claude/skills/
git commit -m "Add team skills: commit, review, test, deploy"

# Team members get skills automatically on pull
git pull
# Skills are immediately available in their Claude Code sessions
```

### Document Skill Expectations

Add a brief comment at the top of each skill explaining when to use it:

```markdown
---
name: release
description: Create and publish a new release
---

# Release Skill

> Use this skill when you are ready to publish a new version.
> This will create a git tag, build the package, and publish to npm.
> **WARNING**: This pushes to the remote and publishes publicly.

## Pre-Release Checks
...
```

### Test Your Skills

After creating a skill, test it in a few different scenarios:

```
# Test with explicit invocation
> /commit

# Test with arguments
> /commit --amend

# Test auto-invoke (if configured)
> Can you commit these changes?

# Test edge cases
> /review  (with no changes to review)
> /deploy  (with failing tests)
```

---

## 11. Exercises

### Exercise 1: Create a Commit Skill

Create a `.claude/skills/commit.md` that:

1. Checks `git status` for staged changes
2. Analyzes the diff to determine the commit type (feat, fix, refactor, etc.)
3. Generates a conventional commit message
4. Includes a scope based on the primary directory affected
5. Adds a body for non-trivial changes
6. References related issues if mentioned in the conversation

Test it by making a small change and running `/commit`.

### Exercise 2: Create a Review Skill

Create a `.claude/skills/review.md` that:

1. Accepts a file path or defaults to recent git changes
2. Checks for at least 5 specific code quality areas
3. Rates each finding by severity (Critical, Warning, Suggestion)
4. Outputs a structured report
5. Optionally offers to fix Critical and Warning issues

### Exercise 3: Create a Refactor Skill

Create a `.claude/skills/refactor.md` that:

1. Takes a function or module name as argument
2. Reads the current implementation
3. Identifies code smells (long functions, deep nesting, duplication)
4. Proposes refactoring steps
5. Applies changes one at a time, running tests after each
6. Reports what was changed and why

### Exercise 4: Skills Audit

Review your project's configuration and identify:

1. Three tasks you do repeatedly that could become skills
2. Instructions currently in CLAUDE.md that would work better as skills
3. Hooks that could be replaced by or combined with skills
4. Skills that should be global vs project-specific

---

## 12. Next Steps

Skills give you reusable instruction bundles for common workflows. But some tasks are too large or complex for a single Claude Code instance to handle efficiently. The next lesson covers **Subagents** — Claude Code's ability to spawn child agents that work on subtasks in parallel, enabling complex multi-file operations, large-scale refactoring, and coordinated development workflows.

**Next**: [07. Subagents and Task Delegation](./07_Subagents.md)
