# Subagents and Task Delegation

**Previous**: [06. Skills and Slash Commands](./06_Skills_and_Slash_Commands.md) | **Next**: [08. Agent Teams](./08_Agent_Teams.md)

---

One of Claude Code's most powerful capabilities is launching **subagents** — specialized AI agents that operate independently within their own context windows. Instead of trying to hold an entire complex task in a single conversation, you can delegate work to focused subagents that explore, plan, or execute tasks in parallel. This lesson covers the mechanics, types, and best practices for subagent-based workflows.

**Difficulty**: ⭐⭐

**Prerequisites**:
- Lesson 02: Claude Code Getting Started
- Lesson 03: CLAUDE.md and Project Setup
- Basic understanding of context windows and token limits

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand what subagents are and how they differ from the main Claude Code session
2. Explain context isolation and why it matters for complex tasks
3. Distinguish between the three built-in subagent types (Explore, Plan, General-Purpose)
4. Use the Task tool to launch subagents with appropriate parameters
5. Run multiple subagents in parallel for independent tasks
6. Use background agents and retrieve their output
7. Create custom agent definitions using the /agents command
8. Write effective YAML configurations for custom agents
9. Decide when to use subagents vs. doing work directly in the main session

---

## Table of Contents

1. [What Are Subagents?](#1-what-are-subagents)
2. [Context Isolation](#2-context-isolation)
3. [Built-in Subagent Types](#3-built-in-subagent-types)
4. [The Task Tool](#4-the-task-tool)
5. [Running Subagents in Parallel](#5-running-subagents-in-parallel)
6. [Background Agents](#6-background-agents)
7. [Custom Agent Definitions](#7-custom-agent-definitions)
8. [When to Use Subagents vs. Direct Work](#8-when-to-use-subagents-vs-direct-work)
9. [Best Practices](#9-best-practices)
10. [Common Patterns and Recipes](#10-common-patterns-and-recipes)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. What Are Subagents?

A **subagent** is a separate Claude instance launched from your main Claude Code session to perform a specific task. Think of it like delegating work to a colleague: you give them a clear brief, they work independently, and they report back with results.

```
┌─────────────────────────────────────────────────────┐
│                  Main Claude Code Session            │
│                                                     │
│   "Refactor this codebase to use dependency         │
│    injection throughout"                            │
│                                                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│   │ Subagent │  │ Subagent │  │ Subagent │         │
│   │    #1    │  │    #2    │  │    #3    │         │
│   │ Explore  │  │ General  │  │ General  │         │
│   │ codebase │  │ Refactor │  │ Refactor │         │
│   │ structure│  │ module A │  │ module B │         │
│   └──────────┘  └──────────┘  └──────────┘         │
│        │              │              │              │
│        ▼              ▼              ▼              │
│   [Analysis]    [Modified A]   [Modified B]         │
└─────────────────────────────────────────────────────┘
```

Key characteristics of subagents:

- **Independent context**: Each subagent has its own context window, separate from the main session
- **Focused scope**: A subagent works on one well-defined task
- **Tool access**: Different subagent types have access to different tool sets
- **Parallel execution**: Multiple subagents can run simultaneously
- **Result reporting**: Subagents return their results to the main session

### Why Subagents Exist

The primary motivation is **context window management**. Claude Code has a finite context window (typically 200K tokens). When working on a large codebase, a single session can quickly fill its context with file contents, search results, and conversation history. Subagents solve this by:

1. **Offloading exploration**: A subagent reads dozens of files, synthesizes findings, and returns a concise summary — saving the main session from consuming tokens on raw file contents
2. **Parallelizing work**: Independent tasks run simultaneously instead of sequentially
3. **Isolating complexity**: Each subagent focuses on one thing, reducing the cognitive load on the main session

---

## 2. Context Isolation

Context isolation is the defining architectural feature of subagents. Each subagent operates in its own context window with no shared memory with the parent session or sibling subagents.

### What the Subagent Receives

When you launch a subagent, it receives:

1. **Your prompt**: The task description you provide
2. **System instructions**: Based on the subagent type (Explore, Plan, or General-Purpose)
3. **CLAUDE.md contents**: The project's CLAUDE.md file is automatically loaded
4. **Tool definitions**: The set of tools available to that subagent type

### What the Subagent Does NOT Receive

- The conversation history from the main session
- Results from other subagents (unless you explicitly include them in the prompt)
- Any files the main session has already read (the subagent must read them again)
- The main session's "understanding" of the codebase

```
┌─────────────────────┐      ┌─────────────────────┐
│   Main Session      │      │   Subagent          │
│                     │      │                     │
│ ✓ Full conversation │      │ ✗ No conversation   │
│   history           │      │   history           │
│ ✓ All tool results  │      │ ✗ Only its own      │
│ ✓ User preferences  │      │   tool results      │
│ ✓ Previous edits    │      │ ✓ CLAUDE.md loaded  │
│                     │      │ ✓ Prompt from parent│
│                     │──────│                     │
│                     │prompt│                     │
│                     │──────│                     │
│                     │◀─────│                     │
│                     │result│                     │
└─────────────────────┘      └─────────────────────┘
```

### Implications of Context Isolation

**Advantage**: Subagents start fresh, so they do not carry irrelevant context that might confuse their work.

**Challenge**: You must provide sufficient context in the prompt. If the main session discovered that "the authentication module is in `src/auth/` and uses JWT tokens," the subagent does not know this unless you tell it.

```
# Bad: Assumes subagent knows context from the main session
Task(prompt="Fix the bug we discussed in the auth module")

# Good: Self-contained prompt with all necessary context
Task(prompt="""Fix the JWT token validation bug in src/auth/validate.py.
The issue: tokens with expired 'nbf' (not before) claims are being
accepted. The validate_token() function on line 45 checks 'exp' but
not 'nbf'. Add 'nbf' validation with a 30-second clock skew tolerance.
Run the existing tests in tests/test_auth.py to verify the fix.""")
```

---

## 3. Built-in Subagent Types

Claude Code provides three built-in subagent types, each with a different tool set and purpose.

### 3.1 Explore Subagent

The **Explore** subagent is a fast, read-only agent optimized for codebase exploration and analysis. It cannot modify any files.

**Available Tools**:
- `Glob` — Find files by pattern
- `Grep` — Search file contents
- `Read` — Read file contents

**Use Cases**:
- Understanding codebase structure
- Finding all usages of a function or class
- Analyzing dependencies between modules
- Mapping out test coverage
- Generating summaries of code areas

```
# Example: Use an Explore subagent to understand a module
Task(
    subagent_type="explore",
    prompt="""Analyze the database module in src/database/.

    I need to understand:
    1. What ORM is used and how models are defined
    2. How database connections are managed (pool? singleton?)
    3. What migration system is in place
    4. How queries are constructed (raw SQL? query builder? ORM?)
    5. Any performance concerns (N+1 queries, missing indexes)

    Provide a structured summary with file paths for key components."""
)
```

**Why use Explore over doing it yourself?** The Explore subagent can read dozens of files across the codebase, consuming tokens in its own context window. It then returns a concise summary to the main session, saving you significant context space.

### 3.2 Plan Subagent

The **Plan** subagent is designed for research and architectural planning. It has read-only access to the codebase plus web access for looking up documentation and best practices.

**Available Tools**:
- `Glob` — Find files by pattern
- `Grep` — Search file contents
- `Read` — Read file contents
- `WebSearch` — Search the web
- `WebFetch` — Fetch and process web pages

**Use Cases**:
- Researching the best approach for a feature
- Looking up library documentation
- Creating implementation plans with step-by-step instructions
- Comparing architectural options
- Investigating error messages or stack traces

```
# Example: Use a Plan subagent to research a migration strategy
Task(
    subagent_type="plan",
    prompt="""We need to migrate our Express.js API from REST to GraphQL.

    Current setup:
    - Express.js 4.x with 47 REST endpoints in src/routes/
    - PostgreSQL with Sequelize ORM in src/models/
    - Authentication via JWT middleware in src/middleware/auth.js

    Research and create a migration plan:
    1. Evaluate Apollo Server vs Mercurius vs graphql-yoga (check latest docs)
    2. Analyze our current route handlers to identify GraphQL type candidates
    3. Propose a phased migration plan (REST + GraphQL coexistence)
    4. Identify risks and mitigation strategies
    5. Estimate the scope of changes needed

    Read the relevant source files to ground your analysis in our actual code."""
)
```

### 3.3 General-Purpose Subagent

The **General-Purpose** subagent has full access to all tools, including file editing, terminal commands, and web access. It can make changes to the codebase.

**Available Tools**:
- All read tools (Glob, Grep, Read)
- All write tools (Edit, Write)
- `Bash` — Execute terminal commands
- `WebSearch` / `WebFetch` — Web access
- `NotebookEdit` — Edit Jupyter notebooks
- And any MCP tools configured in the project

**Use Cases**:
- Implementing features or bug fixes
- Running tests and fixing failures
- Creating new files
- Refactoring code
- Installing dependencies
- Any task that requires modifying the codebase

```
# Example: Use a General-Purpose subagent to implement a feature
Task(
    subagent_type="general-purpose",
    prompt="""Add rate limiting to the API endpoints in src/routes/.

    Requirements:
    1. Use the 'express-rate-limit' package (install it)
    2. Apply a default rate limit of 100 requests per 15 minutes
    3. Apply a stricter limit of 10 requests per minute for auth endpoints
    4. Store rate limit data in Redis (connection config is in src/config.js)
    5. Return standard 429 responses with Retry-After header
    6. Add rate limit headers to all responses (X-RateLimit-*)
    7. Write tests in tests/middleware/test_rate_limit.js
    8. Run the test suite and ensure all tests pass

    Use the existing middleware pattern in src/middleware/ as a template."""
)
```

### Comparison Table

| Feature | Explore | Plan | General-Purpose |
|---------|---------|------|-----------------|
| Read files | Yes | Yes | Yes |
| Search codebase | Yes | Yes | Yes |
| Web search | No | Yes | Yes |
| Edit files | No | No | Yes |
| Run commands | No | No | Yes |
| Write new files | No | No | Yes |
| Speed | Fastest | Fast | Normal |
| Token cost | Lowest | Medium | Highest |
| Model default | sonnet | sonnet | sonnet |

---

## 4. The Task Tool

The **Task** tool is the mechanism for launching subagents from the main Claude Code session. Here is a detailed look at its parameters.

### Task Tool Parameters

```python
Task(
    prompt: str,              # Required: The task description
    subagent_type: str,       # "explore" | "plan" | "general-purpose"
    model: str,               # "opus" | "sonnet" | "haiku" (default: "sonnet")
    max_turns: int,           # Maximum conversation turns (default varies)
    run_in_background: bool,  # Run asynchronously (default: false)
)
```

### Parameter Details

#### `prompt` (Required)

The most important parameter. A well-crafted prompt is the difference between a successful subagent and a wasted invocation. Include:

- **Clear objective**: What should the subagent accomplish?
- **Context**: Relevant file paths, architecture decisions, constraints
- **Expected output**: What should the subagent return?
- **Boundaries**: What should it NOT do?

```python
# Detailed, self-contained prompt
Task(
    prompt="""Search the codebase for all usages of the deprecated
    'request' library (npm package). For each usage:
    1. Note the file path and line number
    2. Identify what HTTP method and URL is being called
    3. Note any special options (headers, auth, timeout)

    Return a structured list that I can use to plan migration to 'fetch'.
    Do NOT modify any files — this is analysis only.""",
    subagent_type="explore"
)
```

#### `subagent_type`

Determines the tool set available to the subagent (see Section 3). Choose the most restrictive type that can accomplish the task:

- Need to analyze code? Use `explore`
- Need to research + analyze? Use `plan`
- Need to change code or run commands? Use `general-purpose`

#### `model`

Select the model that runs the subagent:

- **`sonnet`** (default): Good balance of speed, cost, and capability. Use for most tasks.
- **`opus`**: Most capable model. Use for complex reasoning, architectural decisions, or tasks requiring exceptional quality.
- **`haiku`**: Fastest and cheapest. Use for simple, well-defined tasks like formatting or straightforward search-and-replace.

```python
# Use opus for complex architectural analysis
Task(
    prompt="Design the database schema for a multi-tenant SaaS...",
    subagent_type="plan",
    model="opus"
)

# Use haiku for simple, mechanical tasks
Task(
    prompt="Find all Python files that import 'os.path' ...",
    subagent_type="explore",
    model="haiku"
)
```

#### `max_turns`

Controls how many conversation turns the subagent can take. Each "turn" is one cycle of the agent receiving input and producing output (including tool calls). A higher value allows the subagent to do more work but consumes more tokens.

- Default varies by subagent type
- Set lower for simple tasks to limit cost
- Set higher for complex multi-step tasks

```python
# Simple search task — limit turns
Task(
    prompt="Find all TODO comments in src/",
    subagent_type="explore",
    max_turns=3
)

# Complex implementation — allow more turns
Task(
    prompt="Implement the payment processing module...",
    subagent_type="general-purpose",
    max_turns=30
)
```

#### `run_in_background`

When set to `true`, the subagent runs asynchronously. The main session continues working while the subagent processes in the background. See Section 6 for details.

---

## 5. Running Subagents in Parallel

One of the most powerful patterns is launching multiple subagents simultaneously for independent tasks. This dramatically reduces total execution time.

### When to Parallelize

Subagents should be parallelized when:
- Tasks are **independent** (no subagent depends on another's output)
- Tasks operate on **different files or modules**
- You want to **gather information** from multiple areas simultaneously

### Parallel Launch Pattern

```python
# The main session can launch multiple Task calls simultaneously.
# These will run in parallel, not sequentially.

# Subagent 1: Analyze frontend
Task(
    subagent_type="explore",
    prompt="Analyze the React component hierarchy in src/components/. "
           "Map parent-child relationships and identify prop drilling."
)

# Subagent 2: Analyze backend
Task(
    subagent_type="explore",
    prompt="Analyze the API route structure in src/api/. "
           "List all endpoints, their HTTP methods, and middleware chain."
)

# Subagent 3: Analyze database
Task(
    subagent_type="explore",
    prompt="Analyze the database models in src/models/. "
           "Map relationships between tables and identify indexes."
)
```

### Real-World Example: Multi-File Translation

The study project you are reading uses parallel subagents to translate content from English to Korean:

```python
# Phase 2 of the 3-Phase Content Creation Workflow
# Multiple Sonnet agents translate files in parallel

# Agent 1: Translate lessons 1-2
Task(
    subagent_type="general-purpose",
    model="sonnet",
    prompt="""Translate the following English lesson to Korean:
    - /content/en/Topic/01_Lesson.md → /content/ko/Topic/01_Lesson.md
    - /content/en/Topic/02_Lesson.md → /content/ko/Topic/02_Lesson.md

    Rules:
    - Do NOT translate code blocks
    - Preserve URLs and file paths
    - Use English terms in parentheses: "경사 하강법(Gradient Descent)"
    - Maintain table structure"""
)

# Agent 2: Translate lessons 3-4 (runs in parallel with Agent 1)
Task(
    subagent_type="general-purpose",
    model="sonnet",
    prompt="""Translate the following English lesson to Korean:
    - /content/en/Topic/03_Lesson.md → /content/ko/Topic/03_Lesson.md
    - /content/en/Topic/04_Lesson.md → /content/ko/Topic/04_Lesson.md
    ... (same rules) ..."""
)

# Agents 3-N: More parallel translation tasks...
```

### Anti-Pattern: Parallelizing Dependent Tasks

Do NOT parallelize tasks that depend on each other:

```python
# BAD: Agent 2 needs Agent 1's output
Task(prompt="Analyze the codebase structure...")     # Agent 1
Task(prompt="Refactor based on the analysis...")     # Agent 2 — will fail!

# GOOD: Sequential execution for dependent tasks
result = Task(prompt="Analyze the codebase structure...")  # Agent 1
Task(prompt=f"Refactor based on this analysis: {result}")  # Agent 2
```

---

## 6. Background Agents

Background agents run asynchronously, allowing the main session to continue other work while the subagent processes.

### Launching a Background Agent

```python
Task(
    subagent_type="general-purpose",
    prompt="Run the full test suite and generate a coverage report...",
    run_in_background=True
)
```

When `run_in_background=True`:
1. The main session launches the subagent
2. The main session **continues immediately** without waiting
3. When the background agent finishes, the main session is **notified**
4. The main session can then read the output

### Use Cases for Background Agents

- **Long-running tests**: Launch the test suite in the background while continuing to code
- **Code generation**: Generate boilerplate files in the background
- **Research tasks**: Have a Plan subagent research a topic while you work on something else
- **Build processes**: Kick off a build and continue editing

### Checking Background Agent Output

When a background agent completes, its result becomes available to the main session. The notification includes the subagent's final output, which you can then use in your ongoing work.

```
┌──────────────────────────────────────────────────────────┐
│ Main Session Timeline                                    │
│                                                          │
│ t=0   Launch background agent ──────────────┐            │
│ t=1   Continue working on feature X         │            │
│ t=2   Edit src/components/Header.tsx        │ Background │
│ t=3   Edit src/styles/header.css            │ agent      │
│ t=4   ...                                   │ running    │
│ t=5   ◀── Background agent completed ───────┘            │
│       "All 47 tests passed. Coverage: 89%"               │
│ t=6   Incorporate results and continue                   │
└──────────────────────────────────────────────────────────┘
```

### Limitations of Background Agents

- You cannot send additional input to a background agent after launch
- Background agents have the same tool restrictions as their subagent type
- If a background agent fails, you are notified when it terminates
- Results may be large — the main session receives the full output

---

## 7. Custom Agent Definitions

Beyond the three built-in types, you can define **custom agents** tailored to your project's specific needs using the `/agents` command or by creating YAML configuration files.

### Using the /agents Command

The `/agents` command in Claude Code allows you to manage custom agent definitions:

```bash
# In Claude Code, type:
/agents
```

This shows existing custom agents and allows you to create new ones.

### Agent YAML Configuration

Custom agents are defined in YAML files stored in your project's `.claude/agents/` directory:

```yaml
# .claude/agents/code-reviewer.yaml
name: "Code Reviewer"
description: "Reviews code changes for quality, security, and performance issues"

system_prompt: |
  You are a senior code reviewer. Analyze the provided code changes
  with a focus on:
  1. Security vulnerabilities (injection, auth bypass, data exposure)
  2. Performance issues (N+1 queries, unnecessary allocations, blocking I/O)
  3. Code quality (readability, DRY violations, naming conventions)
  4. Test coverage (are edge cases tested? are mocks appropriate?)

  For each issue found, provide:
  - Severity: Critical / High / Medium / Low
  - File and line number
  - Description of the issue
  - Suggested fix with code example

  Be constructive and specific. Praise good patterns when you see them.

allowed_tools:
  - Glob
  - Grep
  - Read
  - Bash    # For running linters/tests
```

### YAML Configuration Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Display name for the agent |
| `description` | string | No | Brief description of purpose |
| `system_prompt` | string | Yes | Instructions for the agent's behavior |
| `allowed_tools` | list | No | Restrict to specific tools (default: all) |

### Example: Documentation Agent

```yaml
# .claude/agents/doc-writer.yaml
name: "Documentation Writer"
description: "Generates and updates project documentation"

system_prompt: |
  You are a technical documentation specialist. Your job is to
  create clear, accurate documentation for the codebase.

  Guidelines:
  - Use the project's existing documentation style
  - Include code examples from the actual codebase (not invented examples)
  - Write for the target audience: intermediate developers
  - Always include a Table of Contents for documents > 100 lines
  - Use semantic versioning references where applicable
  - Cross-reference related documentation

  Output format: Markdown files following the project's conventions.

allowed_tools:
  - Glob
  - Grep
  - Read
  - Edit
  - Write
```

### Example: Security Auditor Agent

```yaml
# .claude/agents/security-auditor.yaml
name: "Security Auditor"
description: "Scans codebase for security vulnerabilities"

system_prompt: |
  You are a security auditor specializing in web application security.
  Systematically scan the codebase for vulnerabilities including:

  - SQL injection and NoSQL injection
  - Cross-site scripting (XSS)
  - Authentication and authorization flaws
  - Sensitive data exposure (hardcoded secrets, API keys)
  - Insecure deserialization
  - Missing security headers
  - Dependency vulnerabilities

  For each finding, report:
  - CWE ID (if applicable)
  - CVSS score estimate
  - File path and line number
  - Proof of concept (if safe to demonstrate)
  - Remediation recommendation with code

  Prioritize findings by severity. Do NOT modify any files.

allowed_tools:
  - Glob
  - Grep
  - Read
  - Bash    # For running security scanning tools
```

### Invoking Custom Agents

Custom agents are invoked as subagents using the Task tool, referenced by their name:

```python
# The main session launches a custom agent
Task(
    subagent_type="code-reviewer",  # Matches the YAML filename
    prompt="""Review the changes in the last commit.
    Focus especially on the new payment processing module
    in src/payments/."""
)
```

---

## 8. When to Use Subagents vs. Direct Work

Not every task benefits from delegation. Here is a decision framework:

### Use Subagents When

| Scenario | Subagent Type | Reason |
|----------|---------------|--------|
| Exploring an unfamiliar codebase area | Explore | Saves context in main session |
| Researching library choices | Plan | Web access + code analysis |
| Implementing independent modules | General-Purpose | Parallel execution |
| Running tests and fixing failures | General-Purpose | Isolated failure handling |
| Translating documentation | General-Purpose | Parallelizable, mechanical |
| Analyzing code quality across many files | Explore | Broad read access needed |

### Do NOT Use Subagents When

| Scenario | Reason |
|----------|--------|
| Quick single-file edit | Overhead of launching subagent exceeds benefit |
| Task needs main session's context | Subagent lacks conversation history |
| Tasks with many interdependencies | Sequential dependencies negate parallelism |
| Interactive, iterative work | Subagents cannot ask follow-up questions |
| Small, well-understood codebase | Main session can handle it directly |

### Decision Flowchart

```
Is the task complex (>10 files or >5 steps)?
├── No → Do it directly in main session
└── Yes ↓
    Does it require the conversation history?
    ├── Yes → Do it directly (or include context in prompt)
    └── No ↓
        Can it be split into independent parts?
        ├── Yes → Use parallel subagents
        └── No ↓
            Does it require file modification?
            ├── No → Use Explore or Plan subagent
            └── Yes → Use General-Purpose subagent
```

---

## 9. Best Practices

### 9.1 Write Self-Contained Prompts

The subagent has no memory of your conversation. Every prompt must be self-contained:

```python
# Include all necessary context
Task(
    subagent_type="general-purpose",
    prompt="""Project: Flask web application
    Structure: app.py (main), models.py (SQLAlchemy), templates/ (Jinja2)
    Database: SQLite with FTS5 for search
    Port: 5050

    Task: Add a /api/search endpoint that:
    1. Accepts GET requests with ?q=<query> parameter
    2. Uses the existing FTS5 search_index table
    3. Returns JSON: {"results": [...], "count": N, "query": "..."}
    4. Handles empty queries with 400 error
    5. Limits results to 50 per request

    Follow the existing route pattern in app.py.
    Run tests after implementation."""
)
```

### 9.2 Choose the Right Subagent Type

Use the **principle of least privilege**: give the subagent only the tools it needs.

```python
# Analysis task → use explore (no write access needed)
Task(subagent_type="explore", prompt="Count lines of code per module...")

# Research task → use plan (needs web, not file editing)
Task(subagent_type="plan", prompt="Research OAuth 2.0 PKCE flow...")

# Implementation task → use general-purpose (needs all tools)
Task(subagent_type="general-purpose", prompt="Implement OAuth login...")
```

### 9.3 Specify Expected Output Format

Tell the subagent exactly what format you want the results in:

```python
Task(
    subagent_type="explore",
    prompt="""Analyze test coverage gaps in src/api/.

    Return results in this format:

    ## Coverage Analysis

    ### Well-Tested (>80% coverage)
    - file.py: description of what's tested

    ### Partially Tested (40-80%)
    - file.py: what's missing

    ### Untested (<40%)
    - file.py: what needs testing

    ### Recommended Test Additions (priority order)
    1. Test name: description, estimated effort"""
)
```

### 9.4 Avoid Duplicating Work

Do not launch a subagent to do something you have already done in the main session:

```python
# BAD: Main session already read this file
# (reading file contents here that the main session already read)
Task(prompt="Read src/config.py and tell me the database URL...")

# GOOD: Pass the information directly
Task(prompt="The database URL is postgres://localhost:5432/mydb. "
           "Set up connection pooling with max 20 connections...")
```

### 9.5 Handle Subagent Failures Gracefully

Subagents can fail. Plan for it:

- Check the subagent's output for errors or incomplete results
- If a subagent fails, you can retry with a modified prompt
- For critical tasks, verify the subagent's work before proceeding
- Consider using a review subagent to validate another subagent's output

---

## 10. Common Patterns and Recipes

### Pattern 1: Explore-Then-Implement

```python
# Step 1: Explore the codebase area
analysis = Task(
    subagent_type="explore",
    prompt="Analyze the authentication system in src/auth/. "
           "Map all files, their purposes, and how they connect."
)

# Step 2: Use the analysis to guide implementation
Task(
    subagent_type="general-purpose",
    prompt=f"""Based on this analysis of the auth system:
    {analysis}

    Add two-factor authentication (TOTP) support.
    Follow the existing code patterns identified above."""
)
```

### Pattern 2: Parallel Analysis and Merge

```python
# Launch parallel exploration agents
frontend_analysis = Task(
    subagent_type="explore",
    prompt="Analyze frontend code quality in src/components/..."
)

backend_analysis = Task(
    subagent_type="explore",
    prompt="Analyze backend code quality in src/api/..."
)

database_analysis = Task(
    subagent_type="explore",
    prompt="Analyze database schema and queries in src/models/..."
)

# Main session synthesizes all three analyses
# and creates a unified improvement plan
```

### Pattern 3: Implementation with Verification

```python
# Step 1: Implement
Task(
    subagent_type="general-purpose",
    prompt="Implement feature X in src/features/..."
)

# Step 2: Verify with a separate subagent
Task(
    subagent_type="explore",
    prompt="Review the implementation of feature X in src/features/. "
           "Check for: correctness, edge cases, error handling, "
           "consistency with codebase style."
)
```

### Pattern 4: Fan-Out / Fan-In

```python
# Fan out: parallel work on independent modules
for module in ["users", "products", "orders", "payments"]:
    Task(
        subagent_type="general-purpose",
        prompt=f"Add input validation to all endpoints in "
               f"src/api/{module}/. Use zod schemas. "
               f"Write tests for each validator."
    )

# Fan in: main session verifies all modules work together
# (after all subagents complete)
```

---

## 11. Exercises

### Exercise 1: Codebase Exploration

You have a large Python project you have never seen before. Write prompts for three Explore subagents that would give you a comprehensive understanding of:
1. The project's architecture and module structure
2. The testing strategy and coverage
3. The dependency graph and external service integrations

### Exercise 2: Parallel Translation Pipeline

Design a workflow using parallel General-Purpose subagents to translate 10 Markdown lesson files from English to Korean. Consider:
- How many subagents to launch (balance between parallelism and rate limits)
- What context each subagent needs
- How to verify translation quality

### Exercise 3: Custom Agent Definition

Create a YAML configuration for a custom agent that:
- Acts as a "Migration Assistant" for database schema changes
- Can read the codebase and run commands (but not edit directly)
- Analyzes current schema, proposes migration SQL, and validates it
- Returns a migration plan with rollback strategy

### Exercise 4: Decision Making

For each scenario, decide whether to use a subagent or work directly. Justify your choice:
1. Renaming a variable used in 3 files
2. Refactoring a monolithic 2000-line file into 10 modules
3. Adding a single test case to an existing test file
4. Generating API documentation for 30 endpoints
5. Investigating why a specific test is failing intermittently

---

## 12. References

- [Claude Code Documentation: Subagents](https://docs.anthropic.com/en/docs/claude-code)
- [Claude Code Documentation: Task Tool](https://docs.anthropic.com/en/docs/claude-code)
- [Anthropic Blog: Multi-Agent Patterns](https://www.anthropic.com/engineering)
- [Claude Code GitHub: Agent Configuration](https://github.com/anthropics/claude-code)

---

## Next Steps

In the next lesson, [Agent Teams](./08_Agent_Teams.md), we move from individual subagents to coordinated teams of agents. You will learn how multiple specialized agents can share a task list, communicate progress, and work together on large-scale projects — taking the delegation patterns from this lesson to the next level.
