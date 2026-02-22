# Best Practices and Patterns

**Previous**: [20. Advanced Development Workflows](./20_Advanced_Workflows.md) | **Next**: [22. Troubleshooting and Debugging](./22_Troubleshooting.md)

---

Getting the most out of Claude Code requires more than just knowing the features -- it requires developing effective habits around prompting, context management, security, and team coordination. This lesson distills the most impactful practices from real-world usage into actionable patterns you can apply immediately.

**Difficulty**: ⭐⭐

**Prerequisites**:
- Working experience with Claude Code ([Lessons 2-7](./02_Claude_Code_Getting_Started.md))
- Understanding of CLAUDE.md and project configuration ([Lesson 3](./03_CLAUDE_md_and_Project_Setup.md))
- Familiarity with permission modes ([Lesson 4](./04_Permission_Modes.md))
- Experience with hooks and skills ([Lessons 5-6](./05_Hooks.md))

**Learning Objectives**:
- Write effective prompts that get accurate, useful results
- Manage context efficiently across long sessions
- Apply security best practices when working with Claude Code
- Establish team collaboration patterns with shared configuration
- Optimize performance for faster, more focused sessions
- Recognize and avoid common anti-patterns that waste time and tokens

---

## Table of Contents

1. [Effective Prompt Writing](#1-effective-prompt-writing)
2. [Context Management](#2-context-management)
3. [Security Best Practices](#3-security-best-practices)
4. [Team Collaboration Patterns](#4-team-collaboration-patterns)
5. [Performance Optimization](#5-performance-optimization)
6. [Common Anti-Patterns](#6-common-anti-patterns)
7. [Cheat Sheet: Quick Reference](#7-cheat-sheet-quick-reference)
8. [Exercises](#8-exercises)

---

## 1. Effective Prompt Writing

The quality of your prompts directly determines the quality of Claude Code's output. A well-crafted prompt can save hours of iteration; a vague prompt wastes tokens and time.

### 1.1 The SCFO Framework

Structure your prompts around four elements: **Situation, Constraints, Format, Output**.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SCFO Prompt Framework                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  S - Situation (What and Why)                                       │
│  ├── What you want to accomplish                                   │
│  ├── Why you need it (context for better decisions)                │
│  └── Relevant background information                               │
│                                                                     │
│  C - Constraints (Boundaries)                                       │
│  ├── Technical constraints (language, framework, version)          │
│  ├── Style constraints (patterns to follow, conventions)           │
│  ├── What NOT to do (equally important)                            │
│  └── Performance/size requirements                                 │
│                                                                     │
│  F - Format (How to respond)                                        │
│  ├── Expected output structure (JSON, markdown, code only)        │
│  ├── Level of explanation needed                                   │
│  └── Whether to include tests, docs, etc.                         │
│                                                                     │
│  O - Output (What to produce)                                       │
│  ├── Specific files to create or modify                            │
│  ├── Verification steps (run tests, type check)                   │
│  └── Success criteria                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Examples: Bad vs Good Prompts

```
BAD (vague, no context):
> "Fix the bug in the login"

GOOD (specific, contextual):
> "The login endpoint at src/routes/auth.py returns a 500 error when
>  the user enters an email with a '+' character (like user+tag@gmail.com).
>  The error is in the email validation regex on line 45. Fix the regex
>  to accept RFC 5321 compliant email addresses. Run the existing tests
>  to make sure nothing else breaks."
```

```
BAD (too broad):
> "Make the code better"

GOOD (specific scope):
> "Refactor the OrderService class (src/services/order.py):
>  1. Extract the tax calculation logic into a separate TaxCalculator class
>  2. Replace the nested if/else for shipping rules with a strategy pattern
>  3. Add type hints to all public methods
>  Keep all existing tests passing."
```

```
BAD (missing constraints):
> "Add a caching layer"

GOOD (clear constraints):
> "Add Redis caching to the GET /api/products endpoint:
>  - Cache key: products:{category}:{page}:{sort}
>  - TTL: 5 minutes
>  - Invalidate on POST/PUT/DELETE to /api/products
>  - Use the existing Redis connection from config.redis_client
>  - Follow the caching pattern used in src/services/user_service.py
>  - Add tests using fakeredis"
```

### 1.3 Breaking Large Tasks into Steps

For complex tasks, break them into explicit steps:

```
> Implement a WebSocket notification system. Let's do this in phases:
>
> Phase 1: Set up the WebSocket server
> - Add ws dependency to package.json
> - Create src/websocket/server.ts with connection handling
> - Integrate with the existing Express server in src/app.ts
> - Test: client can connect and receive a welcome message
>
> Phase 2: Authentication
> - Verify JWT token on WebSocket upgrade
> - Associate connections with user IDs
> - Test: unauthenticated connections are rejected
>
> Phase 3: Notification dispatch
> - Create a NotificationDispatcher that sends events to connected users
> - Integrate with the existing NotificationService
> - Test: creating an order sends a WebSocket event to the buyer
>
> Start with Phase 1. Don't proceed to Phase 2 until Phase 1 tests pass.
```

### 1.4 Using Plan Mode Effectively

Plan mode is your most powerful tool for exploration and strategy. Use it before committing to implementation:

```
> /plan I need to migrate our authentication from session-based to JWT.
>       What files will be affected? What's the safest migration order?
>       Are there any edge cases I should worry about?
```

Plan mode will:
- Search the codebase for all authentication-related code
- Identify dependencies and the order of changes
- Flag potential issues (like active sessions during migration)
- Suggest a step-by-step implementation plan

**When to use Plan mode:**
- Before any refactoring that touches more than 3 files
- When exploring an unfamiliar part of the codebase
- When you are unsure about the best approach
- Before making irreversible changes (database migrations, API breaking changes)

**When to skip Plan mode:**
- Simple, well-understood changes (fix a typo, add a field)
- Tasks where you already know exactly what needs to change
- Follow-up changes in an ongoing session where context is established

---

## 2. Context Management

Claude Code's effectiveness depends heavily on having the right context at the right time. Managing context well means Claude spends less time searching and more time producing useful output.

### 2.1 CLAUDE.md Optimization

Your CLAUDE.md file is the single most important context document. Keep it focused and well-structured:

```markdown
# CLAUDE.md — Best Practices Template

## Project Overview
One paragraph: what the project does, what stack it uses.

## Architecture
Brief description of the main modules and how they connect.
Use an ASCII diagram if the structure is not obvious.

## Key Commands
```bash
# Development
npm run dev          # Start dev server (port 3000)
npm run test         # Run all tests
npm run test:watch   # Watch mode

# Database
npm run db:migrate   # Run migrations
npm run db:seed      # Seed test data
```

## Conventions
- Use functional components with hooks (no class components)
- API routes follow RESTful naming: /api/v1/{resource}
- Tests co-located with source: Component.tsx → Component.test.tsx
- Error handling: use AppError class from src/utils/errors.ts

## Important Files
- src/config/index.ts — all environment configuration
- src/middleware/auth.ts — JWT verification middleware
- src/types/index.ts — shared TypeScript types

## Do NOT
- Modify files in src/generated/ (auto-generated from OpenAPI)
- Use console.log (use the logger from src/utils/logger.ts)
- Add dependencies without checking bundle size impact
```

**What to include:**
- Commands that Claude will need to run (test, build, lint)
- Conventions that are not obvious from the code
- Important files that Claude should know about
- Things to avoid (common mistakes, generated files)

**What NOT to include:**
- Entire API documentation (too long, use file references instead)
- Detailed commit history or changelog
- Information that changes frequently (live URLs, current sprint goals)
- Anything available in standard config files (package.json, tsconfig.json)

### 2.2 File References

Help Claude find the right files instead of searching the entire codebase:

```
> The payment processing logic is in:
> - src/services/payment.ts (main service)
> - src/providers/stripe.ts (Stripe integration)
> - src/providers/paypal.ts (PayPal integration)
> Add Apple Pay support following the same provider pattern.
```

This is much faster than asking Claude to "find the payment code" -- it eliminates the search step entirely.

### 2.3 Session Length Management

Claude Code sessions have a context window limit. As conversations grow longer, earlier context may be compressed or lost. Manage this proactively:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Session Management Strategy                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Short Sessions (< 30 messages)                                     │
│  ├── Ideal for focused tasks                                       │
│  ├── Full context retained throughout                              │
│  └── No special management needed                                  │
│                                                                     │
│  Medium Sessions (30-80 messages)                                   │
│  ├── Use /compact periodically to summarize context                │
│  ├── Restate key context when switching topics                     │
│  └── Claude may need reminders about earlier decisions             │
│                                                                     │
│  Long Sessions (80+ messages)                                       │
│  ├── Consider starting a fresh session                             │
│  ├── Save important decisions to CLAUDE.md                         │
│  ├── Use subagents for independent research                        │
│  └── /compact is essential                                         │
│                                                                     │
│  Signs you should start a new session:                              │
│  ├── Claude forgets decisions made earlier                          │
│  ├── Responses become less accurate or repetitive                  │
│  ├── Claude re-reads files it already read                         │
│  └── You are switching to a completely different task              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.4 The /compact Command

Use `/compact` to compress the conversation history while preserving key decisions:

```
> /compact

# Claude summarizes the session so far:
# "Session summary: We've been implementing the notification preferences
#  feature. Completed: database migration, API endpoints (GET/PUT/PATCH),
#  service integration with caching. Remaining: frontend components,
#  tests for edge cases."
```

Use `/compact` when:
- You are about halfway through the context window
- Switching focus within the same session
- Claude starts showing signs of forgetting earlier context
- You want to "checkpoint" progress before a complex next step

---

## 3. Security Best Practices

### 3.1 Permission Mode Selection

Choose the appropriate permission mode for your trust level:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Permission Mode Decision Tree                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Is this a trusted repository you own or maintain?                  │
│  ├── YES: Is the task well-defined and low-risk?                   │
│  │   ├── YES → Auto-accept mode (with allow rules)                │
│  │   └── NO  → Default mode (approve each tool use)               │
│  │                                                                  │
│  └── NO: Do you understand what Claude will be doing?              │
│      ├── YES → Default mode (review before approving)             │
│      └── NO  → Plan-only mode (no code execution)                 │
│                                                                     │
│  Is this running in CI/CD (no human present)?                       │
│  ├── YES: Is the environment isolated (container)?                 │
│  │   ├── YES → Bypass mode (auto-accept all)                     │
│  │   └── NO  → NEVER use bypass on shared infrastructure          │
│  └── NO  → Use default or auto-accept as appropriate              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Secrets Management

Never put secrets in places Claude can read or expose them:

```
# BAD: Secrets in CLAUDE.md
## API Keys
- Stripe: sk_live_abc123...
- Database: postgresql://admin:password@db.example.com

# BAD: Secrets in prompts
> "Connect to the database at postgresql://admin:password@db.example.com
>  and run a migration"

# GOOD: Use environment variables
> "Connect to the database using the DATABASE_URL environment variable
>  and run the migration"

# GOOD: Reference secrets by name, not value
> "The Stripe API key is stored in the STRIPE_SECRET_KEY env var.
>  Use it to initialize the Stripe client."
```

In your CLAUDE.md:

```markdown
## Environment Variables
Required env vars (see .env.example for structure):
- DATABASE_URL — PostgreSQL connection string
- REDIS_URL — Redis connection string
- STRIPE_SECRET_KEY — Stripe API key
- JWT_SECRET — JWT signing key

DO NOT hardcode any of these values. Always read from os.environ.
```

### 3.3 Reviewing Generated Code

Always review code that Claude generates, especially for:

```
┌─────────────────────────────────────────────────────────────────────┐
│                Security Review Checklist for Generated Code          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  □ SQL queries use parameterized queries (no string concatenation) │
│  □ User input is validated and sanitized                           │
│  □ Authentication checks are present on protected endpoints        │
│  □ Authorization verifies the user can access the resource         │
│  □ Error messages don't leak internal details                      │
│  □ Sensitive data is not logged                                    │
│  □ File operations validate paths (no path traversal)              │
│  □ External URLs are validated before use                          │
│  □ Cryptographic operations use standard libraries                 │
│  □ Dependencies are from trusted sources                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.4 Safe Patterns for Sensitive Operations

```python
# GOOD: Tell Claude to use environment variables
# In your prompt:
# "Read the API key from the environment, never hardcode it"

import os

def get_api_client():
    """Create an API client using environment configuration."""
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise EnvironmentError(
            "API_KEY environment variable is required. "
            "See .env.example for setup instructions."
        )
    return APIClient(api_key=api_key)


# GOOD: Parameterized queries (Claude should always generate these)
def get_user(db, user_id: int):
    """Fetch a user by ID using parameterized query."""
    return db.execute(
        "SELECT * FROM users WHERE id = %s",  # Parameterized
        (user_id,)
    ).fetchone()


# BAD: String interpolation in SQL (flag this in reviews)
def get_user_UNSAFE(db, user_id):
    """NEVER DO THIS — SQL injection vulnerability."""
    return db.execute(
        f"SELECT * FROM users WHERE id = {user_id}"  # VULNERABLE
    ).fetchone()
```

---

## 4. Team Collaboration Patterns

### 4.1 Shared CLAUDE.md as Team Documentation

CLAUDE.md serves dual purpose: it configures Claude Code AND documents team conventions. Use this to your advantage:

```markdown
# CLAUDE.md

## Team Conventions (Claude Code + human developers)

### Git Workflow
- Feature branches: feature/JIRA-123-short-description
- Commit messages: conventional commits (feat:, fix:, refactor:, etc.)
- All PRs require 1 review before merge
- Squash merge to main

### Code Review Standards
When reviewing code (or generating code for review):
- Functions > 30 lines should be broken up
- Public functions must have docstrings
- New endpoints must have integration tests
- Database queries must use the ORM (no raw SQL except migrations)

### Testing Requirements
- Unit tests for business logic (pytest)
- Integration tests for API endpoints (pytest + httpx)
- E2E tests for critical user flows (Playwright)
- Minimum 80% coverage on new code
```

Because this file is checked into version control, the entire team benefits from the same Claude Code configuration.

### 4.2 Standardized Skills for Common Workflows

Create shared skills that encode team-specific workflows:

```markdown
# .claude/skills/create-api-endpoint.md

## Skill: Create API Endpoint

When asked to create a new API endpoint, follow this checklist:

1. Create the route handler in `src/routes/{resource}.py`
2. Add Pydantic request/response schemas in `src/schemas/{resource}.py`
3. Implement business logic in `src/services/{resource}_service.py`
4. Add the route to `src/routes/__init__.py` router registration
5. Write integration tests in `tests/api/test_{resource}.py`
6. Update `docs/API.md` with the new endpoint
7. Run `pytest tests/` to verify all tests pass
8. Run `mypy src/` to verify type safety

### Template for route handler:

```python
from fastapi import APIRouter, Depends, HTTPException, status
from src.auth import get_current_user
from src.schemas.{resource} import {Resource}Create, {Resource}Response
from src.services.{resource}_service import {Resource}Service

router = APIRouter(prefix="/api/v1/{resources}", tags=["{resources}"])

@router.post("/", response_model={Resource}Response, status_code=status.HTTP_201_CREATED)
async def create_{resource}(
    data: {Resource}Create,
    user = Depends(get_current_user),
    service: {Resource}Service = Depends(),
):
    return await service.create(data, user)
```
```

### 4.3 Code Review with Claude Assistance

Establish a pattern where Claude Code assists with (but does not replace) human code review:

```
# Review workflow for team members:

1. Author submits PR
2. CI runs Claude Code review (automated, catches obvious issues)
3. Human reviewer focuses on:
   - Business logic correctness
   - Design decisions and trade-offs
   - Edge cases specific to domain knowledge
   - Performance implications
4. Author addresses both Claude and human feedback
```

### 4.4 Onboarding New Team Members

Create an onboarding-friendly CLAUDE.md section:

```markdown
## New Developer Onboarding

If you're new to this project, ask Claude Code these questions:

1. "Explain the high-level architecture of this project"
2. "Walk me through how a request flows from the API to the database"
3. "What are the main modules and what does each one do?"
4. "Show me the testing patterns used in this project"
5. "What are the most common tasks I'll need to do?"

### Quick Start
```bash
# Setup
cp .env.example .env  # Edit with your local settings
docker-compose up -d   # Start PostgreSQL and Redis
npm install
npm run db:migrate
npm run db:seed

# Development
npm run dev            # http://localhost:3000
npm run test:watch     # Run tests in watch mode
```
```

---

## 5. Performance Optimization

### 5.1 Start Sessions with Clear Context

The first message sets the tone for the entire session. Front-load the important context:

```
# GOOD: Clear opening message
> I'm working on the payment service (src/services/payment/).
> The issue is that refund processing fails silently when the Stripe
> API returns a 402 error. I need to:
> 1. Add proper error handling for 402 responses
> 2. Retry with exponential backoff for transient errors
> 3. Log failures to our error tracking system (Sentry)
> The relevant files are payment_service.py and stripe_provider.py.

# BAD: Vague opening that requires many follow-up questions
> Something is wrong with payments. Can you help?
```

### 5.2 Use Subagents for Independent Research

When you need information from multiple unrelated parts of the codebase, subagents can search in parallel:

```
> I need to understand three things before implementing the new feature:
> 1. How the current notification system works (src/services/notifications/)
> 2. What WebSocket libraries we already have installed
> 3. How our test fixtures are structured (tests/conftest.py)
> These are independent — investigate all three.
```

Claude Code may spawn subagents to research each topic simultaneously, returning results faster than sequential investigation.

### 5.3 Batch Related Changes

Instead of making changes one file at a time, batch related changes:

```
# SLOW: One change at a time
> "Add the new field to the model"
> (wait for completion)
> "Now update the schema"
> (wait for completion)
> "Now update the API endpoint"
> (wait for completion)

# FAST: Batch related changes
> "Add a 'phone_number' field to the User model, update the
>  Pydantic schema, and update the registration endpoint to
>  accept and validate phone numbers. Update all three files
>  together, then run the tests."
```

### 5.4 Avoid Unnecessary File Re-Reads

Claude Code caches file contents within a session. If you modify a file outside of Claude Code (in your editor), let it know:

```
> I just edited src/config.py in my editor to add a new setting.
> Read the updated file before continuing.
```

Without this hint, Claude Code might work with its cached (stale) version of the file.

---

## 6. Common Anti-Patterns

### 6.1 Over-Broad Prompts

```
# ANTI-PATTERN: Asking Claude to review the entire codebase
> "Review all the code in this project and suggest improvements"
# This wastes tokens, produces superficial results, and often
# exceeds the context window.

# BETTER: Focus on specific areas
> "Review the error handling in src/services/ and identify
>  places where exceptions are silently swallowed"
```

### 6.2 Ignoring Plan Mode for Complex Tasks

```
# ANTI-PATTERN: Jumping straight into implementation
> "Rewrite the database layer to use MongoDB instead of PostgreSQL"
# This will produce incomplete results because Claude doesn't
# understand the full scope of changes needed.

# BETTER: Plan first, then implement
> /plan "What would it take to migrate from PostgreSQL to MongoDB?
>        List all affected files, data migration needs, and risks."
# Then use the plan to execute phase by phase.
```

### 6.3 Not Reviewing Generated Code

```
# ANTI-PATTERN: Auto-accepting everything
# Setting permission mode to bypass for local development
# and never reviewing what Claude generates.

# BETTER: Review security-sensitive code
# Use default permission mode. When Claude generates auth code,
# database queries, or file operations, carefully review them.
```

### 6.4 Using Hooks for Things That Belong in CLAUDE.md

```json
// ANTI-PATTERN: Complex hook that enforces style
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Edit|Write",
      "command": "echo 'Remember to use PEP 8 style and add type hints'"
    }]
  }
}

// BETTER: Put style guidelines in CLAUDE.md
// CLAUDE.md:
// ## Style: PEP 8, type hints on all functions, Google-style docstrings
```

Hooks should be for automated checks (running linters, tests), not for instructions that belong in CLAUDE.md.

### 6.5 Running Too Many Parallel Agents

```
# ANTI-PATTERN: Spawning 20 subagents simultaneously
# This hits rate limits and slows everything down.

# BETTER: Batch into groups of 3-5 parallel agents
# If you have 20 files to process, do 4 batches of 5
# rather than all 20 at once.
```

### 6.6 Treating Claude as an Oracle Instead of a Partner

```
# ANTI-PATTERN: Asking for a definitive answer without context
> "What's the best database for my project?"

# BETTER: Provide context and constraints for a reasoned recommendation
> "I'm building a real-time analytics dashboard that ingests 10K events/sec,
>  requires sub-second query latency, and my team knows PostgreSQL.
>  Should I use PostgreSQL with TimescaleDB, ClickHouse, or something else?
>  What are the trade-offs?"
```

---

## 7. Cheat Sheet: Quick Reference

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Claude Code Best Practices Cheat Sheet               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PROMPTING                                                          │
│  ✓ Be specific: what, where, why, constraints                     │
│  ✓ Break large tasks into phases                                   │
│  ✓ Use /plan before complex changes                                │
│  ✓ Reference specific files when you know them                     │
│  ✗ Vague prompts like "make it better"                            │
│  ✗ Asking to review "everything"                                   │
│                                                                     │
│  CONTEXT                                                            │
│  ✓ Keep CLAUDE.md focused and current                              │
│  ✓ Use /compact in long sessions                                   │
│  ✓ Start new sessions for unrelated tasks                          │
│  ✓ Front-load context in first message                             │
│  ✗ Stuffing CLAUDE.md with everything                             │
│  ✗ 200+ message sessions without compacting                       │
│                                                                     │
│  SECURITY                                                           │
│  ✓ Use environment variables for secrets                           │
│  ✓ Review generated auth/security code carefully                   │
│  ✓ Match permission mode to trust level                            │
│  ✓ Check for SQL injection, path traversal                         │
│  ✗ Hardcoding secrets in CLAUDE.md or prompts                     │
│  ✗ Using bypass mode on untrusted repos                           │
│                                                                     │
│  PERFORMANCE                                                        │
│  ✓ Batch related changes in one prompt                             │
│  ✓ Use subagents for independent research                          │
│  ✓ Choose the right model for the task complexity                  │
│  ✓ Use prompt caching for repeated context                         │
│  ✗ One tiny change per message                                    │
│  ✗ Using Opus for simple tasks                                    │
│                                                                     │
│  TEAM                                                               │
│  ✓ Share CLAUDE.md across the team (checked into git)              │
│  ✓ Standardize skills for common workflows                        │
│  ✓ Use Claude for code review assistance, not replacement          │
│  ✓ Include onboarding guidance in CLAUDE.md                        │
│  ✗ Each developer with incompatible configurations                │
│  ✗ Relying solely on Claude for code review                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Exercises

### Exercise 1: CLAUDE.md Audit (Beginner)

Review the CLAUDE.md file (or create one) for a project you work on. Evaluate it against these criteria:
1. Does it include the essential commands Claude needs (test, build, lint)?
2. Is it under 200 lines? (If longer, identify what to trim.)
3. Does it document non-obvious conventions?
4. Does it list important files that Claude should know about?
5. Does it include any sensitive information that should be removed?

Rewrite it following the template in Section 2.1.

### Exercise 2: Prompt Improvement (Intermediate)

Take these vague prompts and rewrite them using the SCFO framework:

1. "Add error handling to the app"
2. "The tests are slow, fix them"
3. "Create a user management system"
4. "Make the API more secure"
5. "Refactor the database code"

For each rewritten prompt, explain what additional context you added and why.

### Exercise 3: Security Review (Intermediate)

Ask Claude Code to generate a simple REST API with authentication. Then review the generated code against the security checklist in Section 3.3. Document:
1. How many checklist items Claude addressed correctly by default
2. Any security issues you found
3. What additional prompting was needed to fix the issues

### Exercise 4: Team Configuration (Advanced)

Design a complete Claude Code configuration for a team of 5 developers working on a Python/FastAPI project. Create:
1. A CLAUDE.md file
2. Two custom skills (one for creating endpoints, one for adding database models)
3. Permission settings appropriate for the team
4. A hook that runs linting on every file edit
5. Onboarding instructions for new team members

### Exercise 5: Session Optimization (Intermediate)

Conduct two identical coding tasks (e.g., implementing a feature) using different approaches:

**Session A**: Start with a vague prompt, add details incrementally
**Session B**: Start with a complete SCFO prompt, use Plan mode, batch changes

Compare:
- Total number of messages needed
- Time to completion
- Quality of the final result
- How many times you had to correct or redirect Claude

Document your findings and identify which practices made the biggest difference.

---

## References

- Claude Code Documentation - https://docs.anthropic.com/en/docs/claude-code
- Anthropic Prompt Engineering Guide - https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering
- OWASP Secure Coding Practices - https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/

---

## Next Steps

[22. Troubleshooting and Debugging](./22_Troubleshooting.md) covers common issues you may encounter with Claude Code -- permission errors, hook failures, context limits, MCP problems, API errors, and performance issues -- along with systematic approaches to diagnose and resolve them.
