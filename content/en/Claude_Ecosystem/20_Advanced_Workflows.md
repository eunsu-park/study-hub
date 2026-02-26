# Advanced Development Workflows

**Previous**: [19. Models, Pricing, and Optimization](./19_Models_and_Pricing.md) | **Next**: [21. Best Practices and Patterns](./21_Best_Practices.md)

---

Claude Code is not just an AI assistant that answers questions -- it is a development partner capable of executing complex, multi-step workflows. This lesson covers advanced patterns for refactoring, test-driven development, CI/CD integration, large codebase exploration, and end-to-end feature implementation.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Solid experience with Claude Code basics ([Lessons 2-6](./02_Claude_Code_Getting_Started.md))
- Understanding of subagents and task delegation ([Lesson 7](./07_Subagents.md))
- Familiarity with Git workflows and CI/CD concepts
- Experience with testing frameworks (pytest, Jest, etc.)

## Learning Objectives

After completing this lesson, you will be able to:

1. Execute multi-file refactoring safely with Claude Code
2. Practice test-driven development using Claude as a coding partner
3. Integrate Claude Code into CI/CD pipelines for automated code review
4. Explore and understand large, unfamiliar codebases systematically
5. Build database schemas and API endpoints with Claude assistance
6. Maintain documentation in sync with code changes
7. Apply these workflows to a complete feature implementation

---

## Table of Contents

1. [Multi-File Refactoring Patterns](#1-multi-file-refactoring-patterns)
2. [Test-Driven Development with Claude](#2-test-driven-development-with-claude)
3. [CI/CD Pipeline Integration](#3-cicd-pipeline-integration)
4. [Large Codebase Exploration](#4-large-codebase-exploration)
5. [Database and API Development](#5-database-and-api-development)
6. [Documentation Workflows](#6-documentation-workflows)
7. [Case Study: Feature Implementation from Spec to Deploy](#7-case-study-feature-implementation-from-spec-to-deploy)
8. [Exercises](#8-exercises)

---

## 1. Multi-File Refactoring Patterns

Refactoring across multiple files is one of Claude Code's strongest capabilities. The key is a systematic approach: plan first, execute in order, verify after each step.

### 1.1 The Planning Phase

Before making any changes, ask Claude to analyze the scope of the refactoring:

```
> I want to rename the UserService class to AccountService across the entire
> codebase. Before making changes, analyze the impact:
> - Which files import or reference UserService?
> - Are there database migrations that reference this name?
> - Are there API endpoints that expose this name?
> - What tests will need to be updated?
```

Claude Code will use its search tools (Grep, Glob) to build a complete picture before touching any files. This planning step prevents partial refactors that break the build.

### 1.2 Systematic Execution

```
┌─────────────────────────────────────────────────────────────────────┐
│                Multi-File Refactoring Strategy                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: PLAN                                                       │
│  ├── Identify all affected files (grep for references)              │
│  ├── Determine dependency order                                     │
│  ├── Check for dynamic references (string-based, configs)           │
│  └── Estimate scope (number of files, risk level)                   │
│                                                                     │
│  Step 2: PREPARE                                                    │
│  ├── Ensure all tests pass before starting                          │
│  ├── Create a branch for the refactoring                           │
│  └── Commit the current state (clean baseline)                     │
│                                                                     │
│  Step 3: EXECUTE (in dependency order)                              │
│  ├── Core definitions first (class/interface/type definitions)     │
│  ├── Internal consumers next (services, utilities)                  │
│  ├── External interfaces last (API routes, CLI commands)            │
│  └── Tests alongside their source files                            │
│                                                                     │
│  Step 4: VERIFY                                                     │
│  ├── Run tests after each batch of changes                          │
│  ├── Run linter/type checker                                        │
│  ├── Manual smoke test of affected features                        │
│  └── Review diff for unintended changes                            │
│                                                                     │
│  Step 5: COMMIT                                                     │
│  ├── One logical commit per refactoring step                       │
│  └── Clear commit messages explaining the "why"                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Using Subagents for Parallel Changes

For large refactors that span independent modules, Claude Code can use subagents to process multiple files simultaneously:

```
> Refactor the logging system to use structured logging (structlog).
> The affected modules are independent: auth/, payments/, notifications/.
> Use subagents to update each module in parallel, then verify.
```

Claude Code will spawn subagents for each module, each making the necessary changes independently. After all subagents complete, it runs the full test suite to verify integration.

### 1.4 Practical Example: Extracting a Module

A common refactoring pattern is extracting functionality into a separate module:

```
> The user authentication logic is scattered across app.py, routes/auth.py,
> and middleware/session.py. Extract it into a clean auth/ package with:
>
> auth/
> ├── __init__.py       # Public API
> ├── service.py        # Core authentication logic
> ├── middleware.py      # Session middleware
> ├── schemas.py        # Pydantic models
> └── exceptions.py     # Custom exceptions
>
> Requirements:
> 1. No behavior changes (pure refactoring)
> 2. All existing tests must continue to pass
> 3. Update all imports across the codebase
> 4. Add deprecation warnings for old import paths
```

Claude Code handles this by:
1. Reading all affected files to understand the current structure
2. Creating the new package with properly organized code
3. Updating imports throughout the codebase
4. Running tests to verify nothing broke
5. Optionally adding backwards-compatible import aliases

---

## 2. Test-Driven Development with Claude

### 2.1 The Red-Green-Refactor Cycle

TDD with Claude Code follows the classic red-green-refactor cycle, but Claude accelerates each phase:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TDD Cycle with Claude Code                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                    ┌──────────────┐                                 │
│                    │              │                                 │
│             ┌──────│   RED        │◀─────────┐                     │
│             │      │ Write a      │          │                     │
│             │      │ failing test │          │                     │
│             │      └──────────────┘          │                     │
│             │             │                  │                     │
│             │      "Write a test for         │                     │
│             │       a function that..."      │                     │
│             ▼             │                  │                     │
│      ┌──────────────┐    │           ┌──────────────┐             │
│      │              │    ▼           │              │             │
│      │   GREEN      │◀───────       │   REFACTOR   │             │
│      │ Write minimal│                │ Improve code │             │
│      │ code to pass │───────────────▶│ quality      │             │
│      └──────────────┘                └──────────────┘             │
│             │                               │                     │
│      "Implement just                 "Refactor this to            │
│       enough to pass                  be cleaner while            │
│       the test"                       keeping tests green"        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 TDD Session Example

Here is a complete TDD session for building a rate limiter:

**Step 1: RED -- Write the test first**

```
> Write pytest tests for a TokenBucketRateLimiter class with these behaviors:
> - Initialize with max_tokens and refill_rate (tokens per second)
> - allow_request() returns True if tokens available, False otherwise
> - Tokens refill over time based on refill_rate
> - Thread-safe for concurrent access
> Don't implement the class yet -- just the tests.
```

Claude generates the test file:

```python
# tests/test_rate_limiter.py
import time
import threading
import pytest
from rate_limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    """Tests for the token bucket rate limiter."""

    def test_allows_requests_within_limit(self):
        """Requests within the token limit should be allowed."""
        limiter = TokenBucketRateLimiter(max_tokens=5, refill_rate=1.0)
        for _ in range(5):
            assert limiter.allow_request() is True

    def test_rejects_requests_over_limit(self):
        """Requests exceeding the token limit should be rejected."""
        limiter = TokenBucketRateLimiter(max_tokens=3, refill_rate=1.0)
        for _ in range(3):
            limiter.allow_request()
        assert limiter.allow_request() is False

    def test_tokens_refill_over_time(self):
        """Tokens should refill based on the refill rate."""
        limiter = TokenBucketRateLimiter(max_tokens=2, refill_rate=10.0)
        # Exhaust all tokens
        limiter.allow_request()
        limiter.allow_request()
        assert limiter.allow_request() is False
        # Wait for refill (10 tokens/sec = 1 token per 0.1 sec)
        time.sleep(0.15)
        assert limiter.allow_request() is True

    def test_tokens_do_not_exceed_max(self):
        """Refilled tokens should not exceed max_tokens."""
        limiter = TokenBucketRateLimiter(max_tokens=3, refill_rate=100.0)
        time.sleep(0.1)  # Would refill 10 tokens, but max is 3
        allowed = sum(1 for _ in range(10) if limiter.allow_request())
        assert allowed == 3

    def test_thread_safety(self):
        """Rate limiter should be safe for concurrent access."""
        limiter = TokenBucketRateLimiter(max_tokens=100, refill_rate=0.0)
        results = []

        def make_requests():
            for _ in range(20):
                results.append(limiter.allow_request())

        threads = [threading.Thread(target=make_requests) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 100 should be allowed (max_tokens=100, no refill)
        assert sum(results) == 100
        assert len(results) == 200

    def test_zero_refill_rate(self):
        """With zero refill rate, tokens never replenish."""
        limiter = TokenBucketRateLimiter(max_tokens=1, refill_rate=0.0)
        assert limiter.allow_request() is True
        time.sleep(0.1)
        assert limiter.allow_request() is False

    def test_initial_state(self):
        """Rate limiter starts with max_tokens available."""
        limiter = TokenBucketRateLimiter(max_tokens=10, refill_rate=1.0)
        assert limiter.available_tokens == 10
```

**Step 2: GREEN -- Implement minimally**

```
> Now implement TokenBucketRateLimiter in rate_limiter.py.
> Write just enough code to make all tests pass.
```

**Step 3: REFACTOR -- Improve quality**

```
> All tests pass. Now refactor the implementation:
> - Add proper docstrings
> - Improve variable naming
> - Add type hints
> - Keep all tests green
```

### 2.3 Using Plan Mode for Test Coverage Analysis

Before writing implementation code, use Plan mode to analyze what tests are needed:

```
> /plan Review the UserService class and identify all untested code paths.
> List the test cases needed for 100% branch coverage.
```

Plan mode analyzes the code without making changes, producing a prioritized list of test cases. You can then switch to implementation mode:

```
> Now write those 12 test cases you identified, starting with the critical paths.
```

---

## 3. CI/CD Pipeline Integration

### 3.1 Running Claude Code in CI

Claude Code can run non-interactively in CI environments using the `--print` flag (which sends a single prompt and exits) or through the API directly.

```yaml
# .github/workflows/claude-review.yml
name: Claude Code Review

on:
  pull_request:
    types: [opened, synchronize]

permissions:
  contents: read
  pull-requests: write

jobs:
  code-review:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for diff

      - name: Install Claude Code
        run: npm install -g @anthropic-ai/claude-code

      - name: Run Claude Code Review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          # Get the diff for this PR
          DIFF=$(git diff origin/${{ github.base_ref }}...HEAD)

          # Run Claude Code in non-interactive mode
          claude --print "Review this pull request diff for:
          1. Potential bugs or logic errors
          2. Security vulnerabilities
          3. Performance concerns
          4. Style/convention violations
          5. Missing tests

          Format your response as a markdown checklist.

          Diff:
          $DIFF" > review.md

      - name: Post review comment
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const review = fs.readFileSync('review.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Claude Code Review\n\n${review}`
            });
```

### 3.2 Auto-Fix CI Failures

When CI fails, Claude Code can automatically diagnose and fix the issue:

```yaml
# .github/workflows/claude-autofix.yml
name: Claude Auto-Fix

on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]

jobs:
  auto-fix:
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          ref: ${{ github.event.workflow_run.head_branch }}

      - name: Get CI logs
        id: logs
        uses: actions/github-script@v7
        with:
          script: |
            const jobs = await github.rest.actions.listJobsForWorkflowRun({
              owner: context.repo.owner,
              repo: context.repo.repo,
              run_id: ${{ github.event.workflow_run.id }},
            });
            const failedJob = jobs.data.jobs.find(j => j.conclusion === 'failure');
            const logs = await github.rest.actions.downloadJobLogsForWorkflowRun({
              owner: context.repo.owner,
              repo: context.repo.repo,
              job_id: failedJob.id,
            });
            return logs.data.substring(logs.data.length - 5000);  // Last 5000 chars

      - name: Claude Code auto-fix
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          claude --print "The CI pipeline failed. Here are the logs:

          ${{ steps.logs.outputs.result }}

          Diagnose the failure and fix the code. Run the failing tests
          to verify the fix works."

      - name: Create fix PR
        uses: peter-evans/create-pull-request@v6
        with:
          commit-message: "fix: auto-fix CI failure"
          title: "fix: Auto-fix CI failure from Claude Code"
          body: "Automated fix for CI failure. Please review before merging."
          branch: autofix/${{ github.event.workflow_run.head_branch }}
```

### 3.3 Bypass Mode for Containers

In CI environments where interactive prompts are not possible, Claude Code supports Bypass mode that auto-accepts all tool executions:

```bash
# In a Docker container or CI runner
export CLAUDE_CODE_PERMISSION_MODE=bypass

# Claude Code will execute all tools without prompting for approval
# ONLY use this in trusted, isolated environments (CI containers)
claude --print "Run the test suite and fix any failing tests"
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CI/CD Integration Patterns                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Pattern 1: PR Review Bot                                           │
│  ├── Trigger: Pull request opened/updated                          │
│  ├── Action: Review diff, post comments                            │
│  └── Model: Sonnet (good balance of quality and cost)              │
│                                                                     │
│  Pattern 2: Auto-Fix Bot                                            │
│  ├── Trigger: CI failure on feature branch                         │
│  ├── Action: Diagnose failure, apply fix, create PR                │
│  └── Model: Sonnet or Opus (depending on complexity)               │
│                                                                     │
│  Pattern 3: Pre-commit Checks                                       │
│  ├── Trigger: Before commit (via hooks)                            │
│  ├── Action: Lint, format, check for secrets                       │
│  └── Model: Haiku (fast, simple checks)                            │
│                                                                     │
│  Pattern 4: Release Notes Generator                                 │
│  ├── Trigger: Tag push or release creation                         │
│  ├── Action: Summarize changes since last release                  │
│  └── Model: Sonnet (good at summarization)                         │
│                                                                     │
│  Pattern 5: Documentation Sync                                      │
│  ├── Trigger: Merge to main                                        │
│  ├── Action: Update API docs, README, changelog                   │
│  └── Model: Sonnet                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Large Codebase Exploration

### 4.1 The Progressive Understanding Strategy

When exploring an unfamiliar codebase, work from the top down:

```
┌─────────────────────────────────────────────────────────────────────┐
│                Progressive Codebase Exploration                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Level 1: PROJECT OVERVIEW (5 minutes)                              │
│  ├── Read README.md, CLAUDE.md, CONTRIBUTING.md                    │
│  ├── List top-level directory structure                              │
│  ├── Check package.json / pyproject.toml / Cargo.toml              │
│  └── Understand: What does this project do?                        │
│                                                                     │
│  Level 2: ARCHITECTURE (15 minutes)                                 │
│  ├── Identify entry points (main.py, index.ts, cmd/)               │
│  ├── Map the module/package structure                              │
│  ├── Find configuration files                                       │
│  ├── Check database schemas / migrations                           │
│  └── Understand: How is the code organized?                        │
│                                                                     │
│  Level 3: DATA FLOW (30 minutes)                                    │
│  ├── Trace a request from entry to response                        │
│  ├── Identify key interfaces and abstractions                      │
│  ├── Map dependency injection / service wiring                     │
│  ├── Review API routes and their handlers                          │
│  └── Understand: How does data move through the system?            │
│                                                                     │
│  Level 4: DEEP DIVE (as needed)                                     │
│  ├── Focus on the specific module relevant to your task            │
│  ├── Read tests to understand expected behavior                    │
│  ├── Check git blame for recent changes                            │
│  └── Understand: How does this specific piece work?                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Exploration Prompts

Use these prompts for systematic exploration with Claude Code:

```
# Level 1: Project overview
> Explore this project and give me a high-level summary:
> - What does it do?
> - What language/framework is it built with?
> - How is it structured?
> - What are the main entry points?

# Level 2: Architecture mapping
> Map the architecture of this codebase:
> - What are the main modules/packages?
> - What external dependencies does it use?
> - How are the database models structured?
> - What patterns does it follow (MVC, Clean Architecture, etc.)?

# Level 3: Specific feature tracing
> Trace the complete request flow for user authentication:
> - Where does the request enter?
> - What middleware processes it?
> - How is the password verified?
> - What token/session is created?
> - How is the response returned?

# Level 4: Deep dive into a component
> I need to modify the payment processing logic.
> Walk me through:
> - All files involved in payment processing
> - The payment state machine
> - Error handling and retry logic
> - Related tests and their coverage
```

### 4.3 Using Plan Mode and Subagents

For large codebases, combine Plan mode with subagents:

```
# Use Plan mode to create an exploration strategy
> /plan I need to add WebSocket support to this REST API.
> What parts of the codebase do I need to understand first?

# Claude creates a plan without making changes:
# 1. Check current server setup (Express/Fastify/etc.)
# 2. Review existing connection handling
# 3. Identify event patterns that could benefit from WebSocket
# 4. Check authentication middleware compatibility
# 5. Review test infrastructure for adding WebSocket tests

# Then use the plan to guide targeted exploration
> Based on the plan, explore step 1 and 2. Show me the relevant files.
```

### 4.4 Creating Mental Maps

Ask Claude to generate visual maps of the codebase:

```
> Create an ASCII diagram showing the dependency relationships between
> the top-level modules in this project. Include which modules depend
> on which, and mark any circular dependencies.
```

Claude generates:

```
Module Dependency Map:
─────────────────────

  api/
  ├──▶ services/
  │    ├──▶ models/
  │    ├──▶ repositories/
  │    │    └──▶ models/
  │    └──▶ utils/
  ├──▶ middleware/
  │    ├──▶ services/  (auth)
  │    └──▶ utils/     (logging)
  └──▶ schemas/
       └──▶ models/  (for validation)

  workers/
  ├──▶ services/
  └──▶ utils/

  ⚠ Circular: services/email.py → services/user.py → services/email.py
```

---

## 5. Database and API Development

### 5.1 Schema-First Development

Start with the database schema and let Claude generate the rest:

```
> Design a PostgreSQL schema for an e-commerce order management system.
> Requirements:
> - Users can have multiple addresses
> - Orders contain multiple line items
> - Products have variants (size, color)
> - Support for discount codes
> - Audit trail for order status changes
>
> Generate:
> 1. The SQL migration file
> 2. SQLAlchemy models
> 3. Pydantic schemas for the API
> 4. Basic CRUD repository classes
```

### 5.2 API Endpoint Generation

```
> Based on the Order model we just created, generate a complete
> FastAPI router with these endpoints:
>
> POST   /orders              - Create a new order
> GET    /orders              - List orders (with pagination and filters)
> GET    /orders/{id}         - Get order details
> PATCH  /orders/{id}/status  - Update order status
> DELETE /orders/{id}         - Cancel an order (soft delete)
>
> Include:
> - Input validation with Pydantic
> - Proper error handling (404, 422, etc.)
> - Authorization checks (users can only see their own orders)
> - Pagination with cursor-based approach
> - OpenAPI documentation strings
```

### 5.3 Migration Creation

```
> We need to add a "shipping_tracking_number" field to the orders table.
> Generate:
> 1. An Alembic migration for this change
> 2. Update the SQLAlchemy model
> 3. Update the Pydantic schemas
> 4. Update the relevant API endpoints to expose this field
> 5. Add a test for the migration
```

Claude Code handles all five files in a single coordinated change, ensuring consistency across the stack.

---

## 6. Documentation Workflows

### 6.1 Generating Docs from Code

```
> Generate API documentation for all endpoints in src/routes/.
> For each endpoint, document:
> - HTTP method and path
> - Description
> - Request parameters and body schema
> - Response schema with examples
> - Error responses
> - Authentication requirements
>
> Output as a Markdown file at docs/API.md
```

### 6.2 Keeping Docs in Sync

Use hooks to automatically flag when code changes might require documentation updates:

```json
// .claude/settings.json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "command": "python .claude/scripts/check_doc_sync.py \"$CLAUDE_FILE_PATH\""
      }
    ]
  }
}
```

```python
# .claude/scripts/check_doc_sync.py
"""Check if a code change might require documentation updates."""
import sys
import os

filepath = sys.argv[1] if len(sys.argv) > 1 else ""

# Patterns that suggest documentation impact
DOC_TRIGGERS = {
    "routes/": "docs/API.md",
    "models/": "docs/DATABASE.md",
    "config": "docs/CONFIGURATION.md",
    "README": None,  # Already documentation
}

for pattern, doc_file in DOC_TRIGGERS.items():
    if pattern in filepath and doc_file:
        if os.path.exists(doc_file):
            print(f"NOTE: Changes to {filepath} may require updates to {doc_file}")
```

### 6.3 Architecture Decision Records (ADRs)

```
> Create an ADR for our decision to switch from REST to GraphQL
> for the mobile API. Follow the standard ADR format:
> - Title
> - Status
> - Context (why we're considering this)
> - Decision (what we decided)
> - Consequences (positive and negative)
>
> Save to docs/adr/0015-graphql-for-mobile-api.md
```

---

## 7. Case Study: Feature Implementation from Spec to Deploy

Let us walk through a complete feature implementation using Claude Code, from specification to deployment.

### 7.1 The Feature: User Notification Preferences

**Specification**: Users should be able to configure notification preferences (email, SMS, push) for different event types (order updates, promotions, security alerts).

### 7.2 Phase 1: Planning

```
> /plan Design the notification preferences feature.
> Consider:
> - Database schema changes needed
> - API endpoints required
> - Frontend components affected
> - Integration with existing notification service
> - Migration strategy for existing users
> - Test coverage plan
```

Claude Code (in Plan mode) analyzes the existing codebase and produces a plan:

```
Notification Preferences Implementation Plan
=============================================

1. Database Changes (1 migration)
   - New table: notification_preferences
   - Columns: user_id, event_type, channel, enabled, created_at, updated_at
   - Default preferences for existing users (migration data)

2. Backend Changes (4 files)
   - models/notification_preference.py (new)
   - schemas/notification.py (update)
   - routes/notifications.py (update: add 3 endpoints)
   - services/notification_service.py (update: check preferences)

3. Frontend Changes (3 components)
   - NotificationSettings.tsx (new page)
   - NotificationToggle.tsx (new component)
   - settings/index.tsx (add navigation link)

4. Tests (2 files)
   - tests/test_notification_preferences.py (new)
   - tests/test_notification_service.py (update)

5. Migration Strategy
   - Default all existing users to email=on, sms=off, push=on
   - Run migration during maintenance window

Estimated effort: 6 files modified, 4 files created
```

### 7.3 Phase 2: Implementation

Execute the plan step by step:

```
> Let's implement the notification preferences feature following the plan.
> Start with the database migration and model, then the API endpoints,
> then the service integration. Write tests for each layer.
> Use TDD: write the test first, then implement.
```

Claude Code works through each step, running tests as it goes:

```
Step 1: Database migration ✓
  - Created migration 0042_notification_preferences.py
  - Created models/notification_preference.py
  - Tests: 5/5 passing

Step 2: API endpoints ✓
  - GET    /api/v1/users/{id}/notification-preferences
  - PUT    /api/v1/users/{id}/notification-preferences
  - PATCH  /api/v1/users/{id}/notification-preferences/{event_type}
  - Tests: 12/12 passing

Step 3: Service integration ✓
  - Updated NotificationService to check preferences before sending
  - Added preference caching (Redis, 5-minute TTL)
  - Tests: 8/8 passing (including integration tests)

Step 4: Frontend components ✓
  - Created NotificationSettings page
  - Created NotificationToggle component
  - Updated settings navigation
  - Tests: 6/6 passing

All tests: 31/31 passing
```

### 7.4 Phase 3: Review and Polish

```
> Review the complete implementation:
> 1. Check for edge cases we might have missed
> 2. Verify API error handling is comprehensive
> 3. Ensure the migration handles existing data correctly
> 4. Check that the frontend handles loading/error states
> 5. Verify no security issues (IDOR, missing auth checks)
```

### 7.5 Phase 4: Documentation and Deploy

```
> Update the following documentation for the notification preferences feature:
> 1. docs/API.md - add the new endpoints
> 2. CHANGELOG.md - add entry for this feature
> 3. docs/adr/0016-notification-preferences-design.md - create ADR
> 4. README.md - update feature list if needed
```

```
> Create the PR description summarizing all changes and linking to the spec.
```

---

## 8. Exercises

### Exercise 1: Refactoring Practice (Intermediate)

Take a codebase you work with regularly and ask Claude Code to:
1. Identify the three largest functions (by line count)
2. Propose how to break each into smaller, focused functions
3. Execute the refactoring for one of them using TDD

### Exercise 2: CI/CD Setup (Intermediate)

Create a GitHub Actions workflow that uses Claude Code to:
1. Review every pull request and post comments
2. Run on PRs that modify files in `src/` only (skip docs-only PRs)
3. Use Haiku for initial triage, escalate to Sonnet for files with security-sensitive patterns

### Exercise 3: Codebase Exploration (Beginner)

Choose an open-source project you have never worked with and use Claude Code's exploration strategy to:
1. Understand the project structure (Level 1)
2. Map the architecture (Level 2)
3. Trace a specific feature's code path (Level 3)
4. Document your findings in a CLAUDE.md file

### Exercise 4: Full Feature Implementation (Advanced)

Implement a complete feature using the Phase 1-4 workflow:
1. Start with a written specification (2-3 paragraphs)
2. Use Plan mode to create an implementation plan
3. Implement with TDD using Claude Code
4. Review, document, and create a PR

Track how long each phase takes and identify where Claude Code provided the most value.

### Exercise 5: Documentation Generation (Intermediate)

Set up an automated documentation workflow:
1. Write a script that extracts all API endpoints from your codebase
2. Use Claude Code to generate comprehensive API documentation
3. Create a hook that warns when code changes might require doc updates
4. Set up a CI check that verifies docs are up to date

---

## References

- Claude Code Documentation - https://docs.anthropic.com/en/docs/claude-code
- GitHub Actions Documentation - https://docs.github.com/en/actions
- Test-Driven Development by Example - Kent Beck
- Refactoring: Improving the Design of Existing Code - Martin Fowler

---

## Next Steps

[21. Best Practices and Patterns](./21_Best_Practices.md) covers effective prompt writing, context management, security practices, team collaboration patterns, and common anti-patterns to avoid -- all the accumulated wisdom for getting the most out of Claude Code.
