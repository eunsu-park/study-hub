# Agent Teams

**Previous**: [07. Subagents and Task Delegation](./07_Subagents.md) | **Next**: [09. IDE Integration](./09_IDE_Integration.md)

---

While subagents are powerful for delegating isolated tasks, many real-world projects require **coordinated collaboration** between multiple agents. Agent Teams take multi-agent workflows to the next level: instead of independent subagents that report back to a parent, team members share context, coordinate through a shared task list, and communicate progress in real time. This lesson covers the architecture, setup, and practical application of Agent Teams.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Lesson 07: Subagents and Task Delegation
- Understanding of parallel task execution
- Familiarity with project management concepts (task lists, work coordination)

**Learning Objectives**:
- Understand the difference between Agent Teams and isolated subagents
- Explain the team architecture: team lead (orchestrator) + specialized agents
- Describe how agents coordinate work through shared task lists
- Set up and configure an Agent Team for a complex task
- Define team member roles and capabilities appropriately
- Apply Agent Teams to real-world scenarios (migration, review, documentation)
- Identify limitations and cost implications of multi-agent workflows
- Decide when Agent Teams add value vs. when simpler approaches suffice

---

## Table of Contents

1. [What Are Agent Teams?](#1-what-are-agent-teams)
2. [Agent Teams vs. Subagents](#2-agent-teams-vs-subagents)
3. [Team Architecture](#3-team-architecture)
4. [Shared Task Lists](#4-shared-task-lists)
5. [Inter-Agent Communication](#5-inter-agent-communication)
6. [Setting Up Agent Teams](#6-setting-up-agent-teams)
7. [Defining Team Member Roles](#7-defining-team-member-roles)
8. [Practical Examples](#8-practical-examples)
9. [Limitations and Constraints](#9-limitations-and-constraints)
10. [Cost Considerations](#10-cost-considerations)
11. [When Not to Use Teams](#11-when-not-to-use-teams)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. What Are Agent Teams?

An **Agent Team** is a group of specialized AI agents that collaborate on a shared task, coordinated by a team lead (orchestrator). Unlike independent subagents that work in isolation, team members are aware of each other's progress and can adapt their work accordingly.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Team                               │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Team Lead (Opus)                       │   │
│  │           Orchestrates, delegates, synthesizes            │   │
│  └──────────┬──────────────┬──────────────┬─────────────────┘   │
│             │              │              │                      │
│             ▼              ▼              ▼                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │  Specialist  │ │  Specialist  │ │  Specialist  │            │
│  │   Agent A    │ │   Agent B    │ │   Agent C    │            │
│  │ (Security)   │ │ (Performance)│ │ (Style)      │            │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘            │
│         │                │                │                     │
│         └────────────────┴────────────────┘                     │
│                          │                                      │
│                ┌─────────▼─────────┐                            │
│                │  Shared Task List │                            │
│                │  ☑ Task 1 (done)  │                            │
│                │  ▶ Task 2 (active)│                            │
│                │  ☐ Task 3         │                            │
│                │  ☐ Task 4         │                            │
│                └───────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

The key insight is that Agent Teams model how human teams work: a tech lead breaks down a project, assigns work to specialists, monitors progress, and integrates results. The "team lead" agent does the same — decomposing the problem, dispatching work, tracking completion, and handling integration.

---

## 2. Agent Teams vs. Subagents

Understanding the distinction between Agent Teams and subagents is crucial for choosing the right approach.

### Subagents: Independent Workers

```
Main Session ──┬──▶ Subagent A (works alone, reports back)
               ├──▶ Subagent B (works alone, reports back)
               └──▶ Subagent C (works alone, reports back)
```

- Each subagent works in **complete isolation**
- No awareness of what other subagents are doing
- No shared task list or coordination mechanism
- Main session must manually integrate results
- Simple to set up, lightweight

### Agent Teams: Coordinated Collaborators

```
Team Lead ──┬──▶ Agent A ──┐
            ├──▶ Agent B ──┤──▶ Shared Task List + Communication
            └──▶ Agent C ──┘
```

- Agents are **aware** of the team's overall task
- **Shared task list** enables coordination
- Team lead **orchestrates** and resolves conflicts
- Agents can **adapt** based on other agents' progress
- More complex to set up, higher overhead

### Comparison Table

| Dimension | Subagents | Agent Teams |
|-----------|-----------|-------------|
| Context sharing | None (isolated) | Shared via task list and team lead |
| Coordination | Manual (by parent) | Automatic (by team lead) |
| Communication | Report to parent only | Progress updates visible to team |
| Conflict resolution | Parent handles | Team lead handles |
| Setup complexity | Low | Higher |
| Token cost | Lower | Higher (coordination overhead) |
| Best for | Independent, parallelizable tasks | Interconnected, collaborative tasks |
| Failure handling | Parent retries individual agents | Team lead reassigns or adapts |

### When Each Approach Wins

**Subagents win** when tasks are truly independent:
- Translating 10 files in parallel (each file is self-contained)
- Searching for different patterns across the codebase
- Running tests in separate modules

**Agent Teams win** when tasks are interconnected:
- Migrating a codebase where changes in one module affect others
- Comprehensive code review requiring multiple perspectives
- Building a feature that spans frontend, backend, and database

---

## 3. Team Architecture

### The Team Lead (Orchestrator)

The team lead is typically an Opus-level agent responsible for:

1. **Task Decomposition**: Breaking the overall goal into specific work items
2. **Agent Assignment**: Deciding which specialist handles which task
3. **Progress Monitoring**: Tracking completion and identifying blockers
4. **Integration**: Combining outputs from specialists into a coherent result
5. **Conflict Resolution**: Handling overlapping edits or contradictory outputs
6. **Quality Assurance**: Verifying that work meets requirements

```python
# The team lead agent operates with full context awareness
# It maintains the shared task list and coordinates work

# Team lead's mental model:
"""
Overall Goal: Migrate from REST API to GraphQL

Work Breakdown:
1. Schema Design Agent → Define GraphQL types from existing models
2. Resolver Agent → Implement resolvers from route handlers
3. Testing Agent → Create integration tests for GraphQL endpoints
4. Documentation Agent → Update API docs for GraphQL

Dependencies:
- Resolver Agent needs Schema Design Agent's output
- Testing Agent needs both Schema and Resolver outputs
- Documentation Agent can start with Schema, update after Resolvers
"""
```

### Specialist Agents

Specialist agents are focused workers with domain-specific capabilities. Each specialist:

- Has a **clear role definition** (what they do and do not do)
- Works on **assigned tasks** from the shared task list
- Reports **progress and findings** back to the team lead
- Can **flag issues** that need team lead intervention

### Communication Flow

```
┌──────────────────────────────────────────────────────────────┐
│                     Communication Flow                        │
│                                                              │
│  Team Lead                                                   │
│  │                                                           │
│  ├──▶ "Agent A: Design the database schema for users,        │
│  │     products, and orders. Mark your task complete when     │
│  │     done."                                                │
│  │                                                           │
│  │  Agent A:                                                 │
│  │  └──▶ "Schema designed. Created 3 migration files.        │
│  │        Note: 'orders' table needs a polymorphic           │
│  │        'payment_type' column — flagging for Agent B."     │
│  │                                                           │
│  ├──▶ "Agent B: Implement the payment module. Note from      │
│  │     Agent A: orders table has polymorphic payment_type."  │
│  │                                                           │
│  │  Agent B:                                                 │
│  │  └──▶ "Payment module implemented. Tests passing.         │
│  │        Discovered: we need a webhook handler for           │
│  │        async payment notifications."                      │
│  │                                                           │
│  ├──▶ "Agent C: Implement webhook handler for payment         │
│  │     notifications..." (new task based on Agent B's find)  │
│  │                                                           │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Shared Task Lists

The shared task list is the coordination mechanism that makes Agent Teams more than just parallel subagents. It provides visibility into what work is pending, in progress, and completed.

### Task List Structure

```
┌────────────────────────────────────────────────────────────────┐
│  Shared Task List: "Migrate REST API to GraphQL"              │
│                                                                │
│  ☑ 1. Audit existing REST endpoints          [Schema Agent]   │
│  ☑ 2. Design GraphQL type definitions        [Schema Agent]   │
│  ▶ 3. Implement User resolvers               [Resolver Agent] │
│  ▶ 4. Implement Product resolvers            [Resolver Agent] │
│  ☐ 5. Implement Order resolvers              [Resolver Agent] │
│  ☐ 6. Write integration tests                [Testing Agent]  │
│  ☐ 7. Update API documentation               [Docs Agent]     │
│  ☐ 8. Set up GraphQL playground              [Resolver Agent] │
│  ☑ 9. Configure Apollo Server                [Schema Agent]   │
│                                                                │
│  Legend: ☑ = completed, ▶ = in progress, ☐ = pending          │
└────────────────────────────────────────────────────────────────┘
```

### How Agents Update the Task List

The team lead creates and manages the task list using the `TodoWrite` tool. As agents complete their work, the team lead updates task statuses:

```python
# Team lead creates initial task list
TodoWrite(todos=[
    {"content": "Audit existing REST endpoints",
     "status": "pending",
     "activeForm": "Auditing existing REST endpoints"},
    {"content": "Design GraphQL type definitions",
     "status": "pending",
     "activeForm": "Designing GraphQL type definitions"},
    {"content": "Implement User resolvers",
     "status": "pending",
     "activeForm": "Implementing User resolvers"},
    # ... more tasks
])
```

### Task Dependencies

The team lead manages dependencies by controlling task assignment order:

```
Task 1 (Schema) ──┐
                   ├──▶ Task 3 (Resolvers) ──▶ Task 6 (Tests)
Task 2 (Config) ──┘                             │
                                                 ▼
                                          Task 7 (Docs)
```

The team lead ensures that dependent tasks are not started until their prerequisites complete:

```python
# Team lead logic (conceptual):
# 1. Launch Schema Agent for Tasks 1-2
# 2. Wait for Tasks 1-2 to complete
# 3. Launch Resolver Agent for Tasks 3-5 (using Schema Agent's output)
# 4. Wait for Tasks 3-5 to complete
# 5. Launch Testing Agent for Task 6 and Docs Agent for Task 7 (parallel)
```

---

## 5. Inter-Agent Communication

Since agents cannot directly talk to each other (they are separate context windows), all communication flows through the team lead.

### Communication Patterns

**1. Team Lead as Router**

The team lead receives outputs from agents and routes relevant information to other agents:

```
Agent A output: "Found 3 circular dependencies in the models"
     │
     ▼
Team Lead: Includes this finding in Agent B's prompt
     │
     ▼
Agent B prompt: "...Note: Agent A found circular dependencies in
                 User→Order→User, Product→Category→Product,
                 and Payment→Order→Payment. Account for these
                 in your refactoring plan."
```

**2. Progress Broadcasting**

The team lead summarizes overall progress when launching new agent tasks:

```python
Task(
    subagent_type="general-purpose",
    prompt="""Team Progress Update:
    - Schema design: COMPLETE (3 types defined: User, Product, Order)
    - User resolvers: COMPLETE (CRUD + authentication)
    - Product resolvers: IN PROGRESS (another agent is handling this)

    Your task: Implement Order resolvers in src/graphql/resolvers/orders.js
    The Order type references User and Product types.
    See the schema in src/graphql/schema/order.graphql for type definitions.
    Follow the same patterns used in the User resolvers."""
)
```

**3. Issue Escalation**

When a specialist encounters an issue they cannot resolve, they report it to the team lead:

```
Agent B: "Cannot implement payment resolver — the Stripe API
          key is not in the environment variables. This is a
          blocker. Need: STRIPE_SECRET_KEY and STRIPE_WEBHOOK_SECRET
          in .env file."

Team Lead: Decides to either:
  - Create a new task for another agent to set up Stripe config
  - Handle it directly
  - Ask the user for the missing credentials
```

---

## 6. Setting Up Agent Teams

### Step 1: Define the Overall Goal

Start with a clear, well-scoped objective:

```python
team_goal = """
Perform a comprehensive security audit of the web application
in /opt/projects/myapp/. The audit should cover:
1. Static code analysis for common vulnerabilities
2. Dependency vulnerability scanning
3. Authentication and authorization review
4. API endpoint security assessment
5. Configuration and secrets management review

Deliverable: A security report with findings categorized by
severity (Critical/High/Medium/Low) and remediation steps.
"""
```

### Step 2: Design the Team Structure

Determine what specialists you need:

```python
team_structure = {
    "team_lead": {
        "model": "opus",
        "role": "Orchestrate the security audit, synthesize findings"
    },
    "specialists": [
        {
            "name": "Code Analyzer",
            "model": "sonnet",
            "focus": "Static analysis of source code for vulnerabilities"
        },
        {
            "name": "Dependency Auditor",
            "model": "sonnet",
            "focus": "Check all dependencies for known CVEs"
        },
        {
            "name": "Auth Reviewer",
            "model": "sonnet",
            "focus": "Review authentication and authorization logic"
        },
        {
            "name": "API Tester",
            "model": "sonnet",
            "focus": "Assess API endpoint security (headers, CORS, rate limiting)"
        }
    ]
}
```

### Step 3: Create the Task List

The team lead decomposes the goal into concrete tasks:

```python
TodoWrite(todos=[
    {"content": "Scan source code for injection vulnerabilities (SQL, XSS, command)",
     "status": "pending",
     "activeForm": "Scanning for injection vulnerabilities"},
    {"content": "Audit npm/pip dependencies for known CVEs",
     "status": "pending",
     "activeForm": "Auditing dependencies for CVEs"},
    {"content": "Review JWT implementation and session management",
     "status": "pending",
     "activeForm": "Reviewing JWT and session management"},
    {"content": "Check API endpoints for missing auth, rate limits, CORS",
     "status": "pending",
     "activeForm": "Checking API endpoint security"},
    {"content": "Scan for hardcoded secrets and misconfigured environment variables",
     "status": "pending",
     "activeForm": "Scanning for hardcoded secrets"},
    {"content": "Review file upload handling for path traversal and type validation",
     "status": "pending",
     "activeForm": "Reviewing file upload security"},
    {"content": "Compile findings into security report with severity ratings",
     "status": "pending",
     "activeForm": "Compiling security report"}
])
```

### Step 4: Launch and Coordinate

```python
# Phase 1: Independent analysis tasks (parallel)
code_analysis = Task(
    subagent_type="general-purpose",
    model="sonnet",
    prompt="""You are the Code Analyzer on a security audit team.
    Scan all source files in /opt/projects/myapp/src/ for:
    1. SQL injection (raw queries, string concatenation in SQL)
    2. XSS (unescaped output, innerHTML usage)
    3. Command injection (exec, system, eval)
    4. Path traversal (user input in file paths)
    5. Insecure deserialization (pickle, yaml.load)

    For each finding, report:
    - File path and line number
    - Vulnerability type (CWE ID if known)
    - Severity (Critical/High/Medium/Low)
    - Code snippet showing the vulnerability
    - Recommended fix

    Search thoroughly — check every source file."""
)

dependency_audit = Task(
    subagent_type="general-purpose",
    model="sonnet",
    prompt="""You are the Dependency Auditor on a security audit team.
    Check all project dependencies for known vulnerabilities:
    1. Read package.json / requirements.txt / Pipfile
    2. Run 'npm audit' or 'pip-audit' if available
    3. Check for outdated packages with known CVEs
    4. Identify packages that are unmaintained or deprecated

    Report format: package name, version, CVE ID, severity, fix version."""
)

auth_review = Task(
    subagent_type="general-purpose",
    model="sonnet",
    prompt="""You are the Auth Reviewer on a security audit team.
    Review the authentication and authorization system:
    1. How are passwords stored? (bcrypt/argon2/plaintext?)
    2. JWT: algorithm, expiration, refresh token rotation
    3. Session management: storage, fixation prevention
    4. Authorization: RBAC/ABAC implementation, privilege escalation risks
    5. Password reset flow: token generation, expiration, reuse prevention

    Check all files in src/auth/, src/middleware/, and related configs."""
)

# Phase 2: Team lead synthesizes results (after Phase 1 completes)
# The team lead reads all outputs and compiles the final report
```

---

## 7. Defining Team Member Roles

Effective Agent Teams require clear role definitions. Each team member should have:

### Role Definition Template

```yaml
# Template for team member role definition
name: "Role Name"
responsibilities:
  - Specific task area 1
  - Specific task area 2
boundaries:
  - What this agent should NOT do
  - Which files/modules are out of scope
input:
  - What context the agent needs
output:
  - What the agent should produce
  - Expected format
```

### Example: Code Review Team

```yaml
# Team Member 1: Security Reviewer
name: "Security Reviewer"
responsibilities:
  - Identify security vulnerabilities in code changes
  - Check for OWASP Top 10 issues
  - Verify input validation and output encoding
  - Review authentication/authorization changes
boundaries:
  - Do NOT review code style or formatting
  - Do NOT suggest performance optimizations
  - Do NOT modify any code
output:
  - List of security findings with severity ratings
  - Each finding includes: file, line, issue, remediation
```

```yaml
# Team Member 2: Performance Reviewer
name: "Performance Reviewer"
responsibilities:
  - Identify performance bottlenecks
  - Check for N+1 queries, unnecessary allocations
  - Review algorithm complexity
  - Verify caching strategies
boundaries:
  - Do NOT review security aspects
  - Do NOT review code style
  - Do NOT modify any code
output:
  - List of performance findings with impact ratings
  - Include benchmark suggestions where applicable
```

```yaml
# Team Member 3: Style Reviewer
name: "Style Reviewer"
responsibilities:
  - Check adherence to project coding standards
  - Verify naming conventions
  - Review documentation and comments
  - Check for DRY violations and code smells
boundaries:
  - Do NOT review security or performance
  - Do NOT modify any code
output:
  - List of style issues
  - References to violated coding standards
```

```yaml
# Team Lead: Review Coordinator
name: "Review Coordinator"
responsibilities:
  - Assign review tasks to specialists
  - Compile findings from all reviewers
  - Prioritize and deduplicate findings
  - Generate final review report
  - Identify conflicting recommendations
output:
  - Unified code review report
  - Prioritized list of changes needed
```

---

## 8. Practical Examples

### Example 1: Large-Scale Codebase Migration

**Scenario**: Migrate a 50,000-line Python 2 codebase to Python 3.

```
Team Structure:
├── Team Lead (Opus): Plans migration order, tracks progress
├── Syntax Agent (Sonnet): print statements, unicode, division
├── Library Agent (Sonnet): Replace deprecated libraries
├── Test Agent (Sonnet): Fix test suite for Python 3
└── Compatibility Agent (Sonnet): Add __future__ imports, six wrappers

Workflow:
1. Team Lead scans codebase and creates module dependency graph
2. Team Lead orders modules for migration (leaf modules first)
3. Syntax + Library Agents work on independent modules in parallel
4. Test Agent follows behind, running and fixing tests
5. Compatibility Agent handles shared code needing dual compatibility
6. Team Lead integrates results and verifies no regressions
```

```python
# Phase 1: Analysis (Explore subagents, parallel)
module_map = Task(
    subagent_type="explore",
    prompt="Map all Python modules in src/ with their import dependencies. "
           "Identify leaf modules (no internal imports) vs core modules."
)

py2_patterns = Task(
    subagent_type="explore",
    prompt="Find all Python 2-specific patterns in src/: "
           "print statements, unicode literals, old-style classes, "
           "dict.has_key(), map/filter returning lists, etc. "
           "Count occurrences per file."
)

# Phase 2: Migration (General-Purpose subagents, coordinated)
# Team lead uses Phase 1 results to plan migration order
# Then launches agents for each module group
```

### Example 2: Comprehensive Code Review

**Scenario**: Review a 30-file pull request touching authentication, API, and database layers.

```python
# Launch three specialist reviewers in parallel
security_review = Task(
    subagent_type="explore",
    model="sonnet",
    prompt="""Security Review of PR files:
    src/auth/login.py, src/auth/jwt.py, src/auth/permissions.py,
    src/api/users.py, src/api/admin.py, src/middleware/rate_limit.py

    Check for:
    - Authentication bypass
    - Authorization flaws (privilege escalation)
    - Input validation gaps
    - Token handling issues
    - Rate limiting effectiveness

    Report findings with severity ratings."""
)

performance_review = Task(
    subagent_type="explore",
    model="sonnet",
    prompt="""Performance Review of PR files:
    src/models/user.py, src/models/order.py, src/api/search.py,
    src/utils/cache.py, src/middleware/logging.py

    Check for:
    - N+1 query patterns
    - Missing database indexes
    - Unnecessary data loading
    - Cache invalidation issues
    - Logging overhead in hot paths

    Report findings with impact estimates."""
)

style_review = Task(
    subagent_type="explore",
    model="sonnet",
    prompt="""Style Review of all 30 PR files.
    Project standards: PEP 8, type hints required, docstrings for
    public functions, max line length 100.

    Check for:
    - PEP 8 violations
    - Missing type hints
    - Missing or inadequate docstrings
    - DRY violations
    - Inconsistent naming

    Report only significant issues (not nitpicks)."""
)

# Team lead compiles all three reviews into a unified PR review
```

### Example 3: Multi-Language Documentation Generation

**Scenario**: Generate API documentation in English, Korean, and Japanese.

```python
# Step 1: Generate English documentation (authoritative source)
english_docs = Task(
    subagent_type="general-purpose",
    model="opus",
    prompt="""Generate comprehensive API documentation for all endpoints
    in src/api/. Read each route handler and create documentation with:
    - Endpoint URL and HTTP method
    - Request parameters (path, query, body) with types
    - Response format with example JSON
    - Authentication requirements
    - Rate limiting details
    - Error responses

    Save to docs/api/en/ as individual Markdown files per resource."""
)

# Step 2: Parallel translation (after English docs are complete)
korean_docs = Task(
    subagent_type="general-purpose",
    model="sonnet",
    prompt="""Translate all API documentation files from docs/api/en/
    to Korean. Save to docs/api/ko/.
    Rules:
    - Do NOT translate code blocks, URLs, or HTTP methods
    - Use Korean technical terms with English in parentheses
    - Preserve all formatting and structure"""
)

japanese_docs = Task(
    subagent_type="general-purpose",
    model="sonnet",
    prompt="""Translate all API documentation files from docs/api/en/
    to Japanese. Save to docs/api/ja/.
    Rules:
    - Do NOT translate code blocks, URLs, or HTTP methods
    - Use standard Japanese technical terminology
    - Preserve all formatting and structure"""
)
```

### Example 4: Research + Implementation Pipeline

**Scenario**: Add WebSocket support to an existing HTTP API.

```python
# Phase 1: Research (Plan agent)
research = Task(
    subagent_type="plan",
    model="opus",
    prompt="""Research the best approach to add WebSocket support to our
    Express.js application.

    Current stack: Express.js 4.x, Node.js 20, PostgreSQL, Redis
    Requirements: real-time notifications, live data feeds, chat

    Research:
    1. ws vs socket.io vs uWebSockets — performance, features, ecosystem
    2. Authentication strategy for WebSocket connections
    3. Scaling WebSockets across multiple server instances (Redis pub/sub?)
    4. Reconnection and heartbeat strategies
    5. Message format standards (JSON-RPC? custom protocol?)

    Produce a detailed implementation plan with file-by-file changes."""
)

# Phase 2: Implementation (parallel General-Purpose agents)
# Using the research output to guide implementation
server_impl = Task(
    subagent_type="general-purpose",
    model="sonnet",
    prompt=f"""Based on this implementation plan:
    {research}

    Implement the WebSocket server module:
    1. Install chosen library
    2. Create src/websocket/server.js
    3. Integrate with Express HTTP server
    4. Implement authentication middleware for WS connections
    5. Set up Redis pub/sub for multi-instance support"""
)

client_impl = Task(
    subagent_type="general-purpose",
    model="sonnet",
    prompt=f"""Based on this implementation plan:
    {research}

    Implement the WebSocket client library:
    1. Create src/client/websocket.js
    2. Implement auto-reconnection with exponential backoff
    3. Add heartbeat/ping-pong mechanism
    4. Create event subscription API
    5. Write usage examples and documentation"""
)
```

---

## 9. Limitations and Constraints

### Context Window Limits

Each agent in the team still has a finite context window. For very large codebases, even a single agent may not be able to hold all relevant files in context:

```
Problem: Agent needs to review 200 files
         200 files × ~500 lines × ~4 tokens/line = 400K tokens
         Exceeds context window

Solution: Team lead assigns groups of related files to different agents
          Agent A: src/api/ (50 files)
          Agent B: src/models/ (30 files)
          Agent C: src/services/ (40 files)
          Agent D: src/utils/ + src/middleware/ (80 files)
```

### No Direct Agent-to-Agent Communication

Agents cannot talk to each other directly. All communication must route through the team lead:

```
Agent A ──X──▶ Agent B    (NOT possible)
Agent A ──▶ Team Lead ──▶ Agent B    (how it actually works)
```

This means:
- The team lead is a bottleneck for information flow
- The team lead must decide what information to pass between agents
- Some context is inevitably lost in translation

### Conflicting Outputs

When multiple agents modify the same files, conflicts can occur:

```
Agent A edits src/config.js: adds database pool config
Agent B edits src/config.js: adds Redis cache config
Result: The last agent's version wins; Agent A's changes may be lost
```

**Mitigation strategies**:
- Assign non-overlapping file ownership to agents
- Have agents propose changes (as diffs) rather than directly editing
- Team lead reviews and merges conflicting changes
- Use a sequential approach for shared files

### Rate Limiting

Running many agents simultaneously can hit API rate limits:

```
10 Sonnet agents × rapid tool calls = potential rate limit errors

Mitigation:
- Limit parallel agents to 5-7
- Use staggered launches
- Team lead monitors for rate limit errors and pauses if needed
```

### Agent Reliability

Agents can produce incorrect outputs, miss edge cases, or misunderstand instructions:

```
Risk: Agent modifies code incorrectly, breaking functionality
Mitigation:
- Run tests after each agent's work
- Use Explore agents to verify General-Purpose agents' output
- Team lead performs final review
- Keep Git commits granular for easy rollback
```

---

## 10. Cost Considerations

Multi-agent workflows consume more tokens than single-agent work. Understanding the cost structure helps you make informed decisions.

### Token Cost Breakdown

```
Single Agent Workflow:
  Input tokens:  ~50K  (system prompt + conversation + file reads)
  Output tokens: ~10K  (responses + tool calls)
  Total: ~60K tokens

Agent Team (4 specialists + 1 lead):
  Team Lead:     ~30K input + ~8K output = ~38K
  Specialist 1:  ~40K input + ~10K output = ~50K
  Specialist 2:  ~40K input + ~10K output = ~50K
  Specialist 3:  ~40K input + ~10K output = ~50K
  Specialist 4:  ~40K input + ~10K output = ~50K
  Total: ~238K tokens (roughly 4x single agent)
```

### Cost Optimization Strategies

**1. Use the cheapest model that works**:

```python
# Opus for the team lead (needs complex reasoning)
# Sonnet for specialists (good enough for focused tasks)
# Haiku for simple, mechanical tasks (searching, formatting)

team_lead:  model="opus"      # $15/M input, $75/M output
specialist: model="sonnet"    # $3/M input, $15/M output
formatter:  model="haiku"     # $0.25/M input, $1.25/M output
```

**2. Minimize redundant file reading**:

```python
# BAD: Every agent reads the same config file
Agent A: reads src/config.js (duplicated tokens)
Agent B: reads src/config.js (duplicated tokens)
Agent C: reads src/config.js (duplicated tokens)

# GOOD: Team lead reads once, includes relevant info in prompts
Team Lead: reads src/config.js
Agent A prompt: "Database is PostgreSQL on port 5432, pool size 20..."
Agent B prompt: "Redis cache is on port 6379, TTL 300s..."
```

**3. Limit agent scope strictly**:

```python
# BAD: Vague scope leads to unnecessary exploration
Task(prompt="Review everything in the project...")

# GOOD: Tight scope minimizes wasted tokens
Task(prompt="Review ONLY src/api/payments.py for SQL injection. "
           "Do not read other files unless directly imported.")
```

**4. Use Explore before General-Purpose**:

```python
# Explore agents are cheaper (fewer tools, faster completion)
# Use them to narrow down what General-Purpose agents need to do

analysis = Task(subagent_type="explore",
                prompt="Which files need changes for feature X?")
# Only then launch General-Purpose for the specific files identified
```

---

## 11. When Not to Use Teams

Agent Teams are powerful but not always appropriate. Do NOT use teams when:

### The Task is Simple

If a single agent can handle the task in a few turns, the coordination overhead of a team is wasteful:

```
# Do NOT use a team for:
- Fixing a single bug
- Adding a test case
- Updating a configuration value
- Renaming a function across 5 files
```

### The Task is Highly Sequential

If every step depends on the previous step's output, there is no parallelism to exploit:

```
Step 1: Design schema    → Step 2: Create migration
→ Step 3: Update models  → Step 4: Update API
→ Step 5: Update tests   → Step 6: Update docs

This is better handled as a chain of subagents or a single agent.
```

### Context Sharing is Critical

If agents need to share a deep understanding of nuanced design decisions:

```
# A single agent that maintains full context across all decisions
# is better than a team where each agent has partial context
```

### Budget is Constrained

Agent Teams cost 3-5x more than a single agent. If token budget is a concern, use a single agent with sequential subagent calls instead.

### Quick Decision Framework

```
Should I use an Agent Team?

1. Is the task large enough to justify coordination overhead? (>1 hour of single-agent work)
   └── No → Use single agent or subagents

2. Can the work be meaningfully parallelized?
   └── No → Use sequential subagent chain

3. Do agents need to be aware of each other's progress?
   └── No → Use independent subagents

4. Is the budget sufficient for 3-5x token cost?
   └── No → Use single agent with careful context management

5. All yes? → Use an Agent Team
```

---

## 12. Exercises

### Exercise 1: Team Design

Design an Agent Team for migrating a Python project from `unittest` to `pytest`. Define:
1. Team structure (lead + specialists)
2. Shared task list (at least 8 tasks)
3. Task dependencies
4. Estimated token cost vs. single-agent approach

### Exercise 2: Role Definition

Write complete role definitions (following the template in Section 7) for a 3-agent team that performs:
1. Code review
2. Documentation update
3. Test generation

For a pull request that adds a new REST API endpoint.

### Exercise 3: Conflict Resolution

Two agents on your team both modify `src/config.py`:
- Agent A adds a `DATABASE_POOL_SIZE = 20` setting
- Agent B adds a `CACHE_TTL = 300` setting

Both changes are correct and needed. Design a strategy for the team lead to handle this conflict. Consider:
- How to detect the conflict
- How to merge both changes
- How to prevent this in future team setups

### Exercise 4: Cost Analysis

Given this team configuration:
- 1 Opus team lead (estimated 40K input + 15K output tokens)
- 3 Sonnet specialists (estimated 50K input + 12K output tokens each)
- 1 Haiku formatter (estimated 20K input + 8K output tokens)

Calculate:
1. Total token consumption
2. Total cost (use current Anthropic pricing)
3. How much a single Sonnet agent would cost for the same work (estimate)
4. The "coordination premium" (team cost / single agent cost)

---

## 13. References

- [Claude Code Documentation: Multi-Agent Workflows](https://docs.anthropic.com/en/docs/claude-code)
- [Anthropic Research: Multi-Agent Coordination](https://www.anthropic.com/research)
- [Anthropic Blog: Agent Teams](https://www.anthropic.com/engineering)
- [Multi-Agent Systems: A Survey](https://arxiv.org/abs/2402.01680)
- [Claude Code GitHub Repository](https://github.com/anthropics/claude-code)

---

## Next Steps

In the next lesson, [IDE Integration](./09_IDE_Integration.md), we shift from multi-agent orchestration to developer experience. You will learn how to use Claude Code inside VS Code and JetBrains IDEs, where inline diff review and @-mentions make the agent interaction feel like a native part of your development environment.
