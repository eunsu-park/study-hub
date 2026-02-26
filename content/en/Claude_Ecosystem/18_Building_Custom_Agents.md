# Building Custom Agents

**Previous**: [17. Claude Agent SDK](./17_Claude_Agent_SDK.md) | **Next**: [19. Models, Pricing, and Optimization](./19_Models_and_Pricing.md)

---

Building a custom agent means going beyond using the SDK out of the box. It means designing an agent with a specific purpose, equipping it with custom tools, crafting system prompts that constrain its behavior, and deploying it reliably in production. This lesson walks through the full process: from agent design to deployment, with four detailed practical examples and guidance on testing, monitoring, and scaling.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Agent SDK basics from Lesson 17
- Tool use patterns from Lesson 16
- Python 3.10+ or TypeScript/Node.js 18+
- Understanding of async programming

## Learning Objectives

After completing this lesson, you will be able to:

1. Design agents with clear scope, appropriate tools, and safety constraints
2. Build custom tools with JSON Schema definitions, executors, and error handling
3. Write effective system prompts that guide agent behavior reliably
4. Implement four complete agent examples for real-world use cases
5. Test agents at the unit, integration, and evaluation levels
6. Deploy agents with rate limiting, cost monitoring, logging, and security
7. Design multi-agent systems with orchestration patterns

---

## Table of Contents

1. [Designing an Agent](#1-designing-an-agent)
2. [Custom Tool Development](#2-custom-tool-development)
3. [System Prompt Engineering for Agents](#3-system-prompt-engineering-for-agents)
4. [Practical Agent Examples](#4-practical-agent-examples)
5. [Testing Agents](#5-testing-agents)
6. [Production Deployment](#6-production-deployment)
7. [Scaling Patterns](#7-scaling-patterns)
8. [Exercises](#8-exercises)
9. [References](#9-references)

---

## 1. Designing an Agent

Before writing code, answer these design questions:

### 1.1 Agent Design Canvas

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Design Canvas                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PURPOSE                                                      │
│     What problem does this agent solve?                          │
│     What is the expected input and output?                       │
│                                                                  │
│  2. SCOPE                                                        │
│     What CAN the agent do? (capabilities)                        │
│     What CANNOT the agent do? (boundaries)                       │
│     What SHOULD the agent refuse to do? (safety)                 │
│                                                                  │
│  3. TOOLS                                                        │
│     What built-in tools does it need?                            │
│     What custom tools must be created?                           │
│     What external APIs or services does it interact with?        │
│                                                                  │
│  4. CONSTRAINTS                                                  │
│     Maximum execution time?                                      │
│     Token/cost budget?                                           │
│     Permission boundaries?                                       │
│     Rate limits on external services?                            │
│                                                                  │
│  5. FAILURE MODES                                                │
│     What happens when tools fail?                                │
│     What happens when the task is ambiguous?                     │
│     What happens when the agent gets stuck in a loop?            │
│     What is the fallback behavior?                               │
│                                                                  │
│  6. EVALUATION                                                   │
│     How do you measure success?                                  │
│     What does a "good" output look like?                         │
│     What are the edge cases?                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Design Example: Code Review Agent

```
Purpose:   Analyze pull requests for bugs, style issues, and security concerns.
           Output a structured review with severity-ranked findings.

Scope:
  CAN:     Read code, run linters, check patterns, generate review comments
  CANNOT:  Push code, merge PRs, modify files
  REFUSE:  Approving PRs without human review, accessing private credentials

Tools:
  Built-in: Read, Glob, Grep, Bash (restricted to linters)
  Custom:   fetch_pr_diff, post_review_comment, get_style_guide

Constraints:
  - Max 30 turns per review
  - Max 5 minutes wall-clock time
  - Bash restricted to: ruff, mypy, eslint, shellcheck
  - Read-only file access

Failure modes:
  - PR too large (>5000 lines): split into file groups
  - Linter crashes: skip linter, note in review
  - Ambiguous code: flag for human review, don't guess

Evaluation:
  - Percentage of real bugs found (recall)
  - False positive rate (precision)
  - Review completion time
  - Human reviewer agreement rate
```

---

## 2. Custom Tool Development

Custom tools extend the agent beyond its built-in capabilities. Each tool requires a schema definition, an executor, and error handling.

### 2.1 Tool Schema Definition

Define your tool using the Messages API tool format (JSON Schema):

```python
# tools.py — Custom tool definitions

TOOLS = [
    {
        "name": "fetch_pr_diff",
        "description": (
            "Fetch the diff for a GitHub pull request. "
            "Returns the unified diff showing all changed files, "
            "additions, and deletions. Use this to understand what "
            "code was changed in a PR."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Repository in 'owner/name' format, e.g., 'anthropics/claude-code'",
                },
                "pr_number": {
                    "type": "integer",
                    "description": "Pull request number",
                    "minimum": 1,
                },
                "file_filter": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files, e.g., '*.py'",
                },
            },
            "required": ["repo", "pr_number"],
        },
    },
    {
        "name": "run_linter",
        "description": (
            "Run a code linter on a specific file or directory. "
            "Supported linters: ruff (Python), eslint (JavaScript/TypeScript), "
            "shellcheck (Bash). Returns linter output with file, line, "
            "severity, and message for each finding."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "linter": {
                    "type": "string",
                    "enum": ["ruff", "eslint", "shellcheck"],
                    "description": "Which linter to run",
                },
                "target": {
                    "type": "string",
                    "description": "File or directory path to lint",
                },
                "config": {
                    "type": "string",
                    "description": "Optional path to linter config file",
                },
            },
            "required": ["linter", "target"],
        },
    },
    {
        "name": "search_codebase",
        "description": (
            "Search the codebase for patterns, function definitions, "
            "or usage of specific APIs. Returns matching locations "
            "with file path, line number, and surrounding context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "file_type": {
                    "type": "string",
                    "description": "File extension filter, e.g., 'py', 'ts'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                },
            },
            "required": ["pattern"],
        },
    },
]
```

### 2.2 Tool Executor Implementation

```python
# executors.py — Tool execution logic

import json
import subprocess
import re
from pathlib import Path
from typing import Any


class ToolExecutor:
    """Execute custom tools with validation and error handling."""

    def __init__(self, project_root: str, github_token: str | None = None):
        self.project_root = Path(project_root)
        self.github_token = github_token

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict:
        """Route tool calls to the appropriate handler."""
        handlers = {
            "fetch_pr_diff": self._fetch_pr_diff,
            "run_linter": self._run_linter,
            "search_codebase": self._search_codebase,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return handler(**tool_input)
        except Exception as e:
            return {
                "error": f"Tool execution failed: {type(e).__name__}: {str(e)}"
            }

    def _fetch_pr_diff(
        self, repo: str, pr_number: int, file_filter: str | None = None
    ) -> dict:
        """Fetch PR diff from GitHub."""
        # Validate inputs
        if not re.match(r"^[\w.-]+/[\w.-]+$", repo):
            return {"error": f"Invalid repo format: {repo}. Use 'owner/name'."}

        cmd = ["gh", "pr", "diff", str(pr_number), "--repo", repo]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            return {"error": f"GitHub CLI error: {result.stderr.strip()}"}

        diff = result.stdout

        # Apply file filter if provided
        if file_filter:
            import fnmatch
            filtered_lines = []
            include_file = False
            for line in diff.splitlines():
                if line.startswith("diff --git"):
                    # Extract filename
                    parts = line.split(" b/")
                    if len(parts) > 1:
                        filename = parts[1]
                        include_file = fnmatch.fnmatch(filename, file_filter)
                if include_file:
                    filtered_lines.append(line)
            diff = "\n".join(filtered_lines)

        # Truncate very large diffs
        if len(diff) > 50000:
            diff = diff[:50000] + "\n\n... [truncated, diff too large] ..."

        return {
            "repo": repo,
            "pr_number": pr_number,
            "diff": diff,
            "line_count": len(diff.splitlines()),
        }

    def _run_linter(
        self, linter: str, target: str, config: str | None = None
    ) -> dict:
        """Run a linter and return structured results."""
        # Validate target path (prevent directory traversal)
        target_path = (self.project_root / target).resolve()
        if not str(target_path).startswith(str(self.project_root.resolve())):
            return {"error": "Path traversal detected. Target must be within project root."}

        if not target_path.exists():
            return {"error": f"Target not found: {target}"}

        # Build linter command
        commands = {
            "ruff": ["ruff", "check", "--output-format=json"],
            "eslint": ["npx", "eslint", "--format=json"],
            "shellcheck": ["shellcheck", "--format=json"],
        }

        cmd = commands.get(linter)
        if not cmd:
            return {"error": f"Unsupported linter: {linter}"}

        if config:
            config_path = (self.project_root / config).resolve()
            if config_path.exists():
                cmd.extend(["--config", str(config_path)])

        cmd.append(str(target_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.project_root),
            )

            # Parse JSON output
            output = result.stdout.strip()
            if output:
                findings = json.loads(output)
            else:
                findings = []

            return {
                "linter": linter,
                "target": target,
                "findings": findings,
                "finding_count": len(findings) if isinstance(findings, list) else 0,
                "exit_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {"error": f"Linter timed out after 60 seconds"}
        except json.JSONDecodeError:
            return {
                "linter": linter,
                "raw_output": result.stdout[:2000],
                "exit_code": result.returncode,
            }

    def _search_codebase(
        self, pattern: str, file_type: str | None = None, max_results: int = 20
    ) -> dict:
        """Search the codebase using ripgrep."""
        cmd = ["rg", "--json", "--max-count", str(max_results)]

        if file_type:
            cmd.extend(["--type", file_type])

        cmd.extend([pattern, str(self.project_root)])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )

        matches = []
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
                if data.get("type") == "match":
                    match_data = data["data"]
                    matches.append({
                        "file": str(Path(match_data["path"]["text"]).relative_to(
                            self.project_root
                        )),
                        "line": match_data["line_number"],
                        "text": match_data["lines"]["text"].strip(),
                    })
            except (json.JSONDecodeError, KeyError):
                continue

        return {
            "pattern": pattern,
            "match_count": len(matches),
            "matches": matches[:max_results],
        }
```

### 2.3 Input Validation

Always validate inputs before execution:

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class ValidationResult:
    valid: bool
    error: str | None = None

def validate_tool_input(tool_name: str, input_data: dict[str, Any]) -> ValidationResult:
    """Validate tool inputs before execution."""

    if tool_name == "fetch_pr_diff":
        repo = input_data.get("repo", "")
        if not re.match(r"^[\w.-]+/[\w.-]+$", repo):
            return ValidationResult(False, f"Invalid repo format: '{repo}'")
        pr = input_data.get("pr_number", 0)
        if not isinstance(pr, int) or pr < 1:
            return ValidationResult(False, f"Invalid PR number: {pr}")

    elif tool_name == "run_linter":
        linter = input_data.get("linter", "")
        if linter not in ("ruff", "eslint", "shellcheck"):
            return ValidationResult(False, f"Unsupported linter: '{linter}'")
        target = input_data.get("target", "")
        if ".." in target:
            return ValidationResult(False, "Path traversal not allowed")

    elif tool_name == "search_codebase":
        pattern = input_data.get("pattern", "")
        if len(pattern) > 500:
            return ValidationResult(False, "Pattern too long (max 500 chars)")

    return ValidationResult(True)
```

---

## 3. System Prompt Engineering for Agents

The system prompt is the most important lever for shaping agent behavior. A well-crafted system prompt turns a general-purpose model into a focused, reliable agent.

### 3.1 System Prompt Structure

```python
SYSTEM_PROMPT_TEMPLATE = """
# Role
{role_definition}

# Capabilities
You have access to these tools:
{tool_descriptions}

# Constraints
{constraints}

# Output Format
{output_format}

# Safety Guidelines
{safety_guidelines}

# Examples
{examples}
"""
```

### 3.2 Role Definition

Be specific about who the agent is and what it does:

```python
# Weak role definition
role = "You are a helpful assistant."

# Strong role definition
role = """You are a senior code reviewer with 15 years of experience in Python
and TypeScript. Your reviews focus on four areas:

1. **Correctness**: Logic errors, off-by-one errors, race conditions, null handling
2. **Security**: Injection vulnerabilities, authentication bypasses, data exposure
3. **Performance**: Algorithmic complexity, unnecessary allocations, N+1 queries
4. **Maintainability**: Code clarity, naming, documentation, test coverage

You review code methodically: first understand the purpose, then analyze
the implementation, then check edge cases. You provide actionable feedback
with specific code suggestions, not vague recommendations."""
```

### 3.3 Capability Boundaries

Explicitly state what the agent should and should not do:

```python
constraints = """## Constraints

1. **Read-only access**: Do NOT modify any files. Use Read, Glob, and Grep only.
2. **No shell commands**: Do NOT use Bash except for running linters (ruff, eslint).
3. **Scope**: Only review files that are part of the PR diff. Do not review unrelated code.
4. **Time budget**: Complete the review within 20 turns. If the PR is too large,
   focus on the most critical files and note which files were skipped.
5. **Confidence**: If you are not sure about an issue, mark it as "info" severity
   and explain your uncertainty. Never fabricate issues.
6. **No approvals**: Never state that a PR is "approved" or "ready to merge."
   Always recommend human review of your findings."""
```

### 3.4 Output Format

Define exactly how the agent should structure its output:

```python
output_format = """## Output Format

Provide your review in this exact format:

### Summary
[1-2 sentences describing the overall quality and purpose of the PR]

### Findings

| # | Severity | File | Line | Issue | Suggestion |
|---|----------|------|------|-------|------------|
| 1 | critical | ... | ... | ... | ... |
| 2 | warning | ... | ... | ... | ... |

Severity levels:
- **critical**: Bugs, security vulnerabilities, data loss risks
- **warning**: Performance issues, code smells, missing error handling
- **suggestion**: Style improvements, documentation, minor refactoring
- **info**: Observations, questions, things to discuss

### Recommendations
[Prioritized list of actions the PR author should take]"""
```

### 3.5 Safety Guardrails

```python
safety = """## Safety Guidelines

1. Never access files outside the repository root directory.
2. Never execute arbitrary code or commands. Only use approved linters.
3. If you encounter credentials, API keys, or secrets in the code:
   - Flag them as CRITICAL findings
   - Do NOT include the actual secret values in your review output
   - Recommend immediate removal and key rotation
4. If you encounter code that could be used for malicious purposes
   (e.g., reverse shells, keyloggers), flag it but do not explain
   how to use it.
5. Do not make assumptions about the author's intent. If code is
   ambiguous, ask rather than assume."""
```

---

## 4. Practical Agent Examples

### 4.1 Code Review Agent

```python
# code_review_agent.py
import asyncio
import json
from claude_code_sdk import query, ClaudeCodeOptions

SYSTEM_PROMPT = """You are a senior code reviewer. Your task is to review
code changes in a pull request and provide structured feedback.

## Process
1. First, read the PR diff to understand the scope of changes.
2. For each changed file, analyze the code for:
   - Bugs and logic errors
   - Security vulnerabilities
   - Performance issues
   - Code style and readability
3. Run appropriate linters if available.
4. Compile findings into a structured review.

## Output Format
Respond with a JSON object:
{
    "summary": "1-2 sentence overview",
    "overall_quality": "excellent|good|acceptable|needs_work|critical",
    "findings": [
        {
            "severity": "critical|warning|suggestion|info",
            "file": "path/to/file.py",
            "line": 42,
            "category": "bug|security|performance|style|documentation",
            "message": "Description of the issue",
            "suggestion": "How to fix it",
            "code_snippet": "Optional: suggested code change"
        }
    ],
    "metrics": {
        "files_reviewed": 5,
        "total_findings": 12,
        "critical_count": 1,
        "warning_count": 3
    }
}

## Constraints
- Read-only: do not modify any files.
- Focus on the changed code, not the entire file.
- Be specific: include file paths and line numbers.
- Be actionable: every finding should include a suggestion."""


async def review_pr(repo: str, pr_number: int) -> dict:
    """Run a code review agent on a GitHub PR."""
    prompt = (
        f"Review pull request #{pr_number} in repository {repo}. "
        f"Start by running 'gh pr diff {pr_number} --repo {repo}' to get the diff, "
        f"then analyze the changes and provide your review as JSON."
    )

    full_output = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt=SYSTEM_PROMPT,
            max_turns=25,
            allowed_tools=["Read", "Glob", "Grep", "Bash"],
        ),
    ):
        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    full_output += block.text

    # Extract JSON from output
    try:
        json_match = full_output[full_output.index("{"):full_output.rindex("}") + 1]
        return json.loads(json_match)
    except (ValueError, json.JSONDecodeError):
        return {"error": "Failed to parse review output", "raw": full_output}


# Usage
if __name__ == "__main__":
    review = asyncio.run(review_pr("owner/repo", 123))
    print(json.dumps(review, indent=2))
```

### 4.2 Documentation Agent

```python
# doc_agent.py
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions

SYSTEM_PROMPT = """You are a technical documentation specialist. Your task is
to generate and maintain documentation from source code.

## Process
1. Read the source code files in the specified directory.
2. Analyze: module purpose, public API, classes, functions, parameters, return types.
3. Generate documentation in Markdown format.
4. Include: overview, installation, API reference, examples, changelog.

## Documentation Style
- Use clear, concise language. Avoid jargon.
- Include code examples for every public function.
- Document parameters with types and descriptions.
- Note any side effects or exceptions.
- Use standard markdown headings (##, ###).

## Constraints
- Only document public APIs (skip private/internal functions prefixed with _).
- Verify code examples are syntactically correct.
- If existing README exists, update it rather than overwriting.
- Preserve any manually-written sections in existing docs."""


async def generate_docs(project_dir: str, output_file: str = "API.md"):
    """Generate API documentation for a project."""
    prompt = (
        f"Generate comprehensive API documentation for the Python project "
        f"in {project_dir}. Read all .py files, analyze the public API, "
        f"and write the documentation to {output_file}. "
        f"Include code examples for key functions."
    )

    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt=SYSTEM_PROMPT,
            max_turns=30,
            cwd=project_dir,
        ),
    ):
        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text, end="")
        elif message.type == "tool_use":
            print(f"\n  [Tool: {message.tool_name}]")

    print(f"\nDocumentation written to {output_file}")


asyncio.run(generate_docs("/path/to/my-project"))
```

### 4.3 Database Migration Agent

```python
# migration_agent.py
import asyncio
import json
from claude_code_sdk import query, ClaudeCodeOptions

SYSTEM_PROMPT = """You are a database migration specialist. You analyze database
schemas and generate safe migration scripts.

## Process
1. Read the current schema from the provided database or SQL files.
2. Compare with the target schema (if provided) or analyze requested changes.
3. Generate migration SQL that is:
   - Safe: uses transactions, includes rollback
   - Incremental: one change per migration
   - Non-destructive: preserves existing data
   - Documented: includes comments explaining each change

## Migration Template
```sql
-- Migration: <description>
-- Created: <date>
-- Author: migration-agent

BEGIN;

-- Forward migration
<sql statements>

-- Verify migration
<verification queries>

COMMIT;

-- Rollback (run manually if needed)
-- BEGIN;
-- <rollback statements>
-- COMMIT;
```

## Safety Rules
1. NEVER generate DROP TABLE without explicit user confirmation.
2. NEVER generate DELETE or TRUNCATE statements.
3. Always add columns as NULLABLE first, then backfill, then add constraints.
4. Always include an estimated execution time for large tables.
5. Flag any migration that might lock tables for extended periods."""


async def generate_migration(
    schema_file: str,
    change_description: str,
    output_dir: str = "./migrations",
) -> str:
    """Generate a database migration script."""
    prompt = (
        f"Read the database schema from {schema_file}. "
        f"Generate a migration script for this change: {change_description}. "
        f"Save the migration to {output_dir}/ with a timestamped filename. "
        f"Also generate the rollback script."
    )

    result = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt=SYSTEM_PROMPT,
            max_turns=15,
        ),
    ):
        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    result += block.text
                    print(block.text, end="")

    return result


asyncio.run(generate_migration(
    "schema.sql",
    "Add a 'preferences' JSONB column to the users table with a default value of '{}'",
    "./migrations"
))
```

### 4.4 Customer Support Agent

```python
# support_agent.py
import asyncio
import json
from claude_code_sdk import query, ClaudeCodeOptions

SYSTEM_PROMPT = """You are a customer support agent for a software product.
You help users resolve technical issues by searching the knowledge base
and providing clear, step-by-step solutions.

## Process
1. Understand the customer's issue.
2. Search the knowledge base (docs/ directory) for relevant articles.
3. If a solution exists, provide step-by-step instructions.
4. If no solution exists, escalate with a summary of what was tried.

## Communication Style
- Be empathetic and professional.
- Use simple language (avoid technical jargon).
- Provide numbered steps for solutions.
- Include relevant links to documentation.
- Ask clarifying questions when the issue is ambiguous.

## Output Format
{
    "resolution_status": "resolved|escalated|needs_info",
    "response_to_customer": "...",
    "internal_notes": "...",
    "articles_referenced": ["path/to/article.md"],
    "suggested_kb_updates": ["Description of missing documentation"]
}

## Constraints
- Read-only access to the knowledge base.
- Do not make promises about timelines or compensation.
- For billing issues, always escalate to human support.
- For security concerns, flag as urgent and escalate immediately."""


async def handle_support_ticket(
    ticket_description: str,
    kb_directory: str = "./docs",
) -> dict:
    """Handle a customer support ticket."""
    prompt = (
        f"Handle this customer support ticket:\n\n"
        f'"{ticket_description}"\n\n'
        f"Search the knowledge base in {kb_directory}/ for relevant solutions. "
        f"Provide your response as a JSON object."
    )

    full_output = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt=SYSTEM_PROMPT,
            max_turns=15,
            allowed_tools=["Read", "Glob", "Grep"],
        ),
    ):
        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    full_output += block.text

    try:
        json_start = full_output.index("{")
        json_end = full_output.rindex("}") + 1
        return json.loads(full_output[json_start:json_end])
    except (ValueError, json.JSONDecodeError):
        return {
            "resolution_status": "escalated",
            "response_to_customer": "I was unable to process your request automatically. A human agent will follow up shortly.",
            "internal_notes": f"Agent failed to parse response. Raw output: {full_output[:500]}",
            "articles_referenced": [],
            "suggested_kb_updates": [],
        }


# Usage
ticket = "I can't log in to my account. I keep getting 'Invalid credentials' even though I just reset my password."
result = asyncio.run(handle_support_ticket(ticket))
print(json.dumps(result, indent=2))
```

---

## 5. Testing Agents

### 5.1 Unit Testing Individual Tools

Test tool executors independently of the agent:

```python
# tests/test_tools.py
import pytest
from executors import ToolExecutor

@pytest.fixture
def executor(tmp_path):
    """Create a tool executor with a temp project root."""
    # Create some test files
    (tmp_path / "main.py").write_text("def hello():\n    print('hello')\n")
    (tmp_path / "utils.py").write_text("# TODO: implement\ndef util():\n    pass\n")
    return ToolExecutor(project_root=str(tmp_path))


class TestSearchCodebase:
    def test_finds_matching_pattern(self, executor):
        result = executor.execute("search_codebase", {
            "pattern": "def hello",
            "file_type": "py",
        })
        assert result["match_count"] >= 1
        assert any("main.py" in m["file"] for m in result["matches"])

    def test_no_matches_returns_empty(self, executor):
        result = executor.execute("search_codebase", {
            "pattern": "nonexistent_function_xyz",
        })
        assert result["match_count"] == 0
        assert result["matches"] == []

    def test_respects_max_results(self, executor):
        result = executor.execute("search_codebase", {
            "pattern": "def",
            "max_results": 1,
        })
        assert len(result["matches"]) <= 1


class TestRunLinter:
    def test_lints_python_file(self, executor):
        result = executor.execute("run_linter", {
            "linter": "ruff",
            "target": "main.py",
        })
        assert "error" not in result
        assert result["linter"] == "ruff"

    def test_rejects_path_traversal(self, executor):
        result = executor.execute("run_linter", {
            "linter": "ruff",
            "target": "../../etc/passwd",
        })
        assert "error" in result
        assert "traversal" in result["error"].lower()

    def test_unsupported_linter(self, executor):
        result = executor.execute("run_linter", {
            "linter": "unknown_linter",
            "target": "main.py",
        })
        assert "error" in result


class TestValidation:
    def test_unknown_tool(self, executor):
        result = executor.execute("nonexistent_tool", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]
```

### 5.2 Integration Testing the Agent Loop

Test the full agent with controlled inputs:

```python
# tests/test_agent_integration.py
import asyncio
import json
import pytest
from claude_code_sdk import query, ClaudeCodeOptions


@pytest.fixture
def test_project(tmp_path):
    """Create a minimal test project."""
    (tmp_path / "hello.py").write_text(
        'def greet(name: str) -> str:\n    """Greet someone."""\n    return f"Hello, {name}!"\n'
    )
    (tmp_path / "README.md").write_text("# Test Project\nA simple test project.\n")
    return tmp_path


@pytest.mark.asyncio
async def test_agent_reads_file(test_project):
    """Test that the agent can read a file and summarize it."""
    output = ""
    async for message in query(
        prompt=f"Read {test_project}/hello.py and describe what the function does.",
        options=ClaudeCodeOptions(
            max_turns=5,
            allowed_tools=["Read"],
            cwd=str(test_project),
        ),
    ):
        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    output += block.text

    # The agent should mention the greet function
    assert "greet" in output.lower()
    assert "hello" in output.lower()


@pytest.mark.asyncio
async def test_agent_respects_tool_restrictions(test_project):
    """Test that the agent cannot use disallowed tools."""
    tool_names_used = []
    async for message in query(
        prompt=f"List files in {test_project} and read README.md.",
        options=ClaudeCodeOptions(
            max_turns=5,
            allowed_tools=["Read"],  # Glob not allowed
            cwd=str(test_project),
        ),
    ):
        if message.type == "tool_use":
            tool_names_used.append(message.tool_name)

    # Should only use Read, not Glob or Bash
    for tool in tool_names_used:
        assert tool == "Read"
```

### 5.3 Evaluation Metrics

Define metrics to measure agent quality:

```python
# evaluation.py
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentEvaluation:
    """Evaluate agent performance on a set of test cases."""

    test_case_id: str
    expected_output: dict             # What the agent should produce
    actual_output: dict               # What the agent actually produced
    turns_used: int = 0
    tools_used: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        """Compare actual vs expected output (simplified)."""
        if not self.expected_output or not self.actual_output:
            return 0.0

        matches = 0
        total = 0
        for key in self.expected_output:
            total += 1
            if key in self.actual_output:
                if self.expected_output[key] == self.actual_output[key]:
                    matches += 1
        return matches / total if total > 0 else 0.0

    @property
    def efficiency(self) -> float:
        """Score based on turns used (fewer is better)."""
        max_turns = 30
        return max(0, 1 - (self.turns_used / max_turns))


@dataclass
class EvaluationSuite:
    """Run and aggregate agent evaluations."""

    evaluations: list[AgentEvaluation] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.evaluations:
            return {"error": "No evaluations to summarize"}

        return {
            "total_tests": len(self.evaluations),
            "avg_accuracy": sum(e.accuracy for e in self.evaluations) / len(self.evaluations),
            "avg_efficiency": sum(e.efficiency for e in self.evaluations) / len(self.evaluations),
            "avg_turns": sum(e.turns_used for e in self.evaluations) / len(self.evaluations),
            "avg_time": sum(e.elapsed_seconds for e in self.evaluations) / len(self.evaluations),
            "most_used_tools": self._tool_frequency(),
        }

    def _tool_frequency(self) -> dict[str, int]:
        from collections import Counter
        all_tools = [t for e in self.evaluations for t in e.tools_used]
        return dict(Counter(all_tools).most_common(10))
```

---

## 6. Production Deployment

### 6.1 Rate Limiting

```python
import asyncio
import time
from dataclasses import dataclass


@dataclass
class RateLimiter:
    """Simple token bucket rate limiter."""
    max_requests_per_minute: int = 10
    _request_times: list = None

    def __post_init__(self):
        self._request_times = []

    async def acquire(self):
        """Wait until a request slot is available."""
        now = time.time()
        # Remove timestamps older than 1 minute
        self._request_times = [
            t for t in self._request_times if now - t < 60
        ]

        if len(self._request_times) >= self.max_requests_per_minute:
            # Wait until the oldest request expires
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self._request_times.append(time.time())


# Usage
limiter = RateLimiter(max_requests_per_minute=10)

async def rate_limited_agent_call(prompt: str):
    await limiter.acquire()
    async for message in query(prompt=prompt, options=ClaudeCodeOptions(max_turns=10)):
        yield message
```

### 6.2 Cost Monitoring

```python
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class CostTracker:
    """Track and enforce API cost budgets."""
    daily_budget_usd: float = 50.0
    monthly_budget_usd: float = 500.0
    log_file: str = "agent_costs.jsonl"
    _daily_spend: float = 0.0
    _monthly_spend: float = 0.0

    def record_usage(self, input_tokens: int, output_tokens: int, model: str):
        """Record token usage and check budget."""
        # Approximate cost calculation
        pricing = {
            "claude-opus-4-20250514": (15.00, 75.00),      # (input, output) per MTok
            "claude-sonnet-4-20250514": (3.00, 15.00),
            "claude-haiku-3-5-20241022": (0.80, 4.00),
        }

        rates = pricing.get(model, (3.00, 15.00))
        cost = (input_tokens / 1_000_000) * rates[0] + (output_tokens / 1_000_000) * rates[1]

        self._daily_spend += cost
        self._monthly_spend += cost

        # Log to file
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "daily_total": round(self._daily_spend, 4),
            "monthly_total": round(self._monthly_spend, 4),
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return cost

    def check_budget(self) -> bool:
        """Check if spending is within budget."""
        if self._daily_spend >= self.daily_budget_usd:
            raise BudgetExceededError(
                f"Daily budget exceeded: ${self._daily_spend:.2f} / ${self.daily_budget_usd:.2f}"
            )
        if self._monthly_spend >= self.monthly_budget_usd:
            raise BudgetExceededError(
                f"Monthly budget exceeded: ${self._monthly_spend:.2f} / ${self.monthly_budget_usd:.2f}"
            )
        return True


class BudgetExceededError(Exception):
    pass
```

### 6.3 Logging and Observability

```python
import logging
import json
from datetime import datetime

# Configure structured logging
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("agent")


class AgentLogger:
    """Structured logging for agent operations."""

    def __init__(self, agent_name: str, run_id: str):
        self.agent_name = agent_name
        self.run_id = run_id
        self.start_time = datetime.now()

    def _log(self, level: str, event: str, **kwargs):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent_name,
            "run_id": self.run_id,
            "event": event,
            "level": level,
            **kwargs,
        }
        logger.info(json.dumps(entry))

    def task_started(self, prompt: str):
        self._log("info", "task_started", prompt=prompt[:200])

    def tool_called(self, tool_name: str, duration_ms: float):
        self._log("info", "tool_called", tool=tool_name, duration_ms=round(duration_ms, 1))

    def turn_completed(self, turn: int, tokens_used: int):
        self._log("info", "turn_completed", turn=turn, tokens=tokens_used)

    def task_completed(self, success: bool, turns: int):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self._log(
            "info", "task_completed",
            success=success, turns=turns, elapsed_seconds=round(elapsed, 1),
        )

    def error(self, error_type: str, message: str):
        self._log("error", "error", error_type=error_type, message=message)
```

### 6.4 Security Considerations

```
┌─────────────────────────────────────────────────────────────────┐
│              Production Security Checklist                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Environment                                                     │
│  ☐ Run agents in sandboxed containers (Docker, Firecracker)     │
│  ☐ Use read-only filesystem mounts where possible                │
│  ☐ Network isolation: block outbound except allowed APIs         │
│  ☐ Separate credentials per agent (least privilege)              │
│                                                                  │
│  Input Validation                                                │
│  ☐ Sanitize all user inputs before passing to agent prompts     │
│  ☐ Limit prompt length to prevent injection attacks              │
│  ☐ Block prompts that attempt to override system instructions    │
│                                                                  │
│  Tool Safety                                                     │
│  ☐ Whitelist allowed tools explicitly                            │
│  ☐ Validate all tool inputs before execution                     │
│  ☐ Set timeouts on all tool executions                           │
│  ☐ Block file access outside designated directories              │
│  ☐ Restrict Bash to specific commands only                       │
│                                                                  │
│  Output Validation                                               │
│  ☐ Check agent outputs for sensitive data (PII, secrets)        │
│  ☐ Validate JSON outputs against expected schema                 │
│  ☐ Sanitize outputs before displaying to users                   │
│                                                                  │
│  Monitoring                                                      │
│  ☐ Log all tool invocations with inputs and outputs              │
│  ☐ Alert on unusual patterns (excessive tool calls, errors)      │
│  ☐ Track cost per agent run and enforce budgets                  │
│  ☐ Monitor for prompt injection attempts                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Scaling Patterns

### 7.1 Multi-Agent Systems

For complex tasks, decompose work across specialized agents:

```python
# multi_agent.py
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions


async def multi_agent_pipeline(task: str, project_dir: str):
    """Run a multi-agent pipeline: analyze → plan → execute → verify."""

    # Agent 1: Analyst
    print("=== Phase 1: Analysis ===")
    analysis = ""
    async for msg in query(
        prompt=f"Analyze this task and break it into subtasks: {task}",
        options=ClaudeCodeOptions(
            system_prompt="You are a technical analyst. Break complex tasks into clear subtasks.",
            max_turns=10,
            allowed_tools=["Read", "Glob", "Grep"],
            cwd=project_dir,
        ),
    ):
        if msg.type == "assistant":
            for block in msg.content:
                if hasattr(block, "text"):
                    analysis += block.text

    # Agent 2: Planner
    print("\n=== Phase 2: Planning ===")
    plan = ""
    async for msg in query(
        prompt=f"Given this analysis, create a detailed execution plan:\n\n{analysis}",
        options=ClaudeCodeOptions(
            system_prompt="You are a project planner. Create step-by-step execution plans.",
            max_turns=10,
            allowed_tools=["Read", "Glob"],
            cwd=project_dir,
        ),
    ):
        if msg.type == "assistant":
            for block in msg.content:
                if hasattr(block, "text"):
                    plan += block.text

    # Agent 3: Executor
    print("\n=== Phase 3: Execution ===")
    execution = ""
    async for msg in query(
        prompt=f"Execute this plan:\n\n{plan}",
        options=ClaudeCodeOptions(
            system_prompt="You are a software developer. Execute the plan precisely.",
            max_turns=30,
            cwd=project_dir,
        ),
    ):
        if msg.type == "assistant":
            for block in msg.content:
                if hasattr(block, "text"):
                    execution += block.text

    # Agent 4: Verifier
    print("\n=== Phase 4: Verification ===")
    async for msg in query(
        prompt=(
            f"Verify that these changes are correct and complete:\n\n"
            f"Original task: {task}\n"
            f"Plan: {plan}\n"
            f"Execution result: {execution[-2000:]}"
        ),
        options=ClaudeCodeOptions(
            system_prompt="You are a QA engineer. Verify changes thoroughly.",
            max_turns=15,
            allowed_tools=["Read", "Glob", "Grep", "Bash"],
            cwd=project_dir,
        ),
    ):
        if msg.type == "assistant":
            for block in msg.content:
                if hasattr(block, "text"):
                    print(block.text, end="")


asyncio.run(multi_agent_pipeline(
    task="Add input validation to all API endpoints",
    project_dir="/path/to/project"
))
```

### 7.2 Orchestration Patterns

```
┌─────────────────────────────────────────────────────────────────┐
│                Orchestration Patterns                             │
├──────────────────┬──────────────────────────────────────────────┤
│ Pattern          │ Description                                   │
├──────────────────┼──────────────────────────────────────────────┤
│ Sequential       │ A → B → C → D                                │
│ Pipeline         │ Each agent passes output to the next          │
│                  │ Use: analysis → plan → execute → verify       │
├──────────────────┼──────────────────────────────────────────────┤
│ Parallel         │ A ──┐                                         │
│ Fan-Out          │ B ──┼──▶ Merge                               │
│                  │ C ──┘                                         │
│                  │ Use: review multiple files simultaneously     │
├──────────────────┼──────────────────────────────────────────────┤
│ Router           │       ┌── Agent A (if condition X)           │
│                  │ Task ─┤                                       │
│                  │       └── Agent B (if condition Y)           │
│                  │ Use: route by task type (bug/feature/docs)   │
├──────────────────┼──────────────────────────────────────────────┤
│ Supervisor       │ Supervisor Agent                              │
│                  │ ├── Assigns tasks to worker agents           │
│                  │ ├── Monitors progress                         │
│                  │ └── Handles failures and reassignment        │
│                  │ Use: large projects with many subtasks       │
└──────────────────┴──────────────────────────────────────────────┘
```

### 7.3 Parallel Execution

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions


async def parallel_reviews(files: list[str], project_dir: str) -> list[str]:
    """Review multiple files in parallel using separate agent instances."""

    async def review_file(file_path: str) -> str:
        output = ""
        async for msg in query(
            prompt=f"Review the code in {file_path} for bugs and style issues.",
            options=ClaudeCodeOptions(
                max_turns=10,
                allowed_tools=["Read", "Grep"],
                cwd=project_dir,
            ),
        ):
            if msg.type == "assistant":
                for block in msg.content:
                    if hasattr(block, "text"):
                        output += block.text
        return output

    # Run all reviews concurrently
    results = await asyncio.gather(
        *[review_file(f) for f in files],
        return_exceptions=True,
    )

    return [
        r if isinstance(r, str) else f"Error: {r}"
        for r in results
    ]


files = ["src/auth.py", "src/database.py", "src/api.py"]
reviews = asyncio.run(parallel_reviews(files, "/path/to/project"))
for file, review in zip(files, reviews):
    print(f"\n--- {file} ---")
    print(review[:500])
```

---

## 8. Exercises

### Exercise 1: Simple Custom Agent (Beginner)

Build a "File Organizer" agent that:
1. Scans a directory for files.
2. Categorizes them by type (documents, images, code, data).
3. Suggests a directory structure.
4. Optionally moves files into the suggested structure.

Write the system prompt, define any custom tools needed, and implement the agent.

### Exercise 2: Tool Development (Intermediate)

Create a custom tool set for a "Project Health" agent:
1. `check_dependencies` -- reads package.json/requirements.txt and checks for outdated packages
2. `count_todos` -- searches the codebase for TODO/FIXME/HACK comments
3. `measure_complexity` -- uses radon (Python) or similar to measure code complexity
4. `check_test_coverage` -- runs tests with coverage and reports percentage

Implement the tool executors with proper validation and error handling. Write unit tests for each.

### Exercise 3: System Prompt Iteration (Intermediate)

Take the Code Review Agent from section 4.1 and improve its system prompt through iteration:
1. Run it on 5 different PRs (you can use public GitHub repos).
2. Analyze where the reviews are weak (false positives, missed issues, unclear feedback).
3. Revise the system prompt to address each weakness.
4. Re-run and measure improvement.
Document your prompt iterations and the reasoning behind each change.

### Exercise 4: Multi-Agent System (Advanced)

Build a multi-agent system for "Automated Codebase Modernization":
1. **Scanner Agent**: Identifies deprecated patterns, old syntax, missing type hints.
2. **Planner Agent**: Prioritizes changes and creates a migration plan.
3. **Modernizer Agent**: Applies changes to each file.
4. **Verifier Agent**: Runs tests and checks that nothing is broken.

Implement the full pipeline with error handling, progress tracking, and a final report.

### Exercise 5: Production-Ready Agent (Advanced)

Take any agent from section 4 and make it production-ready:
1. Add rate limiting and cost tracking.
2. Implement structured logging with correlation IDs.
3. Add input validation and output sanitization.
4. Write a comprehensive test suite (unit + integration + evaluation).
5. Containerize with Docker.
6. Create a simple REST API wrapper (FastAPI) so the agent can be called via HTTP.
7. Document the deployment process.

---

## 9. References

- Claude Code SDK Documentation - https://docs.anthropic.com/en/docs/claude-code/sdk
- Claude Code Agent Architecture - https://docs.anthropic.com/en/docs/claude-code/overview
- Tool Use Best Practices - https://docs.anthropic.com/en/docs/build-with-claude/tool-use/best-practices
- Anthropic Cookbook: Agents - https://github.com/anthropics/anthropic-cookbook
- Multi-Agent Systems - https://docs.anthropic.com/en/docs/build-with-claude/agentic
- Prompt Engineering for Agents - https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering

---

## Next Lesson

[19. Models, Pricing, and Optimization](./19_Models_and_Pricing.md) covers the practical economics of building with Claude: model selection, pricing tiers, prompt caching, the Batch API for cost savings, and strategies for optimizing token usage and cost in production applications.
