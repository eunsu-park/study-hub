# 커스텀 에이전트 구축

**이전**: [17. Claude Agent SDK](./17_Claude_Agent_SDK.md) | **다음**: [19. 모델, 가격 및 최적화](./19_Models_and_Pricing.md)

---

커스텀 에이전트를 구축한다는 것은 SDK를 그대로 사용하는 것을 넘어서는 일입니다. 특정 목적에 맞는 에이전트를 설계하고, 커스텀 도구를 장착하며, 에이전트의 동작을 제어하는 시스템 프롬프트를 작성하고, 프로덕션 환경에서 안정적으로 배포하는 것을 의미합니다. 이 레슨에서는 에이전트 설계부터 배포까지 전체 과정을 다룹니다. 네 가지 상세한 실용 예제와 함께 테스트, 모니터링, 스케일링에 대한 지침도 포함합니다.

**난이도**: ⭐⭐⭐

**사전 요구 사항**:
- 레슨 17의 Agent SDK 기초
- 레슨 16의 도구 사용 패턴
- Python 3.10+ 또는 TypeScript/Node.js 18+
- 비동기 프로그래밍 이해

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 명확한 범위, 적절한 도구, 안전 제약을 갖춘 에이전트 설계
2. JSON Schema 정의, 실행기(executor), 오류 처리를 포함한 커스텀 도구 구축
3. 에이전트 동작을 안정적으로 안내하는 효과적인 시스템 프롬프트 작성
4. 실제 사용 사례를 위한 4가지 완전한 에이전트 예제 구현
5. 단위, 통합, 평가 수준의 에이전트 테스트
6. 속도 제한(rate limiting), 비용 모니터링, 로깅, 보안을 갖춘 에이전트 배포
7. 오케스트레이션 패턴을 활용한 멀티 에이전트 시스템 설계

---

## 목차

1. [에이전트 설계](#1-에이전트-설계)
2. [커스텀 도구 개발](#2-커스텀-도구-개발)
3. [에이전트를 위한 시스템 프롬프트 엔지니어링](#3-에이전트를-위한-시스템-프롬프트-엔지니어링)
4. [실용적인 에이전트 예제](#4-실용적인-에이전트-예제)
5. [에이전트 테스트](#5-에이전트-테스트)
6. [프로덕션 배포](#6-프로덕션-배포)
7. [스케일링 패턴](#7-스케일링-패턴)
8. [연습 문제](#8-연습-문제)
9. [참고 자료](#9-참고-자료)

---

## 1. 에이전트 설계

코드를 작성하기 전에 다음 설계 질문들에 답해 보세요:

### 1.1 에이전트 설계 캔버스(Agent Design Canvas)

```
┌─────────────────────────────────────────────────────────────────┐
│                    에이전트 설계 캔버스                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 목적(PURPOSE)                                                │
│     이 에이전트는 어떤 문제를 해결하는가?                         │
│     예상 입력과 출력은 무엇인가?                                  │
│                                                                  │
│  2. 범위(SCOPE)                                                  │
│     에이전트가 할 수 있는 것은? (기능)                            │
│     에이전트가 할 수 없는 것은? (경계)                            │
│     에이전트가 거부해야 하는 것은? (안전)                         │
│                                                                  │
│  3. 도구(TOOLS)                                                  │
│     어떤 기본 제공 도구가 필요한가?                               │
│     어떤 커스텀 도구를 만들어야 하는가?                           │
│     어떤 외부 API 또는 서비스와 상호작용하는가?                   │
│                                                                  │
│  4. 제약(CONSTRAINTS)                                            │
│     최대 실행 시간은?                                             │
│     토큰/비용 예산은?                                             │
│     권한 경계는?                                                  │
│     외부 서비스의 속도 제한은?                                    │
│                                                                  │
│  5. 실패 모드(FAILURE MODES)                                     │
│     도구가 실패하면 어떻게 되는가?                                │
│     작업이 모호하면 어떻게 되는가?                                │
│     에이전트가 루프에 빠지면 어떻게 되는가?                       │
│     대체 동작(fallback)은 무엇인가?                               │
│                                                                  │
│  6. 평가(EVALUATION)                                             │
│     성공을 어떻게 측정하는가?                                     │
│     "좋은" 출력은 어떻게 생겼는가?                               │
│     엣지 케이스는 무엇인가?                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 설계 예시: 코드 리뷰 에이전트

```
목적:   풀 리퀘스트(pull request)에서 버그, 스타일 문제, 보안 우려 사항을 분석.
        심각도별로 분류된 구조화된 리뷰를 출력.

범위:
  가능:     코드 읽기, 린터 실행, 패턴 확인, 리뷰 댓글 생성
  불가능:   코드 푸시, PR 병합, 파일 수정
  거부:     사람 검토 없이 PR 승인, 개인 자격 증명 접근

도구:
  기본 제공: Read, Glob, Grep, Bash (린터 제한)
  커스텀:   fetch_pr_diff, post_review_comment, get_style_guide

제약:
  - 리뷰당 최대 30턴
  - 최대 5분 실행 시간
  - Bash 제한: ruff, mypy, eslint, shellcheck만 허용
  - 읽기 전용 파일 접근

실패 모드:
  - PR이 너무 큰 경우 (>5000 줄): 파일 그룹으로 분할
  - 린터 충돌 시: 린터 건너뛰고 리뷰에 기록
  - 모호한 코드: 사람 검토 표시, 추측하지 않음

평가:
  - 실제 버그 발견 비율 (재현율, recall)
  - 오탐지율 (정밀도, precision)
  - 리뷰 완료 시간
  - 사람 리뷰어 동의율
```

---

## 2. 커스텀 도구 개발

커스텀 도구는 기본 내장 기능을 넘어 에이전트를 확장합니다. 각 도구에는 스키마 정의, 실행기(executor), 오류 처리가 필요합니다.

### 2.1 도구 스키마 정의

Messages API 도구 형식(JSON Schema)을 사용하여 도구를 정의하세요:

```python
# tools.py — 커스텀 도구 정의

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

### 2.2 도구 실행기 구현

```python
# executors.py — 도구 실행 로직

import json
import subprocess
import re
from pathlib import Path
from typing import Any


class ToolExecutor:
    """검증 및 오류 처리를 포함한 커스텀 도구 실행기."""

    def __init__(self, project_root: str, github_token: str | None = None):
        self.project_root = Path(project_root)
        self.github_token = github_token

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict:
        """도구 호출을 적절한 핸들러로 라우팅."""
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
        """GitHub에서 PR diff 가져오기."""
        # 입력 값 검증
        if not re.match(r"^[\w.-]+/[\w.-]+$", repo):
            return {"error": f"Invalid repo format: {repo}. Use 'owner/name'."}

        cmd = ["gh", "pr", "diff", str(pr_number), "--repo", repo]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            return {"error": f"GitHub CLI error: {result.stderr.strip()}"}

        diff = result.stdout

        # 파일 필터 적용 (제공된 경우)
        if file_filter:
            import fnmatch
            filtered_lines = []
            include_file = False
            for line in diff.splitlines():
                if line.startswith("diff --git"):
                    # 파일명 추출
                    parts = line.split(" b/")
                    if len(parts) > 1:
                        filename = parts[1]
                        include_file = fnmatch.fnmatch(filename, file_filter)
                if include_file:
                    filtered_lines.append(line)
            diff = "\n".join(filtered_lines)

        # 매우 큰 diff 자르기
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
        """린터를 실행하고 구조화된 결과 반환."""
        # 대상 경로 검증 (디렉토리 순회 방지)
        target_path = (self.project_root / target).resolve()
        if not str(target_path).startswith(str(self.project_root.resolve())):
            return {"error": "Path traversal detected. Target must be within project root."}

        if not target_path.exists():
            return {"error": f"Target not found: {target}"}

        # 린터 명령 구성
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

            # JSON 출력 파싱
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
        """ripgrep을 사용하여 코드베이스 검색."""
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

### 2.3 입력 검증

실행 전에 항상 입력 값을 검증하세요:

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class ValidationResult:
    valid: bool
    error: str | None = None

def validate_tool_input(tool_name: str, input_data: dict[str, Any]) -> ValidationResult:
    """실행 전 도구 입력 값 검증."""

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

## 3. 에이전트를 위한 시스템 프롬프트 엔지니어링

시스템 프롬프트는 에이전트 동작을 형성하는 가장 중요한 요소입니다. 잘 작성된 시스템 프롬프트는 범용 모델을 집중적이고 신뢰할 수 있는 에이전트로 변환합니다.

### 3.1 시스템 프롬프트 구조

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

### 3.2 역할 정의

에이전트가 누구이고 무엇을 하는지 구체적으로 기술하세요:

```python
# 약한 역할 정의
role = "You are a helpful assistant."

# 강한 역할 정의
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

### 3.3 기능 경계

에이전트가 해야 할 것과 하지 말아야 할 것을 명시적으로 기술하세요:

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

### 3.4 출력 형식

에이전트가 출력을 어떻게 구조화해야 하는지 정확하게 정의하세요:

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

### 3.5 안전 가드레일(Safety Guardrails)

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

## 4. 실용적인 에이전트 예제

### 4.1 코드 리뷰 에이전트

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
    """GitHub PR에서 코드 리뷰 에이전트를 실행."""
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

    # 출력에서 JSON 추출
    try:
        json_match = full_output[full_output.index("{"):full_output.rindex("}") + 1]
        return json.loads(json_match)
    except (ValueError, json.JSONDecodeError):
        return {"error": "Failed to parse review output", "raw": full_output}


# 사용 예
if __name__ == "__main__":
    review = asyncio.run(review_pr("owner/repo", 123))
    print(json.dumps(review, indent=2))
```

### 4.2 문서화 에이전트

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
    """프로젝트의 API 문서 생성."""
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

### 4.3 데이터베이스 마이그레이션 에이전트

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
    """데이터베이스 마이그레이션 스크립트 생성."""
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

### 4.4 고객 지원 에이전트

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
    """고객 지원 티켓 처리."""
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


# 사용 예
ticket = "I can't log in to my account. I keep getting 'Invalid credentials' even though I just reset my password."
result = asyncio.run(handle_support_ticket(ticket))
print(json.dumps(result, indent=2))
```

---

## 5. 에이전트 테스트

### 5.1 개별 도구 단위 테스트

에이전트와 독립적으로 도구 실행기를 테스트하세요:

```python
# tests/test_tools.py
import pytest
from executors import ToolExecutor

@pytest.fixture
def executor(tmp_path):
    """임시 프로젝트 루트로 도구 실행기 생성."""
    # 테스트 파일 생성
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

### 5.2 에이전트 루프 통합 테스트

제어된 입력으로 전체 에이전트를 테스트하세요:

```python
# tests/test_agent_integration.py
import asyncio
import json
import pytest
from claude_code_sdk import query, ClaudeCodeOptions


@pytest.fixture
def test_project(tmp_path):
    """최소한의 테스트 프로젝트 생성."""
    (tmp_path / "hello.py").write_text(
        'def greet(name: str) -> str:\n    """Greet someone."""\n    return f"Hello, {name}!"\n'
    )
    (tmp_path / "README.md").write_text("# Test Project\nA simple test project.\n")
    return tmp_path


@pytest.mark.asyncio
async def test_agent_reads_file(test_project):
    """에이전트가 파일을 읽고 요약할 수 있는지 테스트."""
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

    # 에이전트가 greet 함수를 언급해야 함
    assert "greet" in output.lower()
    assert "hello" in output.lower()


@pytest.mark.asyncio
async def test_agent_respects_tool_restrictions(test_project):
    """에이전트가 허용되지 않은 도구를 사용할 수 없는지 테스트."""
    tool_names_used = []
    async for message in query(
        prompt=f"List files in {test_project} and read README.md.",
        options=ClaudeCodeOptions(
            max_turns=5,
            allowed_tools=["Read"],  # Glob 허용 안 됨
            cwd=str(test_project),
        ),
    ):
        if message.type == "tool_use":
            tool_names_used.append(message.tool_name)

    # Read만 사용해야 하고, Glob이나 Bash는 사용하지 않아야 함
    for tool in tool_names_used:
        assert tool == "Read"
```

### 5.3 평가 지표

에이전트 품질을 측정하는 지표를 정의하세요:

```python
# evaluation.py
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentEvaluation:
    """테스트 케이스 집합에 대한 에이전트 성능 평가."""

    test_case_id: str
    expected_output: dict             # 에이전트가 생성해야 하는 결과
    actual_output: dict               # 에이전트가 실제로 생성한 결과
    turns_used: int = 0
    tools_used: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        """실제 출력 대 예상 출력 비교 (단순화)."""
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
        """사용된 턴 수 기반 점수 (적을수록 좋음)."""
        max_turns = 30
        return max(0, 1 - (self.turns_used / max_turns))


@dataclass
class EvaluationSuite:
    """에이전트 평가 실행 및 집계."""

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

## 6. 프로덕션 배포

### 6.1 속도 제한(Rate Limiting)

```python
import asyncio
import time
from dataclasses import dataclass


@dataclass
class RateLimiter:
    """단순 토큰 버킷 속도 제한기."""
    max_requests_per_minute: int = 10
    _request_times: list = None

    def __post_init__(self):
        self._request_times = []

    async def acquire(self):
        """요청 슬롯이 사용 가능할 때까지 대기."""
        now = time.time()
        # 1분보다 오래된 타임스탬프 제거
        self._request_times = [
            t for t in self._request_times if now - t < 60
        ]

        if len(self._request_times) >= self.max_requests_per_minute:
            # 가장 오래된 요청이 만료될 때까지 대기
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self._request_times.append(time.time())


# 사용 예
limiter = RateLimiter(max_requests_per_minute=10)

async def rate_limited_agent_call(prompt: str):
    await limiter.acquire()
    async for message in query(prompt=prompt, options=ClaudeCodeOptions(max_turns=10)):
        yield message
```

### 6.2 비용 모니터링

```python
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class CostTracker:
    """API 비용 예산 추적 및 강제."""
    daily_budget_usd: float = 50.0
    monthly_budget_usd: float = 500.0
    log_file: str = "agent_costs.jsonl"
    _daily_spend: float = 0.0
    _monthly_spend: float = 0.0

    def record_usage(self, input_tokens: int, output_tokens: int, model: str):
        """토큰 사용량 기록 및 예산 확인."""
        # 근사 비용 계산
        pricing = {
            "claude-opus-4-20250514": (15.00, 75.00),      # (입력, 출력) MTok당
            "claude-sonnet-4-20250514": (3.00, 15.00),
            "claude-haiku-3-5-20241022": (0.80, 4.00),
        }

        rates = pricing.get(model, (3.00, 15.00))
        cost = (input_tokens / 1_000_000) * rates[0] + (output_tokens / 1_000_000) * rates[1]

        self._daily_spend += cost
        self._monthly_spend += cost

        # 파일에 기록
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
        """지출이 예산 내에 있는지 확인."""
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

### 6.3 로깅 및 가시성(Observability)

```python
import logging
import json
from datetime import datetime

# 구조화된 로깅 설정
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("agent")


class AgentLogger:
    """에이전트 작업에 대한 구조화된 로깅."""

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

### 6.4 보안 고려 사항

```
┌─────────────────────────────────────────────────────────────────┐
│              프로덕션 보안 체크리스트                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  환경                                                            │
│  ☐ 샌드박스 컨테이너에서 에이전트 실행 (Docker, Firecracker)    │
│  ☐ 가능한 경우 읽기 전용 파일시스템 마운트 사용                  │
│  ☐ 네트워크 격리: 허용된 API를 제외한 아웃바운드 차단            │
│  ☐ 에이전트별 자격 증명 분리 (최소 권한 원칙)                    │
│                                                                  │
│  입력 검증                                                       │
│  ☐ 에이전트 프롬프트에 전달하기 전 모든 사용자 입력 위생 처리   │
│  ☐ 인젝션 공격 방지를 위한 프롬프트 길이 제한                    │
│  ☐ 시스템 지침을 우회하려는 프롬프트 차단                        │
│                                                                  │
│  도구 안전                                                       │
│  ☐ 허용된 도구 명시적 화이트리스트 등록                          │
│  ☐ 실행 전 모든 도구 입력 검증                                   │
│  ☐ 모든 도구 실행에 타임아웃 설정                                │
│  ☐ 지정된 디렉토리 외부 파일 접근 차단                           │
│  ☐ Bash를 특정 명령어만으로 제한                                 │
│                                                                  │
│  출력 검증                                                       │
│  ☐ 에이전트 출력에서 민감한 데이터 확인 (PII, 비밀)             │
│  ☐ 예상 스키마에 대해 JSON 출력 검증                             │
│  ☐ 사용자에게 표시하기 전 출력 위생 처리                         │
│                                                                  │
│  모니터링                                                        │
│  ☐ 입력 및 출력과 함께 모든 도구 호출 기록                       │
│  ☐ 비정상적인 패턴(과도한 도구 호출, 오류)에 대한 경보           │
│  ☐ 에이전트 실행당 비용 추적 및 예산 강제                        │
│  ☐ 프롬프트 인젝션 시도 모니터링                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 스케일링 패턴

### 7.1 멀티 에이전트 시스템

복잡한 작업의 경우, 전문화된 에이전트에 걸쳐 작업을 분해하세요:

```python
# multi_agent.py
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions


async def multi_agent_pipeline(task: str, project_dir: str):
    """멀티 에이전트 파이프라인 실행: 분석 → 계획 → 실행 → 검증."""

    # 에이전트 1: 분석가
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

    # 에이전트 2: 기획자
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

    # 에이전트 3: 실행자
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

    # 에이전트 4: 검증자
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

### 7.2 오케스트레이션 패턴

```
┌─────────────────────────────────────────────────────────────────┐
│                오케스트레이션 패턴                                │
├──────────────────┬──────────────────────────────────────────────┤
│ 패턴             │ 설명                                          │
├──────────────────┼──────────────────────────────────────────────┤
│ 순차(Sequential) │ A → B → C → D                               │
│ 파이프라인       │ 각 에이전트가 출력을 다음으로 전달            │
│                  │ 사용: 분석 → 계획 → 실행 → 검증              │
├──────────────────┼──────────────────────────────────────────────┤
│ 병렬             │ A ──┐                                         │
│ 팬아웃(Fan-Out)  │ B ──┼──▶ 병합                               │
│                  │ C ──┘                                         │
│                  │ 사용: 여러 파일 동시 검토                    │
├──────────────────┼──────────────────────────────────────────────┤
│ 라우터(Router)   │       ┌── 에이전트 A (조건 X인 경우)        │
│                  │ 작업 ─┤                                       │
│                  │       └── 에이전트 B (조건 Y인 경우)        │
│                  │ 사용: 작업 유형별 라우팅 (버그/기능/문서)   │
├──────────────────┼──────────────────────────────────────────────┤
│ 슈퍼바이저       │ 슈퍼바이저 에이전트                           │
│                  │ ├── 워커 에이전트에 작업 할당                │
│                  │ ├── 진행 상황 모니터링                        │
│                  │ └── 실패 처리 및 재할당                      │
│                  │ 사용: 많은 하위 작업이 있는 대규모 프로젝트 │
└──────────────────┴──────────────────────────────────────────────┘
```

### 7.3 병렬 실행

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions


async def parallel_reviews(files: list[str], project_dir: str) -> list[str]:
    """별도의 에이전트 인스턴스를 사용하여 여러 파일을 병렬로 검토."""

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

    # 모든 검토를 동시에 실행
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

## 8. 연습 문제

### 연습 1: 간단한 커스텀 에이전트 (초급)

다음을 수행하는 "파일 정리기" 에이전트를 구축하세요:
1. 디렉토리에서 파일을 스캔합니다.
2. 유형별로 분류합니다 (문서, 이미지, 코드, 데이터).
3. 디렉토리 구조를 제안합니다.
4. 선택적으로 파일을 제안된 구조로 이동합니다.

시스템 프롬프트를 작성하고, 필요한 커스텀 도구를 정의하고, 에이전트를 구현하세요.

### 연습 2: 도구 개발 (중급)

"프로젝트 상태 확인" 에이전트를 위한 커스텀 도구 세트를 만드세요:
1. `check_dependencies` -- package.json/requirements.txt를 읽고 오래된 패키지 확인
2. `count_todos` -- 코드베이스에서 TODO/FIXME/HACK 주석 검색
3. `measure_complexity` -- radon(Python) 또는 유사 도구로 코드 복잡도 측정
4. `check_test_coverage` -- 커버리지로 테스트 실행 및 백분율 보고

적절한 검증 및 오류 처리와 함께 도구 실행기를 구현하고, 각 도구에 대한 단위 테스트를 작성하세요.

### 연습 3: 시스템 프롬프트 반복 (중급)

섹션 4.1의 코드 리뷰 에이전트를 가져와서 반복을 통해 시스템 프롬프트를 개선하세요:
1. 5개의 서로 다른 PR에서 실행합니다 (공개 GitHub 저장소 사용 가능).
2. 리뷰가 약한 부분을 분석합니다 (오탐, 놓친 문제, 불명확한 피드백).
3. 각 약점을 해결하기 위해 시스템 프롬프트를 수정합니다.
4. 재실행하고 개선 사항을 측정합니다.
프롬프트 반복과 각 변경의 이유를 문서화하세요.

### 연습 4: 멀티 에이전트 시스템 (고급)

"자동화된 코드베이스 현대화"를 위한 멀티 에이전트 시스템을 구축하세요:
1. **스캐너 에이전트**: 더 이상 사용되지 않는 패턴, 구식 구문, 누락된 타입 힌트 식별.
2. **플래너 에이전트**: 변경 사항 우선순위 지정 및 마이그레이션 계획 작성.
3. **현대화 에이전트**: 각 파일에 변경 사항 적용.
4. **검증 에이전트**: 테스트 실행 및 아무것도 손상되지 않았는지 확인.

오류 처리, 진행 상황 추적, 최종 보고서를 포함한 전체 파이프라인을 구현하세요.

### 연습 5: 프로덕션 준비 에이전트 (고급)

섹션 4의 에이전트 중 하나를 가져와서 프로덕션 준비 상태로 만드세요:
1. 속도 제한과 비용 추적을 추가합니다.
2. 상관 관계 ID(correlation ID)를 포함한 구조화된 로깅을 구현합니다.
3. 입력 검증과 출력 위생 처리를 추가합니다.
4. 포괄적인 테스트 스위트(단위 + 통합 + 평가)를 작성합니다.
5. Docker로 컨테이너화합니다.
6. 에이전트를 HTTP를 통해 호출할 수 있도록 간단한 REST API 래퍼(FastAPI)를 만듭니다.
7. 배포 프로세스를 문서화합니다.

---

## 9. 참고 자료

- Claude Code SDK 문서 - https://docs.anthropic.com/en/docs/claude-code/sdk
- Claude Code 에이전트 아키텍처 - https://docs.anthropic.com/en/docs/claude-code/overview
- 도구 사용 모범 사례 - https://docs.anthropic.com/en/docs/build-with-claude/tool-use/best-practices
- Anthropic Cookbook: 에이전트 - https://github.com/anthropics/anthropic-cookbook
- 멀티 에이전트 시스템 - https://docs.anthropic.com/en/docs/build-with-claude/agentic
- 에이전트를 위한 프롬프트 엔지니어링 - https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering

---

## 다음 레슨

[19. 모델, 가격 및 최적화](./19_Models_and_Pricing.md)에서는 Claude로 구축하는 실용적인 경제성을 다룹니다: 모델 선택, 가격 계층, 프롬프트 캐싱, 비용 절감을 위한 배치 API(Batch API), 그리고 프로덕션 애플리케이션에서 토큰 사용량과 비용을 최적화하는 전략을 설명합니다.
