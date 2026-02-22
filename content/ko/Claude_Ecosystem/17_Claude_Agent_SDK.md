# Claude Agent SDK

**이전**: [16. 도구 사용과 함수 호출](./16_Tool_Use_and_Function_Calling.md) | **다음**: [18. 커스텀 에이전트 구축](./18_Building_Custom_Agents.md)

---

Claude Agent SDK(`claude-code-sdk`)는 Claude Code CLI를 구동하는 것과 동일한 에이전트 기능에 프로그래밍 방식으로 접근할 수 있게 해줍니다. 레슨 16에서 수동으로 구축한 저수준 도구 사용 루프 대신, Agent SDK는 고수준 인터페이스를 제공합니다: 작업을 정의하고, 에이전트의 도구와 권한을 구성하고, 에이전트 루프가 나머지를 처리하도록 합니다 -- 생각하고, 행동하고, 관찰하고, 작업이 완료될 때까지 반복합니다. 이 레슨에서는 SDK의 아키텍처, 핵심 개념, 구성, 그리고 실용적인 사용 패턴을 다룹니다.

**난이도**: ⭐⭐⭐

**사전 요구 사항**:
- 레슨 16에서의 도구 사용에 대한 이해
- Python 3.10+ 또는 Node.js 18+
- Claude Code CLI 설치 (SDK가 이에 의존함)
- 비동기 프로그래밍에 대한 친숙함 (Python asyncio 또는 TypeScript async/await)

**학습 목표**:
- Agent SDK의 아키텍처와 Claude Code와의 관계 이해하기
- Python과 TypeScript에서 SDK 설치 및 구성하기
- 커스텀 시스템 프롬프트와 도구 구성으로 에이전트 만들기
- 에이전트 실행에서 스트리밍 이벤트 처리하기
- 모델 선택, 턴 제한, 권한 설정 구성하기
- SDK 기반 에이전트와 MCP 서버 통합하기
- 에이전트 워크플로우에서 오류, 재시도, 엣지 케이스 처리하기

---

## 목차

1. [Agent SDK란?](#1-agent-sdk란)
2. [아키텍처](#2-아키텍처)
3. [설치 및 설정](#3-설치-및-설정)
4. [핵심 개념](#4-핵심-개념)
5. [에이전트 만들기 (Python)](#5-에이전트-만들기-python)
6. [에이전트 만들기 (TypeScript)](#6-에이전트-만들기-typescript)
7. [구성 옵션](#7-구성-옵션)
8. [에이전트 응답 다루기](#8-에이전트-응답-다루기)
9. [오류 처리 및 재시도](#9-오류-처리-및-재시도)
10. [SDK 컨텍스트에서의 훅(Hook)](#10-sdk-컨텍스트에서의-훅hook)
11. [MCP 서버 통합](#11-mcp-서버-통합)
12. [연습 문제](#12-연습-문제)
13. [참고 자료](#13-참고-자료)

---

## 1. Agent SDK란?

Agent SDK는 Claude Code 에이전트 루프를 프로그래밍 가능한 인터페이스로 감싸는 라이브러리입니다. Claude Code CLI와 동일한 기능 -- 파일 읽기, 코드 작성, 명령 실행, 웹 검색 -- 을 활용하는 애플리케이션을 구축할 수 있지만, 여러분 자신의 코드로 제어됩니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                   추상화 수준                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  수준 3: Claude Code CLI                                         │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  대화형 터미널 인터페이스                            │        │
│  │  사람이 루프에 참여 (권한 프롬프트)                 │        │
│  │  세션 관리, 대화 기록                                │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  수준 2: Claude Agent SDK    ◀── 이 레슨                        │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  프로그래밍 방식의 에이전트 제어                     │        │
│  │  내장 도구 (Read, Write, Bash, Glob 등)             │        │
│  │  자동 에이전트 루프 (생각 → 행동 → 관찰)            │        │
│  │  컨텍스트 창 관리 및 압축                            │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  수준 1: Claude API (Messages + 도구 사용)                       │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  수동 도구 사용 루프를 포함한 원시 API 호출          │        │
│  │  대화 상태, 도구, 재시도를 직접 관리                 │        │
│  │  최대 제어, 최대 복잡성                               │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  수준 0: HTTP / REST                                             │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  api.anthropic.com에 대한 원시 HTTP 요청             │        │
│  │  SDK 없음, 수동 JSON 구성                            │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**원시 API와의 주요 차이점:**
- **내장 도구**: 파일 작업(Read, Write, Edit, Glob, Grep), 셸(Bash), 웹(WebFetch, WebSearch)이 자동으로 포함됩니다.
- **에이전트 루프**: SDK가 생각-행동-관찰 사이클을 자동으로 처리합니다. `while stop_reason == "tool_use"` 루프를 작성할 필요가 없습니다.
- **컨텍스트 관리**: SDK가 컨텍스트 창 한계를 처리하고, 창이 꽉 찰 때 오래된 메시지를 압축합니다.
- **권한 모델**: 에이전트가 사용할 수 있는 도구와 접근할 수 있는 내용을 제어하는 구성 가능한 권한.

---

## 2. 아키텍처

Agent SDK는 Claude Code CLI 프로세스와 통신하고, 이것은 다시 Claude API를 호출합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                   SDK 아키텍처                                   │
│                                                                   │
│  여러분의 애플리케이션                                            │
│  ┌──────────────────┐                                            │
│  │  Python / TS     │                                            │
│  │  애플리케이션    │                                            │
│  │  코드            │                                            │
│  └────────┬─────────┘                                            │
│           │  SDK API 호출                                         │
│           ▼                                                       │
│  ┌──────────────────┐                                            │
│  │  claude-code-sdk │  Agent SDK 라이브러리                       │
│  │  ┌────────────┐  │                                            │
│  │  │ 에이전트   │  │  생각 → 행동 → 관찰 → 반복               │
│  │  │ 루프       │  │                                            │
│  │  └────────────┘  │                                            │
│  │  ┌────────────┐  │                                            │
│  │  │ 도구 관리자│  │  내장 + 커스텀 도구                        │
│  │  └────────────┘  │                                            │
│  │  ┌────────────┐  │                                            │
│  │  │ 컨텍스트   │  │  창 관리, 압축                             │
│  │  │ 관리자     │  │                                            │
│  │  └────────────┘  │                                            │
│  └────────┬─────────┘                                            │
│           │  서브프로세스 통신                                     │
│           ▼                                                       │
│  ┌──────────────────┐                                            │
│  │  Claude Code CLI │  claude 바이너리                            │
│  └────────┬─────────┘                                            │
│           │  HTTPS                                                │
│           ▼                                                       │
│  ┌──────────────────┐                                            │
│  │  Claude API      │  api.anthropic.com                         │
│  └──────────────────┘                                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

에이전트 루프는 고정된 사이클을 따릅니다:

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  생각    │────▶│  행동    │────▶│  관찰    │──┐
│          │     │          │     │          │  │
│  작업을  │     │  도구    │     │  도구    │  │
│  분석하고│     │  사용    │     │  결과    │  │
│  다음    │     │  (Read,  │     │  처리    │  │
│  단계 계획│     │  Write,  │     │          │  │
│          │     │  Bash,   │     │          │  │
│          │     │  등)     │     │          │  │
└──────────┘     └──────────┘     └────┬─────┘  │
     ▲                                 │        │
     │                                 │        │
     │     ┌──────────────┐            │        │
     │     │   완료       │◀───────────┘        │
     │     │              │  (작업 완료 시)     │
     │     │  최종 결과   │                     │
     │     │  반환        │                     │
     │     └──────────────┘                     │
     │                                          │
     └──────────────────────────────────────────┘
              (더 많은 작업이 필요한 경우)
```

---

## 3. 설치 및 설정

### 3.1 사전 요구 사항

Agent SDK는 Claude Code CLI가 설치되어 있어야 합니다:

```bash
# Claude Code CLI 설치 (아직 설치되지 않은 경우)
npm install -g @anthropic-ai/claude-code

# 설치 확인
claude --version
```

### 3.2 Python SDK

```bash
# Python SDK 설치
pip install claude-code-sdk

# 또는 uv 사용
uv add claude-code-sdk
```

### 3.3 TypeScript SDK

```bash
# TypeScript SDK 설치
npm install @anthropic-ai/claude-code-sdk

# 또는 선호하는 패키지 관리자 사용
pnpm add @anthropic-ai/claude-code-sdk
yarn add @anthropic-ai/claude-code-sdk
```

### 3.4 인증

SDK는 Claude Code CLI와 동일한 인증을 사용합니다. 인증되었는지 확인하십시오:

```bash
# 옵션 1: 대화형으로 로그인
claude login

# 옵션 2: API 키 직접 설정
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## 4. 핵심 개념

### 4.1 에이전트 루프

에이전트 루프는 핵심 실행 모델입니다. 작업(자연어 프롬프트)을 제공하면, 에이전트는 작업이 완료되거나 한계에 도달할 때까지 생각-행동-관찰 사이클을 반복합니다.

각 반복:
1. **생각(Think)**: Claude가 현재 상태를 분석하고 다음에 할 일을 결정합니다.
2. **행동(Act)**: Claude가 도구를 호출합니다 (파일 읽기, 명령 실행, 검색 등).
3. **관찰(Observe)**: 도구 결과가 Claude에게 분석을 위해 반환됩니다.
4. **결정(Decide)**: Claude가 다른 작업을 수행하거나 작업이 완료되었음을 선언합니다.

### 4.2 내장 도구

Agent SDK에는 Claude Code의 모든 내장 도구가 포함됩니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                   내장 도구                                      │
├──────────────────┬──────────────────────────────────────────────┤
│ 카테고리         │ 도구                                          │
├──────────────────┼──────────────────────────────────────────────┤
│ 파일 시스템      │ Read     - 파일 내용 읽기                    │
│                  │ Write    - 파일 생성 또는 덮어쓰기            │
│                  │ Edit     - 파일에 대한 타겟 편집 수행         │
│                  │ Glob     - 패턴으로 파일 찾기                 │
│                  │ Grep     - 파일 내용 검색 (ripgrep)           │
├──────────────────┼──────────────────────────────────────────────┤
│ 셸               │ Bash     - 셸 명령 실행                       │
├──────────────────┼──────────────────────────────────────────────┤
│ 웹               │ WebFetch - 웹 페이지 가져오기 및 파싱         │
│                  │ WebSearch - 웹 검색                           │
├──────────────────┼──────────────────────────────────────────────┤
│ Jupyter          │ NotebookEdit - Jupyter 노트북 셀 편집         │
├──────────────────┼──────────────────────────────────────────────┤
│ MCP              │ 연결된 MCP 서버가 노출한 모든 도구            │
└──────────────────┴──────────────────────────────────────────────┘
```

### 4.3 컨텍스트 창 관리

SDK는 컨텍스트 창을 자동으로 관리합니다:
- 대화가 커짐에 따라, 오래된 메시지가 새 콘텐츠를 위한 공간을 만들기 위해 **압축(compacted)**(요약)됩니다.
- 이전에 읽은 파일 내용은 더 이상 즉시 관련성이 없는 경우 요약될 수 있습니다.
- 시스템 프롬프트와 최근 메시지는 항상 전체 형태로 유지됩니다.

### 4.4 스트리밍 이벤트

에이전트 실행은 애플리케이션이 처리할 수 있는 이벤트 스트림을 생성합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                   이벤트 유형                                    │
├──────────────────┬──────────────────────────────────────────────┤
│ 이벤트           │ 설명                                          │
├──────────────────┼──────────────────────────────────────────────┤
│ assistant        │ 에이전트의 텍스트 출력 (생각/답변)            │
│ tool_use         │ 에이전트가 도구를 호출하는 중                 │
│ tool_result      │ 도구 실행 결과                                │
│ result           │ 에이전트 작업의 최종 결과                     │
│ error            │ 실행 중 오류                                  │
└──────────────────┴──────────────────────────────────────────────┘
```

---

## 5. 에이전트 만들기 (Python)

### 5.1 기본 사용법

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions, Message

async def main():
    # 간단한 단발성 작업
    messages: list[Message] = []

    async for message in query(
        prompt="Read the file README.md and summarize its contents.",
        options=ClaudeCodeOptions(
            max_turns=10,  # 최대 에이전트 루프 반복 횟수
        ),
    ):
        if message.type == "assistant":
            # 에이전트의 텍스트 출력 (생각, 설명, 최종 답변)
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text, end="")
        elif message.type == "tool_use":
            # 에이전트가 도구를 사용하는 중
            print(f"\n  [Tool: {message.tool_name}]")
        elif message.type == "result":
            print(f"\n\n--- Task Complete ---")

asyncio.run(main())
```

### 5.2 시스템 프롬프트 포함

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions

async def code_review(file_path: str):
    """파일에 대한 자동 코드 리뷰를 실행합니다."""
    system_prompt = """You are a senior code reviewer. Analyze the given file for:
1. Code quality and readability
2. Potential bugs or edge cases
3. Performance considerations
4. Security vulnerabilities
5. Adherence to best practices

Provide your review in a structured format with severity levels
(critical, warning, suggestion) for each finding."""

    async for message in query(
        prompt=f"Review the code in {file_path} and provide detailed feedback.",
        options=ClaudeCodeOptions(
            system_prompt=system_prompt,
            max_turns=5,
        ),
    ):
        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text, end="")

    print()  # 최종 줄바꿈

asyncio.run(code_review("src/app.py"))
```

### 5.3 구조화된 결과 처리

```python
import asyncio
import json
from claude_code_sdk import query, ClaudeCodeOptions

async def analyze_codebase(directory: str) -> dict:
    """코드베이스를 분석하고 구조화된 결과를 반환합니다."""
    prompt = f"""Analyze the codebase in {directory}. Return a JSON object with:
{{
    "total_files": <number>,
    "languages": {{"python": <count>, "javascript": <count>, ...}},
    "largest_files": [
        {{"path": "<path>", "lines": <count>}},
        ...
    ],
    "potential_issues": [
        {{"file": "<path>", "issue": "<description>", "severity": "high|medium|low"}},
        ...
    ]
}}

Use the Glob and Read tools to explore the codebase. Return ONLY the JSON object."""

    full_response = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(max_turns=20),
    ):
        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    full_response += block.text

    # 응답에서 JSON 추출
    try:
        # 응답에서 JSON 찾기
        json_start = full_response.index("{")
        json_end = full_response.rindex("}") + 1
        return json.loads(full_response[json_start:json_end])
    except (ValueError, json.JSONDecodeError):
        return {"error": "Failed to parse structured response", "raw": full_response}

result = asyncio.run(analyze_codebase("/path/to/project"))
print(json.dumps(result, indent=2))
```

---

## 6. 에이전트 만들기 (TypeScript)

### 6.1 기본 사용법

```typescript
import { query, ClaudeCodeOptions, Message } from "@anthropic-ai/claude-code-sdk";

async function main() {
  const options: ClaudeCodeOptions = {
    maxTurns: 10,
  };

  for await (const message of query({
    prompt: "List all Python files in the current directory and count the total lines of code.",
    options,
  })) {
    switch (message.type) {
      case "assistant":
        for (const block of message.content) {
          if ("text" in block) {
            process.stdout.write(block.text);
          }
        }
        break;

      case "tool_use":
        console.log(`\n  [Tool: ${message.toolName}]`);
        break;

      case "result":
        console.log("\n\n--- Task Complete ---");
        break;
    }
  }
}

main();
```

### 6.2 구성 포함

```typescript
import { query, ClaudeCodeOptions } from "@anthropic-ai/claude-code-sdk";

async function generateTests(filePath: string) {
  const systemPrompt = `You are a testing expert. Generate comprehensive unit tests
for the given code. Use pytest for Python, vitest for TypeScript.
Cover: happy paths, edge cases, error conditions, and boundary values.`;

  for await (const message of query({
    prompt: `Read ${filePath} and write comprehensive unit tests for it. Save the tests to a file.`,
    options: {
      systemPrompt,
      maxTurns: 15,
      allowedTools: ["Read", "Write", "Glob", "Bash"],
    },
  })) {
    if (message.type === "assistant") {
      for (const block of message.content) {
        if ("text" in block) {
          process.stdout.write(block.text);
        }
      }
    }
  }
}

generateTests("src/utils.ts");
```

---

## 7. 구성 옵션

### 7.1 사용 가능한 옵션

```python
from claude_code_sdk import ClaudeCodeOptions

options = ClaudeCodeOptions(
    # 모델 선택
    model="claude-sonnet-4-20250514",        # 사용할 모델

    # 턴 제한
    max_turns=25,                             # 최대 에이전트 루프 반복 횟수

    # 시스템 프롬프트
    system_prompt="You are a helpful assistant.",

    # 작업 디렉토리
    cwd="/path/to/project",                   # 작업 디렉토리 설정

    # 권한 설정
    permission_mode="default",                # "default", "acceptEdits",
                                              # "bypassPermissions", "plan"

    # 도구 제한
    allowed_tools=["Read", "Write", "Bash"],  # 특정 도구 허용 목록
    disallowed_tools=["WebFetch"],            # 특정 도구 차단 목록

    # MCP 서버
    mcp_servers=[                             # MCP 서버 연결
        {
            "name": "my-server",
            "command": "node",
            "args": ["/path/to/server.js"],
        }
    ],
)
```

### 7.2 모델 선택

```python
# 복잡한 작업에는 가장 유능한 모델 사용
options = ClaudeCodeOptions(model="claude-opus-4-20250514")

# 균형 잡힌 성능/비용에는 Sonnet 사용
options = ClaudeCodeOptions(model="claude-sonnet-4-20250514")

# 간단하고 빠른 작업에는 Haiku 사용
options = ClaudeCodeOptions(model="claude-haiku-3-5-20241022")
```

### 7.3 권한 모드

```
┌─────────────────┬─────────────────────────────────────────────────┐
│ 모드             │ 설명                                             │
├─────────────────┼─────────────────────────────────────────────────┤
│ default         │ 파일 쓰기와 셸 명령에 대한 승인이 있는           │
│                 │ 표준 권한                                         │
├─────────────────┼─────────────────────────────────────────────────┤
│ acceptEdits     │ 파일 편집 자동 승인; 셸 명령은 여전히 프롬프트   │
├─────────────────┼─────────────────────────────────────────────────┤
│ bypassPermissions│ 모든 것 자동 승인 (샌드박스 환경에서만 사용)    │
├─────────────────┼─────────────────────────────────────────────────┤
│ plan            │ 읽기 전용; 쓰기 또는 셸 명령 없음                │
│                 │ (탐색 및 분석용)                                  │
└─────────────────┴─────────────────────────────────────────────────┘
```

자동화된 CI/CD 또는 샌드박스 환경의 경우:

```python
# 격리로 안전성이 관리되는 Docker 컨테이너 또는 CI 환경에서
options = ClaudeCodeOptions(
    permission_mode="bypassPermissions",
    max_turns=50,
)
```

### 7.4 도구 제한

에이전트가 사용할 수 있는 도구를 제어합니다:

```python
# 읽기 전용 분석: 쓰기 없음, 셸 없음
options = ClaudeCodeOptions(
    allowed_tools=["Read", "Glob", "Grep"],
)

# 전체 개발: 웹을 제외한 모든 도구
options = ClaudeCodeOptions(
    disallowed_tools=["WebFetch", "WebSearch"],
)
```

---

## 8. 에이전트 응답 다루기

### 8.1 이벤트 기반 처리

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions

async def process_events():
    """다양한 이벤트 유형 처리를 시연합니다."""
    tool_calls = []
    text_output = []
    errors = []

    async for message in query(
        prompt="Find all TODO comments in the codebase and list them.",
        options=ClaudeCodeOptions(max_turns=15),
    ):
        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    text_output.append(block.text)

        elif message.type == "tool_use":
            tool_calls.append({
                "tool": message.tool_name,
                "input": getattr(message, "tool_input", None),
            })

        elif message.type == "error":
            errors.append(str(message))

        elif message.type == "result":
            pass  # 최종 결과

    # 요약
    print(f"Tool calls made: {len(tool_calls)}")
    for tc in tool_calls:
        print(f"  - {tc['tool']}")
    print(f"Errors: {len(errors)}")
    print(f"Response length: {sum(len(t) for t in text_output)} characters")

asyncio.run(process_events())
```

### 8.2 진행 상황 추적

```python
import asyncio
import time
from claude_code_sdk import query, ClaudeCodeOptions

async def run_with_progress(prompt: str):
    """진행 상황 추적과 함께 에이전트 작업을 실행합니다."""
    start_time = time.time()
    turn_count = 0
    tool_count = 0

    print(f"Task: {prompt}")
    print("-" * 60)

    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(max_turns=20),
    ):
        if message.type == "assistant":
            turn_count += 1
            elapsed = time.time() - start_time
            print(f"  [Turn {turn_count} | {elapsed:.1f}s]", end="")

            for block in message.content:
                if hasattr(block, "text"):
                    # 각 텍스트 블록의 처음 100자 출력
                    preview = block.text[:100].replace("\n", " ")
                    print(f" {preview}...")

        elif message.type == "tool_use":
            tool_count += 1
            print(f"    -> Tool: {message.tool_name}")

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Completed in {total_time:.1f}s | {turn_count} turns | {tool_count} tool calls")

asyncio.run(run_with_progress(
    "Read package.json and suggest dependency updates"
))
```

---

## 9. 오류 처리 및 재시도

### 9.1 에이전트 오류 처리

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions

async def safe_agent_run(prompt: str, max_retries: int = 3) -> str:
    """오류 처리 및 재시도와 함께 에이전트 작업을 실행합니다."""
    for attempt in range(max_retries):
        try:
            result_text = ""
            async for message in query(
                prompt=prompt,
                options=ClaudeCodeOptions(max_turns=10),
            ):
                if message.type == "assistant":
                    for block in message.content:
                        if hasattr(block, "text"):
                            result_text += block.text

                elif message.type == "error":
                    raise RuntimeError(f"Agent error: {message}")

            return result_text

        except RuntimeError as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise

        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}")
            raise

    return ""  # 여기에 도달해서는 안 됩니다

result = asyncio.run(safe_agent_run("Summarize the project structure"))
print(result)
```

### 9.2 턴 제한 처리

에이전트가 `max_turns`에 도달하면 작업이 완료되지 않아도 중지됩니다. 이를 처리합니다:

```python
async def run_with_continuation(prompt: str, max_total_turns: int = 50):
    """턴 제한에 도달하면 계속하며 에이전트 작업을 실행합니다."""
    turns_used = 0
    batch_size = 10
    full_result = ""
    continuation_prompt = prompt

    while turns_used < max_total_turns:
        remaining = min(batch_size, max_total_turns - turns_used)
        task_complete = False

        async for message in query(
            prompt=continuation_prompt,
            options=ClaudeCodeOptions(max_turns=remaining),
        ):
            if message.type == "assistant":
                for block in message.content:
                    if hasattr(block, "text"):
                        full_result += block.text
                turns_used += 1

            elif message.type == "result":
                task_complete = True

        if task_complete:
            break

        # 완료되지 않은 경우 컨텍스트와 함께 계속합니다
        continuation_prompt = (
            "Continue the previous task. Here is what you have done so far:\n"
            f"{full_result[-500:]}\n\n"  # 컨텍스트의 마지막 500자
            "Please continue and complete the task."
        )

    return full_result
```

---

## 10. SDK 컨텍스트에서의 훅(Hook)

훅(레슨 5에서 다룸)은 Agent SDK와도 함께 작동합니다. 에이전트 동작을 가로채고 수정할 수 있습니다:

```python
# .claude/settings.json (프로젝트 디렉토리에서)
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python /path/to/validate_command.py"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "python /path/to/lint_file.py"
          }
        ]
      }
    ]
  }
}
```

이러한 훅 구성이 있는 디렉토리에서 에이전트가 실행되면, 훅이 자동으로 실행됩니다. 이는 다음에 유용합니다:
- **유효성 검사**: 실행 전 명령 확인
- **린팅(Linting)**: 에이전트가 파일을 쓴 후 자동 린팅
- **로깅**: 감사를 위해 모든 도구 호출 추적
- **보안**: 위험한 작업 차단

### 프로그래밍 방식의 훅 유사 동작

설정 파일 없이 훅 유사 동작을 원한다면, 스트리밍 루프에서 이벤트를 처리하십시오:

```python
async def agent_with_guards(prompt: str):
    """프로그래밍 방식의 안전 확인과 함께 에이전트를 실행합니다."""
    blocked_patterns = ["rm -rf", "DROP TABLE", "sudo"]

    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(max_turns=15),
    ):
        if message.type == "tool_use":
            # 위험한 패턴에 대한 도구 입력 확인
            tool_input = str(getattr(message, "tool_input", ""))
            for pattern in blocked_patterns:
                if pattern in tool_input:
                    print(f"  [BLOCKED] Tool {message.tool_name} "
                          f"contains blocked pattern: {pattern}")
                    # 참고: 이것은 기록만 합니다; SDK가 실제 실행을 관리합니다.
                    # 실제 차단을 위해서는 훅이나 permission_mode를 사용하십시오.

        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text, end="")
```

---

## 11. MCP 서버 통합

Agent SDK는 MCP 서버에 연결하여 커스텀 도구와 리소스로 에이전트의 기능을 확장할 수 있습니다:

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions

async def agent_with_mcp():
    """MCP 서버 통합으로 에이전트를 실행합니다."""
    options = ClaudeCodeOptions(
        max_turns=15,
        mcp_servers=[
            {
                "name": "database",
                "command": "python",
                "args": ["/path/to/db-mcp-server/server.py"],
                "env": {
                    "DB_PATH": "/path/to/production.db",
                },
            },
            {
                "name": "weather",
                "command": "node",
                "args": ["/path/to/weather-server/dist/index.js"],
                "env": {
                    "WEATHER_API_KEY": "your-key-here",
                },
            },
        ],
    )

    async for message in query(
        prompt="Query the database for all users in Tokyo, "
               "then check the current weather there.",
        options=options,
    ):
        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text, end="")
        elif message.type == "tool_use":
            print(f"\n  [Tool: {message.tool_name}]")

asyncio.run(agent_with_mcp())
```

에이전트는 동일한 작업에서 내장 도구(Read, Bash 등)와 MCP 도구(데이터베이스 쿼리, 날씨 조회) 모두를 사용하여 필요에 따라 체이닝할 수 있습니다.

### 원격 MCP 서버

Streamable HTTP 전송을 사용하는 원격 MCP 서버의 경우:

```python
options = ClaudeCodeOptions(
    mcp_servers=[
        {
            "name": "remote-db",
            "type": "url",
            "url": "https://mcp.internal.company.com/database",
            "headers": {
                "Authorization": "Bearer your-token-here",
            },
        },
    ],
)
```

---

## 12. 연습 문제

### 연습 1: 간단한 에이전트 작업 (초급)

Agent SDK를 사용하여 Python 파일을 읽고, 그 안에 정의된 함수의 수를 세고, 요약을 출력하는 Python 스크립트를 작성하십시오. 여러분 자신의 파일 중 하나로 테스트하십시오.

### 연습 2: 코드 분석 파이프라인 (중급)

다음을 수행하는 에이전트 파이프라인을 구축하십시오:
1. 프로젝트 디렉토리에서 모든 Python 파일을 스캔합니다.
2. 각 파일에서 누락된 독스트링(Docstring), 타입 힌트(Type Hint), 테스트 커버리지를 확인합니다.
3. Markdown 형식으로 보고서를 생성합니다.
4. 보고서를 `code_analysis_report.md`에 저장합니다.

적절한 도구 제한(Read, Glob, Grep, Write)과 함께 Agent SDK를 사용하십시오.

### 연습 3: 이벤트 기반 대시보드 (중급)

에이전트를 실행하고 다음을 보여주는 실시간 대시보드를 표시하는 스크립트를 만드십시오:
- 현재 턴 번호
- 지금까지 사용된 도구 (횟수 포함)
- 경과 시간
- 생성된 텍스트의 문자 수
- 현재 상태 (생각 중/도구 사용/완료)

화면 제어 문자 또는 `rich` 같은 라이브러리를 사용하여 표시하십시오.

### 연습 4: 다중 에이전트 조율 (고급)

다음과 같이 여러 Agent SDK 호출을 순차적으로 사용하는 시스템을 구축하십시오:
1. 에이전트 1 (분석): 코드베이스를 분석하고 리팩토링이 필요한 영역을 식별합니다.
2. 에이전트 2 (계획): 에이전트 1의 출력을 받아 상세한 리팩토링 계획을 만듭니다.
3. 에이전트 3 (실행): 리팩토링 계획을 실행합니다.
4. 에이전트 4 (검토): 변경 사항을 검토하고 문제점을 보고합니다.

각 에이전트의 출력을 다음 에이전트의 프롬프트에 컨텍스트로 전달하십시오.

### 연습 5: MCP 통합을 포함한 에이전트 (고급)

여러분이 구축한 커스텀 MCP 서버(레슨 13에서)에 연결하는 Agent SDK 애플리케이션을 만드십시오. 에이전트는 다음을 수행해야 합니다:
1. MCP 도구를 사용하여 외부 소스에서 데이터를 가져옵니다.
2. 내장 도구(계산을 위한 Bash, 출력을 위한 Write)를 사용하여 데이터를 처리합니다.
3. 요약 보고서를 생성합니다.

MCP 서버의 stdio 및 HTTP 전송 모두로 테스트하십시오.

---

## 13. 참고 자료

- Claude Code SDK 문서 - https://docs.anthropic.com/en/docs/claude-code/sdk
- Claude Code SDK Python 패키지 - https://pypi.org/project/claude-code-sdk/
- Claude Code SDK TypeScript 패키지 - https://www.npmjs.com/package/@anthropic-ai/claude-code-sdk
- Claude Code 아키텍처 - https://docs.anthropic.com/en/docs/claude-code/overview
- 에이전트 루프 문서 - https://docs.anthropic.com/en/docs/claude-code/sdk#the-agent-loop
- MCP 통합 - https://docs.anthropic.com/en/docs/claude-code/mcp

---

## 다음 레슨

[18. 커스텀 에이전트 구축](./18_Building_Custom_Agents.md)에서는 Agent SDK를 더 발전시켜, 커스텀 도구 개발, 에이전트를 위한 시스템 프롬프트 엔지니어링, 실용적인 에이전트 예제(코드 리뷰, 문서화, 데이터베이스 마이그레이션, 고객 지원), 테스트 전략, 프로덕션 배포 패턴을 다룹니다.
