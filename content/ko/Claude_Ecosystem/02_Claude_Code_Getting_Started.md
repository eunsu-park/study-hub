# Claude Code: 시작하기

**이전**: [01. Claude 소개](./01_Introduction_to_Claude.md) | **다음**: [03. CLAUDE.md와 프로젝트 설정](./03_CLAUDE_md_and_Project_Setup.md)

---

Claude Code는 터미널에서 직접 실행되는 커맨드라인 AI 코딩 어시스턴트입니다. 웹 기반 AI 도구와 달리, Claude Code는 실제 프로젝트의 컨텍스트에서 동작합니다 — 파일을 읽고, 코드를 편집하고, 명령을 실행하고, 테스트를 수행하고, git 커밋을 만들 수 있습니다. 이 레슨은 설치, 첫 번째 세션, 그리고 Claude Code를 효과적인 개발 파트너로 만드는 기본 워크플로우를 안내합니다.

**난이도**: ⭐

**선행 조건**:
- [01. Claude 소개](./01_Introduction_to_Claude.md)
- 터미널/커맨드라인 기초 (**Shell_Script** 토픽 참조)
- 코드 에디터와 작업할 프로젝트
- Node.js 18+ 설치 (npm 설치용)

**학습 목표**:
- Claude Code 설치 및 인증하기
- 대화형 세션 시작하고 탐색하기
- Claude Code가 사용하는 핵심 도구 이해하기 (Read, Write, Edit, Bash, Glob, Grep)
- 읽기-편집-테스트-커밋 워크플로우 따르기
- 세션 관리를 위한 기본 슬래시 명령어 사용하기
- 처음부터 끝까지 실용적인 디버깅 연습 완료하기

---

## 목차

1. [Claude Code란 무엇인가?](#1-claude-code란-무엇인가)
2. [설치](#2-설치)
3. [인증](#3-인증)
4. [첫 번째 세션](#4-첫-번째-세션)
5. [도구 시스템](#5-도구-시스템)
6. [핵심 워크플로우](#6-핵심-워크플로우)
7. [세션 관리](#7-세션-관리)
8. [필수 슬래시 명령어](#8-필수-슬래시-명령어)
9. [실용 연습: 버그 수정](#9-실용-연습-버그-수정)
10. [작업 디렉토리와 프로젝트 범위](#10-작업-디렉토리와-프로젝트-범위)
11. [효과적인 사용을 위한 팁](#11-효과적인-사용을-위한-팁)
12. [연습 문제](#12-연습-문제)
13. [다음 단계](#13-다음-단계)

---

## 1. Claude Code란 무엇인가?

Claude Code는 **에이전트형 코딩 도구(agentic coding tool)** 입니다 — 단순히 코드에 대한 질문에 답하는 것이 아니라 행동을 취합니다. 작업을 설명하면 Claude Code는:

1. 컨텍스트를 이해하기 위해 프로젝트 파일을 **읽습니다**
2. 코드베이스를 기반으로 접근 방식을 **계획합니다**
3. 변경 사항을 구현하기 위해 파일을 **편집합니다**
4. 테스트, 빌드, 검증을 위해 명령을 **실행합니다**
5. 결과를 바탕으로 **반복합니다** (오류 수정, 접근 방식 조정)

이 에이전트 루프는 작업이 완료될 때까지 계속됩니다. 제어권은 사용자에게 있습니다 — Claude Code는 변경하기 전에 허가를 요청합니다(별도로 설정하지 않은 경우).

```
┌─────────────────────────────────────────────────────────┐
│                   Claude Code 에이전트 루프              │
│                                                         │
│     작업 설명                                           │
│           │                                             │
│           ▼                                             │
│     ┌───────────┐                                       │
│     │   읽기    │ ← 코드베이스 이해                     │
│     └─────┬─────┘                                       │
│           ▼                                             │
│     ┌───────────┐                                       │
│     │   계획    │ ← 접근 방식 결정                      │
│     └─────┬─────┘                                       │
│           ▼                                             │
│     ┌───────────┐                                       │
│     │   편집    │ ← 변경 사항 적용                      │
│     └─────┬─────┘                                       │
│           ▼                                             │
│     ┌───────────┐      ┌──────────┐                     │
│     │   테스트  │─────▶│  오류?   │──── 예 ──┐          │
│     └───────────┘      └──────────┘          │          │
│                              │               │          │
│                             아니오            │          │
│                              │               ▼          │
│                              ▼        ┌───────────┐    │
│                         ┌────────┐    │   수정    │    │
│                         │  완료  │    └─────┬─────┘    │
│                         └────────┘          │          │
│                                             └──▶ 읽기  │
└─────────────────────────────────────────────────────────┘
```

### Claude Code가 아닌 것

- **IDE 플러그인이 아님** (IDE 통합은 존재 — 레슨 09 참조)
- **코드 완성 엔진이 아님** (줄별 제안이 아닌 전체 작업 처리)
- **챗봇이 아님** (파일시스템과 터미널에 실제 행동을 취함)

---

## 2. 설치

### 방법 1: npm (권장)

```bash
# npm으로 전역 설치
npm install -g @anthropic-ai/claude-code

# 설치 확인
claude --version
```

### 방법 2: Homebrew (macOS)

```bash
# Homebrew로 설치
brew install claude-code

# 설치 확인
claude --version
```

### 시스템 요구사항

| 요구사항 | 최소 사양 |
|---------|----------|
| **OS** | macOS 12+, Ubuntu 20.04+, Windows (WSL2 경유) |
| **Node.js** | 18.0 이상 |
| **RAM** | 4 GB (8 GB 권장) |
| **터미널** | 최신 터미널 에뮬레이터 |
| **셸** | bash, zsh, 또는 fish |

### 업데이트

```bash
# 최신 버전으로 업데이트
npm update -g @anthropic-ai/claude-code

# 또는 Homebrew로
brew upgrade claude-code
```

---

## 3. 인증

Claude Code는 기능을 사용하기 위해 API 키 또는 활성 Anthropic 계정이 필요합니다. 두 가지 인증 방법이 있습니다.

### 방법 1: 대화형 로그인

```bash
# 로그인 흐름 시작
claude login

# 인증을 위해 브라우저가 열립니다
# 로그인 후 CLI가 자격 증명을 로컬에 저장합니다
```

### 방법 2: API 키

```bash
# API 키를 환경 변수로 설정
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# 또는 셸 프로필에 추가 (~/.zshrc, ~/.bashrc)
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-..."' >> ~/.zshrc
source ~/.zshrc
```

### 인증 확인

```bash
# Claude Code 시작 — 인증이 올바르면 프롬프트가 표시됩니다
claude

# 인증 문제가 있으면 수정 방법에 대한 지침과 함께
# 명확한 오류 메시지가 표시됩니다
```

### 팀을 위한 인증

조직을 통해 Claude Code를 사용하는 경우(Claude for Work 또는 Enterprise):

```bash
# 조직 계정으로 로그인
claude login

# CLI가 조직 멤버십을 자동으로 감지합니다
# 조직 정책 (모델 접근, 속도 제한) 적용
```

---

## 4. 첫 번째 세션

### 세션 시작

프로젝트 디렉토리로 이동하여 Claude Code를 시작합니다:

```bash
# 프로젝트로 이동
cd ~/projects/my-app

# 대화형 세션 시작
claude
```

Claude Code 프롬프트가 표시됩니다:

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

### 자연어 상호작용

일반 영어(또는 지원되는 모든 언어)로 요청을 입력합니다. Claude Code는 자연어를 이해하고 이를 행동으로 변환합니다.

```
> What does this project do?

Claude가 수행하는 작업:
1. README.md, package.json 또는 동등한 파일 읽기
2. 디렉토리 구조 스캔
3. 프로젝트의 목적, 기술 스택, 구조 요약 제공
```

```
> Find all TODO comments in the codebase

Claude가 수행하는 작업:
1. Grep 도구로 모든 파일에서 "TODO" 검색
2. 파일별로 결과 정리하여 표시
3. 선택적으로 TODO에 대한 수정 사항 제안
```

```
> Add input validation to the user registration endpoint

Claude가 수행하는 작업:
1. 등록 엔드포인트 코드 찾기
2. 현재 구현 읽기
3. 유효성 검사 로직 제안
4. 파일 편집 허가 요청
5. 유효성 검사 코드로 파일 편집
6. 검증을 위한 테스트 실행 제안
```

### 권한 프롬프트

기본적으로 Claude Code는 행동을 취하기 전에 허가를 요청합니다. 다음과 같은 프롬프트를 보게 됩니다:

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

- **y**: 이 특정 행동 허용
- **n**: 이 행동 거부
- **always**: 이 세션의 모든 유사한 행동 허용

---

## 5. 도구 시스템

Claude Code는 내장 **도구(tools)** 세트를 통해 동작합니다 — 각각 특정 유형의 행동을 위해 설계되었습니다. 이 도구들을 이해하면 Claude Code의 동작을 예측하고 안내하는 데 도움이 됩니다.

### 도구 참조 테이블

| 도구 | 목적 | 예시 |
|------|------|------|
| **Read** | 파일 내용 읽기 | 구조 이해를 위해 `src/app.py` 읽기 |
| **Write** | 파일 생성 또는 덮어쓰기 | 새 `test_auth.py` 파일 생성 |
| **Edit** | 기존 파일에 대한 타겟 편집 | `utils.py`의 함수 서명 변경 |
| **Bash** | 셸 명령 실행 | `pytest`, `npm test`, `git status` 실행 |
| **Glob** | 이름 패턴으로 파일 찾기 | 모든 `*.test.ts` 파일 찾기 |
| **Grep** | 파일 내용 검색 | `deprecated_function`의 모든 사용처 찾기 |
| **WebFetch** | 웹 콘텐츠 가져오기 | API 문서 가져오기 |
| **WebSearch** | 웹 검색 | 라이브러리 최신 버전 찾기 |
| **NotebookEdit** | Jupyter 노트북 편집 | `analysis.ipynb`의 셀 수정 |

### 도구들이 함께 동작하는 방식

일반적인 작업은 순서대로 여러 도구를 사용합니다:

```
작업: "test_auth.py의 실패하는 테스트 수정하기"

1단계: Bash       → `pytest test_auth.py` 실행하여 실패 확인
2단계: Read       → test_auth.py를 읽어 테스트 이해
3단계: Grep       → 테스트되는 함수 검색
4단계: Read       → 버그가 있는 소스 파일 읽기
5단계: Edit       → 소스 파일의 버그 수정
6단계: Bash       → `pytest test_auth.py` 실행하여 수정 확인
7단계: (완료)     → 성공 보고
```

### 도구 권한

각 도구는 권한 설정을 통해 개별적으로 제어할 수 있습니다 (레슨 04에서 자세히 다룸). 기본적으로:

- **Read, Glob, Grep**: 프롬프트 없이 허용 (읽기 전용 작업)
- **Edit, Write**: 승인 필요 (파일을 수정함)
- **Bash**: 승인 필요 (명령을 실행함)

---

## 6. 핵심 워크플로우

Claude Code의 기본 워크플로우는 네 단계를 따릅니다: **읽기(Read) → 편집(Edit) → 테스트(Test) → 커밋(Commit)**. 이는 숙련된 개발자가 작업하는 방식을 반영합니다 — 코드를 이해하고, 변경하고, 제대로 작동하는지 확인하고, 진행 상황을 저장합니다.

### 1단계: 읽기 (이해)

변경을 시작하기 전에 Claude Code는 컨텍스트를 구축하기 위해 관련 파일을 읽습니다.

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

### 2단계: 편집 (변경)

Claude Code는 기존 파일에는 Edit 도구를, 새 파일에는 Write를 사용하여 타겟 변경을 합니다.

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

### 3단계: 테스트 (검증)

편집 후 Claude Code는 아무 것도 깨지지 않았는지 확인하기 위해 프로젝트의 테스트 스위트를 실행합니다.

```
> Now run the tests to make sure everything works

Claude:
  Tool: Bash("npm test")

  "All 47 tests pass. The rate limiting is working correctly.
   I also see the existing test for the login route passes."
```

### 4단계: 커밋 (저장)

테스트가 통과하면 Claude Code는 잘 형식화된 git 커밋을 만들 수 있습니다.

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

## 7. 세션 관리

### 세션 수명

Claude Code 세션은 터미널이 열려 있는 동안 지속됩니다. 대화 기록은 시간이 지남에 따라 컨텍스트를 축적하여, Claude가 이전 대화를 참조할 수 있습니다.

```bash
# 세션 시작 (종료할 때까지 실행)
claude

# 세션 내에서 많은 교환을 할 수 있습니다
> Fix the login bug
> Now add tests for it
> Update the documentation
> Commit everything
> /exit
```

### 세션 재개

```bash
# 가장 최근 세션 재개
claude --resume

# 특정 세션 계속하기
claude --resume <session-id>
```

### 원샷 모드

대화형 세션이 필요 없는 빠른 작업에:

```bash
# 단일 프롬프트를 실행하고 종료
claude -p "How many TODO comments are in this project?"

# 입력을 Claude에 파이프
cat error.log | claude -p "Explain this error and suggest a fix"

# 특정 모델 사용
claude -p "Explain this code" --model claude-opus-4-20250514
```

### 컨텍스트 관리

대화가 길어지면 더 많은 토큰을 소비합니다. 컨텍스트를 관리하기 위한 전략:

```
> /compact

# 지금까지의 대화를 요약하여 컨텍스트 공간을 확보합니다
# Claude가 이전 세부 사항을 잊어버리는 것을 느낄 때 긴 세션 중에 유용합니다
```

```
> /clear

# 대화를 완전히 지우고 새로 시작합니다
# 완전히 다른 작업으로 전환할 때 사용하세요
```

---

## 8. 필수 슬래시 명령어

Claude Code는 세션 관리와 일반적인 작업을 위한 내장 슬래시 명령어를 제공합니다.

### 명령어 참조

| 명령어 | 설명 |
|--------|------|
| `/help` | 모든 사용 가능한 명령어와 사용 팁 표시 |
| `/exit` | 현재 세션 종료 |
| `/clear` | 대화 기록 지우기 |
| `/compact` | 컨텍스트 공간 확보를 위해 대화 요약 |
| `/cost` | 현재 세션의 토큰 사용량과 비용 표시 |
| `/doctor` | 설정 및 연결 문제 진단 |
| `/init` | 현재 프로젝트를 위한 CLAUDE.md 파일 초기화 |
| `/model` | 현재 모델 표시 또는 변경 |
| `/config` | 설정 보기 또는 편집 |

### /cost 사용하기

`/cost` 명령어는 개발 세션 중 지출을 추적하는 데 유용합니다:

```
> /cost

Session cost:
  Input tokens:  45,230  ($0.14)
  Output tokens: 12,847  ($0.19)
  Total cost:    $0.33

  Model: claude-sonnet-4-20250514
  Session duration: 23 minutes
```

### /doctor 사용하기

작동이 올바르지 않을 때 `/doctor`가 진단을 실행합니다:

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

## 9. 실용 연습: 버그 수정

완전하고 현실적인 디버깅 세션을 진행해 보겠습니다. 이는 문제 식별부터 커밋된 수정까지 전체 Claude Code 워크플로우를 보여줍니다.

### 시나리오

Python Flask 애플리케이션에 실패하는 테스트가 있습니다. 테스트 보고서는 다음을 보여줍니다:

```
FAILED tests/test_api.py::test_create_user - AssertionError:
  Expected status code 201, got 400
```

### 단계별 세션

```bash
# 프로젝트로 이동하여 Claude Code 시작
cd ~/projects/user-service
claude
```

**교환 1: 실패 이해하기**

```
> The test test_create_user in tests/test_api.py is failing.
  It expects a 201 but gets 400. Can you investigate and fix it?
```

Claude Code가 조사를 시작합니다:

```
Tool: Bash("python -m pytest tests/test_api.py::test_create_user -v")

FAILED tests/test_api.py::test_create_user - AssertionError:
  assert 400 == 201

Tool: Read("tests/test_api.py")

# Claude가 테스트를 읽고 확인:
def test_create_user(client):
    response = client.post("/api/users", json={
        "name": "Alice",
        "email": "alice@example.com",
        "age": 25
    })
    assert response.status_code == 201

Tool: Grep("create_user", type="py")

# 라우트 핸들러 찾기
Tool: Read("src/routes/users.py")
```

**교환 2: 근본 원인 파악하기**

Claude가 설명합니다:

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

**교환 3: 수정 적용하기**

```
> Make role optional with a default of "user"
```

```
Tool: Edit("src/routes/users.py")

  - "role": {"type": "string", "required": True},
  + "role": {"type": "string", "required": False, "default": "user"},

Tool: Edit("src/routes/users.py")

  # 핸들러 함수에 기본값 로직 추가:
  + role = data.get("role", "user")
```

**교환 4: 검증하기**

```
Tool: Bash("python -m pytest tests/test_api.py -v")

tests/test_api.py::test_create_user PASSED
tests/test_api.py::test_list_users PASSED
tests/test_api.py::test_delete_user PASSED

3 passed in 0.42s
```

**교환 5: 커밋하기**

```
> Commit this fix

Tool: Bash("git add src/routes/users.py")
Tool: Bash('git commit -m "Make role field optional with default value

The create_user endpoint required a role field after a recent
update, breaking backward compatibility. Made the field optional
with a default of \"user\" to maintain API compatibility."')
```

---

## 10. 작업 디렉토리와 프로젝트 범위

Claude Code는 **작업 디렉토리(working directory)** — `claude` 명령을 시작한 디렉토리 — 의 컨텍스트 내에서 동작합니다. 이는 Claude가 볼 수 있는 파일과 로드하는 프로젝트 컨텍스트를 결정합니다.

### 프로젝트 루트 감지

Claude Code는 다음을 찾아 프로젝트 루트를 자동으로 감지합니다:

1. `CLAUDE.md` 파일
2. `.git` 디렉토리
3. `package.json`, `pyproject.toml`, `Cargo.toml` 또는 유사한 프로젝트 파일
4. `.claude/` 디렉토리

### 범위 규칙

```
/Users/you/projects/my-app/     ← 프로젝트 루트 (.git 있음)
├── CLAUDE.md                    ← 세션 시작 시 읽힘
├── src/
│   ├── app.py                   ← 범위 내 ✓
│   └── utils.py                 ← 범위 내 ✓
├── tests/
│   └── test_app.py              ← 범위 내 ✓
└── node_modules/                ← 범위 내 (보통 무시됨)

/Users/you/other-project/        ← 범위 밖 ✗
```

### 여러 프로젝트

여러 프로젝트에 걸쳐 작업해야 하는 경우:

```bash
# 옵션 1: 상위 디렉토리에서 Claude Code 시작
cd ~/projects
claude
# Claude가 이제 모든 하위 디렉토리에 접근 가능

# 옵션 2: 별도의 세션 사용
# 터미널 1
cd ~/projects/frontend && claude

# 터미널 2
cd ~/projects/backend && claude
```

---

## 11. 효과적인 사용을 위한 팁

### 구체적으로 표현하기

```
# 덜 효과적
> Fix the bug

# 더 효과적
> The /api/users endpoint returns 500 when the email field
  contains unicode characters. Fix the validation logic.
```

### 컨텍스트 제공하기

```
# 덜 효과적
> Add caching

# 더 효과적
> Add Redis caching to the get_user_by_id function in
  src/services/user.py. Cache entries should expire after
  5 minutes. We're using the redis-py library.
```

### Claude가 검증하도록 하기

```
# 좋은 패턴: 변경 후 Claude에게 테스트 실행 요청
> Fix the sorting bug in utils.py and run the tests to confirm
```

### 반복적 개선 사용하기

```
> Add pagination to the list endpoint
# Claude가 기본 페이지네이션 구현

> The offset-based pagination is fine but add a total_count
  field in the response so the frontend knows how many pages
  there are
# Claude가 구현 개선

> Now add tests for the pagination edge cases — empty results,
  last page, invalid page numbers
# Claude가 포괄적인 테스트 추가
```

---

## 12. 연습 문제

### 연습 문제 1: 설치 확인

Claude Code를 설치하고 작동을 확인하세요:

1. npm 또는 Homebrew로 설치
2. `claude login` 또는 API 키로 인증
3. `claude --version` 실행하고 버전 확인
4. `claude`로 세션 시작하고 "What directory am I in?" 질문
5. `/doctor` 실행하여 설정 확인
6. `/cost` 실행하여 세션 비용 확인
7. `/exit`으로 종료

### 연습 문제 2: 코드베이스 탐색

현재 작업 중인 프로젝트로 이동하여 Claude Code 세션을 시작하세요:

1. Claude에게 프로젝트 구조 설명 요청
2. 특정 모듈을 임포트하는 모든 파일 찾기 요청
3. 헷갈리는 함수 설명 요청
4. 잠재적 버그나 코드 스멜 식별 요청

### 연습 문제 3: 버그 수정 연습

프로젝트 중 하나에 의도적인 버그를 만드세요(또는 테스트 프로젝트 사용):

1. 버그 도입 (예: 하나씩 오류, 변수 이름 오타)
2. Claude Code를 시작하고 원인을 밝히지 않고 증상 설명
3. Claude가 조사하고 수정 사항을 제안하도록 두기
4. 수정 사항 검토, 승인하고 테스트 실행
5. Claude에게 커밋 만들기 요청

---

## 13. 다음 단계

이제 Claude Code가 설치되었고 핵심 워크플로우를 이해했습니다. 다음 레슨은 **CLAUDE.md** — 프로젝트의 관례, 코딩 표준, 테스트 절차에 대해 Claude Code를 가르치는 프로젝트 설정 파일 — 를 다룹니다. 잘 작성된 CLAUDE.md는 팀의 관행을 따르는 데 필요한 컨텍스트를 제공하여 Claude Code의 효과를 크게 향상시킵니다.

**다음**: [03. CLAUDE.md와 프로젝트 설정](./03_CLAUDE_md_and_Project_Setup.md)
