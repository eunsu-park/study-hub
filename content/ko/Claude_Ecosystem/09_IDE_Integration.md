# IDE 통합(IDE Integration)

**이전**: [08. 에이전트 팀](./08_Agent_Teams.md) | **다음**: [10. Claude 데스크톱 애플리케이션](./10_Claude_Desktop.md)

---

Claude Code는 터미널에만 국한되지 않습니다. 가장 널리 사용되는 두 IDE 계열인 **VS Code**와 **JetBrains**에 깊이 통합되어, AI 보조 코딩을 편집기 내부에서 직접 사용할 수 있게 합니다. 이 레슨에서는 IDE 내에서 Claude Code를 최대한 활용하기 위한 설치, 주요 기능, 키보드 단축키, 워크플로 팁을 다룹니다.

**난이도**: ⭐

**전제 조건**:
- 레슨 02: Claude Code 시작하기 (CLI 기초)
- VS Code 또는 JetBrains IDE 중 하나에 대한 기본 지식

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. VS Code용 Claude Code 확장 프로그램 설치 및 설정
2. JetBrains IDE용 Claude Code 플러그인 설치 및 설정
3. IDE 내에서 대화형 코딩을 위한 Claude Code 패널 사용
4. 인라인 차이(diff)를 검토하고 제안된 변경 사항 수락/거절
5. @-멘션을 사용하여 파일 컨텍스트를 효율적으로 제공
6. IDE 내에서 플랜 모드(Plan mode) 검토 탐색
7. 일반적인 작업에 대한 키보드 단축키 적용
8. 터미널 전용 워크플로와 IDE 통합 워크플로 비교하여 적합한 방법 선택

---

## 목차

1. [개요: 터미널 + IDE](#1-개요-터미널--ide)
2. [VS Code 확장 프로그램](#2-vs-code-확장-프로그램)
3. [JetBrains 플러그인](#3-jetbrains-플러그인)
4. [인라인 차이 검토](#4-인라인-차이-검토)
5. [파일 컨텍스트를 위한 @-멘션](#5-파일-컨텍스트를-위한--멘션)
6. [IDE에서의 플랜 모드](#6-ide에서의-플랜-모드)
7. [터미널 통합](#7-터미널-통합)
8. [키보드 단축키 참조](#8-키보드-단축키-참조)
9. [터미널 전용 vs. IDE 워크플로](#9-터미널-전용-vs-ide-워크플로)
10. [효과적인 IDE + Claude Code 워크플로 팁](#10-효과적인-ide--claude-code-워크플로-팁)
11. [문제 해결](#11-문제-해결)
12. [연습 문제](#12-연습-문제)
13. [참고 자료](#13-참고-자료)

---

## 1. 개요: 터미널 + IDE

Claude Code는 두 가지 수준에서 동작합니다:

1. **CLI(터미널)**: 핵심 경험. 터미널에서 자연어를 입력하면 Claude Code가 코드를 읽고, 편집하고, 실행합니다.
2. **IDE 확장 프로그램/플러그인**: 편집기 내에 통합된 패널로, CLI 경험을 시각적으로 향상시킵니다 — 인라인 차이(diff), 파일 인식 컨텍스트, 편집기 네이티브 상호작용.

```
┌──────────────────────────────────────────────────────────────┐
│                         Your IDE                             │
│                                                              │
│  ┌──────────────────────────────┬──────────────────────────┐ │
│  │                              │    Claude Code Panel     │ │
│  │       Editor Area            │                          │ │
│  │                              │  > Refactor the login    │ │
│  │  - Inline diffs shown here   │    handler to use async  │ │
│  │  - Accept/reject changes     │                          │ │
│  │  - See proposed edits in     │  Claude: I'll refactor   │ │
│  │    context                   │  the login handler...    │ │
│  │                              │                          │ │
│  │                              │  [Plan] [Accept] [Reject]│ │
│  └──────────────────────────────┴──────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Terminal (integrated) — Claude Code CLI also available  │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

IDE 통합은 별도의 제품이 **아닙니다**. 동일한 Claude Code 엔진이 편집기의 네이티브 UI를 통해 표현된 것입니다. 설정, CLAUDE.md, 권한, 훅(hook)은 모두 동일하게 작동합니다.

---

## 2. VS Code 확장 프로그램

### 2.1 설치

**방법 1: VS Code 마켓플레이스**

1. VS Code를 엽니다
2. 확장 프로그램으로 이동합니다 (macOS: `Cmd+Shift+X`, Windows/Linux: `Ctrl+Shift+X`)
3. "Claude Code"를 검색합니다
4. 공식 Anthropic 확장 프로그램에서 **설치**를 클릭합니다

**방법 2: 커맨드 라인**

```bash
code --install-extension anthropic.claude-code
```

**방법 3: Claude Code CLI에서**

터미널에 Claude Code가 이미 설치되어 있는 경우:

```bash
claude install-extension vscode
```

### 2.2 초기 설정

설치 후:

1. 활동 바(왼쪽 사이드바)에 Claude Code 아이콘이 나타납니다
2. 아이콘을 클릭하여 Claude Code 패널을 엽니다
3. 아직 인증되지 않은 경우 로그인 메시지가 표시됩니다
4. 확장 프로그램이 기존의 `~/.claude/` 설정을 자동으로 감지합니다

```
┌─────────────────────────────────┐
│  VS Code Activity Bar          │
│                                 │
│  📁 Explorer                    │
│  🔍 Search                     │
│  🔀 Source Control              │
│  🐛 Run and Debug              │
│  📦 Extensions                  │
│  🤖 Claude Code  ◀── NEW       │
│                                 │
└─────────────────────────────────┘
```

### 2.3 Claude Code 패널 열기

Claude Code 패널을 여는 방법은 여러 가지입니다:

| 방법 | macOS | Windows/Linux |
|--------|-------|---------------|
| 키보드 단축키 | `Cmd+Esc` | `Ctrl+Esc` |
| 활동 바 | Claude Code 아이콘 클릭 | Claude Code 아이콘 클릭 |
| 명령 팔레트 | `Cmd+Shift+P` → "Claude Code: Open" | `Ctrl+Shift+P` → "Claude Code: Open" |

패널은 사이드바(기본적으로 오른쪽)로 열리거나, 아래쪽 패널 영역으로 드래그할 수 있습니다.

### 2.4 Claude Code 패널

패널은 CLI와 유사한 대화형 인터페이스를 제공합니다:

```
┌─────────────────────────────────────┐
│  Claude Code                    [⚙] │
│─────────────────────────────────────│
│                                     │
│  Session: my-project (active)       │
│  Model: claude-sonnet-4-6    │
│  Mode: normal                       │
│                                     │
│  You: Refactor the UserService      │
│  class to use dependency injection  │
│  instead of direct imports.         │
│                                     │
│  Claude: I'll refactor UserService  │
│  to use constructor injection.      │
│  Let me read the current code...    │
│                                     │
│  📄 Reading src/services/user.ts    │
│  ✏️  Editing src/services/user.ts   │
│  📄 Reading src/app.ts              │
│  ✏️  Editing src/app.ts             │
│                                     │
│  [View Changes] [Accept All]        │
│                                     │
│─────────────────────────────────────│
│  > Type your message...        [⏎]  │
└─────────────────────────────────────┘
```

### 2.5 상태 바 표시기

VS Code 상태 바에 Claude Code의 현재 상태가 표시됩니다:

```
┌─────────────────────────────────────────────────────────────┐
│  Status Bar (bottom of VS Code)                             │
│                                                             │
│  🤖 Claude: Ready  |  Mode: Normal  |  Tokens: 12.4K/200K  │
│  └── Agent status    └── Permission    └── Context usage    │
└─────────────────────────────────────────────────────────────┘
```

상태 표시기:
- **Ready(준비)**: Claude Code가 유휴 상태로 입력을 기다리는 중
- **Thinking(생각 중)**: 요청을 처리하는 중
- **Editing(편집 중)**: 파일을 변경하는 중
- **Waiting(대기 중)**: 승인을 기다리는 중 (플랜 모드 또는 권한 요청 시)
- 토큰 카운터는 현재 컨텍스트 창 사용량을 표시

### 2.6 설정 동기화

VS Code 확장 프로그램은 CLI와 설정을 공유합니다:

- **CLAUDE.md**: 동일한 프로젝트 지시사항 적용
- **권한(Permissions)**: `.claude/settings.json`의 동일한 허용/거부 규칙
- **훅(Hooks)**: 동일한 훅 설정이 활성화
- **스킬(Skills)**: 동일한 `/skill` 명령어 사용 가능
- **모델 선택**: 세션별로 독립적으로 변경 가능

VS Code 확장 프로그램에만 적용되는 설정은 VS Code의 설정에서 확인할 수 있습니다 (`Cmd+,`):

```json
{
  "claude-code.panelPosition": "right",
  "claude-code.fontSize": 14,
  "claude-code.showTokenCount": true,
  "claude-code.autoOpenPanel": false,
  "claude-code.theme": "auto"
}
```

---

## 3. JetBrains 플러그인

### 3.1 설치

Claude Code 플러그인은 모든 JetBrains IDE에서 사용 가능합니다:
- IntelliJ IDEA
- PyCharm
- WebStorm
- GoLand
- PhpStorm
- CLion
- Rider
- RubyMine

**설치 단계**:

1. JetBrains IDE를 엽니다
2. **설정/환경 설정(Settings/Preferences)** → **플러그인(Plugins)**으로 이동합니다
3. 마켓플레이스 탭에서 "Claude Code"를 검색합니다
4. **설치**를 클릭하고 IDE를 재시작합니다

```bash
# 대안: 커맨드 라인에서 설치 (JetBrains Toolbox가 설정된 경우)
# 플러그인 ID는 다를 수 있으니 JetBrains 마켓플레이스에서 정확한 ID를 확인하세요
```

### 3.2 VS Code와의 기능 동등성

JetBrains 플러그인은 VS Code 확장 프로그램과 동일한 핵심 기능을 제공합니다:

| 기능 | VS Code | JetBrains |
|---------|---------|-----------|
| 채팅 패널 | 예 | 예 |
| 인라인 차이 | 예 | 예 |
| @-멘션 | 예 | 예 |
| 플랜 모드 검토 | 예 | 예 |
| 터미널 통합 | 예 | 예 |
| 상태 바 | 예 | 예 |
| CLI와 설정 동기화 | 예 | 예 |

### 3.3 JetBrains 전용 워크플로

JetBrains IDE는 몇 가지 고유한 통합 지점을 제공합니다:

**인스펙션(Inspections) 통합**: Claude Code의 발견 사항이 JetBrains의 내장 코드 인스펙션과 함께 표시될 수 있습니다:

```
┌──────────────────────────────────────────────┐
│  Code Inspection Results                      │
│                                              │
│  ⚠️ JetBrains: Unused import 'os'           │
│  ⚠️ JetBrains: Method may be static         │
│  🤖 Claude: Potential SQL injection on L45   │
│  🤖 Claude: Missing null check on L78       │
└──────────────────────────────────────────────┘
```

**도구 창(Tool Window)**: Claude Code 도구 창은 IDE의 어느 위치에든 도킹할 수 있습니다:

```
View → Tool Windows → Claude Code
```

**액션 시스템(Action System) 통합**: Claude Code 액션이 JetBrains의 액션 시스템에 나타납니다:

```
Ctrl+Shift+A (Find Action) → Type "Claude"

Results:
  Claude Code: Open Panel
  Claude Code: Ask About Selection
  Claude Code: Explain Code
  Claude Code: Generate Tests
  Claude Code: Review File
```

**오른쪽 클릭 컨텍스트 메뉴**:

```
Right-click on selected code →
  Claude Code →
    Ask About Selection
    Explain This Code
    Refactor Selection
    Generate Tests for Selection
    Find Bugs in Selection
```

---

## 4. 인라인 차이 검토

IDE 통합의 가장 효과적인 기능 중 하나는 **인라인 차이(inline diff) 검토**입니다. Claude Code가 파일에 변경 사항을 제안하면, 채팅 패널의 텍스트로만 볼 수 있는 것이 아니라 편집기에서 직접 확인할 수 있습니다.

### 작동 방식

```
┌──────────────────────────────────────────────────────────────┐
│  src/services/user.ts                                        │
│──────────────────────────────────────────────────────────────│
│  10 │ export class UserService {                             │
│  11 │-  private db = new Database();     // REMOVED (red)    │
│  12 │-  private cache = new Cache();     // REMOVED (red)    │
│  11 │+  constructor(                     // ADDED (green)    │
│  12 │+    private db: Database,          // ADDED (green)    │
│  13 │+    private cache: Cache,          // ADDED (green)    │
│  14 │+  ) {}                             // ADDED (green)    │
│  15 │                                                        │
│  16 │   async getUser(id: string) {                          │
│  17 │     // unchanged code...                               │
│──────────────────────────────────────────────────────────────│
│  [Accept Change] [Reject Change] [Edit Manually]             │
└──────────────────────────────────────────────────────────────┘
```

### 차이 탐색

Claude Code가 여러 파일을 변경하면, 파일 간에 탐색할 수 있습니다:

| 동작 | macOS | Windows/Linux |
|--------|-------|---------------|
| 다음 차이 | `Cmd+Option+]` | `Ctrl+Alt+]` |
| 이전 차이 | `Cmd+Option+[` | `Ctrl+Alt+[` |
| 현재 차이 수락 | `Cmd+Enter` | `Ctrl+Enter` |
| 현재 차이 거절 | `Cmd+Backspace` | `Ctrl+Backspace` |
| 모든 차이 수락 | `Cmd+Shift+Enter` | `Ctrl+Shift+Enter` |
| 모든 차이 거절 | `Cmd+Shift+Backspace` | `Ctrl+Shift+Backspace` |

### 부분 수락

동일한 파일 내에서 일부 변경 사항은 수락하고 다른 것은 거절할 수 있습니다:

```
File: src/config.ts

Change 1: Add database pool config     [Accept ✓]
Change 2: Change port from 3000→8080   [Reject ✗]  (I want to keep 3000)
Change 3: Add Redis connection string   [Accept ✓]
```

이러한 세밀한 제어는 변경 사항이 원자적으로 적용되는 터미널에 비해 IDE 통합의 주요 장점 중 하나입니다.

---

## 5. 파일 컨텍스트를 위한 @-멘션

IDE 통합에서 **@-멘션**을 사용하여 메시지에서 파일과 코드 범위를 직접 참조할 수 있습니다. 이를 통해 파일 위치를 설명할 필요 없이 정확한 컨텍스트를 제공할 수 있습니다.

### 파일 멘션

`@` 다음에 파일명을 입력하여 참조합니다:

```
You: Refactor @src/auth/login.ts to use the error handling
     pattern from @src/utils/errors.ts
```

Claude Code는 두 파일의 전체 내용을 컨텍스트로 받아, 변경하기 전에 기존 코드를 이해합니다.

### 라인 범위 멘션

파일 내의 특정 라인을 참조합니다:

```
You: The function at @src/api/users.ts:45-80 has a bug.
     It doesn't handle the case where the user ID is null.
```

Claude Code는 파일의 45-80번 라인만 읽어 관련 코드에 집중합니다.

### 디렉토리 멘션

전체 디렉토리를 참조합니다:

```
You: Analyze the test coverage in @tests/api/ and identify
     which endpoints are missing test cases.
```

### 심볼 멘션

일부 IDE 통합에서는 심볼(함수, 클래스, 변수)을 직접 참조할 수 있습니다:

```
You: Explain what @UserService.authenticate does and whether
     it properly handles token expiration.
```

### @-멘션 자동완성

IDE는 `@`를 입력할 때 자동완성을 제공합니다:

```
You: Refactor @
              ┌──────────────────────────┐
              │  📄 src/auth/login.ts    │
              │  📄 src/auth/jwt.ts      │
              │  📄 src/auth/register.ts │
              │  📁 src/api/            │
              │  📁 src/models/         │
              └──────────────────────────┘
```

이는 전체 파일 경로를 입력하는 것보다 빠르고 경로 오류를 방지합니다.

### @-멘션 예시

| 입력 | Claude가 받는 것 |
|-------|---------------------|
| `@package.json` | package.json의 전체 내용 |
| `@src/app.ts:1-20` | src/app.ts의 1-20번 라인 |
| `@tests/` | tests/ 디렉토리의 파일 목록 |
| `@.env.example` | .env.example의 내용 |
| `@tsconfig.json` | tsconfig.json의 전체 내용 |

---

## 6. IDE에서의 플랜 모드

플랜 모드(Plan mode, 레슨 04에서 자세히 다룸)는 IDE에서 향상된 지원을 제공합니다.

### 플랜 모드 활성화

```
You: /plan Migrate our Express app to use TypeScript strict mode
```

또는 패널 컨트롤에서 플랜 모드를 전환합니다:

```
┌────────────────────────────────────────┐
│  Mode: [Normal ▼]                      │
│         ├── Normal                     │
│         ├── Plan                       │
│         └── Auto-accept               │
└────────────────────────────────────────┘
```

### 플랜 검토 인터페이스

IDE에서 플랜 모드는 제안된 변경 사항의 구조화된 뷰를 제공합니다:

```
┌──────────────────────────────────────────────────────────────┐
│  Plan: Migrate to TypeScript Strict Mode                     │
│──────────────────────────────────────────────────────────────│
│                                                              │
│  Phase 1: Configuration                                      │
│  ☐ Update tsconfig.json: enable "strict": true               │
│  ☐ Update tsconfig.json: enable "noImplicitAny": true        │
│                                                              │
│  Phase 2: Fix Type Errors (estimated: 47 files)              │
│  ☐ src/models/user.ts: Add types to 3 functions              │
│  ☐ src/models/order.ts: Add types to 5 functions             │
│  ☐ src/api/users.ts: Fix 12 implicit 'any' parameters       │
│  ☐ ... (44 more files)                                       │
│                                                              │
│  Phase 3: Verification                                       │
│  ☐ Run tsc --noEmit to verify zero errors                    │
│  ☐ Run test suite to verify no regressions                   │
│                                                              │
│  [Approve Plan] [Edit Plan] [Cancel]                         │
└──────────────────────────────────────────────────────────────┘
```

플랜을 검토하고, 편집(단계 추가/제거)하고, Claude Code가 실행을 시작하기 전에 승인할 수 있습니다. 실행 중에는 각 단계가 완료됨으로 표시됩니다.

---

## 7. 터미널 통합

IDE의 통합 터미널은 Claude Code와 원활하게 작동합니다.

### IDE 터미널에서 Claude Code 사용

IDE의 터미널에서 `claude` CLI를 직접 사용할 수 있습니다:

```bash
# In the IDE's integrated terminal
claude "Run the test suite and fix any failing tests"
```

이는 특정 작업에는 터미널 경험을 선호하고, 다른 작업에는 IDE 패널을 사용하는 경우에 유용합니다.

### 터미널 명령 출력

Claude Code가 터미널 명령을 실행할 때(Bash 도구를 통해), 출력은 다음 두 곳에서 볼 수 있습니다:

1. Claude Code 패널 (요약)
2. IDE의 터미널 (전체 출력)

```
Claude Code Panel:
  Running: npm test
  Result: 45/47 tests passed, 2 failures
  [Show Full Output]

IDE Terminal:
  $ npm test
  PASS src/auth/__tests__/login.test.ts (2.1s)
  PASS src/api/__tests__/users.test.ts (1.8s)
  FAIL src/api/__tests__/orders.test.ts (3.2s)
    ● OrderAPI › POST /orders › should validate required fields
      Expected: 400
      Received: 500
  ...
```

### 터미널 공유

IDE 터미널에서 명령을 수동으로 실행하면, 이를 참조하여 Claude Code가 출력을 확인할 수 있습니다:

```
You: I just ran `npm test` in the terminal and got failures.
     Check the terminal output and fix the failing tests.
```

---

## 8. 키보드 단축키 참조

### VS Code 단축키

| 동작 | macOS | Windows/Linux |
|--------|-------|---------------|
| Claude Code 패널 열기 | `Cmd+Esc` | `Ctrl+Esc` |
| Claude Code 패널 토글 | `Cmd+Shift+Esc` | `Ctrl+Shift+Esc` |
| 메시지 전송 | `Enter` | `Enter` |
| 입력창에서 줄 바꿈 | `Shift+Enter` | `Shift+Enter` |
| 현재 작업 취소 | `Escape` | `Escape` |
| 모든 변경 사항 수락 | `Cmd+Shift+Enter` | `Ctrl+Shift+Enter` |
| 모든 변경 사항 거절 | `Cmd+Shift+Backspace` | `Ctrl+Shift+Backspace` |
| 다음 차이 | `Cmd+Option+]` | `Ctrl+Alt+]` |
| 이전 차이 | `Cmd+Option+[` | `Ctrl+Alt+[` |
| 현재 차이 수락 | `Cmd+Enter` | `Ctrl+Enter` |
| 현재 차이 거절 | `Cmd+Backspace` | `Ctrl+Backspace` |
| Claude Code 입력에 포커스 | `Cmd+L` | `Ctrl+L` |
| 대화 지우기 | `Cmd+K` | `Ctrl+K` |
| 선택 항목에 대해 질문 | `Cmd+Shift+L` | `Ctrl+Shift+L` |

### JetBrains 단축키

| 동작 | macOS | Windows/Linux |
|--------|-------|---------------|
| Claude Code 열기 | `Cmd+Esc` | `Ctrl+Esc` |
| 메시지 전송 | `Enter` | `Enter` |
| 줄 바꿈 | `Shift+Enter` | `Shift+Enter` |
| 작업 취소 | `Escape` | `Escape` |
| 모든 변경 사항 수락 | `Cmd+Shift+Enter` | `Ctrl+Shift+Enter` |
| Claude 액션 찾기 | `Cmd+Shift+A` | `Ctrl+Shift+A` |
| 선택 항목에 대해 질문 | `Cmd+Shift+L` | `Ctrl+Shift+L` |

### 단축키 사용자 지정

**VS Code**: 키보드 단축키 열기 (`Cmd+K Cmd+S`) 후 "Claude Code" 검색

**JetBrains**: 설정 → 키맵(Keymap) → "Claude Code" 검색

```json
// VS Code keybindings.json example
[
  {
    "key": "cmd+shift+c",
    "command": "claude-code.openPanel",
    "when": "editorTextFocus"
  },
  {
    "key": "cmd+shift+e",
    "command": "claude-code.explainSelection",
    "when": "editorHasSelection"
  }
]
```

---

## 9. 터미널 전용 vs. IDE 워크플로

두 방식 모두 장점이 있습니다. 선택에 도움이 되는 비교를 살펴보세요:

### 터미널 전용 워크플로

**장점**:
- IDE 의존성 없음 — SSH, 컨테이너, 어떤 머신에서도 동작
- 빠른 시작 (IDE 오버헤드 없음)
- Claude Code 출력을 위한 전체 화면
- 서버 사이드 작업, DevOps, 자동화에 적합
- 스크립트 가능 (Claude Code를 다른 도구로 파이프)
- 모든 운영 체제에서 동일하게 작동

**단점**:
- 시각적 차이 검토 없음 (변경 사항이 텍스트로만 설명됨)
- 전체 파일 경로를 직접 입력해야 함 (@-멘션 자동완성 없음)
- 편집기 컨텍스트에서 변경 사항을 볼 수 없음
- 변경 사항을 부분적으로 수락/거절하기 어려움
- 터미널과 편집기 사이를 수동으로 전환해야 함

```
Terminal workflow:
  Terminal ──▶ Claude Code ──▶ Changes applied ──▶ Open editor to review
```

### IDE 통합 워크플로

**장점**:
- 편집기 컨텍스트에서 인라인 차이 검토
- 파일에 대한 자동완성이 포함된 @-멘션
- 주변 코드와 함께 변경 사항 확인
- 개별 변경에 대한 세밀한 수락/거절 제어
- 터미널과 편집기 간의 컨텍스트 전환 없음
- 구조화된 시각적 검토가 포함된 플랜 모드

**단점**:
- VS Code 또는 JetBrains IDE 필요
- IDE + 확장 프로그램으로 인한 추가 메모리 사용
- 확장 프로그램 업데이트가 CLI 릴리스보다 늦을 수 있음
- 일부 고급 CLI 기능이 UI에 노출되지 않을 수 있음
- IDE별 버그 및 특이사항

```
IDE workflow:
  Editor ──▶ Claude Code Panel ──▶ Inline diffs ──▶ Accept/reject in place
```

### 권장 접근 방식

대부분의 개발자는 **두 가지 모두** 사용하며, 작업에 따라 선택합니다:

| 작업 | 권장 방법 |
|------|-------------|
| 코드에 대한 빠른 질문 | IDE (코드 선택 → 질문) |
| 여러 파일 리팩토링 | IDE (인라인 차이 검토) |
| 서버 사이드 디버깅 | 터미널 (SSH 접근) |
| CI/CD 스크립팅 | 터미널 (스크립트 가능) |
| 코드 리뷰 지원 | IDE (시각적 차이) |
| 코드베이스 탐색 | 터미널 또는 IDE |
| 테스트 작성 | IDE (코드 옆에서 테스트 확인) |
| Docker/배포 작업 | 터미널 |

---

## 10. 효과적인 IDE + Claude Code 워크플로 팁

### 팁 1: 질문 전에 선택하기

Claude Code에 질문하기 전에 편집기에서 관련 코드를 선택하세요. 선택 영역은 자동으로 컨텍스트로 포함됩니다:

```
1. Select lines 45-80 in src/api/users.ts
2. Cmd+Shift+L (Ask about selection)
3. "Why does this function return undefined when the user has no orders?"
```

### 팁 2: 분할 뷰 사용

편집기 옆에 Claude Code 패널을 분할 뷰로 열어 두세요:

```
┌───────────────────────┬────────────────────┐
│                       │                    │
│   Editor              │   Claude Code      │
│   (your code)         │   (conversation)   │
│                       │                    │
│                       │                    │
└───────────────────────┴────────────────────┘
```

### 팁 3: 수락 전에 변경 사항 검토

Claude Code를 신뢰하더라도, 항상 인라인 차이를 수락하기 전에 검토하세요. 다음을 확인하세요:
- 주변 코드에서 의도치 않은 부작용
- 모듈의 인터페이스(공개 API)를 깨뜨리는 변경 사항
- 설정 가능해야 할 하드코딩된 값
- 새 코드에서 누락된 에러 처리

### 팁 4: 대규모 변경에는 플랜 모드 사용

여러 파일에 걸친 변경 사항에는 먼저 플랜 모드를 사용하세요:

```
You: /plan Add authentication middleware to all API routes
```

플랜을 검토한 후 승인하세요. 이렇게 하면 Claude Code가 편집을 시작할 때 예상치 못한 상황을 방지할 수 있습니다.

### 팁 5: 터미널과 IDE 결합

복잡한 워크플로에는 두 가지를 모두 사용하세요:

```
1. IDE: Ask Claude Code to explain the codebase architecture
2. Terminal: Use Claude Code CLI to run migration scripts
3. IDE: Review and refine the generated code
4. Terminal: Run tests and fix failures
```

### 팁 6: 패널에 집중 유지

단일 세션에서 긴 대화를 피하세요. 주제가 크게 바뀌면 새 세션을 시작하세요:

```
Session 1: "Help me debug the login flow" (focused)
Session 2: "Now help me optimize the database queries" (new topic)
```

### 팁 7: Claude와 함께 편집기 기능 활용

IDE의 내장 기능을 Claude Code와 함께 사용하세요:

- **정의로 이동(Go to Definition)**: Claude에게 리팩토링을 요청하기 전에 IDE를 사용하여 호출 위치를 파악하세요
- **참조 찾기(Find References)**: Claude가 함수를 수정하기 전에 모든 사용 위치를 확인하세요
- **Git 블레임(Git Blame)**: 변경 요청 전에 코드의 이력을 파악하세요
- **디버거(Debugger)**: Claude의 변경 사항이 올바르게 작동하는지 코드를 단계별로 실행하여 확인하세요

---

## 11. 문제 해결

### 확장 프로그램이 로드되지 않는 경우

```
Symptom: Claude Code icon not appearing in Activity Bar
Solutions:
1. Check VS Code version (requires VS Code 1.85+)
2. Reload VS Code window (Cmd+Shift+P → "Reload Window")
3. Check extension is enabled (Extensions → Claude Code → Enable)
4. Check developer console (Help → Toggle Developer Tools → Console)
```

### 인증 문제

```
Symptom: "Not authenticated" error in the IDE panel
Solutions:
1. Run 'claude auth' in the terminal first
2. Check ~/.claude/credentials exists
3. Ensure your API key or subscription is valid
4. Try logging out and back in: 'claude auth logout && claude auth login'
```

### 패널이 응답하지 않는 경우

```
Symptom: Claude Code panel shows spinner indefinitely
Solutions:
1. Cancel current operation (Escape)
2. Check network connectivity
3. Check Claude Code CLI works in terminal independently
4. Restart the extension (Cmd+Shift+P → "Claude Code: Restart")
```

### 차이 검토가 표시되지 않는 경우

```
Symptom: Claude says it edited files but no inline diffs appear
Solutions:
1. Check the file is open in the editor
2. Click "View Changes" in the Claude Code panel
3. Check Source Control panel for pending changes
4. Ensure the file is not in .gitignore (diffs use git)
```

### 높은 메모리 사용량

```
Symptom: IDE becomes slow when Claude Code is active
Solutions:
1. Close long conversation sessions (start fresh)
2. Reduce claude-code.maxHistoryItems in settings
3. Close unused editor tabs
4. Increase VS Code memory limit in settings
```

---

## 12. 연습 문제

### 연습 1: VS Code 설정

1. VS Code에 Claude Code 확장 프로그램 설치
2. CLAUDE.md 파일이 있는 프로젝트 열기
3. `Cmd+Esc`를 사용하여 Claude Code 패널 열기
4. 프로젝트 구조를 설명해 달라고 요청
5. CLAUDE.md 지시사항이 따르고 있는지 확인

### 연습 2: 인라인 차이 검토

1. Claude Code에 프로젝트의 함수를 리팩토링해 달라고 요청
2. 편집기에서 인라인 차이 검토
3. 하나의 변경은 수락하고 다른 하나는 거절
4. 파일이 선택한 내용을 반영하는지 확인

### 연습 3: @-멘션 워크플로

1. 여러 파일이 있는 프로젝트 열기
2. Claude Code 패널에서 @-멘션을 사용하여 두 파일 참조
3. 두 파일의 코드 패턴을 비교해 달라고 Claude에게 요청
4. 특정 질문을 위해 라인 범위 멘션(`@file.ts:10-30`) 사용

### 연습 4: 플랜 모드 검토

1. Claude Code 패널에서 플랜 모드로 전환
2. 여러 파일에 걸친 변경을 요청 (예: "모든 API 엔드포인트에 입력 유효성 검사 추가")
3. Claude Code가 제안하는 플랜 검토
4. 동의하지 않는 단계를 제거하도록 플랜 편집
5. 수정된 플랜을 승인하고 실행 관찰

### 연습 5: 터미널 + IDE 비교

두 가지 방식으로 동일한 작업 수행:

1. **터미널**: `claude` CLI를 사용하여 모듈에 대한 테스트 파일 추가
2. **IDE**: Claude Code 패널을 사용하여 다른 모듈에 대한 테스트 파일 추가
3. 경험 비교: 속도, 제어, 검토 품질
4. 어떤 방식을 선호하는지와 그 이유를 기록

---

## 13. 참고 자료

- [Claude Code VS Code Extension](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code)
- [Claude Code JetBrains Plugin](https://plugins.jetbrains.com/plugin/claude-code)
- [Claude Code Documentation: IDE Integration](https://docs.anthropic.com/en/docs/claude-code)
- [VS Code Extension API](https://code.visualstudio.com/api)
- [JetBrains Plugin Development](https://plugins.jetbrains.com/docs/intellij/)

---

## 다음 단계

다음 레슨 [Claude 데스크톱 애플리케이션](./10_Claude_Desktop.md)에서는 독립형 Claude 데스크톱 앱을 살펴봅니다 — macOS와 Windows용 전용 애플리케이션으로, 병렬 세션 관리, 인터페이스 내에서 웹 앱을 직접 실행하는 앱 미리보기(App Preview), 그리고 PR 모니터링 및 CI 수정 워크플로를 위한 깊은 GitHub 통합을 제공합니다.
