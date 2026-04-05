# Claude 데스크톱 애플리케이션

**이전**: [09. IDE 통합](./09_IDE_Integration.md) | **다음**: [11. Cowork: AI 디지털 동료](./11_Cowork.md)

---

Claude Desktop은 macOS와 Windows용 독립형 애플리케이션으로, 로컬 개발 환경과 깊이 통합된 전용 환경에서 Claude의 AI 기능을 제공합니다. claude.ai의 웹 인터페이스와 달리, 데스크톱 앱은 git 워크트리(worktree) 격리를 통한 병렬 코딩 세션, 웹 애플리케이션을 실행하고 볼 수 있는 앱 미리보기(App Preview), 그리고 풀 리퀘스트 모니터링 및 CI 실패 자동 수정을 위한 GitHub 통합과 같은 기능을 제공합니다. 이 레슨에서는 데스크톱 앱의 기능, 워크플로, 그리고 CLI 및 IDE 경험과의 보완적 관계를 다룹니다.

**난이도**: ⭐

**전제 조건**:
- 레슨 01: Claude 소개 (Claude 제품 생태계 이해)
- 레슨 02: Claude Code 시작하기 (Claude Code에 대한 기본 친숙도)
- Git 기초 (브랜치, 커밋, 워크트리)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Claude Desktop이 무엇이며 claude.ai 및 Claude Code CLI와 어떻게 다른지 이해
2. macOS 또는 Windows에서 Claude Desktop 설치 및 설정
3. git 워크트리 격리를 통한 병렬 세션으로 동시 작업
4. Claude 내에서 웹 애플리케이션을 실행하고 보기 위한 앱 미리보기 사용
5. PR 모니터링 및 CI 수정 워크플로를 위한 GitHub 통합 활용
6. 데스크톱 전용 설정 구성
7. Claude Desktop이 Claude Code CLI와 어떻게 통합되는지 이해

---

## 목차

1. [Claude Desktop이란?](#1-claude-desktop이란)
2. [데스크톱 vs. 웹 vs. CLI](#2-데스크톱-vs-웹-vs-cli)
3. [설치 및 설정](#3-설치-및-설정)
4. [병렬 세션과 Git 워크트리](#4-병렬-세션과-git-워크트리)
5. [시각적 차이 검토](#5-시각적-차이-검토)
6. [앱 미리보기](#6-앱-미리보기)
7. [GitHub 통합](#7-github-통합)
8. [데스크톱 설정 및 구성](#8-데스크톱-설정-및-구성)
9. [Claude Code CLI와의 통합](#9-claude-code-cli와의-통합)
10. [세션 지속성](#10-세션-지속성)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. Claude Desktop이란?

Claude Desktop은 Anthropic의 독립형 데스크톱 애플리케이션으로, Claude의 AI 기능을 네이티브 앱으로 로컬 머신에 제공합니다. claude.ai의 대화형 인터페이스와 Claude Code의 로컬 도구 기능을 결합하여, 개발 워크플로를 위해 설계된 전용 애플리케이션으로 구현했습니다.

```
┌────────────────────────────────────────────────────────────────┐
│                     Claude Desktop                             │
│                                                                │
│  ┌──────┐  ┌──────────────────────────────────────────────┐   │
│  │      │  │                                              │   │
│  │ Side │  │              Main Workspace                  │   │
│  │ bar  │  │                                              │   │
│  │      │  │  Conversation + Code + App Preview           │   │
│  │ ───  │  │                                              │   │
│  │ Sess │  │  Claude: I've updated the login page.        │   │
│  │ ion  │  │  Here's what it looks like:                  │   │
│  │ List │  │                                              │   │
│  │      │  │  ┌──────────────────────────────────┐        │   │
│  │ ───  │  │  │     App Preview                  │        │   │
│  │ PR   │  │  │     (Live web app rendering)     │        │   │
│  │ Mon  │  │  │                                  │        │   │
│  │ itor │  │  │     [Login Page Preview]         │        │   │
│  │      │  │  │                                  │        │   │
│  │      │  │  └──────────────────────────────────┘        │   │
│  │      │  │                                              │   │
│  └──────┘  └──────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

주요 특성:
- **네이티브 애플리케이션**: macOS 또는 Windows 앱으로 실행 (브라우저 아님)
- **로컬 파일 접근**: 로컬 머신의 파일을 읽고 수정
- **도구 실행**: 터미널 명령, 빌드, 테스트를 로컬에서 실행
- **지속적인 세션**: 앱 재시작 후에도 대화가 유지됨
- **Git 인식**: 저장소 구조를 이해하고 워크트리를 생성할 수 있음

---

## 2. 데스크톱 vs. 웹 vs. CLI

Claude는 여러 인터페이스를 통해 사용할 수 있습니다. 비교표를 살펴보세요:

| 기능 | claude.ai (웹) | Claude Desktop | Claude Code (CLI) |
|---------|----------------|----------------|-------------------|
| **플랫폼** | 브라우저 | macOS, Windows | 터미널 (모든 OS) |
| **로컬 파일 접근** | 아니오 (업로드만) | 예 | 예 |
| **터미널 명령** | 아니오 | 예 | 예 |
| **Git 통합** | 아니오 | 예 (워크트리, PR) | 예 (기본 git) |
| **앱 미리보기** | 아니오 | 예 | 아니오 |
| **병렬 세션** | 탭 (격리 없음) | Git 워크트리 격리 | 여러 터미널 |
| **PR 모니터링** | 아니오 | 예 (GitHub) | `gh` CLI 통해 |
| **MCP 지원** | 제한적 | 예 | 예 |
| **오프라인** | 아니오 | 아니오 | 아니오 |
| **적합한 용도** | 일반 Q&A, 글쓰기 | 개발, 프로토타이핑 | 파워 유저, 자동화 |

### 각 도구를 사용하는 경우

- **claude.ai**: 일반적인 질문, 글쓰기 작업, 아티팩트를 사용한 빠른 프로토타이핑, 로컬 코드가 필요하지 않은 경우
- **Claude Desktop**: 전체 개발 세션, 웹 앱의 시각적 피드백, PR 관리, 병렬 기능 개발
- **Claude Code CLI**: 서버 사이드 작업, CI/CD 통합, 스크립팅, SSH 세션, 자동화

---

## 3. 설치 및 설정

### macOS 설치

1. [claude.ai/download](https://claude.ai/download) 또는 Mac App Store에서 다운로드
2. 다운로드한 `.dmg` 파일 열기
3. Claude를 응용 프로그램 폴더로 드래그
4. 응용 프로그램 또는 Spotlight에서 Claude 실행
5. Anthropic 계정으로 로그인

```bash
# Verify installation
ls /Applications/Claude.app
# or
open -a Claude
```

### Windows 설치

1. [claude.ai/download](https://claude.ai/download)에서 다운로드
2. 설치 프로그램(`.exe`) 실행
3. 설치 마법사를 따름
4. 시작 메뉴 또는 검색에서 Claude 실행

### 첫 번째 설정

첫 실행 시:
1. **로그인**: Anthropic 계정 사용 (claude.ai와 동일)
2. **권한 부여**: 메시지가 표시되면 로컬 파일 시스템 접근 허용
3. **프로젝트 폴더 선택**: 작업할 디렉토리 선택
4. **도구 구성**: 로컬 도구 접근 활성화/비활성화 (터미널, 파일 편집)

```
┌──────────────────────────────────────────────┐
│  Welcome to Claude Desktop                   │
│                                              │
│  ☑ Allow file system access                  │
│  ☑ Allow terminal command execution          │
│  ☑ Allow network access                      │
│  ☐ Allow unrestricted tool use               │
│                                              │
│  Project folder: [~/projects/myapp]  [Browse]│
│                                              │
│  [Get Started]                               │
└──────────────────────────────────────────────┘
```

---

## 4. 병렬 세션과 Git 워크트리

Claude Desktop의 주목할 만한 기능 중 하나는 **여러 코딩 세션을 병렬로 실행**하는 기능으로, 각 세션은 자체 **git 워크트리(worktree)**에 격리됩니다. 이는 Claude가 코드 충돌 없이 두 가지 기능을 동시에 작업할 수 있음을 의미합니다.

### Git 워크트리란?

Git 워크트리는 동일한 저장소에 연결된 여러 작업 디렉토리를 가질 수 있게 합니다. 각 워크트리는 다른 브랜치를 체크아웃하며, 한 워크트리의 변경 사항은 다른 워크트리에 영향을 미치지 않습니다.

```bash
# Standard git workflow: one working directory
my-project/     # Only one branch checked out at a time

# With worktrees: multiple working directories
my-project/           # Main worktree (e.g., main branch)
my-project-feature-a/ # Worktree for feature-a branch
my-project-feature-b/ # Worktree for feature-b branch
```

### Claude Desktop의 워크트리 활용 방식

새 병렬 세션을 시작할 때, Claude Desktop은:

1. 작업을 위한 새 git 브랜치 생성
2. 해당 브랜치를 위한 새 워크트리 디렉토리 생성
3. 해당 워크트리 내에서 세션 전체를 실행
4. 완료 시 워크트리를 병합하거나 삭제

```
┌──────────────────────────────────────────────────────────────┐
│  Claude Desktop - Parallel Sessions                          │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │  Session 1       │  │  Session 2       │                 │
│  │  Branch: feat-a  │  │  Branch: feat-b  │                 │
│  │  Worktree:       │  │  Worktree:       │                 │
│  │  /tmp/myapp-a/   │  │  /tmp/myapp-b/   │                 │
│  │                  │  │                  │                 │
│  │  "Add user       │  │  "Fix payment    │                 │
│  │   profiles page" │  │   validation"    │                 │
│  │                  │  │                  │                 │
│  │  Status: Working │  │  Status: Testing │                 │
│  └──────────────────┘  └──────────────────┘                 │
│                                                              │
│  Both sessions work independently — no conflicts             │
└──────────────────────────────────────────────────────────────┘
```

### 병렬 세션 시작

```
1. Click "New Session" in the sidebar
2. Select "Parallel Session (new worktree)"
3. Name the branch (e.g., "feature/user-profiles")
4. Describe the task
5. Claude creates the worktree and begins working
```

### 워크트리 격리의 이점

- **개발 중 병합 충돌 없음**: 각 세션은 별도의 디렉토리에 있는 별도의 파일에서 작업
- **독립적인 테스트**: 한 워크트리에서 실행된 테스트가 다른 워크트리에 영향을 미치지 않음
- **깔끔한 롤백**: 세션이 잘못되면 워크트리를 삭제 — 메인 브랜치는 그대로 유지
- **진정한 병렬성**: Claude가 두 가지 기능을 동시에 구현 가능

### 결과 병합

병렬 세션이 완료된 후:

```
┌──────────────────────────────────────────────┐
│  Session "feat-user-profiles" Complete        │
│                                              │
│  Changes:                                    │
│  + src/pages/UserProfile.tsx (new)           │
│  + src/api/users.ts (modified)               │
│  + tests/UserProfile.test.tsx (new)          │
│                                              │
│  [Create PR]  [Merge to main]  [Discard]     │
└──────────────────────────────────────────────┘
```

---

## 5. 시각적 차이 검토

Claude Desktop은 코드 변경 사항을 검토하기 위한 풍부한 시각적 차이(diff) 인터페이스를 제공하며, IDE 통합과 유사하지만 독립형 경험으로 제공됩니다.

### 나란히 보기(Side-by-Side) 차이 뷰

```
┌───────────────────────────┬───────────────────────────┐
│  Before                   │  After                    │
│───────────────────────────│───────────────────────────│
│  10: class UserService {  │  10: class UserService {  │
│  11:   db = new DB();     │  11:   constructor(       │
│  12:                      │  12:     private db: DB,  │
│  13:   getUser(id) {      │  13:   ) {}               │
│  14:     return this.db   │  14:                      │
│  15:       .query(id);    │  15:   getUser(id: string)│
│  16:   }                  │  16:     return this.db   │
│  17: }                    │  17:       .query(id);    │
│                           │  18:   }                  │
│                           │  19: }                    │
└───────────────────────────┴───────────────────────────┘
│  [Accept] [Reject] [Edit] │ File 1 of 3 [◀ ▶]       │
└───────────────────────────────────────────────────────┘
```

### 통합 차이 뷰(Unified Diff View)

통합 뷰를 선호하는 경우:

```
  10   class UserService {
- 11     db = new DB();
- 12
- 13     getUser(id) {
+ 11     constructor(
+ 12       private db: DB,
+ 13     ) {}
+ 14
+ 15     getUser(id: string) {
  16       return this.db
  17         .query(id);
  18     }
  19   }
```

### 변경 요약

Claude Desktop은 파일 전반에 걸친 모든 변경 사항의 요약을 제공합니다:

```
┌──────────────────────────────────────────────────────────┐
│  Change Summary                                          │
│                                                          │
│  3 files changed, 24 insertions(+), 12 deletions(-)     │
│                                                          │
│  📄 src/services/user.ts      +8  -4   [View Diff]     │
│  📄 src/app.ts                +12 -6   [View Diff]     │
│  📄 tests/user.test.ts        +4  -2   [View Diff]     │
│                                                          │
│  [Accept All] [Review Each] [Reject All]                 │
└──────────────────────────────────────────────────────────┘
```

---

## 6. 앱 미리보기

앱 미리보기(App Preview)는 Claude Desktop만의 고유한 기능으로, **Claude 인터페이스 내에서 직접 웹 애플리케이션을 실행**하고 라이브 결과를 볼 수 있습니다.

### 앱 미리보기 작동 방식

Claude가 웹 애플리케이션을 빌드하거나 수정할 때, 개발 서버를 시작하고 결과를 데스크톱 앱에서 직접 렌더링할 수 있습니다:

```
┌──────────────────────────────────────────────────────────────┐
│  Claude: I've updated the dashboard. Here's the preview:     │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  App Preview - http://localhost:3000/dashboard           ││
│  │  ────────────────────────────────────────────────────────││
│  │                                                          ││
│  │  ┌─────────────────────────────────────────────────┐    ││
│  │  │  Dashboard                          [Settings]  │    ││
│  │  │                                                 │    ││
│  │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  │    ││
│  │  │  │ Users     │  │ Revenue   │  │ Orders    │  │    ││
│  │  │  │ 1,234     │  │ $45,678   │  │ 567       │  │    ││
│  │  │  └───────────┘  └───────────┘  └───────────┘  │    ││
│  │  │                                                 │    ││
│  │  │  [Chart showing weekly trends]                  │    ││
│  │  │                                                 │    ││
│  │  └─────────────────────────────────────────────────┘    ││
│  │                                                          ││
│  │  Console: No errors                         [Refresh]   ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
│  The layout looks good. Should I adjust the chart colors?    │
└──────────────────────────────────────────────────────────────┘
```

### 앱 미리보기 시작

앱 미리보기는 Claude가 다음을 수행할 때 자동으로 활성화됩니다:
1. 개발 서버 시작 (`npm run dev`, `flask run` 등)
2. HTML 파일 생성
3. 볼 수 있는 출력이 있는 웹 애플리케이션 생성

수동으로 미리보기를 요청할 수도 있습니다:

```
You: Start the dev server and show me the login page
```

### 콘솔 로그 모니터링

앱 미리보기에는 다음을 보여주는 콘솔 패널이 포함됩니다:
- JavaScript 콘솔 출력 (`console.log`, `console.error`)
- 네트워크 요청 상태
- 런타임 오류 및 경고

```
┌──────────────────────────────────────────────────────┐
│  Console Output                                      │
│                                                      │
│  [LOG]  App started on port 3000                     │
│  [LOG]  Connected to database                        │
│  [WARN] Deprecation: findDOMNode is deprecated       │
│  [ERR]  TypeError: Cannot read property 'map' of     │
│         undefined at Dashboard.tsx:45                 │
│                                                      │
│  [Clear]  [Filter: All ▼]  [Auto-scroll ☑]          │
└──────────────────────────────────────────────────────┘
```

### 실시간 오류 감지 및 자동 수정

콘솔에 오류가 표시되면, Claude Desktop이 이를 감지하고 수정을 제안할 수 있습니다:

```
Claude: I see a TypeError in Dashboard.tsx at line 45 — the
        'orders' array is undefined on initial render. This is
        because the API call hasn't completed yet. I'll add a
        loading state and null check.

        [Auto-Fix] [Show Error Details] [Ignore]
```

**자동 수정(Auto-Fix)**을 클릭하면 Claude가 다음을 수행합니다:
1. 오류 세부 정보와 스택 추적 읽기
2. 관련 파일로 이동
3. 수정 사항 적용 (예: 널 체크, 로딩 상태 추가)
4. 미리보기를 새로 고침하여 수정 확인

### 지원 프레임워크

앱 미리보기는 localhost에서 서비스하는 모든 프레임워크와 작동합니다:

| 프레임워크 | 명령어 | 자동 감지 |
|-----------|---------|--------------|
| React (Vite) | `npm run dev` | 예 |
| Next.js | `npm run dev` | 예 |
| Vue | `npm run dev` | 예 |
| Svelte | `npm run dev` | 예 |
| Flask | `flask run` | 예 |
| Express | `node server.js` | 예 |
| Django | `python manage.py runserver` | 예 |
| Static HTML | 직접 파일 렌더링 | 예 |

---

## 7. GitHub 통합

Claude Desktop은 GitHub과 통합하여 PR 모니터링, CI 상태 추적, CI 실패에 대한 자동 수정을 제공합니다.

### PR 모니터링

사이드바에 현재 저장소의 활성 풀 리퀘스트가 표시됩니다:

```
┌──────────────────────────────────────┐
│  Pull Requests                       │
│                                      │
│  ▶ #142 Add user profiles     [Open]│
│    Branch: feat/user-profiles        │
│    CI: ✓ Passing                     │
│    Reviews: 1/2 approved             │
│                                      │
│  ▶ #141 Fix payment validation [Open]│
│    Branch: fix/payment-val           │
│    CI: ✗ Failing (2 checks)         │
│    Reviews: 0/2 approved             │
│                                      │
│  ▶ #140 Update dependencies   [Open]│
│    Branch: chore/deps                │
│    CI: ⏳ Running                    │
│    Reviews: Not requested            │
│                                      │
└──────────────────────────────────────┘
```

### CI 체크 상태

PR을 클릭하면 세부 CI 체크 상태를 볼 수 있습니다:

```
┌──────────────────────────────────────────────────────────┐
│  PR #141: Fix payment validation                         │
│                                                          │
│  CI Checks:                                              │
│  ✓ lint          Passed (12s)                            │
│  ✗ test          Failed (45s)  [View Logs]               │
│  ✗ build         Failed (23s)  [View Logs]               │
│  ⏭ deploy        Skipped (depends on build)              │
│                                                          │
│  test failure:                                           │
│  FAIL src/payments/__tests__/validate.test.ts            │
│    ● should reject negative amounts                      │
│      Expected: ValidationError                           │
│      Received: undefined                                 │
│                                                          │
│  [Auto-Fix CI]  [View PR on GitHub]                      │
└──────────────────────────────────────────────────────────┘
```

### CI 실패 자동 수정

CI 체크가 실패하면, Claude Desktop이 문제를 자동으로 수정할 수 있습니다:

```
1. Click [Auto-Fix CI]
2. Claude reads the CI logs and error messages
3. Claude identifies the failing code
4. Claude creates a fix commit
5. Claude pushes to the PR branch
6. CI re-runs automatically

Workflow:
  CI fails → Claude reads logs → Identifies fix → Commits → Pushes → CI passes
```

자동 수정 흐름 예시:

```
Claude: CI check 'test' failed. I found the issue:

        In src/payments/validate.ts:23, the validateAmount()
        function doesn't throw for negative values. The test
        expects a ValidationError for negative amounts.

        Fix: Add a check for amount < 0 at the start of
        validateAmount().

        [Apply Fix and Push]  [Show Diff First]
```

### 인라인 댓글로 코드 리뷰

Claude Desktop은 PR에 인라인 댓글을 추가하여 코드 리뷰에 참여할 수 있습니다:

```
┌──────────────────────────────────────────────────────────────┐
│  PR #142 Review - src/pages/UserProfile.tsx                  │
│                                                              │
│  45 │ const [user, setUser] = useState(null);                │
│  46 │                                                        │
│  47 │ useEffect(() => {                                      │
│  48 │   fetch(`/api/users/${userId}`)                        │
│     │   🤖 Claude: Missing error handling for the fetch      │
│     │      call. If the API returns a 404 or 500, the        │
│     │      component will show a blank page. Add a           │
│     │      try/catch and set an error state.                  │
│  49 │     .then(r => r.json())                               │
│  50 │     .then(setUser);                                    │
│  51 │ }, [userId]);                                          │
│     │   🤖 Claude: userId is not in the dependency array     │
│     │      type — if it's a string, this could cause         │
│     │      re-renders on reference changes. Consider         │
│     │      memoizing or using a stable reference.            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. 데스크톱 설정 및 구성

### 설정 접근

```
macOS:  Claude → Settings (or Cmd+,)
Windows: File → Settings (or Ctrl+,)
```

### 일반 설정

```json
{
  "appearance": {
    "theme": "system",            // "light", "dark", "system"
    "fontSize": 14,
    "fontFamily": "SF Mono",
    "sidebarPosition": "left"
  },
  "sessions": {
    "autoSave": true,             // Persist sessions across restarts
    "maxParallelSessions": 4,
    "defaultWorktreeLocation": "/tmp/claude-worktrees"
  },
  "tools": {
    "allowFileAccess": true,
    "allowTerminal": true,
    "allowNetwork": true,
    "confirmBeforeExecution": true   // Prompt before running commands
  },
  "github": {
    "enabled": true,
    "autoMonitorPRs": true,
    "autoFixCI": false,           // Require manual approval for CI fixes
    "reviewComments": true
  }
}
```

### 프로젝트 전용 설정

Claude Desktop은 Claude Code CLI와 동일한 프로젝트 수준 설정을 따릅니다:

- **CLAUDE.md**: 자동으로 로드되는 프로젝트 지시사항
- **.claude/settings.json**: 권한 규칙 및 도구 구성
- **.claude/agents/**: 사용자 정의 에이전트 정의

### 모델 선택

```
┌──────────────────────────────────────────┐
│  Model Selection                         │
│                                          │
│  ○ Claude Opus 4.6     (Most capable)      │
│  ● Claude Sonnet 4.6   (Balanced)          │
│  ○ Claude Haiku 4.5  (Fastest)           │
│                                          │
│  Extended thinking: [Off ▼]              │
│  ├── Off                                 │
│  ├── Low (4K budget)                     │
│  ├── Medium (16K budget)                 │
│  └── High (32K budget)                   │
│                                          │
└──────────────────────────────────────────┘
```

---

## 9. Claude Code CLI와의 통합

Claude Desktop과 Claude Code CLI는 동일한 기본 엔진을 공유합니다. 경쟁 관계가 아닌 상호 보완적입니다.

### 공유 구성

두 도구는 동일한 구성 소스에서 읽습니다:

```
~/.claude/                    # Shared between Desktop and CLI
├── settings.json             # Global settings
├── credentials               # Authentication
└── projects/                 # Project-specific settings

project/.claude/              # Shared project settings
├── settings.json
├── agents/
└── skills/

project/CLAUDE.md             # Shared project instructions
```

### 두 도구 함께 사용

일반적인 워크플로는 동일한 프로젝트의 다른 측면에 두 도구를 모두 사용하는 것입니다:

```
Workflow: Feature Development

1. Claude Desktop: Start a parallel session for "feature/auth"
   - Claude works on the auth module with App Preview
   - You see the login page rendering in real time

2. Claude Code CLI (in terminal):
   - Meanwhile, use CLI for server-side configuration
   - Set up database migrations
   - Run performance benchmarks

3. Claude Desktop: Review the auth module changes
   - Visual diff review
   - Create a PR from the Desktop app

4. Claude Code CLI:
   - Use CLI to check CI status: gh pr checks 142
   - Run final integration tests
```

### 충돌 방지

Claude Desktop과 Claude Code CLI가 동일한 디렉토리에서 작업하는 경우:

```
Conflict scenario:
  Desktop editing src/app.ts  ←──→  CLI editing src/app.ts
  Result: One overwrites the other's changes

Prevention:
  - Use git worktrees in Desktop (separate directory)
  - Work on different files simultaneously
  - Coordinate: finish one session before starting another in the same directory
```

---

## 10. 세션 지속성

Claude Desktop은 애플리케이션 재시작 시에도 세션을 보존합니다.

### 보존되는 것

- **대화 이력**: 모든 메시지와 도구 출력
- **세션 상태**: 어떤 파일이 읽혔는지, 어떤 변경이 이루어졌는지
- **워크트리 연관**: 세션이 사용하는 git 워크트리/브랜치
- **앱 미리보기 상태**: 어떤 서버가 실행 중이었는지 (하지만 재시작 필요)

### 보존되지 않는 것

- **실행 중인 프로세스**: 앱이 닫히면 개발 서버, 테스트, 빌드가 중단됨
- **진행 중인 작업**: 앱이 닫힐 때 실행 중이던 도구 호출은 손실됨
- **컨텍스트 창 위치**: AI는 전체 이력을 가지고 시작하지만 오래된 부분을 요약할 수 있음

### 세션 재개

```
1. Open Claude Desktop
2. Sidebar shows previous sessions:

   Recent Sessions:
   ├── "Add user profiles" (2 hours ago) - feat/user-profiles
   ├── "Fix payment bug" (yesterday) - fix/payment-val
   └── "Refactor database" (3 days ago) - refactor/db-layer

3. Click a session to resume
4. Claude loads the conversation history and continues

Claude: "Welcome back! Last time we were working on the user
         profiles feature. We completed the ProfilePage component
         and the API endpoints. The remaining work is:
         - Add profile picture upload
         - Write tests
         - Update the navigation

         Should I continue with the profile picture upload?"
```

### 세션 관리

```
┌──────────────────────────────────────────────┐
│  Session Management                          │
│                                              │
│  Active Sessions: 2 of 4 max                 │
│                                              │
│  📌 feat/user-profiles   [Resume] [Archive]  │
│  📌 fix/payment-val      [Resume] [Archive]  │
│                                              │
│  Archived Sessions:                          │
│  📁 refactor/db-layer    [Restore] [Delete]  │
│  📁 chore/update-deps    [Restore] [Delete]  │
│                                              │
│  [New Session]  [Clean Up Worktrees]         │
└──────────────────────────────────────────────┘
```

### 워크트리 정리

세션이 보관되거나 삭제되면, 관련 git 워크트리를 정리할 수 있습니다:

```bash
# Claude Desktop can do this automatically, or you can do it manually:
git worktree list
# /Users/you/myapp               abcd123 [main]
# /tmp/claude-worktrees/myapp-a  ef45678 [feat/user-profiles]
# /tmp/claude-worktrees/myapp-b  90abcde [fix/payment-val]

# Remove a worktree
git worktree remove /tmp/claude-worktrees/myapp-b

# Prune stale worktree references
git worktree prune
```

---

## 11. 연습 문제

### 연습 1: 설치 및 첫 번째 세션

1. Claude Desktop 다운로드 및 설치
2. 프로젝트 디렉토리 열기
3. 대화 시작: "이 프로젝트의 구조를 설명해 주세요"
4. Claude가 로컬 파일을 읽고 정보를 제공하는 방식 관찰

### 연습 2: 병렬 세션

1. Claude Desktop에서 git 저장소 열기
2. 두 개의 병렬 세션 생성:
   - 세션 1: "웹사이트에 푸터 컴포넌트 추가"
   - 세션 2: "헤더 내비게이션 링크 수정"
3. 각 세션이 별도의 워크트리에서 작업함을 관찰
4. 변경 사항이 독립적인지 확인 (충돌 없음)
5. 두 세션의 브랜치를 병합

### 연습 3: 앱 미리보기

1. 웹 프로젝트 열기 (React, Vue, 또는 일반 HTML)
2. Claude에게 시각적 변경을 요청 (예: "배경을 다크 모드로 변경")
3. 앱 미리보기가 실시간으로 업데이트되는 것을 관찰
4. 의도적인 오류를 도입하고 다음을 관찰:
   - 오류를 보여주는 콘솔 로그
   - Claude가 자동 수정을 제안하는 것

### 연습 4: GitHub 통합

1. Claude Desktop 세션에서 PR 생성
2. 사이드바에서 CI 체크 모니터링
3. CI가 실패하면 [Auto-Fix CI] 기능 사용
4. 수정 사항이 푸시되기 전에 검토

### 연습 5: CLI + 데스크톱 워크플로

1. Claude Desktop에서 작업 시작 (예: "새 API 엔드포인트 생성")
2. Claude가 작업하는 동안 터미널을 열고 Claude Code CLI 사용
3. CLI로 보완적인 작업 수행 (예: "데이터베이스 마이그레이션 설정")
4. 두 도구의 변경 사항이 함께 작동하는지 확인

---

## 12. 참고 자료

- [Claude Desktop Download](https://claude.ai/download)
- [Claude Desktop Documentation](https://docs.anthropic.com/en/docs/claude-desktop)
- [Git Worktrees Documentation](https://git-scm.com/docs/git-worktree)
- [GitHub CLI (gh)](https://cli.github.com/)
- [Anthropic Blog: Claude Desktop Features](https://www.anthropic.com/news)

---

## 다음 단계

다음 레슨 [Cowork: AI 디지털 동료](./11_Cowork.md)에서는 Cowork를 살펴봅니다 — Claude를 자율적인 디지털 동료로 실행하는 Anthropic의 제품으로, 코딩을 넘어 프로젝트 관리, 문서 처리, 플러그인 및 MCP 커넥터를 통한 워크플로 자동화를 포함하는 더 넓은 작업을 처리합니다.
