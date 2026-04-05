# 훅과 이벤트 기반 자동화

**이전**: [04. 권한 모드와 보안](./04_Permission_Modes.md) | **다음**: [06. 스킬과 슬래시 명령어](./06_Skills_and_Slash_Commands.md)

---

훅(Hook)은 특정 이벤트가 발생할 때 Claude Code가 자동으로 실행하는 셸 명령어입니다. Claude가 파일을 편집하면 훅이 Prettier나 Black으로 자동 포맷팅합니다. Claude가 작업을 완료하면 알림을 트리거할 수 있습니다. Claude가 위험한 명령어를 실행하려 할 때 훅이 이를 가로채어 차단할 수 있습니다. 훅은 Claude Code를 인터랙티브 어시스턴트에서 이벤트 기반 자동화 파이프라인으로 변환하며 — 모델이 지침을 기억하는 것에 의존하지 않고 결정론적으로 동작합니다.

**난이도**: ⭐⭐

**선행 학습**:
- [03. CLAUDE.md와 프로젝트 설정](./03_CLAUDE_md_and_Project_Setup.md)
- [04. 권한 모드와 보안](./04_Permission_Modes.md)
- 셸 스크립팅 기초 (**Shell_Script** 토픽 참조)
- JSON 설정 파일에 대한 기초 지식

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 네 가지 훅 타입과 각각이 실행되는 시점 이해
2. 매처(matcher)와 명령어가 포함된 JSON 훅 설정 작성
3. 훅 컨텍스트 데이터에 접근하는 환경 변수 사용
4. 포맷팅, 린팅, 테스트, 알림을 위한 실용적인 훅 구축
5. 훅이 실패하거나 예상치 못한 결과를 낼 때 디버깅
6. 훅(결정론적)과 CLAUDE.md 지침(제안적)의 차이 구분

---

## 목차

1. [훅이란?](#1-훅이란)
2. [훅 타입](#2-훅-타입)
3. [설정 형식](#3-설정-형식)
4. [훅 매처](#4-훅-매처)
5. [환경 변수](#5-환경-변수)
6. [실용적인 예시](#6-실용적인-예시)
7. [훅 실행 흐름](#7-훅-실행-흐름)
8. [오류 처리와 디버깅](#8-오류-처리와-디버깅)
9. [훅 vs CLAUDE.md 지침](#9-훅-vs-claudemd-지침)
10. [고급 패턴](#10-고급-패턴)
11. [연습 문제](#11-연습-문제)
12. [다음 단계](#12-다음-단계)

---

## 1. 훅이란?

**훅**은 특정 이벤트가 발생할 때 Claude Code가 실행하는 셸 명령어입니다. 훅의 특징:

- **결정론적**: 이벤트가 발생하면 항상 실행됨 — 모델이 따를 수도 있고 따르지 않을 수도 있는 CLAUDE.md 지침과 달리
- **설정 가능**: 자연어가 아닌 설정 JSON 파일로 정의됨
- **범위 지정**: 특정 도구, 파일 경로, 명령어 패턴을 타겟으로 할 수 있음
- **비차단 또는 차단**: 일부 훅은 Claude의 작업 진행을 막을 수 있음

훅은 Git 훅이나 CI/CD 파이프라인 트리거처럼 생각하되, Claude Code의 도구 사용을 위한 것입니다.

```
훅 없이:
  Claude가 파일 편집 → 완료 (파일이 포맷팅되지 않을 수 있음)

자동 포맷 훅과 함께:
  Claude가 파일 편집 → 훅: prettier --write file → 완료 (파일이 포맷팅됨)
```

---

## 2. 훅 타입

Claude Code는 도구 실행 라이프사이클의 서로 다른 시점에 트리거되는 네 가지 훅 타입을 지원합니다.

### 훅 타입 개요

| 훅 타입 | 실행 시점 | 차단 가능? | 일반적인 용도 |
|---------|----------|-----------|--------------|
| **PreToolUse** | 도구 실행 전 | Yes | 유효성 검사, 위험한 명령어 차단 |
| **PostToolUse** | 도구 실행 후 | No | 포맷팅, 린팅, 알림 |
| **Notification** | Claude가 알림 전송 시 | No | 커스텀 알림 라우팅 |
| **Stop** | Claude가 턴 완료 시 | No | 최종 검사, 요약, 테스트 실행 |

### 라이프사이클 다이어그램

```
                    사용자가 메시지 전송
                           │
                           ▼
                    Claude가 도구 사용 결정
                           │
                    ┌──────┴──────┐
                    │ PreToolUse  │ ← 도구 차단 가능
                    │   훅        │
                    └──────┬──────┘
                           │
                    도구가 차단됨? ──Yes──▶ Claude가 조정
                           │
                          No
                           │
                    ┌──────┴──────┐
                    │  도구       │
                    │  실행       │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │ PostToolUse │ ← 포맷, 린트, 알림
                    │   훅        │
                    └──────┬──────┘
                           │
                    Claude가 계속하거나 완료
                           │
                    ┌──────┴──────┐
                    │   Stop      │ ← 최종 검사
                    │   훅        │
                    └─────────────┘
```

---

## 3. 설정 형식

훅은 설정 JSON 파일(`.claude/settings.json`, `~/.claude/settings.json`, 또는 `.claude/settings.local.json`)에서 설정됩니다.

### 기본 구조

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "<패턴>",
        "command": "<셸 명령어>"
      }
    ],
    "PostToolUse": [
      {
        "matcher": "<패턴>",
        "command": "<셸 명령어>"
      }
    ],
    "Notification": [
      {
        "command": "<셸 명령어>"
      }
    ],
    "Stop": [
      {
        "command": "<셸 명령어>"
      }
    ]
  }
}
```

### 설정 필드

| 필드 | 필수 여부 | 설명 |
|------|----------|------|
| `matcher` | 아니오* | 도구 이름 또는 파일 경로와 매칭하는 패턴. 생략하면 해당 타입의 모든 이벤트에 훅 실행. |
| `command` | 예 | 실행할 셸 명령어. 환경 변수를 통해 컨텍스트 수신. |

*매처는 `Notification`과 `Stop` 훅에는 적용되지 않습니다.

### 최소 예시

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "echo '파일이 편집됨: $CLAUDE_FILE_PATH'"
      }
    ]
  }
}
```

---

## 4. 훅 매처

매처는 어떤 도구 호출이 훅을 트리거할지 결정합니다. 도구 이름과 파일 경로 패턴을 지원합니다.

### 도구 이름 매처

| 매처 | 매칭 대상 |
|------|----------|
| `"Edit"` | 모든 파일 편집 |
| `"Write"` | 모든 파일 쓰기 (새 파일 생성) |
| `"Bash"` | 모든 Bash 명령어 실행 |
| `"Read"` | 모든 파일 읽기 |
| `"Glob"` | 모든 파일 검색 |
| `"Grep"` | 모든 내용 검색 |
| `"NotebookEdit"` | 모든 Jupyter 노트북 편집 |

### 복합 매처

동일한 이벤트 타입에 여러 훅을 정의할 수 있습니다:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "npx prettier --write $CLAUDE_FILE_PATH"
      },
      {
        "matcher": "Write",
        "command": "npx prettier --write $CLAUDE_FILE_PATH"
      },
      {
        "matcher": "Bash",
        "command": "echo '명령어 실행됨: $CLAUDE_BASH_COMMAND'"
      }
    ]
  }
}
```

### 파일 경로 매처

일부 훅은 글로브 패턴을 사용하여 파일 경로와 매칭할 수 있습니다:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit:*.py",
        "command": "black $CLAUDE_FILE_PATH"
      },
      {
        "matcher": "Edit:*.ts",
        "command": "npx prettier --write $CLAUDE_FILE_PATH"
      },
      {
        "matcher": "Edit:*.go",
        "command": "gofmt -w $CLAUDE_FILE_PATH"
      }
    ]
  }
}
```

---

## 5. 환경 변수

훅이 실행될 때 Claude Code는 이벤트에 대한 컨텍스트를 제공하는 환경 변수를 설정합니다.

### 사용 가능한 환경 변수

| 변수 | 사용 가능한 곳 | 설명 |
|------|--------------|------|
| `CLAUDE_FILE_PATH` | PostToolUse (Edit, Write) | 편집/작성된 파일의 절대 경로 |
| `CLAUDE_BASH_COMMAND` | PreToolUse, PostToolUse (Bash) | 실행될 또는 실행된 명령어 |
| `CLAUDE_TOOL_NAME` | 모든 훅 | 도구 이름 (Edit, Bash 등) |
| `CLAUDE_EXIT_CODE` | PostToolUse (Bash) | Bash 명령어의 종료 코드 |
| `CLAUDE_NOTIFICATION` | Notification | 알림 메시지 텍스트 |
| `CLAUDE_PROJECT_DIR` | 모든 훅 | 프로젝트 루트의 절대 경로 |

### 명령어에서 환경 변수 사용

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "echo \"파일 편집됨: $CLAUDE_FILE_PATH, 프로젝트: $CLAUDE_PROJECT_DIR\""
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "echo \"실행 예정: $CLAUDE_BASH_COMMAND\""
      }
    ]
  }
}
```

---

## 6. 실용적인 예시

### 예시 1: 편집 후 자동 포맷

가장 일반적인 훅: Claude가 파일을 편집한 후 자동으로 포맷팅합니다.

**JavaScript/TypeScript 프로젝트 (Prettier)**:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "npx prettier --write \"$CLAUDE_FILE_PATH\" 2>/dev/null || true"
      },
      {
        "matcher": "Write",
        "command": "npx prettier --write \"$CLAUDE_FILE_PATH\" 2>/dev/null || true"
      }
    ]
  }
}
```

**Python 프로젝트 (Black + isort)**:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit:*.py",
        "command": "black \"$CLAUDE_FILE_PATH\" && isort \"$CLAUDE_FILE_PATH\""
      }
    ]
  }
}
```

**Go 프로젝트 (gofmt)**:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit:*.go",
        "command": "gofmt -w \"$CLAUDE_FILE_PATH\""
      }
    ]
  }
}
```

### 예시 2: 편집 후 린터 실행

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit:*.py",
        "command": "ruff check \"$CLAUDE_FILE_PATH\" --fix --quiet"
      },
      {
        "matcher": "Edit:*.ts",
        "command": "npx eslint \"$CLAUDE_FILE_PATH\" --fix --quiet"
      }
    ]
  }
}
```

### 예시 3: 코드 변경 후 테스트 실행

소스 파일이 편집될 때마다 테스트 스위트를 실행합니다:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit:src/**",
        "command": "npm test --silent 2>&1 | tail -5"
      }
    ]
  }
}
```

> **참고**: 모든 편집에 대한 테스트 훅은 워크플로우를 상당히 느리게 만들 수 있으므로 주의하세요. 대신 Stop 이벤트에서만 테스트를 실행하는 것을 고려하세요.

### 예시 4: 커스텀 알림

Claude가 작업을 완료할 때 데스크톱 알림 전송:

**macOS**:

```json
{
  "hooks": {
    "Stop": [
      {
        "command": "osascript -e 'display notification \"Claude Code가 작업을 완료했습니다\" with title \"Claude Code\"'"
      }
    ],
    "Notification": [
      {
        "command": "osascript -e \"display notification \\\"$CLAUDE_NOTIFICATION\\\" with title \\\"Claude Code\\\"\""
      }
    ]
  }
}
```

**Linux (notify-send)**:

```json
{
  "hooks": {
    "Stop": [
      {
        "command": "notify-send 'Claude Code' '작업 완료'"
      }
    ]
  }
}
```

### 예시 5: 위험한 명령어 차단

PreToolUse 훅을 사용하여 위험한 명령어를 가로채고 차단합니다:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "case \"$CLAUDE_BASH_COMMAND\" in *'rm -rf /'*|*'dd if='*|*'mkfs'*|*': >'*) echo '차단됨: 위험한 명령어 감지됨' >&2; exit 1;; esac"
      }
    ]
  }
}
```

PreToolUse 훅이 0이 아닌 상태로 종료되면 도구 호출이 차단되고 Claude는 훅의 stderr 출력을 오류 메시지로 받습니다.

### 예시 6: Git 커밋 전 검사

Claude가 커밋하기 전에 검사를 실행합니다:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "if echo \"$CLAUDE_BASH_COMMAND\" | grep -q 'git commit'; then npm test --silent || (echo '커밋 전 테스트가 통과해야 합니다' >&2; exit 1); fi"
      }
    ]
  }
}
```

### 예시 7: 모든 작업 로깅

Claude가 하는 모든 것의 감사 추적 생성:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "command": "echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) | $CLAUDE_TOOL_NAME | $CLAUDE_FILE_PATH $CLAUDE_BASH_COMMAND\" >> /tmp/claude-audit.log"
      }
    ]
  }
}
```

---

## 7. 훅 실행 흐름

정확한 실행 흐름을 이해하면 올바른 훅을 작성하고 문제를 디버깅하는 데 도움이 됩니다.

### 실행 순서

동일한 이벤트에 여러 훅이 매칭되면 설정에 정의된 순서대로 실행됩니다:

```json
{
  "hooks": {
    "PostToolUse": [
      { "matcher": "Edit:*.py", "command": "isort $CLAUDE_FILE_PATH" },
      { "matcher": "Edit:*.py", "command": "black $CLAUDE_FILE_PATH" },
      { "matcher": "Edit:*.py", "command": "ruff check $CLAUDE_FILE_PATH" }
    ]
  }
}
```

`.py` 파일 편집 시 실행 순서:
1. isort (임포트 정렬)
2. black (코드 포맷팅)
3. ruff (오류 검사)

### 타이밍

```
훅 타이밍:

PreToolUse:   도구 실행 전에 실행. 0이 아닌 값으로 종료하면 도구가 차단됨.
              Claude는 훅의 stderr을 오류 메시지로 받음.

PostToolUse:  도구 완료 후 실행. 도구의 작업을 되돌릴 수 없음.
              출력은 Claude에게 컨텍스트로 제공됨.

Notification: Claude가 알림 메시지를 생성할 때 실행.
              Claude의 동작에 영향을 주지 않음.

Stop:         Claude가 응답 턴을 완료할 때 실행.
              출력은 다음 턴에 Claude에게 표시됨.
```

### 훅의 설정 병합

서로 다른 설정 파일의 훅은 병합됩니다 (재정의되지 않음):

```
전역 훅 (~/.claude/settings.json):
  PostToolUse: [format_hook]

프로젝트 훅 (.claude/settings.json):
  PostToolUse: [lint_hook]

로컬 훅 (.claude/settings.local.json):
  PostToolUse: [notify_hook]

유효 훅:
  PostToolUse: [format_hook, lint_hook, notify_hook]
```

---

## 8. 오류 처리와 디버깅

### 훅 실패

훅 명령어가 실패할 때 (0이 아닌 상태로 종료):

| 훅 타입 | 실패 시 동작 |
|---------|------------|
| **PreToolUse** | 도구 호출 **차단**; Claude가 오류 메시지 수신 |
| **PostToolUse** | 오류 기록; Claude에게 알려지지만 편집은 유지 |
| **Notification** | 오류가 조용히 기록됨 |
| **Stop** | 오류 기록; 출력이 다음 턴에 Claude에게 표시됨 |

### 일반적인 오류

**명령어를 찾을 수 없음**:

```json
// 문제: prettier가 전역으로 설치되지 않음
{ "command": "prettier --write $CLAUDE_FILE_PATH" }

// 해결: npx 또는 전체 경로 사용
{ "command": "npx prettier --write $CLAUDE_FILE_PATH" }
// 또는
{ "command": "./node_modules/.bin/prettier --write $CLAUDE_FILE_PATH" }
```

**파일 경로에 따옴표 누락**:

```json
// 문제: 공백이 있는 경로에서 오류 발생
{ "command": "black $CLAUDE_FILE_PATH" }

// 해결: 변수를 항상 따옴표로 감싸기
{ "command": "black \"$CLAUDE_FILE_PATH\"" }
```

**훅이 잘못된 파일 타입에 실행됨**:

```json
// 문제: prettier가 바이너리 파일에서 실패
{ "matcher": "Edit", "command": "npx prettier --write $CLAUDE_FILE_PATH" }

// 해결: 원하는 파일 타입만 매칭
{ "matcher": "Edit:*.{ts,tsx,js,jsx,json,css,md}", "command": "npx prettier --write $CLAUDE_FILE_PATH" }
```

### 디버깅 기법

**1. echo 문 추가**:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "echo \"디버그: $CLAUDE_TOOL_NAME 도구로 $CLAUDE_FILE_PATH 편집 중\" && npx prettier --write \"$CLAUDE_FILE_PATH\""
      }
    ]
  }
}
```

**2. 파일에 로그**:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "command": "echo \"$(date): $CLAUDE_TOOL_NAME $CLAUDE_FILE_PATH\" >> /tmp/claude-hooks.log"
      }
    ]
  }
}
```

**3. 오류를 우아하게 억제**:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "npx prettier --write \"$CLAUDE_FILE_PATH\" 2>/dev/null || true"
      }
    ]
  }
}
```

`|| true`는 prettier가 오류(예: 지원되지 않는 파일 타입)를 만나도 훅이 실패를 보고하지 않도록 합니다.

---

## 9. 훅 vs CLAUDE.md 지침

이 구분은 효과적인 Claude Code 설정에 있어 핵심적입니다. 훅과 CLAUDE.md는 서로 다른 역할을 하며 서로 다른 목적으로 사용되어야 합니다.

### 비교표

| 측면 | 훅 | CLAUDE.md |
|------|----|-----------|
| **성격** | 결정론적 자동화 | 자연어 제안 |
| **적용** | 트리거될 때 항상 실행 | 모델이 따를 수도 있고 아닐 수도 있음 |
| **범위** | 특정 도구 이벤트 | 일반 프로젝트 컨텍스트 |
| **언어** | 셸 명령어 (JSON 설정) | 마크다운 (자연어) |
| **유연성** | 엄격하고 정확한 동작 | 컨텍스트에 따라 적응 가능 |
| **예시** | 코드 포맷, 린터 실행 | 코딩 스타일 선호도 |

### 훅을 사용해야 할 때

```
반드시 발생해야 하는 것에 훅 사용:
  ✓ 코드 포맷팅 (prettier, black, gofmt)
  ✓ 임포트 정렬 (isort, organize-imports)
  ✓ 위험한 명령어 차단
  ✓ 감사 로깅
  ✓ 데스크톱 알림
  ✓ 커밋 전 유효성 검사
```

### CLAUDE.md를 사용해야 할 때

```
발생해야 하는 것에 CLAUDE.md 사용:
  ✓ 코딩 스타일 선호도
  ✓ 아키텍처 가이드라인
  ✓ 명명 규칙
  ✓ 테스트 작성 패턴
  ✓ API 설계 관례
  ✓ 문서화 표준
```

### 결합 전략

최선의 접근 방식은 둘 모두를 사용하는 것입니다:

```markdown
# CLAUDE.md
## 코드 스타일
- 4칸 들여쓰기 사용
- 함수 이름은 snake_case
- 함수 시그니처에 항상 타입 힌트 추가
```

```json
// .claude/settings.json — 훅
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit:*.py",
        "command": "black --line-length 100 \"$CLAUDE_FILE_PATH\" && isort \"$CLAUDE_FILE_PATH\""
      }
    ]
  }
}
```

CLAUDE.md는 Claude에게 **어떤** 스타일로 작성할지 알려줍니다. 훅은 Claude의 초기 출력이 약간 벗어나더라도 출력이 규칙을 준수하도록 **보장**합니다.

---

## 10. 고급 패턴

### 조건부 훅

파일이나 프로젝트 상태에 따라 다른 명령어를 실행합니다:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "case \"$CLAUDE_FILE_PATH\" in *.py) black \"$CLAUDE_FILE_PATH\";; *.ts|*.tsx) npx prettier --write \"$CLAUDE_FILE_PATH\";; *.go) gofmt -w \"$CLAUDE_FILE_PATH\";; esac"
      }
    ]
  }
}
```

### 훅 스크립트

복잡한 훅의 경우 인라인 명령어 대신 외부 스크립트를 사용하세요:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": ".claude/hooks/post-edit.sh"
      }
    ]
  }
}
```

`.claude/hooks/post-edit.sh`:

```bash
#!/bin/bash
set -euo pipefail

FILE="$CLAUDE_FILE_PATH"
EXT="${FILE##*.}"

case "$EXT" in
    py)
        echo "Python 포맷팅: $FILE"
        black "$FILE" 2>/dev/null
        isort "$FILE" 2>/dev/null
        ruff check "$FILE" --fix --quiet 2>/dev/null || true
        ;;
    ts|tsx|js|jsx)
        echo "TypeScript/JavaScript 포맷팅: $FILE"
        npx prettier --write "$FILE" 2>/dev/null
        npx eslint --fix "$FILE" 2>/dev/null || true
        ;;
    go)
        echo "Go 포맷팅: $FILE"
        gofmt -w "$FILE"
        ;;
    rs)
        echo "Rust 포맷팅: $FILE"
        rustfmt "$FILE" 2>/dev/null || true
        ;;
    *)
        echo ".$EXT 파일에 대한 포매터가 설정되지 않음"
        ;;
esac
```

실행 권한 부여:

```bash
chmod +x .claude/hooks/post-edit.sh
```

### 최종 유효성 검사를 위한 Stop 훅

Claude가 완료된 후 포괄적인 검사를 실행합니다:

```json
{
  "hooks": {
    "Stop": [
      {
        "command": ".claude/hooks/final-check.sh"
      }
    ]
  }
}
```

`.claude/hooks/final-check.sh`:

```bash
#!/bin/bash

echo "=== 최종 유효성 검사 ==="

# 커밋되지 않은 변경사항 확인
if ! git diff --quiet; then
    echo "경고: 커밋되지 않은 변경사항이 감지됨"
    git diff --stat
fi

# 빠른 린터 검사 실행
if command -v npx &>/dev/null && [ -f "package.json" ]; then
    echo "린트 검사 중..."
    npx eslint src/ --quiet 2>/dev/null && echo "린트: 통과" || echo "린트: 문제 발견됨"
fi

# 테스트 상태 확인
if [ -f "package.json" ] && grep -q '"test"' package.json; then
    echo "테스트 실행 중..."
    npm test --silent 2>&1 | tail -3
fi
```

---

## 11. 연습 문제

### 연습 1: 기본 훅 설정

다음을 수행하는 훅이 포함된 `.claude/settings.json` 생성:

1. 편집 후 Python 파일을 `black`으로 자동 포맷
2. 편집 후 JavaScript 파일을 `prettier`로 자동 포맷
3. Claude가 실행하는 모든 Bash 명령어를 `/tmp/claude-commands.log`에 로깅
4. Claude가 작업 완료 시 macOS 알림 전송

### 연습 2: 커밋 전 유효성 검사 훅

다음을 수행하는 PreToolUse 훅 작성:

1. Claude가 `git commit`을 실행하려 할 때 감지
2. 커밋 허용 전 `npm test` 실행
3. 테스트 실패 시 커밋 차단
4. 테스트 통과 시 커밋 허용

### 연습 3: 다중 언어 포맷 스크립트

다음을 수행하는 `.claude/hooks/format.sh` 스크립트 생성:

1. `$CLAUDE_FILE_PATH`에서 파일 확장자 감지
2. 적절한 포매터 실행 (black, prettier, gofmt, rustfmt)
3. 포매터가 없는 경우 우아하게 처리 (포매터가 설치되지 않아도 오류 없음)
4. 수행한 작업을 `/tmp/claude-format.log`에 로깅

Edit과 Write 이벤트 모두에 대한 PostToolUse 훅으로 설정하세요.

### 연습 4: 깨진 훅 디버깅

다음 훅 설정에서 세 가지 버그를 찾아 수정하세요:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "prettier --write $CLAUDE_FILE_PATH"
      },
      {
        "matcher": "Bash",
        "command": "echo 'Ran: $CLAUDE_BASH_COMMAND' >> ~/claude.log"
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Edit",
        "command": "if [ ! -f $CLAUDE_FILE_PATH ]; then echo File not found; exit 1; fi"
      }
    ]
  }
}
```

---

## 12. 다음 단계

훅은 Claude Code 도구 사용에 대한 결정론적 자동화를 제공합니다. 그러나 자동화는 전체 그림의 일부에 불과합니다 — 특정 작업에 대해 Claude가 따르는 재사용 가능한 **지침**도 필요합니다. 다음 레슨에서는 **스킬과 슬래시 명령어**를 다룹니다 — 커스텀 지침, 워크플로우, 관례를 하나의 명령어로 호출할 수 있는 재사용 가능한 단위로 패키징하는 시스템입니다.

**다음**: [06. 스킬과 슬래시 명령어](./06_Skills_and_Slash_Commands.md)
