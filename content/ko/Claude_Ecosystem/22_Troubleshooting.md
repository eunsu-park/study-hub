# 문제 해결 및 디버깅

**이전**: [21. 모범 사례와 패턴](./21_Best_Practices.md)

---

적절한 설정과 모범 사례를 따르더라도 Claude Code를 사용할 때 문제가 발생할 수 있습니다. 이 레슨은 가장 일반적인 문제들 -- 권한 오류와 훅 실패에서 컨텍스트 창 한계, MCP 연결 문제, API 오류까지 -- 을 진단하고 해결하는 체계적인 접근 방법을 제공합니다. 무언가 잘못될 때 참조 가이드로 사용하세요.

**난이도**: ⭐⭐

**사전 요구 사항**:
- 작동하는 Claude Code 설치 ([레슨 2](./02_Claude_Code_Getting_Started.md))
- 권한 모드(Permission Modes) 이해 ([레슨 4](./04_Permission_Modes.md))
- 훅(Hooks) 숙지 ([레슨 5](./05_Hooks.md))
- MCP 기본 지식 ([레슨 12](./12_Model_Context_Protocol.md))
- API 기초 이해 ([레슨 15](./15_Claude_API_Fundamentals.md))

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 권한 오류를 체계적으로 진단하고 해결
2. 훅 설정 및 실행 실패 디버깅
3. 컨텍스트 창 한계를 효과적으로 관리
4. MCP 서버 연결 문제 해결
5. 적절한 재시도 전략으로 API 오류 처리
6. 성능 문제 파악 및 해결
7. 내장 진단을 위한 `/doctor` 명령어 사용
8. 자체 진단이 충분하지 않을 때 도움을 구하는 곳 파악

---

## 목차

1. [권한 오류](#1-권한-오류)
2. [훅 실패](#2-훅-실패)
3. [컨텍스트 창 문제](#3-컨텍스트-창-문제)
4. [MCP 연결 문제](#4-mcp-연결-문제)
5. [API 오류](#5-api-오류)
6. [성능 문제](#6-성능-문제)
7. [도구 실행 문제](#7-도구-실행-문제)
8. [/doctor 명령어](#8-doctor-명령어)
9. [도움받을 곳](#9-도움받을-곳)
10. [문제 해결 결정 트리](#10-문제-해결-결정-트리)
11. [연습 문제](#11-연습-문제)

---

## 1. 권한 오류

권한 오류는 Claude Code를 처음 시작할 때 가장 흔한 문제입니다. 현재 권한 모드에서 허용하지 않는 도구를 Claude가 사용하려 할 때 발생합니다.

### 1.1 파일 작업에서 "Permission denied"

**증상**: Claude가 파일을 읽거나, 편집하거나, 생성할 수 없다고 보고합니다.

```
Error: Permission denied — cannot write to /path/to/file.py
```

**진단 체크리스트:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Permission Denied — 진단 단계                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 권한 모드 확인                                                   │
│     $ claude config get permission_mode                             │
│     → "plan-only"이면 Claude가 파일을 편집할 수 없음                │
│                                                                     │
│  2. 허용/거부 규칙 확인                                              │
│     $ cat .claude/settings.json                                     │
│     → 파일 경로와 일치하는 거부 규칙 찾기                            │
│                                                                     │
│  3. 파일 시스템 권한 확인                                            │
│     $ ls -la /path/to/file.py                                      │
│     → OS 수준에서 파일이 읽기 전용일 수 있음                         │
│                                                                     │
│  4. 파일이 제한된 디렉토리에 있는지 확인                             │
│     → 일부 디렉토리는 기본적으로 차단됨 (예: .git/)                 │
│                                                                     │
│  5. 설정 계층 구조 확인                                              │
│     ~/.claude/settings.json      (전역 — 엔터프라이즈)              │
│     .claude/settings.json        (프로젝트 — git에 체크인)          │
│     .claude/local_settings.json  (로컬 — git에 없음)                │
│     → 상위 수준 거부는 하위 수준 허용을 재정의함                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**해결책:**

```json
// .claude/settings.json — 허용 규칙 추가
{
  "permissions": {
    "allow": [
      "Edit:*",           // 모든 파일 편집 허용
      "Write:src/**",     // src/에서만 쓰기 허용
      "Bash:npm test",    // npm test 실행 허용
      "Bash:npm run *"    // npm run 명령어 허용
    ],
    "deny": [
      "Edit:.env*",       // env 파일 편집 금지
      "Write:.git/**",    // .git/에 쓰기 금지
      "Bash:rm -rf *"     // 재귀 삭제 금지
    ]
  }
}
```

```bash
# OS 수준 파일 권한 수정
chmod 644 /path/to/file.py

# 디렉토리 권한 수정
chmod 755 /path/to/directory/
```

### 1.2 "현재 권한 모드에서 도구를 사용할 수 없음"

**증상**: Claude가 현재 모드에서 도구를 사용할 수 없다고 말합니다.

```
The Bash tool is not allowed in plan-only mode.
```

**근본 원인**: 권한 모드가 사용 가능한 도구를 제한합니다.

```
┌───────────────────────────────────────────────────────────┐
│  권한 모드 → 사용 가능한 도구                               │
├────────────────┬──────────────────────────────────────────┤
│ plan-only      │ Read, Glob, Grep만 가능 (수정 없음)       │
│ default        │ 모든 도구 (승인 프롬프트 포함)             │
│ auto-accept    │ 허용 규칙에 맞는 모든 도구 (허용된 것은   │
│                │ 프롬프트 없음, 나머지는 프롬프트)          │
│ bypass         │ 모든 도구 (프롬프트 없음)                  │
└────────────────┴──────────────────────────────────────────┘
```

**해결책**: 적절한 권한 모드로 전환:

```bash
# 현재 모드 확인
claude config get permission_mode

# 현재 세션의 모드 전환
# (Claude Code 내 설정 메뉴 사용)
# 또는 설정에서 지정:
```

```json
// .claude/local_settings.json
{
  "permission_mode": "default"
}
```

### 1.3 설정 계층 구조 충돌

허용/거부 규칙이 무시되는 것처럼 보일 때 설정 계층 구조 전반의 충돌을 확인하세요:

```bash
# 권한에 영향을 줄 수 있는 모든 설정 파일 확인
cat ~/.claude/settings.json 2>/dev/null        # 엔터프라이즈/전역
cat .claude/settings.json 2>/dev/null           # 프로젝트 (공유)
cat .claude/local_settings.json 2>/dev/null     # 로컬 (개인)
```

**해결 우선순위**: 엔터프라이즈 > 프로젝트 > 로컬. 엔터프라이즈 설정에서 도구를 거부하면 프로젝트 및 로컬 설정이 해당 거부를 재정의할 수 없습니다.

---

## 2. 훅 실패

훅(Hook)은 Claude Code 생명주기의 특정 지점에서 외부 명령을 실행합니다. 실패하면 Claude가 작업을 완료하지 못할 수 있습니다.

### 2.1 훅 명령어를 찾을 수 없음

**증상:**

```
Hook error: command not found: my-lint-script
```

**진단:**

```bash
# 명령어가 존재하는지 확인
which my-lint-script

# 프로젝트의 스크립트인지 확인
ls -la .claude/scripts/my-lint-script

# 실행 권한이 있는지 확인
file .claude/scripts/my-lint-script
```

**일반적인 원인과 수정:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  훅 "Command Not Found" — 일반적인 원인                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 스크립트가 PATH에 없음                                           │
│     수정: 절대 경로 또는 ./상대/경로 사용                            │
│     "command": "/usr/local/bin/eslint"                              │
│     "command": "./.claude/scripts/lint.sh"                          │
│                                                                     │
│  2. 셔뱅(Shebang) 줄 누락                                           │
│     수정: #!/bin/bash 또는 #!/usr/bin/env python3 추가              │
│                                                                     │
│  3. 실행 권한 없음                                                   │
│     수정: chmod +x .claude/scripts/lint.sh                          │
│                                                                     │
│  4. 잘못된 셸 (bash vs zsh)                                         │
│     수정: 셔뱅에서 셸을 명시적으로 지정                              │
│     #!/bin/bash (macOS에서 sh != bash이므로 #!/bin/sh 사용 금지)    │
│                                                                     │
│  5. 훅 컨텍스트에서 Node/Python이 PATH에 없음                       │
│     수정: 전체 경로 사용: /usr/local/bin/node                        │
│     또는: /usr/bin/env node                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 훅 타임아웃

**증상**: Claude Code가 멈추고 훅 타임아웃을 보고합니다.

```
Hook timed out after 10000ms: pre-commit-check
```

**진단 및 수정:**

```bash
# 훅 명령어를 직접 실행하고 시간 측정
time ./.claude/scripts/pre-commit-check.sh

# 느린 경우 병목 지점 파악
# 일반적인 원인:
# - 훅에서 전체 테스트 슈트 실행 (일부만 실행해야 함)
# - 네트워크 호출 (API 검사, 패키지 다운로드)
# - 대용량 파일 처리
```

**해결책:**

```json
// 훅 설정에서 타임아웃 증가
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "./.claude/scripts/check.sh",
        "timeout": 30000  // 30초 (기본값 10초)
      }
    ]
  }
}
```

```bash
# 훅을 더 빠르게 만들기
# 모든 테스트를 실행하는 대신:
pytest tests/  # 느림 (30초)

# 빠르고 관련된 테스트만 실행:
pytest tests/unit/ -x --timeout=5  # 빠름 (2초)
```

### 2.3 오류를 반환하는 훅

**증상**: 훅이 실행되지만 0이 아닌 상태로 종료되어 Claude를 차단합니다.

```
Hook failed with exit code 1: lint-check
Output: src/utils.py:15:1: E302 expected 2 blank lines, found 1
```

**진단:**

```bash
# 훅 명령어를 직접 실행하여 전체 출력 확인
./.claude/scripts/lint-check.sh src/utils.py
echo $?  # 종료 코드 확인
```

**핵심 인사이트**: 0이 아닌 상태로 종료하는 훅은 Claude에 보고되며, 그러면 문제를 수정하고 재시도할 수 있습니다. 이것은 실제로 많은 훅(린팅 같은)의 의도된 동작입니다. 그러나 훅이 거짓 양성을 보고하고 있다면 훅 자체를 수정해야 합니다.

```bash
# 예시: 너무 엄격한 린트 훅
# 이전 (사소한 스타일 문제에서 실패):
#!/bin/bash
flake8 "$1" --max-line-length=79

# 이후 (AI 생성 코드에 더 합리적):
#!/bin/bash
flake8 "$1" --max-line-length=100 --ignore=E501,W503
```

### 2.4 단계별 훅 디버깅

```bash
# 1단계: 훅 설정 확인
cat .claude/settings.json | python3 -m json.tool

# 2단계: 어떤 훅이 실패하는지 파악
# 훅 스크립트에 디버그 출력 추가:
#!/bin/bash
echo "DEBUG: Hook started" >&2
echo "DEBUG: Args = $@" >&2
echo "DEBUG: Working directory = $(pwd)" >&2
echo "DEBUG: PATH = $PATH" >&2

# 실제 명령어 실행
eslint "$1"
EXIT_CODE=$?

echo "DEBUG: Exit code = $EXIT_CODE" >&2
exit $EXIT_CODE

# 3단계: 훅에 사용 가능한 환경 변수 확인
# 훅은 환경 변수를 통해 컨텍스트를 받습니다:
# $CLAUDE_FILE_PATH — 작업 중인 파일
# $CLAUDE_TOOL_NAME — 사용 중인 도구
# $CLAUDE_SESSION_ID — 현재 세션 식별자
```

---

## 3. 컨텍스트 창 문제

### 3.1 "Context too long" 오류

**증상**: Claude가 대화가 컨텍스트 창 한계를 초과했다고 보고합니다.

```
Error: Total token count exceeds model context window (200K tokens).
```

**즉각적인 해결책:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  컨텍스트 창 초과 — 빠른 수정                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. /compact                                                        │
│     핵심 컨텍스트를 보존하면서 대화 내역 압축                        │
│     현재 작업을 계속해야 할 때 최선                                  │
│                                                                     │
│  2. 새 세션 시작                                                     │
│     새로운 200K 컨텍스트 창                                          │
│     다른 작업으로 전환할 때 최선                                     │
│                                                                     │
│  3. 서브에이전트 사용                                                │
│     각 서브에이전트는 자체 컨텍스트 창을 가짐                        │
│     독립적인 연구나 병렬 작업에 최선                                 │
│                                                                     │
│  4. CLAUDE.md 크기 축소                                              │
│     필수적이지 않은 정보 제거                                        │
│     콘텐츠를 인라인으로 포함하는 대신 파일 링크                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 컨텍스트 저하 증상

하드 한계에 도달하기 전에 컨텍스트 품질이 점진적으로 저하됩니다:

```
┌─────────────────────────────────────────────────────────────────────┐
│  컨텍스트 저하 징조 (발생 순서)                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  초기 경고 징조:                                                     │
│  ⚠ Claude가 세션 초반에 이미 읽은 파일을 다시 읽음                  │
│  ⚠ Claude가 이미 답변한 질문을 다시 함                              │
│  ⚠ 응답이 세션 초반의 오래된 정보를 참조함                          │
│                                                                     │
│  중간 정도 저하:                                                     │
│  ⚠ Claude가 20개 이상 메시지 전에 내린 결정을 잊음                  │
│  ⚠ 생성된 코드가 이전 패턴과 모순됨                                 │
│  ⚠ Claude가 전체 작업 구조를 놓침                                   │
│                                                                     │
│  심각한 저하:                                                        │
│  ⚠ Claude가 다단계 지침을 따를 수 없음                              │
│  ⚠ 응답이 프로젝트 특화 대신 일반적이 됨                            │
│  ⚠ /compact이 더 이상 도움이 되지 않음 — 세션이 너무 늦어짐         │
│                                                                     │
│  조치: 초기 경고 징조에서 /compact 사용.                             │
│  중간 정도 저하에서 새 세션 시작.                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 예방 전략

```python
# 전략 1: 프롬프트를 집중적으로 유지
# 전체 코드베이스를 컨텍스트로 포함하는 대신
# 특정 파일을 참조:

# 나쁜 예: "여기 내 전체 프로젝트입니다: [50,000 토큰의 코드]"
# 좋은 예: "관련 파일은 src/auth/service.py와 src/auth/middleware.py입니다"

# 전략 2: CLAUDE.md를 현명하게 사용
# CLAUDE.md는 200줄 미만이어야 함
# 콘텐츠가 아닌 포인터 포함

# 전략 3: 대형 작업 분해
# 하나의 200개 메시지 세션 대신:
# 세션 1: 기능 계획 (10개 메시지)
# 세션 2: 백엔드 구현 (30개 메시지)
# 세션 3: 프론트엔드 구현 (30개 메시지)
# 세션 4: 테스트 및 문서 작성 (20개 메시지)

# 전략 4: 연구를 위한 서브에이전트 사용
# "서브에이전트를 사용하여 인증 모듈을 조사하고
#  내가 알아야 할 주요 인터페이스를 보고해주세요."
# 서브에이전트의 작업은 메인 컨텍스트 창을 소비하지 않음.
```

---

## 4. MCP 연결 문제

### 4.1 서버가 시작되지 않음

**증상**: Claude Code가 초기화될 때 MCP 서버가 시작되지 않습니다.

```
MCP error: Failed to start server "my-mcp-server"
```

**진단:**

```bash
# 1단계: MCP 설정 확인
cat .claude/mcp_settings.json | python3 -m json.tool

# 2단계: 서버를 수동으로 시작 시도
# stdio 전송의 경우:
node /path/to/mcp-server/index.js

# Python 서버의 경우:
python /path/to/mcp-server/main.py

# 3단계: 의존성 설치 여부 확인
cd /path/to/mcp-server
npm install  # 또는 pip install -r requirements.txt

# 4단계: 포트 충돌 확인 (HTTP 전송)
lsof -i :3001  # 포트가 이미 사용 중인지 확인
```

**일반적인 MCP 설정 문제:**

```json
// .claude/mcp_settings.json

{
  "mcpServers": {
    "my-database": {
      // 잘못됨: 상대 경로 (올바르게 해석되지 않을 수 있음)
      "command": "./mcp-servers/database/index.js",

      // 올바름: 절대 경로
      "command": "/home/user/project/mcp-servers/database/index.js",

      // 올바름: npm 패키지에 npx 사용
      "command": "npx",
      "args": ["-y", "@mcp/database-server"],

      // 서버에 필요한 환경 변수
      "env": {
        "DATABASE_URL": "postgresql://localhost:5432/mydb"
      }
    }
  }
}
```

### 4.2 전송 오류

**증상**: 서버가 시작되지만 통신이 실패합니다.

```
MCP transport error: unexpected end of stream
MCP transport error: invalid JSON-RPC message
```

**전송 유형별 진단:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  MCP 전송 디버깅                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  stdio 전송 (가장 일반적):                                           │
│  ├── 서버는 MCP 메시지 외에 stdout에 출력하면 안 됨                 │
│  ├── 디버그 로깅은 반드시 stderr로 가야 함                          │
│  ├── 확인: 서버가 시작 시 배너를 출력하는가?                        │
│  │   → stderr로 리다이렉트: console.error() 사용 (console.log() 금지)│
│  └── 확인: 서버가 stdin에서 올바르게 읽고 있는가?                   │
│                                                                     │
│  HTTP/SSE 전송:                                                      │
│  ├── 확인: 서버가 실행 중이고 수신 대기 중인가?                     │
│  │   → curl http://localhost:3001/health                           │
│  ├── 확인: 브라우저 기반이면 CORS 설정                              │
│  ├── 확인: HTTPS 엔드포인트의 SSL/TLS 인증서                        │
│  └── 확인: 프록시 또는 방화벽이 연결을 차단하는가                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**가장 일반적인 stdio 문제 수정:**

```javascript
// 잘못됨: stdout에 출력 (MCP 프로토콜 방해)
console.log("Server starting...");

// 올바름: stderr에 출력 (방해하지 않음)
console.error("Server starting...");

// 올바름: 적절한 로거 사용
const logger = {
  info: (msg) => process.stderr.write(`[INFO] ${msg}\n`),
  error: (msg) => process.stderr.write(`[ERROR] ${msg}\n`),
};
logger.info("Server starting...");
```

### 4.3 인증 실패

**증상**: MCP 서버가 인증으로 인해 연결을 거부합니다.

```
MCP error: Authentication failed for server "my-api"
```

**해결책:**

```json
// 환경 변수를 통해 인증 자격 증명 전달
{
  "mcpServers": {
    "my-api": {
      "command": "npx",
      "args": ["-y", "@mcp/api-server"],
      "env": {
        "API_KEY": "${env:MY_API_KEY}",
        "API_SECRET": "${env:MY_API_SECRET}"
      }
    }
  }
}
```

```bash
# Claude Code를 시작하기 전에 환경 변수가 설정되어 있는지 확인
export MY_API_KEY="your-api-key-here"
export MY_API_SECRET="your-api-secret-here"
claude  # 환경 변수를 사용 가능하게 하여 Claude Code 시작
```

### 4.4 MCP Inspector로 디버깅

MCP Inspector는 MCP 서버를 테스트하기 위한 진단 도구입니다:

```bash
# MCP Inspector 설치
npx @modelcontextprotocol/inspector

# MCP 서버를 대화식으로 테스트
# Inspector는 다음을 위한 UI를 제공합니다:
# - MCP 서버에 연결
# - 사용 가능한 도구, 리소스, 프롬프트 나열
# - 개별 도구 호출 실행
# - 요청/응답 메시지 검사
# - 오류 세부 정보 보기

# 특정 서버 테스트
npx @modelcontextprotocol/inspector node /path/to/your/server.js
```

### 4.5 일반적인 MCP 설정 실수

```json
// 실수 1: 인수가 필요한 명령에 "args" 누락
{
  "mcpServers": {
    "my-server": {
      "command": "python main.py --port 3001"  // 잘못됨: 명령 문자열에 인수 포함
    }
  }
}
// 수정:
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["main.py", "--port", "3001"]  // 올바름: 인수 분리
    }
  }
}

// 실수 2: 의존성 설치 잊음
// 수정: 설정 전에 npm install 또는 pip install 실행

// 실수 3: 잘못된 작업 디렉토리
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["server.js"],
      "cwd": "/correct/working/directory"  // 작업 디렉토리 지정
    }
  }
}
```

---

## 5. API 오류

### 5.1 401 Unauthorized (미인증)

**증상:**

```
Error: 401 Unauthorized — Invalid API key
```

**진단:**

```bash
# API 키가 설정되었는지 확인
echo $ANTHROPIC_API_KEY | head -c 10  # 처음 10자만 표시

# 키 형식 확인 (sk-ant-로 시작해야 함)
# 다른 공급자를 사용하는 경우 해당 키 형식 확인

# 일반적인 문제:
# - 키를 후행 공백과 함께 복사
# - 잘못된 환경의 키 (테스트 vs 프로덕션)
# - 키가 취소되거나 만료됨
# - 잘못된 셸 프로필에 설정됨 (.bashrc vs .zshrc)
```

**해결책:**

```bash
# API 키 올바르게 설정
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# 작동하는지 확인
curl -s https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "content-type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "Hi"}]
  }' | python3 -m json.tool

# 설정 파일을 사용하는 경우
# ~/.claude/config.json
{
  "api_key_source": "environment"  // 또는 "keychain"
}
```

### 5.2 429 Rate Limited (속도 제한)

**증상:**

```
Error: 429 Too Many Requests — Rate limit exceeded
```

**속도 제한 이해:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  속도 제한 유형                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 분당 요청 수 (RPM)                                               │
│     → 짧은 기간에 너무 많은 API 호출                                │
│     → 해결책: 요청 간격 조정, 배치 API 사용                         │
│                                                                     │
│  2. 분당 토큰 수 (TPM)                                               │
│     → 너무 빠르게 너무 많은 텍스트 전송                             │
│     → 해결책: 프롬프트 크기 줄이기, 캐싱 사용                       │
│                                                                     │
│  3. 일일 토큰 수 (TPD)                                               │
│     → 일일 할당량 소진                                               │
│     → 해결책: 토큰 사용량 최적화, 플랜 업그레이드                   │
│                                                                     │
│  응답의 속도 제한 헤더:                                               │
│  x-ratelimit-limit-requests: 100                                   │
│  x-ratelimit-remaining-requests: 23                                │
│  x-ratelimit-reset-requests: 2024-01-01T00:01:00Z                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**커스텀 코드에서 백오프(Backoff) 구현:**

```python
import time
import anthropic
from anthropic import RateLimitError

def call_with_backoff(client, max_retries=5, **kwargs):
    """속도 제한 시 지수적 백오프(Exponential Backoff)로 API 호출."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise  # 최대 재시도 후 포기

            # 지수적 백오프: 1초, 2초, 4초, 8초, 16초
            wait_time = 2 ** attempt
            print(f"속도 제한. 재시도 {attempt + 1} 전 {wait_time}초 대기...")
            time.sleep(wait_time)

    raise RuntimeError("최대 재시도 초과")

# 사용
client = anthropic.Anthropic()
response = call_with_backoff(
    client,
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Claude Code 사용자의 경우**: Claude Code는 내장 재시도 로직으로 속도 제한을 자동으로 처리합니다. 지속적인 속도 제한 오류가 발생하면 일반적으로 너무 많은 병렬 에이전트를 실행하고 있다는 의미입니다. 동시 서브에이전트 수를 줄이세요.

### 5.3 529 Overloaded (과부하)

**증상:**

```
Error: 529 — API is temporarily overloaded
```

이것은 속도 제한과 다릅니다 -- 모든 사용자에 걸쳐 API 자체가 높은 부하 상태임을 의미합니다.

**해결책:**

```python
import time
import random
import anthropic
from anthropic import APIStatusError

def call_with_jitter(client, max_retries=3, **kwargs):
    """과부하 오류에 대한 지터(Jitter)를 사용한 재시도."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except APIStatusError as e:
            if e.status_code == 529 and attempt < max_retries - 1:
                # 썬더링 허드(Thundering Herd) 방지를 위한 지터된 백오프
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"API 과부하. {wait_time:.1f}초 대기...")
                time.sleep(wait_time)
            else:
                raise
```

**Claude Code 사용자의 경우**: 몇 분 기다렸다가 다시 시도하세요. 문제가 지속되면 https://status.anthropic.com에서 서비스 상태를 확인하세요.

### 5.4 타임아웃 오류

**증상:**

```
Error: Request timed out after 60000ms
```

**일반적인 원인:**
- 매우 큰 입력 (컨텍스트 창 한계에 가까움)
- 처리하는 데 더 오래 걸리는 복잡한 추론 작업
- 네트워크 연결 문제

**해결책:**

```python
# 크거나 복잡한 요청에 대한 타임아웃 증가
client = anthropic.Anthropic(
    timeout=120.0  # 기본 60초 대신 2분
)

# 또는 요청별 타임아웃
response = client.messages.create(
    model="claude-opus-4-20250514",
    max_tokens=8192,
    messages=[{"role": "user", "content": very_long_prompt}],
    timeout=180.0  # 이 특정 요청에 3분
)
```

---

## 6. 성능 문제

### 6.1 느린 응답 시간

**진단:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  느린 응답 — 근본 원인 분석                                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  느린 것이 Claude의 응답인가요 아니면 도구 실행인가요?               │
│                                                                     │
│  Claude 응답이 느린 경우:                                            │
│  ├── 큰 입력 컨텍스트 → 컨텍스트 줄이기, 캐싱 사용                 │
│  ├── 복잡한 작업 → 예상됨; Opus가 Sonnet보다 느림                   │
│  ├── 확장 사고(Extended Thinking) 활성화 → 더 많은 시간 할당        │
│  └── 서비스 저하 → status.anthropic.com 확인                        │
│                                                                     │
│  도구 실행이 느린 경우:                                              │
│  ├── Bash 명령이 오래 걸림 → 실행 중인 것 확인                      │
│  ├── 대용량 파일 읽기 → 타겟팅된 읽기 사용                          │
│  ├── 큰 코드베이스에서 Grep → 더 구체적인 패턴 사용                 │
│  └── MCP 서버 응답 느림 → MCP 서버 디버그                           │
│                                                                     │
│  세션 시작이 느린 경우:                                              │
│  ├── 큰 CLAUDE.md → 필수 사항으로 다듬기                            │
│  ├── 많은 MCP 서버 → 필요한 것만 활성화                             │
│  └── 느린 네트워크 → 인터넷 연결 확인                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 높은 토큰 사용량

토큰 사용량 (및 비용)이 예상보다 높은 경우:

```
# 높은 토큰 사용량의 원인:
1. CLAUDE.md가 너무 큼 (모든 메시지와 함께 전송됨)
2. 일부분만 관련될 때 전체 파일 포함
3. 불필요한 단어가 포함된 장황한 프롬프트
4. 반복되는 컨텍스트에 프롬프트 캐싱을 사용하지 않음
5. Sonnet으로 충분할 때 Opus 사용
6. Claude가 이미 컨텍스트에 있는 파일을 다시 읽음

# 진단 단계:
# API 응답에서 사용량 확인
response = client.messages.create(...)
print(f"입력 토큰:  {response.usage.input_tokens}")
print(f"출력 토큰: {response.usage.output_tokens}")
# 입력 토큰이 지속적으로 높으면 컨텍스트 크기를 확인하세요
```

### 6.3 세션 시작 지연

```bash
# CLAUDE.md 크기 확인 (200줄 미만이어야 함)
wc -l CLAUDE.md

# MCP 서버 수 확인
cat .claude/mcp_settings.json | python3 -c "
import json, sys
config = json.load(sys.stdin)
servers = config.get('mcpServers', {})
print(f'구성된 MCP 서버: {len(servers)}')
for name in servers:
    print(f'  - {name}')
"

# 시작이 느린 경우 문제를 격리하기 위해
# 일시적으로 MCP 서버 비활성화 시도
```

---

## 7. 도구 실행 문제

### 7.1 Bash 명령 실패

**증상**: 터미널에서 작동하는 Bash 명령이 Claude가 실행할 때 실패합니다.

**일반적인 원인:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Bash 명령 실패 — 일반적인 원인                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 작업 디렉토리 불일치                                             │
│     Claude의 작업 디렉토리는 Bash 호출 간에 재설정됨                │
│     수정: 상대 경로가 아닌 절대 경로 사용                            │
│                                                                     │
│  2. 환경 변수 차이                                                   │
│     Claude의 셸이 전체 환경을 갖지 않을 수 있음                     │
│     수정: 변수를 명시적으로 설정하거나 .env 파일 사용               │
│                                                                     │
│  3. 셸 차이 (bash vs zsh)                                           │
│     일부 구문은 zsh에서 작동하지만 bash에서는 안 됨 (또는 반대)     │
│     수정: 가능한 경우 POSIX 호환 구문 사용                          │
│     참고: declare -A (bash)는 zsh에서 사용 불가                     │
│                                                                     │
│  4. 대화형 명령                                                      │
│     사용자 입력이 필요한 명령은 중단되거나 실패함                   │
│     수정: 비대화형 플래그 사용 (예: -y, --yes, --batch)            │
│                                                                     │
│  5. 타임아웃                                                         │
│     2분 이상 걸리는 명령은 타임아웃 됨                              │
│     수정: 오래 실행되는 명령에 --timeout 플래그 사용               │
│                                                                     │
│  6. 권한 거부                                                        │
│     명령이 허용 목록에 없음                                          │
│     수정: 설정의 허용 규칙에 추가                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 파일 편집 충돌

**증상**: Claude의 파일 편집이 예상한 내용과 일치하지 않아 실패합니다.

```
Error: old_string not found in file (content may have changed)
```

**원인:**
- Claude가 작업하는 동안 IDE에서 파일을 편집함
- 다른 도구 (포매터, 린터)가 파일을 수정함
- 이전 편집 후 훅이 파일을 수정함

**해결책:**

```
# Claude에게 파일을 다시 읽도록 지시
> "파일이 외부에서 수정되었습니다. src/app.py를 다시 읽고
>  편집을 다시 시도하세요."

# 자동 포매터를 사용하는 경우 Claude가 활발히 편집하는 동안
# 저장 시 실행되지 않도록 구성하거나, Claude의 훅을 편집이
# 완료된 후 포매터를 실행하도록 구성

# IDE가 외부 변경을 감지하도록 구성하여 충돌 방지:
# VS Code: "files.watcherExclude" 또는 일시적으로 자동 저장 비활성화
```

### 7.3 Git 작업 오류

**증상**: Git 명령이 예기치 않게 실패합니다.

```
error: Your local changes to the following files would be overwritten by merge
```

**일반적인 시나리오와 수정:**

```bash
# 시나리오 1: 커밋되지 않은 변경이 체크아웃 차단
# Claude가 브랜치를 전환하려 하지만 커밋되지 않은 작업이 있음
git stash  # 변경 사항 임시 저장
git checkout target-branch
git stash pop  # 변경 사항 복원

# 시나리오 2: 병합 충돌
# Claude는 명시적으로 요청하지 않는 한 자동으로 병합 충돌을
# 해결하면 안 됩니다
# Claude에게 다음과 같이 말하세요: "이 파일들에 병합 충돌이 있습니다.
# 해결하는 데 도움을 주세요."

# 시나리오 3: 분리된 HEAD 상태
git checkout main  # 브랜치로 돌아가기
# 또는 현재 상태에서 브랜치 생성:
git checkout -b recovery-branch
```

---

## 8. /doctor 명령어

Claude Code에는 일반적인 설정 문제를 확인하는 내장 진단 명령이 있습니다:

```
> /doctor

Claude Code 진단 보고서
==============================

✓ Claude Code 버전: 1.x.x (최신)
✓ API 키: 설정됨 (sk-ant-...xxxx)
✓ API 연결: 정상 (응답 시간: 145ms)
✓ 권한 모드: default
✓ CLAUDE.md: 발견됨 (142줄)
✓ 설정: .claude/settings.json 유효
⚠ MCP 서버: 3개 중 2개 정상
  ✓ filesystem: 연결됨
  ✓ database: 연결됨
  ✗ slack: 시작 실패 (command not found: npx)
✓ 훅: 2개 설정됨, 모두 유효
✓ Git: 저장소 감지됨, 깨끗한 작업 트리
✓ 셸: /bin/zsh
✓ Node.js: v20.11.0
✓ Python: 3.12.1
```

**다음의 경우 /doctor 사용:**
- 새 프로젝트를 시작할 때
- 설정을 변경한 후
- 원인을 알 수 없는 문제가 발생했을 때
- Claude Code를 업데이트한 후
- 새 팀원을 온보딩할 때

---

## 9. 도움받을 곳

### 9.1 문서

```
┌─────────────────────────────────────────────────────────────────────┐
│  문서 리소스                                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  공식 문서                                                           │
│  └── https://docs.anthropic.com                                    │
│      ├── Claude Code 가이드                                          │
│      ├── API 참조                                                    │
│      ├── MCP 사양                                                    │
│      └── 프롬프트 엔지니어링 가이드                                  │
│                                                                     │
│  GitHub                                                             │
│  └── https://github.com/anthropics/claude-code                     │
│      ├── 이슈: 버그 보고, 기존 이슈 검색                            │
│      ├── 토론: 질문하기, 패턴 공유                                  │
│      └── README: 설치 및 빠른 시작                                  │
│                                                                     │
│  MCP 사양                                                            │
│  └── https://modelcontextprotocol.io                               │
│      ├── 프로토콜 사양                                               │
│      ├── 서버 레지스트리                                             │
│      └── SDK 문서                                                    │
│                                                                     │
│  API 상태                                                            │
│  └── https://status.anthropic.com                                  │
│      ├── 현재 서비스 상태                                            │
│      ├── 인시던트 내역                                               │
│      └── 알림 구독                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 효과적인 버그 보고서 작성

GitHub에 이슈를 제출할 때 다음을 포함하세요:

```markdown
## 버그 보고서 템플릿

**환경:**
- Claude Code 버전: (claude --version)
- OS: macOS 15.3 / Ubuntu 24.04 / Windows 11
- 셸: zsh / bash
- Node.js 버전: (node --version)
- Python 버전: (python3 --version)

**설명:**
무슨 일이 있었나요? 무엇을 예상했나요?

**재현 단계:**
1. 이 구조의 프로젝트에서 Claude Code 시작: ...
2. 이 프롬프트 입력: ...
3. Claude가 이 도구를 실행: ...
4. 오류 발생: ...

**설정:**
- 권한 모드: default
- CLAUDE.md: [첨부하거나 요약]
- 설정: [.claude/settings.json 첨부]
- MCP 서버: [설정된 서버 나열]

**오류 출력:**
```
[전체 오류 메시지 여기에 붙여넣기]
```

**스크린샷 (해당하는 경우):**
[문제를 보여주는 스크린샷 첨부]
```

### 9.3 커뮤니티 리소스

```
- GitHub 토론: 질문하기, 팁 공유
  https://github.com/anthropics/claude-code/discussions

- Anthropic Discord: 실시간 커뮤니티 도움
  https://discord.gg/anthropic

- Stack Overflow: [claude-code] 또는 [anthropic] 태그로 질문
```

---

## 10. 문제 해결 결정 트리

무언가 잘못될 때 이 결정 트리를 사용하여 문제의 범주를 빠르게 파악하세요:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 문제 해결 결정 트리                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  어떤 종류의 오류가 발생했나요?                                       │
│  │                                                                  │
│  ├── "Permission denied" 또는 "not allowed"                        │
│  │   └── 1절로 이동: 권한 오류                                      │
│  │                                                                  │
│  ├── "Hook failed" 또는 "hook timeout"                              │
│  │   └── 2절로 이동: 훅 실패                                        │
│  │                                                                  │
│  ├── Claude가 컨텍스트를 잊거나 "context too long"                  │
│  │   └── 3절로 이동: 컨텍스트 창 문제                               │
│  │                                                                  │
│  ├── "MCP error" 또는 "server not starting"                         │
│  │   └── 4절로 이동: MCP 연결 문제                                  │
│  │                                                                  │
│  ├── HTTP 오류 (401, 429, 500, 529)                                 │
│  │   └── 5절로 이동: API 오류                                       │
│  │                                                                  │
│  ├── 느린 응답 또는 높은 비용                                        │
│  │   └── 6절로 이동: 성능 문제                                      │
│  │                                                                  │
│  ├── Bash/Edit/Git 명령 실패                                        │
│  │   └── 7절로 이동: 도구 실행 문제                                 │
│  │                                                                  │
│  ├── 불확실 / 여러 문제                                              │
│  │   └── /doctor 실행 (8절)                                         │
│  │                                                                  │
│  └── 위의 어느 것도 해당하지 않음                                    │
│      └── 9절로 이동: 도움받을 곳                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 11. 연습 문제

### 연습 1: 권한 설정 (초급)

다음을 수행하는 `.claude/settings.json`을 만드세요:
1. Claude가 `src/`와 `tests/`에 있는 파일만 편집하도록 허용
2. `npm test`, `npm run lint`, `npm run build` 실행 허용
3. `.env` 파일 편집 거부
4. `rm`, `docker`, `sudo` 명령 실행 거부

허용된 작업과 거부된 작업을 수행하도록 Claude에게 요청하여 설정을 테스트하세요.

### 연습 2: 훅 디버깅 (중급)

다음을 수행하는 훅 스크립트를 만드세요:
1. Claude가 편집하는 모든 파일에서 ESLint (또는 flake8) 실행
2. stderr에 디버그 로깅 포함
3. 린터가 설치되지 않은 경우를 처리
4. 15초 타임아웃 설정

훅에 의도적으로 버그를 도입하고 2.4절의 단계를 사용하여 디버깅을 연습하세요.

### 연습 3: 컨텍스트 관리 (중급)

Claude Code 세션을 시작하고 의도적으로 컨텍스트 한계에 가깝게 밀어붙이세요:
1. 5개의 큰 파일 읽기 (각 500줄 이상)
2. 코드에 관해 30개 메시지 대화
3. `/compact` 사용 후 무엇이 보존되는지 관찰
4. 대화를 계속하면서 컨텍스트 저하가 시작될 때 기록
5. 문서화: 저하 전까지 몇 개의 메시지? `/compact`이 도움이 되었나요?

### 연습 4: API 오류 처리 (고급)

다음을 수행하는 Python 스크립트를 작성하세요:
1. 속도 제한을 유발하기 위해 50번의 빠른 API 호출
2. 지터(Jitter)가 있는 지수적 백오프 구현
3. 타이밍 정보와 함께 각 재시도 시도 로깅
4. 성공률과 평균 응답 시간 보고

```python
# 시작 코드
import anthropic
import time

client = anthropic.Anthropic()
results = {"success": 0, "rate_limited": 0, "errors": 0}

# 여기에 구현하세요...
# 추적: 시도, 재시도, 총 시간, 성공률
```

### 연습 5: 전체 진단 (초급)

Claude Code 설치에서 `/doctor`를 실행하고:
1. 발견된 경고 또는 오류 문서화
2. 발견된 각 문제 수정
3. `/doctor`를 다시 실행하여 수정 사항 확인
4. 새 기기에서 Claude Code를 설정할 때 확인할 항목 체크리스트 생성

---

## 참고 자료

- Claude Code 문서 - https://docs.anthropic.com/en/docs/claude-code
- Anthropic API 참조 - https://docs.anthropic.com/en/api
- MCP 사양 - https://modelcontextprotocol.io
- Anthropic 상태 페이지 - https://status.anthropic.com
- GitHub 이슈 - https://github.com/anthropics/claude-code/issues

---

## 결론

이 레슨으로 Claude Ecosystem 토픽이 완성됩니다. 이제 설치부터 고급 워크플로우까지 Claude Code에 대한 포괄적인 이해와 문제가 발생했을 때를 위한 문제 해결 참조서를 갖추었습니다. 가장 효과적인 Claude Code 사용자들은 좋은 프롬프팅 습관 (레슨 21)과 체계적인 디버깅 스킬 (이 레슨)을 결합하여 생산적이고 효율적인 개발 세션을 유지합니다.

계속 학습하기 위해 새로운 시나리오를 만날 때마다 이전 레슨을 다시 방문하고, Claude Code가 계속 발전함에 따라 Anthropic 문서를 최신 상태로 유지하세요.
