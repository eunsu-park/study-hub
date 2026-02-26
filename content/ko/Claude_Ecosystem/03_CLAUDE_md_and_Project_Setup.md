# CLAUDE.md와 프로젝트 설정

**이전**: [02. Claude Code: 시작하기](./02_Claude_Code_Getting_Started.md) | **다음**: [04. 권한 모드와 보안](./04_Permission_Modes.md)

---

CLAUDE.md는 Claude Code 생태계에서 가장 중요한 설정 파일입니다. 프로젝트 루트에 위치하는 일반 마크다운 파일로, Claude에게 프로젝트별 컨텍스트를 제공합니다 — 코딩 표준, 아키텍처 결정사항, 테스트 절차, 배포 노트, 그리고 Claude가 코드베이스를 효과적으로 다루는 데 도움이 되는 모든 정보를 담습니다. 이 레슨에서는 CLAUDE.md를 심층적으로 다루며, `.claude/` 디렉토리 구조와 설정 계층 구조까지 함께 설명합니다.

**난이도**: ⭐

**선행 학습**:
- [02. Claude Code: 시작하기](./02_Claude_Code_Getting_Started.md)
- 마크다운 문법 기초 지식
- Claude Code로 설정할 프로젝트

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. CLAUDE.md가 존재하는 이유와 Claude가 이를 활용하는 방법 이해
2. 다양한 프로젝트 유형에 맞는 효과적인 CLAUDE.md 작성
3. `.claude/` 디렉토리 구조 설정
4. 설정 계층 구조(전역, 프로젝트, 로컬) 탐색
5. 커밋 대상과 gitignore 대상에 대한 올바른 결정
6. 팀 공유 Claude Code 설정의 모범 사례 적용

---

## 목차

1. [CLAUDE.md가 중요한 이유](#1-claudemd가-중요한-이유)
2. [Claude의 CLAUDE.md 읽기 방식](#2-claude의-claudemd-읽기-방식)
3. [CLAUDE.md 구조와 섹션](#3-claudemd-구조와-섹션)
4. [실전 CLAUDE.md 예시](#4-실전-claudemd-예시)
5. [.claude/ 디렉토리](#5-claude-디렉토리)
6. [설정 계층 구조](#6-설정-계층-구조)
7. [커밋 대상 vs Gitignore 대상](#7-커밋-대상-vs-gitignore-대상)
8. [새 프로젝트 초기화](#8-새-프로젝트-초기화)
9. [CLAUDE.md vs 설정 파일: 상황에 따른 선택](#9-claudemd-vs-설정-파일-상황에-따른-선택)
10. [모범 사례](#10-모범-사례)
11. [연습 문제](#11-연습-문제)
12. [다음 단계](#12-다음-단계)

---

## 1. CLAUDE.md가 중요한 이유

CLAUDE.md 없이는 Claude Code가 매 세션을 프로젝트 관련 지식이 전혀 없는 상태로 시작합니다. 코드는 읽을 수 있지만 다음 사항을 알지 못합니다:

- 코딩 표준 (탭 vs 스페이스, 명명 규칙)
- 테스트 실행 방법 (`pytest`? `npm test`? `make test`?)
- 어떤 디렉토리에 무엇이 있는지 (`src/`가 메인 소스인지? 아니면 `lib/`인지?)
- 아키텍처 결정사항과 그 근거
- 배포 절차와 환경 세부사항
- 따라야 할 패턴과 피해야 할 패턴

CLAUDE.md는 이 간극을 메웁니다. Claude를 범용 AI 어시스턴트에서 프로젝트 관례를 따르는 컨텍스트 인식 팀원으로 변환시켜 줍니다.

### CLAUDE.md 유무 비교

**CLAUDE.md 없이** — Claude가 일반적인 패턴을 바탕으로 추측합니다:

```
> 사용자 프로필용 새 API 엔드포인트 추가

Claude는 다음과 같이 행동할 수 있습니다:
- 사용하지 않는 프레임워크를 사용
- 파일을 잘못된 디렉토리에 배치
- 프로젝트와 다른 명명 규칙 사용
- 프로젝트의 유효성 검사 패턴 무시
- 기대하는 테스트를 실행하지 않음
```

**CLAUDE.md 있이** — Claude가 정확한 관례를 따릅니다:

```markdown
# CLAUDE.md 내용:
## API 관례
- 라우트는 `src/routes/<resource>.ts`에 위치
- 요청 유효성 검사에 Zod 사용
- 모든 엔드포인트는 `tests/routes/`에 대응하는 테스트 필요
- RESTful 명명 준수: GET /api/users/:id, POST /api/users
- 라우트 변경 후 `npm test` 실행
```

```
> 사용자 프로필용 새 API 엔드포인트 추가

Claude는 다음을 수행합니다:
- src/routes/profiles.ts 생성 (올바른 디렉토리)
- Zod 스키마 사용 (프로젝트의 유효성 검사 라이브러리)
- tests/routes/profiles.test.ts 생성
- RESTful 패턴 준수
- npm test로 검증 실행
```

---

## 2. Claude의 CLAUDE.md 읽기 방식

Claude Code 세션을 시작하면 다음 과정이 진행됩니다:

```
1. Claude Code가 프로젝트 루트를 감지
2. 프로젝트 루트의 CLAUDE.md 읽기 (존재하는 경우)
3. 상위 디렉토리의 CLAUDE.md 파일 읽기 (모노레포의 경우)
4. 내용을 시스템 프롬프트에 포함
5. Claude가 세션 전체에서 해당 지침 사용
```

### CLAUDE.md 탐색 순서

```
~/projects/my-monorepo/          ← CLAUDE.md (먼저 읽음)
├── packages/
│   ├── frontend/                ← CLAUDE.md (이후 읽음)
│   │   └── src/
│   └── backend/                 ← CLAUDE.md (이후 읽음)
│       └── src/
└── shared/                      ← CLAUDE.md 없음
```

`~/projects/my-monorepo/packages/frontend/`에서 `claude`를 실행하면 Claude는 다음을 읽습니다:
1. `~/projects/my-monorepo/CLAUDE.md` (상위)
2. `~/projects/my-monorepo/packages/frontend/CLAUDE.md` (현재)

지침이 병합되며, 더 구체적인(더 깊은) 지침이 우선 적용됩니다.

### 토큰 예산(Token Budget)

CLAUDE.md 내용은 컨텍스트 윈도우의 토큰을 소비합니다. 일반적인 CLAUDE.md는 500~3,000 토큰을 사용하며, 이는 200K 윈도우의 극히 일부입니다. 그러나 극단적으로 긴 CLAUDE.md(10,000+ 토큰)는 낭비가 될 수 있습니다. 간결하고 관련성 있게 유지하세요.

---

## 3. CLAUDE.md 구조와 섹션

잘 정리된 CLAUDE.md는 일관된 구조를 따릅니다. 다음은 권장 템플릿입니다:

```markdown
# 프로젝트명

프로젝트가 수행하는 작업과 목적에 대한 간략한 설명.

## 프로젝트 구조

```
src/
├── routes/     # API 엔드포인트
├── services/   # 비즈니스 로직
├── models/     # 데이터베이스 모델
├── middleware/  # Express 미들웨어
└── utils/      # 공유 유틸리티
tests/
├── unit/       # 단위 테스트
├── integration/ # 통합 테스트
└── fixtures/   # 테스트 데이터
```

## 기술 스택

- **런타임**: TypeScript 5.3을 사용하는 Node.js 20
- **프레임워크**: Express 4.18
- **데이터베이스**: Prisma ORM을 사용하는 PostgreSQL 16
- **테스트**: Jest + Supertest
- **린팅**: ESLint + Prettier

## 개발 명령어

```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev

# 모든 테스트 실행
npm test

# 특정 테스트 파일 실행
npm test -- tests/unit/auth.test.ts

# 린트 및 포맷
npm run lint
npm run format

# 프로덕션 빌드
npm run build
```

## 코딩 표준

- TypeScript strict 모드 사용 — `any` 타입 금지
- `let` 대신 `const` 선호; `var` 절대 사용 금지
- 함수는 30줄 미만; 더 길면 헬퍼로 분리
- 기본 내보내기 대신 명명된 내보내기(named exports) 사용
- 오류 처리: `src/errors/`의 커스텀 오류 클래스 항상 사용

## API 관례

- RESTful 라우트: `GET /api/resource`, `POST /api/resource`
- 모든 요청 본문은 Zod 스키마로 유효성 검사
- 응답 형식: `{ data: T, meta?: { pagination } }`
- Authorization 헤더의 JWT를 통한 인증
- 모든 공개 엔드포인트에 속도 제한 적용

## 테스트 요구사항

- 모든 새 기능에 단위 테스트 필수
- API 엔드포인트에 통합 테스트
- 테스트 파일은 소스 구조를 반영: `src/foo.ts` → `tests/unit/foo.test.ts`
- 최소 코드 커버리지 80%
- 커밋 전 `npm test` 실행

## Git 관례

- 브랜치 명명: `feature/description`, `fix/description`, `chore/description`
- 커밋 메시지: conventional commits (feat:, fix:, chore:, docs:)
- 머지 전 항상 main 기반으로 리베이스
- 기능 브랜치는 스쿼시 머지
```

### 섹션별 설명

| 섹션 | 목적 | 중요한 이유 |
|------|------|------------|
| **프로젝트 구조** | 디렉토리 레이아웃 | Claude가 올바른 위치에 파일 배치 |
| **기술 스택** | 언어, 프레임워크, 버전 | Claude가 올바른 API와 패턴 사용 |
| **개발 명령어** | 빌드, 테스트, 실행 방법 | Claude가 올바른 명령어 실행 |
| **코딩 표준** | 스타일 규칙, 패턴 | Claude가 관례 준수 |
| **API 관례** | 엔드포인트 설계 규칙 | Claude가 일관된 API 생성 |
| **테스트 요구사항** | 테스트 대상과 방법 | Claude가 적절한 테스트 작성 |
| **Git 관례** | 커밋과 브랜치 규칙 | Claude가 올바른 커밋 생성 |

---

## 4. 실전 CLAUDE.md 예시

### Python 데이터 사이언스 프로젝트

```markdown
# Data Pipeline

여러 거래소의 금융 데이터를 처리하는 ETL 파이프라인.

## 프로젝트 구조

```
pipeline/
├── extractors/    # 데이터 소스 커넥터
├── transformers/  # 데이터 정제 및 변환
├── loaders/       # 데이터베이스 및 파일 작성기
├── models/        # SQLAlchemy 모델
├── config/        # 환경별 설정
└── tests/
```

## 환경

- Python 3.12, pyenv로 관리
- 의존성: pip + requirements.txt (poetry 사용 안 함)
- 가상 환경: `source .venv/bin/activate`

## 명령어

```bash
python -m pytest tests/ -v          # 테스트 실행
python -m pytest tests/ --cov=pipeline  # 커버리지 포함
python -m mypy pipeline/            # 타입 검사
python -m ruff check pipeline/      # 린팅
python -m ruff format pipeline/     # 포맷팅
```

## 코드 스타일

- 100자 줄 제한의 PEP 8
- 모든 함수 시그니처에 타입 힌트 필수
- 독스트링(Docstring): Google 스타일
- `os.path` 대신 `pathlib.Path` 사용
- 프로덕션 코드에서 `print()` 대신 `logging` 모듈 사용
- DTO에는 데이터클래스(Dataclasses), 유효성 검사에는 Pydantic 사용

## 중요 사항

- `.env` 파일 또는 API 키는 절대 커밋 금지
- `extractors/` 모듈은 속도 제한된 API 호출 사용 — 항상 `@retry` 데코레이터 사용
- 데이터베이스 마이그레이션은 Alembic 사용: `alembic upgrade head`
- CI에서 mypy + ruff + pytest 실행; 모두 통과해야 함
```

### TypeScript 모노레포

```markdown
# Commerce Platform

이커머스 플랫폼 모노레포 (웹, API, 공유 패키지).

## 모노레포 구조

```
packages/
├── web/          # Next.js 프론트엔드 (포트 3000)
├── api/          # NestJS 백엔드 (포트 4000)
├── shared/       # 공유 타입 및 유틸리티
├── ui/           # 공유 React 컴포넌트 라이브러리
└── config/       # 공유 ESLint, TSConfig 등
```

## 패키지 매니저

- pnpm (npm 또는 yarn 사용 금지)
- 루트에서 `pnpm install`
- `pnpm --filter @commerce/web dev`로 특정 패키지 실행

## 핵심 규칙

1. 공유 타입은 `packages/shared/types/`에 위치 — 타입 중복 절대 금지
2. API 응답은 `shared/types/api.ts`의 타입과 일치해야 함
3. UI 컴포넌트에는 Storybook 스토리 필수
4. 모든 패키지는 `packages/config/`의 공유 ESLint 설정 사용
5. 데이터베이스 쿼리는 Prisma 사용 — 마이그레이션에만 Raw SQL 허용

## 테스트

```bash
pnpm test                           # 모든 패키지
pnpm --filter @commerce/api test    # API만
pnpm --filter @commerce/web e2e     # E2E 테스트 (Playwright)
```
```

### C/C++ 임베디드 프로젝트

```markdown
# Sensor Controller Firmware

STM32F4 센서 컨트롤러 보드 펌웨어.

## 빌드 시스템

- CMake 3.25+
- ARM GCC 툴체인: `arm-none-eabi-gcc`
- 빌드: `mkdir build && cd build && cmake .. && make`
- 플래시: `make flash` (ST-Link 필요)

## 코드 스타일

- C11 표준, `-Wall -Wextra -Werror`로 컴파일
- 4칸 들여쓰기, 탭 금지
- 함수와 변수에 snake_case
- 매크로와 상수에 UPPER_SNAKE_CASE
- 하드웨어 레지스터에 헝가리안 표기법: `reg_`, `pin_`, `irq_`

## 메모리 제약

- 플래시: 총 512KB, 380KB 사용 가능
- RAM: 총 128KB, 96KB 사용 가능
- 스택 크기: 태스크당 4KB
- 동적 메모리 할당(malloc/free) 절대 사용 금지
- 모든 버퍼는 정적으로 할당

## 중요 사항

- ISR(인터럽트 서비스 루틴) 함수는 50줄 미만이며 비블로킹이어야 함
- `drivers/`의 HAL 레이어 사용 — 레지스터 직접 접근 금지
- FreeRTOS 태스크는 `tasks/`에 위치, 각각 전용 .c/.h 쌍 사용
```

---

## 5. .claude/ 디렉토리

CLAUDE.md 외에도 Claude Code는 추가 설정을 위해 `.claude/` 디렉토리를 사용합니다.

### 디렉토리 구조

```
.claude/
├── settings.json        # 프로젝트 수준 설정 (커밋)
├── settings.local.json  # 로컬 재정의 (gitignore)
└── skills/              # 프로젝트별 스킬 (커밋)
    ├── commit.md
    └── review.md
```

### settings.json

모든 팀원에게 적용되는 프로젝트 수준 설정:

```json
{
  "permissions": {
    "allow": [
      "Bash(npm test)",
      "Bash(npm run lint)",
      "Bash(npm run build)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push --force)"
    ]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "command": "npx prettier --write $CLAUDE_FILE_PATH"
      }
    ]
  }
}
```

### settings.local.json

커밋해서는 안 되는 개인 재정의 설정:

```json
{
  "permissions": {
    "allow": [
      "Bash(docker compose up)"
    ]
  },
  "model": "claude-opus-4-20250514"
}
```

---

## 6. 설정 계층 구조

Claude Code는 세 가지 수준의 설정을 읽으며, 더 구체적인 설정이 일반 설정을 재정의합니다.

### 우선순위 순서 (높은 것부터 낮은 것까지)

```
┌─────────────────────────────────┐
│  1. 로컬 설정 (가장 높음)       │  .claude/settings.local.json
│     개인 재정의                 │  git에 커밋 안 함
├─────────────────────────────────┤
│  2. 프로젝트 설정               │  .claude/settings.json
│     팀 공유 규칙                │  git에 커밋
├─────────────────────────────────┤
│  3. 전역 설정 (가장 낮음)       │  ~/.claude/settings.json
│     사용자 전체 기본값          │  모든 프로젝트에 적용
└─────────────────────────────────┘
```

### 전역 설정 (~/.claude/settings.json)

모든 프로젝트에 적용되는 설정:

```json
{
  "permissions": {
    "deny": [
      "Bash(rm -rf /)",
      "Bash(sudo *)"
    ]
  },
  "model": "claude-sonnet-4-20250514"
}
```

### 병합 방식

설정은 교체가 아니라 병합됩니다. 허용(allow)과 거부(deny) 목록이 연결됩니다. 동일한 권한이 허용과 거부 목록 모두에 나타나면 거부가 우선합니다.

```
전역:    allow: ["Bash(git *)"]     deny: ["Bash(rm -rf *)"]
프로젝트: allow: ["Bash(npm test)"]  deny: ["Bash(git push --force)"]
로컬:    allow: ["Bash(docker *)"]

결과:    allow: ["Bash(git *)", "Bash(npm test)", "Bash(docker *)"]
         deny:  ["Bash(rm -rf *)", "Bash(git push --force)"]
```

### 유효 설정 확인

```bash
# Claude Code 세션 내부에서
> /config

# 병합된 유효 설정 표시
# 각 설정이 어느 파일에서 왔는지 포함
```

---

## 7. 커밋 대상 vs Gitignore 대상

커밋 여부에 대한 올바른 결정은 팀이 일관된 경험을 공유하면서 개인 설정을 보존하도록 합니다.

### 커밋할 파일

| 파일 | 이유 |
|------|------|
| `CLAUDE.md` | 전체 팀이 공유하는 프로젝트 컨텍스트 |
| `.claude/settings.json` | 공유 권한 규칙 및 훅 |
| `.claude/skills/*.md` | 공유 커스텀 스킬 |

### Gitignore할 파일

| 파일 | 이유 |
|------|------|
| `.claude/settings.local.json` | 개인 설정 (모델 선택, 추가 권한) |
| `.claude/credentials` | 인증 토큰 |

### 권장 .gitignore 항목

```gitignore
# Claude Code 로컬 설정
.claude/settings.local.json
.claude/credentials
```

### 팀 워크플로우

```
개발자 A가 프로젝트 설정 생성:
  1. 프로젝트 관례를 담은 CLAUDE.md 작성
  2. 공유 규칙이 담긴 .claude/settings.json 생성
  3. 두 파일 모두 커밋

개발자 B가 저장소를 클론:
  1. `claude` 실행 — CLAUDE.md와 설정 자동으로 적용
  2. 개인 설정용 .claude/settings.local.json 생성
  3. 개발자 A와 동일한 관례로 작업
```

---

## 8. 새 프로젝트 초기화

### /init 사용

기존 프로젝트에 CLAUDE.md를 가장 빠르게 생성하는 방법:

```bash
cd ~/projects/my-app
claude
```

```
> /init

Claude가 수행하는 작업:
1. 프로젝트 구조 스캔
2. 언어, 프레임워크, 도구 감지
3. 기존 설정 파일 읽기 (package.json, pyproject.toml 등)
4. 프로젝트에 맞춘 CLAUDE.md 생성
5. 검토 및 승인 요청
```

### 수동 생성

더 많은 제어를 원한다면 CLAUDE.md를 직접 생성합니다:

```bash
# 파일 생성
touch CLAUDE.md

# 에디터로 열기
code CLAUDE.md

# 또는 Claude의 도움 받기
claude -p "이 프로젝트를 읽고 포괄적인 CLAUDE.md 파일을 생성해 주세요"
```

### 템플릿으로 시작하기

```bash
# Claude Code 세션 내부에서
> 이 프로젝트에 맞는 CLAUDE.md를 생성해 주세요. 다음 섹션을 포함하세요:
  - 프로젝트 구조
  - 버전이 포함된 기술 스택
  - 개발 명령어 (설치, 테스트, 린트, 빌드)
  - 코딩 표준
  - Git 관례
  - 코드에서 추론할 수 있는 중요한 아키텍처 결정사항
```

---

## 9. CLAUDE.md vs 설정 파일: 상황에 따른 선택

CLAUDE.md와 설정 파일은 서로 다른 목적을 가집니다. 이 구분을 이해하는 것이 효과적인 설정의 핵심입니다.

### CLAUDE.md = 제안적 컨텍스트

CLAUDE.md는 Claude가 지침으로 사용하는 **제안과 컨텍스트**를 제공합니다. Claude는 이 지침을 따르려 하지만 상황에 따라 다르게 행동할 수 있습니다.

```markdown
# CLAUDE.md에서 — 제안과 컨텍스트

## 코딩 표준
- 4칸 들여쓰기 사용
- 함수는 30줄 미만
- 항상 타입 힌트 추가

## 아키텍처 참고사항
- 헥사고날(Hexagonal) 아키텍처 사용
- 도메인 로직은 인프라에 의존하면 안 됨
```

### 설정 파일 = 결정적 규칙

설정 파일은 Claude Code가 기계적으로 적용하는 **엄격한 규칙**을 정의합니다. 권한은 허용 또는 거부이며 해석의 여지가 없습니다.

```json
// .claude/settings.json에서 — 결정적 규칙
{
  "permissions": {
    "allow": ["Bash(npm test)"],
    "deny": ["Bash(rm -rf *)"]
  }
}
```

### 결정 가이드

| 질문 | CLAUDE.md 사용 | 설정 파일 사용 |
|------|:--------------:|:--------------:|
| "PEP 8 스타일 사용" | Yes | |
| "`rm -rf` 절대 실행 금지" | | Yes |
| "상속보다 합성 선호" | Yes | |
| "편집 후 Prettier로 자동 포맷" | | Yes (훅) |
| "API는 REST 관례 사용" | Yes | |
| "`npm test` 항상 허용" | | Yes |
| "데이터베이스 마이그레이션은 검토 필요" | Yes | |
| "모든 `sudo` 명령어 차단" | | Yes |

---

## 10. 모범 사례

### CLAUDE.md를 집중적으로 유지

```markdown
# 좋음: 구체적이고 실행 가능한 지침
## 테스트
- 모든 테스트에 `pytest tests/ -v` 실행
- 빠른 단위 테스트만 `pytest tests/unit/` 실행
- 최소 커버리지 80% 필요

# 나쁨: 모호하고 일반적인 지침
## 테스트
- 좋은 테스트 작성
- 제대로 동작하는지 확인
```

### 프로젝트 발전에 따라 CLAUDE.md 업데이트

```markdown
# 좋은 관행: 주요 결정에 날짜 기록

## 아키텍처 결정사항
- **2025-01**: 공개 API를 REST에서 GraphQL로 마이그레이션
- **2025-03**: MongoDB에서 PostgreSQL로 전환
- **2025-06**: 주문 서비스에 헥사고날 아키텍처 채택
```

### 근거를 위한 주석 사용

```markdown
## 코드 스타일

- `JSON.parse(JSON.stringify())` 대신 `structuredClone()` 사용
  (Node 18+을 타겟으로 하므로 structuredClone 항상 사용 가능)

- TypeScript에서 `enum` 사용 금지 — 대신 `as const` 객체 사용
  (enum은 혼란스러운 JavaScript를 생성하고 타입 내로잉(type-narrowing) 문제 있음)
```

### 모노레포에서 CLAUDE.md 계층화

```
my-monorepo/
├── CLAUDE.md                   # 공유 관례, 모노레포 명령어
├── packages/
│   ├── frontend/
│   │   └── CLAUDE.md           # React/Next.js 특화 지침
│   ├── backend/
│   │   └── CLAUDE.md           # NestJS 특화 지침
│   └── shared/
│       └── CLAUDE.md           # 공유 라이브러리 관례
```

루트 CLAUDE.md:
```markdown
# My Monorepo

## 패키지 매니저
- pnpm 사용 (npm 또는 yarn 사용 금지)
- 항상 모노레포 루트에서 명령어 실행

## 공유 규칙
- 모든 패키지는 `pnpm lint`와 `pnpm test`를 통과해야 함
- 공유 타입은 `packages/shared/`에 위치
```

패키지 CLAUDE.md:
```markdown
# Frontend Package

## 프레임워크
- App Router를 사용하는 Next.js 14 (Pages Router 사용 금지)
- 스타일링에 Tailwind CSS
- 기본적으로 서버 컴포넌트, 필요할 때만 'use client'

## 테스트
- `pnpm --filter frontend test`로 단위 테스트
- `pnpm --filter frontend e2e`로 Playwright 테스트
```

### 지나치게 긴 CLAUDE.md 파일 피하기

CLAUDE.md의 각 토큰은 매 턴마다 컨텍스트 윈도우에서 소비됩니다. 100~300줄을 목표로 하세요. CLAUDE.md가 500줄을 초과하면 다음을 고려하세요:

1. 상세 문서는 별도 파일로 이동하고 참조
2. 패키지별 지침에는 하위 디렉토리 CLAUDE.md 사용
3. 루트 CLAUDE.md에는 가장 자주 필요한 정보만 유지

---

## 11. 연습 문제

### 연습 1: CLAUDE.md 생성

현재 작업 중인 기존 프로젝트를 가져와 처음부터 CLAUDE.md를 생성하세요:

1. 프로젝트 구조 문서화
2. 버전이 포함된 기술 스택 목록 작성
3. 개발 명령어 작성 (설치, 테스트, 린트, 빌드)
4. 프로젝트에 특화된 코딩 표준 정의
5. 아키텍처 결정사항이나 중요 사항 추가
6. 200줄 이내로 유지

### 연습 2: 설정 구성

다음 요구사항에 맞는 `.claude/settings.json` 생성:

1. 프롬프트 없이 `npm test`와 `npm run lint` 허용
2. 모든 `rm -rf` 명령어 거부
3. `git push --force` 거부
4. force push를 제외한 모든 `git` 명령어 허용

### 연습 3: 모노레포 구성

다음을 포함하는 모노레포의 CLAUDE.md 계층 구조 설계:

- React 프론트엔드 (`packages/web/`)
- Python FastAPI 백엔드 (`packages/api/`)
- 공유 protobuf 정의 패키지 (`packages/proto/`)

루트 CLAUDE.md와 각 패키지의 CLAUDE.md를 작성하세요. 루트 vs 각 패키지에 무엇이 들어가야 할지 고려하세요.

### 연습 4: 검토 및 개선

이미 CLAUDE.md가 있다면, 이 레슨의 모범 사례와 비교하여 검토하세요:

1. Claude가 관례를 따를 만큼 충분히 구체적인가?
2. Claude가 실행해야 하는 명령어가 포함되어 있는가?
3. 컨텍스트 토큰을 낭비하지 않을 만큼 간결한가?
4. Claude가 가장 자주 실수하는 상황을 다루고 있는가?

---

## 12. 다음 단계

CLAUDE.md와 프로젝트 설정이 완료되면 Claude Code 세션이 프로젝트를 인식하게 됩니다. 다음 레슨에서는 **권한 모드(Permission Modes)**를 다룹니다 — Claude Code가 할 수 있는 것과 없는 것을 제어하는 보안 모델입니다. 권한을 이해하는 것은 특히 프로덕션 코드베이스나 팀 환경에서 생산성과 안전성 사이의 균형을 맞추는 데 필수적입니다.

**다음**: [04. 권한 모드와 보안](./04_Permission_Modes.md)
