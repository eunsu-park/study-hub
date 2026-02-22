# 권한 모드와 보안

**이전**: [03. CLAUDE.md와 프로젝트 설정](./03_CLAUDE_md_and_Project_Setup.md) | **다음**: [05. 훅과 이벤트 기반 자동화](./05_Hooks.md)

---

Claude Code는 파일을 읽고, 코드를 작성하며, 컴퓨터에서 임의의 셸 명령어를 실행할 수 있습니다. 이러한 강력한 기능은 견고한 권한 시스템을 필요로 합니다. Claude Code는 서로 다른 신뢰 수준과 워크플로우를 위해 설계된 다섯 가지 고유한 권한 모드를 제공합니다. 이 레슨에서는 각 모드를 상세히 다루고, 허용/거부 규칙 설정 방법을 설명하며, 개인 및 팀 사용을 위한 보안 모범 사례를 정립합니다.

**난이도**: ⭐⭐

**선행 학습**:
- [02. Claude Code: 시작하기](./02_Claude_Code_Getting_Started.md)
- [03. CLAUDE.md와 프로젝트 설정](./03_CLAUDE_md_and_Project_Setup.md)
- 파일 시스템 권한과 셸 명령어에 대한 이해

**학습 목표**:
- 권한 제어가 존재하는 이유와 다루는 위협 모델 이해
- 다섯 가지 권한 모드 각각 설정
- 세밀한 제어를 위한 글로브(glob) 패턴으로 허용/거부 규칙 작성
- 읽기 및 편집 작업을 제한하는 파일 접근 규칙 설정
- 다양한 워크플로우에 적합한 모드 선택
- 최소 권한 원칙을 포함한 보안 모범 사례 적용

---

## 목차

1. [권한이 중요한 이유](#1-권한이-중요한-이유)
2. [권한 모드 개요](#2-권한-모드-개요)
3. [기본 모드](#3-기본-모드)
4. [자동 수락 모드](#4-자동-수락-모드)
5. [플랜 모드](#5-플랜-모드)
6. [묻지 않기 모드](#6-묻지-않기-모드)
7. [바이패스 모드](#7-바이패스-모드)
8. [허용 및 거부 규칙](#8-허용-및-거부-규칙)
9. [파일 접근 규칙](#9-파일-접근-규칙)
10. [올바른 모드 선택](#10-올바른-모드-선택)
11. [보안 모범 사례](#11-보안-모범-사례)
12. [연습 문제](#12-연습-문제)
13. [다음 단계](#13-다음-단계)

---

## 1. 권한이 중요한 이유

Claude Code는 샌드박스에 갇힌 챗봇이 아닙니다 — 파일 시스템과 터미널에서 직접 동작합니다. 단 하나의 실수로:

- **파일 삭제**: 잘못된 디렉토리에서의 `rm -rf`
- **코드 덮어쓰기**: 프로덕션 설정에 대한 잘못된 편집
- **비밀 정보 노출**: 환경 변수를 로그에 출력하는 명령어 실행
- **빌드 중단**: 호환되지 않는 의존성 설치 또는 빌드 파일 수정
- **코드 푸시**: 원격 저장소에 영향을 주는 Git 작업

가 발생할 수 있습니다.

권한 시스템은 안전망입니다. Claude Code가 명시적으로 승인한 작업만 수행하도록 보장하며, 또는 환경을 신뢰하는 경우 더 빠른 반복을 위해 제어를 완화할 수 있습니다.

### 위협 모델

```
┌─────────────────────────────────────────────────────┐
│                  신뢰 수준                           │
│                                                     │
│  높은 신뢰                          낮은 신뢰       │
│  ◀─────────────────────────────────────────────▶    │
│                                                     │
│  일회용 VM    개인 프로젝트         프로덕션         │
│  CI 컨테이너  사이드 프로젝트       클라이언트 코드  │
│  플레이그라운드 알려진 코드베이스   공유 머신        │
│                                                     │
│  ← 바이패스 모드  Sonnet 모드 →    플랜 모드 →      │
│  ← 자동 수락     기본 모드 →       묻지 않기 →      │
└─────────────────────────────────────────────────────┘
```

---

## 2. 권한 모드 개요

| 모드 | 프롬프트? | 편집? | 명령어? | 적합한 경우 |
|------|-----------|-------|---------|------------|
| **기본** | 각 작업마다 Yes | 승인 필요 | 승인 필요 | 학습, 민감한 코드 |
| **자동 수락** | No | 자동 승인 | 자동 승인 | 신뢰하는 프로젝트, 빠른 반복 |
| **플랜** | N/A | No | No (읽기 전용) | 분석, 아키텍처 검토 |
| **묻지 않기** | 알 수 없는 작업만 | 규칙 기반 | 규칙 기반 | 설정된 워크플로우 |
| **바이패스** | No | Yes, 샌드박스 없음 | Yes, 샌드박스 없음 | 컨테이너, VM, CI/CD |

### 모드 설정 방법

```bash
# 특정 모드로 시작
claude --mode default
claude --mode auto-accept
claude --mode plan
claude --mode bypass

# 세션 내부에서 모드 전환 (설정이 허용하는 경우)
> /mode plan
> /mode auto-accept
```

---

## 3. 기본 모드

기본 모드는 가장 신중한 옵션입니다. Claude Code는 모든 파일 편집과 모든 명령어 실행 전에 명시적 승인을 요청합니다. 읽기 작업(Read, Glob, Grep)은 아무것도 수정하지 않으므로 프롬프트 없이 허용됩니다.

### 프롬프트가 나타나는 경우

```
프롬프트 없이 허용:
  ✓ 파일 내용 읽기
  ✓ 파일 검색 (Glob)
  ✓ 파일 내용 검색 (Grep)
  ✓ 웹 검색 및 가져오기

승인 필요:
  ⚠ 기존 파일 편집
  ⚠ 새 파일 작성
  ⚠ Bash 명령어 실행
  ⚠ Jupyter 노트북 편집
```

### 프롬프트 모습

Claude가 파일을 편집하려 할 때:

```
Claude가 src/auth/login.ts를 편집하려 합니다

  @@ -15,7 +15,10 @@
   export async function login(req: Request, res: Response) {
  -  const { email, password } = req.body;
  +  const { email, password } = req.body;
  +  if (!email || !password) {
  +    return res.status(400).json({ error: 'Email and password required' });
  +  }

이 편집을 허용하시겠습니까? [y/n/always]
```

Claude가 명령어를 실행하려 할 때:

```
Claude가 실행하려 합니다: npm test

허용하시겠습니까? [y/n/always]
```

### 응답 옵션

| 옵션 | 효과 |
|------|------|
| `y` | 이 특정 작업 허용 |
| `n` | 이 특정 작업 거부 |
| `always` | 세션의 나머지 동안 유사한 모든 작업 허용 |

### 기본 모드 사용 시기

- Claude Code를 처음 사용하며 작동 방식을 학습 중일 때
- 중요한 프로덕션 코드 작업 시
- 변경사항이 발생하기 전에 모든 변경을 검토하려 할 때
- 다른 사람에게 Claude Code를 시연할 때

---

## 4. 자동 수락 모드

자동 수락 모드는 모든 도구 호출을 사전 승인합니다. Claude Code는 파일을 읽고, 편집하고, 명령어를 실행하면서 묻지 않습니다. 이는 반복적 개발에 가장 빠른 모드이지만 모델과 코드베이스 모두에 대한 신뢰가 필요합니다.

### 자동 수락 모드 시작

```bash
# 명령줄에서
claude --mode auto-accept

# 또는 세션 내부에서
> /mode auto-accept
```

### 자동 수락에서 일어나는 일

```
> 인증 버그를 수정하고 테스트를 실행하세요

Claude가 자동으로 수행합니다:
  1. 관련 파일 읽기 (프롬프트 없음)
  2. 소스 파일 편집 (프롬프트 없음)
  3. `npm test` 실행 (프롬프트 없음)
  4. 테스트 실패 시, 오류 출력 읽고 수정 (프롬프트 없음)
  5. 테스트 재실행 (프롬프트 없음)
  6. 결과 보고
```

### 안전 고려사항

자동 수락 모드에서도 일부 안전 조치는 유지됩니다:

1. **거부 규칙 적용**: 설정에 거부 규칙이 있으면 모드에 관계없이 적용됨
2. **샌드박스**: 파일 작업은 기본적으로 프로젝트 디렉토리로 샌드박스 처리됨
3. **네트워크 제한**: Bash에서의 아웃바운드 네트워크 호출은 기본적으로 제한됨

### 자동 수락 모드 사용 시기

- 버전 관리가 있는 개인 프로젝트 작업 시
- 빠른 반복 (수정 → 테스트 → 수정 → 테스트)을 원할 때
- 프로젝트에 안전망 역할의 좋은 테스트 커버리지가 있을 때
- 대규모 리팩토링을 진행하며 `git diff`로 마지막에 변경사항을 검토하려 할 때

### 위험 완화

```bash
# 자동 수락 사용 전, 깨끗한 git 상태 확인
git status          # 커밋되지 않은 변경사항 없어야 함
git stash           # 진행 중인 작업 스태시

# 자동 수락 세션 시작
claude --mode auto-accept

# 세션 후, 모든 변경사항 검토
git diff            # Claude가 만든 모든 변경사항 검토
git diff --stat     # 변경된 파일 요약

# 문제가 발생한 경우
git checkout .      # 모든 변경사항 되돌리기 (파괴적)
git stash pop       # 스태시한 작업 복원
```

---

## 5. 플랜 모드

플랜 모드는 **읽기 전용**입니다. Claude Code는 파일을 읽고, 코드베이스를 검색하고, 코드를 분석할 수 있지만 편집이나 명령어를 실행할 수 없습니다. 분석, 아키텍처 검토, 계획 수립 세션에 이상적입니다.

### 플랜 모드 시작

```bash
claude --mode plan
```

### 플랜 모드에서 Claude가 할 수 있는 것

```
✓ 파일 읽기 및 코드 이해
✓ Glob과 Grep으로 검색
✓ 아키텍처와 패턴 분석
✓ 변경 제안 (구현은 않음)
✓ 계획 및 액션 아이템 생성
✓ 참고 정보 웹 검색

✗ 파일 편집 또는 작성
✗ Bash 명령어 실행
✗ 테스트 또는 빌드 실행
✗ Git 커밋 생성
```

### 사용 사례

**아키텍처 검토**:
```
> 이 프로젝트의 의존성 그래프를 분석하세요. 순환 의존성을
  식별하고 이를 해결하는 방법을 제안하세요.
```

**코드 감사**:
```
> 보안 취약점에 대해 인증 흐름을 검토하세요.
  SQL 인젝션, XSS, CSRF, 안전하지 않은 토큰 처리를 확인하세요.
```

**계획 수립**:
```
> 멀티 테넌시(multi-tenancy) 지원을 추가해야 합니다. 현재
  데이터 모델을 분석하고 마이그레이션 계획을 제안하세요.
  변경이 필요한 모든 파일을 나열하세요.
```

**온보딩**:
```
> 이 프로젝트의 아키텍처를 설명하세요. API 게이트웨이에서
  데이터베이스까지 요청이 어떻게 흐르는지 설명하세요.
```

---

## 6. 묻지 않기 모드

묻지 않기 모드는 사전 설정된 규칙을 사용하여 작업을 자동으로 승인하거나 거부합니다. 허용 규칙에 일치하는 작업은 자동으로 실행됩니다. 거부 규칙에 일치하는 작업은 차단됩니다. 어느 쪽에도 해당하지 않는 작업은 기본적으로 거부됩니다(안전 폐쇄, fail-closed).

### 설정

설정 파일을 통해 묻지 않기 모드를 구성합니다:

```json
// .claude/settings.json
{
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Edit",
      "Write",
      "Bash(npm test)",
      "Bash(npm run lint)",
      "Bash(npm run build)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git add *)",
      "Bash(git commit *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)",
      "Bash(git reset --hard *)",
      "Bash(curl *)",
      "Bash(wget *)",
      "Bash(sudo *)"
    ]
  }
}
```

### 규칙 매칭 작동 방식

규칙은 다음 순서로 평가됩니다:
1. 거부 목록 확인 — 일치하면 작업 **차단**
2. 허용 목록 확인 — 일치하면 작업 **허용**
3. 일치하는 규칙 없으면 — 작업 **거부** (안전 폐쇄)

```
작업: Bash("npm test")
  → 거부 목록: 일치 없음
  → 허용 목록: "Bash(npm test)"와 일치
  → 결과: 허용됨

작업: Bash("rm -rf /tmp/cache")
  → 거부 목록: "Bash(rm -rf *)"와 일치
  → 결과: 차단됨

작업: Bash("python script.py")
  → 거부 목록: 일치 없음
  → 허용 목록: 일치 없음
  → 결과: 거부됨 (일치하는 규칙 없음)
```

### 묻지 않기 모드 사용 시기

- 알려진 명령어로 잘 정의된 워크플로우가 있을 때
- 전면적 승인 없는 자동화를 원할 때
- 특정 명령어가 항상 허용되어야 하는 팀 환경
- 제어된 명령어 셋을 가진 CI/CD 파이프라인

---

## 7. 바이패스 모드

바이패스 모드는 모든 권한 검사와 샌드박싱을 비활성화합니다. Claude Code가 사용자 계정의 전체 권한으로 실행됩니다. 이 모드는 **격리된 환경 전용**으로 설계되었습니다 — 컨테이너, 가상 머신, CI/CD 러너.

### 바이패스 모드 시작

```bash
# 컨테이너 또는 VM에서만 사용
claude --mode bypass
```

### 바이패스가 비활성화하는 것

```
바이패스 모드에서 비활성화되는 것:
  ✗ 권한 프롬프트 (모든 작업 자동 승인)
  ✗ 파일 시스템 샌드박싱 (임의 경로 접근 가능)
  ✗ 네트워크 제한 (아웃바운드 호출 가능)
  ✗ 거부 규칙 (거부 규칙도 무시됨)
```

### 적합한 환경

```
바이패스 모드에 안전한 환경:
  ✓ Docker 컨테이너 (일회용)
  ✓ GitHub Actions 러너
  ✓ CI/CD 빌드 에이전트
  ✓ 개발 VM
  ✓ 클라우드 기반 개발 환경 (Codespaces, Gitpod)

바이패스 모드 절대 사용 금지:
  ✗ 개인 머신
  ✗ 프로덕션 서버
  ✗ 공유 워크스테이션
  ✗ 프로젝트 외부에 민감한 데이터가 있는 머신
```

### Docker 예시

```dockerfile
FROM node:20-slim

# Claude Code 설치
RUN npm install -g @anthropic-ai/claude-code

# API 키 설정 (프로덕션에서는 시크릿 사용)
ENV ANTHROPIC_API_KEY=sk-ant-api03-...

WORKDIR /app
COPY . .

# 컨테이너 내부에서 바이패스 모드로 Claude Code 실행
CMD ["claude", "--mode", "bypass", "-p", "모든 테스트를 실행하고 실패한 것을 수정하세요"]
```

---

## 8. 허용 및 거부 규칙

규칙은 Claude Code가 할 수 있는 것을 세밀하게 제어합니다. 와일드카드와 특정 도구 타겟팅을 지원하는 패턴 매칭 구문을 사용합니다.

### 규칙 구문

```
ToolName(pattern)
```

여기서:
- `ToolName`은 도구 이름: `Bash`, `Edit`, `Write`, `Read`, `Glob`, `Grep`
- `pattern`은 도구 인수에 대한 글로브 스타일 매칭
- `*`는 임의의 문자와 매칭

### Bash 규칙

```json
{
  "permissions": {
    "allow": [
      "Bash(npm test)",
      "Bash(npm test *)",
      "Bash(npm run lint)",
      "Bash(npx prettier *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(python -m pytest *)",
      "Bash(make *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push --force *)",
      "Bash(git reset --hard *)",
      "Bash(sudo *)",
      "Bash(curl *)",
      "Bash(wget *)",
      "Bash(pip install *)",
      "Bash(npm install *)"
    ]
  }
}
```

### Edit 및 Write 규칙

```json
{
  "permissions": {
    "allow": [
      "Edit(src/**)",
      "Edit(tests/**)",
      "Write(tests/**)"
    ],
    "deny": [
      "Edit(.env*)",
      "Edit(*.pem)",
      "Edit(*.key)",
      "Write(.env*)",
      "Edit(package-lock.json)",
      "Edit(pnpm-lock.yaml)"
    ]
  }
}
```

### Read 규칙

읽기 작업도 민감한 파일에 대해 제한할 수 있습니다:

```json
{
  "permissions": {
    "deny": [
      "Read(.env*)",
      "Read(**/*.pem)",
      "Read(**/*.key)",
      "Read(**/credentials*)"
    ]
  }
}
```

### 완전한 설정 예시

다음은 Node.js 프로젝트를 위한 포괄적인 설정 파일입니다:

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Edit(src/**)",
      "Edit(tests/**)",
      "Edit(docs/**)",
      "Write(src/**)",
      "Write(tests/**)",
      "Bash(npm test)",
      "Bash(npm test *)",
      "Bash(npm run lint)",
      "Bash(npm run lint:fix)",
      "Bash(npm run build)",
      "Bash(npx prettier --write *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(git branch *)",
      "Bash(git checkout *)",
      "Bash(node -e *)",
      "Bash(npx tsc --noEmit)"
    ],
    "deny": [
      "Read(.env*)",
      "Read(**/*.pem)",
      "Edit(.env*)",
      "Edit(package-lock.json)",
      "Edit(node_modules/**)",
      "Write(.env*)",
      "Write(node_modules/**)",
      "Bash(rm -rf *)",
      "Bash(sudo *)",
      "Bash(git push *)",
      "Bash(git reset --hard *)",
      "Bash(npm publish *)",
      "Bash(curl *)",
      "Bash(wget *)"
    ]
  }
}
```

---

## 9. 파일 접근 규칙

도구 수준의 권한 외에도, Claude Code는 사용하는 도구에 관계없이 Claude가 읽거나 편집할 수 있는 파일을 제한하는 경로 기반 규칙을 지원합니다.

### 경로 패턴

파일 접근 규칙은 글로브 패턴을 사용합니다:

| 패턴 | 매칭 대상 |
|------|----------|
| `src/**` | `src/` 하위의 모든 파일 (재귀적) |
| `*.ts` | 현재 디렉토리의 모든 TypeScript 파일 |
| `**/*.test.ts` | 프로젝트 어디서나 모든 테스트 파일 |
| `config/*.json` | `config/` 직접 하위의 JSON 파일 |
| `!.env*` | 모든 `.env` 파일 제외 (거부 패턴) |

### 실용적인 설정

**Claude를 특정 디렉토리로 제한**:

```json
{
  "permissions": {
    "allow": [
      "Edit(src/**)",
      "Edit(tests/**)"
    ],
    "deny": [
      "Edit(infrastructure/**)",
      "Edit(deployment/**)",
      "Edit(.github/**)"
    ]
  }
}
```

**설정 파일 보호**:

```json
{
  "permissions": {
    "deny": [
      "Edit(*.config.js)",
      "Edit(*.config.ts)",
      "Edit(tsconfig.json)",
      "Edit(package.json)",
      "Edit(.eslintrc*)",
      "Edit(Dockerfile*)",
      "Edit(docker-compose*)"
    ]
  }
}
```

**특정 파일에 읽기 전용 접근**:

```json
{
  "permissions": {
    "allow": [
      "Read(infrastructure/**)"
    ],
    "deny": [
      "Edit(infrastructure/**)",
      "Write(infrastructure/**)"
    ]
  }
}
```

---

## 10. 올바른 모드 선택

### 결정 플로우차트

```
환경이 일회용인가 (컨테이너, VM)?
├── Yes → 바이패스 모드
│
└── No → Claude가 변경사항을 만들기를 원하는가?
    ├── No → 플랜 모드 (읽기 전용)
    │
    └── Yes → 잘 정의된 규칙이 있는가?
        ├── Yes → 묻지 않기 모드 (규칙 기반)
        │
        └── No → 코드베이스에 좋은 테스트가 있다고 신뢰하는가?
            ├── Yes → 자동 수락 모드
            │
            └── No → 기본 모드 (수동 승인)
```

### 시나리오별 모드 비교

| 시나리오 | 권장 모드 | 근거 |
|----------|----------|------|
| Claude Code 처음 사용 | 기본 | 신뢰하기 전에 Claude가 하는 일 파악 |
| 개인 프로젝트, 좋은 테스트 | 자동 수락 | 테스트 안전망이 있는 빠른 반복 |
| 팀 프로젝트, 공유 규칙 | 묻지 않기 | 개발자 간 일관된 권한 |
| 프로덕션 긴급 수정 | 기본 | 중요한 변경에 최대 제어 |
| 아키텍처 검토 | 플랜 | 읽기 전용 분석, 실수로 인한 변경 방지 |
| CI/CD 파이프라인 | 바이패스 | 격리된 컨테이너, 사람 없음 |
| 코드베이스 탐색 | 플랜 | 변경 없이 코드 이해 |
| 대규모 리팩토링 | 자동 수락 | 개별적으로 승인하기에 너무 많은 변경 |
| 클라이언트 코드 검토 | 플랜 | 보안 및 책임을 위한 읽기 전용 |

### 세션 중 모드 전환

세션 중에 모드를 변경할 수 있습니다:

```
> /mode plan
# 읽기 전용 모드에서 코드베이스 분석

> /mode auto-accept
# 구현을 위해 자동 수락으로 전환

> /mode default
# 민감한 변경을 위해 기본으로 전환
```

---

## 11. 보안 모범 사례

### 최소 권한 원칙(Principle of Least Privilege)

현재 작업에 필요한 권한만 Claude Code에 부여하세요. 제한적으로 시작하고 필요에 따라 확장하세요.

```json
// 좋음: 구체적인 권한
{
  "permissions": {
    "allow": [
      "Bash(npm test)",
      "Bash(npm run lint)",
      "Edit(src/**)",
      "Edit(tests/**)"
    ]
  }
}

// 나쁨: 지나치게 관대한 권한
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Edit(*)",
      "Write(*)"
    ]
  }
}
```

### 비밀 정보 보호

비밀 정보가 담긴 파일에 대한 접근을 항상 거부하세요:

```json
{
  "permissions": {
    "deny": [
      "Read(.env*)",
      "Read(**/*.pem)",
      "Read(**/*.key)",
      "Read(**/credentials*)",
      "Read(**/secrets*)",
      "Edit(.env*)",
      "Write(.env*)"
    ]
  }
}
```

### 위험한 명령어 제한

```json
{
  "permissions": {
    "deny": [
      "Bash(rm -rf *)",
      "Bash(sudo *)",
      "Bash(chmod 777 *)",
      "Bash(git push --force *)",
      "Bash(git reset --hard *)",
      "Bash(git clean -f *)",
      "Bash(> *)",
      "Bash(curl * | sh)",
      "Bash(curl * | bash)",
      "Bash(eval *)"
    ]
  }
}
```

### 버전 관리를 안전망으로 사용

```bash
# Claude Code 세션 시작 전:
# 1. 모든 변경사항이 커밋되었는지 확인
git status  # 깨끗해야 함

# 2. 작업 브랜치 생성
git checkout -b feature/claude-changes

# 3. 세션 후, 모든 변경사항 검토
git diff main...HEAD

# 4. 문제가 발생한 경우 언제든지 돌아갈 수 있음
git checkout main
git branch -D feature/claude-changes
```

### 팀 보안 체크리스트

Claude Code를 사용하는 팀을 위한 공유 보안 정책 수립:

```markdown
## Claude Code 보안 정책 (팀을 위한 예시)

1. **커밋된 설정**: 모든 팀원이 `.claude/settings.json` 사용
2. **최소 거부 목록**: rm -rf, sudo, git push --force, curl|sh
3. **비밀 정보 보호**: .env, .pem, .key 파일 모든 작업에서 거부
4. **모드 제한**: 바이패스 모드는 CI/CD 컨테이너에서만
5. **코드 검토**: 모든 Claude 생성 변경사항은 PR 검토 통과
6. **감사**: 각 세션 종료 시 `/cost` 실행; 이상 사항 보고
```

---

## 12. 연습 문제

### 연습 1: 모드 선택

각 시나리오에 적합한 권한 모드를 선택하고 이유를 설명하세요:

1. 방금 클론한 새 오픈 소스 프로젝트 탐색
2. 회사 결제 처리 서비스의 버그 수정
3. GitHub Actions 워크플로우에서 Claude Code 실행
4. 개인 블로그 코드베이스의 대규모 리팩토링
5. 주니어 개발자가 처음으로 Claude Code 사용

### 연습 2: 권한 규칙 작성

다음 요구사항에 맞는 Python Django 프로젝트용 `.claude/settings.json` 생성:

1. `python manage.py test`와 `python manage.py migrate` 실행 허용
2. `apps/`와 `tests/` 디렉토리의 파일 편집 허용
3. `settings.py`, `urls.py`, 마이그레이션 파일 편집 거부
4. 모든 `pip install` 명령어 거부 (의존성은 수동으로 관리)
5. git status, diff, log 허용, push 및 force 작업 거부

### 연습 3: 보안 감사

다음 설정 파일을 검토하고 모든 보안 문제를 식별하세요:

```json
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Edit(*)",
      "Write(*)",
      "Read(*)"
    ],
    "deny": [
      "Bash(rm -rf /)"
    ]
  }
}
```

최소 5가지 문제를 나열하고 수정된 버전을 제공하세요.

### 연습 4: 점진적 신뢰 확장

Claude Code를 사용하는 새 팀원을 위한 권한 진행 과정 설계:

- 1주차: 어떤 모드와 규칙?
- 2-3주차: 어떻게 권한을 확장하는가?
- 2개월 이상: 안정적인 설정은 무엇인가?

각 단계에 대한 이유를 문서화하세요.

---

## 13. 다음 단계

이제 시스템에 대한 Claude Code의 접근을 제어하는 방법을 이해했습니다. 다음 레슨에서는 **훅(Hooks)**을 소개합니다 — Claude Code 작업에 반응하여 셸 명령어를 실행하는 이벤트 기반 자동화입니다. 훅을 사용하면 수동 개입 없이 포맷팅, 린팅, 테스트, 알림을 자동화할 수 있어 Claude Code를 완전히 자동화된 개발 워크플로우로 만들 수 있습니다.

**다음**: [05. 훅과 이벤트 기반 자동화](./05_Hooks.md)
