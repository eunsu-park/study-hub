# 고급 개발 워크플로우

**이전**: [19. 모델, 가격 및 최적화](./19_Models_and_Pricing.md) | **다음**: [21. 모범 사례와 패턴](./21_Best_Practices.md)

---

Claude Code는 질문에 답하는 AI 어시스턴트에 그치지 않습니다. 복잡한 다단계 워크플로우를 실행할 수 있는 개발 파트너입니다. 이 레슨에서는 리팩토링, 테스트 주도 개발(TDD), CI/CD 통합, 대규모 코드베이스 탐색, 종단 간 기능 구현을 위한 고급 패턴을 다룹니다.

**난이도**: ⭐⭐⭐

**사전 요구 사항**:
- Claude Code 기초에 대한 탄탄한 경험 ([레슨 2-6](./02_Claude_Code_Getting_Started.md))
- 서브에이전트(subagent) 및 작업 위임 이해 ([레슨 7](./07_Subagents.md))
- Git 워크플로우 및 CI/CD 개념에 대한 친숙함
- 테스트 프레임워크 경험 (pytest, Jest 등)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Claude Code로 다중 파일 리팩토링을 안전하게 실행
2. Claude를 코딩 파트너로 활용한 테스트 주도 개발 실습
3. 자동화된 코드 리뷰를 위해 CI/CD 파이프라인에 Claude Code 통합
4. 크고 낯선 코드베이스를 체계적으로 탐색하고 이해
5. Claude 지원으로 데이터베이스 스키마 및 API 엔드포인트 구축
6. 코드 변경에 맞춰 문서 동기화 유지
7. 완전한 기능 구현에 이러한 워크플로우 적용

---

## 목차

1. [다중 파일 리팩토링 패턴](#1-다중-파일-리팩토링-패턴)
2. [Claude와 함께하는 테스트 주도 개발](#2-claude와-함께하는-테스트-주도-개발)
3. [CI/CD 파이프라인 통합](#3-cicd-파이프라인-통합)
4. [대규모 코드베이스 탐색](#4-대규모-코드베이스-탐색)
5. [데이터베이스 및 API 개발](#5-데이터베이스-및-api-개발)
6. [문서화 워크플로우](#6-문서화-워크플로우)
7. [사례 연구: 명세서부터 배포까지 기능 구현](#7-사례-연구-명세서부터-배포까지-기능-구현)
8. [연습 문제](#8-연습-문제)

---

## 1. 다중 파일 리팩토링 패턴

여러 파일에 걸친 리팩토링은 Claude Code의 가장 강력한 기능 중 하나입니다. 핵심은 체계적인 접근 방식입니다: 먼저 계획하고, 순서대로 실행하며, 각 단계 후 검증합니다.

### 1.1 계획 단계

변경을 시작하기 전에 Claude에게 리팩토링의 범위를 분석하도록 요청하세요:

```
> I want to rename the UserService class to AccountService across the entire
> codebase. Before making changes, analyze the impact:
> - Which files import or reference UserService?
> - Are there database migrations that reference this name?
> - Are there API endpoints that expose this name?
> - What tests will need to be updated?
```

Claude Code는 파일을 건드리기 전에 검색 도구(Grep, Glob)를 사용하여 전체 그림을 파악합니다. 이 계획 단계는 빌드를 망가뜨리는 불완전한 리팩토링을 방지합니다.

### 1.2 체계적인 실행

```
┌─────────────────────────────────────────────────────────────────────┐
│                다중 파일 리팩토링 전략                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1단계: 계획(PLAN)                                                  │
│  ├── 영향받는 모든 파일 식별 (참조 검색)                            │
│  ├── 의존성 순서 결정                                               │
│  ├── 동적 참조 확인 (문자열 기반, 설정 파일)                        │
│  └── 범위 추정 (파일 수, 위험 수준)                                 │
│                                                                     │
│  2단계: 준비(PREPARE)                                               │
│  ├── 시작 전 모든 테스트 통과 확인                                  │
│  ├── 리팩토링을 위한 브랜치 생성                                    │
│  └── 현재 상태 커밋 (깨끗한 기준선)                                 │
│                                                                     │
│  3단계: 실행(EXECUTE, 의존성 순서로)                                │
│  ├── 핵심 정의 먼저 (클래스/인터페이스/타입 정의)                  │
│  ├── 내부 소비자 다음 (서비스, 유틸리티)                            │
│  ├── 외부 인터페이스 마지막 (API 라우트, CLI 명령)                  │
│  └── 소스 파일과 함께 테스트                                        │
│                                                                     │
│  4단계: 검증(VERIFY)                                                │
│  ├── 각 변경 배치 후 테스트 실행                                    │
│  ├── 린터/타입 검사기 실행                                          │
│  ├── 영향받은 기능의 수동 스모크 테스트                             │
│  └── 의도하지 않은 변경을 위해 diff 검토                           │
│                                                                     │
│  5단계: 커밋(COMMIT)                                                │
│  ├── 리팩토링 단계당 하나의 논리적 커밋                             │
│  └── "왜"를 설명하는 명확한 커밋 메시지                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 병렬 변경에 서브에이전트 사용

독립적인 모듈에 걸친 대규모 리팩토링의 경우, Claude Code는 서브에이전트(subagent)를 사용하여 여러 파일을 동시에 처리할 수 있습니다:

```
> Refactor the logging system to use structured logging (structlog).
> The affected modules are independent: auth/, payments/, notifications/.
> Use subagents to update each module in parallel, then verify.
```

Claude Code는 각 모듈에 대해 서브에이전트를 생성하여 독립적으로 필요한 변경을 수행합니다. 모든 서브에이전트가 완료된 후 전체 테스트 스위트를 실행하여 통합을 검증합니다.

### 1.4 실용 예시: 모듈 추출

일반적인 리팩토링 패턴은 기능을 별도 모듈로 추출하는 것입니다:

```
> The user authentication logic is scattered across app.py, routes/auth.py,
> and middleware/session.py. Extract it into a clean auth/ package with:
>
> auth/
> ├── __init__.py       # Public API
> ├── service.py        # Core authentication logic
> ├── middleware.py      # Session middleware
> ├── schemas.py        # Pydantic models
> └── exceptions.py     # Custom exceptions
>
> Requirements:
> 1. No behavior changes (pure refactoring)
> 2. All existing tests must continue to pass
> 3. Update all imports across the codebase
> 4. Add deprecation warnings for old import paths
```

Claude Code는 다음 방식으로 처리합니다:
1. 영향받는 모든 파일을 읽어 현재 구조 파악
2. 제대로 정리된 코드로 새 패키지 생성
3. 코드베이스 전반에 걸쳐 임포트 업데이트
4. 아무것도 손상되지 않았는지 확인하기 위해 테스트 실행
5. 선택적으로 하위 호환 임포트 별칭 추가

---

## 2. Claude와 함께하는 테스트 주도 개발

### 2.1 적-녹-리팩토링(Red-Green-Refactor) 사이클

Claude Code와 함께하는 TDD(테스트 주도 개발)는 전통적인 적-녹-리팩토링 사이클을 따르지만, Claude가 각 단계를 가속화합니다:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Claude Code와 함께하는 TDD 사이클                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                    ┌──────────────┐                                 │
│                    │              │                                 │
│             ┌──────│   RED(적)    │◀─────────┐                     │
│             │      │ 실패하는 테스트│          │                     │
│             │      │ 작성         │          │                     │
│             │      └──────────────┘          │                     │
│             │             │                  │                     │
│             │      "...함수에 대한            │                     │
│             │       테스트를 작성해줘"        │                     │
│             ▼             │                  │                     │
│      ┌──────────────┐    │           ┌──────────────┐             │
│      │              │    ▼           │              │             │
│      │  GREEN(녹)   │◀───────       │   REFACTOR   │             │
│      │ 테스트를     │               │ 코드 품질     │             │
│      │ 통과시키는   │───────────────▶│ 개선         │             │
│      │ 최소 코드    │               └──────────────┘             │
│      └──────────────┘                      │                     │
│             │                       "테스트가 통과된               │
│      "테스트를 통과시키기               상태에서 이것을             │
│       위한 최소한의 코드만              더 깔끔하게               │
│       구현해줘"                         리팩토링해줘"             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 TDD 세션 예시

속도 제한기(rate limiter)를 구축하기 위한 완전한 TDD 세션입니다:

**1단계: RED -- 먼저 테스트 작성**

```
> Write pytest tests for a TokenBucketRateLimiter class with these behaviors:
> - Initialize with max_tokens and refill_rate (tokens per second)
> - allow_request() returns True if tokens available, False otherwise
> - Tokens refill over time based on refill_rate
> - Thread-safe for concurrent access
> Don't implement the class yet -- just the tests.
```

Claude가 테스트 파일을 생성합니다:

```python
# tests/test_rate_limiter.py
import time
import threading
import pytest
from rate_limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    """토큰 버킷 속도 제한기 테스트."""

    def test_allows_requests_within_limit(self):
        """토큰 한도 내의 요청은 허용되어야 함."""
        limiter = TokenBucketRateLimiter(max_tokens=5, refill_rate=1.0)
        for _ in range(5):
            assert limiter.allow_request() is True

    def test_rejects_requests_over_limit(self):
        """토큰 한도를 초과하는 요청은 거부되어야 함."""
        limiter = TokenBucketRateLimiter(max_tokens=3, refill_rate=1.0)
        for _ in range(3):
            limiter.allow_request()
        assert limiter.allow_request() is False

    def test_tokens_refill_over_time(self):
        """토큰은 보충 속도에 따라 시간이 지남에 따라 다시 채워져야 함."""
        limiter = TokenBucketRateLimiter(max_tokens=2, refill_rate=10.0)
        # 모든 토큰 소진
        limiter.allow_request()
        limiter.allow_request()
        assert limiter.allow_request() is False
        # 보충 대기 (10 토큰/초 = 0.1초당 1 토큰)
        time.sleep(0.15)
        assert limiter.allow_request() is True

    def test_tokens_do_not_exceed_max(self):
        """보충된 토큰은 max_tokens를 초과하지 않아야 함."""
        limiter = TokenBucketRateLimiter(max_tokens=3, refill_rate=100.0)
        time.sleep(0.1)  # 10개 토큰이 보충되어야 하지만 최대값은 3
        allowed = sum(1 for _ in range(10) if limiter.allow_request())
        assert allowed == 3

    def test_thread_safety(self):
        """속도 제한기는 동시 접근에 안전해야 함."""
        limiter = TokenBucketRateLimiter(max_tokens=100, refill_rate=0.0)
        results = []

        def make_requests():
            for _ in range(20):
                results.append(limiter.allow_request())

        threads = [threading.Thread(target=make_requests) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 정확히 100개가 허용되어야 함 (max_tokens=100, 보충 없음)
        assert sum(results) == 100
        assert len(results) == 200

    def test_zero_refill_rate(self):
        """보충 속도가 0이면 토큰이 절대 보충되지 않아야 함."""
        limiter = TokenBucketRateLimiter(max_tokens=1, refill_rate=0.0)
        assert limiter.allow_request() is True
        time.sleep(0.1)
        assert limiter.allow_request() is False

    def test_initial_state(self):
        """속도 제한기는 max_tokens를 사용 가능한 상태로 시작해야 함."""
        limiter = TokenBucketRateLimiter(max_tokens=10, refill_rate=1.0)
        assert limiter.available_tokens == 10
```

**2단계: GREEN -- 최소한으로 구현**

```
> Now implement TokenBucketRateLimiter in rate_limiter.py.
> Write just enough code to make all tests pass.
```

**3단계: REFACTOR -- 품질 개선**

```
> All tests pass. Now refactor the implementation:
> - Add proper docstrings
> - Improve variable naming
> - Add type hints
> - Keep all tests green
```

### 2.3 테스트 커버리지 분석에 Plan 모드 사용

구현 코드를 작성하기 전에 Plan 모드를 사용하여 어떤 테스트가 필요한지 분석하세요:

```
> /plan Review the UserService class and identify all untested code paths.
> List the test cases needed for 100% branch coverage.
```

Plan 모드는 변경 없이 코드를 분석하여 우선순위가 지정된 테스트 케이스 목록을 생성합니다. 그런 다음 구현 모드로 전환할 수 있습니다:

```
> Now write those 12 test cases you identified, starting with the critical paths.
```

---

## 3. CI/CD 파이프라인 통합

### 3.1 CI에서 Claude Code 실행

Claude Code는 `--print` 플래그(단일 프롬프트를 전송하고 종료)를 사용하거나 API를 통해 직접 CI 환경에서 비대화형으로 실행할 수 있습니다.

```yaml
# .github/workflows/claude-review.yml
name: Claude Code Review

on:
  pull_request:
    types: [opened, synchronize]

permissions:
  contents: read
  pull-requests: write

jobs:
  code-review:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # diff를 위한 전체 히스토리

      - name: Install Claude Code
        run: npm install -g @anthropic-ai/claude-code

      - name: Run Claude Code Review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          # 이 PR의 diff 가져오기
          DIFF=$(git diff origin/${{ github.base_ref }}...HEAD)

          # Claude Code를 비대화형 모드로 실행
          claude --print "Review this pull request diff for:
          1. Potential bugs or logic errors
          2. Security vulnerabilities
          3. Performance concerns
          4. Style/convention violations
          5. Missing tests

          Format your response as a markdown checklist.

          Diff:
          $DIFF" > review.md

      - name: Post review comment
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const review = fs.readFileSync('review.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Claude Code Review\n\n${review}`
            });
```

### 3.2 CI 실패 자동 수정

CI가 실패하면 Claude Code가 자동으로 문제를 진단하고 수정할 수 있습니다:

```yaml
# .github/workflows/claude-autofix.yml
name: Claude Auto-Fix

on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]

jobs:
  auto-fix:
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          ref: ${{ github.event.workflow_run.head_branch }}

      - name: Get CI logs
        id: logs
        uses: actions/github-script@v7
        with:
          script: |
            const jobs = await github.rest.actions.listJobsForWorkflowRun({
              owner: context.repo.owner,
              repo: context.repo.repo,
              run_id: ${{ github.event.workflow_run.id }},
            });
            const failedJob = jobs.data.jobs.find(j => j.conclusion === 'failure');
            const logs = await github.rest.actions.downloadJobLogsForWorkflowRun({
              owner: context.repo.owner,
              repo: context.repo.repo,
              job_id: failedJob.id,
            });
            return logs.data.substring(logs.data.length - 5000);  // 마지막 5000자

      - name: Claude Code auto-fix
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          claude --print "The CI pipeline failed. Here are the logs:

          ${{ steps.logs.outputs.result }}

          Diagnose the failure and fix the code. Run the failing tests
          to verify the fix works."

      - name: Create fix PR
        uses: peter-evans/create-pull-request@v6
        with:
          commit-message: "fix: auto-fix CI failure"
          title: "fix: Auto-fix CI failure from Claude Code"
          body: "Automated fix for CI failure. Please review before merging."
          branch: autofix/${{ github.event.workflow_run.head_branch }}
```

### 3.3 컨테이너를 위한 바이패스(Bypass) 모드

대화형 프롬프트가 불가능한 CI 환경에서 Claude Code는 모든 도구 실행을 자동으로 수락하는 바이패스(Bypass) 모드를 지원합니다:

```bash
# Docker 컨테이너 또는 CI 러너에서
export CLAUDE_CODE_PERMISSION_MODE=bypass

# Claude Code는 승인 요청 없이 모든 도구를 실행
# 신뢰할 수 있는 격리된 환경(CI 컨테이너)에서만 사용
claude --print "Run the test suite and fix any failing tests"
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CI/CD 통합 패턴                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  패턴 1: PR 리뷰 봇                                                 │
│  ├── 트리거: 풀 리퀘스트 열기/업데이트                              │
│  ├── 액션: diff 검토, 댓글 게시                                     │
│  └── 모델: Sonnet (품질과 비용의 균형)                              │
│                                                                     │
│  패턴 2: 자동 수정 봇                                               │
│  ├── 트리거: 기능 브랜치에서 CI 실패                               │
│  ├── 액션: 실패 진단, 수정 적용, PR 생성                           │
│  └── 모델: Sonnet 또는 Opus (복잡도에 따라)                        │
│                                                                     │
│  패턴 3: 사전 커밋 검사                                             │
│  ├── 트리거: 커밋 전 (훅을 통해)                                    │
│  ├── 액션: 린트, 형식 지정, 시크릿 확인                            │
│  └── 모델: Haiku (빠른 간단한 검사)                                │
│                                                                     │
│  패턴 4: 릴리스 노트 생성기                                         │
│  ├── 트리거: 태그 푸시 또는 릴리스 생성                             │
│  ├── 액션: 마지막 릴리스 이후 변경 사항 요약                       │
│  └── 모델: Sonnet (요약에 능함)                                    │
│                                                                     │
│  패턴 5: 문서 동기화                                                │
│  ├── 트리거: main에 병합                                            │
│  ├── 액션: API 문서, README, 변경 로그 업데이트                    │
│  └── 모델: Sonnet                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 대규모 코드베이스 탐색

### 4.1 점진적 이해 전략

낯선 코드베이스를 탐색할 때는 위에서 아래로 작업하세요:

```
┌─────────────────────────────────────────────────────────────────────┐
│                점진적 코드베이스 탐색                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  레벨 1: 프로젝트 개요 (5분)                                        │
│  ├── README.md, CLAUDE.md, CONTRIBUTING.md 읽기                    │
│  ├── 최상위 디렉토리 구조 나열                                      │
│  ├── package.json / pyproject.toml / Cargo.toml 확인               │
│  └── 이해: 이 프로젝트는 무엇을 하는가?                            │
│                                                                     │
│  레벨 2: 아키텍처 (15분)                                            │
│  ├── 진입점 식별 (main.py, index.ts, cmd/)                         │
│  ├── 모듈/패키지 구조 매핑                                          │
│  ├── 설정 파일 찾기                                                 │
│  ├── 데이터베이스 스키마 / 마이그레이션 확인                        │
│  └── 이해: 코드는 어떻게 구성되어 있는가?                          │
│                                                                     │
│  레벨 3: 데이터 흐름 (30분)                                         │
│  ├── 진입부터 응답까지 요청 추적                                    │
│  ├── 핵심 인터페이스 및 추상화 식별                                 │
│  ├── 의존성 주입 / 서비스 와이어링 매핑                             │
│  ├── API 라우트 및 핸들러 검토                                      │
│  └── 이해: 데이터는 시스템을 통해 어떻게 이동하는가?               │
│                                                                     │
│  레벨 4: 심층 탐색 (필요에 따라)                                    │
│  ├── 작업과 관련된 특정 모듈에 집중                                 │
│  ├── 예상 동작을 이해하기 위해 테스트 읽기                          │
│  ├── 최근 변경 사항을 위해 git blame 확인                           │
│  └── 이해: 이 특정 부분은 어떻게 작동하는가?                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 탐색 프롬프트

Claude Code로 체계적인 탐색을 위해 다음 프롬프트를 사용하세요:

```
# 레벨 1: 프로젝트 개요
> Explore this project and give me a high-level summary:
> - What does it do?
> - What language/framework is it built with?
> - How is it structured?
> - What are the main entry points?

# 레벨 2: 아키텍처 매핑
> Map the architecture of this codebase:
> - What are the main modules/packages?
> - What external dependencies does it use?
> - How are the database models structured?
> - What patterns does it follow (MVC, Clean Architecture, etc.)?

# 레벨 3: 특정 기능 추적
> Trace the complete request flow for user authentication:
> - Where does the request enter?
> - What middleware processes it?
> - How is the password verified?
> - What token/session is created?
> - How is the response returned?

# 레벨 4: 컴포넌트 심층 탐색
> I need to modify the payment processing logic.
> Walk me through:
> - All files involved in payment processing
> - The payment state machine
> - Error handling and retry logic
> - Related tests and their coverage
```

### 4.3 Plan 모드와 서브에이전트 사용

대규모 코드베이스의 경우, Plan 모드와 서브에이전트를 결합하세요:

```
# Plan 모드로 탐색 전략 수립
> /plan I need to add WebSocket support to this REST API.
> What parts of the codebase do I need to understand first?

# Claude가 변경 없이 계획을 생성:
# 1. 현재 서버 설정 확인 (Express/Fastify/기타)
# 2. 기존 연결 처리 검토
# 3. WebSocket의 이점을 받을 이벤트 패턴 식별
# 4. 인증 미들웨어 호환성 확인
# 5. WebSocket 테스트 추가를 위한 테스트 인프라 검토

# 계획을 사용하여 목표적 탐색 안내
> Based on the plan, explore step 1 and 2. Show me the relevant files.
```

### 4.4 멘탈 맵 생성

Claude에게 코드베이스의 시각적 맵을 생성하도록 요청하세요:

```
> Create an ASCII diagram showing the dependency relationships between
> the top-level modules in this project. Include which modules depend
> on which, and mark any circular dependencies.
```

Claude가 생성합니다:

```
모듈 의존성 맵:
─────────────────────

  api/
  ├──▶ services/
  │    ├──▶ models/
  │    ├──▶ repositories/
  │    │    └──▶ models/
  │    └──▶ utils/
  ├──▶ middleware/
  │    ├──▶ services/  (auth)
  │    └──▶ utils/     (logging)
  └──▶ schemas/
       └──▶ models/  (for validation)

  workers/
  ├──▶ services/
  └──▶ utils/

  ⚠ 순환 참조: services/email.py → services/user.py → services/email.py
```

---

## 5. 데이터베이스 및 API 개발

### 5.1 스키마 우선 개발

데이터베이스 스키마부터 시작하고 Claude가 나머지를 생성하게 하세요:

```
> Design a PostgreSQL schema for an e-commerce order management system.
> Requirements:
> - Users can have multiple addresses
> - Orders contain multiple line items
> - Products have variants (size, color)
> - Support for discount codes
> - Audit trail for order status changes
>
> Generate:
> 1. The SQL migration file
> 2. SQLAlchemy models
> 3. Pydantic schemas for the API
> 4. Basic CRUD repository classes
```

### 5.2 API 엔드포인트 생성

```
> Based on the Order model we just created, generate a complete
> FastAPI router with these endpoints:
>
> POST   /orders              - Create a new order
> GET    /orders              - List orders (with pagination and filters)
> GET    /orders/{id}         - Get order details
> PATCH  /orders/{id}/status  - Update order status
> DELETE /orders/{id}         - Cancel an order (soft delete)
>
> Include:
> - Input validation with Pydantic
> - Proper error handling (404, 422, etc.)
> - Authorization checks (users can only see their own orders)
> - Pagination with cursor-based approach
> - OpenAPI documentation strings
```

### 5.3 마이그레이션 생성

```
> We need to add a "shipping_tracking_number" field to the orders table.
> Generate:
> 1. An Alembic migration for this change
> 2. Update the SQLAlchemy model
> 3. Update the Pydantic schemas
> 4. Update the relevant API endpoints to expose this field
> 5. Add a test for the migration
```

Claude Code는 스택 전반에 걸쳐 일관성을 보장하면서 단일 조율된 변경으로 다섯 개 파일 모두를 처리합니다.

---

## 6. 문서화 워크플로우

### 6.1 코드에서 문서 생성

```
> Generate API documentation for all endpoints in src/routes/.
> For each endpoint, document:
> - HTTP method and path
> - Description
> - Request parameters and body schema
> - Response schema with examples
> - Error responses
> - Authentication requirements
>
> Output as a Markdown file at docs/API.md
```

### 6.2 문서 동기화 유지

코드 변경이 문서 업데이트를 필요로 할 수 있을 때 자동으로 알림을 보내는 훅을 사용하세요:

```json
// .claude/settings.json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "command": "python .claude/scripts/check_doc_sync.py \"$CLAUDE_FILE_PATH\""
      }
    ]
  }
}
```

```python
# .claude/scripts/check_doc_sync.py
"""코드 변경이 문서 업데이트를 필요로 할 수 있는지 확인."""
import sys
import os

filepath = sys.argv[1] if len(sys.argv) > 1 else ""

# 문서 영향을 시사하는 패턴
DOC_TRIGGERS = {
    "routes/": "docs/API.md",
    "models/": "docs/DATABASE.md",
    "config": "docs/CONFIGURATION.md",
    "README": None,  # 이미 문서
}

for pattern, doc_file in DOC_TRIGGERS.items():
    if pattern in filepath and doc_file:
        if os.path.exists(doc_file):
            print(f"참고: {filepath} 변경은 {doc_file} 업데이트가 필요할 수 있습니다")
```

### 6.3 아키텍처 결정 기록(ADR)

```
> Create an ADR for our decision to switch from REST to GraphQL
> for the mobile API. Follow the standard ADR format:
> - Title
> - Status
> - Context (why we're considering this)
> - Decision (what we decided)
> - Consequences (positive and negative)
>
> Save to docs/adr/0015-graphql-for-mobile-api.md
```

---

## 7. 사례 연구: 명세서부터 배포까지 기능 구현

Claude Code를 사용한 명세서부터 배포까지의 완전한 기능 구현을 살펴보겠습니다.

### 7.1 기능: 사용자 알림 설정

**명세서**: 사용자는 다양한 이벤트 유형(주문 업데이트, 프로모션, 보안 알림)에 대한 알림 설정(이메일, SMS, 푸시)을 구성할 수 있어야 합니다.

### 7.2 1단계: 계획

```
> /plan Design the notification preferences feature.
> Consider:
> - Database schema changes needed
> - API endpoints required
> - Frontend components affected
> - Integration with existing notification service
> - Migration strategy for existing users
> - Test coverage plan
```

Claude Code(Plan 모드)가 기존 코드베이스를 분석하고 계획을 생성합니다:

```
알림 설정 구현 계획
=============================================

1. 데이터베이스 변경 (마이그레이션 1개)
   - 새 테이블: notification_preferences
   - 컬럼: user_id, event_type, channel, enabled, created_at, updated_at
   - 기존 사용자에 대한 기본 설정 (마이그레이션 데이터)

2. 백엔드 변경 (파일 4개)
   - models/notification_preference.py (신규)
   - schemas/notification.py (업데이트)
   - routes/notifications.py (업데이트: 엔드포인트 3개 추가)
   - services/notification_service.py (업데이트: 설정 확인)

3. 프론트엔드 변경 (컴포넌트 3개)
   - NotificationSettings.tsx (신규 페이지)
   - NotificationToggle.tsx (신규 컴포넌트)
   - settings/index.tsx (내비게이션 링크 추가)

4. 테스트 (파일 2개)
   - tests/test_notification_preferences.py (신규)
   - tests/test_notification_service.py (업데이트)

5. 마이그레이션 전략
   - 기존 모든 사용자: 이메일=on, sms=off, push=on으로 기본 설정
   - 유지 관리 창 동안 마이그레이션 실행

예상 작업량: 파일 6개 수정, 파일 4개 생성
```

### 7.3 2단계: 구현

계획을 단계별로 실행합니다:

```
> Let's implement the notification preferences feature following the plan.
> Start with the database migration and model, then the API endpoints,
> then the service integration. Write tests for each layer.
> Use TDD: write the test first, then implement.
```

Claude Code가 각 단계를 실행하면서 테스트를 실행합니다:

```
1단계: 데이터베이스 마이그레이션 ✓
  - 마이그레이션 0042_notification_preferences.py 생성
  - models/notification_preference.py 생성
  - 테스트: 5/5 통과

2단계: API 엔드포인트 ✓
  - GET    /api/v1/users/{id}/notification-preferences
  - PUT    /api/v1/users/{id}/notification-preferences
  - PATCH  /api/v1/users/{id}/notification-preferences/{event_type}
  - 테스트: 12/12 통과

3단계: 서비스 통합 ✓
  - 전송 전 설정을 확인하도록 NotificationService 업데이트
  - 설정 캐싱 추가 (Redis, 5분 TTL)
  - 테스트: 8/8 통과 (통합 테스트 포함)

4단계: 프론트엔드 컴포넌트 ✓
  - NotificationSettings 페이지 생성
  - NotificationToggle 컴포넌트 생성
  - 설정 내비게이션 업데이트
  - 테스트: 6/6 통과

전체 테스트: 31/31 통과
```

### 7.4 3단계: 검토 및 마무리

```
> Review the complete implementation:
> 1. Check for edge cases we might have missed
> 2. Verify API error handling is comprehensive
> 3. Ensure the migration handles existing data correctly
> 4. Check that the frontend handles loading/error states
> 5. Verify no security issues (IDOR, missing auth checks)
```

### 7.5 4단계: 문서화 및 배포

```
> Update the following documentation for the notification preferences feature:
> 1. docs/API.md - add the new endpoints
> 2. CHANGELOG.md - add entry for this feature
> 3. docs/adr/0016-notification-preferences-design.md - create ADR
> 4. README.md - update feature list if needed
```

```
> Create the PR description summarizing all changes and linking to the spec.
```

---

## 8. 연습 문제

### 연습 1: 리팩토링 실습 (중급)

정기적으로 작업하는 코드베이스를 가져와 Claude Code에게 다음을 요청하세요:
1. 가장 큰 함수 3개 식별 (줄 수 기준)
2. 각각을 더 작고 집중된 함수들로 분리하는 방법 제안
3. TDD를 사용하여 그 중 하나의 리팩토링 실행

### 연습 2: CI/CD 설정 (중급)

Claude Code를 사용하는 GitHub Actions 워크플로우를 만드세요:
1. 모든 풀 리퀘스트를 검토하고 댓글 게시
2. `src/`의 파일을 수정하는 PR에서만 실행 (문서 전용 PR 건너뜀)
3. 초기 분류에는 Haiku를 사용하고, 보안에 민감한 패턴이 있는 파일에는 Sonnet으로 에스컬레이션

### 연습 3: 코드베이스 탐색 (초급)

이전에 작업해본 적 없는 오픈소스 프로젝트를 선택하고 Claude Code의 탐색 전략을 사용하여:
1. 프로젝트 구조 이해 (레벨 1)
2. 아키텍처 매핑 (레벨 2)
3. 특정 기능의 코드 경로 추적 (레벨 3)
4. CLAUDE.md 파일에 발견 사항 문서화

### 연습 4: 전체 기능 구현 (고급)

1-4단계 워크플로우를 사용하여 완전한 기능을 구현하세요:
1. 작성된 명세서로 시작 (2-3 문단)
2. Plan 모드를 사용하여 구현 계획 수립
3. Claude Code로 TDD를 사용하여 구현
4. 검토, 문서화, PR 생성

각 단계에 소요되는 시간을 추적하고 Claude Code가 가장 큰 가치를 제공한 곳을 식별하세요.

### 연습 5: 문서화 생성 (중급)

자동화된 문서화 워크플로우를 설정하세요:
1. 코드베이스에서 모든 API 엔드포인트를 추출하는 스크립트 작성
2. Claude Code를 사용하여 포괄적인 API 문서 생성
3. 코드 변경이 문서 업데이트를 필요로 할 수 있을 때 경고하는 훅 생성
4. 문서가 최신 상태인지 확인하는 CI 검사 설정

---

## 참고 자료

- Claude Code 문서 - https://docs.anthropic.com/en/docs/claude-code
- GitHub Actions 문서 - https://docs.github.com/en/actions
- 예제를 통한 테스트 주도 개발 - Kent Beck
- 리팩토링: 기존 코드의 설계 개선 - Martin Fowler

---

## 다음 단계

[21. 모범 사례와 패턴](./21_Best_Practices.md)에서는 효과적인 프롬프트 작성, 컨텍스트 관리, 보안 관행, 팀 협업 패턴, 피해야 할 일반적인 안티패턴을 다룹니다. Claude Code를 최대한 활용하기 위한 축적된 지혜를 배웁니다.
