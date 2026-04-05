# 레슨 3: CI 기초

**이전**: [버전 관리 워크플로우](./02_Version_Control_Workflows.md) | **다음**: [GitHub Actions 심화](./04_GitHub_Actions_Deep_Dive.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 지속적 통합의 원칙과 빈번한 통합이 리스크를 줄이는 이유를 설명할 수 있다
2. 빌드, 테스트, 배포 단계와 적절한 품질 게이트를 포함하는 CI 파이프라인을 설계할 수 있다
3. 지속적 통합, 지속적 전달, 지속적 배포를 구분할 수 있다
4. 빌드 산출물에 대한 아티팩트 관리 전략을 구현할 수 있다
5. 다양한 시나리오(push, pull request, 예약, 수동)에 대한 빌드 트리거를 구성할 수 있다
6. 빠른 피드백, 밀폐형 빌드, 테스트 병렬화를 포함한 CI 모범 사례를 적용할 수 있다

---

지속적 통합(CI)은 개발자가 변경 사항을 푸시할 때마다 자동으로 코드를 빌드하고 테스트하는 실천 방법입니다. 1990년대 후반 Extreme Programming 커뮤니티에 의해 개척되었으며, 2006년 Martin Fowler의 영향력 있는 글에서 공식화되었습니다. CI는 전체 DevOps 파이프라인의 기초입니다. 신뢰할 수 있고 빠른 CI 프로세스 없이는 지속적 전달과 지속적 배포가 불가능합니다. 이 레슨에서는 효과적인 CI 파이프라인을 구축하기 위한 개념, 아키텍처, 모범 사례를 다룹니다.

> **비유 -- 조립 라인 품질 관리:** CI는 끝에서 한 번의 대규모 검사를 하는 대신, 조립 라인의 모든 공정에서 품질 검사를 하는 것과 같습니다. 용접이 불량하면 작업자가 즉시 발견합니다. 공정별 검사가 없으면 불량 용접이 전체 조립 과정에 전파되고, 최종 검사에서 거의 완성된 제품을 불합격 처리하게 됩니다 -- 모든 후속 작업이 낭비됩니다.

## 1. 지속적 통합이란?

지속적 통합은 개발자가 공유 메인라인에 코드 변경 사항을 **빈번하게 머지**하고, 각 머지가 **자동화된 빌드 및 테스트 스위트에 의해 검증**되는 실천 방법입니다.

### 핵심 원칙

```
┌──────────────────────────────────────────────────────────────┐
│                   CI Core Principles                          │
│                                                               │
│  1. Maintain a single source repository                      │
│  2. Automate the build                                       │
│  3. Make the build self-testing                               │
│  4. Everyone commits to mainline every day                   │
│  5. Every commit triggers a build                            │
│  6. Keep the build fast (< 10 minutes ideal)                 │
│  7. Test in a clone of the production environment            │
│  8. Make the latest build artifacts easy to access            │
│  9. Everyone can see what is happening (transparency)         │
│  10. Automate deployment                                      │
│                                                               │
│  -- Martin Fowler, "Continuous Integration" (2006)           │
└──────────────────────────────────────────────────────────────┘
```

### CI 피드백 루프

```
Developer pushes code
        │
        ▼
┌─────────────────┐
│   CI Server     │
│   detects push  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Build       │────▶│     Test        │────▶│   Report        │
│  (compile,      │     │  (unit, lint,   │     │  (pass/fail,    │
│   package)      │     │   integration)  │     │   coverage)     │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                  ┌───────────────┐
                                                  │  Notify dev   │
                                                  │  (Slack, email,│
                                                  │   PR status)   │
                                                  └───────────────┘
         Feedback time: ideally < 10 minutes
```

---

## 2. CI vs CD vs CD

이 세 가지 용어는 종종 혼동되지만 자동화 수준이 점진적으로 증가하는 별개의 실천 방법을 나타냅니다.

### 지속적 통합 (CI)

```
Code ──▶ Build ──▶ Unit Tests ──▶ Integration Tests ──▶ Feedback
                                                         │
                                            Developers fix issues
                                            before merging
```

**정의**: 모든 코드 변경을 자동으로 빌드하고 테스트합니다.
**산출물**: main 브랜치의 검증되고 테스트된 코드.
**사람의 역할**: 개발자가 언제 릴리스할지 결정해야 합니다.

### 지속적 전달 (Continuous Delivery)

```
Code ──▶ Build ──▶ Test ──▶ Staging Deploy ──▶ Acceptance Tests
                                                       │
                                               Manual approval
                                                       │
                                                       ▼
                                              Production Deploy
```

**정의**: 모든 변경이 자동으로 빌드, 테스트되고 릴리스 준비가 됩니다. 프로덕션 배포에는 **수동 승인**(버튼 클릭)이 필요합니다.
**산출물**: 항상 릴리스 준비된 아티팩트.
**사람의 역할**: 누군가가 "프로덕션에 배포"를 클릭합니다.

### 지속적 배포 (Continuous Deployment)

```
Code ──▶ Build ──▶ Test ──▶ Staging Deploy ──▶ Acceptance Tests
                                                       │
                                            Automatic (no human)
                                                       │
                                                       ▼
                                              Production Deploy
```

**정의**: 자동화된 테스트를 통과한 모든 변경이 **프로덕션에 자동으로 배포**됩니다. 사람의 승인 게이트가 없습니다.
**산출물**: 커밋 후 수 분 내에 프로덕션의 코드.
**사람의 역할**: 없음 (초기 설정 이후).

### 비교 표

| 측면 | CI | 지속적 전달 | 지속적 배포 |
|------|----|-----------|-----------|
| 빌드 자동화 | 예 | 예 | 예 |
| 테스트 자동화 | 예 | 예 | 예 |
| 스테이징 배포 | 선택 | 자동 | 자동 |
| 프로덕션 배포 | 수동 | 수동 (원클릭) | 자동 |
| 롤백 | 수동 | 수동 또는 자동 | 자동 |
| 릴리스 빈도 | 팀 일정에 따라 | 팀이 결정할 때 | 모든 성공적인 빌드마다 |
| 배포당 리스크 | 다양 | 낮음 | 매우 낮음 |

---

## 3. CI 파이프라인 아키텍처

### 파이프라인 단계

일반적인 CI 파이프라인은 순차적인 단계로 구성되며, 각 단계에는 하나 이상의 병렬 작업이 포함됩니다.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Source   │───▶│  Build   │───▶│  Test    │───▶│ Security │───▶│ Artifact │
│          │    │          │    │          │    │  Scan    │    │ Publish  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
   trigger        compile        unit tests     SAST/SCA        docker push
   checkout        package       integration    license check   upload to
   dependencies    lint          coverage                       registry
```

### 단계별 상세

#### 소스 단계

```bash
# Triggered by a code push or pull request
# CI server checks out the code

git clone --depth 1 https://github.com/org/repo.git
cd repo
git checkout $COMMIT_SHA
```

#### 빌드 단계

```bash
# Install dependencies
pip install -r requirements.txt

# Compile (if applicable)
go build ./...

# Lint code
flake8 src/
eslint src/**/*.js

# Type check
mypy src/
```

#### 테스트 단계

```bash
# Unit tests (fast, isolated)
pytest tests/unit/ --junitxml=reports/unit.xml

# Integration tests (may need external services)
docker compose up -d postgres redis
pytest tests/integration/ --junitxml=reports/integration.xml
docker compose down

# Code coverage
pytest --cov=src --cov-report=xml:reports/coverage.xml

# Coverage threshold enforcement
coverage report --fail-under=80
```

#### 보안 스캔 단계

```bash
# Static Application Security Testing (SAST)
bandit -r src/ -f json -o reports/sast.json

# Software Composition Analysis (SCA)
safety check -r requirements.txt --json > reports/sca.json

# Container image scanning
trivy image myapp:latest --format json --output reports/trivy.json
```

#### 아티팩트 단계

```bash
# Build Docker image
docker build -t myapp:${COMMIT_SHA} .

# Tag with version
docker tag myapp:${COMMIT_SHA} registry.example.com/myapp:${COMMIT_SHA}
docker tag myapp:${COMMIT_SHA} registry.example.com/myapp:latest

# Push to registry
docker push registry.example.com/myapp:${COMMIT_SHA}
docker push registry.example.com/myapp:latest
```

---

## 4. 빌드 트리거

### 트리거 유형

| 트리거 | 시점 | 사용 사례 |
|--------|------|-----------|
| **Push** | 브랜치에 코드 푸시 | 모든 커밋에 CI 실행 |
| **Pull request** | PR 생성/업데이트 | 머지 전 검증 |
| **Schedule** | cron 기반 타이밍 | 야간 빌드, 의존성 검사 |
| **Manual** | 사용자가 버튼 클릭 | 프로덕션 배포, 일회성 작업 |
| **Tag** | Git 태그 생성 | 릴리스 빌드 |
| **API/Webhook** | 외부 이벤트 | 크로스 저장소 트리거, ChatOps |

### 트리거 구성 예시

```yaml
# GitHub Actions trigger examples
on:
  # Run on push to main or develop
  push:
    branches: [main, develop]
    paths-ignore:
      - '*.md'
      - 'docs/**'

  # Run on PRs targeting main
  pull_request:
    branches: [main]

  # Nightly security scan at 2 AM UTC
  schedule:
    - cron: '0 2 * * *'

  # Manual trigger with parameters
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        type: choice
        options:
          - staging
          - production
```

### 경로 기반 트리거링

```yaml
# Only run backend CI when backend code changes
# Avoids wasting CI resources on unrelated changes
on:
  push:
    paths:
      - 'src/backend/**'
      - 'tests/backend/**'
      - 'requirements.txt'
      - '.github/workflows/backend-ci.yml'
```

---

## 5. 아티팩트 관리

아티팩트는 CI 빌드의 산출물로, 컴파일된 바이너리, Docker 이미지, 테스트 리포트, 패키지 등입니다.

### 아티팩트 유형

| 아티팩트 | 형식 | 저장소 |
|----------|------|--------|
| Docker 이미지 | OCI 이미지 레이어 | 컨테이너 레지스트리 (ECR, GCR, Docker Hub) |
| Python 패키지 | `.whl`, `.tar.gz` | PyPI, 프라이빗 인덱스 |
| npm 패키지 | `.tgz` | npm 레지스트리, Artifactory |
| Java 패키지 | `.jar`, `.war` | Maven Central, Nexus |
| 바이너리 | 플랫폼별 실행 파일 | GitHub Releases, S3 |
| 테스트 리포트 | JUnit XML, HTML | CI 서버, S3 |
| 커버리지 리포트 | XML, HTML | CI 서버, Codecov |

### 버저닝 전략

```bash
# Semantic Versioning (SemVer): MAJOR.MINOR.PATCH
# MAJOR: breaking changes
# MINOR: new features, backwards compatible
# PATCH: bug fixes, backwards compatible

v1.0.0   # Initial release
v1.1.0   # New feature added
v1.1.1   # Bug fix
v2.0.0   # Breaking change

# Git-based versioning for Docker images
# Use commit SHA for traceability
docker build -t myapp:abc123f .

# CalVer (Calendar Versioning)
# Used by Ubuntu (22.04), pip (24.0)
v2024.03.15

# Build number
build-42
```

### Docker 이미지 태깅 전략

```bash
# Tag with multiple identifiers for flexibility
IMAGE=registry.example.com/myapp

# Commit SHA -- exact traceability
docker tag myapp $IMAGE:${GIT_SHA:0:7}

# Branch name -- latest build of a branch
docker tag myapp $IMAGE:${BRANCH_NAME}

# Semantic version (on tagged releases)
docker tag myapp $IMAGE:v1.2.3
docker tag myapp $IMAGE:v1.2
docker tag myapp $IMAGE:v1

# Latest (use with caution -- ambiguous in production)
docker tag myapp $IMAGE:latest
```

---

## 6. CI 모범 사례

### 빌드 속도 유지

```
Target build times:
  Unit tests:        < 2 minutes
  Full CI pipeline:  < 10 minutes
  Nightly/full:      < 30 minutes

Techniques for speed:
  ✓ Parallelize tests across multiple runners
  ✓ Cache dependencies between builds
  ✓ Use incremental builds (only rebuild what changed)
  ✓ Run slow tests in a separate pipeline (nightly)
  ✓ Use fast base images (Alpine, distroless)
```

### 테스트 피라미드

```
                    /\
                   /  \          End-to-End Tests
                  / E2E\         (few, slow, expensive)
                 /──────\
                /        \       Integration Tests
               /Integration\     (moderate number)
              /──────────────\
             /                \   Unit Tests
            /    Unit Tests    \  (many, fast, cheap)
           /────────────────────\

  More unit tests, fewer E2E tests.
  Fast feedback at the base, confidence at the top.
```

### 밀폐형 빌드

밀폐형(격리된) 빌드는 실행 장소나 시점에 관계없이 동일한 결과를 생성합니다.

```dockerfile
# Bad: Non-hermetic -- depends on host state
FROM python:3.11
RUN pip install -r requirements.txt  # Versions may change over time
COPY . .

# Good: Hermetic -- pinned versions, reproducible
FROM python:3.11.7-slim@sha256:abc123...  # Pinned image digest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  # requirements.txt has pinned versions
COPY . .
```

```
# requirements.txt -- pin exact versions
flask==3.0.2
requests==2.31.0
SQLAlchemy==2.0.25
```

### 빠른 실패

```yaml
# Run linting and type checking first (fastest checks)
# If they fail, don't waste time on slower tests
stages:
  - lint        # 30 seconds
  - build       # 1 minute
  - unit-test   # 2 minutes
  - integration # 5 minutes
  - e2e         # 10 minutes

# Each stage only runs if previous stages passed
```

### 코드로서의 CI 파이프라인

```yaml
# Store CI configuration in the repository alongside the code
# This ensures:
#   - CI changes are reviewed like code (via PRs)
#   - CI configuration is versioned with the code
#   - Anyone can understand the build process
#   - Reproducible builds across branches

# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install flake8
      - run: flake8 src/

  test:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest tests/ --junitxml=reports/test-results.xml
      - uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: reports/test-results.xml
```

---

## 7. 일반적인 CI 패턴

### 매트릭스 빌드

여러 버전, OS 또는 구성에서 동시에 테스트합니다.

```yaml
jobs:
  test:
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
        os: [ubuntu-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pytest tests/
```

### Fan-Out / Fan-In

독립적인 작업을 병렬로 실행한 후 결과를 집계합니다.

```
              ┌── Unit Tests ─────────┐
              │                        │
Build ───────├── Integration Tests ──├──── Deploy
              │                        │
              └── Security Scan ──────┘
              (fan-out: parallel)    (fan-in: wait for all)
```

### 카나리 빌드

전체 롤아웃 전에 프로덕션의 일부에 먼저 배포합니다.

```
Build ──▶ Test ──▶ Deploy to 5% ──▶ Monitor (30 min)
                                          │
                                    ┌─────┴─────┐
                                    │            │
                               Healthy?     Unhealthy?
                                    │            │
                                    ▼            ▼
                             Deploy to       Rollback
                               100%
```

---

## 8. CI 메트릭 및 모니터링

### 주요 CI 메트릭

| 메트릭 | 목표 | 중요한 이유 |
|--------|------|------------|
| **빌드 소요 시간** | < 10분 | 느린 빌드는 개발자 생산성을 저해 |
| **빌드 성공률** | > 95% | 불안정한 빌드는 CI에 대한 신뢰를 침식 |
| **대기 시간** | < 2분 | 긴 대기열은 러너가 부족함을 의미 |
| **테스트 커버리지** | > 80% | 임계값 미만의 커버리지는 리스크를 증가 |
| **불안정 테스트율** | < 1% | 불안정 테스트는 시간 낭비와 불신을 초래 |
| **깨진 빌드 MTTR** | < 30분 | 깨진 main은 전체 팀을 차단 |

### 빌드 상태 모니터링

```bash
# Track build times over the last 30 days
# (pseudo-query for a CI analytics platform)
SELECT
  date_trunc('day', started_at) as day,
  avg(duration_seconds) as avg_duration,
  count(*) as build_count,
  count(*) filter (where status = 'success') * 100.0 / count(*) as success_rate
FROM builds
WHERE started_at > now() - interval '30 days'
GROUP BY 1
ORDER BY 1;
```

---

## 연습 문제

### 연습 문제 1: CI 파이프라인 설계

Python 웹 애플리케이션(Flask + PostgreSQL)을 위한 완전한 CI 파이프라인을 설계하십시오. 다음을 명시하십시오:
1. 모든 단계와 순서
2. 각 단계에서 실행되는 내용 (구체적인 명령어)
3. 병렬로 실행할 수 있는 단계
4. 품질 게이트 (빌드를 실패시키는 조건)
5. 생성되는 아티팩트
6. 예상 총 파이프라인 소요 시간

### 연습 문제 2: CI vs CD 결정

팀에 현재 CI(자동 빌드 및 테스트)가 있습니다. CTO가 지속적 배포(통과한 모든 빌드를 프로덕션에 자동 배포)로 전환하길 원합니다. 다음을 나열하십시오:
1. 지속적 배포를 도입하기 전에 팀이 갖춰야 할 다섯 가지 전제 조건
2. 지속적 배포를 너무 일찍 도입할 때의 세 가지 리스크
3. CI에서 지속적 전달, 지속적 배포로 이동하는 단계적 계획

### 연습 문제 3: 불안정 테스트 분류

CI 파이프라인의 불안정 테스트율이 15%입니다. 개발자들이 실패를 무시하고 빌드가 통과할 때까지 재실행합니다.
1. 이것이 왜 위험한지 설명하십시오
2. 불안정 테스트를 식별하고 분류하기 위한 체계적인 접근법을 제안하십시오
3. 실제 실패를 숨기지 않는 불안정 테스트 격리 메커니즘을 설계하십시오
4. 시간에 따른 불안정 테스트 감소를 추적하기 위한 메트릭을 정의하십시오

### 연습 문제 4: 빌드 시간 최적화

CI 파이프라인이 25분 걸립니다. 내역은 다음과 같습니다:
- 체크아웃 + 의존성 설치: 4분
- Lint + 타입 검사: 2분
- 단위 테스트: 5분
- 통합 테스트: 10분
- Docker 빌드 + 푸시: 4분

총 시간을 10분 이내로 줄이기 위한 최소 다섯 가지 구체적인 최적화를 제안하십시오. 각 최적화에 대해 예상 시간 절감과 트레이드오프를 설명하십시오.

---

**이전**: [버전 관리 워크플로우](./02_Version_Control_Workflows.md) | [개요](00_Overview.md) | **다음**: [GitHub Actions 심화](./04_GitHub_Actions_Deep_Dive.md)

**License**: CC BY-NC 4.0
