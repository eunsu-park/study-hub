# 레슨 09: 형상 관리(Configuration Management)

**이전**: [08. 검증과 확인](./08_Verification_and_Validation.md) | **다음**: [10. 프로젝트 관리](./10_Project_Management.md)

---

소프트웨어 시스템은 결코 완성되지 않습니다 — 지속적으로 변경되고, 확장되며, 다양한 환경에 배포됩니다. 이러한 변경을 규율 있게 관리하지 않으면, 프로젝트는 혼돈에 빠집니다: 팀원들이 서로의 작업을 덮어쓰고, 운영 환경에서 아무도 재현할 수 없는 버전이 실행되며, 핫픽스 하나가 찾는 데 석 달이 걸리는 회귀 버그를 유발합니다. 소프트웨어 형상 관리(Software Configuration Management, SCM)는 이를 방지하는 규율입니다. 모든 산출물에 정확한 식별자를 부여하고, 변경 방식을 통제하며, 언제든 시스템의 특정 버전을 재현 가능하게 만듭니다.

**난이도**: ⭐⭐⭐

**선수 학습**:
- 버전 관리(Git 토픽 권장)에 대한 기본 이해
- 소프트웨어 개발 생명주기 이해 (레슨 02)
- 일반 프로그래밍 경험

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 소프트웨어 형상 관리(SCM)를 정의하고 소프트웨어 공학에서의 역할을 설명한다
2. 형상 항목(Configuration Item)을 식별·분류하고 기준선(Baseline)을 수립한다
3. 버전 관리를 위한 브랜치 전략을 설명하고 비교한다
4. 빌드 관리와 재현 가능한 빌드의 특성을 설명한다
5. 시맨틱 버저닝(Semantic Versioning)을 적용하고 릴리스 프로세스를 설계한다
6. 공식적인 변경 관리 프로세스의 단계를 구별한다
7. 환경 관리와 코드형 인프라(Infrastructure as Code)와의 관계를 설명한다
8. 잠금 파일(Lockfile), 버전 고정(Pinning), 취약점 스캔을 활용하여 의존성을 안전하게 관리한다

---

## 목차

1. [소프트웨어 형상 관리란?](#1-소프트웨어-형상-관리란)
2. [형상 항목과 기준선](#2-형상-항목과-기준선)
3. [버전 관리 개념](#3-버전-관리-개념)
4. [빌드 관리](#4-빌드-관리)
5. [릴리스 관리](#5-릴리스-관리)
6. [변경 관리](#6-변경-관리)
7. [형상 감사](#7-형상-감사)
8. [환경 관리](#8-환경-관리)
9. [의존성 관리](#9-의존성-관리)
10. [도구 개요](#10-도구-개요)
11. [요약](#11-요약)
12. [연습 문제](#12-연습-문제)
13. [더 읽을거리](#13-더-읽을거리)

---

## 1. 소프트웨어 형상 관리란?

### 1.1 정의

**소프트웨어 형상 관리(Software Configuration Management, SCM)**는 요구사항 문서부터 운영 바이너리까지, 소프트웨어 개발 과정에서 생산되는 모든 산출물에 대한 변경을 식별, 추적, 통제하는 규율입니다.

IEEE는 이를 다음과 같이 정의합니다:
> "형상 항목의 기능적·물리적 특성을 식별하고 문서화하며, 해당 특성에 대한 변경을 통제하고, 변경 처리 및 구현 상태를 기록·보고하며, 명시된 요구사항과의 준수 여부를 검증하기 위해 기술적·관리적 지도와 감시를 적용하는 규율."

다소 관료적으로 들릴 수 있지만, SCM은 실제로 시급한 문제들을 해결합니다:

| SCM 없이 발생하는 문제 | SCM 해결책 |
|---------------------|--------------|
| "운영 환경에 어떤 버전이 배포되어 있는가?" | 기준선(Baseline)과 릴리스 태깅 |
| "누가 빌드를 망쳤고 어떻게 된 건가?" | 버전 관리 이력 + CI/CD 로그 |
| "지난 달 릴리스를 재현할 수 없다" | 잠긴 의존성, VCS의 빌드 스크립트 |
| "두 개발자가 같은 파일을 편집했다" | 브랜치 전략과 병합 워크플로우 |
| "모듈 A의 변경이 모듈 B를 망가뜨렸다" | 변경 관리 + 영향 분석 |
| "개발 환경이 운영 환경과 다르게 동작한다" | 환경 관리 / 코드형 인프라(IaC) |

### 1.2 SCM의 네 가지 핵심 기능

```
┌─────────────────────────────────────────────────────────────────┐
│  소프트웨어 형상 관리(Software Configuration Management)         │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  식별(Identification)│    │  통제(Control)   │                  │
│  │                  │    │                  │                  │
│  │ 모든 산출물에    │    │ 변경을 체계적으로│                  │
│  │ 이름 부여 및 추적│    │ 관리            │                  │
│  └──────────────────┘    └──────────────────┘                  │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  회계(Accounting) │    │  감사(Auditing)   │                  │
│  │                  │    │                  │                  │
│  │ 모든 CI의 상태   │    │ 산출물이 기준선  │                  │
│  │ 기록; 변경 이력  │    │ 과 일치하는지    │                  │
│  │ 보고            │    │ 검증; 프로세스 확인│                  │
│  └──────────────────┘    └──────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 SCM과 DevOps

SCM은 DevOps보다 수십 년 앞서 등장했습니다 (1950년대 방위 계약에서 유래). DevOps는 SCM의 원칙을 채택하고 자동화합니다:

| SCM 개념 | DevOps/현대적 실천 |
|-------------|------------------------|
| 형상 식별(Configuration identification) | Git 커밋과 태그; Docker 이미지 다이제스트 |
| 버전 관리(Version control) | Git, GitHub/GitLab |
| 변경 통제(Change control) | 풀 리퀘스트(Pull Request), 코드 리뷰 |
| 빌드 관리(Build management) | CI 파이프라인 (GitHub Actions, Jenkins) |
| 릴리스 관리(Release management) | CD 파이프라인, GitOps |
| 환경 관리(Environment management) | 코드형 인프라(Terraform, Ansible) |
| 감사(Auditing) | 감사 로그, SBOM (소프트웨어 부품 명세서) |

---

## 2. 형상 항목과 기준선

### 2.1 형상 항목(Configuration Item)

**형상 항목(Configuration Item, CI)**은 형상 통제 대상에 포함되는 모든 산출물 — 고유한 식별자가 부여되고 변경이 추적되는 것 — 입니다. CI는 소스 코드만을 의미하지 않습니다.

| 범주 | 예시 |
|----------|----------|
| **소스 코드** | 애플리케이션 코드, 스크립트, 테스트 코드, 데이터베이스 마이그레이션 |
| **빌드 산출물** | 컴파일된 바이너리, 컨테이너 이미지, 패키지 |
| **문서** | 요구사항, 설계 문서, 사용자 매뉴얼, API 명세 |
| **테스트 산출물** | 테스트 계획, 테스트 케이스, 테스트 데이터, 테스트 스크립트 |
| **설정 파일** | `application.yaml`, `.env.example`, nginx 설정, Kubernetes 매니페스트 |
| **서드파티 컴포넌트** | 벤더 라이브러리, 라이선스 파일 |
| **인프라 코드** | Terraform `.tf` 파일, Ansible 플레이북, Dockerfile |
| **프로젝트 관리** | 프로젝트 계획, 위험 관리 대장, 릴리스 노트 |

**CI 선정 기준**: 다음 조건을 만족하면 CI로 지정해야 합니다:
- 변경을 추적하고 감사해야 하는 경우
- 동시에 여러 버전이 존재하는 경우
- 해당 산출물의 무결성이 시스템 무결성에 영향을 미치는 경우
- 이전 상태로 되돌릴 필요가 있을 수 있는 경우

### 2.2 CI 명명 및 식별

모든 CI에는 다음을 포함하는 고유 식별자가 필요합니다:
- **이름(Name)**: 설명적이고 명명 규칙과 일관된 이름
- **유형(Type)**: 소스, 문서, 테스트 산출물 등
- **버전(Version)**: 개정판을 구별하는 번호 또는 해시
- **변형(Variant)**: 선택적, 플랫폼별 버전용 (예: `linux-amd64`, `macos-arm64`)

```
명명 체계 예시:
  {project}-{component}-{version}-{variant}.{ext}

  myapp-api-1.4.2-linux-amd64.tar.gz
  myapp-api-1.4.2-windows-amd64.zip
  myapp-docs-1.4.2.pdf
```

### 2.3 기준선(Baseline)

**기준선(Baseline)**은 하나 이상의 CI에 대해 공식적으로 검토·합의된 스냅샷으로, 이후 개발의 고정된 참조점 역할을 합니다. 특정 시점의 "공식 버전"입니다.

| 기준선 유형 | 수립 시점 | 내용 |
|---------------|----------------|----------|
| **기능 기준선(Functional Baseline)** | 시스템 요구사항 검토 후 | 승인된 요구사항 명세 |
| **할당 기준선(Allocated Baseline)** | 예비 설계 검토 후 | 승인된 아키텍처 및 고수준 설계 |
| **제품 기준선(Product Baseline)** | 최종 인수 테스트 후 | 릴리스를 위한 전체 소스 코드, 문서, 테스트 |
| **운영 기준선(Operational Baseline)** | 운영 중 | 모든 패치를 포함한 배포 구성 |

현대적 실천에서 기준선은 메인 브랜치의 Git 태그에 해당합니다:

```bash
# v1.4.2 제품 기준선 수립
git tag -a v1.4.2 -m "Product baseline: Q1 2024 release

Approved by: Product team 2024-03-28
Change control ticket: CCB-2024-031
Includes: api-service, worker-service, migrations 001-047"

git push origin v1.4.2
```

---

## 3. 버전 관리 개념

버전 관리 시스템(VCS)은 SCM의 기술적 근간입니다. 이 절은 전략에 초점을 맞추며, 세부 기술은 Git 토픽에서 다룹니다.

### 3.1 브랜치 전략

브랜치 전략은 팀이 동시 개발과 릴리스를 관리하기 위해 브랜치를 사용하는 방법을 정의합니다.

#### Gitflow

정기적 릴리스(월별/분기별)를 하는 팀에 적합합니다.

```
main          ─────────────●──────────────────●─────────
                          ↑ v1.0              ↑ v2.0
develop       ──────●──────────────●──────────────────●──
                    ↑              ↑
feature/login ──────●              │
                                   │
feature/pay   ────────────────────●│
                                    │
release/2.0   ───────────────────────●────────●──────────
                                    (test)   (merge)
hotfix/2.0.1  ─────────────────────────────────────●─────
```

**브랜치 구성**:
- `main`: 운영 준비된 코드만 포함; 항상 릴리스 가능
- `develop`: 통합 브랜치; 최신 개발 변경사항
- `feature/*`: 개별 기능, develop에서 분기
- `release/*`: 릴리스 준비 (버그 수정만, 신기능 없음)
- `hotfix/*`: 긴급 운영 수정, main에서 분기

#### 트렁크 기반 개발(Trunk-Based Development, TBD)

지속적 배포(매일 또는 더 자주 릴리스)를 하는 팀에 적합합니다.

```
main (trunk)  ──●──●──●──●──●──●──●──●──●──●──●──●──
               ↑  ↑     ↑           ↑
             feat1 feat2 feat3     feat4
              (단기 브랜치, 1~2일 이내)
```

모든 개발자가 최소 하루에 한 번 main에 통합합니다. 기능 플래그(Feature Flag)가 운영 환경에서 무엇을 활성화할지 제어하여, 배포와 릴리스를 분리합니다.

#### 전략 선택

| 요소 | Gitflow | 트렁크 기반 개발 |
|--------|---------|-------------|
| 릴리스 주기 | 정기적 (주/월 단위) | 지속적 (매일) |
| 팀 규모 | 모든 규모 | 소규모·훈련된 팀에 더 적합 |
| 테스트 자동화 | 중간 수준 허용 | 강력한 자동화 테스트 필수 |
| 기능 플래그 필요 여부 | 불필요 | 필요 |
| 장기 브랜치 | 있음 (기능 브랜치) | 없음 (2일 미만) |

### 3.2 태깅과 릴리스 브랜치

태그는 특정 커밋을 중요한 것으로 표시합니다. 모든 릴리스에는 태그를 붙여야 합니다.

```bash
# 어노테이션 태그 (릴리스에 권장 — 메타데이터 포함)
git tag -a v1.4.2 -m "Release v1.4.2 - Q1 2024"

# 버전 순으로 태그 목록 표시
git tag --sort=version:refname

# 이전 버전에 긴급 패치를 위한 릴리스 브랜치 생성
git checkout -b release/1.3 v1.3.0
git cherry-pick abc1234  # 핫픽스 적용
git tag -a v1.3.1 -m "Hotfix v1.3.1"
```

---

## 4. 빌드 관리

**빌드(Build)**는 소스 코드와 리소스를 배포 가능한 산출물로 변환하는 프로세스입니다. 빌드 관리는 이 프로세스가 신뢰할 수 있고, 재현 가능하며, 자동화되도록 보장합니다.

### 4.1 좋은 빌드 시스템의 특성

| 특성 | 설명 | 예시 |
|----------|-------------|---------|
| **재현 가능(Reproducible)** | 동일한 입력은 항상 동일한 출력을 생성 | 잠긴 의존성; 무작위 요소 없음 |
| **자동화(Automated)** | 수동 단계 불필요 | CI 파이프라인이 모든 커밋에서 빌드 실행 |
| **빠름(Fast)** | 증분 빌드 — 변경된 것만 재빌드 | Make, Gradle 증분 컴파일 |
| **멱등성(Idempotent)** | 빌드를 두 번 실행해도 동일한 결과 | 빌드 자체에 부작용 없음 |
| **문서화(Documented)** | 빌드 프로세스가 구전이 아닌 코드로 정의됨 | VCS의 `Makefile`, `build.gradle`, `pyproject.toml` |
| **검증(Verified)** | 빌드에 품질 게이트 포함 | 파이프라인의 테스트, 린팅, 보안 스캔 |

### 4.2 빌드 스크립트와 도구

```makefile
# Makefile: 폴리글랏 프로젝트의 공통 빌드 자동화
.PHONY: all clean test lint build docker

# 기본 타겟
all: lint test build

# 의존성 설치
deps:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# 린터 실행
lint:
	flake8 src/ --max-line-length=100
	mypy src/

# 커버리지와 함께 테스트 실행
test:
	pytest tests/ -v --cov=src --cov-fail-under=85

# 운영 산출물 빌드
build:
	python -m build --wheel
	@echo "Build artifact: dist/"

# git SHA를 태그로 사용하여 Docker 이미지 빌드
docker:
	docker build \
	  --build-arg GIT_SHA=$(shell git rev-parse --short HEAD) \
	  --build-arg BUILD_DATE=$(shell date -u +%Y-%m-%dT%H:%M:%SZ) \
	  -t myapp:$(shell git rev-parse --short HEAD) \
	  -t myapp:latest \
	  .

# 생성된 산출물 정리
clean:
	rm -rf dist/ build/ *.egg-info .coverage htmlcov/ .mypy_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
```

### 4.3 재현 가능한 빌드(Reproducible Build)

**재현 가능한 빌드**는 동일한 소스를 입력받으면 비트 단위로 동일한 출력을 생성합니다. 이를 통해:
- 배포된 바이너리가 공개된 소스 코드에 해당하는지 검증 가능
- 공급망 공격(빌드 환경 변조) 탐지 가능
- 빌드 산출물의 신뢰할 수 있는 캐싱 가능

```dockerfile
# 재현성을 위해 설계된 Dockerfile
# 1. 기본 이미지를 태그가 아닌 특정 다이제스트로 고정
FROM python:3.12.3-slim@sha256:a1e3204e39b5f3e2c3f74bc7fdf14e2ae1dfba9ab8bb1cde0f3a5b1c5e2c2d3f

# 2. 파일 메타데이터의 고정된 타임스탬프 설정
ARG SOURCE_DATE_EPOCH=1711670400

# 3. 모든 시스템 패키지 고정
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5=15.6-0+deb12u1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. 잠금 파일(lockfile)에서 설치 (requirements.txt만 사용하지 않음)
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY src/ ./src/

# 5. 출처(provenance) 메타데이터 포함
ARG GIT_SHA
ARG BUILD_DATE
LABEL org.opencontainers.image.revision=$GIT_SHA \
      org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.source="https://github.com/org/myapp"

CMD ["python", "-m", "myapp"]
```

### 4.4 아티팩트 저장소(Artifact Repository)

빌드 산출물은 **아티팩트 저장소** — 바이너리, 패키지, 컨테이너 이미지를 위한 버전 관리되고 검색 가능한 저장소 — 에 보관해야 합니다.

| 아티팩트 유형 | 저장소 |
|---------------|------------|
| Python 패키지 | PyPI, Artifactory, AWS CodeArtifact |
| Java/JVM 패키지 | Maven Central, Nexus |
| Docker 이미지 | Docker Hub, ECR, GCR, GHCR |
| Helm 차트 | Helm repo, OCI 레지스트리 |
| 일반 바이너리 | Artifactory, S3 (버전 관리 활성화) |

배포 시 소스에서 빌드하지 마십시오. 한 번 빌드하고, 산출물을 저장하고, 산출물을 배포하십시오.

---

## 5. 릴리스 관리

릴리스 관리는 소프트웨어 버전이 패키징되고, 버전이 지정되며, 사용자에게 전달되는 방식을 관장합니다.

### 5.1 시맨틱 버저닝(Semantic Versioning, SemVer)

시맨틱 버저닝(semver.org)은 버전 번호의 범용적인 문법을 제공합니다:

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

예시:
  1.0.0          초기 안정 릴리스
  1.0.1          패치: 하위 호환 버그 수정
  1.1.0          마이너: 하위 호환 신기능 추가
  2.0.0          메이저: 호환성을 깨는 변경 (API 비호환)
  2.1.0-alpha.1  프리릴리스: 불안정, 운영 사용 불가
  2.1.0-rc.1     릴리스 후보: 기능 완성, 최종 테스트 단계
  1.0.0+build.42 빌드 메타데이터: 정보 제공용, 우선순위 비교에서 무시
```

**버전 우선순위 규칙**:
```
1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-beta < 1.0.0-rc.1 < 1.0.0
```

**증가 시점**:
- **PATCH**: 공개 API를 변경하지 않고 버그를 수정할 때
- **MINOR**: 하위 호환 방식으로 기능을 추가할 때
- **MAJOR**: 비호환 API 변경을 할 때

라이브러리의 경우, 엄격한 SemVer는 하위 소비자를 보호합니다. 애플리케이션(라이브러리가 아닌)의 경우, MAJOR 버전은 종종 제품 세대나 연간 릴리스를 추적합니다.

### 5.2 릴리스 노트(Release Notes)

릴리스 노트는 변경사항을 사용자에게 전달합니다. 좋은 릴리스 노트는 다음 질문에 답합니다:
- 무엇이 변경되었는가?
- 나(사용자)에게 왜 중요한가?
- 내가 무언가를 해야 하는가?

```markdown
# Release Notes: v2.3.0 (2024-03-28)

## What's New

- **Bulk export**: export up to 10,000 records to CSV in one operation (#1847)
- **Dark mode**: full dark mode support across all screens (#2103)
- **API rate limiting**: new `X-RateLimit-*` response headers (#2251)

## Bug Fixes

- Fixed: session expired users were redirected to blank page (#2289)
- Fixed: decimal amounts rounded incorrectly for JPY and KRW (#2301)

## Breaking Changes

- **API**: `GET /users` no longer returns the `password_hash` field.
  Update any clients that read this field. (#2275)

## Deprecations

- `POST /api/v1/auth/login` is deprecated. Use `POST /api/v2/auth/login`.
  v1 will be removed in v3.0.0 (Q3 2024).

## Upgrade Notes

Run the following migration before upgrading:
```
python manage.py migrate --run-syncdb
```

## Known Issues

- Safari 16.x: dark mode toggle may require a page refresh.
  Workaround: use the keyboard shortcut Cmd+Shift+D.
```

### 5.3 릴리스 프로세스

```
기능 동결(Feature freeze)
      │
      ▼
릴리스 브랜치 생성 (release/2.3)
      │
      ▼
릴리스 브랜치에서 회귀 테스트
      │
      ├── 버그 발견? → 릴리스 브랜치에서 수정 → main으로 cherry-pick
      │
      ▼
릴리스 후보 (v2.3.0-rc.1)
      │
      ▼
스테이징 배포 + 인수 테스트
      │
      ├── 문제 발견? → rc 번호 증가 (v2.3.0-rc.2)
      │
      ▼
운영 배포 (단계적/블루-그린)
      │
      ▼
커밋 태깅: v2.3.0
      │
      ▼
릴리스 브랜치를 main에 병합
      │
      ▼
릴리스 노트 게시 + 이해관계자 통보
```

---

## 6. 변경 관리

변경 관리는 기준선이 설정된 CI에 대한 모든 수정이 통제된 프로세스를 따르도록 보장합니다. 시스템 무결성을 훼손하는 임의적인 변경을 방지합니다.

### 6.1 변경 요청 프로세스

```
1. 개시(INITIATION)
   누구나 변경 요청(CR) 또는 변경 요구서(RFC)를 제출할 수 있음
   └── CR 포함 내용: 설명, 정당성, 영향 받는 컴포넌트, 긴급도

2. 영향 분석(IMPACT ANALYSIS)
   기술팀이 평가:
   ├── 어떤 CI가 영향을 받는가?
   ├── 구현 노력/비용은 얼마인가?
   ├── 위험은 무엇인가?
   └── 하지 않을 경우의 위험은 무엇인가?

3. 검토 및 승인(REVIEW AND APPROVAL)
   변경 통제 위원회(CCB)가 검토:
   ├── 승인 → 일정 계획 및 담당자 지정
   ├── 연기 → 조건부로 백로그에 추가
   ├── 거부 → 사유 문서화
   └── 추가 정보 요청으로 반려

4. 구현(IMPLEMENTATION)
   개발자가 통제된 브랜치에서 변경 사항을 구현

5. 검증(VERIFICATION)
   테스트로 변경 사항이 작동하고 회귀가 없음을 확인

6. 릴리스(RELEASE)
   변경 사항이 배포되고 CR이 종료됨
   변경 로그가 업데이트됨
```

### 6.2 변경 통제 위원회(Change Control Board, CCB)

CCB(변경 자문 위원회(Change Advisory Board, CAB)라고도 함)는 변경에 대한 의사결정 기구입니다. 구성은 조직 규모에 따라 다릅니다:

| 조직 규모 | CCB 구성원 | 회의 주기 |
|-------------------|-------------|-----------------|
| 소규모 스타트업 | 기술 리드 + PM | 비동기 (Slack/GitHub) |
| 중간 규모 회사 | 엔지니어링 매니저, QA 리드, PM, 운영 | 주간 |
| 대기업 | CTO, 엔지니어링/QA/운영 VP, 보안 | 격주; P1 긴급 위원회 |
| 규제 산업 (금융, 의료) | 위 + 컴플라이언스 담당자, 법무 | 공식적, 문서화 |

애자일(Agile) 팀에서는 CCB 기능이 별도 기구 대신 스프린트 계획과 풀 리퀘스트 리뷰에 내재화되는 경우가 많습니다.

### 6.3 긴급 변경(Emergency Change)

중요한 운영 장애의 경우, 일반 CCB 주기는 너무 느립니다. **긴급 변경 프로세스**를 통해 다음이 허용됩니다:
1. 신속 승인 (전체 CCB 대신 승인자 2명)
2. 몇 시간 이내 구현 및 배포
3. **사후 문서화**: 변경 후(사전이 아닌) 문서 작성 완료
4. 48~72시간 이내 의무적 사후 검토

```
긴급 변경 트리거:
  운영 장애 | 보안 침해 | 데이터 손상 위험

신속 승인 (다음 중 2인): 기술 리드, 엔지니어링 매니저, 온콜 SRE

배포 전 롤백 계획 필수

72시간 이내 사후 조치:
  - 근본 원인 분석(RCA)
  - 전체 CCB 방식의 문서화
  - 교훈(Lessons learned)
```

### 6.4 영향 분석(Impact Analysis)

변경을 승인하기 전에, CCB는 그것이 무엇에 영향을 미치는지 이해해야 합니다:

```python
# 단순화된 의존성 그래프 영향 분석
def find_impacted_modules(changed_module: str, dependency_graph: dict) -> set:
    """
    BFS from changed module to find all modules that depend on it.
    dependency_graph[A] = [B, C] means A depends on B and C.
    We want to find all modules that (transitively) depend on changed_module.
    """
    # Reverse the graph: who depends on me?
    reverse_graph = {}
    for module, deps in dependency_graph.items():
        for dep in deps:
            reverse_graph.setdefault(dep, set()).add(module)

    impacted = set()
    queue = [changed_module]
    while queue:
        current = queue.pop(0)
        for dependent in reverse_graph.get(current, []):
            if dependent not in impacted:
                impacted.add(dependent)
                queue.append(dependent)
    return impacted

# Example
graph = {
    "checkout": ["cart", "payment", "inventory"],
    "order_history": ["checkout"],
    "admin_dashboard": ["checkout", "order_history"],
    "notification_service": ["checkout"],
}
print(find_impacted_modules("payment", graph))
# {'checkout', 'order_history', 'admin_dashboard', 'notification_service'}
```

---

## 7. 형상 감사(Configuration Auditing)

형상 감사는 빌드되고 배포된 것이 승인되고 문서화된 것과 일치하는지 검증합니다. 두 가지 유형이 있습니다:

### 7.1 기능 형상 감사(Functional Configuration Audit, FCA)

CI의 실제 성능이 요구사항과 일치하는지 검증합니다. "이 빌드가 해야 할 모든 것을 하는가?"라는 질문에 답합니다.

**체크리스트**:
- 계획된 모든 기능이 존재하고 작동함
- 이전 감사에서 발견된 모든 결함이 해결됨
- 테스트 결과가 품질 완료 기준을 충족함
- 릴리스 노트가 변경된 내용을 정확하게 설명함

### 7.2 물리 형상 감사(Physical Configuration Audit, PCA)

CI가 물리적으로 문서화된 설명과 일치하는지 검증합니다. "릴리스하려는 것이 정확히 승인되고 테스트된 것인가?"라는 질문에 답합니다.

**체크리스트**:
- 코드의 버전 번호가 빌드 산출물의 버전 번호와 일치함
- 빌드 매니페스트에 나열된 모든 파일이 존재함
- 산출물의 체크섬이 기록된 값과 일치함
- 승인되지 않은 파일이 포함되지 않음
- 산출물의 의존성이 승인된 의존성 목록과 일치함

```bash
# 체크섬으로 산출물 무결성 검증
sha256sum myapp-2.3.0-linux-amd64.tar.gz > myapp-2.3.0.sha256
cat myapp-2.3.0.sha256
# 7a3b9c1d... myapp-2.3.0-linux-amd64.tar.gz

# 검증 (수신자 또는 감사자가 수행)
sha256sum --check myapp-2.3.0.sha256
# myapp-2.3.0-linux-amd64.tar.gz: OK
```

### 7.3 소프트웨어 부품 명세서(Software Bill of Materials, SBOM)

SBOM은 소프트웨어 산출물의 모든 컴포넌트 — 자체 개발 코드, 오픈소스 의존성, 그리고 그 라이선스 — 에 대한 완전한 목록입니다. 다음을 위해 필요합니다:
- 공급망 보안 (행정명령 14028, 2021)
- 라이선스 컴플라이언스
- 취약점 대응 (CVE가 공개될 때 보유 항목 파악)

```bash
# syft를 사용하여 SPDX 형식으로 SBOM 생성
syft myapp:2.3.0 -o spdx-json > myapp-2.3.0.sbom.json

# Python 프로젝트용 SBOM 생성
pip install cyclonedx-bom
cyclonedx-py --poetry -o sbom.json
```

---

## 8. 환경 관리

소프트웨어는 각기 다른 목적을 위한 여러 환경에서 실행됩니다. 이러한 환경을 일관되게 관리하는 것은 신뢰할 수 있는 배포를 위해 필수적입니다.

### 8.1 표준 환경

```
개발자 워크스테이션
        │
        │ (커밋 + PR)
        ▼
CI 환경 (임시 — 빌드마다 생성, 완료 후 삭제)
        │
        │ (main으로 병합)
        ▼
개발/통합 환경(Development / Integration Environment)
        │
        │ (예정된 승격)
        ▼
스테이징/사전 운영 환경(Staging / Pre-production Environment)
        │
        │ (승인 게이트)
        ▼
운영 환경(Production Environment)
```

| 환경 | 목적 | 접근 주체 | 데이터 |
|-------------|---------|----------------|------|
| **CI** | 자동화된 빌드 및 테스트 | CI 시스템 | 합성 테스트 데이터 |
| **개발** | 통합 테스트; 데모 | 개발자, QA | 익명화 또는 합성 데이터 |
| **스테이징** | 릴리스 전 검증 | QA, PM, 이해관계자 | 운영 스냅샷 (익명화) |
| **운영** | 최종 사용자 서비스 | 최종 사용자 | 실제 데이터 |

**환경 드리프트(Environment drift)** — 스테이징이 운영과 다르게 동작하는 현상 — 은 "내 컴퓨터에서는 됐는데" 문제의 가장 흔한 원인 중 하나입니다. 코드형 인프라(IaC)가 이를 해결합니다.

### 8.2 코드형 인프라(Infrastructure as Code, IaC)

IaC는 인프라(서버, 데이터베이스, 네트워크, 로드 밸런서)를 버전 관리에 저장된 선언적 설정 파일로 정의합니다. 동일한 코드가 모든 환경을 프로비저닝하여 일관성을 보장합니다.

```hcl
# Terraform: 웹 애플리케이션 스택 프로비저닝
# environments/staging/main.tf와 environments/prod/main.tf는
# 동일한 모듈을 공유 — 입력값만 다름

module "web_app" {
  source = "../../modules/web_app"

  # 환경별 값 (terraform.tfvars에서 설정)
  environment     = var.environment     # "staging" 또는 "prod"
  instance_type   = var.instance_type   # "t3.small" vs "t3.large"
  min_instances   = var.min_instances   # 1 vs 3
  max_instances   = var.max_instances   # 2 vs 10
  db_instance     = var.db_instance     # "db.t3.micro" vs "db.r6g.large"
  enable_backups  = var.enable_backups  # false vs true
  alert_endpoints = var.alert_endpoints # 개발팀 vs 운영팀 + PagerDuty
}
```

```yaml
# Ansible: 애플리케이션 서버 일관성 있게 구성
---
- name: Configure web application servers
  hosts: "{{ target_env }}_web"  # staging_web 또는 prod_web
  become: true
  vars_files:
    - "vars/{{ target_env }}.yml"  # 환경별 변수

  roles:
    - common          # OS 기준: NTP, 로깅, 보안 패치
    - nginx           # 웹 서버, TLS 종료
    - app_deploy      # 아티팩트 저장소에서 애플리케이션 산출물 배포
    - monitoring      # Prometheus 노드 익스포터 설치

  tasks:
    - name: Ensure application is running
      systemd:
        name: myapp
        state: started
        enabled: true
```

### 8.3 설정 파일과 시크릿(Secrets)

애플리케이션 설정은 환경마다 다릅니다 (데이터베이스 URL, API 키, 기능 플래그). 세 가지 계층:

```
계층 1: 민감하지 않은 환경별 설정
  → VCS의 환경별 설정 파일에 저장
  → 예: application-staging.yaml, application-prod.yaml

계층 2: 민감한 설정 (자격 증명, API 키)
  → 절대 VCS에 저장하지 말 것
  → 시크릿 관리 시스템(Vault, AWS Secrets Manager, GCP Secret Manager)에 저장
  → 런타임에 환경 변수 또는 마운트된 시크릿으로 주입

계층 3: 기능 플래그(Feature Flag)
  → 기능 플래그 서비스(LaunchDarkly, Unleash, Flagsmith)에 저장
  → 재배포 없이 변경 가능
```

```bash
# .env.example (VCS에 커밋 — 필요한 변수를 보여주되 값은 없음)
DATABASE_URL=postgresql://user:password@host:5432/dbname
REDIS_URL=redis://localhost:6379
STRIPE_API_KEY=sk_...
JWT_SECRET=...
LOG_LEVEL=INFO
FEATURE_FLAG_NEW_UI=false

# .env (커밋하지 않음 — 각 환경이 실제 값을 채움)
# 배포 시 시크릿 매니저에서 가져옴:
#   aws secretsmanager get-secret-value --secret-id prod/myapp
```

---

## 9. 의존성 관리

현대 소프트웨어는 수백 또는 수천 개의 서드파티 라이브러리로 구축됩니다. 이러한 의존성을 안전하게 관리하는 것은 SCM의 중요한 책임입니다.

### 9.1 잠금 파일(Lockfile)

**잠금 파일(Lockfile)**은 특정 시점에 해결된 모든 의존성(전이적 의존성 포함)의 정확한 버전을 기록합니다. 모든 개발자와 모든 CI 빌드가 정확히 동일한 버전을 사용하도록 보장합니다.

```
요구사항 파일 (의도 기술)            잠금 파일 (실제 기록)
──────────────────────────────────      ─────────────────────────────
requests>=2.28.0                        requests==2.31.0
flask>=3.0.0                            flask==3.0.2
                                        werkzeug==3.0.1
                                        click==8.1.7
                                        jinja2==3.1.3
                                        markupsafe==2.1.5
                                        ...47개의 추가 전이적 의존성
```

| 생태계 | 요구사항 파일 | 잠금 파일 |
|-----------|------------------|----------|
| Python (pip) | `requirements.txt` | `requirements.lock` (pip-compile) |
| Python (Poetry) | `pyproject.toml` | `poetry.lock` |
| Node.js (npm) | `package.json` | `package-lock.json` |
| Node.js (Yarn) | `package.json` | `yarn.lock` |
| Ruby | `Gemfile` | `Gemfile.lock` |
| Go | `go.mod` | `go.sum` |
| Rust | `Cargo.toml` | `Cargo.lock` |

**규칙**: 애플리케이션의 잠금 파일은 항상 커밋하십시오. 라이브러리의 경우 잠금 파일 커밋은 선택적입니다 (하위 소비자가 자체적으로 해결).

### 9.2 버전 고정(Version Pinning)

**버전 고정(Pinning)**은 범위 대신 정확한 버전을 지정합니다. 다음의 경우에 중요합니다:
- 운영 배포 (예측 가능성)
- 보안 스캔 (CVE 데이터베이스에서 특정 버전을 명확하게 확인 가능)

```python
# 미고정 (운영 환경에 위험)
requests>=2.28.0
flask~=3.0

# 고정 (운영 환경에 안전)
requests==2.31.0
flask==3.0.2
```

버전 고정 전략: `requirements.txt`가 아닌 잠금 파일에서 고정합니다. 이렇게 하면 라이브러리 호환성을 위한 의도(범위)와 재현성을 위한 보장(정확한 버전) 모두를 얻을 수 있습니다.

### 9.3 의존성 취약점 스캔

버전이 고정되고 잠금 파일이 있더라도, 잠금 파일이 생성된 후 CVE가 공개되면 의존성이 취약해질 수 있습니다.

```bash
# Python: pip-audit이 OSV 및 PyPA 어드바이저리 데이터베이스 대상 스캔
pip install pip-audit
pip-audit -r requirements.lock

# 출력:
# Found 2 known vulnerabilities in 1 package
# Name      Version  ID                    Fix Versions
# ──────────────────────────────────────────────────────
# cryptography 38.0.1  GHSA-jfh8-c2jp-jvq8  39.0.1
# cryptography 38.0.1  GHSA-w7pp-m8wf-vj6r  39.0.1

# Node.js: npm audit
npm audit --audit-level=high

# Docker 이미지 스캔
docker scout cves myapp:2.3.0
```

취약점 스캔을 CI 파이프라인에 통합하십시오 — 운영 의존성의 고심각도 취약점 발견 시 빌드를 실패시킵니다.

### 9.4 의존성 업데이트 전략

| 전략 | 설명 | 위험도 |
|----------|-------------|------|
| **수동 업데이트** | 개발자가 업데이트 시기 결정 | 자주 구식이 됨; 누적 위험 높음 |
| **자동 PR** (Dependabot, Renovate) | 봇이 각 업데이트에 PR 오픈 | PR이 많음; 좋은 테스트가 있으면 관리 가능 |
| **정기적 업데이트** | 스프린트마다 의존성 업데이트에 시간 할당 | 균형 잡힌; 예측 가능한 작업량 |
| **LTS 유지** | 장기 지원(Long-Term Support) 버전만 사용 | 보수적; 기능을 놓치지만 안정적 |

권장 사항: Dependabot 또는 Renovate를 사용하여 패치 업데이트는 자동 병합(테스트 통과 시), 마이너 및 메이저 업데이트는 수동 검토합니다.

---

## 10. 도구 개요

### 10.1 SCM 기능별 도구

| 기능 | 도구 | 사용 사례 |
|----------|------|----------|
| **버전 관리** | Git | 범용 소스 관리 |
| **코드 호스팅/리뷰** | GitHub, GitLab, Bitbucket | PR 워크플로우, 코드 리뷰, CODEOWNERS |
| **CI/CD** | GitHub Actions, Jenkins, GitLab CI, CircleCI | 자동화된 빌드, 테스트, 배포 파이프라인 |
| **아티팩트 저장** | JFrog Artifactory, Nexus, AWS ECR, GitHub Packages | 버전 관리된 바이너리 저장 |
| **IaC 프로비저닝** | Terraform, Pulumi | 코드형 클라우드 인프라 |
| **설정 관리** | Ansible, Chef, Puppet | 코드형 서버 설정 |
| **컨테이너 오케스트레이션** | Kubernetes, Docker Compose | 환경 일관성 |
| **시크릿 관리** | HashiCorp Vault, AWS Secrets Manager | 안전한 자격 증명 저장 |
| **의존성 관리** | pip-tools, Poetry, Dependabot, Renovate | 잠금 파일, 자동 업데이트 |
| **취약점 스캔** | Snyk, Dependabot, Trivy, pip-audit | 의존성 및 이미지의 CVE 탐지 |
| **SBOM 생성** | syft, CycloneDX | 소프트웨어 부품 명세서 |
| **기능 플래그** | LaunchDarkly, Unleash | 배포와 릴리스 분리 |

### 10.2 완전한 SCM 툴체인 예시

```
개발자가 코드 커밋 (Git)
           │
           ▼
풀 리퀘스트 오픈 (GitHub)
  └── 코드 리뷰 (CODEOWNERS가 필수 리뷰어 지정)
  └── 상태 체크 통과 필수:
        ├── 린팅 (GitHub Actions)
        ├── 단위 테스트 + 커버리지 (GitHub Actions)
        ├── 보안 스캔: Semgrep (GitHub Actions)
        └── 의존성 취약점 확인: Dependabot 알림
           │
           ▼ (PR이 main에 병합)
           │
CI 파이프라인 (GitHub Actions)
  ├── Docker 이미지 빌드
  ├── 통합 테스트 실행
  ├── ECR에 이미지 푸시 (git SHA로 태깅)
  └── SBOM 생성 (syft)
           │
           ▼
CD 파이프라인: 스테이징에 배포
  ├── Terraform apply (IaC 변경 시)
  ├── Ansible 플레이북 (서버 설정)
  └── Kubernetes 롤링 업데이트
           │
           ▼
수동 승인 게이트
           │
           ▼
CD 파이프라인: 운영에 배포 (블루/그린)
  ├── 기존 버전 옆에 새 버전 배포
  ├── 트래픽 전환: 10% → 50% → 100% (30분에 걸쳐)
  └── 오류율이 임계값 초과 시 자동 롤백
           │
           ▼
릴리스 태깅: v2.3.0 (Git 태그)
릴리스 노트 게시 (GitHub Release)
아티팩트 저장소에 SBOM 보관
```

---

## 11. 요약

소프트웨어 형상 관리는 대규모 소프트웨어 개발을 다룰 수 있게 만드는 기반입니다. 이것 없이는 시스템이 신뢰할 수 없게 되고, 릴리스가 예측 불가능해지며, 팀은 무언가를 안전하게 변경할 수 있다는 확신을 잃습니다.

핵심 요점:

- **형상 항목(Configuration Item)**은 추적이 필요한 모든 산출물 — 소스 코드, 문서, 설정 파일, 인프라 코드, 빌드 산출물 — 입니다. 모든 CI는 고유한 식별자와 버전을 가집니다.
- **기준선(Baseline)**은 참조점 역할을 하는 공식적으로 승인된 스냅샷입니다. 현대적 실천에서 메인 브랜치의 Git 태그에 해당합니다.
- **브랜치 전략** — 정기적 릴리스를 위한 Gitflow, 지속적 배포를 위한 트렁크 기반 개발 — 은 동시 변경 관리를 위한 워크플로우 구조를 제공합니다.
- **빌드 관리**는 버전 관리된 빌드 스크립트를 통한 자동화되고 재현 가능한 빌드를 의미합니다. 한 번 빌드하고, 산출물을 저장하고, 산출물을 배포합니다.
- **시맨틱 버저닝(Semantic Versioning)**은 버전 번호에 정확한 의미를 부여합니다. MAJOR.MINOR.PATCH는 소비자에게 호환성 보장을 알립니다.
- **변경 관리**는 수정에 규율을 가져옵니다: 모든 변경은 요청되고, 영향이 분석되고, 승인되고, 구현되고, 검증되고, 종료됩니다.
- **IaC를 사용한 환경 관리**는 개발, 스테이징, 운영 환경이 일관되고 재현 가능하도록 보장합니다 — "스테이징에서는 됐는데 운영에서는 안 된다"는 문제를 제거합니다.
- **의존성 관리**는 재현성을 위한 잠금 파일, 예측 가능성을 위한 버전 고정, 그리고 보안 문제를 신속하게 발견하기 위한 자동화된 취약점 스캔이 필요합니다.

---

## 12. 연습 문제

**연습 문제 1 — 형상 항목 식별**

다음과 같은 새로운 마이크로서비스에 대해 SCM을 설정하고 있습니다:
- Python FastAPI 애플리케이션
- PostgreSQL과 Redis 사용
- Terraform을 사용하여 AWS에 배포
- React 프론트엔드 동반
- OpenAPI 형식의 API 문서

(a) 이 시스템에 대해 유형별로 분류하여 최소 15개의 형상 항목을 나열하십시오.
(b) CI가 아닌(버전 관리되어서는 안 되는) 세 가지 산출물을 식별하고 이유를 설명하십시오.
(c) SCM 식별 요구사항을 충족하는 Docker 이미지 태그 명명 체계를 설계하십시오.

**연습 문제 2 — 브랜치 전략**

8명의 개발자로 구성된 팀이 이커머스 플랫폼을 개발하고 있습니다. 현재는 격주로 릴리스하지만, 6개월 이내에 일일 릴리스로 전환하고자 합니다.

(a) 현재 2주 릴리스 주기에 적합한 브랜치 전략을 권장하십시오. 주요 브랜치를 보여주는 간단한 다이어그램을 그리십시오.
(b) 트렁크 기반 개발로의 전환 계획을 설명하십시오. 전환 전에 팀이 갖추어야 할 것은 무엇입니까?
(c) 다음 릴리스 작업이 진행 중인 동안 2주 전 릴리스에 대한 핫픽스가 필요합니다. 단계별로 정확한 Git 워크플로우를 설명하십시오.

**연습 문제 3 — 의존성 분석**

Python 웹 애플리케이션의 `pyproject.toml`이 다음을 지정합니다:

```toml
[tool.poetry.dependencies]
python = "^3.11"
django = ">=4.2,<5.0"
celery = "^5.3"
redis = ">=4.6"
boto3 = "*"
```

(a) boto3에 `*` (임의 버전)를 사용하는 것의 위험은 무엇입니까?
(b) `django = "4.2.8"` 대신 `django = ">=4.2,<5.0"`를 사용합니다. 이 접근 방식의 장점과 단점은 무엇입니까?
(c) 다음을 수행하는 Dependabot 설정(`dependabot.yml`)을 작성하십시오: 주간으로 업데이트 확인, 모든 패치 업데이트를 그룹화, 메이저 버전 업데이트에 대해 `@security-team`의 검토 요구.

**연습 문제 4 — 변경 통제**

운영 시스템이 금융 거래를 처리합니다. 한 개발자가 레거시 MD5 기반 세션 토큰을 암호학적으로 안전한 랜덤 토큰(256비트, base64 인코딩)으로 교체하는 변경을 제안합니다. 예상 구현 시간: 4시간.

(a) 영향 분석을 포함하여 이 변경에 대한 완전한 변경 요청 문서를 작성하십시오.
(b) 변경 통제 위원회가 이 변경을 승인하기 전에 물어야 할 질문은 무엇입니까?
(c) 이 변경이 운영 환경에 배포되기 전에 통과해야 하는 검증 단계를 설계하십시오.

**연습 문제 5 — 릴리스 관리**

SaaS 제품이 버전 `2.7.3`에 있습니다. 다음 각 변경에 대해 적절한 다음 버전 번호를 결정하고 이유를 설명하십시오:

(a) 회의 알림의 시간대 처리를 수정하는 버그 수정.
(b) 사용자가 프로필에서 토글할 수 있는 새로운 "다크 모드" 설정.
(c) REST API의 `/users` 엔드포인트가 이제 JSON 응답에서 `id` 대신 `user_id`를 반환합니다.
(d) API 변경 없이 UI에 일본어와 한국어 지원 추가.
(e) 12개월 전에 제거 예고된 deprecated SOAP API 엔드포인트 제거.

---

## 13. 더 읽을거리

- **도서**:
  - *Continuous Delivery* — Jez Humble and David Farley. 빌드, 테스트, 배포 파이프라인을 포함한 자동화된 소프트웨어 배포의 결정판 가이드.
  - *The Phoenix Project* — Gene Kim, Kevin Behr, George Spafford. 설득력 있는 스토리를 통해 DevOps와 변경 관리 개념을 보여주는 소설.
  - *Software Configuration Management Patterns* — Steve Berczuk and Brad Appleton. 애자일 팀을 위한 SCM 실용 패턴.
  - *Infrastructure as Code* (2nd ed.) — Kief Morris. 코드로 인프라를 관리하기 위한 종합 가이드.

- **표준**:
  - IEEE Std 828-2012 — 시스템 및 소프트웨어 공학의 형상 관리를 위한 IEEE 표준
  - NIST SP 800-128 — 정보 시스템의 보안 중심 형상 관리 가이드
  - Semantic Versioning 2.0.0 — https://semver.org/

- **도구 문서**:
  - Git Reference — https://git-scm.com/doc
  - Terraform Documentation — https://developer.hashicorp.com/terraform/docs
  - Ansible Documentation — https://docs.ansible.com/
  - Dependabot Documentation — https://docs.github.com/en/code-security/dependabot
  - Syft (SBOM generator) — https://github.com/anchore/syft

- **아티클 및 명세서**:
  - "Gitflow Workflow" — Atlassian Bitbucket 가이드
  - "Trunk Based Development" — https://trunkbaseddevelopment.com/
  - Reproducible Builds — https://reproducible-builds.org/
  - NTIA Software Bill of Materials — https://www.ntia.gov/sbom

---

## 연습 문제

### 연습 1: 형상 항목(Configuration Item) 식별

PostgreSQL을 사용하는 새로운 Python FastAPI 서비스를 위해 SCM을 설정하고 있습니다. 이 서비스는 Docker로 컨테이너화되고, Terraform을 통해 AWS에 배포되며, OpenAPI 형식의 API 문서가 있습니다.

(a) 범주별로 (소스 코드, 빌드 산출물, 문서, 설정 파일, 인프라 코드) 최소 12개의 형상 항목(CI)을 나열하세요.
(b) 각 CI에 대해 버전 관리 전략(Git 저장소, 아티팩트 레지스트리, 시크릿 매니저)을 명시하고 선택 이유를 설명하세요.
(c) 버전 관리하면 안 되는 산출물 두 가지를 식별하고 이유를 설명하세요. 각각을 대신 어떻게 관리해야 하는지 설명하세요.

### 연습 2: 브랜칭 전략(Branching Strategy) 설계

여섯 명의 개발자로 구성된 팀이 SaaS HR 플랫폼을 구축하고 있습니다. 현재는 월별로 릴리스하지만, 3개월 내에 주간 릴리스로 전환하려고 합니다.

(a) 현재 월별 릴리스 주기에 적합한 브랜칭 전략을 설계하세요. 각 장기 유지 브랜치와 그 목적을 설명하세요.
(b) 프로덕션 보안 취약점이 발견되었습니다. 진행 중인 다음 릴리스를 방해하지 않고 핫픽스(Hotfix)를 생성하고 배포하기 위한 정확한 Git 명령어(브랜치 생성, 수정, 태그, 병합)를 단계별로 설명하세요.
(c) 트렁크 기반 개발(Trunk-Based Development)로 안전하게 전환하기 전에 팀이 갖춰야 할 세 가지 기술적 전제 조건을 나열하세요.

### 연습 3: 의미적 버전 관리(Semantic Versioning) 결정

라이브러리가 현재 버전 `3.2.1`에 있습니다. 각 변경에 대해 올바른 다음 버전 번호를 결정하고, 어떤 SemVer 규칙이 적용되는지 설명하세요.

(a) 날짜 범위 계산에서 off-by-one 오류를 수정하는 버그 픽스.
(b) 새로운 `batch_process()` 메서드가 공개 API에 추가되고, 기존 메서드는 변경 없음.
(c) `process()` 메서드의 반환 유형이 `dict`에서 타입이 지정된 `Result` 데이터클래스로 변경 — dict를 언팩하는 기존 호출자가 깨짐.
(d) `3.0.0` 이후 더 이상 사용되지 않는(Deprecated) `legacy_process()` 메서드가 제거됨.
(e) 내부 구현 세부 사항이 더 빠른 알고리즘을 사용하도록 변경됨; 공개 API는 동일함.

### 연습 4: 의존성 위험(Dependency Risk) 분석

Python 웹 애플리케이션의 `pyproject.toml`이 다음을 선언합니다:

```toml
[tool.poetry.dependencies]
python = "^3.11"
fastapi = ">=0.100,<1.0"
sqlalchemy = "^2.0"
pydantic = "*"
httpx = "^0.25"
```

(a) 다섯 가지 의존성을 버전 해결 위험(Version-Resolution Risk) 높음에서 낮음 순으로 순위를 매기세요. 각 순위에 대한 근거를 제시하세요.
(b) `pydantic` 지정자 `"*"`가 오늘 버전 `1.10.14`로 해결됩니다. 6개월 후, breaking API가 있는 `pydantic 2.0`이 출시됩니다. 이로 인한 위험은 무엇이며, 잠금 파일(Lockfile)은 이를 어떻게 완화합니까?
(c) Poetry, Git, 자동화된 테스트가 있는 CI 파이프라인을 사용하는 팀에서 `sqlalchemy`를 `2.0.x`에서 `2.1.x`로 안전하게 업그레이드하는 데 필요한 세 가지 단계를 작성하세요.

### 연습 5: 변경 요청(Change Request) 작성

현재 프로덕션 API는 HTTP 기본 인증(Basic Authentication)을 사용합니다. 보안 팀이 OAuth 2.0 Bearer 토큰으로의 마이그레이션을 의무화했습니다. 이 변경은 인증 미들웨어, 모든 API 클라이언트(내부 및 외부), 개발자 문서, 세 가지 서드파티 통합에 영향을 미칩니다.

다음 내용을 포함하는 완전한 변경 요청(Change Request) 문서를 작성하세요:
- 변경 설명 및 비즈니스 정당성
- 영향을 받는 형상 항목(최소 여섯 가지 나열)
- 영향 분석(일정, 노력, 위험)
- 롤백 계획(Rollback Plan)
- 검증 및 인수 기준(Verification and Acceptance Criteria)

---

**이전**: [08. 검증과 확인](./08_Verification_and_Validation.md) | **다음**: [10. 프로젝트 관리](./10_Project_Management.md)
