# 레슨 2: 버전 관리 워크플로우

**이전**: [DevOps 기초](./01_DevOps_Fundamentals.md) | **다음**: [CI 기초](./03_CI_Fundamentals.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. GitFlow, GitHub Flow, 트렁크 기반 개발을 비교 및 대조하고, 주어진 팀과 프로젝트에 적합한 워크플로우를 선택할 수 있다
2. Feature flag를 구현하여 배포와 기능 릴리스를 분리할 수 있다
3. 모노레포 vs 폴리레포 전략의 장단점을 평가할 수 있다
4. CI/CD 파이프라인을 지원하고 머지 충돌을 최소화하는 브랜칭 전략을 설계할 수 있다
5. 속도와 품질의 균형을 맞추는 브랜치 보호 규칙과 코드 리뷰 관행을 적용할 수 있다

---

버전 관리 워크플로우는 팀이 코드에 대해 어떻게 협업하고, 릴리스를 관리하며, 변경 사항을 통합하는지를 정의합니다. 올바른 브랜칭 전략은 배포 빈도, 리드 타임, 변경 실패율 -- 네 가지 DORA 메트릭 중 세 가지에 직접적인 영향을 미칩니다. 잘못된 워크플로우를 선택하면 머지 지옥이 발생하고, 릴리스가 느려지며, 리스크가 증가합니다. 이 레슨에서는 가장 일반적인 워크플로우, 장단점, 그리고 팀의 필요에 맞게 매칭하는 방법을 살펴봅니다.

> **비유 -- 고속도로 설계:** 브랜칭 전략은 고속도로 설계와 같습니다. GitFlow는 많은 진입/출구 램프가 있는 복잡한 인터체인지입니다(강력하지만 혼란스러움). 트렁크 기반 개발은 모든 사람이 빠르게 합류하는 단일 고속 차선입니다. GitHub Flow는 간단한 진입/출구 램프가 있는 2차선 고속도로입니다. 최적의 설계는 교통량(팀 규모)과 속도 요구사항(배포 빈도)에 따라 달라집니다.

## 1. GitFlow

GitFlow는 Vincent Driessen이 2010년에 소개한 것으로, 엄격한 머지 규칙을 가진 여러 장기 브랜치를 사용합니다.

### 브랜치 구조

```
main (production)
  │
  ├── hotfix/payment-bug ──────────────────────────▶ main + develop
  │
  develop (integration)
  │
  ├── release/v2.1 ───────────────────────────────▶ main + develop
  │
  ├── feature/user-auth ──────────────────────────▶ develop
  │
  └── feature/dashboard ──────────────────────────▶ develop
```

### 브랜치 유형

| 브랜치 | 용도 | 수명 | 머지 대상 |
|--------|------|------|-----------|
| `main` | 프로덕션 준비 완료된 코드 | 영구 | -- |
| `develop` | 기능 통합 브랜치 | 영구 | -- |
| `feature/*` | 새 기능 개발 | 임시 | `develop` |
| `release/*` | 릴리스 준비 및 안정화 | 임시 | `main` + `develop` |
| `hotfix/*` | 긴급 프로덕션 수정 | 임시 | `main` + `develop` |

### GitFlow 실습

```bash
# Start a new feature
git checkout develop
git checkout -b feature/user-auth

# Work on the feature...
git add .
git commit -m "Add user authentication module"

# Finish the feature -- merge back to develop
git checkout develop
git merge --no-ff feature/user-auth
git branch -d feature/user-auth

# Prepare a release
git checkout develop
git checkout -b release/v2.1

# Stabilize the release (bug fixes only)
git commit -m "Fix login redirect bug"

# Finish the release
git checkout main
git merge --no-ff release/v2.1
git tag -a v2.1.0 -m "Release v2.1.0"

git checkout develop
git merge --no-ff release/v2.1
git branch -d release/v2.1

# Emergency hotfix
git checkout main
git checkout -b hotfix/payment-bug
git commit -m "Fix payment processing null pointer"

git checkout main
git merge --no-ff hotfix/payment-bug
git tag -a v2.1.1 -m "Hotfix v2.1.1"

git checkout develop
git merge --no-ff hotfix/payment-bug
git branch -d hotfix/payment-bug
```

### GitFlow 장단점

| 장점 | 단점 |
|------|------|
| 관심사의 명확한 분리 | 복잡함 -- 관리할 브랜치가 많음 |
| 병렬 릴리스 작업 지원 | 장기 브랜치가 머지 충돌을 유발 |
| 버전별 소프트웨어(모바일 앱, 데스크톱)에 적합 | 느린 피드백 루프 |
| 명시적인 릴리스 프로세스 | 지속적 배포와 호환되지 않음 |
| 예정된 릴리스가 있는 팀에 적합 | `develop` 브랜치가 병목이 될 수 있음 |

---

## 2. GitHub Flow

GitHub Flow는 하나의 장기 브랜치(`main`)와 단기 기능 브랜치를 사용하는 단순화된 워크플로우입니다.

### 브랜치 구조

```
main (always deployable)
  │
  ├── feature/user-auth ──── PR ──── review ──── merge ──── deploy
  │
  ├── fix/login-bug ──────── PR ──── review ──── merge ──── deploy
  │
  └── feature/dashboard ──── PR ──── review ──── merge ──── deploy
```

### 워크플로우 단계

```bash
# 1. Create a branch from main
git checkout main
git pull origin main
git checkout -b feature/user-profile

# 2. Make changes and commit
git add .
git commit -m "Add user profile page"

# 3. Push and open a pull request
git push origin feature/user-profile
gh pr create --title "Add user profile page" \
  --body "Implements the user profile page with avatar upload"

# 4. Discuss and review
# Team members review the code, CI runs tests

# 5. Merge to main
gh pr merge --squash

# 6. Deploy
# Automatic deployment triggered on main merge
```

### GitHub Flow 규칙

1. **`main`은 항상 배포 가능** -- main에 깨진 코드를 절대 커밋하지 않음
2. **main에서 브랜치 생성** -- 모든 작업은 최신 main 체크아웃에서 시작
3. **PR을 일찍 개설** -- 작업 중인 내용의 논의를 위해 draft PR 사용
4. **머지 후 배포** -- main에 머지할 때마다 배포를 트리거
5. **머지 후 브랜치 삭제** -- 브랜치 목록을 깔끔하게 유지

### GitHub Flow 장단점

| 장점 | 단점 |
|------|------|
| 단순함 -- 두 가지 브랜치 유형만 존재 | 명시적인 릴리스 프로세스 없음 |
| 빠른 피드백 루프 | 모든 머지가 배포됨 (견고한 CI/CD 필요) |
| 작고 빈번한 변경을 장려 | 여러 버전 관리가 어려움 |
| 지속적 배포와 자연스럽게 맞음 | 릴리스 준비를 위한 스테이징 브랜치 없음 |
| 새 팀원이 배우기 쉬움 | 불완전한 기능에 Feature flag 필요 |

---

## 3. 트렁크 기반 개발

트렁크 기반 개발(TBD)은 단순함을 더욱 추구합니다. 개발자가 트렁크(main)에 직접 커밋하거나 매우 짧은 수명의 브랜치(1일 미만)를 사용합니다.

### 브랜치 구조

```
main (trunk)
  │
  ├── [direct commit by dev A]
  │
  ├── short-lived-branch (< 1 day) ──── merge ──── [auto-deploy]
  │
  ├── [direct commit by dev B]
  │
  └── short-lived-branch (< 4 hours) ── merge ──── [auto-deploy]
```

### 핵심 실천 사항

```bash
# Option A: Commit directly to main (small teams)
git checkout main
git pull --rebase origin main
# Make a small change
git add .
git commit -m "Add input validation to login form"
git push origin main

# Option B: Short-lived branch (larger teams)
git checkout main
git pull --rebase origin main
git checkout -b add-validation

# Work for a few hours at most
git add .
git commit -m "Add input validation to login form"
git push origin add-validation

# Create PR, get quick review, merge same day
gh pr create --title "Add login validation"
# After approval (ideally within hours):
gh pr merge --squash
```

### 트렁크 기반 개발 규칙

1. **브랜치는 하루 미만 존재** (이상적으로는 수 시간)
2. **하루에 최소 한 번 트렁크에 머지** -- "아프면 더 자주 하라"
3. **Feature flag로 미완성 작업 제어** -- 코드는 배포되지만 활성화되지 않음
4. **포괄적인 자동화 테스팅** -- 브랜치가 아니라 테스트가 안전망
5. **장기 브랜치 없음** -- develop 없음, release 브랜치 없음

### 워크플로우 비교

```
┌─────────────────────────────────────────────────────────────────┐
│           Workflow Comparison Matrix                             │
│                                                                  │
│  Dimension          GitFlow    GitHub Flow   Trunk-Based         │
│  ──────────────     ────────   ──────────   ───────────          │
│  Complexity         High       Low          Very Low             │
│  Deploy Frequency   Low        High         Very High            │
│  Branch Lifetime    Days-Weeks Hours-Days   Hours                │
│  Merge Conflicts    High       Medium       Low                  │
│  CI/CD Fit          Poor       Good         Excellent            │
│  Team Size          Any        Small-Medium Small-Large          │
│  Release Model      Scheduled  Continuous   Continuous           │
│  Feature Flags      Optional   Recommended  Required             │
│  Best For           Versioned  Web apps     High-performing      │
│                     software   & services   teams                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Feature Flag

Feature flag(기능 토글이라고도 함)는 배포와 릴리스를 분리하여, 불완전하거나 위험한 기능을 배포하되 플래그 뒤에 숨길 수 있게 합니다.

### Feature Flag 유형

| 유형 | 용도 | 수명 | 예시 |
|------|------|------|------|
| **릴리스 플래그** | 미완성 기능 숨김 | 수일 ~ 수 주 | `new_dashboard_enabled` |
| **실험 플래그** | A/B 테스팅 | 수일 ~ 수 주 | `checkout_v2_experiment` |
| **운영 플래그** | 운영적 제어 | 영구 | `circuit_breaker_payments` |
| **권한 플래그** | 사용자별 접근 | 영구 | `beta_features_enabled` |

### 구현 예시

```python
# Simple feature flag implementation
import os
import json

class FeatureFlags:
    """Simple file-based feature flag system."""

    def __init__(self, config_path="features.json"):
        with open(config_path) as f:
            self._flags = json.load(f)

    def is_enabled(self, flag_name: str, default: bool = False) -> bool:
        return self._flags.get(flag_name, default)

# features.json
# {
#     "new_dashboard": false,
#     "dark_mode": true,
#     "experimental_search": false
# }

flags = FeatureFlags()

# In application code
def get_dashboard():
    if flags.is_enabled("new_dashboard"):
        return render_new_dashboard()
    return render_old_dashboard()
```

```python
# Feature flag with environment variable override
import os

def is_feature_enabled(flag_name: str) -> bool:
    """Check feature flag, with env var override for testing."""
    env_key = f"FEATURE_{flag_name.upper()}"
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return env_val.lower() in ("true", "1", "yes")
    # Fall back to config file or remote service
    return get_flag_from_config(flag_name)
```

### Feature Flag 모범 사례

```
DO:
  ✓ Give flags descriptive names: "new_checkout_flow" not "flag_42"
  ✓ Set an expiration date for temporary flags
  ✓ Track all active flags in a registry
  ✓ Remove flags once the feature is fully rolled out
  ✓ Test both paths (flag on and flag off)

DON'T:
  ✗ Leave stale flags in the codebase for months
  ✗ Nest feature flags (flag A controls flag B)
  ✗ Use flags as a substitute for proper configuration
  ✗ Skip testing the "flag off" path
```

---

## 5. 모노레포 vs 폴리레포

### 모노레포

모든 프로젝트, 서비스, 라이브러리가 단일 저장소에 존재합니다.

```
monorepo/
├── services/
│   ├── api/
│   ├── web/
│   └── worker/
├── libs/
│   ├── auth/
│   ├── database/
│   └── logging/
├── tools/
│   ├── linter/
│   └── deploy/
└── package.json / BUILD files
```

**모노레포 도구:**
- **Bazel** -- Google의 빌드 시스템, 언어 무관
- **Nx** -- JavaScript/TypeScript용 모노레포 툴킷
- **Turborepo** -- JS/TS 모노레포용 고성능 빌드 시스템
- **Lerna** -- npm 멀티 패키지 관리

### 폴리레포

각 프로젝트, 서비스, 라이브러리가 자체 저장소를 가집니다.

```
org/api-service/          (own repo, own CI/CD)
org/web-frontend/         (own repo, own CI/CD)
org/worker-service/       (own repo, own CI/CD)
org/auth-library/         (own repo, published as package)
org/database-library/     (own repo, published as package)
```

### 비교

| 측면 | 모노레포 | 폴리레포 |
|------|----------|----------|
| **코드 공유** | 직접 import, 즉시 사용 | 패키지 게시, 버전 관리 |
| **원자적 변경** | 하나의 PR로 여러 서비스 변경 | 여러 저장소에 걸친 복수 PR |
| **CI/CD** | 영향받는 부분만 빌드 | 각 저장소가 자체 파이프라인 보유 |
| **코드 소유권** | 디렉토리별 CODEOWNERS | 저장소별 권한 |
| **의존성 관리** | 단일 lockfile, 일관된 버전 | 각 저장소가 자체 의존성 관리 |
| **온보딩** | 한 번 클론, 전체 확인 | 여러 저장소 클론 |
| **규모 문제** | 대규모 체크아웃, 느린 git 작업 | 의존성 버전 드리프트 |
| **사용 기업** | Google, Meta, Microsoft | Netflix, Amazon, Spotify |

### 선택 기준

```
Choose Monorepo when:
  - Teams share many libraries and interfaces
  - Atomic cross-service changes are frequent
  - You have tooling to support it (Bazel, Nx)
  - Code consistency is a priority

Choose Polyrepo when:
  - Teams are highly autonomous
  - Services are loosely coupled
  - Teams use different languages/frameworks
  - You want independent deployment cycles
  - Repository access control is important
```

---

## 6. 브랜치 보호 및 코드 리뷰

### 브랜치 보호 규칙

```yaml
# GitHub branch protection (configured via UI or API)
# Settings > Branches > Branch protection rules

main:
  required_reviews: 2                    # At least 2 approvals
  dismiss_stale_reviews: true            # Re-review after new pushes
  require_code_owner_review: true        # CODEOWNERS must approve
  require_status_checks:
    - ci/build
    - ci/test
    - ci/lint
  require_branches_up_to_date: true      # Branch must be current with main
  restrict_pushes: true                  # No direct pushes to main
  require_signed_commits: false          # Optional GPG signing
  require_linear_history: true           # Squash or rebase only (no merge commits)
```

```bash
# Set up branch protection via GitHub CLI
gh api repos/{owner}/{repo}/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["ci/build","ci/test"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":2}'
```

### CODEOWNERS 파일

```
# .github/CODEOWNERS
# Each line defines who must review changes to matching files

# Default reviewers for everything
*                       @org/platform-team

# Frontend team owns all frontend code
/src/frontend/          @org/frontend-team

# Backend team owns API code
/src/api/               @org/backend-team

# Security team must review auth changes
/src/auth/              @org/security-team
/src/api/middleware/     @org/security-team

# Infrastructure team owns IaC
/terraform/             @org/infra-team
/ansible/               @org/infra-team
/.github/workflows/     @org/platform-team
```

### 효과적인 코드 리뷰

```
Code Review Checklist:
──────────────────────
Functionality:
  [ ] Does the code do what the PR description says?
  [ ] Are edge cases handled?
  [ ] Are error paths tested?

Design:
  [ ] Is the code in the right place architecturally?
  [ ] Are abstractions appropriate (not over/under-engineered)?
  [ ] Does it follow existing patterns in the codebase?

Readability:
  [ ] Are variable/function names clear?
  [ ] Are complex sections commented?
  [ ] Would a new team member understand this code?

Testing:
  [ ] Are there tests for new functionality?
  [ ] Do tests cover edge cases and error paths?
  [ ] Are tests readable and maintainable?

Security:
  [ ] No hardcoded secrets or credentials?
  [ ] Input validation present?
  [ ] SQL injection / XSS prevention?
```

---

## 7. 커밋 컨벤션

### Conventional Commits

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**유형:**

| 유형 | 설명 |
|------|------|
| `feat` | 새로운 기능 |
| `fix` | 버그 수정 |
| `docs` | 문서만 변경 |
| `style` | 포맷팅, 누락된 세미콜론 (코드 변경 없음) |
| `refactor` | 버그 수정도 기능 추가도 아닌 코드 변경 |
| `perf` | 성능 개선 |
| `test` | 테스트 추가 또는 수정 |
| `ci` | CI 구성 변경 |
| `chore` | 유지보수 작업, 의존성 업데이트 |

**예시:**

```bash
# Feature with scope
git commit -m "feat(auth): add OAuth2 login with Google"

# Bug fix
git commit -m "fix(api): handle null response from payment gateway"

# Breaking change (noted in footer)
git commit -m "feat(api): change user endpoint response format

BREAKING CHANGE: GET /users now returns paginated results.
The response shape changed from an array to an object with
'data' and 'pagination' fields."

# CI change
git commit -m "ci: add Python 3.12 to test matrix"
```

---

## 연습 문제

### 연습 문제 1: 워크플로우 선택

세 가지 다른 팀의 컨설턴트로서 각 팀에 적합한 브랜칭 전략을 추천하고 근거를 제시하십시오:

1. **팀 A**: 4명의 개발자가 App Store에 분기별 릴리스하는 모바일 앱을 개발 중입니다. 프로덕션에서 여러 버전을 동시에 유지해야 합니다.
2. **팀 B**: 15명의 개발자가 지속적 배포를 사용하는 SaaS 웹 애플리케이션을 개발 중입니다. 하루에 10회 이상 배포합니다.
3. **팀 C**: 3명의 개발자가 오픈소스 라이브러리를 개발 중입니다. 외부 기여자가 PR을 자주 제출합니다. npm에 버전별 릴리스를 게시합니다.

각 팀에 대해 어떤 워크플로우를, 왜 선택했는지, 그리고 어떤 구체적인 실천 사항(Feature flag, 브랜치 보호 규칙)을 채택해야 하는지 설명하십시오.

### 연습 문제 2: Feature Flag 구현

새로운 검색 알고리즘을 점진적으로 출시하기 위한 Feature flag 시스템을 설계하십시오:
1. 플래그와 가능한 상태(boolean, 백분율 롤아웃, 사용자 세그먼트)를 정의하십시오
2. 플래그를 확인하는 애플리케이션 로직의 의사코드를 작성하십시오
3. CI에서 이전 코드 경로와 새 코드 경로를 모두 테스트하는 방법을 설명하십시오
4. 롤아웃 계획: 각 단계별 백분율, 모니터링할 메트릭, 롤백 기준을 수립하십시오

### 연습 문제 3: 모노레포 마이그레이션 계획

조직에 5개의 마이크로서비스가 별도의 저장소에 있고, 3개의 공통 라이브러리를 공유합니다. 공유 라이브러리에 버전 드리프트 문제가 있습니다(서비스 A는 auth-lib v1.2 사용, 서비스 B는 v1.5 사용). 마이그레이션 계획을 수립하십시오:
1. 이 특정 시나리오에서 모노레포로 마이그레이션하는 것의 장단점을 나열하십시오
2. 단계적 마이그레이션 계획을 제안하십시오 (어떤 저장소를 먼저, 어떤 도구 사용)
3. CI/CD 파이프라인이 어떻게 변경되는지 설명하십시오
4. 리스크와 완화 전략을 식별하십시오

---

**이전**: [DevOps 기초](./01_DevOps_Fundamentals.md) | [개요](00_Overview.md) | **다음**: [CI 기초](./03_CI_Fundamentals.md)

**License**: CC BY-NC 4.0
