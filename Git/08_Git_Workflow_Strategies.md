# 08. Git 워크플로우 전략

## 학습 목표
- 다양한 Git 브랜치 전략 이해
- 팀 규모와 프로젝트에 맞는 워크플로우 선택
- Git Flow, GitHub Flow, Trunk-based Development 비교
- 릴리스 관리 및 버전 전략 수립

## 목차
1. [워크플로우 개요](#1-워크플로우-개요)
2. [Git Flow](#2-git-flow)
3. [GitHub Flow](#3-github-flow)
4. [Trunk-based Development](#4-trunk-based-development)
5. [GitLab Flow](#5-gitlab-flow)
6. [워크플로우 선택 가이드](#6-워크플로우-선택-가이드)
7. [연습 문제](#7-연습-문제)

---

## 1. 워크플로우 개요

### 1.1 브랜치 전략의 중요성

```
좋은 브랜치 전략의 특징:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ✓ 명확한 규칙     - 팀원 모두가 이해하고 따를 수 있음      │
│  ✓ 충돌 최소화     - 병합 충돌과 통합 문제 감소             │
│  ✓ 코드 품질 유지  - 리뷰와 테스트를 강제                   │
│  ✓ 릴리스 용이     - 배포 프로세스가 명확함                 │
│  ✓ 롤백 가능       - 문제 발생 시 빠른 복구                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 주요 워크플로우 비교

```
┌────────────────────────────────────────────────────────────────┐
│                      워크플로우 비교                            │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│              │   Git Flow   │ GitHub Flow  │  Trunk-based     │
├──────────────┼──────────────┼──────────────┼──────────────────┤
│ 복잡도       │     높음     │     낮음     │     낮음         │
│ 브랜치 수    │     많음     │     적음     │     최소         │
│ 릴리스 주기  │   정기적     │    수시      │     수시         │
│ 팀 규모      │   중~대규모  │   소~중규모  │   소~대규모      │
│ 배포 빈도    │     낮음     │     높음     │     매우 높음    │
│ CI/CD 의존도 │     낮음     │     높음     │     매우 높음    │
│ 적합한 경우  │ 릴리스 버전  │ SaaS/웹 앱   │ 지속 배포        │
│              │ 관리 필요    │              │                  │
└──────────────┴──────────────┴──────────────┴──────────────────┘
```

---

## 2. Git Flow

### 2.1 Git Flow 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                        Git Flow 브랜치                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  main (master)  ──●────────●────────────────●─────────●──▶     │
│                   │        ↑                ↑         ↑         │
│                   │        │                │         │         │
│  hotfix          │       ●─●               │         │         │
│                   │        │                │         │         │
│  release         │        │    ●───────●───┘         │         │
│                   │        │    ↑       │             │         │
│                   │        │    │       │             │         │
│  develop ────────●────────●────●───────●─────────────●──▶     │
│                   ↓        ↑    ↑       ↑             │         │
│                   │        │    │       │             │         │
│  feature/A       ●────●───┘    │       │             │         │
│                                │       │             │         │
│  feature/B                    ●───●───┘             │         │
│                                                      │         │
│  feature/C                                          ●─●       │
│                                                                 │
│  영구 브랜치: main, develop                                     │
│  임시 브랜치: feature/*, release/*, hotfix/*                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 브랜치 역할

```
┌─────────────────────────────────────────────────────────────┐
│ 브랜치         │ 용도                                        │
├─────────────────────────────────────────────────────────────┤
│ main          │ 프로덕션 릴리스 (항상 배포 가능 상태)        │
│ develop       │ 개발 통합 브랜치 (다음 릴리스 준비)          │
│ feature/*     │ 새 기능 개발 (develop에서 분기)              │
│ release/*     │ 릴리스 준비 (버그 수정, 문서화)              │
│ hotfix/*      │ 긴급 버그 수정 (main에서 분기)               │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Git Flow 명령어

```bash
# git-flow 도구 설치
brew install git-flow-avh  # macOS
apt-get install git-flow   # Ubuntu

# Git Flow 초기화
git flow init

# 대화형 설정
# Branch name for production releases: [main]
# Branch name for "next release" development: [develop]
# Feature branches? [feature/]
# Release branches? [release/]
# Hotfix branches? [hotfix/]
# Version tag prefix? [v]

# Feature 브랜치
git flow feature start user-auth
# ... 작업 ...
git flow feature finish user-auth

# Release 브랜치
git flow release start 1.2.0
# ... 버그 수정, 문서화 ...
git flow release finish 1.2.0

# Hotfix 브랜치
git flow hotfix start 1.2.1
# ... 긴급 수정 ...
git flow hotfix finish 1.2.1
```

### 2.4 Git Flow 수동 실행

```bash
# ===== Feature 브랜치 =====
# 시작
git checkout develop
git checkout -b feature/user-auth

# 작업 후 완료
git checkout develop
git merge --no-ff feature/user-auth
git branch -d feature/user-auth

# ===== Release 브랜치 =====
# 시작
git checkout develop
git checkout -b release/1.2.0

# 버그 수정 후 완료
git checkout main
git merge --no-ff release/1.2.0
git tag -a v1.2.0 -m "Version 1.2.0"

git checkout develop
git merge --no-ff release/1.2.0
git branch -d release/1.2.0

# ===== Hotfix 브랜치 =====
# 시작
git checkout main
git checkout -b hotfix/1.2.1

# 수정 후 완료
git checkout main
git merge --no-ff hotfix/1.2.1
git tag -a v1.2.1 -m "Hotfix 1.2.1"

git checkout develop
git merge --no-ff hotfix/1.2.1
git branch -d hotfix/1.2.1
```

### 2.5 Git Flow 장단점

```
장점:
✓ 명확한 브랜치 역할 분리
✓ 릴리스 버전 관리 용이
✓ 병렬 개발 지원
✓ 핫픽스와 기능 개발 분리

단점:
✗ 복잡한 브랜치 구조
✗ 느린 통합 주기
✗ CI/CD와 맞지 않을 수 있음
✗ 소규모 팀에 과도함
```

---

## 3. GitHub Flow

### 3.1 GitHub Flow 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                       GitHub Flow                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  main ──●──────●──────●──────●──────●──────●──────●──▶          │
│          ↑      ↑      ↑      ↑      ↑      ↑      ↑            │
│          │      │      │      │      │      │      │            │
│  PR #1  ●──●───┘      │      │      │      │      │            │
│                       │      │      │      │      │            │
│  PR #2            ●──●┘      │      │      │      │            │
│                              │      │      │      │            │
│  PR #3                   ●───┘      │      │      │            │
│                                     │      │      │            │
│  PR #4                          ●──●┘      │      │            │
│                                            │      │            │
│  PR #5                                 ●───┘      │            │
│                                                   │            │
│  PR #6                                       ●───●┘            │
│                                                                 │
│  규칙:                                                          │
│  1. main은 항상 배포 가능                                       │
│  2. 모든 변경은 브랜치에서 시작                                 │
│  3. Pull Request로 리뷰                                         │
│  4. 리뷰 후 main에 병합                                         │
│  5. main 병합 후 즉시 배포                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 GitHub Flow 워크플로우

```bash
# 1. 브랜치 생성
git checkout main
git pull origin main
git checkout -b feature/add-login

# 2. 작업 및 커밋
git add .
git commit -m "Add login functionality"

# 3. 푸시
git push -u origin feature/add-login

# 4. Pull Request 생성 (GitHub UI 또는 CLI)
gh pr create --title "Add login functionality" --body "Description..."

# 5. 코드 리뷰
# - 리뷰어 지정
# - CI 테스트 통과 확인
# - 피드백 반영

# 6. 병합
gh pr merge --squash  # 또는 GitHub UI에서

# 7. 브랜치 삭제
git checkout main
git pull
git branch -d feature/add-login
git push origin --delete feature/add-login

# 8. 배포 (자동 또는 수동)
```

### 3.3 브랜치 네이밍 컨벤션

```bash
# 기능 개발
feature/user-authentication
feature/shopping-cart
feature/JIRA-123-payment-integration

# 버그 수정
bugfix/login-error
bugfix/JIRA-456-cart-calculation

# 개선
improvement/performance-optimization
improvement/code-refactoring

# 문서
docs/api-documentation
docs/readme-update

# 실험
experiment/new-algorithm
spike/caching-solution
```

### 3.4 Pull Request 템플릿

```markdown
<!-- .github/pull_request_template.md -->
## 변경 사항
<!-- 이 PR에서 변경한 내용을 설명해주세요 -->

## 변경 이유
<!-- 왜 이 변경이 필요한지 설명해주세요 -->

## 테스트
- [ ] 단위 테스트 추가/수정
- [ ] 통합 테스트 추가/수정
- [ ] 수동 테스트 완료

## 체크리스트
- [ ] 코드 스타일 가이드 준수
- [ ] 문서 업데이트 (필요한 경우)
- [ ] Breaking changes 없음 (있다면 설명)

## 관련 이슈
Closes #123

## 스크린샷 (UI 변경 시)
<!-- 스크린샷을 첨부해주세요 -->
```

### 3.5 GitHub Flow 장단점

```
장점:
✓ 단순하고 이해하기 쉬움
✓ 빠른 피드백 루프
✓ CI/CD와 잘 맞음
✓ 지속적 배포에 적합

단점:
✗ 릴리스 버전 관리 부재
✗ 여러 버전 유지보수 어려움
✗ 대규모 팀에서 병목 가능
✗ 긴 개발 주기 기능에 부적합
```

---

## 4. Trunk-based Development

### 4.1 Trunk-based 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                   Trunk-based Development                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  main ──●──●──●──●──●──●──●──●──●──●──●──●──●──●──●──▶         │
│          │  │  │  │  │  │  │  │  │  │  │  │  │  │  │           │
│          ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑           │
│          │  │  │  │  │  │  │  │  │  │  │  │  │  │  │           │
│  개발자  A  B  A  C  B  A  B  C  A  B  C  A  B  A  C           │
│                                                                 │
│  특징:                                                          │
│  • 모든 개발자가 trunk(main)에 직접 커밋                        │
│  • 또는 매우 짧은 수명의 브랜치 사용 (1-2일)                    │
│  • 하루에 여러 번 통합                                          │
│  • Feature Flags로 미완성 기능 제어                             │
│  • 강력한 CI/CD 필수                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Feature Flags

```javascript
// 기능 플래그 예시 (JavaScript)
const featureFlags = {
  newCheckout: process.env.FEATURE_NEW_CHECKOUT === 'true',
  darkMode: process.env.FEATURE_DARK_MODE === 'true',
  experimentalSearch: process.env.FEATURE_EXPERIMENTAL_SEARCH === 'true',
};

// 사용
function renderCheckout() {
  if (featureFlags.newCheckout) {
    return <NewCheckout />;
  }
  return <OldCheckout />;
}

// 점진적 롤아웃
function isFeatureEnabled(userId, feature, percentage) {
  const hash = hashFunction(userId + feature);
  return (hash % 100) < percentage;
}
```

```yaml
# Feature Flag 서비스 (예: LaunchDarkly, Unleash)
# unleash 설정 예시
features:
  - name: new-checkout
    enabled: true
    strategies:
      - name: gradualRollout
        parameters:
          percentage: 25
      - name: userWithId
        parameters:
          userIds: "user-123,user-456"
```

### 4.3 Trunk-based 워크플로우

```bash
# 방법 1: 직접 main에 커밋 (소규모 팀)
git checkout main
git pull --rebase origin main
# ... 작업 ...
git add .
git commit -m "Add feature X"
git pull --rebase origin main  # 충돌 해결
git push origin main

# 방법 2: 짧은 수명 브랜치 (일반적)
git checkout main
git pull origin main
git checkout -b short-lived/add-feature

# 작업 (최대 1-2일)
git add .
git commit -m "Add feature X"

# 빠른 통합
git checkout main
git pull origin main
git merge --no-ff short-lived/add-feature
git push origin main
git branch -d short-lived/add-feature
```

### 4.4 릴리스 브랜치 (선택적)

```
┌─────────────────────────────────────────────────────────────────┐
│               Trunk-based with Release Branches                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  main ──●──●──●──●──●──●──●──●──●──●──●──●──●──●──●──▶         │
│               │              │              │                   │
│               ▼              ▼              ▼                   │
│  release/1.0 ──●─────▶       │              │                   │
│                              │              │                   │
│  release/1.1 ────────────────●──●──▶       │                   │
│                                             │                   │
│  release/1.2 ───────────────────────────────●──●──▶            │
│                                                                 │
│  • 릴리스 브랜치는 main에서 생성                                │
│  • 릴리스 브랜치에서는 버그 수정만                              │
│  • Cherry-pick으로 수정사항 반영                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```bash
# 릴리스 브랜치 생성
git checkout main
git checkout -b release/1.2
git push -u origin release/1.2

# 버그 수정 (main에서 작업 후 cherry-pick)
git checkout main
git commit -m "Fix critical bug"
git checkout release/1.2
git cherry-pick <commit-hash>

# 또는 릴리스에서 직접 수정 후 main에 반영
git checkout release/1.2
git commit -m "Fix bug in release"
git checkout main
git cherry-pick <commit-hash>
```

### 4.5 Trunk-based 장단점

```
장점:
✓ 통합 지옥(integration hell) 방지
✓ 매우 빠른 피드백
✓ 지속적 배포에 최적
✓ 코드 충돌 최소화
✓ 항상 릴리스 가능 상태

단점:
✗ 강력한 CI/CD 인프라 필수
✗ Feature Flags 관리 복잡
✗ 불완전한 기능이 main에 존재
✗ 높은 테스트 커버리지 필요
✗ 팀의 성숙도 필요
```

---

## 5. GitLab Flow

### 5.1 GitLab Flow 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitLab Flow                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  방식 1: 환경 브랜치                                            │
│                                                                 │
│  main ──●──●──●──●──●──●──●──▶                                  │
│              │     │     │                                      │
│              ▼     ▼     ▼                                      │
│  staging ────●─────●─────●──▶                                   │
│                    │     │                                      │
│                    ▼     ▼                                      │
│  production ───────●─────●──▶                                   │
│                                                                 │
│  방식 2: 릴리스 브랜치                                          │
│                                                                 │
│  main ──●──●──●──●──●──●──●──▶                                  │
│              │           │                                      │
│              ▼           ▼                                      │
│  2.3-stable ──●──●──▶   │                                      │
│                         │                                      │
│  2.4-stable ─────────────●──●──▶                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 환경 기반 GitLab Flow

```bash
# 1. Feature 브랜치에서 개발
git checkout main
git checkout -b feature/new-feature
# ... 개발 ...
git push -u origin feature/new-feature

# 2. Merge Request → main
# 코드 리뷰 후 병합

# 3. main → staging (자동 또는 수동)
git checkout staging
git merge main
git push origin staging
# → 스테이징 환경 자동 배포

# 4. staging → production (승인 후)
git checkout production
git merge staging
git push origin production
# → 프로덕션 환경 자동 배포
```

### 5.3 릴리스 기반 GitLab Flow

```bash
# 릴리스 브랜치 생성
git checkout main
git checkout -b 2.4-stable
git push -u origin 2.4-stable

# main에서 버그 수정
git checkout main
git commit -m "Fix critical bug"

# 릴리스 브랜치에 cherry-pick
git checkout 2.4-stable
git cherry-pick <commit-hash>
git push origin 2.4-stable

# 릴리스 태그
git tag v2.4.1
git push origin v2.4.1
```

---

## 6. 워크플로우 선택 가이드

### 6.1 선택 기준

```
┌─────────────────────────────────────────────────────────────────┐
│                      워크플로우 선택 매트릭스                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  배포 빈도                                                      │
│  높음 │         Trunk-based ◄───┐                              │
│       │              ▲          │                              │
│       │              │          │ CI/CD 성숙도                 │
│       │         GitHub Flow     │                              │
│       │              ▲          │                              │
│       │              │          │                              │
│  낮음 │         Git Flow        │                              │
│       └──────────────────────────┘                              │
│       작음         팀 규모         큼                           │
│                                                                 │
│  결정 트리:                                                     │
│  1. 정기 릴리스가 필요한가? → Yes → Git Flow                   │
│  2. CI/CD가 잘 갖춰져 있는가? → Yes → Trunk-based              │
│  3. 단순함이 중요한가? → Yes → GitHub Flow                     │
│  4. 여러 환경이 있는가? → Yes → GitLab Flow                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 상황별 추천

```
┌────────────────────────────────────────────────────────────────┐
│ 상황                              │ 추천 워크플로우             │
├───────────────────────────────────┼────────────────────────────┤
│ 스타트업, 작은 팀 (1-5명)         │ GitHub Flow                │
│ 웹 애플리케이션, SaaS             │ GitHub Flow / Trunk-based  │
│ 모바일 앱 (앱스토어 배포)         │ Git Flow                   │
│ 오픈소스 프로젝트                 │ Git Flow / GitHub Flow     │
│ 기업 소프트웨어 (정기 릴리스)     │ Git Flow                   │
│ DevOps 성숙 조직                  │ Trunk-based                │
│ 여러 환경 (dev/staging/prod)      │ GitLab Flow                │
│ 마이크로서비스                    │ GitHub Flow / Trunk-based  │
│ 레거시 시스템 유지보수            │ Git Flow                   │
└───────────────────────────────────┴────────────────────────────┘
```

### 6.3 하이브리드 접근법

```
많은 팀이 순수한 워크플로우 대신 하이브리드 방식을 사용합니다:

예시: GitHub Flow + Release 브랜치
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  main ──●──●──●──●──●──●──●──●──●──●──●──●──▶                   │
│          ↑  ↑  ↑  │     ↑  ↑  ↑  │     ↑  ↑                    │
│  PR     ●  ●  ●  │     ●  ●  ●  │     ●  ●                    │
│                  ▼              ▼                               │
│  release/1.0 ────●──●──▶       │                               │
│                                 │                               │
│  release/1.1 ───────────────────●──●──▶                        │
│                                                                 │
│  • 일반 개발은 GitHub Flow 방식                                 │
│  • 릴리스가 필요할 때만 릴리스 브랜치 생성                      │
│  • 핫픽스는 릴리스 브랜치에서 main으로 병합                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 버전 관리 전략

```bash
# Semantic Versioning (SemVer)
# MAJOR.MINOR.PATCH
# 1.2.3

# MAJOR: 호환되지 않는 API 변경
# MINOR: 하위 호환되는 기능 추가
# PATCH: 하위 호환되는 버그 수정

# 태그 생성
git tag -a v1.2.3 -m "Release version 1.2.3"
git push origin v1.2.3

# 자동 버전 관리 도구
# npm version patch  # 1.2.3 → 1.2.4
# npm version minor  # 1.2.3 → 1.3.0
# npm version major  # 1.2.3 → 2.0.0

# Conventional Commits + 자동 버전 관리
# feat: → minor
# fix: → patch
# BREAKING CHANGE: → major
```

---

## 7. 연습 문제

### 연습 1: Git Flow 실습
```bash
# 요구사항:
# 1. Git Flow 초기화
# 2. feature/login 브랜치 생성 및 작업
# 3. release/1.0.0 생성
# 4. hotfix/1.0.1 시뮬레이션
# 5. 모든 단계 문서화

# 명령어 작성:
```

### 연습 2: GitHub Flow 실습
```bash
# 요구사항:
# 1. 브랜치 생성 규칙 정의
# 2. PR 템플릿 작성
# 3. 브랜치 보호 규칙 설정 (GitHub)
# 4. 자동 병합 후 삭제 설정

# 설정 및 명령어 작성:
```

### 연습 3: Feature Flags 구현
```javascript
// 요구사항:
// 1. Feature Flag 관리 시스템 설계
// 2. 점진적 롤아웃 로직 구현
// 3. A/B 테스트 지원
// 4. 환경별 설정 분리

// 코드 작성:
```

### 연습 4: 워크플로우 전환 계획
```markdown
# 요구사항:
# 현재 Git Flow를 사용하는 팀이 GitHub Flow로 전환하려 합니다.
# 전환 계획을 작성하세요.

# 포함할 내용:
# 1. 현재 상태 분석
# 2. 전환 단계
# 3. 팀 교육 계획
# 4. 롤백 계획
# 5. 성공 지표
```

---

## 다음 단계

- [09_고급_Git_기법](09_고급_Git_기법.md) - hooks, submodules, worktrees
- [10_모노레포_관리](10_모노레포_관리.md) - 대규모 저장소 관리
- [07_GitHub_Actions](07_GitHub_Actions.md) - CI/CD 자동화 복습

## 참고 자료

- [Git Flow 원본 문서](https://nvie.com/posts/a-successful-git-branching-model/)
- [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Trunk Based Development](https://trunkbaseddevelopment.com/)
- [GitLab Flow](https://docs.gitlab.com/ee/topics/gitlab_flow.html)

---

[← 이전: GitHub Actions](07_GitHub_Actions.md) | [다음: 고급 Git 기법 →](09_고급_Git_기법.md) | [목차](00_Overview.md)
