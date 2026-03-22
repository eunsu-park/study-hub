# 버전 관리와 협업

> **토픽**: Programming
> **레슨**: 14 of 16
> **선수 지식**: 기본 Git 지식(clone, commit, push, pull), 명령줄 익숙함
> **목표**: 전문 소프트웨어 개발을 위한 브랜칭 전략, 코드 리뷰 관행, CI/CD 기초, 협업 워크플로우 마스터

## 소개

버전 관리는 현대 소프트웨어 개발의 기초입니다. 이를 통해:
- **히스토리 추적**: 모든 변경 사항이 컨텍스트와 함께 기록됨
- **협업**: 여러 개발자가 충돌 없이 동시에 작업
- **실험**: 위험한 변경을 두려움 없이 시도
- **롤백 기능**: 실수를 쉽게 되돌림
- **코드 리뷰**: 병합 전 체계적 품질 게이트

이 레슨은 팀을 생산적으로 만드는 버전 관리 도구 위에 구축된 **워크플로우**—인간 프로세스—를 다룹니다.

## 버전 관리의 간략한 역사

진화를 이해하면 현대 도구가 왜 그런 방식으로 작동하는지 이해하는 데 도움이 됩니다:

1. **RCS(1982)**: 단일 파일 잠금, 한 번에 한 개발자
2. **CVS(1986)**: 다중 파일 버전 관리, 동시 편집
3. **SVN/Subversion(2000)**: 중앙 집중식 서버, 원자적 커밋, 더 나은 바이너리 처리
4. **Git(2005)**: 분산, 브랜칭 우선 설계, 매우 빠름
5. **Mercurial(2005)**: 분산, Git보다 단순, 적은 채택

**주요 패러다임 전환**: **중앙 집중식(Centralized)**(SVN) → **분산(Distributed)**(Git)

### 중앙 집중식 vs 분산

**중앙 집중식(SVN)**:
```
        중앙 서버
             |
     +-------+-------+
     |       |       |
  개발자 A 개발자 B 개발자 C
```
- 서버의 단일 진실 공급원
- 커밋에 네트워크 연결 필요
- 브랜치가 비쌈(서버에 전체 복사본)

**분산(Git)**:
```
    원격 저장소(GitHub/GitLab)
             |
     +-------+-------+
     |       |       |
   로컬    로컬    로컬
  저장소 A 저장소 B 저장소 C
```
- 모든 개발자가 전체 히스토리 보유
- 커밋은 로컬(빠르고, 오프라인 가능)
- 브랜치가 저렴(커밋에 대한 포인터)
- Push/pull로 원격과 동기화

## Git 기초(빠른 복습)

워크플로우로 들어가기 전에 다음 개념을 이해하고 있는지 확인하세요:

### 저장소, 커밋, 브랜치

```bash
# 저장소: 모든 히스토리를 포함하는 .git 폴더
git init

# 커밋: 특정 시점의 프로젝트 스냅샷
git add file.txt
git commit -m "Add feature X"

# 브랜치: 커밋에 대한 이동 가능한 포인터
git branch feature-login
git checkout feature-login  # 또는: git checkout -b feature-login
```

### 세 가지 트리

```
작업 디렉토리  →  스테이징 영역  →  저장소
              (git add)      (git commit)
```

```bash
# 파일 수정
echo "Hello" > file.txt

# 변경 사항 스테이징
git add file.txt

# 저장소에 커밋
git commit -m "Add greeting"
```

### Merge vs Rebase

**Merge**: 브랜치를 결합하고 히스토리 보존
```bash
git checkout main
git merge feature-branch
```
```
    A---B---C  main
         \   \
          D---E  feature-branch
```

**Rebase**: 다른 브랜치 위에 커밋을 재생하여 히스토리 재작성
```bash
git checkout feature-branch
git rebase main
```
```
    A---B---C  main
             \
              D'---E'  feature-branch (rebased)
```

**언제 사용**:
- **Merge**: 공개 브랜치, 완전한 히스토리 보존
- **Rebase**: 비공개 브랜치, 더 깨끗한 선형 히스토리

## 브랜칭 전략

브랜칭 전략은 속도, 안정성, 협업의 균형을 맞추기 위해 **팀이 코드 변경을 조직하는 방법**을 정의합니다.

### 1. Git Flow

**Git Flow**는 여러 장기 브랜치를 가진 구조화된 브랜칭 모델입니다.

**브랜치**:
- **main**: 프로덕션 준비 코드
- **develop**: 기능을 위한 통합 브랜치
- **feature/**: 새 기능(develop에서 분기)
- **release/**: 릴리스 준비(develop에서 분기)
- **hotfix/**: 긴급 수정(main에서 분기)

**워크플로우**:
```bash
# 새 기능 시작
git checkout develop
git checkout -b feature/user-authentication

# 기능 작업
git commit -m "Add login form"
git commit -m "Add password validation"

# develop으로 다시 병합
git checkout develop
git merge feature/user-authentication

# 릴리스 준비
git checkout -b release/v1.2.0
# 버그 수정, 버전 번호 업데이트
git checkout main
git merge release/v1.2.0
git tag v1.2.0

# 프로덕션 핫픽스
git checkout main
git checkout -b hotfix/security-patch
git commit -m "Fix CVE-2024-1234"
git checkout main
git merge hotfix/security-patch
git tag v1.2.1
git checkout develop
git merge hotfix/security-patch
```

**장점**:
- 명확한 관심사 분리
- 예정된 릴리스에 적합(예: 월별)
- 기능과 릴리스의 병렬 개발

**단점**:
- 많은 브랜치로 복잡
- 소규모 팀이나 지속적 배포에는 오버헤드
- develop에서 병합 충돌이 누적될 수 있음

**최적**: 예정된 릴리스가 있는 전통적 소프트웨어(분기별, 월별).

### 2. GitHub Flow

**GitHub Flow**는 하나의 장기 브랜치 `main`을 가진 단순화된 워크플로우입니다.

**워크플로우**:
```bash
# main에서 기능 브랜치 생성
git checkout main
git pull origin main
git checkout -b add-search-feature

# 변경
git commit -m "Add search endpoint"
git commit -m "Add search UI"

# Push하고 Pull Request 열기
git push origin add-search-feature
# GitHub에서 PR 열기

# 코드 리뷰와 CI 통과 후, PR 병합
# 브랜치 삭제
```

**규칙**:
1. `main`은 항상 배포 가능
2. 설명적인 브랜치 이름 생성(`fix-login-bug`, `patch-1` 아님)
3. 논의를 위해 PR을 일찍 열기
4. 병합 후 `main`에서 배포

**장점**:
- 단순: 추적할 브랜치가 하나
- 빠른 반복
- 지속적 배포와 작동

**단점**:
- 강력한 CI/CD 필요
- main의 불완전한 기능에 기능 플래그 필요
- 대규모 팀에는 구조가 적음

**최적**: 지속적 배포를 하는 웹 애플리케이션, 중소 규모 팀.

### 3. 트랙 기반 개발

**트랙 기반 개발(Trunk-Based Development)**은 단기 브랜치(< 1일)와 빈번한 통합을 강조합니다.

**워크플로우**:
```bash
# 작은 변경은 main에 직접 커밋
git checkout main
git pull origin main
# 작은 변경
git commit -m "Refactor user service"
git push origin main

# 큰 변경은 단기 브랜치 사용
git checkout -b refactor-database
# 몇 시간 작업
git commit -m "Extract repository pattern"
git push origin refactor-database
# PR 열기, 빠른 리뷰, 당일 병합
```

**규칙**:
- 하루에 최소 한 번 `main`에 커밋
- 불완전한 기능에 기능 플래그 사용
- 장기 브랜치 없음
- 엄격한 자동화 테스트

**장점**:
- 병합 충돌 최소화(빈번한 통합)
- 작고 점진적인 변경 장려
- 빠른 피드백 루프

**단점**:
- 성숙한 CI/CD와 테스트 필요
- 기능 플래그가 복잡성 추가
- 강한 엔지니어링 규율 없이는 위험

**최적**: 고성능 팀, 지속적 배포를 하는 SaaS 제품.

### 비교 표

| 전략 | 브랜치 복잡도 | 릴리스 주기 | 팀 규모 | CI/CD 필요 |
|----------|-------------------|-----------------|-----------|----------------|
| Git Flow | 높음 | 예정됨 | 대규모 | 보통 |
| GitHub Flow | 낮음 | 지속적 | 중소 | 높음 |
| 트랙 기반 | 매우 낮음 | 지속적 | 모두 | 매우 높음 |

## 풀 리퀘스트 / 머지 리퀘스트

**풀 리퀘스트(PR)**는 병합 전 코드 리뷰와 논의 메커니즘입니다.

### 좋은 PR의 구조

**1. 제목**: 간결하고 설명적
```
✅ OAuth2로 사용자 인증 추가
❌ 코드 업데이트
```

**2. 설명**: 컨텍스트와 테스트 지침
```markdown
## 요약
Google 로그인을 사용한 OAuth2 인증 플로우 구현.

## 변경사항
- OAuth2 클라이언트 라이브러리 추가
- 로그인/콜백 라우트 생성
- Redis에 사용자 세션 저장
- 인증 미들웨어 추가

## 테스트
1. Redis 시작: `docker run -p 6379:6379 redis`
2. `.env`에 환경 변수 설정
3. `/login` 방문하여 Google로 로그인
4. `/dashboard`로 리다이렉트 확인

## 스크린샷
[로그인 플로우 스크린샷]

## 관련 이슈
Closes #123
```

**3. 범위**: 작은 PR이 더 빠르게 리뷰됨
```
✅ 50-200줄: 빠른 리뷰
⚠️ 200-500줄: 시간 소요
❌ 1000+줄: 피하기; 여러 PR로 분할
```

### 작은 PR vs 큰 PR

**작은 PR**:
- 리뷰하기 쉬움(인지 부하 적음)
- 더 빠른 병합(컨텍스트 전환 적음)
- 낮은 위험(제한된 영향 범위)
- 더 빈번한 통합(충돌 적음)

**큰 PR**:
- 리뷰 피로: 리뷰어가 주의 깊게 읽지 않고 승인
- 장기 브랜치: 병합 충돌 누적
- 높은 위험: 큰 변경은 되돌리기 어려움

**전략**: 작업을 점진적 PR로 분할:
```
❌ 하나의 PR: "사용자 대시보드 구현"(1500줄)

✅ 점진적 PR:
  1. "사용자 모델과 데이터베이스 스키마 추가"(100줄)
  2. "사용자 서비스 API 추가"(150줄)
  3. "대시보드 UI 추가"(200줄)
  4. "대시보드 데이터 가져오기 추가"(100줄)
```

## 코드 리뷰 모범 사례

코드 리뷰는 **기술**입니다. 작성자와 리뷰어 모두 신중해야 합니다.

### 무엇을 확인할 것인가

**1. 정확성**: 코드가 작동하는가?
- 논리 오류
- 엣지 케이스 처리
- 오류 처리

**2. 설계**: 잘 구조화되었는가?
- 관심사 분리
- 적절한 추상화
- 디자인 패턴 사용

**3. 복잡도**: 이해하기 쉬운가?
- 지나치게 영리한 코드
- 명확하지 않은 논리에 대한 주석 누락

**4. 테스트**: 적절히 테스트되었는가?
- 새 코드에 대한 테스트 커버리지
- 엣지 케이스 테스트
- 테스트가 읽기 쉬운가

**5. 보안**: 취약점이 있는가?
- SQL 인젝션, XSS 위험
- 코드의 시크릿
- 인증/권한 부여

**6. 성능**: 효율적인가?
- 불필요한 루프
- 비효율적인 데이터베이스 쿼리(N+1)
- 메모리 누수

**7. 스타일**: 규칙을 따르는가?
- 네이밍 규칙
- 코드 포맷팅
- 문서화

### 건설적 피드백

**친절하고 구체적으로**:
```
❌ "잘못되었습니다."
✅ "이 함수는 `data`가 null이면 예외를 던질 수 있습니다.
   23줄에 null 체크를 추가하는 것을 고려하세요."

❌ "나쁜 네이밍."
✅ "변수 이름 `d`가 불명확합니다. 명확성을 위해
   `daysUntilExpiration`으로 변경하는 것을 고려하세요."
```

**요구하지 말고 질문하기**:
```
❌ "HashMap을 사용하도록 변경하세요."
✅ "O(n) 대신 O(1) 조회를 위해 여기서 HashMap을 사용할 수 있을까요?"
```

**좋은 작업 칭찬**:
```
✅ "좋은 리팩토링! 훨씬 더 읽기 쉽습니다."
✅ "엣지 케이스에 대한 훌륭한 테스트 커버리지."
```

### 자동화할 수 있는 것은 자동화

기계가 확인할 수 있는 것에 인간 시간을 낭비하지 마세요:

- **린터**: `eslint`, `pylint`, `rubocop`
- **포매터**: `prettier`, `black`, `gofmt`
- **타입 체커**: `TypeScript`, `mypy`, `Flow`
- **보안 스캐너**: `Snyk`, `Dependabot`

CI에서 자동으로 실행되도록 구성하세요.

### 코드 리뷰 체크리스트

팀별 체크리스트 생성:
```markdown
## 코드 리뷰 체크리스트

### 기능
- [ ] 코드가 의도대로 작동함
- [ ] 엣지 케이스 처리됨
- [ ] 오류 처리가 적절함

### 테스트
- [ ] 새 코드에 테스트 있음
- [ ] 테스트가 포괄적임
- [ ] 로컬에서 테스트 통과

### 설계
- [ ] 코드가 모듈화되고 SOLID 원칙 따름
- [ ] 불필요한 복잡성 없음

### 보안
- [ ] 하드코딩된 시크릿 없음
- [ ] 입력 유효성 검증 있음
- [ ] 권한 확인이 올바름

### 문서화
- [ ] 공개 API 문서화됨
- [ ] README 업데이트됨(필요시)
```

## 병합 전략

PR을 병합할 때 세 가지 옵션이 있습니다:

### 1. 병합 커밋

전체 히스토리를 보존하는 병합 커밋 생성:
```
    A---B---C  main
         \   \
          D---E  feature (F에서 병합 커밋)
               \
                F  main (병합 후)
```

```bash
git checkout main
git merge --no-ff feature-branch
```

**장점**: 전체 히스토리 보존, 전체 기능 되돌리기 쉬움
**단점**: 많은 병합 커밋으로 히스토리 복잡

### 2. 스쿼시와 병합

모든 기능 커밋을 하나로 결합:
```
    A---B---C---D  main
         \
          E---F---G  feature (D로 스쿼시됨)
```

```bash
git checkout main
git merge --squash feature-branch
git commit -m "Add feature X (squashed)"
```

**장점**: 깨끗한 선형 히스토리, 기능당 하나의 커밋
**단점**: 개별 커밋 히스토리 손실, bisect 어려움

### 3. 리베이스와 병합

main 위에 커밋 재생:
```
    A---B---C  main
             \
              D'---E'  feature (rebased)
```

```bash
git checkout feature-branch
git rebase main
git checkout main
git merge feature-branch  # Fast-forward 병합
```

**장점**: 선형 히스토리, 개별 커밋 보존
**단점**: 히스토리 재작성(공개 브랜치는 리베이스하지 말 것)

### 권장사항

- **스쿼시**: 작은 기능, 버그 수정(가장 일반적)
- **리베이스**: 보존할 가치가 있는 잘 구조화된 커밋
- **병합 커밋**: 대규모 기능, 릴리스 브랜치

## CI/CD 기초

**지속적 통합(CI)**: 모든 push에서 자동으로 코드 빌드 및 테스트.
**지속적 전달(CD)**: main 브랜치를 항상 배포 가능하게 유지.
**지속적 배포**: 테스트 통과 후 프로덕션에 자동 배포.

### CI 파이프라인 예제

```yaml
# .github/workflows/ci.yml (GitHub Actions)
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Run linter
        run: npm run lint

      - name: Run tests
        run: npm test

      - name: Build
        run: npm run build

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run security scan
        run: npm audit
```

### CD 파이프라인 예제

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run tests
        run: npm test

      - name: Build Docker image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Push to registry
        run: docker push myapp:${{ github.sha }}

      - name: Deploy to production
        run: kubectl set image deployment/myapp myapp=myapp:${{ github.sha }}
```

### 파이프라인 단계

일반적인 순서의 단계:
1. **코드 체크아웃**: 저장소 복제
2. **의존성 설치**: `npm install`, `pip install`
3. **린트**: 코드 스타일 확인
4. **테스트**: 단위, 통합 테스트 실행
5. **보안 스캔**: 취약점 확인
6. **빌드**: 컴파일, 번들
7. **배포**: 스테이징/프로덕션으로 푸시

**빠른 피드백**: 빠르게 실패—느린 것(E2E 테스트) 전에 빠른 확인(린트) 실행.

## 모노레포 vs 폴리레포

### 모노레포

**모든 프로젝트를 하나의 저장소에**:
```
company-repo/
├── services/
│   ├── api/
│   ├── frontend/
│   └── worker/
├── libraries/
│   ├── shared-utils/
│   └── ui-components/
└── tools/
```

**장점**:
- 프로젝트 간 원자적 변경
- 코드 공유가 쉬움
- 단일 CI/CD 구성
- 단순화된 의존성 관리

**단점**:
- 대규모 저장소(clone, checkout 느림)
- 관련 없는 변경에 대해 CI 실행
- 툴링 필요(Bazel, Nx, Turborepo)

**사용**: Google, Facebook, Microsoft

### 폴리레포

**각 프로젝트에 대해 별도 저장소**:
```
company-api/
company-frontend/
company-worker/
shared-utils/
ui-components/
```

**장점**:
- 명확한 소유권 경계
- 더 빠른 clone
- 대상 CI(영향받은 저장소만)

**단점**:
- 저장소 간 변경이 고통스러움
- 의존성 버전 지옥
- 코드 중복

**사용**: 대부분의 중소 기업

### 권장사항

- **모노레포**: 프로젝트가 긴밀하게 결합되고, 빈번한 프로젝트 간 변경
- **폴리레포**: 프로젝트가 독립적인 마이크로서비스

## 시맨틱 버전 관리

**시맨틱 버전 관리(SemVer)**는 버전 번호를 통해 변경의 성격을 전달합니다:

```
MAJOR.MINOR.PATCH
  2  . 3   . 1
```

- **MAJOR**: 파괴적 변경(호환되지 않는 API 변경)
- **MINOR**: 새 기능(하위 호환)
- **PATCH**: 버그 수정(하위 호환)

**예제**:
- `1.0.0 → 1.0.1`: 버그 수정
- `1.0.1 → 1.1.0`: 새 기능 추가
- `1.1.0 → 2.0.0`: 파괴적 변경(API 변경됨)

**사전 릴리스 버전**:
- `1.0.0-alpha.1`: 알파 릴리스
- `1.0.0-beta.2`: 베타 릴리스
- `1.0.0-rc.1`: 릴리스 후보

## 커밋 메시지 규칙

좋은 커밋 메시지는 다음을 가능하게 합니다:
- 변경 사항의 빠른 이해
- 자동화된 변경 로그 생성
- 히스토리의 쉬운 탐색

### 관례적 커밋

형식:
```
<type>(<scope>): <description>

[선택적 본문]

[선택적 바닥글]
```

**타입**:
- `feat`: 새 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 포맷팅(코드 변경 없음)
- `refactor`: 코드 재구조화
- `test`: 테스트 추가
- `chore`: 툴링, 의존성

**예제**:
```
feat(auth): OAuth2 로그인 추가

OAuth2를 사용한 Google 로그인 플로우 구현.
사용자가 이제 Google 계정으로 로그인할 수 있습니다.

Closes #123

---

fix(api): getUser 엔드포인트에서 null 사용자 ID 처리

이전에는 null ID가 500 오류를 발생시켰습니다.
이제 오류 메시지와 함께 400 Bad Request 반환.

---

docs(readme): 설치 지침 업데이트

---

refactor(db): 저장소 패턴 추출

더 나은 관심사 분리를 위해
서비스에서 저장소로 데이터베이스 논리 이동.
```

### 의미 있는 메시지 작성

**나쁨**:
```
git commit -m "fix bug"
git commit -m "update code"
git commit -m "changes"
```

**좋음**:
```
git commit -m "fix: 사용자 등록에서 경쟁 조건 방지"
git commit -m "refactor: 이메일 유효성 검증을 유틸리티 함수로 추출"
git commit -m "feat: 사용자 목록 엔드포인트에 페이지네이션 추가"
```

**팁**:
- 명령형 사용: "Add feature" not "Added feature"
- 첫 줄: 간결한 요약(50자)
- 본문: **왜**를 설명, 무엇이 아니라(코드가 무엇을 보여줌)

## 연습 문제

### 연습 문제 1: 브랜칭 전략 선택

다음 특성을 가진 팀을 이끌고 있습니다:
- 5명의 개발자
- Heroku에 배포되는 웹 애플리케이션
- 하루에 여러 번 배포
- 불완전한 기능에 기능 플래그 사용

어떤 브랜칭 전략을 추천하시겠습니까? 선택을 정당화하세요.

### 연습 문제 2: 코드 리뷰 체크리스트 작성

Flask와 PostgreSQL을 사용하는 Python 웹 API 프로젝트를 위한 코드 리뷰 체크리스트를 만드세요. 기능, 보안, 성능, 스타일을 다루는 최소 10개 항목을 포함하세요.

### 연습 문제 3: CI/CD 파이프라인 설계

다음을 수행하는 Node.js 애플리케이션을 위한 CI/CD 파이프라인을 설계하세요:
1. 모든 브랜치의 push에서 실행
2. 린팅, 테스트, 보안 스캔 실행
3. Docker 이미지 빌드
4. `main`에 대한 push에서 스테이징에 배포
5. git 태그(`v*`)에서 프로덕션에 배포

YAML로 파이프라인 구성 작성(GitHub Actions 또는 GitLab CI).

### 연습 문제 4: PR 품질 평가

이 풀 리퀘스트를 평가하고 피드백을 제공하세요:

**제목**: "사용자 관련 업데이트"
**설명**: (비어 있음)
**변경사항**: 45개 파일 변경, 2,300줄 추가, 800 삭제
**커밋**: "wip", "fix", "more changes"와 같은 메시지가 있는 37개 커밋

문제는 무엇입니까? 작성자가 어떻게 개선해야 합니까?

### 연습 문제 5: 시맨틱 버전 관리

API가 현재 버전 `2.3.5`입니다. 각 시나리오에 대한 다음 버전 번호를 결정하세요:

1. 인증 미들웨어의 버그 수정
2. 기존 엔드포인트에 새로운 선택적 매개변수 추가
3. 폐기된 엔드포인트 제거
4. 내부 캐싱 개선(API 변경 없음)
5. JSON 응답의 필드 이름을 `user_name`에서 `username`으로 변경

## 요약

효과적인 버전 관리 워크플로우는 팀이 깨뜨리지 않고 빠르게 움직일 수 있게 합니다:

- **브랜칭 전략**: Git Flow(구조화), GitHub Flow(단순), 트랙 기반(빠름)
- **풀 리퀘스트**: 명확한 설명이 있는 작고 집중된 PR이 더 빠르게 리뷰됨
- **코드 리뷰**: 건설적이고, 스타일 체크 자동화, 설계와 정확성에 집중
- **병합 전략**: 스쿼시(깨끗한 히스토리), 리베이스(커밋 보존), 병합(전체 히스토리)
- **CI/CD**: 자동화된 테스트와 배포가 인적 오류를 줄이고 전달 가속화
- **모노레포 vs 폴리레포**: 긴밀한 결합에는 모노레포, 독립성에는 폴리레포
- **시맨틱 버전 관리**: MAJOR.MINOR.PATCH가 변경의 영향 전달
- **커밋 메시지**: 관례적 커밋은 자동화와 명확성 가능

훌륭한 워크플로우는 **속도**(기능 빠르게 배송)와 **품질**(버그 방지, 코드베이스 유지) 사이의 균형을 맞춥니다.

## 내비게이션

[← 이전: API 설계](13_API_Design.md) | [다음: 소프트웨어 아키텍처 →](15_Software_Architecture.md)
