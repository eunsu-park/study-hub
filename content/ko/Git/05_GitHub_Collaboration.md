# GitHub 협업

**이전**: [GitHub 시작하기](./04_GitHub_Getting_Started.md) | **다음**: [Git 고급 명령어](./06_Git_Advanced.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Collaborator 모델과 Fork & Pull Request 모델의 차이를 구분할 수 있다
2. 저장소를 Fork하고, 변경 사항을 만들고, Pull Request를 제출할 수 있다
3. Pull Request에 코멘트를 달고 승인하는 방식으로 코드 리뷰(Code Review)를 수행할 수 있다
4. GitHub Issues를 사용하여 버그, 기능 요청, 작업을 추적할 수 있다
5. 리뷰 및 CI 요구사항을 강제하기 위한 브랜치 보호 규칙(Branch Protection Rule)을 설정할 수 있다
6. Fork 기반 오픈소스 기여 워크플로우를 처음부터 끝까지 적용할 수 있다

---

코드를 작성하는 것은 일의 절반에 불과하며, 나머지 절반은 다른 사람들과 협력하는 것입니다. GitHub의 협업 기능인 Pull Request, 코드 리뷰(Code Review), Issues, 브랜치 보호는 버그를 조기에 발견하고, 팀 전체에 지식을 공유하며, main 브랜치를 안정적으로 유지하는 구조화된 프로세스를 제공합니다. 오픈소스에 기여하든 팀 프로젝트에서 작업하든, 이러한 기술은 필수적입니다.

## 1. 협업 워크플로우 개요

GitHub에서 협업하는 두 가지 주요 방식:

| 방식 | 설명 | 사용 경우 |
|------|------|----------|
| **Collaborator** | 저장소에 직접 푸시 권한 | 팀 프로젝트 |
| **Fork & PR** | 복제 후 Pull Request | 오픈소스 기여 |

---

## 2. Fork (포크)

다른 사람의 저장소를 내 계정으로 복사합니다.

### Fork 하는 방법

1. 원본 저장소 페이지 방문
2. 우측 상단 "Fork" 버튼 클릭
3. 내 계정으로 복사됨

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  원본: octocat/hello-world                              │
│         │                                               │
│         │ Fork                                          │
│         ▼                                               │
│  내 계정: myname/hello-world  ← 독립적인 복사본          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Fork 후 작업 흐름

```bash
# 1. Fork한 저장소 클론
git clone git@github.com:myname/hello-world.git
cd hello-world

# 2. 원본 저장소를 upstream으로 추가
git remote add upstream git@github.com:octocat/hello-world.git

# 3. remote 확인
git remote -v
# origin    git@github.com:myname/hello-world.git (fetch)
# origin    git@github.com:myname/hello-world.git (push)
# upstream  git@github.com:octocat/hello-world.git (fetch)
# upstream  git@github.com:octocat/hello-world.git (push)
```

### 원본 저장소와 동기화

```bash
# 1. 원본의 최신 변경 가져오기
git fetch upstream

# 2. main 브랜치에 병합
git switch main
git merge upstream/main

# 3. 내 Fork에 반영
git push origin main
```

---

## 3. Pull Request (PR)

변경 사항을 원본 저장소에 반영해달라고 요청합니다.

### Pull Request 생성 과정

```bash
# 1. 새 브랜치에서 작업
git switch -c feature/add-greeting

# 2. 변경 후 커밋
echo "Hello, World!" > greeting.txt
git add .
git commit -m "feat: 인사말 파일 추가"

# 3. 내 Fork에 푸시
git push origin feature/add-greeting
```

### GitHub에서 PR 생성

1. GitHub에서 "Compare & pull request" 버튼 클릭
2. PR 정보 작성:
   - **제목**: 변경 사항 요약
   - **설명**: 상세 내용, 관련 이슈
3. "Create pull request" 클릭

### PR 템플릿 예시

```markdown
## 변경 사항
- 인사말 출력 기능 추가
- greeting.txt 파일 생성

## 관련 이슈
Closes #123

## 테스트
- [x] 로컬에서 동작 확인
- [x] 기존 기능에 영향 없음

## 스크린샷
(필요시 첨부)
```

### PR 워크플로우

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  1. Fork & Clone                                             │
│         ↓                                                    │
│  2. 브랜치 생성 & 작업                                         │
│         ↓                                                    │
│  3. Push to Fork                                             │
│         ↓                                                    │
│  4. Create Pull Request                                      │
│         ↓                                                    │
│  5. Code Review (리뷰어 피드백)                                │
│         ↓                                                    │
│  6. 수정 필요시 추가 커밋                                       │
│         ↓                                                    │
│  7. Merge (관리자가 병합)                                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Code Review (코드 리뷰)

PR을 통해 코드를 검토하고 피드백을 주고받습니다.

### 리뷰 요청하기

1. PR 페이지에서 "Reviewers" 클릭
2. 리뷰어 선택

### 리뷰 작성하기

1. "Files changed" 탭에서 변경 내용 확인
2. 라인별로 코멘트 추가 가능
3. 리뷰 완료:
   - **Comment**: 일반 코멘트
   - **Approve**: 승인
   - **Request changes**: 수정 요청

### 리뷰 피드백 반영

```bash
# 피드백 받은 내용 수정
git add .
git commit -m "fix: 리뷰 피드백 반영"
git push origin feature/add-greeting

# PR에 자동으로 커밋 추가됨
```

---

## 5. Issues (이슈)

버그, 기능 요청, 질문 등을 관리합니다.

### Issue 작성

1. 저장소의 "Issues" 탭
2. "New issue" 클릭
3. 제목과 설명 작성

### Issue 템플릿 예시

**버그 리포트:**
```markdown
## 버그 설명
로그인 버튼 클릭 시 에러 발생

## 재현 방법
1. 로그인 페이지 이동
2. 이메일/비밀번호 입력
3. 로그인 버튼 클릭
4. 에러 메시지 확인

## 예상 동작
메인 페이지로 이동

## 환경
- OS: macOS 14.0
- Browser: Chrome 120
```

**기능 요청:**
```markdown
## 기능 설명
다크 모드 지원

## 필요한 이유
눈의 피로 감소

## 추가 정보
(디자인 참고 자료 등)
```

### Issue와 PR 연결

```markdown
# PR 설명에서 이슈 참조
Fixes #42
Closes #42
Resolves #42

# 위 키워드 사용 시 PR 머지되면 이슈 자동 종료
```

---

## 6. GitHub 협업 실습

### 실습 1: 오픈소스 기여 시뮬레이션

```bash
# 1. 연습용 저장소 Fork (GitHub 웹에서)
# https://github.com/octocat/Spoon-Knife

# 2. Fork한 저장소 클론
git clone git@github.com:myname/Spoon-Knife.git
cd Spoon-Knife

# 3. upstream 설정
git remote add upstream git@github.com:octocat/Spoon-Knife.git

# 4. 브랜치 생성
git switch -c my-contribution

# 5. 파일 수정
echo "My name is here!" >> contributors.txt

# 6. 커밋 & 푸시
git add .
git commit -m "Add my name to contributors"
git push origin my-contribution

# 7. GitHub에서 Pull Request 생성
```

### 실습 2: 팀 협업 시나리오

```bash
# === 팀원 A (저장소 관리자) ===
# 1. 저장소 생성 및 초기 설정
mkdir team-project
cd team-project
git init
echo "# Team Project" > README.md
git add .
git commit -m "initial commit"
git remote add origin git@github.com:teamA/team-project.git
git push -u origin main

# 2. Collaborator 추가 (GitHub Settings > Collaborators)

# === 팀원 B ===
# 1. 저장소 클론
git clone git@github.com:teamA/team-project.git
cd team-project

# 2. 브랜치에서 작업
git switch -c feature/login
echo "login feature" > login.js
git add .
git commit -m "feat: 로그인 기능 구현"
git push origin feature/login

# 3. GitHub에서 PR 생성

# === 팀원 A ===
# 1. PR 리뷰 및 머지
# 2. 머지 후 로컬 업데이트
git pull origin main
```

---

## 7. 유용한 GitHub 기능

### Labels (라벨)

이슈/PR 분류:
- `bug`: 버그
- `enhancement`: 기능 개선
- `documentation`: 문서
- `good first issue`: 입문자용

### Milestones (마일스톤)

이슈들을 버전/스프린트로 그룹화

### Projects (프로젝트 보드)

칸반 보드 스타일로 작업 관리:
- To Do
- In Progress
- Done

### GitHub Actions

자동화 워크플로우:
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: npm test
```

---

## 명령어 요약

| 명령어 | 설명 |
|--------|------|
| `git remote add upstream URL` | 원본 저장소 추가 |
| `git fetch upstream` | 원본 변경 가져오기 |
| `git merge upstream/main` | 원본과 병합 |
| `git push origin 브랜치` | Fork에 푸시 |

---

## 핵심 용어 정리

| 용어 | 설명 |
|------|------|
| **Fork** | 저장소를 내 계정으로 복사 |
| **Pull Request** | 변경 사항 반영 요청 |
| **Code Review** | 코드 검토 |
| **Merge** | 브랜치/PR 병합 |
| **Issue** | 버그/기능 요청 관리 |
| **upstream** | 원본 저장소 |
| **origin** | 내 원격 저장소 |

---

## 학습 완료!

Git/GitHub 기초 학습을 완료했습니다. 다음 주제로 넘어가기 전에 실제 프로젝트에서 연습해보세요!

### 추천 연습

1. GitHub에서 관심 있는 오픈소스 프로젝트 찾기
2. 문서 오타 수정으로 첫 기여 시도
3. 개인 프로젝트를 GitHub에 올리고 관리

---

## 연습 문제

### 연습 1: Fork 후 Pull Request 제출
1. `https://github.com/octocat/Spoon-Knife` 저장소를 Fork합니다.
2. Fork한 저장소를 클론하고, 원본을 가리키는 `upstream`을 추가한 뒤 `git remote -v`로 두 원격을 확인합니다.
3. `add-my-name` 브랜치를 만들고, 파일에 GitHub 사용자명을 추가한 뒤 커밋하고 Fork에 푸시합니다.
4. 원본 저장소를 대상으로 Pull Request(PR)를 엽니다. 이 레슨에서 다룬 PR 템플릿 형식에 맞게 제목과 설명을 작성합니다.

### 연습 2: 코드 리뷰(Code Review) 연습
파트너와 함께(또는 두 개의 GitHub 계정을 사용하여):
1. 한 사람이 작은 코드 변경이 담긴 PR을 엽니다.
2. 리뷰어가 "Files changed" 탭에서 특정 라인에 최소 두 개의 인라인 코멘트를 남깁니다.
3. 작성자가 피드백을 반영하여 새 커밋을 푸시합니다 — 새 커밋이 열린 PR에 자동으로 추가되는지 확인합니다.
4. 리뷰어가 승인(Approve)하고, 작성자(또는 쓰기 권한이 있는 리뷰어)가 PR을 머지합니다.

### 연습 3: Issues 워크플로우
1. 자신의 저장소에서 이 레슨의 템플릿 형식을 사용하여 버그 리포트 이슈를 작성합니다. 재현 단계, 예상 동작, 환경 정보를 포함합니다.
2. 가상의 개선 사항에 대한 기능 요청 이슈를 작성합니다.
3. 설명에 `Closes #<이슈번호>`를 포함한 PR을 엽니다. PR을 머지하고 이슈가 자동으로 닫히는지 확인합니다.

### 연습 4: Fork 동기화
1. 연습 1에서 만든 Fork를 사용하여 업스트림(upstream) 변경 사항을 시뮬레이션합니다(저장소 소유자에게 요청하거나, 테스트용 저장소라면 GitHub 웹 에디터로 직접 커밋 생성).
2. `git fetch upstream`을 실행하고 `git log upstream/main`으로 새 커밋을 확인한 뒤 로컬 `main`에 병합합니다.
3. `git push origin main`으로 업데이트된 `main`을 Fork에 푸시합니다.

### 연습 5: 브랜치 보호 규칙(Branch Protection Rules)
관리자 권한이 있는 저장소에서:
1. **Settings → Branches**로 이동하여 `main`에 대한 브랜치 보호 규칙을 추가합니다.
2. "Require a pull request before merging"과 "Require status checks to pass before merging"을 활성화합니다.
3. `main`으로 직접 푸시를 시도하고 거부되는지 확인합니다.
4. 브랜치 보호 규칙이 팀 환경에서 코드 품질을 어떻게 향상시키는지 간략히 서술합니다.
